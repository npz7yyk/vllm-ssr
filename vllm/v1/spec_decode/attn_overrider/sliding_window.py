import torch

from vllm.config import VllmConfig
from .abstract import AbstractAttentionOverrider


class SlidingWindowAttentionOverrider(AbstractAttentionOverrider):
    """An attention overrider that implements sliding window attention."""

    # Wether this overrider uses a private attention implementation.
    use_private_attention = False

    def __init__(
        self,
        # Base class args.
        vllm_config: VllmConfig,
        device: torch.device,
        # Sliding window specific args.
        window_size: int = 64,
        **_kwargs
    ):
        super().__init__(vllm_config, device)
        assert window_size % self.block_size == 0
        self.window_size = window_size
        self.window_num_blocks = window_size // self.block_size
        self.max_block_slot = self.max_model_len // self.block_size - 1
        self.num_speculative_tokens = \
            vllm_config.speculative_config.num_speculative_tokens
        self.offset = torch.arange(
            -self.window_num_blocks + 1, 1, 1,
            dtype=torch.int32, device=device
        )

        # Whether to enable sliding window attention.
        self._enable_sliding_window = False

        # Lazy allocation of self-maintained block_table
        self.block_table = None

    def ready_to_draft(self):
        return self._enable_sliding_window

    def _overriden_kv_insert(self, *args, **kwargs):
        # Always use the original kv insert function.
        return self.original_kv_insert_func(*args, **kwargs)

    def _overriden_attention(self, *args, **kwargs):
        # If not in draft phase, use the original attention function.
        # Besides, determine whether to enable sliding window attention.
        if not self.in_draft:
            max_seqlen_q = kwargs['max_seqlen_q']
            self._enable_sliding_window = \
                max_seqlen_q <= self.num_speculative_tokens + 1
            return self.original_attention_func(*args, **kwargs)

        # If not enabling sliding window, cannot reach here.
        assert self._enable_sliding_window

        # Since all attention layers share the same attn_metadata,
        # we only need to modify it for the first layer of the model.
        layer_index, _ = self._get_layer()

        # Modify the attn_metadata for sliding window attention.
        if layer_index == 0:
            # Get the necessary arguments.
            seq_lens: torch.Tensor = kwargs['seqused_k']
            block_table: torch.Tensor = kwargs['block_table']
            batch_size = seq_lens.shape[0]
            if self.block_table is None or \
                    self.block_table.shape[0] < batch_size:
                self.block_table = torch.zeros_like(block_table)

            # Compute the sliding window attention blocks.
            tail_slots = (seq_lens // self.block_size).unsqueeze_(-1)
            tail_slots.clamp_(self.window_num_blocks, self.max_block_slot)
            window_slots = tail_slots.expand(-1, self.window_num_blocks)
            window_slots = window_slots + self.offset
            # Keep the first block as the attention sink.
            self.block_table[:batch_size, 0] = block_table[:batch_size, 0]
            # Copy the sliding window blocks.
            self.block_table[:batch_size, 1: self.window_num_blocks + 1] = \
                torch.gather(block_table, 1, window_slots)

            # Compute the seq_lens for the sliding window attention.
            extra_lens = (seq_lens - self.window_size - self.block_size)
            extra_blocks = torch.div(
                extra_lens + self.block_size - 1,
                self.block_size, rounding_mode='floor'
            ).clamp_min_(0)

            # Store the modified seq_lens and max_seq_len.
            self.seqused_k = seq_lens - extra_blocks * self.block_size
            self.max_seqlen_k = self.seqused_k.max().item()

        # Store the original attn_metadata.
        original_block_table = kwargs['block_table']
        original_seqused_k = kwargs['seqused_k']
        original_max_seqlen_k = kwargs['max_seqlen_k']

        # # Update kwargs with modified attn_metadata.
        kwargs['block_table'] = self.block_table
        kwargs['seqused_k'] = self.seqused_k
        kwargs['max_seqlen_k'] = self.max_seqlen_k

        # Call the original attention function with modified attn_metadata.
        rst = self.original_attention_func(*args, **kwargs)

        # Restore the original attn_metadata in kwargs.
        kwargs['block_table'] = original_block_table
        kwargs['seqused_k'] = original_seqused_k
        kwargs['max_seqlen_k'] = original_max_seqlen_k

        return rst
