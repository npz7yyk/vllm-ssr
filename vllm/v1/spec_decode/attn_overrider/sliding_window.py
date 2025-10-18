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
        self.offset = torch.arange(
            -self.window_num_blocks + 1, 1, 1,
            dtype=torch.int32, device=device
        )

    def _overriden_kv_insert(self, *args, **kwargs):
        # Always use the original kv insert function.
        return self.original_kv_insert_func(*args, **kwargs)

    def _overriden_attention(self, *args, **kwargs):
        # If not in draft phase, use the original attention function.
        if not self.in_draft:
            return self.original_attention_func(*args, **kwargs)

        # Since all attention layers share the same attn_metadata,
        # we only need to modify it for the first layer of the model.
        layer_index, _ = self._get_layer()

        # Modify the attn_metadata for sliding window attention.
        if layer_index == 0:
            # Get the necessary arguments.
            seq_lens: torch.Tensor = kwargs['seqused_k']
            block_table: torch.Tensor = kwargs['block_table']

            # Compute the sliding window attention blocks.
            tail_slots = (seq_lens // self.block_size).unsqueeze_(-1)
            tail_slots.clamp_(self.window_num_blocks, self.max_block_slot)
            window_slots = tail_slots.expand(-1, self.window_num_blocks)
            window_slots = window_slots + self.offset
            # Keep the first block as the attention sink.
            # block_table is a tensor, only needs to modify once.
            block_table[:, 1: self.window_num_blocks + 1] = \
                torch.gather(block_table, 1, window_slots)

            # Compute the seq_lens for the sliding window attention.
            extra_lens = (seq_lens - self.window_size - self.block_size)
            extra_blocks = torch.div(
                extra_lens + self.block_size - 1,
                self.block_size, rounding_mode='floor'
            ).clamp_min_(0)
            # seq_lens is a tensor, only needs to modify once.
            seq_lens -= extra_blocks * self.block_size
            self.max_seq_len = seq_lens.max().item()

        # max_seq_len is an integer, needs to modify every time.
        kwargs['max_seqlen_k'] = self.max_seq_len

        # Call the original attention function with modified attn_metadata.
        return self.original_attention_func(*args, **kwargs)
