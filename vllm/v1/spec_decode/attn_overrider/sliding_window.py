import torch

from vllm.config import VllmConfig
from .abstract import AbstractAttentionOverrider, override_return


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
        self.offset = torch.arange(
            -self.window_num_blocks + 1, 1, 1,
            dtype=torch.int32, device=device
        )

    @override_return(use_private_attention)
    def __call__(self, layer_index: int, *_args, **_kwargs):
        # Since all attention layers share the same attn_metadata,
        # we only need to modify it for the first layer of the model.
        if layer_index > 0:
            return

        attn_metadata = self._get_attn_metadata()
        positions = attn_metadata.seq_lens

        # Compute the sliding window attention blocks.
        tail_slots = (positions // self.block_size).unsqueeze_(-1)
        tail_slots.clamp_min_(self.window_num_blocks)
        window_slots = tail_slots.expand(-1, self.window_num_blocks)
        window_slots = window_slots + self.offset
        # Keep the first block as the attention sink.
        attn_metadata.block_table[:, 1: self.window_num_blocks + 1] = \
            torch.gather(attn_metadata.block_table, 1, window_slots)

        # Compute the seq_lens for the sliding window attention.
        extra_lens = (positions - self.window_size - self.block_size)
        extra_blocks = torch.div(
            extra_lens + self.block_size - 1,
            self.block_size, rounding_mode='floor'
        ).clamp_min_(0)
        attn_metadata.seq_lens -= extra_blocks * self.block_size
        attn_metadata.max_seq_len = attn_metadata.seq_lens.max()
