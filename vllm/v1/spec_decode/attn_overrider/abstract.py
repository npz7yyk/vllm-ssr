import abc
import functools
import torch

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata


def override_return(always_value):
    """
    Wrap a function so its return value is always `always_value`.
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            fn(*args, **kwargs)
            return always_value
        return wrapper
    return deco


class AbstractAttentionOverrider(abc.ABC):
    """ An interface for overriding attention computation. """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.device = device

        self.block_size = vllm_config.cache_config.block_size
        self.max_model_len = vllm_config.model_config.max_model_len

    def _get_attn_metadata(self) -> FlashAttentionMetadata:
        """Get the attention metadata from the forward context.

        Returns:
            The attention metadata.
        """
        attn_metadata = get_forward_context().attn_metadata
        # Assume: All attention layers share the same attn_metadata.
        if isinstance(attn_metadata, dict):
            attn_metadata = list(attn_metadata.values())[0]
        return attn_metadata
