import abc
import functools
import re
import torch

from vllm.attention.layer import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

logger = init_logger(__name__)


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


# FIXME: LayerIndexer has strong assumptions on the layer names.
#        It may not work for some models.
class LayerIndexer:
    """A utility class to iterate over layer indices."""

    def __init__(self, vllm_config: VllmConfig):
        # Get attention layer names from the vllm config.
        attn_layer_names = \
            set(get_layers_from_vllm_config(vllm_config, Attention).keys())
        # Collect layer indices from layer names.
        between_dots = re.compile(r'\.(\d+)\.')
        any_digits = re.compile(r'\d+')
        layer_indices = []
        # Try to extract the layer index from the layer name.
        for name in attn_layer_names:
            # Try1: match digits between dots.
            match = between_dots.search(name)
            if match:
                layer_indices.append(int(match.group(1)))
                continue
            # Try2: match any digits, take the first match.
            match = any_digits.search(name)
            if match:
                layer_indices.append(int(match.group(0)))
                continue
            # Failed to parse the layer index.
            layer_indices = list(range(len(attn_layer_names)))
            logger.warning(
                f"Failed to parse layer index from attention layer: {name}, "
                f"Try fallback to use 0, 1, ..., {len(attn_layer_names) - 1}."
            )
            break

        # Build the layer index iterator.
        self._layer_indices = sorted(set(layer_indices))
        self._num_layers = len(self._layer_indices)
        self._current_pos = 0

    @property
    def current_layer_index(self) -> int:
        # Return the current layer and move to the next one.
        rst = self._layer_indices[self._current_pos]
        self._current_pos = (self._current_pos + 1) % self._num_layers
        return rst


class AbstractAttentionOverrider(abc.ABC):
    """ An interface for overriding attention computation. """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.device = device
        self.layer_indexer = LayerIndexer(vllm_config)

        self.block_size = vllm_config.cache_config.block_size
        self.max_model_len = vllm_config.model_config.max_model_len

    def _get_layer_index(self) -> int:
        """Get the current layer index from the layer indexer.

        Returns:
            The current layer index.
        """
        return self.layer_indexer.current_layer_index

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
