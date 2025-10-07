import abc
import functools
import re
import torch

from vllm.attention.layer import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
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


# FIXME: LayerIndexer has strong assumptions on the layer names.
#        It may not work for some models.
class LayerIndexer:
    """A utility class to iterate over layer indices."""

    def __init__(self, vllm_config: VllmConfig):
        # Get attention layers from the vllm config.
        attn_layers = get_layers_from_vllm_config(vllm_config, Attention)

        # Collect layer indices from layer names.
        between_dots = re.compile(r'\.(\d+)\.')
        any_digits = re.compile(r'\d+')
        layer_with_indices: list[tuple[int, Attention]] = []
        # Try to extract the layer index from the layer name.
        for name, layer in attn_layers.items():
            # Try1: match digits between dots.
            match = between_dots.search(name)
            if match:
                layer_with_indices.append((int(match.group(1)), layer))
                continue
            # Try2: match any digits, take the first match.
            match = any_digits.search(name)
            if match:
                layer_with_indices.append((int(match.group(0)), layer))
                continue
            # Failed to parse the layer index.
            raise ValueError(f"Cannot parse layer index from layer {name}.")

        # Build the layer index iterator.
        self._layer_with_indices = sorted(layer_with_indices)
        self._num_layers = len(self._layer_with_indices)
        self._current_pos = 0

    @property
    def current_layer(self) -> tuple[int, Attention]:
        return self._layer_with_indices[self._current_pos]

    def step(self):
        if self._current_pos + 1 < self._num_layers:
            self._current_pos += 1
        else:
            self._current_pos = 0


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

    def _get_layer(self) -> tuple[int, Attention]:
        """Get the current layer index and object.

        Returns:
            The current layer index and object.
        """
        rst = self.layer_indexer.current_layer
        self.layer_indexer.step()
        return rst

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
