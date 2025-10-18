import abc
import re
import torch

from vllm import _custom_ops as ops
from vllm.attention.layer import Attention
from vllm.attention.ops.triton_unified_attention import unified_attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl


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

    def is_reset(self) -> bool:
        return self._current_pos == 0

    @property
    def layers(self) -> list[Attention]:
        return [layer for _, layer in self._layer_with_indices]

    @property
    def num_layers(self) -> int:
        return self._num_layers

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
        # Basic attributes.
        self.vllm_config = vllm_config
        self.device = device
        self.layer_indexer = LayerIndexer(vllm_config)

        # Commonly used attributes.
        self.block_size = vllm_config.cache_config.block_size
        self.max_model_len = vllm_config.model_config.max_model_len

        # Save original functions.
        self.original_kv_insert_func = ops.reshape_and_cache_flash
        self.original_attention_func = unified_attention

        # Override the original kv insert function
        def overriden_kv_insert(*args, **kwargs):
            return self._overriden_kv_insert(*args, **kwargs)
        import vllm.v1.attention.backends.triton_attn as triton_attn
        triton_attn.ops.reshape_and_cache_flash = overriden_kv_insert

        # Override the original attention function
        def overriden_attention(*args, **kwargs):
            return self._overriden_attention(*args, **kwargs)
        for layer in self.layer_indexer.layers:
            assert isinstance(layer.impl, TritonAttentionImpl)
            layer.impl.unified_attention = overriden_attention

        # The current speculative step.
        # 0 means target verification.
        self.current_draft_step = 0

    @abc.abstractmethod
    def _overriden_kv_insert(self):
        # Switch implementations based on self.in_draft
        pass

    @abc.abstractmethod
    def _overriden_attention(self):
        # Switch implementations based on self.in_draft
        pass

    def ready_to_draft(self):
        return True

    @property
    def in_draft(self) -> bool:
        return self.current_draft_step > 0

    def set_draft_step(self, step: int):
        assert self.layer_indexer.is_reset()
        self.current_draft_step = step

    def _get_layer(self) -> tuple[int, Attention]:
        """Get the current layer index and object.

        Returns:
            The current layer index and object.
        """
        rst = self.layer_indexer.current_layer
        self.layer_indexer.step()
        return rst
