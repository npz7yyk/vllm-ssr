import abc
import re
import torch

from vllm import _custom_ops as ops
from vllm.attention.layer import Attention
from vllm.attention.ops.triton_unified_attention import unified_attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
from vllm.v1.utils import CpuGpuBuffer


# FIXME: This function has strong assumptions on the layer names.
#        It may not work for some models.
def _parse_layer_index(name: str) -> int:
    """ Parse the layer index from the layer name.

    Args:
        name: The name of the layer.

    Returns:
        The layer index extracted from the layer name.

    """
    # Try1: match digits between dots.
    match = re.search(r'\.(\d+)\.', name)
    if match:
        return int(match.group(1))
    # Try2: match any digits, take the first match.
    match = re.search(r'\d+', name)
    if match:
        return int(match.group(0))
    raise ValueError(f"Cannot parse layer index from layer {name}.")


def _should_override(layer: Attention) -> bool:
    """Check if the given attention layer should be overridden.

    Currently identified attention layers that should NOT be overridden:
        1. Sliding window attention layers.
    TODO: Add more if needed.

    Args:
        layer: The attention layer to check.

    Returns:
        True if the layer should be overridden, False otherwise.
    """
    assert isinstance(layer.impl, TritonAttentionImpl)

    # Check for sliding window attention.
    if layer.impl.sliding_window != (-1, -1):
        return False

    return True


class LayerIndexer:
    """A utility class to iterate over layer indices."""

    def __init__(self, vllm_config: VllmConfig):
        # Get attention layers from the vllm config.
        attn_layers = get_layers_from_vllm_config(vllm_config, Attention)
        self.attn_layers = list(attn_layers.values())

        # Collect layer indices from layer names.
        layer_metadata_list = []
        # Try to extract the layer index from the layer name.
        for name, layer in attn_layers.items():
            layer_index = _parse_layer_index(name)
            should_override = _should_override(layer)
            layer_metadata_list.append((layer_index, should_override, layer))

        # Build the layer index iterator.
        self._layer_metadata_list: list[tuple[int, bool, Attention]] = \
            sorted(layer_metadata_list, key=lambda x: x[0])
        self._num_layers = len(self._layer_metadata_list)
        self._ptr = 0

        # Collect override layer indices.
        self.override_layer_indices = [
            idx for idx, should_override, _ in self._layer_metadata_list
            if should_override
        ]
        self.min_override_layer_index = min(self.override_layer_indices)

    def is_reset(self) -> bool:
        return self._ptr == 0

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def current_layer(self) -> tuple[int, bool, Attention]:
        return self._layer_metadata_list[self._ptr]

    def step(self):
        if self._ptr + 1 < self._num_layers:
            self._ptr += 1
        else:
            self._ptr = 0


class AbstractAttentionOverrider(abc.ABC):
    """ An interface for overriding attention computation. """

    # Whether metadata remapping is needed.
    needs_metadata_remapping: bool = False

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
        for idx, layer in enumerate(self.layer_indexer.attn_layers):
            assert isinstance(layer.impl, TritonAttentionImpl)
            layer.impl.unified_attention = overriden_attention
        self.effective_layer_indices = \
            self.layer_indexer.override_layer_indices
        self.min_effective_layer_index = \
            self.layer_indexer.min_override_layer_index

        # The current speculative step.
        # 0 means target verification.
        self.current_draft_step = 0

        # Metadata mappings (index to metadata slot).
        self.index_to_metadata_buffer = CpuGpuBuffer(
            vllm_config.scheduler_config.max_num_seqs,
            dtype=torch.int32,
            device=self.device,
            pin_memory=True)
        self.index_to_metadata = self.index_to_metadata_buffer.gpu
        # Indicate whether each request is new.
        self.is_new_req_buffer = CpuGpuBuffer(
            vllm_config.scheduler_config.max_num_seqs,
            dtype=torch.bool,
            device=self.device,
            pin_memory=True)
        self.is_new_req = torch.zeros(
            vllm_config.scheduler_config.max_num_seqs,
            dtype=torch.bool,
            device=self.device)

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

    def _get_layer(self) -> tuple[int, bool, Attention]:
        """Get the current layer metadata and step the layer indexer.

        Returns:
            A tuple of (layer_index, should_override, layer).
        """
        rst = self.layer_indexer.current_layer
        self.layer_indexer.step()
        return rst

    def update_metadata_mappings(self, batch_size: int):
        """Update the request metadata mappings.

        Args:
            batch_size: The current batch size.
        """
        self.index_to_metadata_buffer.copy_to_gpu(batch_size)
        self.is_new_req_buffer.copy_to_gpu(batch_size)
        # Use logical_or to guarantee all new requests are marked.
        self.is_new_req[:batch_size].logical_or_(
            self.is_new_req_buffer.gpu[:batch_size])
