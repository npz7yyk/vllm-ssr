# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional, Callable, TYPE_CHECKING

import functools
import numpy as np
import re
import torch
import torch.nn as nn

from vllm.attention.layer import Attention
from vllm.attention.utils.fa_utils import is_flash_attn_varlen_func_available
from vllm.config import (CompilationLevel, VllmConfig,
                         get_layers_from_vllm_config)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.utils import is_pin_memory_available
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import compute_probs
from vllm.v1.spec_decode.attn_overrider import build_attention_overrider
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)

PADDING_SLOT_ID = -1


def _wrap_func(enter: str = "", exit: str = "") -> Callable:
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            inst = args[0] if args else None
            enter_fn = getattr(inst, enter) if enter else lambda: None
            exit_fn = getattr(inst, exit) if exit else lambda: None

            if enter_fn:
                enter_fn()
            try:
                return fn(*args, **kwargs)
            finally:
                if exit_fn:
                    exit_fn()
        return wrapper
    return deco


class LayerIndexer:
    """A utility class to iterate over layer indices."""

    def __init__(self, attn_layer_names: set[str]):
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


class SSRProposer:

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner: "GPUModelRunner",
    ):
        self.vllm_config = vllm_config
        self.device = device
        self.speculative_config = vllm_config.speculative_config
        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method

        self.runner = runner
        self.dtype = vllm_config.model_config.dtype
        self.max_model_len = vllm_config.model_config.max_model_len
        self.block_size = vllm_config.cache_config.block_size
        self.vocab_size = vllm_config.model_config.get_vocab_size()
        self.num_speculative_tokens = \
            self.speculative_config.num_speculative_tokens
        self.max_num_tokens = \
            vllm_config.scheduler_config.max_num_batched_tokens
        self.token_arange_np = np.arange(self.max_num_tokens)

        self.is_multimodal_model = vllm_config.model_config \
            .is_multimodal_model
        assert not vllm_config.model_config.is_multimodal_model, \
            "SSR is not supported for multimodal models yet."

        assert not self.vllm_config.compilation_config.cudagraph_mode. \
            has_full_cudagraphs(), \
            "Full CUDA graphs are not supported for SSR."
        self.use_cuda_graph = (self.vllm_config.compilation_config.level
                               == CompilationLevel.PIECEWISE and
                               not self.vllm_config.model_config.enforce_eager)
        self.cudagraph_batch_sizes = list(
            reversed(
                self.vllm_config.compilation_config.cudagraph_capture_sizes))

        max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.arange = torch.arange(
            # Need +1 here because the arange is used to set query_start_loc,
            # which has one more element than batch_size.
            max_batch_size + 1,
            device=device,
            dtype=torch.int32,
        )
        # Lazy allocation of buffers.
        self._draft_probs = None
        self._sampled_token_ids = None

        # Determine allowed attention backends once during initialization.
        self.allowed_attn_types = (FlashAttentionMetadata,)
        import vllm.v1.attention.backends.flash_attn as flash_attn_module
        self.flash_attn_module = flash_attn_module
        assert is_flash_attn_varlen_func_available(), \
            "FlashAttention with variable length support is required for SSR."
        self._original_func = None

    def _sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata
    ) -> SamplerOutput:
        """
        Sample from the logits with sampling_metadata.
        """
        return self.runner.sampler(logits, sampling_metadata)

    def enable_attn_override(self):
        # Override function for attention to enable SSR.
        def overrided_attention(*args, **kwargs):
            layer_index = self.layer_indexer.current_layer_index
            use_private_attention = self.attention_overrider(
                layer_index=layer_index,
                *args, **kwargs
            )
            # Attention already conducted inside the overrider.
            if use_private_attention:
                return
            # The overrider only modifies the attention metadata.
            # NOTE: max_seqlen_k (int) may be changed in the overrider.
            kwargs["max_seqlen_k"] = self.attn_metadata.max_seq_len
            # Call the original attention function.
            self._original_func(*args, **kwargs)

        # Save the original function for the first time.
        if self._original_func is None:
            self._original_func = self.flash_attn_module.flash_attn_varlen_func
        # Override the function.
        self.flash_attn_module.flash_attn_varlen_func = overrided_attention

    def disable_attn_override(self):
        # Restore the original function.
        self.flash_attn_module.flash_attn_varlen_func = self._original_func

    def get_draft_probs(self, running_req_mask: torch.Tensor) -> torch.Tensor:
        # [batch_size * num_speculative_tokens, vocab_size]
        running_req_mask = running_req_mask[:self.batch_size]
        return self._draft_probs[:self.batch_size][running_req_mask]. \
            reshape(-1, self.vocab_size)

    @property
    def sampled_token_ids(self) -> torch.Tensor:
        # [batch_size * num_speculative_tokens]
        return self._sampled_token_ids[:self.batch_size]

    @_wrap_func(enter="enable_attn_override", exit="disable_attn_override")
    def propose(
        self,
        next_token_ids: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embeds: Optional[list[torch.Tensor]] = None,
    ) -> torch.Tensor:
        # For prototyping, mm, dp, pp, and kv_connector etc. are not supported.
        # TODO(Yikang): Add them back if needed.

        # Determine the batch size and the number of tokens to compute.
        num_tokens = next_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]
        self.batch_size = batch_size
        last_token_indices = self.arange[:batch_size]
        self.input_ids[:batch_size] = next_token_ids

        # FIXME: Lazy allocation of buffers. Not a good practice.
        if self._draft_probs is None or \
                self._draft_probs.shape[0] < batch_size:
            self._draft_probs = torch.empty(
                (batch_size, self.num_speculative_tokens, self.vocab_size),
                dtype=torch.float32, device=self.device
            )
            self._sampled_token_ids = torch.empty(
                (batch_size, self.num_speculative_tokens),
                dtype=torch.float32, device=self.device
            )

        # FIXME: need to consider multiple kv_cache_groups
        attn_metadata = self.runner.attn_groups[0][0].metadata_builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )
        assert isinstance(attn_metadata, self.allowed_attn_types)
        # At this moment, we assume all attention layers belong to the same KV
        # cache group, thus using the same attention metadata.
        self.attn_metadata = attn_metadata
        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata
        if self.use_cuda_graph and \
                num_tokens <= self.cudagraph_batch_sizes[-1]:
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)
        else:
            num_input_tokens = num_tokens

        # Update attention metadata and prepare inputs.
        positions = attn_metadata.seq_lens.long()
        # Mask out the position ids that exceed the max model length.
        # Otherwise, we may get out-of-range error in RoPE.
        exceeds_max_model_len = positions >= self.max_model_len
        clamped_positions = torch.where(exceeds_max_model_len, 0, positions)
        # Update the attention metadata.
        attn_metadata.num_actual_tokens = num_tokens
        attn_metadata.max_query_len = 1
        attn_metadata.query_start_loc = self.arange[:batch_size + 1]
        attn_metadata.max_seq_len += 1
        attn_metadata.seq_lens += 1
        # Consider max model length.
        attn_metadata.max_seq_len = \
            min(attn_metadata.max_seq_len, self.max_model_len)
        # For the requests that exceed the max model length, we set the
        # sequence length to 1 to minimize their overheads in attention.
        attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)
        # Compute the slot mapping.
        block_numbers = clamped_positions // self.block_size
        block_ids = attn_metadata.block_table.gather(
            dim=1, index=block_numbers.view(-1, 1))
        block_ids = block_ids.view(-1)
        attn_metadata.slot_mapping = \
            (block_ids * self.block_size + clamped_positions % self.block_size)
        # Mask out the slot mappings that exceed the max model length.
        # Otherwise, the KV cache will be inadvertently updated with the
        # padding tokens.
        attn_metadata.slot_mapping.masked_fill_(
            exceeds_max_model_len, PADDING_SLOT_ID)
        # copy inputs to buffer for cudagraph
        self.positions[:batch_size] = clamped_positions
        positions = self.positions[:num_input_tokens]

        if self.is_multimodal_model:
            raise NotImplementedError
            input_ids = self.input_ids[:num_tokens]
            inputs_embeds = self.model.get_input_embeddings(
                input_ids,
                multimodal_embeddings=mm_embeds or None,
            )
            self.inputs_embeds[:num_tokens] = inputs_embeds
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            input_ids = None
        else:
            inputs_embeds = None
            input_ids = self.input_ids[:num_input_tokens]

        uniform_decode = True
        batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens,
                                           uniform_decode=uniform_decode)
        cudagraph_runtime_mode, batch_descriptor = \
            self.runner.cudagraph_dispatcher.dispatch(batch_descriptor)

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_descriptor,
        ):
            hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=inputs_embeds,
            )

        hidden_states = hidden_states[last_token_indices]
        logits = self.model.compute_logits(hidden_states, None)
        self._draft_probs[:batch_size, 0] = compute_probs(
            logits, self.arange[1: batch_size + 1], sampling_metadata)
        self._sampled_token_ids[:batch_size, 0] = \
            self._sample(logits, sampling_metadata).sampled_token_ids.flatten()

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            return self.sampled_token_ids

        # Speculatively sample multiple tokens.
        for step in range(1, self.num_speculative_tokens):
            self.input_ids[:batch_size] = \
                self._sampled_token_ids[:batch_size, step - 1]
            self.positions[:batch_size] += 1

            # Update attention metadata and prepare inputs.
            # Mask out the position ids that exceed the max model length.
            # Otherwise, we may get out-of-range error in RoPE.
            positions = self.positions[:batch_size]
            exceeds_max_model_len = positions >= self.max_model_len
            positions = torch.where(exceeds_max_model_len, 0, positions)
            # Update the attention metadata.
            attn_metadata.max_seq_len += 1
            attn_metadata.seq_lens += 1
            # Consider max model length.
            attn_metadata.max_seq_len = \
                min(attn_metadata.max_seq_len, self.max_model_len)
            # For the requests that exceed the max model length, we set the
            # sequence length to 1 to minimize their overheads in attention.
            attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)
            # Compute the slot mapping.
            block_numbers = positions // self.block_size
            block_ids = attn_metadata.block_table.gather(
                dim=1, index=block_numbers.view(-1, 1))
            block_ids = block_ids.view(-1)
            attn_metadata.slot_mapping = \
                (block_ids * self.block_size + positions % self.block_size)
            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            attn_metadata.slot_mapping.masked_fill_(
                exceeds_max_model_len, PADDING_SLOT_ID)

            # Run the model.
            # Use persistent buffers for CUDA graphs.
            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_descriptor,
            ):
                hidden_states = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    inputs_embeds=inputs_embeds,
                )

            # Get the logits and sample the next token.
            hidden_states = hidden_states[last_token_indices]
            logits = self.model.compute_logits(hidden_states, None)
            self._draft_probs[:batch_size, step] = compute_probs(
                logits, self.arange[1: batch_size + 1], sampling_metadata)
            self._sampled_token_ids[:batch_size, step] = self._sample(
                logits, sampling_metadata).sampled_token_ids.flatten()

        # [batch_size, num_speculative_tokens]
        return self.sampled_token_ids

    def prepare_inputs(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        # [batch_size]
        num_rejected_tokens: torch.Tensor
    ) -> tuple[CommonAttentionMetadata, torch.Tensor]:
        """
        This function is used to prepare the inputs for the spec decode.
        It updates to the common_attn_metadata to account for the rejected
        tokens (and newly sampled tokens). It also returns the token indices
        of the tokens that should be fed to the speculator.
        """
        # E.g.
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1, q1 + q2, q1 + q2 + q3]
        #  common_attn_metadata.seq_lens{_cpu}: [s1, s2, s3]
        #  num_rejected_tokens: [n1, n2, n3]
        # This function computes the intermediate values:
        #  num_tokens_per_req: [q1 - n1, q2 - n2, q3 - n3]
        # And returns:
        #  common_attn_metadata.query_start_loc{_cpu}:
        #       [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        #  common_attn_metadata.seq_lens{_cpu}:
        #       [s1 - n1 + 1, s2 - n2 + 1, s3 - n3 + 1]
        #  token_indices: [0, 1, ..., q1 - n1 - 1,
        #                 q1, q1 + 1, ..., q1 + q2 - n2 - 1,
        #                 q1 + q2, q1 + q2 + 1, ..., q1 + q2 + q3 - n3 - 1]

        device = common_attn_metadata.query_start_loc.device
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        new_seq_lens_cpu = common_attn_metadata.seq_lens_cpu \
            - num_rejected_tokens

        # [0, q1, q1 + q2, q1 + q2 + q3] -> [q1, q2, q3]
        new_query_len_per_req = (query_start_loc_cpu[1:] -
                                 query_start_loc_cpu[:-1])
        # [q1, q2, q3] -> [q1 - n1, q2 - n2, q3 - n3]
        new_num_tokens_per_req = new_query_len_per_req - num_rejected_tokens
        new_num_tokens_per_req_np = new_num_tokens_per_req.numpy()

        # [q1 - n1, q2 - n2, q3 - n3] ->
        # [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
        new_query_start_loc_cpu = torch.zeros(
            query_start_loc_cpu.shape,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available())
        new_query_start_loc_np = new_query_start_loc_cpu.numpy()
        np.cumsum(new_num_tokens_per_req_np, out=new_query_start_loc_np[1:])

        total_num_tokens = new_query_start_loc_np[-1]
        # Example assuming num_tokens_per_req_np = [2, 4, 3]
        # this implies that `new_query_start_locs` is:
        # [0, 2, 6, 9] ->
        # [0, 0, 2, 2, 2, 2, 6, 6, 6]
        #  _r1_  ____r2____  ___r3__
        new_query_start_locs_expanded = np.repeat(new_query_start_loc_np[:-1],
                                                  new_num_tokens_per_req_np)
        # [0, 1, 2, 3, 4, 5, 6, 7, 8] ->
        # [0, 1, 0, 1, 2, 3, 0, 1, 2]
        #  _r1_  ____r2____  ___r3__
        token_offests = self.token_arange_np[:total_num_tokens] \
            - new_query_start_locs_expanded

        # Expand starting positions to match token pattern
        # [0, q1, q1 + q2] ->
        # [0, 0, q1, q1, q1, q1, q1 + q2, q1 + q2, q1 + q2]
        #  _r1_  _____r2_______  ___________r3____________
        old_query_start_locs_expanded = np.repeat(
            query_start_loc_cpu[:-1].numpy(), new_num_tokens_per_req_np)
        # Final token indices are:
        # [0, 1,                                // req 1
        #  q1 + 0, q1 + 1, q1 + 2, q1 + 3,       // req 2
        #  q1 + q2 + 0, q1 + q2 + 1, q1 + q2 + 2] // req 3
        token_indices_np = token_offests + old_query_start_locs_expanded
        token_indices = torch.from_numpy(token_indices_np).to(
            device, non_blocking=True)

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc_cpu.to(device,
                                                       non_blocking=True),
            seq_lens=new_seq_lens_cpu.to(device, non_blocking=True),
            query_start_loc_cpu=new_query_start_loc_cpu,
            seq_lens_cpu=new_seq_lens_cpu,
            num_computed_tokens_cpu=common_attn_metadata.
            num_computed_tokens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=new_seq_lens_cpu.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[token_indices],
            causal=True,
        )

        return spec_common_attn_metadata, token_indices

    def load_model(self, target_model: nn.Module) -> None:
        # Load the target model.
        self.model = target_model
        self.attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, Attention).keys())
        # FIXME: LayerIndexer has strong assumptions on the layer names.
        #        It may not work for some models.
        self.layer_indexer = LayerIndexer(self.attn_layer_names)

        # Reuse runner buffers for inputs and positions.
        # This is a MUST when using CUDA graphs.
        self.input_ids = self.runner.input_ids.gpu
        self.positions = self.runner.positions.gpu

        # Build the attention overrider.
        self.attention_overrider = build_attention_overrider(
            vllm_config=self.vllm_config,
            device=self.device,
        )
