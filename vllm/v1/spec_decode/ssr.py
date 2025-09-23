# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
from dataclasses import replace
from importlib.util import find_spec
from typing import Optional, Protocol, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from vllm.attention.layer import Attention
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import (CompilationLevel, CUDAGraphMode, VllmConfig,
                         get_layers_from_vllm_config)
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import supports_multimodal
from vllm.platforms import current_platform
from vllm.utils import is_pin_memory_available, round_up
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionMetadata, _DEFAULT_MAX_NUM_SPLITS_FOR_CUDA_GRAPH)
from vllm.v1.attention.backends.tree_attn import (TreeAttentionMetadata,
                                                  TreeAttentionMetadataBuilder)
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import compute_probs
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)

PADDING_SLOT_ID = -1


class SSRProposer:

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner: "GPUModelRunner",
    ):
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method

        self.runner = runner
        self.dtype = vllm_config.model_config.dtype
        self.max_model_len = vllm_config.model_config.max_model_len
        self.block_size = vllm_config.cache_config.block_size
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
            # We need +1 here because the arange is used to set query_start_loc,
            # which has one more element than batch_size.
            max_batch_size + 1,
            device=device,
            dtype=torch.int32,
        )

        # Determine allowed attention backends once during initialization.
        self.allowed_attn_types: tuple
        if current_platform.is_rocm():
            rocm_types = [TritonAttentionMetadata, FlashAttentionMetadata]
            # vllm.v1.attention.backends.rocm_aiter_fa is an optional backend
            if find_spec("vllm.v1.attention.backends.rocm_aiter_fa"):
                from vllm.v1.attention.backends.rocm_aiter_fa import (
                    AiterFlashAttentionMetadata)
                rocm_types.append(AiterFlashAttentionMetadata)
            self.allowed_attn_types = tuple(rocm_types)
        else:
            self.allowed_attn_types = (FlashAttentionMetadata,
                                       TreeAttentionMetadata)

        # Parse the speculative token tree.
        spec_token_tree = self.speculative_config.speculative_token_tree
        self.tree_choices: list[tuple[int,
                                      ...]] = ast.literal_eval(spec_token_tree)
        tree_depth = len(self.tree_choices[-1])
        # Precompute per-level properties of the tree.
        num_drafts_per_level = [0] * tree_depth
        for node in self.tree_choices:
            num_drafts_per_level[len(node) - 1] += 1
        self.cu_drafts_per_level = [num_drafts_per_level[0]]
        self.child_drafts_per_level = [num_drafts_per_level[0]]
        for level in range(1, tree_depth):
            self.cu_drafts_per_level.append(self.cu_drafts_per_level[-1] +
                                            num_drafts_per_level[level])
            self.child_drafts_per_level.append(num_drafts_per_level[level] //
                                               num_drafts_per_level[level - 1])
        # Precompute draft position offsets in flattened tree.
        self.tree_draft_pos_offsets = torch.arange(
            1,
            len(self.tree_choices) + 1,
            device=device,
            dtype=torch.int32,
        ).repeat(max_batch_size, 1)

    def _sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata
    ) -> SamplerOutput:
        """
        Sample from the logits with sampling_metadata.
        """
        return self.runner.sampler(logits, sampling_metadata)

    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens]
        target_positions: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embeds: Optional[list[torch.Tensor]] = None,
    ) -> torch.Tensor:
        # For prototyping, mm, dp, pp, and kv_connector etc. are not supported.
        # TODO(Yikang): Add them back if needed.

        # Determine the batch size and the number of tokens to compute.
        if self.is_self_speculation:
            num_tokens = next_token_ids.shape[0]
            batch_size = next_token_ids.shape[0]
            last_token_indices = self.arange[:batch_size]
            self.input_ids[:batch_size] = next_token_ids
        else:
            raise NotImplementedError

        # FIXME: need to consider multiple kv_cache_groups
        attn_metadata: FlashAttentionMetadata | TreeAttentionMetadata = \
            self.runner.attn_groups[0][0].metadata_builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        # At this moment, we assume all eagle layers belong to the same KV
        # cache group, thus using the same attention metadata.
        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata
        if self.use_cuda_graph and \
                num_tokens <= self.cudagraph_batch_sizes[-1]:
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)
        else:
            num_input_tokens = num_tokens

        # Update attention metadata and prepare inputs.
        if self.is_self_speculation:
            positions = attn_metadata.seq_lens.long()
            # Mask out the position ids that exceed the max model length.
            # Otherwise, we may get out-of-range error in RoPE.
            exceeds_max_model_len = positions >= self.max_model_len
            clamped_positions = torch.where(exceeds_max_model_len, 0,
                                            positions)

            attn_metadata.num_actual_tokens = num_tokens
            attn_metadata.max_query_len = 1
            attn_metadata.query_start_loc = self.arange[:batch_size + 1]
            attn_metadata.max_seq_len += 1
            attn_metadata.seq_lens += 1
            # Consider max model length.
            attn_metadata.max_seq_len = min(attn_metadata.max_seq_len,
                                            self.max_model_len)
            # For the requests that exceed the max model length, we set the
            # sequence length to 1 to minimize their overheads in attention.
            attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)

            # Compute the slot mapping.
            block_numbers = clamped_positions // self.block_size
            block_ids = attn_metadata.block_table.gather(
                dim=1, index=block_numbers.view(-1, 1))
            block_ids = block_ids.view(-1)
            attn_metadata.slot_mapping = (block_ids * self.block_size +
                                          clamped_positions % self.block_size)
            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            attn_metadata.slot_mapping.masked_fill_(exceeds_max_model_len,
                                                    PADDING_SLOT_ID)
            # copy inputs to buffer for cudagraph
            self.positions[:batch_size] = clamped_positions
            positions = self.positions[:num_input_tokens]
        else:
            raise NotImplementedError

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
        logits: torch.Tensor = self.model.compute_logits(hidden_states, None)

        if isinstance(attn_metadata, TreeAttentionMetadata):
            raise NotImplementedError
            # Draft using tree attention.
            draft_token_ids_list = self.propose_tree(
                batch_size=batch_size,
                logits=logits,
                positions=positions,
                hidden_states=hidden_states,
                common_attn_metadata=common_attn_metadata,
            )
            # [batch_size, num_tree_tokens]
            return torch.cat(draft_token_ids_list, dim=1)

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1 or True:
            # [batch_size, vocab_size]
            self.draft_probs = \
                compute_probs(
                    logits,
                    self.arange[1: batch_size + 1],
                    sampling_metadata
                )
            sampler_output = self._sample(logits, sampling_metadata)
            # [batch_size, 1]
            return sampler_output.sampled_token_ids.view(batch_size, 1)

        # TODO: Currently, MTP module released by deepseek only has
        # one layer. Adapt this code to support multiple layers once
        # there's a multi-layer MTP module.
        assert isinstance(attn_metadata, self.allowed_attn_types)

        # return draft_token_ids
        assert False

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
        # Check whether to use self-speculation or a separate draft model.
        if self.vllm_config.speculative_config.model == \
                self.vllm_config.model_config.model:
            # Self-speculation.
            self.model = target_model
            self.attn_layer_names = set(
                get_layers_from_vllm_config(self.vllm_config, Attention).keys()
            )
            self.is_self_speculation = True
            # Reuse GPU buffers in runner.
            self.input_ids = self.runner.input_ids.gpu
            self.positions = self.runner.positions.gpu
            return

        # Use a separate draft model.
        raise NotImplementedError

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
    ) -> None:
        with set_forward_context(None, self.vllm_config,
                                 num_tokens=num_tokens):
            input_ids = self.input_ids[:num_tokens]
            inputs_embeds = None
            self.model(
                input_ids=input_ids,
                positions=self.positions[:num_tokens],
                hidden_states=self.hidden_states[:num_tokens],
                inputs_embeds=inputs_embeds,
            )
