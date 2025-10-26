# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from vllm.config import CompilationLevel, VllmConfig
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.utils import is_pin_memory_available
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import compute_probs
from vllm.v1.spec_decode.attn_overrider import build_attention_overrider
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
        self.allowed_attn_types = (TritonAttentionMetadata,)

        # Used to resolve the request ID to metadata index mapping.
        self.req_id_to_metadata: dict[str, int] = {}
        self.metadata_in_use: list[bool] = [False] * max_batch_size

    def _sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata
    ) -> SamplerOutput:
        """
        Sample from the logits with sampling_metadata.
        """
        return self.runner.sampler(logits, sampling_metadata)

    def remap_metadata(
        self,
        req_id_to_index: dict[str, int],
        num_draft_tokens: np.ndarray,
    ):
        # 1. Update the index to metadata mapping.
        # Remove finished requests.
        finished_req_ids = \
            set(self.req_id_to_metadata.keys()) - set(req_id_to_index.keys())
        for req_id in finished_req_ids:
            metadata_index = self.req_id_to_metadata[req_id]
            self.metadata_in_use[metadata_index] = False
            del self.req_id_to_metadata[req_id]

        # Build the index to request ID mapping.
        # Number of requests is small, so we use a simple loop here.
        index_to_req_id = {v: k for k, v in req_id_to_index.items()}

        # Build the index to metadata tensor.
        # TODO (Yikang): Can be optimized by checking against the history.
        index_to_metadata = []
        is_new_req = []
        metadata_ptr = 0
        for idx in range(len(index_to_req_id)):
            req_id = index_to_req_id[idx]
            metadata_index = self.req_id_to_metadata.get(req_id, -1)
            # If metadata_index is -1, it means a new request
            if metadata_index == -1:
                while self.metadata_in_use[metadata_ptr]:
                    metadata_ptr += 1
                metadata_index = metadata_ptr
                self.req_id_to_metadata[req_id] = metadata_index
                self.metadata_in_use[metadata_index] = True
                metadata_ptr += 1
                is_new_req.append(True)
            else:
                is_new_req.append(False)
            index_to_metadata.append(metadata_index)
        self.attention_overrider.update_metadata_mappings(
            index_to_metadata, is_new_req)

        # 2. Remap the draft probabilities based on the new request IDs.
        # Major computation is conducted on CPU.
        # Will be conducted in the runner's CPU logic region.
        # Early exit if no draft tokens.
        if num_draft_tokens is None:
            return

        # Skip remmapping if possible.
        prev_req_id_to_index = self.req_id_to_index
        if all(n == self.num_speculative_tokens for n in num_draft_tokens) \
                and req_id_to_index == prev_req_id_to_index:
            self.cu_gather_indices = None
            return

        gather_indices = []
        for idx, count in enumerate(num_draft_tokens):
            req_id = index_to_req_id[idx]
            # Nothing to gather for this request.
            if count == 0:
                continue
            assert req_id in prev_req_id_to_index
            start = prev_req_id_to_index[req_id] * self.vllm_config.\
                speculative_config.num_speculative_tokens
            end = start + count
            gather_indices += list(range(start, end))
        self.cu_gather_indices = torch.tensor(
            gather_indices, pin_memory=self.runner.pin_memory
        ).to(self.device, non_blocking=True)

    @property
    def draft_probs(self) -> torch.Tensor:
        buffer = self._draft_probs[:self.batch_size].view(-1, self.vocab_size)
        return buffer if self.cu_gather_indices is None else \
            buffer.index_select(0, self.cu_gather_indices)

    @property
    def sampled_token_ids(self) -> torch.Tensor:
        return self._sampled_token_ids[:self.batch_size]

    def propose(
        self,
        next_token_ids: torch.Tensor,
        common_attn_metadata: dict[int, CommonAttentionMetadata],
        sampling_metadata: SamplingMetadata,
        mm_embeds: Optional[list[torch.Tensor]] = None,
    ) -> torch.Tensor:
        # For prototyping, mm, dp, pp, and kv_connector etc. are not supported.
        # TODO(Yikang): Add them back if needed.

        # Determine the batch size and the number of tokens to compute.
        num_tokens = next_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]
        self.batch_size = batch_size
        self.req_id_to_index = dict(self.runner.input_batch.req_id_to_index)
        last_token_indices = self.arange[:batch_size]
        self.input_ids[:batch_size] = next_token_ids

        # Do not draft if the attention overrider is not ready.
        # Yikang: MUST be placed after registering req_id_to_index.
        if not self.attention_overrider.ready_to_draft():
            return None

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

        # Build attention metadata for all attention groups.
        attn_metadatas: list[TritonAttentionMetadata] = []
        per_layer_attn_metadata = {}
        for kv_cache_group_id, attn_metadata in common_attn_metadata.items():
            # FIXME: Now assume each kv cache group only has one attn group.
            attn_group = self.runner.attn_groups[kv_cache_group_id][0]
            # Build attention metadata for this attn group.
            attn_metadata = attn_group.metadata_builder.build(
                common_prefix_len=0,
                common_attn_metadata=attn_metadata,
            )
            assert isinstance(attn_metadata, self.allowed_attn_types)
            attn_metadatas.append(attn_metadata)
            # Register per-layer attention metadata.
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

        # Determine the number of input tokens for cudagraph.
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
        self.positions[:batch_size] = clamped_positions
        positions = self.positions[:num_input_tokens]

        # Update the attention metadata.
        for attn_metadata in attn_metadatas:
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
            attn_metadata.slot_mapping = (block_ids * self.block_size +
                                          clamped_positions % self.block_size)
            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            attn_metadata.slot_mapping.masked_fill_(
                exceeds_max_model_len, PADDING_SLOT_ID)

        if self.is_multimodal_model:
            raise NotImplementedError
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
        self.attention_overrider.set_draft_step(1)
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
            self.attention_overrider.set_draft_step(0)
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
            for attn_metadata in attn_metadatas:
                attn_metadata.max_seq_len += 1
                attn_metadata.seq_lens += 1
                # Consider max model length.
                attn_metadata.max_seq_len = \
                    min(attn_metadata.max_seq_len, self.max_model_len)
                # For the requests that exceed the max model length, we set the
                # sequence length to 1 to minimize the overheads in attention.
                attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)
                # Compute the slot mapping.
                block_numbers = positions // self.block_size
                block_ids = attn_metadata.block_table.gather(
                    dim=1, index=block_numbers.view(-1, 1))
                block_ids = block_ids.view(-1)
                attn_metadata.slot_mapping = \
                    (block_ids * self.block_size + positions % self.block_size)
                # Mask out the slot mappings that exceed the max model length.
                # Otherwise, the KV cache will be inadvertently updated with
                # the padding tokens.
                attn_metadata.slot_mapping.masked_fill_(
                    exceeds_max_model_len, PADDING_SLOT_ID)

            # Run the model.
            # Use persistent buffers for CUDA graphs.
            self.attention_overrider.set_draft_step(step + 1)
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
        self.attention_overrider.set_draft_step(0)
        return self.sampled_token_ids

    def prepare_inputs(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        # [batch_size]
        num_rejected_tokens: torch.Tensor
    ) -> CommonAttentionMetadata:
        """
        This function is used to prepare the inputs for the spec decode.
        It updates to the common_attn_metadata to account for the rejected
        tokens (and newly sampled tokens).
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

        return CommonAttentionMetadata(
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

    def load_model(self, target_model: nn.Module) -> None:
        # Load the target model.
        self.model = target_model

        # Reuse runner buffers for inputs and positions.
        # This is a MUST when using CUDA graphs.
        self.input_ids = self.runner.input_ids.gpu
        self.positions = self.runner.positions.gpu

        # Build the attention overrider.
        self.attention_overrider = build_attention_overrider(
            vllm_config=self.vllm_config,
            device=self.device,
        )
