import torch
import triton
import triton.language as tl

from vllm.attention.ops.triton_ssr_attention import ssr_unified_attention
from vllm.config import VllmConfig
from vllm.vllm_flash_attn import flash_attn_with_kvcache
from .abstract import AbstractAttentionOverrider


@triton.jit
def gather_kv_pair_kernel_reversed(
    topk_indices_ptr,
    context_lens_ptr,
    num_next_indices_ptr,
    block_table_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_output_ptr,
    v_output_ptr,
    batch_size,
    topk,
    num_kv_heads,
    head_dim,
    page_size,
    stride_si_batch,
    stride_si_topk,
    stride_bt_batch,
    stride_bt_page,
    stride_cache_page,
    stride_cache_pos,
    stride_cache_head,
    stride_cache_dim,
    stride_out_batch,
    stride_out_pos,
    stride_out_head,
    stride_out_dim,
    HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    num_positions = topk
    positions_per_batch = num_positions * num_kv_heads
    batch_idx = pid // positions_per_batch
    remainder = pid % positions_per_batch
    pos_idx = remainder // num_kv_heads
    head_idx = remainder % num_kv_heads
    if batch_idx >= batch_size or pos_idx >= num_positions:
        return
    # Load parameters for this batch (assuming contiguous)
    context_len = tl.load(context_lens_ptr + batch_idx).to(tl.int64)
    num_next = tl.load(num_next_indices_ptr + batch_idx).to(tl.int32)

    # Compute valid_len as context_len.clamp(max=topk)
    valid_len = tl.minimum(context_len, topk)

    # Calculate the new effective valid length after insertion
    total_len = valid_len + num_next
    effective_valid_len = tl.minimum(total_len, topk)

    # Skip if this position is beyond effective valid length
    if pos_idx >= effective_valid_len:
        return

    # Determine the sequence position to retrieve
    if total_len <= topk:
        # Case 1: enough space
        if pos_idx < valid_len:
            # Original valid index
            seq_pos = tl.load(
                topk_indices_ptr + batch_idx * stride_si_batch +
                pos_idx * stride_si_topk
            ).to(tl.int64)
        else:
            # New index: pos_idx in [valid_len, valid_len + num_next)
            offset_in_new = pos_idx - valid_len
            seq_pos = context_len + offset_in_new
    else:
        # Case 2: overflow, keep first (topk - num_next) original indices
        cutoff = topk - num_next
        if pos_idx < cutoff:
            # Original valid index (kept)
            seq_pos = tl.load(
                topk_indices_ptr + batch_idx * stride_si_batch +
                pos_idx * stride_si_topk
            ).to(tl.int64)
        else:
            # New index: pos_idx in [topk - num_next, topk)
            offset_in_new = pos_idx - cutoff
            seq_pos = context_len + offset_in_new
    dim_offsets = tl.arange(0, HEAD_DIM)
    mask = dim_offsets < head_dim
    # Reverse the output position: first input goes to last output position
    reversed_pos_idx = topk - 1 - pos_idx
    out_offset = (
        batch_idx * stride_out_batch +
        reversed_pos_idx * stride_out_pos +
        head_idx * stride_out_head
    )
    k_out_base = k_output_ptr + out_offset
    v_out_base = v_output_ptr + out_offset
    # Convert sequence position to physical page
    page_idx = seq_pos // page_size
    page_offset = seq_pos % page_size
    physical_page = tl.load(
        block_table_ptr + batch_idx * stride_bt_batch +
        page_idx * stride_bt_page
    )
    cache_offset = (
        physical_page * stride_cache_page +
        page_offset * stride_cache_pos +
        head_idx * stride_cache_head
    )
    k_base = k_cache_ptr + cache_offset
    v_base = v_cache_ptr + cache_offset
    k_vals = tl.load(k_base + dim_offsets * stride_cache_dim, mask, other=0.0)
    v_vals = tl.load(v_base + dim_offsets * stride_cache_dim, mask, other=0.0)
    # Store at reversed position
    tl.store(k_out_base + dim_offsets * stride_out_dim, k_vals, mask=mask)
    tl.store(v_out_base + dim_offsets * stride_out_dim, v_vals, mask=mask)


def gather_kv_cache_triton(
    topk_indices: torch.Tensor,
    context_lens: torch.Tensor,
    num_next_indices: torch.Tensor,
    block_table: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, topk = topk_indices.shape
    _num_pages, page_size, num_kv_heads, head_dim = k_cache.shape
    grid = (batch_size * topk * num_kv_heads,)
    HEAD_DIM = triton.next_power_of_2(head_dim)
    gather_kv_pair_kernel_reversed[grid](
        topk_indices,
        context_lens,
        num_next_indices,
        block_table,
        k_cache,
        v_cache,
        k_out,
        v_out,
        batch_size,
        topk,
        num_kv_heads,
        head_dim,
        page_size,
        topk_indices.stride(0),
        topk_indices.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        k_out.stride(0),
        k_out.stride(1),
        k_out.stride(2),
        k_out.stride(3),
        HEAD_DIM=HEAD_DIM,
    )
    return k_out, v_out


@triton.jit
def sum_reduce_with_seqlen_mask_kernel(
    input_ptr,  # [sum(lens), num_q_head, max_seqlen]
    output_ptr,  # [batch_size, max_seqlen] - PRE-FILLED with -inf
    cu_seqlens_q_ptr,  # [batch_size + 1]
    seqlens_ptr,  # [batch_size] - valid seqlen for each batch
    num_q_head: tl.constexpr,  # int
    stride_input_token,
    stride_input_head,
    stride_input_seq,
    stride_output_batch,
    stride_output_seq,
    BLOCK_SEQ: tl.constexpr,
):
    """
    Sum reduction over tokens and heads, respecting per-sequence valid lengths.

    OPTIMIZATION: Output buffer is PRE-FILLED with -inf, so this kernel only
    needs to write valid positions. Invalid positions remain as -inf.

    Args:
        input_ptr: [sum(lens), num_q_head, max_seqlen]
                   Where sum(lens) is total number of tokens across batch
                   lens[i] = number of tokens in sequence i (typically 1-16)
        output_ptr: [batch_size, max_seqlen] - PRE-FILLED with -inf
        cu_seqlens_q_ptr: [batch_size + 1] cumulative lens
                        e.g., [0, 5, 12, 15] means lens = [5, 7, 3]
        seqlens_ptr: [batch_size] valid sequence positions
    """
    batch_idx = tl.program_id(0)
    seq_block_idx = tl.program_id(1)

    # Load token boundaries for this batch
    token_start = tl.load(cu_seqlens_q_ptr + batch_idx)
    token_end = tl.load(cu_seqlens_q_ptr + batch_idx + 1)
    num_tokens = token_end - token_start  # lens[batch_idx], typically 1-16

    # Load valid sequence length
    valid_seqlen = tl.load(seqlens_ptr + batch_idx)

    # Compute sequence positions for this block and clamp to valid range
    seq_offsets = seq_block_idx * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    seq_is_valid = seq_offsets < valid_seqlen

    # Initialize accumulator for valid positions only
    acc = tl.zeros((BLOCK_SEQ,), dtype=tl.float32)

    # Sum over all tokens in this sequence
    for token_offset in range(num_tokens):
        token_idx = token_start + token_offset

        # Sum over all query heads
        for head_idx in range(num_q_head):
            input_offset = (
                token_idx * stride_input_token +
                head_idx * stride_input_head +
                seq_offsets * stride_input_seq
            )

            vals = tl.load(
                input_ptr + input_offset,
                mask=seq_is_valid,
                other=0.0
            )

            # Accumulate
            acc += vals

    # Store only valid positions (invalid ones remain as pre-filled -inf)
    output_offset = \
        batch_idx * stride_output_batch + seq_offsets * stride_output_seq
    tl.store(
        output_ptr + output_offset,
        acc,
        mask=seq_is_valid
    )


def sum_reduce_with_masking(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seqlens: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:
    """
    Sum reduction over tokens and heads with per-sequence masking.

    Args:
        input_tensor: [sum(lens), num_q_head, max_seqlen]
                     sum(lens) = total tokens, typically batch_size * (1-16)
        output_tensor: [batch_size, max_seqlen] - PRE-FILLED with -inf
        cu_seqlens: [batch_size + 1] - cumulative token counts
        seqlens: [batch_size] - valid sequence length for each batch element
        max_seqlen: int - maximum sequence length (32k-64k)

    Returns:
        output_tensor with:
        - [:, :seqlen[i]] = sum of input over tokens and heads
        - [:, seqlen[i]:] = -inf (pre-filled, kernel doesn't touch these)

    Example:
        batch_size = 3
        lens = [5, 7, 3]  # tokens per sequence
        cu_seqlens = [0, 5, 12, 15]
        seqlens = [2048, 32768, 512]  # valid positions per sequence

        input: [15, 8, 65536]  # 15 tokens, 8 heads, 64k max_seqlen
        output: [3, 65536] pre-filled with -inf

        Kernel writes:
        output[0, :2048] = sum(input[0:5, :, :2048])  # sum 5 tokens, 8 heads
        output[1, :32768] = sum(input[5:12, :, :32768])
        output[2, :512] = sum(input[12:15, :, :512])

        Rest remains -inf (pre-filled)
    """
    batch_size = cu_seqlens.shape[0] - 1
    num_q_head = input_tensor.shape[1]

    # Use 256-element blocks for good memory coalescing
    BLOCK_SEQ = 256
    num_seq_blocks = triton.cdiv(max_seqlen, BLOCK_SEQ)

    grid = (batch_size, num_seq_blocks)

    sum_reduce_with_seqlen_mask_kernel[grid](
        input_tensor,
        output_tensor,
        cu_seqlens,
        seqlens,
        num_q_head,
        input_tensor.stride(0),
        input_tensor.stride(1),
        input_tensor.stride(2),
        output_tensor.stride(0),
        output_tensor.stride(1),
        BLOCK_SEQ=BLOCK_SEQ,
    )

    return output_tensor


class SSRAttentionOverrider(AbstractAttentionOverrider):
    """An attention overrider that implements SSR sparse attention."""

    def __init__(
        self,
        # Base class args.
        vllm_config: VllmConfig,
        device: torch.device,
        # SSR-specific args.
        top_k: int = 256,
        **_kwargs
    ):
        super().__init__(vllm_config, device)
        self.topk = top_k

        # Whether to enable the top-k mechanism.
        self.enable_topk = False

        # The number of speculative tokens
        self.num_speculative_tokens = \
            vllm_config.speculative_config.num_speculative_tokens
        # Get the number of layers.
        self.num_layers = self.layer_indexer.num_layers
        # Get the head dimension and number of heads.
        self.num_kv_heads = self.vllm_config.model_config.get_num_kv_heads(
            self.vllm_config.parallel_config)
        self.num_q_heads = \
            self.vllm_config.model_config.get_num_attention_heads(
                self.vllm_config.parallel_config)
        self.head_dim = self.vllm_config.model_config.get_head_size()

        # Pre-allocation of cuda buffers.
        # Used for indexing.
        self.arrange = torch.arange(
            vllm_config.scheduler_config.max_num_seqs,
            dtype=torch.int32, device=device)
        # Store context lengths first, then store the left padding size.
        self.context_lens = torch.empty(
            vllm_config.scheduler_config.max_num_seqs,
            dtype=torch.int32, device=device)

        # Lazy allocation of topk slots and KV caches.
        self.topk_seq_indices: list[torch.Tensor] = []
        self.topk_k_caches: list[torch.Tensor] = []
        self.topk_v_caches: list[torch.Tensor] = []

    def ready_to_draft(self):
        return self.enable_topk

    def _draft_kv_insert(
        self, key: torch.Tensor, value: torch.Tensor, *_args, **_kwargs
    ):
        # Insert later in the draft attention function.
        self.last_k = key[:self.batch_size]
        self.last_v = value[:self.batch_size]

    def _overriden_kv_insert(self, *args, **kwargs):
        # Get the current layer index.
        self.layer_index, self.should_override, _ = self._get_layer()

        # Determine whether to insert into the topk cache.
        if self.enable_topk and self.in_draft and self.should_override:
            # Insert into the topk cache.
            self._draft_kv_insert(*args, **kwargs)
        else:
            # Insert into the full cache.
            self.original_kv_insert_func(*args, **kwargs)

    def _retrieve_topk_kv_cache(
        self,
        layer_index: int,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        topk_seq_indices: torch.Tensor,
    ):
        # Expand k and v to have the batch dimension.
        k_out = self.topk_k_caches[layer_index][:self.batch_size]
        v_out = self.topk_v_caches[layer_index][:self.batch_size]
        gather_kv_cache_triton(
            topk_seq_indices,
            self.context_lens[:self.batch_size],
            self.num_accepted_tokens,
            self.block_table, k_cache, v_cache, k_out, v_out,
        )

    def _draft_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        seqused_k: torch.Tensor,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        rotary_interleaved=True,
        alibi_slopes=None,
        *_args, **_kwargs
    ) -> bool:
        # Get the current layer index.
        layer_index = self.layer_index

        # Build the top-k kv cache for the next few speculative steps.
        if self.current_draft_step == 1:
            # Things to do only once in the first speculative step.
            if layer_index == self.min_effective_layer_index:
                # Determine how many new indices to add for each sequence.
                self.num_accepted_tokens: torch.Tensor = \
                    seqused_k - self.context_lens[:self.batch_size] - 1
                self.num_accepted_tokens.clamp_min_(0)
                self.cache_leftpad: torch.Tensor = \
                    self.topk - self.context_lens[:self.batch_size] - \
                    self.num_accepted_tokens
                self.cache_leftpad.clamp_min_(0)

            # Gather the top-k KV from the full cache.
            # Need to be done for all layers, but only in the first step.
            topk_seq_indices = \
                self.topk_seq_indices[layer_index][:self.batch_size]

            # Gather the top-k KV from the full kv cache.
            # They will be reused in the next few speculative steps.
            self._retrieve_topk_kv_cache(layer_index, k, v, topk_seq_indices)

        # Use the top-k KV cache for attention.
        flash_attn_with_kvcache(
            q=q.unsqueeze(1),
            k_cache=self.topk_k_caches[layer_index][:self.batch_size],
            v_cache=self.topk_v_caches[layer_index][:self.batch_size],
            out=out.unsqueeze(1),
            k=self.last_k.unsqueeze(1),
            v=self.last_v.unsqueeze(1),
            cache_seqlens=self.topk + self.current_draft_step - 1,
            cache_leftpad=self.cache_leftpad,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            rotary_interleaved=rotary_interleaved,
            alibi_slopes=alibi_slopes,
        )
        # Free the last_k and last_v references.
        self.last_k, self.last_v = None, None

    def _may_allocate_topk_mem(self):
        # Do nothing if already allocated and the batch size is sufficient.
        batch_size = self.batch_size
        if len(self.topk_seq_indices) and \
                self.topk_scores_dummy.shape[0] >= batch_size:
            return

        # Allocate the attention score buffer, shared by all layers.
        self.attn_scores = torch.empty(
            (batch_size * (self.num_speculative_tokens + 1),
             self.num_q_heads, self.max_model_len),
            dtype=self.vllm_config.model_config.dtype, device=self.device)

        # Allocate the top-k slots.
        self.topk_seq_indices = [torch.empty(
            (batch_size, self.topk), dtype=torch.long, device=self.device)
            if idx in self.effective_layer_indices else None
            for idx in range(self.num_layers)]
        self.topk_scores_dummy = torch.empty(
            (batch_size, self.topk), dtype=torch.float32, device=self.device)

        # Allocate the top-k KV caches.
        self.topk_k_caches = [torch.empty(
            (batch_size, self.topk + self.num_speculative_tokens,
             self.num_kv_heads, self.head_dim),
            dtype=self.vllm_config.model_config.dtype, device=self.device)
            if idx in self.effective_layer_indices else None
            for idx in range(self.num_layers)]
        self.topk_v_caches = [torch.empty(
            (batch_size, self.topk + self.num_speculative_tokens,
             self.num_kv_heads, self.head_dim),
            dtype=self.vllm_config.model_config.dtype, device=self.device)
            if idx in self.effective_layer_indices else None
            for idx in range(self.num_layers)]

    def _overriden_attention(self, *args, **kwargs):
        # Determine whether to enable topk attention.
        if not self.in_draft and self.layer_index == 0:
            max_seqlen_q: int = kwargs['max_seqlen_q']
            self.enable_topk = max_seqlen_q <= self.num_speculative_tokens + 1

        # Upddate metadata at the first effective layer.
        if self.enable_topk and not self.in_draft and \
                self.layer_index == self.min_effective_layer_index:
            seqlens: torch.Tensor = kwargs['seqused_k']
            cu_seqlens_q: torch.Tensor = kwargs['cu_seqlens_q']
            self.batch_size = seqlens.shape[0]
            self.block_table = kwargs['block_table'][:self.batch_size]
            self.max_seqlen_q: int = kwargs['max_seqlen_q']
            # Register number of query tokens.
            self.num_query_tokens = cu_seqlens_q[-1]
            # Allocate memory if needed.
            self._may_allocate_topk_mem()
            # Track the context lengths for each sequence.
            # Ultimate goal is to determine top-k slots for each sequence.
            seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            self.context_lens[:self.batch_size] = seqlens - seqlens_q + 1
            # Check whether it is uniform decode.
            self.uniform_decode = \
                self.max_seqlen_q * self.batch_size == self.num_query_tokens

        # Determine which attention function to use.
        # If not using top-k, fall back to the original attention function.
        if not self.enable_topk or not self.should_override:
            self.original_attention_func(*args, **kwargs)
            return

        # Use top-k attention when it is time to draft.
        if self.in_draft:
            # Conduct draft attention with top-k mechanism.
            self._draft_attention(*args, **kwargs)
            return

        # Target attention with top-k collection.
        attn_scores = self.attn_scores[:self.num_query_tokens]
        ssr_unified_attention(
            *args, **kwargs,
            attn_scores=attn_scores.fill_(-torch.inf),
        )

        # Now extract the topk indices from attn_scores
        max_seqlen_k = kwargs['max_seqlen_k']
        focus_len = max(max_seqlen_k, self.topk)
        # (cu_seqlens_q[-1], num_query_heads, focus_len), not contiguous
        attn_scores = attn_scores[..., :focus_len]

        # Decide whether this is uniform_decode or not
        # If uniform_decode, we can do a simple reshape and sum-reduce
        # Otherwise we need to do a more complex reduction with triton
        if self.uniform_decode:
            attn_scores = attn_scores.view(self.batch_size, -1, focus_len)
            attn_scores_reduced = attn_scores.sum(dim=1, dtype=torch.float32)
        else:
            # Non-uniform decode
            attn_scores_reduced = torch.full(
                (self.batch_size, focus_len),
                float('-inf'),
                dtype=torch.float32,
                device=attn_scores.device,
            )
            cu_seqlens_q = kwargs['cu_seqlens_q']
            seqused_k = kwargs['seqused_k']
            sum_reduce_with_masking(
                attn_scores,
                attn_scores_reduced,
                cu_seqlens_q,
                seqused_k,
                focus_len,
            )

        # Compute topk indices and copy to output
        topk_seq_indices = self.topk_seq_indices[self.layer_index]
        topk_seq_indices = topk_seq_indices[:self.batch_size]
        topk_scores_dummy = self.topk_scores_dummy[:self.batch_size]
        torch.topk(
            attn_scores_reduced, self.topk, -1, largest=True, sorted=True,
            out=(topk_scores_dummy, topk_seq_indices)
        )
