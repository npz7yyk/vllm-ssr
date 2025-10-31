import torch
import triton
import triton.language as tl

from vllm.config import VllmConfig
from .abstract import AbstractAttentionOverrider


@triton.jit
def _compute_min_max(
    index_to_metadata,  # [max_num_seqs]
    keys_ptr,           # [block_num, block_size, num_head, head_dim]
    min_keys_ptr,       # [max_num_seqs, num_blocks, num_head, head_dim]
    max_keys_ptr,       # [max_num_seqs, num_blocks, num_head, head_dim]
    block_table_ptr,    # [batch_size, max_blocks] - block IDs for each request
    update_begins,      # [batch_size] - start index for each request
    update_ends,        # [batch_size] - end index for each request
    block_size,
    batch_size,
    max_blocks,
    num_head,
    head_dim,
):
    """
    Triton kernel to compute min/max of keys for a batch of blocks.
    
    For each metadata (metadata ID) and its range [update_begin:update_end], 
    extracts the corresponding blocks from block_table and computes element-wise min/max
    of keys for each block. Directly stores results in min_keys and max_keys.
    """

    head_id = tl.program_id(0)      # head dimension: [0, num_head)
    head_dim_id = tl.program_id(1)  # head_dim dimension: [0, head_dim)
    metadata_id = tl.program_id(2)  # metadata dimension: [0, batch_size)
    
    if head_id >= num_head or head_dim_id >= head_dim or metadata_id >= batch_size:
        return
    
    # Use metadata positions to do the actual computation, not batch index
    metadata = tl.load(index_to_metadata + metadata_id)
    update_begin = tl.load(update_begins + metadata)
    update_end = tl.load(update_ends + metadata)
    
    # Process each block in the range [update_begin:update_end]
    for block_idx in range(update_begin, update_end):
        block_table_offset = metadata * max_blocks + block_idx
        block_id = tl.load(block_table_ptr + block_table_offset)
        
        block_start = block_id * block_size
        
        offset = block_start * num_head * head_dim + head_id * head_dim + head_dim_id
        min_val = tl.load(keys_ptr + offset)
        max_val = min_val
        
        # Iterate through the rest of the block to find min/max
        for i in range(1, block_size):
            offset = (block_start + i) * num_head * head_dim + head_id * head_dim + head_dim_id
            val = tl.load(keys_ptr + offset, eviction_policy='evict_last')
            min_val = tl.cast(tl.minimum(min_val, val), min_val.dtype)
            max_val = tl.cast(tl.maximum(max_val, val), max_val.dtype)
        
        # Store results directly in min_keys and max_keys
        out_offset = metadata * max_blocks * num_head * head_dim + \
                     block_idx * num_head * head_dim + \
                     head_id * head_dim + head_dim_id
        tl.store(min_keys_ptr + out_offset, min_val)
        tl.store(max_keys_ptr + out_offset, max_val)


@triton.jit
def _compute_block_attn_weight(
    index_to_metadata,      # [max_num_seqs]
    min_max_keys_ptr,       # [2, batch_size, max_blocks, num_head, head_dim]
    q_ptr,                  # [batch_size, num_head, q_per_group, head_dim]
    output_ptr,             # [batch_size, max_blocks]
    valid_lengths,          # [batch_size]
    batch_size: tl.constexpr,
    max_blocks: tl.constexpr,
    num_head: tl.constexpr,
    head_dim: tl.constexpr,
    q_per_group: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Triton kernels to compute block attention weights.
    
    Does the same thing as  einsum("bhqd,mblhd -> mblhq") then amax(dim=0).mean(dim=(1,2))

    """
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    
    if batch_idx >= batch_size or block_idx >= max_blocks:
        return
    
    # Use metadata positions to do the actual computation, not batch index
    metadata = tl.load(index_to_metadata + batch_idx)
    valid_len = tl.load(valid_lengths + metadata)
    
    if block_idx >= valid_len:
        return
    
    score_sum = 0.0

    # Iterate over heads in blocks for better cache utilization
    for head_start in range(0, num_head, BLOCK_HEAD):
        head_range = tl.arange(0, BLOCK_HEAD)
        head_idx = head_start + head_range
        head_mask = head_idx < num_head

        # Iterate over q_per_group dimension
        for q_start in range(0, q_per_group, BLOCK_Q):
            q_range = tl.arange(0, BLOCK_Q)
            q_idx = q_start + q_range
            q_mask = q_idx < q_per_group
            
            # Compute dot products for min and max keys
            min_dot_product = tl.zeros((BLOCK_HEAD, BLOCK_Q), dtype=tl.float32)
            max_dot_product = tl.zeros((BLOCK_HEAD, BLOCK_Q), dtype=tl.float32)
            
            # Iterate over head_dim with vectorized loads to compute complete dot products
            for d_start in range(0, head_dim, BLOCK_D):
                d_range = tl.arange(0, BLOCK_D)
                d_idx = d_start + d_range
                d_mask = d_idx < head_dim
                
                min_key_offset = (0 * batch_size * max_blocks * num_head * head_dim +
                              metadata * max_blocks * num_head * head_dim +
                              block_idx * num_head * head_dim +
                              head_idx[:, None] * head_dim +
                              d_idx[None, :])
                min_keys = tl.load(min_max_keys_ptr + min_key_offset, 
                               mask=head_mask[:, None] & d_mask[None, :],
                               other=0.0)
    
                max_key_offset = (1 * batch_size * max_blocks * num_head * head_dim +
                              metadata * max_blocks * num_head * head_dim +
                              block_idx * num_head * head_dim +
                              head_idx[:, None] * head_dim +
                              d_idx[None, :])
                max_keys = tl.load(min_max_keys_ptr + max_key_offset, 
                               mask=head_mask[:, None] & d_mask[None, :],
                               other=0.0)

                q_offset = (batch_idx * num_head * q_per_group * head_dim +
                            head_idx[:, None, None] * q_per_group * head_dim +
                            q_idx[None, :, None] * head_dim +
                            d_idx[None, None, :])
                queries = tl.load(q_ptr + q_offset,
                                  mask=head_mask[:, None, None] & q_mask[None, :, None] & d_mask[None, None, :],
                                  other=0.0)
                
                # Accumulate dot products
                min_dot_product += tl.sum(min_keys[:, None, :] * queries, axis=2)
                max_dot_product += tl.sum(max_keys[:, None, :] * queries, axis=2)
            
            # Take element-wise max between min and max dot products and sum
            combined_mask = head_mask[:, None] & q_mask[None, :]
            max_scores = tl.maximum(min_dot_product, max_dot_product)
            # Sum only valid elements
            score_sum += tl.sum(tl.where(combined_mask, max_scores, 0.0))
    
    # Compute mean over (num_head * q_per_group), accomodate to vllm
    total_elements = num_head * q_per_group
    mean_score = score_sum / total_elements

    output_offset = batch_idx * max_blocks + block_idx
    tl.store(output_ptr + output_offset, mean_score)


@triton.jit
def _copy_blocks_to_sparse_table(
    top_k_block_idx,        # [batch_size, k]
    block_table_ptr,        # [batch_size, max_blocks]
    sparse_block_table_ptr, # [batch_size, output_len]
    valid_lengths,          # [batch_size]
    num_total_blocks,       # [batch_size]
    batch_size,
    k,
    output_len,
    max_blocks,
):
    """
    Triton kernel to copy top-k selected blocks and potential last 1-2 blocks
    (potential 2 if draft step is larger than 16) to sparse_block_table.
    
    The kernel first copies the selected blocks to the front of the queue and
    copies the last one or two invalid blocks(not validated yet) to the queue.
    """
    batch_id = tl.program_id(0)
    col_idx = tl.program_id(1)
    
    if batch_id >= batch_size or col_idx >= output_len:
        return
    
    # Use metadata positions to do the actual computation, not batch index
    valid_len = tl.load(valid_lengths + batch_id)
    total_blocks = tl.load(num_total_blocks + batch_id)
    # Copy topk.
    if col_idx < k:
        top_k_idx = tl.load(top_k_block_idx + batch_id * k + col_idx)
        block_id = tl.load(block_table_ptr + batch_id * max_blocks + top_k_idx)
        tl.store(sparse_block_table_ptr + batch_id * max_blocks + col_idx, block_id)
    # Copy from end blocks (last 1-2 blocks)
    else:
        end_idx = col_idx - k
        block_pos = valid_len + end_idx
        if block_pos < total_blocks:
            block_id = tl.load(block_table_ptr + batch_id * max_blocks + block_pos)
            tl.store(sparse_block_table_ptr + batch_id * max_blocks + col_idx, block_id)


class Quest:
    def __init__(self,
                 max_num_seqs: int,
                 num_head: int,
                 head_dim: int,
                 layer_num:int,
                 block_num: int,
                 dtype: str,
                 device: str):
        self.computed_block_num = 0
        self.valid_lengths = torch.zeros(max_num_seqs, dtype=torch.int32, device = device)
        self.prev_valid_lengths = torch.zeros(max_num_seqs, dtype=torch.int32, device = device)
        self.dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        self.num_head = num_head
        self.device = device
        self.head_dim = head_dim
        self.block_num = block_num
        # Set up min, max key cache per block and block mapping.
        self.min_max_keys = \
            torch.zeros((layer_num, 2, max_num_seqs, block_num, num_head, head_dim), 
            dtype=self.dtype, device=device).contiguous()
        # self.max_keys = \
        #     torch.zeros((layer_num, max_num_seqs, block_num, num_head, head_dim), 
        #     dtype=self.dtype, device=device).contiguous()
        
        # Pre-allocate attention weights buffer to avoid repeated allocations
        self.estimation_attn_weights_batch = torch.zeros((max_num_seqs, block_num), 
                                       dtype=self.dtype, device=device).contiguous()
        self.regular = torch.arange(max_num_seqs, device=device)
    
    def compute_min_max_store(self, index_to_metadata, batch_size, block_table, layer_idx, keys, block_size):
        """
        Batched version: compute min/max for multiple blocks and store directly.
        
        """
        update_begin = self.prev_valid_lengths
        update_end = self.valid_lengths
        
        max_blocks = block_table.shape[1]

        grid = (self.num_head, self.head_dim, batch_size)

        # Call the triton kernel to do computation of min and max keys
        _compute_min_max[grid](
            index_to_metadata,
            keys,
            self.min_max_keys[layer_idx, 0],
            self.min_max_keys[layer_idx, 1],
            block_table,
            update_begin,
            update_end,
            block_size,
            batch_size,
            max_blocks,
            self.num_head,
            self.head_dim,
        )

    def compute_block_attn_weight(self,
                                  index_to_metadata: torch.Tensor,
                                  batch_size: int,
                                  q: torch.Tensor,
                                  layer_index: int):
        """
        Batched version: compute estimation attention score for each block and 
        store in estimation_attn_weights_batch

        """
        # Need to zero out the estimation attention
        self.estimation_attn_weights_batch.zero_()

        # Reshape q
        total_query_heads = q.shape[1]
        q_per_group = total_query_heads // self.num_head
        q_reshape = q.view(batch_size, self.num_head, q_per_group, -1)
        
        grid = (batch_size, self.block_num)
        
        BLOCK_HEAD = min(32, self.num_head)
        BLOCK_Q = min(8, q_per_group)
        BLOCK_D = 64 if self.head_dim >= 64 else 32
        
        # Call the triton kernel to do attention score computation
        _compute_block_attn_weight[grid](
            index_to_metadata,
            self.min_max_keys[layer_index],
            q_reshape,
            self.estimation_attn_weights_batch,
            self.valid_lengths,
            batch_size,
            self.block_num,
            self.num_head,
            self.head_dim,
            q_per_group,
            BLOCK_HEAD=BLOCK_HEAD,
            BLOCK_Q=BLOCK_Q,
            BLOCK_D=BLOCK_D,
        )


class QuestAttentionOverrider(AbstractAttentionOverrider):
    """An attention overrider that implements Quest attention."""
    needs_metadata_remapping = True

    def __init__(
        self,
        # Base class args.
        vllm_config: VllmConfig,
        device: torch.device,
        # Quest specific args.
        budget: int,
        max_batch_size: int,
        **_kwargs
    ):
        super().__init__(vllm_config, device)
        max_num_seqs = max_batch_size
        device = self.vllm_config.device_config.device
        num_head = self.vllm_config.model_config.get_num_kv_heads(self.vllm_config.parallel_config)
        head_dim = self.vllm_config.model_config.get_head_size()
        num_layer = self.vllm_config.model_config.get_num_layers(self.vllm_config.parallel_config)
        dtype = self.vllm_config.model_config.dtype
        
        # Calculate maximum blocks per request: ceil(max_model_len / block_size)
        max_model_len = self.vllm_config.model_config.max_model_len
        self.block_size = self.vllm_config.cache_config.block_size
        max_blocks_per_request = (max_model_len + self.block_size - 1) // self.block_size

        # Initialize metadata for estimation use
        self.block_table = None
        self.seq_lens = None
        self.sparse_block_table = torch.zeros((max_num_seqs, max_blocks_per_request), dtype=torch.int32, device=device)
        self.sparse_seq_lens = torch.zeros(max_num_seqs, dtype=torch.int32, device=device)
        self.max_seqlen_k = None
        self.sparse_max_seqlen_k = None
        self.num_speculative_tokens = \
            vllm_config.speculative_config.num_speculative_tokens
        self._enable_quest = False

        # Initialize quest
        self.quest = Quest(max_num_seqs=max_num_seqs,
                           num_head=num_head,
                           head_dim=head_dim,
                           layer_num=num_layer,
                           block_num=max_blocks_per_request,
                           dtype=dtype,
                           device=device)
        self.k = budget // self.block_size

    def ready_to_draft(self):
        return self._enable_quest
    
    def copy_to_buffer(self, top_k_block_idx: torch.Tensor, block_table: torch.Tensor, 
                      valid_length: torch.Tensor, num_total_blocks: torch.Tensor, k: int) -> torch.Tensor:
        """
        Copy top-k selected blocks and last 1-2 blocks to sparse_block_table.

        """
        batch_size = self.batch_size
        max_blocks = block_table.shape[1]
        
        # Calculate maximum end blocks
        max_end_blocks = (num_total_blocks - valid_length).max().item()
        output_len = k + max_end_blocks
        
        # Call the triton kernel to do actual copy
        grid = (batch_size, output_len)
        _copy_blocks_to_sparse_table[grid](
            top_k_block_idx,
            block_table,
            self.sparse_block_table,
            valid_length,
            num_total_blocks,
            batch_size,
            k,
            output_len,
            max_blocks,
        )

    def _overriden_kv_insert(self, *args, **kwargs):
        # Always use the original kv insert function.
        return self.original_kv_insert_func(*args, **kwargs)
    
    def _overriden_attention(self, *args, **kwargs):
        # Runtime profile variables.
        # min_max_begin = torch.cuda.Event(enable_timing=True)
        # min_max_end = torch.cuda.Event(enable_timing=True)
        # attn_begin = torch.cuda.Event(enable_timing=True)
        # attn_end = torch.cuda.Event(enable_timing=True)
        # copy_begin = torch.cuda.Event(enable_timing=True)
        # copy_end = torch.cuda.Event(enable_timing=True)
        all_total_begin = torch.cuda.Event(enable_timing=True)
        all_total_end = torch.cuda.Event(enable_timing=True)
        all_total_begin.record()
        if not self.in_draft:
            max_seqlen_q = kwargs['max_seqlen_q']
            self._enable_quest = \
                max_seqlen_q <= self.num_speculative_tokens + 1
            return self.original_attention_func(*args, **kwargs)
        
        assert self._enable_quest

        layer_index, should_override, _ = self._get_layer()

        # If this layer is sliding window attention, use original attention.
        if not should_override:
            return self.original_attention_func(*args, **kwargs)

        if layer_index == self.min_effective_layer_index:
            self.block_table = kwargs["block_table"]
            self.seq_lens = kwargs["seqused_k"]
            self.batch_size = self.seq_lens.shape[0]
            self.max_seqlen_k = kwargs["max_seqlen_k"]
            self.num_total_blocks = (self.seq_lens + self.block_size - 1) // self.block_size
            self.is_in_order = torch.all(self.quest.regular[:self.batch_size] == self.index_to_metadata[:self.batch_size]).item()
            # If it is the first draft step, we need to update quest minmax and valid length.
            if (self.current_draft_step == 1):
                self.num_valid_blocks = self.seq_lens // self.block_size
                if self.is_new_req.any():
                    new_request_ids = self.index_to_metadata[:self.batch_size][self.is_new_req[:self.batch_size]]
                    self.quest.prev_valid_lengths[new_request_ids] = 0
                    self.quest.min_max_keys[:,:,new_request_ids] = 0
                    old_request_ids = self.index_to_metadata[:self.batch_size][~self.is_new_req[:self.batch_size]]
                    self.quest.prev_valid_lengths[old_request_ids] = self.quest.valid_lengths[old_request_ids]
                    self.is_new_req.fill_(False)
                else:
                    self.quest.prev_valid_lengths[:] = self.quest.valid_lengths[:]
                self.quest.valid_lengths[self.index_to_metadata[:self.batch_size]] = self.num_valid_blocks

        q = kwargs['q']
        self.quest.compute_min_max_store(index_to_metadata = self.index_to_metadata,
                                         batch_size = self.batch_size,
                                         block_table=self.block_table,
                                         layer_idx=layer_index,
                                         keys=kwargs['k'],
                                         block_size=self.block_size)

        self.quest.compute_block_attn_weight(index_to_metadata=self.index_to_metadata[:self.batch_size],
                                             batch_size = self.batch_size,
                                             q=q,
                                             layer_index=layer_index)
        
        valid_lens = self.quest.valid_lengths[self.index_to_metadata[:self.batch_size]]

        # k_val should be min(self.k, min valid_length) to ensure all sequences have enough blocks
        min_valid_len = valid_lens.min().item()
        k_val = min(self.k, min_valid_len)
        
        _, top_k_block_idx = self.quest.estimation_attn_weights_batch.topk(k_val, dim=1, sorted=False)
        
        self.copy_to_buffer(
            top_k_block_idx=top_k_block_idx,
            block_table=self.block_table,
            valid_length=valid_lens,
            num_total_blocks=self.num_total_blocks,
            k=k_val
        )

        if layer_index == self.min_effective_layer_index:
            self.sparse_seq_lens[:self.batch_size] = k_val * self.block_size + \
            self.seq_lens - self.quest.valid_lengths[self.index_to_metadata[:self.batch_size]] * self.block_size
            self.sparse_max_seqlen_k = self.sparse_seq_lens.max().item()

        kwargs["block_table"] = self.sparse_block_table[:self.batch_size]
        kwargs["seqused_k"] = self.sparse_seq_lens[:self.batch_size]
        kwargs["max_seqlen_k"] = self.sparse_max_seqlen_k
        
        rst = self.original_attention_func(*args, **kwargs)

        kwargs["block_table"] = self.block_table
        kwargs["seqused_k"] = self.seq_lens
        kwargs["max_seqlen_k"] = self.max_seqlen_k
        all_total_end.record()

        return rst