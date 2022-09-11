#include <MSFO.cuh>

struct params
{
    const KeyT* all_key_first;
    ValT* all_idx_first;
    size_t* d_slot_size_cnt;
    size_t* d_prefix_sum;
    size_t num_to_start_lookup;
    size_t num_to_end_lookup;
    size_t num_shards_;
    size_t num_tables_;
    size_t table_id_;
    mutableViewT* submap_mutable_views;
    ViewT* submap_views;
    uint32_t num_submaps;
    size_t submap_idx;
    atomicT* num_successes;
    atomicT* start_idx;
};

template <uint32_t block_size,
          uint32_t tile_size,
          typename mutableViewT,
          typename ViewT>
__global__ void ker_FusedBatchLookupOrCreate(params p)
{
    auto _grid = cooperative_groups::this_grid();
    auto _block = cooperative_groups::this_thread_block();
    auto tile = cooperative_groups::tiled_partition<tile_size>(_block);
    auto tid = _grid.thread_rank();
    auto key_idx = tid / tile_size;
    ValT tmp;

    //get params
    auto all_key_first = p.all_key_first;
    auto all_idx_first = p.all_idx_first;
    auto d_slot_size_cnt = p.d_slot_size_cnt;
    auto d_prefix_sum = p.d_prefix_sum;
    auto num_to_start_lookup = p.num_to_start_lookup;
    auto num_to_end_lookup = p.num_to_end_lookup;
    auto num_shards_ = p.num_shards_;
    auto num_tables_ = p.num_tables_;
    auto table_id_ = p.table_id_;
    auto submap_mutable_views = p.submap_mutable_views;
    auto submap_views = p.submap_views;
    auto num_submaps = p.num_submaps;
    auto submap_idx = p.submap_idx;
    auto num_successes = p.num_successes;
    auto start_idx = p.start_idx;

    // calculate which shard to start and the offset inside this shard
    size_t shard_start_idx;
    size_t shard_start_offset_idx;
    size_t shard_end_idx;
    size_t shard_end_offset_idx;
    // get shard start offset
    size_t num_of_keys = 0;
    for(auto i = 0; i < num_shards_; i++)
    {
        if(num_of_keys + d_slot_size_cnt[table_id_ + i * num_tables_] > num_to_start_lookup)
        {
            shard_start_idx = i;
            shard_start_offset_idx = num_to_start_lookup - num_of_keys; // included
            break;
        }
        else
        {
            num_of_keys += d_slot_size_cnt[table_id_ + i * num_tables_];
        }
    }

    // get shard end offset
    for(auto i = shard_start_idx; i < num_shards_; i++)
    {
        if(num_of_keys + d_slot_size_cnt[table_id_ + i * num_tables_] >= num_to_end_lookup)
        {
            shard_end_idx = i;
            shard_end_offset_idx = num_to_end_lookup - num_of_keys; // not included
            break;
        }
        else
        {
            num_of_keys += d_slot_size_cnt[table_id_ + i * num_tables_];
        }
    }

    
    for(int sid = shard_start_idx; sid <= shard_end_idx; sid++)
    {
        // get key offset and num of keys in this  shard need to lookup
        const KeyT* key_first;
        ValT* idx_first = all_idx_first + num_to_start_lookup;
        size_t num_to_lookup_this_shard;
        if(sid == shard_start_idx) // at starting shard, need to offset start key
        {
            key_first = all_key_first + d_prefix_sum[sid * num_tables_ + table_id_] + shard_start_offset_idx;
            if(sid == shard_end_idx) // also at starting shard, need chunked lookup size
            {
                num_to_lookup_this_shard = shard_end_offset_idx - shard_start_offset_idx;
            }
            else // all the way to the end of shard
            {
                num_to_lookup_this_shard = d_slot_size_cnt[sid * num_tables_ + table_id_] - shard_start_offset_idx;
            }
        }
        else
        {
            key_first = all_key_first + d_prefix_sum[sid * num_tables_ + table_id_];
            if(sid == shard_end_idx)
            {
                num_to_lookup_this_shard = shard_end_offset_idx;
            }
            else // all the way to the end of shard
            {
                num_to_lookup_this_shard = d_slot_size_cnt[sid * num_tables_ + table_id_];
            }
        }
        // Lookup keys
        for(size_t id = key_idx; id < num_to_lookup_this_shard; id += (gridDim.x * blockDim.x) / tile_size)
        {
            ValT found_value  = empty_val_sentinel;
            KeyT key =  key_first[id];
            for(auto i = 0; i < num_submaps; ++i)
            {
                auto submap_view = submap_views[i];
                auto found = submap_view.find(tile, key);
                if (found != submap_view.end())
                {
                    found_value = found->second;
                    break;
                }
            }

            if (found_value == empty_val_sentinel)
            {
                if (tile.thread_rank() == 0)
                {
                    tmp = start_idx->fetch_add(1); // Use your own way to distribute idx
                }
                found_value = tile.shfl(tmp, 0);
                auto insert_pair = cuco::pair_type<KeyT, ValT>{key, found_value};
                if (submap_mutable_views[submap_idx].insert(tile, insert_pair) &&
                tile.thread_rank() == 0)
                {
                    num_successes->fetch_add(1);
                }
            }

            if (tile.thread_rank() == 0)
            {
                idx_first[id] = found_value;
            }
            
        }

        _grid.sync(); // Sync before we goes to next shard
    }
}








void MSFO::cucoHashtable::FusedBatchLookupOrCreate(KeyT* all_key_first, /* GPU */
                                                   size_t* slot_size_cnt, /* CPU */
                                                   size_t table_id_,
                                                   size_t num_tables_,
                                                   size_t num_shards_)
{
    std::lock_guard<std::mutex> lock(mutex_);
    // get number of all keys
    // O(num_tables_)
    size_t num_to_lookup = 0;
    for(int i = 0; i < num_shards_; i++)
    {
        num_to_lookup += slot_size_cnt[i * num_tables_ + table_id_];
    }
    map_->reserve(map_->get_size() + num_to_lookup); // reserve space
    // No need to resize when size goes down, thrust will not free excessive memory anyway
    if(num_to_lookup > mem_index_.size()) 
        mem_index_.resize(num_to_lookup);
    // exclusive scan to get prefix sum as total offset
    // this step can be done in fused op
    // O(num_tables_ * num_shards_)
    size_t* prefix_sum = new size_t[num_tables_ * num_tables_]();
    for(int i = 1; i < num_tables_ * num_shards_; i++)
    {
        prefix_sum[i] = prefix_sum[i - 1] + slot_size_cnt[i - 1];
    }

    // Move to GPU
    // O(num_tables_ * num_shards_)
    size_t* d_slot_size_cnt;
    size_t* d_prefix_sum;
    CUCO_CUDA_TRY(cudaMalloc((void**)&d_slot_size_cnt, sizeof(size_t) * num_shards_ * num_tables_));
    CUCO_CUDA_TRY(cudaMalloc((void**)&d_prefix_sum, sizeof(size_t) * num_shards_ * num_tables_));
    CUCO_CUDA_TRY(cudaMemcpy(d_slot_size_cnt, slot_size_cnt, sizeof(size_t) * num_shards_ * num_tables_, cudaMemcpyHostToDevice));
    CUCO_CUDA_TRY(cudaMemcpy(d_prefix_sum, prefix_sum, sizeof(size_t) * num_shards_ * num_tables_, cudaMemcpyHostToDevice));

    // Lookup keys here
    size_t num_to_insert = num_to_lookup;
    size_t submap_idx = 0;
    size_t num_to_start_lookup = 0;
    size_t num_to_end_lookup;
    while (num_to_insert > 0)
    {
        std::size_t capacity_remaining = map_->get_max_load_factor() * 
                                        map_->get_submaps()[submap_idx]->get_capacity() - 
                                        map_->get_submaps()[submap_idx]->get_size();

        if (capacity_remaining >= map_->get_min_insert_size())
        {
            *(map_->get_num_successes()) = 0;
            CUCO_CUDA_TRY(cudaMemPrefetchAsync(map_->get_num_successes(), sizeof(atomicT), device_id_));
            auto n = std::min(capacity_remaining, num_to_insert);
            num_to_end_lookup = num_to_start_lookup + n;
            // Read this before change the size of  grid/block
            // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#grid-synchronization-cg
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, device_id_);
            auto const block_size = 128;
            auto const tile_size = 4;
            auto const grid_size = deviceProp.multiProcessorCount * 10;
            params p = {
                all_key_first,
                thrust::raw_pointer_cast(mem_index_.data()),
                d_slot_size_cnt,
                d_prefix_sum,
                num_to_start_lookup,
                num_to_end_lookup,
                num_shards_,
                num_tables_,
                table_id_,
                map_->get_submap_mutable_views().data().get(),
                map_->get_submap_views().data().get(),
                map_->get_submaps().size(),
                submap_idx,
                map_->get_num_successes(),
                start_idx
            };
            void* kernel_args = (void*)&p;
            CUCO_CUDA_TRY(
                cudaLaunchCooperativeKernel
                (
                    (void*)ker_FusedBatchLookupOrCreate<block_size, tile_size, mutableViewT, ViewT>,
                    grid_size, block_size,
                    &kernel_args,
                    0,
                    stream_
                )
            );

            CUCO_CUDA_TRY(cudaStreamSynchronize(stream_));
            size_t h_num_successes = map_->get_num_successes()->load();
            map_->update_submap_sizes(submap_idx, h_num_successes);
            num_to_start_lookup = num_to_end_lookup;
            num_to_insert -= n;
        }
        submap_idx++;
    }
    // Free memory
    delete[] prefix_sum;
    cudaFree(d_slot_size_cnt);
    cudaFree(d_prefix_sum);
    return;
}