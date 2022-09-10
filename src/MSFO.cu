#include <MSFO.cuh>



void MSFO::cucoHashtable::FusedBatchLookupOrCreate(KeyT* all_key, /* GPU */
                                                   size_t* slot_size_cnt, /* CPU */
                                                   size_t table_id_,
                                                   size_t num_tables_,
                                                   size_t num_shards_)
{
    // get number of all keys
    // O(num_tables_)
    size_t num_to_lookup = 0;
    for(int i = 0; i < num_tables_; i++)
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
    for(int i = 1; i < num_tables_ * num_tables_; i++)
    {
        prefix_sum[i] = prefix_sum[i - 1] + slot_size_cnt[i - 1];
    }
    // Move to GPU
    // O(num_tables_ * num_shards_)
    size_t* d_slot_size_cnt;
    size_t* d_prefix_sum;
    cudaMalloc((void**)&d_slot_size_cnt, sizeof(size_t) * num_shards_ * num_tables_);
    cudaMalloc((void**)&d_prefix_sum, sizeof(size_t) * num_shards_ * num_tables_);
    
    // atomic counter
    cuda::atomic<std::size_t, cuda::thread_scope_device>* start_idx;
    cudaMallocManaged((void**)&start_idx, sizeof(cuda::atomic<std::size_t, cuda::thread_scope_device>));
    start_idx->store(0);
    cudaMemPrefetchAsync(start_idx, sizeof(cuda::atomic<std::size_t, cuda::thread_scope_device>), device_id_, stream_);

    



    std::cout << "Number of new allocated keys are " << start_idx->load() << "." << std::endl;
    // Free memory
    delete[] prefix_sum;
    cudaFree(d_slot_size_cnt);
    cudaFree(d_prefix_sum);
    cudaFree(start_idx);
    return;
}