#include <cuda_runtime.h>
#include <cuco/dynamic_map.cuh>
#include <cuco/allocator.hpp>
#include <memory>
#include <limits>

#define KeyT int64_t
#define ValT size_t

static constexpr KeyT empty_key_sentinel = std::numeric_limits<KeyT>::max();
static constexpr ValT empty_val_sentinel = std::numeric_limits<ValT>::max();

namespace MSFO{


class cucoHashtable
{
    public:
        cucoHashtable()
        {
            CUCO_CUDA_TRY(cudaStreamCreate(&stream_));
            CUCO_CUDA_TRY(cudaGetDevice(&device_id_));
            map_ = std::make_unique<cuco::dynamic_map<KeyT, ValT>>(100000,
                                                                   cuco::sentinel::empty_key{empty_key_sentinel},
                                                                   cuco::sentinel::empty_value{empty_val_sentinel});
        }

        /* params
         * @ input
         * all_key : array contains all keys from all shards, acquired from tf input ids
         * slot_size_count : each slot size, acquired from tf input, 
         *                   length of this array should equal to
         *                   num_tables_ * num_shards_
         * table_id_ : current table's id range from [0, num_tables)
         * num_tables_ : total number of tables
         * num_shards_ : total number of shards
         */
        void FusedBatchLookupOrCreate(KeyT* all_key,
                                      size_t* slot_size_cnt,
                                      size_t table_id_,
                                      size_t num_tables_,
                                      size_t num_shards_);



        ~cucoHashtable()
        {
            map_.reset();
        }

    private:
        cudaStream_t stream_;
        int device_id_;
        std::unique_ptr<cuco::dynamic_map<KeyT, ValT>> map_;
};

}