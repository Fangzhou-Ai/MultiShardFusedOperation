#include <gtest/gtest.h>
#include <MSFO.cuh>

#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/distance.h>

TEST(MSFO_TEST, init)
{
    MSFO::cucoHashtable table;
    EXPECT_EQ(table.get_size(), 0);
}

TEST(MSFO_TEST, one_shard_one_table)
{
    MSFO::cucoHashtable table;
    size_t cnt = 1000000;
    thrust::device_vector<KeyT> keys(cnt);
    thrust::sequence(keys.begin(), keys.end());
    table.FusedBatchLookupOrCreate(thrust::raw_pointer_cast(keys.data()), &cnt);
    EXPECT_EQ(table.get_size(), cnt);
    // verify key
    thrust::host_vector<size_t> h_mem_idx(table.get_mem_idx());
    thrust::sort(h_mem_idx.begin(), h_mem_idx.end());
    for(auto i = 0; i < cnt; i++)
    {
        EXPECT_EQ(h_mem_idx[i], i);
    }
}

TEST(MSFO_TEST, two_shards_one_table_no_overlap)
{
    MSFO::cucoHashtable table;
    size_t cnt = 1000000;
    size_t cnt_array[2] = {cnt, cnt};
    thrust::device_vector<KeyT> keys(cnt * 2);
    // no overlapping
    thrust::sequence(keys.begin(), keys.end());
    table.FusedBatchLookupOrCreate(thrust::raw_pointer_cast(keys.data()), cnt_array, 0, 1, 2);
    EXPECT_EQ(table.get_size(), cnt * 2);
    // verify key
    thrust::host_vector<size_t> h_mem_idx(table.get_mem_idx());
    thrust::sort(h_mem_idx.begin(), h_mem_idx.end());
    for(auto i = 0; i < cnt * 2; i++)
    {
        EXPECT_EQ(h_mem_idx[i], i);
    }
}

TEST(MSFO_TEST, two_shards_one_table_partial_overlap)
{
    MSFO::cucoHashtable table;
    size_t cnt = 1000000;
    size_t cnt_array[2] = {cnt, cnt};
    thrust::device_vector<KeyT> keys(cnt * 2);
    // partial overlapping
    thrust::sequence(keys.begin(), keys.begin() + cnt);
    thrust::sequence(keys.begin() + cnt, keys.end(), cnt / 2);
    table.FusedBatchLookupOrCreate(thrust::raw_pointer_cast(keys.data()), cnt_array, 0, 1, 2);
    EXPECT_EQ(table.get_size(), cnt * 3 / 2);
    // verify key
    thrust::host_vector<size_t> h_mem_idx(table.get_mem_idx());
    thrust::sort(h_mem_idx.begin(), h_mem_idx.end());
    auto new_end = thrust::unique(h_mem_idx.begin(), h_mem_idx.end());
    EXPECT_EQ(thrust::distance(h_mem_idx.begin(), new_end), cnt * 3 / 2);
    for(auto i = 0; i < cnt * 3 / 2; i++)
    {
        EXPECT_EQ(h_mem_idx[i], i);
    }
}

TEST(MSFO_TEST, two_shards_one_table_full_overlap)
{
    MSFO::cucoHashtable table;
    size_t cnt = 1000000;
    size_t cnt_array[2] = {cnt, cnt};
    thrust::device_vector<KeyT> keys(cnt * 2);
    // Full overlapping
    thrust::sequence(keys.begin(), keys.begin() + cnt);
    thrust::sequence(keys.begin() + cnt, keys.end());
    table.FusedBatchLookupOrCreate(thrust::raw_pointer_cast(keys.data()), cnt_array, 0, 1, 2);
    EXPECT_EQ(table.get_size(), cnt);
    // verify key
    thrust::host_vector<size_t> h_mem_idx(table.get_mem_idx());
    thrust::sort(h_mem_idx.begin(), h_mem_idx.end());
    auto new_end = thrust::unique(h_mem_idx.begin(), h_mem_idx.end());
    EXPECT_EQ(thrust::distance(h_mem_idx.begin(), new_end), cnt);
    for(auto i = 0; i < cnt; i++)
    {
        EXPECT_EQ(h_mem_idx[i], i);
    }
}


TEST(MSFO_TEST, two_shards_two_tables_partial_overlap)
{
    MSFO::cucoHashtable table1;
    MSFO::cucoHashtable table2;
    size_t cnt = 1000000;
    size_t cnt_array[4] = {cnt, cnt, cnt, cnt};
    thrust::device_vector<KeyT> keys(cnt * 4);
    // partial overlapping
    thrust::sequence(keys.begin(), keys.begin() + cnt);
    thrust::sequence(keys.begin() + 2 * cnt, keys.begin() + 3 * cnt, cnt / 2);
    thrust::sequence(keys.begin() + cnt, keys.begin() + 2 * cnt);
    thrust::sequence(keys.begin() + 3 * cnt, keys.end(), cnt / 2);
    table1.FusedBatchLookupOrCreate(thrust::raw_pointer_cast(keys.data()), cnt_array, 0, 2, 2);
    table2.FusedBatchLookupOrCreate(thrust::raw_pointer_cast(keys.data()), cnt_array, 1, 2, 2);
    EXPECT_EQ(table1.get_size(), cnt * 3 / 2);
    EXPECT_EQ(table2.get_size(), cnt * 3 / 2);
    // verify key
    // table 1
    thrust::host_vector<size_t> h_mem_idx1(table1.get_mem_idx());
    thrust::sort(h_mem_idx1.begin(), h_mem_idx1.end());
    auto new_end1 = thrust::unique(h_mem_idx1.begin(), h_mem_idx1.end());
    EXPECT_EQ(thrust::distance(h_mem_idx1.begin(), new_end1), cnt * 3 / 2);
    for(auto i = 0; i < cnt * 3 / 2; i++)
    {
        EXPECT_EQ(h_mem_idx1[i], i);
    }
    // table 2
    thrust::host_vector<size_t> h_mem_idx2(table2.get_mem_idx());
    thrust::sort(h_mem_idx2.begin(), h_mem_idx2.end());
    auto new_end2 = thrust::unique(h_mem_idx2.begin(), h_mem_idx2.end());
    EXPECT_EQ(thrust::distance(h_mem_idx2.begin(), new_end2), cnt * 3 / 2);
    for(auto i = 0; i < cnt * 3 / 2; i++)
    {
        EXPECT_EQ(h_mem_idx2[i], i);
    }
}

