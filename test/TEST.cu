#include <gtest/gtest.h>
#include <MSFO.cuh>

#include <thrust/sequence.h>

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
}

