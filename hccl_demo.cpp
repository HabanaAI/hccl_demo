/******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 ******************************************************************************/

// C++ Standard Libraries
#include <iostream>
#include <exception>
#include <vector>
#include <cstdlib>  // for getenv, mkstemp
#include <chrono>   // for Bandwidth calculation
#include <iomanip>  // for setprecision
#include <unistd.h>
#include <functional>
#include <algorithm>
#include <sstream>
#include <numeric>
#include <fstream>

// HCCL :: Habana Collective Communications Library
#include <hccl.h>

// Synapse :: Habana Synapse training API
#include <synapse_api.h>

#if AFFINITY_ENABLED
#include "affinity.h"
#endif

#define DATA_ELEMENTS_MAX 13
#define DEFAULT_TEST_SIZE 33554432
#define DEFAULT_TEST_LOOP 10
#define DEFAULT_BOX_SIZE  8
#define NUMBER_OF_WARMUPS 100

using namespace std;
using Clock = chrono::high_resolution_clock;

#define CHECK_HCCL_STATUS(x)                                                                                           \
    {                                                                                                                  \
        const auto _res = (x);                                                                                         \
        if (_res != hcclSuccess)                                                                                       \
            throw runtime_error {"In function " + string {__FUNCTION__} +                                              \
                                 "(): " #x " failed: " + hcclGetErrorString(_res)};                                    \
    }

#define CHECK_SYNAPSE_STATUS(x)                                                                                        \
    {                                                                                                                  \
        const auto _res = (x);                                                                                         \
        if (_res != synSuccess)                                                                                        \
            throw runtime_error {"In function " + string {__FUNCTION__} +                                              \
                                 "(): " #x " failed with synapse error: " + to_string((_res))};                        \
    }

struct hccl_demo_data
{
    synDeviceId     device_handle;
    synStreamHandle device_to_host_stream;
    synStreamHandle host_to_device_stream;
    synStreamHandle collective_stream;
    size_t          nranks;
    hcclComm_t      hccl_comm;
    size_t          num_iters;
};

struct hccl_demo_stats
{
    float  avg_duration_in_sec;
    float  rank_duration_in_sec;
    size_t num_iters;
};

ostream& log()
{
    return cout;
}

hcclResult_t get_avg_duration(hccl_demo_data& demo_data, hccl_demo_stats& stat)
{
    float&      rank_duration       = stat.rank_duration_in_sec;
    float&      avg_duration        = stat.avg_duration_in_sec;
    uint64_t    data_size           = sizeof(stat.rank_duration_in_sec);
    uint64_t    count               = data_size / sizeof(float);
    const void* input_host_data_ptr = reinterpret_cast<void*>(&rank_duration);

    uint64_t input_dev_ptr {};
    uint64_t output_dev_ptr {};

    CHECK_SYNAPSE_STATUS(synDeviceMalloc(demo_data.device_handle, data_size, 0, 0, &input_dev_ptr));
    CHECK_SYNAPSE_STATUS(synDeviceMalloc(demo_data.device_handle, data_size, 0, 0, &output_dev_ptr));
    CHECK_SYNAPSE_STATUS(synHostMap(demo_data.device_handle, data_size, input_host_data_ptr));
    CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.host_to_device_stream,
                                         (uint64_t) input_host_data_ptr,
                                         data_size,
                                         input_dev_ptr,
                                         HOST_TO_DRAM));
    CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.host_to_device_stream));
    CHECK_HCCL_STATUS(hcclAllReduce((const void*) input_dev_ptr,
                                    (void*) output_dev_ptr,
                                    count,
                                    hcclFloat32,
                                    hcclSum,
                                    demo_data.hccl_comm,
                                    demo_data.collective_stream));
    CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.collective_stream));

    const void* output_host_data_ptr = reinterpret_cast<void*>(&avg_duration);

    CHECK_SYNAPSE_STATUS(synHostMap(demo_data.device_handle, data_size, output_host_data_ptr));
    CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.device_to_host_stream,
                                         output_dev_ptr,
                                         data_size,
                                         (uint64_t) output_host_data_ptr,
                                         DRAM_TO_HOST));
    CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.device_to_host_stream));

    avg_duration = avg_duration / demo_data.nranks;

    return hcclSuccess;
}

hccl_demo_stats benchmark(hccl_demo_data& demo_data, const function<void()>& fn)
{
    // Warmup run
    hccl_demo_stats stat;
    auto            num_warmup_iters = size_t {NUMBER_OF_WARMUPS};

    for (size_t iter = 0; iter < num_warmup_iters; ++iter)
    {
        fn();
    }

    CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.collective_stream));

    // Actual iterations
    auto start_time = Clock::now();

    for (size_t iter = 0; iter < demo_data.num_iters; ++iter)
    {
        fn();
    }

    CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.collective_stream));

    auto duration             = Clock::now() - start_time;
    stat.rank_duration_in_sec = chrono::duration_cast<chrono::duration<double>>(duration).count();
    stat.rank_duration_in_sec = stat.rank_duration_in_sec / demo_data.num_iters;

    CHECK_HCCL_STATUS(get_avg_duration(demo_data, stat));

    return stat;
}

bool should_report_stat(int rank)
{
    return rank == 0;
}

inline string format_bw(const double bytes_per_sec)
{
    stringstream ss;
    ss << fixed << setprecision(3) << bytes_per_sec / 1e6 << " MB/s";
    return ss.str();
}

string get_print_delimiter(size_t length, char delimiter)
{
    stringstream ss;

    for (size_t i = 0; i < length; i++)
    {
        ss << delimiter;
    }
    return ss.str();
}

string get_demo_test_type()
{
    static bool is_cached = false;
    static auto test_type = string {"broadcast"};
    if (!is_cached)
    {
        char* env_value = getenv("HCCL_DEMO_TEST");
        test_type       = (env_value != nullptr) ? string(env_value) : test_type;
        is_cached       = true;
    }
    return test_type;
}

int get_demo_box_size()
{
    static bool is_cached = false;
    static auto box_size  = DEFAULT_BOX_SIZE;
    if (!is_cached)
    {
        char* env_value = getenv("HCCL_BOX_SIZE");
        box_size        = (env_value != nullptr) ? atoi(env_value) : box_size;
        is_cached       = true;
    }
    return box_size;
}

int get_demo_test_root()
{
    static bool is_cached = false;
    static auto test_root = 0;
    if (!is_cached)
    {
        char* env_value = getenv("HCCL_DEMO_TEST_ROOT");
        test_root       = (env_value != nullptr) ? atoi(env_value) : test_root;
        is_cached       = true;
    }
    return test_root;
}

int get_demo_test_size()
{
    static bool is_cached = false;
    static auto test_size = DEFAULT_TEST_SIZE;
    if (!is_cached)
    {
        char* env_value = getenv("HCCL_DEMO_TEST_SIZE");
        test_size       = (env_value != nullptr) ? atoi(env_value) : test_size;
        is_cached       = true;
    }
    return test_size;
}

int get_demo_test_loop()
{
    static bool is_cached = false;
    static auto test_loop = DEFAULT_TEST_LOOP;
    if (!is_cached)
    {
        char* env_value = getenv("HCCL_DEMO_TEST_LOOP");
        test_loop       = (env_value != nullptr) ? atoi(env_value) : test_loop;
        is_cached       = true;
    }
    return test_loop;
}

string get_demo_csv_path()
{
    static bool is_cached = false;
    static auto csv_path  = string {""};
    if (!is_cached)
    {
        char* env_value = getenv("HCCL_DEMO_CSV_PATH");
        csv_path        = (env_value != nullptr) ? string(env_value) : csv_path;
        is_cached       = true;
    }
    return csv_path;
}

int get_nranks()
{
    static bool is_cached   = false;
    static auto test_nranks = 0;
    if (!is_cached)
    {
        char* env_value = getenv("HCCL_NRANKS");
        test_nranks     = (env_value != nullptr) ? atoi(env_value) : test_nranks;
        is_cached       = true;
    }
    return test_nranks;
}

int get_hccl_rank()
{
    static bool is_cached = false;
    static auto test_rank = -1;
    if (!is_cached)
    {
        char* env_value = getenv("HCCL_RANK");
        test_rank       = (env_value != nullptr) ? atoi(env_value) : test_rank;
        is_cached       = true;
    }
    return test_rank;
}

void describe_stat(const string&          stat_name,
                   const hccl_demo_stats& stats,
                   size_t                 data_size,
                   double                 factor,
                   int                    hccl_rank,
                   int                    loop,
                   const string&          test_type,
                   const string&          dtype)
{
    auto avg_bandwidth = (double) data_size / stats.avg_duration_in_sec;
    avg_bandwidth      = avg_bandwidth * factor;
    auto rank_bandwith = (double) data_size / stats.rank_duration_in_sec;
    rank_bandwith      = rank_bandwith * factor;

    if (should_report_stat(hccl_rank))
    {
        stringstream ss;
        sleep(1);
        size_t delimiter_size = stat_name.length() + string {"[BENCHMARK]"}.length() + 1;
        ss << get_print_delimiter(delimiter_size, '#') << '\n';
        ss << "[BENCHMARK] " << stat_name << '\n';
        ss << "[BENCHMARK]     Bandwidth     : " << format_bw(avg_bandwidth);
        ss << '\n' << get_print_delimiter(delimiter_size, '#') << '\n';
        log() << ss.str();
    }

    // Write results to csv file
    auto csv_path = get_demo_csv_path();
    if (!csv_path.empty())
    {
        ofstream output;
        output.open(csv_path, ofstream::out | ofstream::app);
        output << test_type << "," << hccl_rank << "," << dtype << "," << data_size << "," << loop << ","
               << format_bw(rank_bandwith) << endl;
        output.close();
    }
}

hcclResult_t send_recv_test(
    void* out_dev_ptr, const void* input_dev_ptr, size_t count, hcclComm_t comm, hcclStream_t stream, int peerRank)
{
    hcclGroupStart();

    CHECK_HCCL_STATUS(hcclSend((const void*) input_dev_ptr, count, hcclFloat32, peerRank, comm, stream));
    CHECK_HCCL_STATUS(hcclRecv((void*) out_dev_ptr, count, hcclFloat32, peerRank, comm, stream));

    hcclGroupEnd();

    return hcclSuccess;
}

int main()
{
    try
    {
        log() << "Running HCCL Demo :: A simple program demonstrating HCCL usage from C++" << endl;
        hccl_demo_data demo_data;
        demo_data.nranks    = get_nranks();
        demo_data.num_iters = get_demo_test_loop();
        int hccl_rank       = get_hccl_rank();

        // Initialize Synapse API context
        CHECK_SYNAPSE_STATUS(synInitialize());

        // Acquire device
        const synModuleId device_module_id = hccl_rank % get_demo_box_size();
        CHECK_SYNAPSE_STATUS(synDeviceAcquireByModuleId(&demo_data.device_handle, device_module_id));

#if AFFINITY_ENABLED
        setupAffinity(device_module_id);
#endif

        // Create Streams
        CHECK_SYNAPSE_STATUS(
            synStreamCreate(&demo_data.collective_stream, demo_data.device_handle, STREAM_TYPE_NETWORK_COLLECTIVE, 0));
        CHECK_SYNAPSE_STATUS(synStreamCreate(&demo_data.device_to_host_stream,
                                             demo_data.device_handle,
                                             STREAM_TYPE_COPY_DEVICE_TO_HOST,
                                             0));
        CHECK_SYNAPSE_STATUS(synStreamCreate(&demo_data.host_to_device_stream,
                                             demo_data.device_handle,
                                             STREAM_TYPE_COPY_HOST_TO_DEVICE,
                                             0));

        // Generate unique id
        hcclUniqueId  unique_id {};
        constexpr int master_mpi_rank = 0;

        if (hccl_rank == master_mpi_rank)
        {
            CHECK_HCCL_STATUS(hcclGetUniqueId(&unique_id));
        }

        // Create new HCCL communicator
        CHECK_HCCL_STATUS(hcclCommInitRank(&demo_data.hccl_comm, demo_data.nranks, unique_id, hccl_rank));

        uint64_t input_dev_ptr {};
        uint64_t output_dev_ptr {};

        // Allocate buffers on the HPU device
        uint64_t    data_size           = get_demo_test_size();
        uint64_t    count               = data_size / sizeof(float);
        auto        input_host_data     = vector<float>(count, hccl_rank + 1);
        const void* input_host_data_ptr = reinterpret_cast<void*>(input_host_data.data());

        CHECK_SYNAPSE_STATUS(synDeviceMalloc(demo_data.device_handle, data_size, 0, 0, &input_dev_ptr));
        CHECK_SYNAPSE_STATUS(synDeviceMalloc(demo_data.device_handle, data_size, 0, 0, &output_dev_ptr));
        CHECK_SYNAPSE_STATUS(synHostMap(demo_data.device_handle, data_size, input_host_data_ptr));
        CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.host_to_device_stream,
                                             (uint64_t) input_host_data_ptr,
                                             data_size,
                                             input_dev_ptr,
                                             HOST_TO_DRAM));
        CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.host_to_device_stream));

        string test_type = get_demo_test_type();

        if (test_type == "broadcast")
        {
            double broadcast_factor = 1;
            int    root             = get_demo_test_root();

            for (uint64_t i = 0; i < count; ++i)
            {
                input_host_data[i] = i + hccl_rank;
            }

            // Copy from input_host_data_ptr to input_dev_ptr (to be used in benchmark)
            CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.host_to_device_stream,
                                                 (uint64_t) input_host_data_ptr,
                                                 data_size,
                                                 input_dev_ptr,
                                                 HOST_TO_DRAM));

            CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.host_to_device_stream));

            // Run HCCL Broadcast collective
            auto stat = benchmark(demo_data, [&]() {
                CHECK_HCCL_STATUS(hcclBroadcast((const void*) input_dev_ptr,
                                                (void*) output_dev_ptr,
                                                input_host_data.size(),
                                                hcclFloat32,
                                                root,
                                                demo_data.hccl_comm,
                                                demo_data.collective_stream));
            });

            // Correctness check

            bool is_ok = true;

            auto        output_host_data     = vector<float>(input_host_data.size());
            const void* output_host_data_ptr = reinterpret_cast<void*>(output_host_data.data());

            CHECK_SYNAPSE_STATUS(synHostMap(demo_data.device_handle, data_size, output_host_data_ptr));
            CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.device_to_host_stream,
                                                 output_dev_ptr,
                                                 data_size,
                                                 (uint64_t) output_host_data_ptr,
                                                 DRAM_TO_HOST));
            CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.device_to_host_stream));

            for (size_t i = 0; i < input_host_data.size(); ++i)
            {
                if (abs(output_host_data[i] - (float) (i + root)) != 0)
                {
                    is_ok = false;
                }
            }

            log() << "Broadcast hccl_rank=" << hccl_rank << " root=" << root << " size=" << data_size << " <float>"
                  << " Input Buffer [" << input_host_data[0] << " " << input_host_data[1] << " " << input_host_data[2]
                  << " " << input_host_data[3] << " ...]"
                  << " Output Buffer [" << output_host_data[0] << " " << output_host_data[1] << " "
                  << output_host_data[2] << " " << output_host_data[3] << " ...]"
                  << " which is " << (is_ok ? "fine." : "bad.") << endl;

            CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, input_host_data_ptr));
            CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, output_host_data_ptr));

            // End of correctness check

            describe_stat("Broadcast(count=" + to_string(input_host_data.size()) +
                              ", dtype=fp32, iterations=" + to_string(demo_data.num_iters) + ")",
                          stat,
                          data_size,
                          broadcast_factor,
                          hccl_rank,
                          demo_data.num_iters,
                          test_type,
                          "float");
        }
        else if (test_type == "all_reduce")
        {
            double allreduce_factor = ((double) (2 * (demo_data.nranks - 1))) / ((double) demo_data.nranks);

            // Fill input data, example:
            // Input        |   Output
            // G0 G1 G2 G3      G0 G1 G2 G3
            // 0  1  2  3   =>  6  6  6  6
            // 4  5  6  7       22 22 22 22
            // 8  9  10 11      38 38 38 38
            // 12 13 14 15      54 54 54 54

            for (uint64_t i = 0; i < count; ++i)
            {
                // We want to make sure we use different values on each cell and between ranks,
                // but we don't want the summation to get too big, that is why we modulo by DATA_ELEMENTS_MAX.
                input_host_data[i] = hccl_rank + (demo_data.nranks * (i % DATA_ELEMENTS_MAX));
            }

            //Copy from input_host_data_ptr to input_dev_ptr (to be used in benchmark)
            CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.host_to_device_stream,
                                                 (uint64_t) input_host_data_ptr,
                                                 data_size,
                                                 input_dev_ptr,
                                                 HOST_TO_DRAM));
            CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.host_to_device_stream));

            // Run HCCL AllReduce collective
            auto stat = benchmark(demo_data, [&]() {
                CHECK_HCCL_STATUS(hcclAllReduce((const void*) input_dev_ptr,
                                                (void*) output_dev_ptr,
                                                input_host_data.size(),
                                                hcclFloat32,
                                                hcclSum,
                                                demo_data.hccl_comm,
                                                demo_data.collective_stream));
            });

            // Correctness check

            bool is_ok = true;

            auto        output_host_data     = vector<float>(input_host_data.size());
            const void* output_host_data_ptr = reinterpret_cast<void*>(output_host_data.data());

            CHECK_SYNAPSE_STATUS(synHostMap(demo_data.device_handle, data_size, output_host_data_ptr));
            CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.device_to_host_stream,
                                                 output_dev_ptr,
                                                 data_size,
                                                 (uint64_t) output_host_data_ptr,
                                                 DRAM_TO_HOST));
            CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.device_to_host_stream));

            int start = 0;
            int end   = demo_data.nranks - 1;
            int expected;
            int addCommSize;

            for (size_t i = 0; i < input_host_data.size(); ++i)
            {
                addCommSize = demo_data.nranks * (i % DATA_ELEMENTS_MAX);

                // Arithmetic progression
                expected = ((start + addCommSize) + (end + addCommSize)) * demo_data.nranks / 2;
                if (abs(output_host_data[i] - expected) != 0)
                {
                    is_ok = false;
                }
            }
            log() << "Allreduce hccl_rank=" << hccl_rank << " size=" << data_size << " <float>"
                  << " Input Buffer [" << input_host_data[0] << " " << input_host_data[1] << " " << input_host_data[2]
                  << " " << input_host_data[3] << " ...]"
                  << " reduced to Output Buffer [" << output_host_data[0] << " " << output_host_data[1] << " "
                  << output_host_data[2] << " " << output_host_data[3] << " ...]"
                  << " which is " << (is_ok ? "fine." : "bad.") << endl;

            CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, input_host_data_ptr));
            CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, output_host_data_ptr));

            // End of correctness check

            describe_stat("hcclAllReduce(src!=dst, count=" + to_string(input_host_data.size()) +
                              ", dtype=fp32, iterations=" + to_string(demo_data.num_iters) + ")",
                          stat,
                          data_size,
                          allreduce_factor,
                          hccl_rank,
                          demo_data.num_iters,
                          test_type,
                          "float");
        }
        else if (test_type == "reduce_scatter")
        {
            double reduce_scatter_factor = ((double) (demo_data.nranks - 1)) / ((double) demo_data.nranks);

            // Fill input data, example:
            // Input        |   Output
            // G0 G1 G2 G3      G0 G1 G2 G3
            // 0  1  2  3   =>  6  22 38 54
            // 4  5  6  7
            // 8  9  10 11
            // 12 13 14 15

            for (uint64_t i = 0; i < count; ++i)
            {
                // We want to make sure we use different values on each cell and between ranks,
                // but we don't want the summation to get too big, that is why we modulo by DATA_ELEMENTS_MAX.
                input_host_data[i] = hccl_rank + (demo_data.nranks * (i % DATA_ELEMENTS_MAX));
            }

            //Copy from input_host_data_ptr to input_dev_ptr (to be used in benchmark)
            CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.host_to_device_stream,
                                                 (uint64_t) input_host_data_ptr,
                                                 data_size,
                                                 input_dev_ptr,
                                                 HOST_TO_DRAM));
            CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.host_to_device_stream));

            // Run HCCL ReduceScatter collective
            auto stat = benchmark(demo_data, [&]() {
                CHECK_HCCL_STATUS(hcclReduceScatter((const void*) input_dev_ptr,
                                                    (void*) output_dev_ptr,
                                                    input_host_data.size() / demo_data.nranks,
                                                    hcclFloat32,
                                                    hcclSum,
                                                    demo_data.hccl_comm,
                                                    demo_data.collective_stream));
            });

            // Correctness check

            bool        is_ok                = true;
            auto        output_host_data     = vector<float>(input_host_data.size() / demo_data.nranks);
            const void* output_host_data_ptr = reinterpret_cast<void*>(output_host_data.data());

            CHECK_SYNAPSE_STATUS(
                synHostMap(demo_data.device_handle, data_size / demo_data.nranks, output_host_data_ptr));
            CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.device_to_host_stream,
                                                 output_dev_ptr,
                                                 data_size / demo_data.nranks,
                                                 (uint64_t) output_host_data_ptr,
                                                 DRAM_TO_HOST));
            CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.device_to_host_stream));

            int start;
            int end;
            int expected = 0;

            for (size_t i = 0; i < output_host_data.size(); ++i)
            {
                start = (hccl_rank * output_host_data.size()) % DATA_ELEMENTS_MAX * demo_data.nranks;
                end   = start + (demo_data.nranks - 1);
                // Arithmetic progression
                expected = (((start + demo_data.nranks * i) % (demo_data.nranks * DATA_ELEMENTS_MAX)) +
                            ((end + demo_data.nranks * i) % (demo_data.nranks * DATA_ELEMENTS_MAX))) *
                           demo_data.nranks / 2;
                if (abs(output_host_data[i] - expected) != 0)
                {
                    is_ok = false;
                }
            }

            log() << "ReduceScatter hccl_rank=" << hccl_rank << " size=" << data_size << " <float>"
                  << " Input Buffer [" << input_host_data[0] << " " << input_host_data[1] << " " << input_host_data[2]
                  << " " << input_host_data[3] << " ...]"
                  << " reduced to Output Buffer [" << output_host_data[0] << " " << output_host_data[1] << " "
                  << output_host_data[2] << " " << output_host_data[3] << " ...]"
                  << " which is " << (is_ok ? "fine." : "bad.") << endl;

            CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, input_host_data_ptr));
            CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, output_host_data_ptr));

            // End of correctness

            describe_stat("hcclReduceScatter(src!=dst, count=" + to_string(input_host_data.size()) +
                              ", dtype=fp32, iterations=" + to_string(demo_data.num_iters) + ")",
                          stat,
                          data_size,
                          reduce_scatter_factor,
                          hccl_rank,
                          demo_data.num_iters,
                          test_type,
                          "float");
        }
        else if (test_type == "all_gather")
        {
            double all_gather_factor = ((double) (demo_data.nranks - 1));
            CHECK_SYNAPSE_STATUS(
                synDeviceMalloc(demo_data.device_handle, data_size * demo_data.nranks, 0, 0, &output_dev_ptr));

            // Fill input data, example:
            // Input        |   Output
            // G0 G1 G2 G3      G0 G1 G2 G3
            // 0  2  4  6   =>  0  0  0  0
            // 1  3  5  7       1  1  1  1
            //                  2  2  2  2
            //                  3  3  3  3
            //                  4  4  4  4
            //                  5  5  5  5
            //                  6  6  6  6
            //                  7  7  7  7

            for (uint64_t i = 0; i < count; ++i)
            {
                input_host_data[i] = hccl_rank * count + i;
            }

            //Copy from input_host_data_ptr to input_dev_ptr (to be used in benchmark)
            CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.host_to_device_stream,
                                                 (uint64_t) input_host_data_ptr,
                                                 data_size,
                                                 input_dev_ptr,
                                                 HOST_TO_DRAM));

            CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.host_to_device_stream));

            // Run HCCL AllGather collective
            auto stat = benchmark(demo_data, [&]() {
                CHECK_HCCL_STATUS(hcclAllGather((const void*) input_dev_ptr,
                                                (void*) output_dev_ptr,
                                                input_host_data.size(),
                                                hcclFloat32,
                                                demo_data.hccl_comm,
                                                demo_data.collective_stream));
            });

            // Correctness check

            bool        is_ok                = true;
            auto        output_host_data     = vector<float>(input_host_data.size() * demo_data.nranks);
            const void* output_host_data_ptr = reinterpret_cast<void*>(output_host_data.data());

            CHECK_SYNAPSE_STATUS(
                synHostMap(demo_data.device_handle, data_size * demo_data.nranks, output_host_data_ptr));
            CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.device_to_host_stream,
                                                 output_dev_ptr,
                                                 data_size * demo_data.nranks,
                                                 (uint64_t) output_host_data_ptr,
                                                 DRAM_TO_HOST));
            CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.device_to_host_stream));

            for (size_t i = 0; i < output_host_data.size(); ++i)
            {
                if (output_host_data[i] != i)
                {
                    is_ok = false;
                }
            }

            log() << "AllGather hccl_rank=" << hccl_rank << " size=" << data_size << " <float>"
                  << " Input Buffer [" << input_host_data[0] << " " << input_host_data[1] << " " << input_host_data[2]
                  << " " << input_host_data[3] << " ...]"
                  << " gathered to Output Buffer [" << output_host_data[0] << " " << output_host_data[1] << " "
                  << output_host_data[2] << " " << output_host_data[3] << " ...]"
                  << " which is " << (is_ok ? "fine." : "bad.") << endl;

            CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, input_host_data_ptr));
            CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, output_host_data_ptr));

            // End of correctness check

            describe_stat("hcclAllGather(src!=dst, count=" + to_string(input_host_data.size()) +
                              ", dtype=fp32, iterations=" + to_string(demo_data.num_iters) + ")",
                          stat,
                          data_size,
                          all_gather_factor,
                          hccl_rank,
                          demo_data.num_iters,
                          test_type,
                          "float");
        }
        else if (test_type == "send_recv")
        {
            double send_recv_factor = 1;
            int    peerRank         = get_hccl_rank() % 2 ? get_hccl_rank() - 1 : get_hccl_rank() + 1;

            auto stat = benchmark(demo_data, [&]() {
                CHECK_HCCL_STATUS(send_recv_test((void*) output_dev_ptr,
                                                 (const void*) input_dev_ptr,
                                                 (uint64_t) input_host_data.size(),
                                                 demo_data.hccl_comm,
                                                 demo_data.collective_stream,
                                                 peerRank));
            });

            // Correctness check
            bool is_ok = true;

            auto        output_host_data     = vector<float>(input_host_data.size());
            const void* output_host_data_ptr = reinterpret_cast<void*>(output_host_data.data());

            CHECK_SYNAPSE_STATUS(synHostMap(demo_data.device_handle, data_size, output_host_data_ptr));

            CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.device_to_host_stream,
                                                 output_dev_ptr,
                                                 data_size,
                                                 (uint64_t) output_host_data_ptr,
                                                 DRAM_TO_HOST));
            CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.device_to_host_stream));

            for (size_t i = 0; i < input_host_data.size(); ++i)
            {
                if (abs(output_host_data[i] - (float) (peerRank + 1)) != 0)
                {
                    is_ok = false;
                }
            }

            log() << "SendRecv hccl_rank=" << hccl_rank << " peerRank=" << peerRank << " size=" << data_size
                  << " <float>"
                  << " Input Buffer [" << input_host_data[0] << " " << input_host_data[1] << " " << input_host_data[2]
                  << " " << input_host_data[3] << " ...]"
                  << " Output Buffer [" << output_host_data[0] << " " << output_host_data[1] << " "
                  << output_host_data[2] << " " << output_host_data[3] << " ...]"
                  << " which is " << (is_ok ? "fine." : "bad.") << endl;

            CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, input_host_data_ptr));
            CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, output_host_data_ptr));

            // End of correctness check
            describe_stat("hcclSendRecv(src!=dst, count=" + to_string(input_host_data.size()) +
                              ", dtype=fp32, iterations=" + to_string(demo_data.num_iters) + ")",
                          stat,
                          data_size,
                          send_recv_factor,
                          hccl_rank,
                          demo_data.num_iters,
                          test_type,
                          "float");
        }
        else
        {
            log() << "Unknown test type (" << test_type << ")" << endl;
            return -1;
        }

        CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.collective_stream));

        // Destroy HCCL communicator
        CHECK_HCCL_STATUS(hcclCommDestroy(demo_data.hccl_comm));

        CHECK_SYNAPSE_STATUS(synDeviceFree(demo_data.device_handle, input_dev_ptr, 0));
        CHECK_SYNAPSE_STATUS(synDeviceFree(demo_data.device_handle, output_dev_ptr, 0));

        // Clean up HCCL
        CHECK_SYNAPSE_STATUS(synDeviceRelease(demo_data.device_handle));

        // Destroy synapse api context
        CHECK_SYNAPSE_STATUS(synDestroy());
    }
    catch (const exception& ex)
    {
        log() << "error: " << ex.what() << endl;
        return -1;
    }
    return 0;
}
