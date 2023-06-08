/******************************************************************************
# Copyright (c) 2022 Habana Labs, Ltd.
# SPDX-License-Identifier: Apache-2.0
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

constexpr int INVALID_RANK = -1;

#if MPI_ENABLED
// Open MPI (v4.0.2)
#include <mpi.h>

#define CHECK_MPI_STATUS(x)                                                                                            \
    {                                                                                                                  \
        const auto _res = (x);                                                                                         \
        if (_res != MPI_SUCCESS)                                                                                       \
            throw std::runtime_error {"In function " + std::string {__FUNCTION__} +                                    \
                                      "(): " #x " failed with code: " + std::to_string(_res)};                         \
    }

#endif  //MPI_ENABLED

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
    std::string     ranks_list;
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
    hccl_demo_stats stat;

    // Warmup to sync all the gaudis on the device.
    hcclBarrier(demo_data.hccl_comm, demo_data.collective_stream);

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
#if MPI_ENABLED
        char* env_value = getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
#else
        char* env_value = getenv("HCCL_BOX_SIZE");
#endif
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

uint64_t get_demo_test_size()
{
    static bool is_cached = false;
    static uint64_t test_size = DEFAULT_TEST_SIZE;
    if (!is_cached)
    {
        char* env_value = getenv("HCCL_DEMO_TEST_SIZE");
        test_size       = (env_value != nullptr) ? strtoull(env_value, NULL, 0) : test_size;
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
#if MPI_ENABLED
    int mpi_size {};
    CHECK_MPI_STATUS(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));
    return mpi_size;
#endif  // MPI_ENABLED

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

int verify_mpi_configuration()
{
    bool mpi_enabled = false;
#if MPI_ENABLED
    mpi_enabled = true;
#endif  // MPI_ENABLED

    static auto mpi_requested = false;
    char*       env_value     = getenv("HCCL_DEMO_MPI_REQUESTED");
    mpi_requested             = (env_value != nullptr) ? atoi(env_value) : mpi_requested;

    return mpi_requested ^ mpi_enabled;
}

int get_hccl_rank()
{
#if MPI_ENABLED
    int mpi_rank {};
    CHECK_MPI_STATUS(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
    return mpi_rank;
#endif  // MPI_ENABLED

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

std::string get_ranks_list()
{
    static const string default_ranks_list = string {""};
    const char*         env_value          = getenv("HCCL_RANKS_LIST");
    return (env_value != nullptr) ? string(env_value) : default_ranks_list;
}

void describe_stat(const string&          stat_name,
                   const hccl_demo_stats& stats,
                   size_t                 data_size,
                   double                 factor,
                   int                    hccl_rank,
                   int                    loop,
                   const string&          test_type,
                   const string&          dtype,
                   const bool             reportingRank)
{
    auto algo_bandwidth = (double) data_size / stats.avg_duration_in_sec;
    auto avg_bandwidth  = algo_bandwidth * factor;
    auto rank_bandwith = (double) data_size / stats.rank_duration_in_sec;
    rank_bandwith      = rank_bandwith * factor;

    if (reportingRank)
    {
        stringstream ss;
        sleep(1);
        size_t delimiter_size = stat_name.length() + string {"[BENCHMARK]"}.length() + 1;
        ss << get_print_delimiter(delimiter_size, '#') << '\n';
        ss << "[BENCHMARK] " << stat_name << '\n';
        ss << "[BENCHMARK]     NW Bandwidth   : " << format_bw(avg_bandwidth) << '\n';
        ss << "[BENCHMARK]     Algo Bandwidth : " << format_bw(algo_bandwidth);
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

hcclResult_t send_recv_test(void*                 out_dev_ptr,
                            const void*           input_dev_ptr,
                            const size_t          count,
                            hcclComm_t            comm,
                            const synStreamHandle stream,
                            const int             recvFromRank,
                            const int             sendToRank)
{
    hcclGroupStart();

    CHECK_HCCL_STATUS(hcclSend((const void*) input_dev_ptr, count, hcclFloat32, sendToRank, comm, stream));
    CHECK_HCCL_STATUS(hcclRecv((void*) out_dev_ptr, count, hcclFloat32, recvFromRank, comm, stream));

    hcclGroupEnd();

    return hcclSuccess;
}

using RanksVector = std::vector<int>;

static hcclResult_t send_recv_ranks_test(std::vector<void*>&   output_dev_ptrs,
                                         const void*           input_dev_ptr,
                                         const size_t          count,
                                         hcclComm_t            comm,
                                         const synStreamHandle stream,
                                         const RanksVector&    recvRanks,
                                         const RanksVector&    sendRanks)
{
    hcclGroupStart();

    for (const int sendRank : sendRanks)
    {
        CHECK_HCCL_STATUS(hcclSend((const void*) input_dev_ptr, count, hcclFloat32, sendRank, comm, stream));
    }

    size_t rankCount = 0;
    for (const int recvRank : recvRanks)
    {
        CHECK_HCCL_STATUS(hcclRecv((void*) output_dev_ptrs[rankCount], count, hcclFloat32, recvRank, comm, stream));
        rankCount++;
    }

    hcclGroupEnd();

    return hcclSuccess;
}

static bool send_recv_test_driver(hccl_demo_data&           demo_data,
                                  const std::string&        test_type,
                                  const int                 hccl_rank,
                                  const uint64_t            data_size,
                                  const uint64_t            count,
                                  const std::vector<float>& input_host_data,
                                  const uint64_t            input_dev_ptr,
                                  const uint64_t            output_dev_ptr)
{
    //
    // This test does the following whether it's a single box or scaleout.
    // For single box, exchange buffer with adjacent rank. If odd number of ranks then last rank does self send/recv.
    // Fill input data with rank number, example:
    // Input        |   Output
    // G0 G1 G2 G3      G0 G1 G2 G3
    // 1  2  3  4   =>  2  1  4  3
    // 1  2  3  4       2  1  4  3
    // 1  2  3  4       2  1  4  3
    // 1  2  3  4       2  1  4  3

    // For scaleout test, exchange buffer with next peer rank in ring manner.
    //
    // Example:
    // 4 boxes: R0 -> R8 & R0 <- R24, R8 <- R0 & R8 -> R16, R16 <- R8 & R16 -> R24, R24 <- R16 & R24 ->R0 etc.
    // 2 boxes: R0 <> R8, R1 <> R9, etc.
    //
    // In both cases, each rank does 1 send and 1 recv from another (same) rank.

    bool         is_ok            = true;
    const double send_recv_factor = 1;

    const unsigned int boxSize    = static_cast<unsigned>(get_demo_box_size());
    const unsigned int numOfRanks = demo_data.nranks;
    const unsigned int numOfBoxes = numOfRanks / boxSize;

    const unsigned int ranksPerBox = numOfRanks / numOfBoxes;

    const unsigned myRank   = static_cast<unsigned>(hccl_rank);
    const unsigned myBoxNum = myRank / boxSize;

    int sendToRank   = INVALID_RANK;
    int recvFromRank = INVALID_RANK;

    if (numOfBoxes > 1)
    // scaleout
    {
        // Do ring with adjacent boxes
        const unsigned targetSendBox = myBoxNum == numOfBoxes - 1 ? 0 : myBoxNum + 1;
        sendToRank                   = targetSendBox * ranksPerBox + (myRank % ranksPerBox);
        const unsigned targetRecvBox = myBoxNum == 0 ? numOfBoxes - 1 : myBoxNum - 1;
        recvFromRank                 = targetRecvBox * ranksPerBox + (myRank % ranksPerBox);
    }
    else
    // single box
    {
        // send / recv from adjacent even/odd pairs ranks, i.e. R0 <>R1, R2<>R3.
        // in case of odd number of ranks - last rank will do send/recv with self.
        sendToRank   = (myRank % 2) != 0                                       ? myRank - 1
                       : ((numOfRanks % 2) && (myRank == numOfRanks - 1)) != 0 ? myRank
                                                                               : myRank + 1;
        recvFromRank = sendToRank;
    }

    const void* input_host_data_ptr = reinterpret_cast<const void*>(input_host_data.data());

    auto stat = benchmark(demo_data, [&]() {
        CHECK_HCCL_STATUS(send_recv_test((void*) output_dev_ptr,
                                         (const void*) input_dev_ptr,
                                         (uint64_t) input_host_data.size(),
                                         demo_data.hccl_comm,
                                         demo_data.collective_stream,
                                         recvFromRank,
                                         sendToRank));
    });

    // Correctness check

    auto        output_host_data     = vector<float>(input_host_data.size());
    const void* output_host_data_ptr = reinterpret_cast<void*>(output_host_data.data());

    CHECK_SYNAPSE_STATUS(synHostMap(demo_data.device_handle, data_size, output_host_data_ptr));

    CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.device_to_host_stream,
                                         output_dev_ptr,
                                         data_size,
                                         (uint64_t) output_host_data_ptr,
                                         DRAM_TO_HOST));
    CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.device_to_host_stream));

    for (size_t i = 0; i < output_host_data.size(); ++i)
    {
        if (abs(output_host_data[i] - (float) (recvFromRank + 1)) != 0)
        {
            is_ok = false;
        }
    }

    log() << "SendRecv hccl_rank=" << hccl_rank << " recvRank=" << recvFromRank << " size=" << data_size << " <float>"
          << " Input Buffer [" << input_host_data[0] << " " << input_host_data[1] << " " << input_host_data[2] << " "
          << input_host_data[3] << " ...]"
          << " Output Buffer [" << output_host_data[0] << " " << output_host_data[1] << " " << output_host_data[2]
          << " " << output_host_data[3] << " ...]"
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
                  "float",
                  should_report_stat(hccl_rank));
    return is_ok;
}

struct RanksPairSendRecv
{
    int sendFromRank;
    int recvInRank;
};

static std::vector<RanksPairSendRecv> parseRanksList(const std::string& ranksListSt, const int maxRankNumber)
{
    std::vector<RanksPairSendRecv> ranksList;
    std::stringstream              ss(ranksListSt);

    std::vector<int> tempRanksVector;
    std::string      token;

    while (std::getline(ss, token, ','))
    {
        const int rankNum = std::stoi(token);
        if ((rankNum >= 0) && (rankNum <= (int) (maxRankNumber)))
        {
            tempRanksVector.push_back(rankNum);
        }
        else
        {
            throw std::runtime_error {" Invalid rank number " + std::to_string(rankNum) + ", maxRankNumber=" +
                                      std::to_string(maxRankNumber) + ", ranksListSt=" + ranksListSt};
        }
    }

    if (tempRanksVector.size() % 2 != 0)
    {
        throw std::runtime_error {" Invalid ranks pairs, ranksListSt=" + ranksListSt};
    }

    if ((tempRanksVector.size() > 0) && (tempRanksVector.size() % 2 == 0))
    {
        const size_t pairsNum = tempRanksVector.size() / 2;
        for (size_t count = 0; count < pairsNum; count++)
        {
            const int sendFromRank = tempRanksVector[count * 2];
            const int recvInRank   = tempRanksVector[count * 2 + 1];
            ranksList.push_back({sendFromRank, recvInRank});
        }
    }

    return ranksList;
}

static bool send_recv_ranks_test_driver(hccl_demo_data&           demo_data,
                                        const std::string&        test_type,
                                        const int                 hccl_rank,
                                        const uint64_t            data_size,
                                        const uint64_t            count,
                                        const std::vector<float>& input_host_data,
                                        const uint64_t            input_dev_ptr,
                                        const uint64_t            output_dev_ptr)
{
    //
    // This test performs send_recv from/to specific ranks given as a list
    // A single rank can send to one or many ranks and can also recv from one or many ranks.
    // It supports both scaleup and scaleout send/recv.
    // It reports adjusted B/W according to number of recvs.

    bool   is_ok                  = true;
    double send_recv_ranks_factor = 1;

    const unsigned int numOfRanks = demo_data.nranks;

    const std::string                    ranksListStr   = demo_data.ranks_list;
    const std::vector<RanksPairSendRecv> ranksPairsList = parseRanksList(ranksListStr, numOfRanks - 1);

    // Determine how many ranks we send to or recv from, and builds a list of each
    int    reportingReceiverRank = INVALID_RANK;  // if this rank is receiver, report its adjusted BW
    int    reportingSenderRank   = INVALID_RANK;  // if this rank is sender report its BW
    size_t numberOfSenders       = 0;

    RanksVector sendToRanks;
    RanksVector recvFromRanks;

    for (const auto ranksPair : ranksPairsList)
    {
        numberOfSenders++;
        const int sendingFromRank = ranksPair.sendFromRank;
        const int receivingInRank = ranksPair.recvInRank;
        if (hccl_rank == sendingFromRank)
        {
            reportingSenderRank = sendingFromRank;
            sendToRanks.push_back(receivingInRank);
            if (true)
            {
                log() << "Rank " << hccl_rank << ", Going to send to rank " << receivingInRank << std::endl;
            }
        }
        else if (hccl_rank == receivingInRank)
        {
            reportingReceiverRank = receivingInRank;
            recvFromRanks.push_back(sendingFromRank);
            if (true)
            {
                log() << "Rank " << hccl_rank << ", Going to receive from rank " << sendingFromRank << std::endl;
            }
        }
    }

    if (hccl_rank == reportingReceiverRank)
    {
        send_recv_ranks_factor = (float) (recvFromRanks.size());
        log() << "hccl_rank=" << hccl_rank << ", numberOfSenders=" << numberOfSenders
              << ", reportingReceiverRank=" << reportingReceiverRank << ", reportingSenderRank=" << reportingSenderRank
              << std::endl;
    }

    // allocate output buffer per recv rank - could be 0 if not receiving
    std::vector<void*>    output_dev_ptrs(recvFromRanks.size(), 0);
    std::vector<uint64_t> output_dev_ptrs_uint(recvFromRanks.size(), 0);
    for (size_t rankCount = 0; rankCount < output_dev_ptrs.size(); rankCount++)
    {
        CHECK_SYNAPSE_STATUS(
            synDeviceMalloc(demo_data.device_handle, data_size, 0, 0, &(output_dev_ptrs_uint[rankCount])));
        output_dev_ptrs[rankCount] = (void*) (output_dev_ptrs_uint[rankCount]);
    }

    const void* input_host_data_ptr = reinterpret_cast<const void*>(input_host_data.data());

    auto stat = benchmark(demo_data, [&]() {
        CHECK_HCCL_STATUS(send_recv_ranks_test(output_dev_ptrs,
                                               (const void*) input_dev_ptr,
                                               (uint64_t) input_host_data.size(),
                                               demo_data.hccl_comm,
                                               demo_data.collective_stream,
                                               recvFromRanks,
                                               sendToRanks));
    });

    // Correctness check

    auto        output_host_data     = vector<float>(input_host_data.size());
    const void* output_host_data_ptr = reinterpret_cast<void*>(output_host_data.data());
    CHECK_SYNAPSE_STATUS(synHostMap(demo_data.device_handle, data_size, output_host_data_ptr));

    for (size_t rankIndex = 0; rankIndex < recvFromRanks.size(); rankIndex++)
    {
        output_host_data.assign(input_host_data.size(), 0);  // clear host test buffer
        const int recvFromRank = recvFromRanks[rankIndex];

        CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.device_to_host_stream,
                                             (output_dev_ptrs_uint[rankIndex]),
                                             data_size,
                                             (uint64_t) output_host_data_ptr,
                                             DRAM_TO_HOST));
        CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.device_to_host_stream));

        for (size_t i = 0; i < output_host_data.size(); ++i)
        {
            if (abs(output_host_data[i] - (float) (recvFromRank + 1)) != 0)
            {
                is_ok = false;
            }
        }

        log() << "SendRecv hccl_rank=" << hccl_rank << " recvRank=" << recvFromRank << " size=" << data_size
              << " <float>"
              << " Input Buffer [" << input_host_data[0] << " " << input_host_data[1] << " " << input_host_data[2]
              << " " << input_host_data[3] << " ...]"
              << " Output Buffer [" << output_host_data[0] << " " << output_host_data[1] << " " << output_host_data[2]
              << " " << output_host_data[3] << " ...]"
              << " which is " << (is_ok ? "fine." : "bad.") << endl;

        CHECK_SYNAPSE_STATUS(synDeviceFree(demo_data.device_handle, output_dev_ptrs_uint[rankIndex], 0));
    }

    CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, input_host_data_ptr));
    CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, output_host_data_ptr));

    // End of correctness check
    describe_stat("hcclSendRecv(src!=dst, count=" + to_string(input_host_data.size()) +
                      ", dtype=fp32, iterations=" + to_string(demo_data.num_iters) + ")",
                  stat,
                  data_size,
                  send_recv_ranks_factor,
                  hccl_rank,
                  demo_data.num_iters,
                  test_type,
                  "float",
                  ((hccl_rank == reportingReceiverRank) || (hccl_rank == reportingSenderRank)));
    return is_ok;
}

int main()
{
    bool is_ok = true;
    try
    {
        log() << "Running HCCL Demo :: A simple program demonstrating HCCL usage from C++" << endl;

        if (verify_mpi_configuration())
        {
            throw runtime_error {
                "HCCL demo compilation and user instruction regarding run type (MPI/pure) are non compatible. \nPlease "
                "consider to build the demo with the correct instructions or run with -clean"};
        }

#if MPI_ENABLED
        log() << "MPI enabled. Make sure that HCCL demo is launched with mpirun." << std::endl;
        // Initialize the Open MPI execution context.
        CHECK_MPI_STATUS(MPI_Init(NULL, NULL));
#endif  //MPI_ENABLED

        hccl_demo_data demo_data;
        demo_data.nranks    = get_nranks();
        demo_data.num_iters = get_demo_test_loop();
        demo_data.ranks_list = get_ranks_list();

        const int hccl_rank = get_hccl_rank();

        // Initialize Synapse API context
        CHECK_SYNAPSE_STATUS(synInitialize());

        // Acquire device
        const synModuleId device_module_id = hccl_rank % get_demo_box_size();
        CHECK_SYNAPSE_STATUS(synDeviceAcquireByModuleId(&demo_data.device_handle, device_module_id));

#if AFFINITY_ENABLED
        if (setupAffinity(device_module_id) != 0)
        {
            throw runtime_error {"Affinity setting for HCCL demo failed."};
        }
#endif
        // Create Streams
        CHECK_SYNAPSE_STATUS(
            synStreamCreateGeneric(&demo_data.collective_stream, demo_data.device_handle, 0));
        CHECK_SYNAPSE_STATUS(synStreamCreateGeneric(&demo_data.device_to_host_stream,
                                             demo_data.device_handle,
                                             0));
        CHECK_SYNAPSE_STATUS(synStreamCreateGeneric(&demo_data.host_to_device_stream,
                                             demo_data.device_handle,
                                             0));

        // Generate unique id
        hcclUniqueId  unique_id {};
        constexpr int master_mpi_rank = 0;

        if (hccl_rank == master_mpi_rank)
        {
            CHECK_HCCL_STATUS(hcclGetUniqueId(&unique_id));
        }

#if MPI_ENABLED
        CHECK_MPI_STATUS(MPI_Bcast(&unique_id, sizeof(unique_id), MPI_BYTE, master_mpi_rank, MPI_COMM_WORLD));
#endif  // MPI_ENABLED

        // Create new HCCL communicator
        CHECK_HCCL_STATUS(hcclCommInitRank(&demo_data.hccl_comm, demo_data.nranks, unique_id, hccl_rank));

        uint64_t input_dev_ptr {};
        uint64_t output_dev_ptr {};

        // Allocate buffers on the HPU device
        uint64_t    data_size           = get_demo_test_size();  // bytes
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
        const bool is_root_rank = should_report_stat(hccl_rank);
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
                          "float",
                          is_root_rank);
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
                          "float",
                          is_root_rank);
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
                          "float",
                          is_root_rank);
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
                          "float",
                          is_root_rank);
        }
        else if (test_type == "all2all")
        {
            double all2all_factor = ((double) (demo_data.nranks - 1)) / ((double) demo_data.nranks);

            // Fill input data, example:
            // Input        |   Output
            // G0 G1 G2 G3      G0 G1 G2 G3
            // 0  2  4  6   =>  0  4  8  12
            // 1  3  5  7       1  5  9  13
            // 4  5  8  10      2  6  10 14
            // 5  6  9  11      3  7  11 15
            // 8  10 12 14      4  8  12 16
            // 9  11 13 15      5  9  13 17
            // 12 14 16 18      6  10 14 18
            // 13 15 17 19      7  11 15 19
            uint64_t chunkSize = count / demo_data.nranks;
            for (uint64_t i = 0; i < count / chunkSize; ++i)
            {
                // We want to make sure we use different values on each cell and between ranks,
                // but we don't want the summation to get too big, that is why we modulo by DATA_ELEMENTS_MAX.
                for (uint64_t j = 0; j < chunkSize; ++j)
                {
                    int val                            = hccl_rank * chunkSize + j + demo_data.nranks * i;
                    input_host_data[i * chunkSize + j] = (val % DATA_ELEMENTS_MAX);
                }
            }

            // Copy from input_host_data_ptr to input_dev_ptr (to be used in benchmark)
            CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.host_to_device_stream,
                                                 (uint64_t) input_host_data_ptr,
                                                 data_size,
                                                 input_dev_ptr,
                                                 HOST_TO_DRAM));
            CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.host_to_device_stream));

            // Run HCCL AlltoAll collective
            auto stat = benchmark(demo_data, [&]() {
                CHECK_HCCL_STATUS(hcclAlltoAll((const void*) input_dev_ptr,
                                               (void*) output_dev_ptr,
                                               input_host_data.size(),
                                               hcclFloat32,
                                               demo_data.hccl_comm,
                                               demo_data.collective_stream));
            });

            // Correctness check
            auto        output_host_data     = vector<float>(input_host_data.size());
            const void* output_host_data_ptr = reinterpret_cast<void*>(output_host_data.data());

            CHECK_SYNAPSE_STATUS(synHostMap(demo_data.device_handle, data_size, output_host_data_ptr));
            CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.device_to_host_stream,
                                                 output_dev_ptr,
                                                 data_size,
                                                 (uint64_t) output_host_data_ptr,
                                                 DRAM_TO_HOST));
            CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.device_to_host_stream));

            int start = hccl_rank * (count / chunkSize);
            int expected;

            for (size_t i = 0; i < output_host_data.size(); ++i)
            {
                expected = ((start + i) % DATA_ELEMENTS_MAX);
                if ((float) output_host_data[i] != (float) expected)
                {
                    is_ok = false;
                }
            }

            log() << "All2All hccl_rank=" << hccl_rank << " size=" << data_size << " <float>"
                  << " Input Buffer [" << input_host_data[0] << " " << input_host_data[1] << " " << input_host_data[2]
                  << " " << input_host_data[3] << " ...]"
                  << " distributed to Output Buffer [" << output_host_data[0] << " " << output_host_data[1] << " "
                  << output_host_data[2] << " " << output_host_data[3] << " ...]"
                  << " which is " << (is_ok ? "fine." : "bad.") << endl;

            CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, input_host_data_ptr));
            CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, output_host_data_ptr));

            // End of correctness check

            describe_stat("hcclAlltoAll(src!=dst, count=" + to_string(input_host_data.size()) +
                              ", dtype=fp32, iterations=" + to_string(demo_data.num_iters) + ")",
                          stat,
                          data_size,
                          all2all_factor,
                          hccl_rank,
                          demo_data.num_iters,
                          test_type,
                          "float",
                          is_root_rank);
        }
        else if (test_type == "send_recv")
        {
            if (demo_data.ranks_list.length() > 0)
            {
                if (hccl_rank == get_demo_test_root())
                {
                    log() << "Will perform ranks send_recv test with list: " << get_ranks_list() << std::endl;
                }
                is_ok = send_recv_ranks_test_driver(demo_data,
                                                    test_type,
                                                    hccl_rank,
                                                    data_size,
                                                    count,
                                                    input_host_data,
                                                    input_dev_ptr,
                                                    output_dev_ptr);
            }
            else
            {
                is_ok = send_recv_test_driver(demo_data,
                                              test_type,
                                              hccl_rank,
                                              data_size,
                                              count,
                                              input_host_data,
                                              input_dev_ptr,
                                              output_dev_ptr);
            }
        }
        else if (test_type == "reduce")
        {
            double reduce_factor = 1;
            int    root          = get_demo_test_root();
            // Fill input data, example:
            // root = G1
            // Input        |   Output
            // G0 G1 G2 G3      G0 G1 G2 G3
            // 0  1  2  3   =>      6
            // 4  5  6  7          22
            // 8  9  10 11         38
            // 12 13 14 15         54

            for (uint64_t i = 0; i < count; ++i)
            {
                input_host_data[i] = hccl_rank + (demo_data.nranks * (i % DATA_ELEMENTS_MAX));
            }

            // Copy from input_host_data_ptr to input_dev_ptr (to be used in benchmark)
            CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.host_to_device_stream,
                                                 (uint64_t) input_host_data_ptr,
                                                 data_size,
                                                 input_dev_ptr,
                                                 HOST_TO_DRAM));

            CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.host_to_device_stream));

            // Run HCCL Reduce collective
            auto stat = benchmark(demo_data, [&]() {
                CHECK_HCCL_STATUS(hcclReduce((const void*) input_dev_ptr,
                                             (void*) output_dev_ptr,
                                             input_host_data.size(),
                                             hcclFloat32,
                                             hcclSum,
                                             root,
                                             demo_data.hccl_comm,
                                             demo_data.collective_stream));
            });

            // Correctness check

            auto        output_host_data     = std::vector<float>(input_host_data.size());
            const void* output_host_data_ptr = reinterpret_cast<void*>(output_host_data.data());

            log() << "Reduce hccl_rank=" << hccl_rank << " root=" << root << " size=" << data_size << " <float>"
                  << " Input Buffer [" << input_host_data[0] << " " << input_host_data[1] << " " << input_host_data[2]
                  << " " << input_host_data[3] << " ...]";

            // The correctness check is relevant for the root's output buffer only
            if (hccl_rank == root)
            {
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
                    if (std::abs(output_host_data[i] - expected) != 0)
                    {
                        is_ok = false;
                    }
                }

                log() << " Output Buffer [" << output_host_data[0] << " " << output_host_data[1] << " "
                      << output_host_data[2] << " " << output_host_data[3] << " ...]"
                      << " which is " << (is_ok ? "fine." : "bad.") << std::endl;
            }
            else
            {
                log() << std::endl;
            }

            CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, input_host_data_ptr));
            CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, output_host_data_ptr));

            // End of correctness check

            describe_stat("Reduce(count=" + std::to_string(input_host_data.size()) +
                              ", dtype=fp32, iterations=" + std::to_string(demo_data.num_iters) + ")",
                          stat,
                          data_size,
                          reduce_factor,
                          hccl_rank,
                          demo_data.num_iters,
                          test_type,
                          "float",
                          is_root_rank);
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

#if MPI_ENABLED
        CHECK_MPI_STATUS(MPI_Finalize());
#endif  // MPI_ENABLED

        if (!is_ok)
        {
            throw runtime_error {"Collective operation has failed on corretness."};
        }
    }
    catch (const exception& ex)
    {
        log() << "HCCL demo error: " << ex.what() << endl;
        return -1;
    }
    return 0;
}
