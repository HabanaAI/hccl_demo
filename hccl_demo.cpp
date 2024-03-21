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
#include <cmath>  // for pow

// HCCL :: Habana Collective Communications Library
#include <hccl.h>

// Synapse :: Habana Synapse training API
#include <synapse_api.h>

#if AFFINITY_ENABLED
#include "affinity.h"
#endif

#define DATA_ELEMENTS_MAX 13
#define DEFAULT_TEST_SIZE 33554432
#define DEFAULT_TEST_SIZE_RANGE_MIN 0
#define DEFAULT_TEST_SIZE_RANGE_MAX 0
#define DEFAULT_TEST_SIZE_RANGE_INC 1
#define DEFAULT_TEST_LOOP 10
#define DEFAULT_BOX_SIZE  8
#define ALLOCATED_HBM_SIZE          (1024 * 1024 * 1024)  // 1G
#define AMOUNT_JUMBO_BUFFERS (2)

constexpr int INVALID_RANK    = -1;
constexpr int master_mpi_rank = 0;

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
    hcclDataType_t  hccl_data_type;
    std::string     str_data_type;
    size_t          num_iters;
    std::string     ranks_list;
    int             hccl_rank;
    int             mpi_root_rank;
    uint64_t        usable_memory;
};

struct hccl_demo_stats
{
    float  rank_duration_in_sec;
    size_t num_iters;
};

struct hccl_demo_report_entry
{
    uint64_t data_size;
    uint64_t count;
    float    time;
    double   algo_bw;
    double   avg_bw;
    string   data_type;
    string   reduction_op;
};

vector<hccl_demo_report_entry> report_entry_vec;

ostream& log()
{
    return cout;
}

uint64_t get_data_type_size(const string& data_type)
{
    uint64_t data_size = sizeof(float);
    if (data_type == "float")
    {
        data_size = sizeof(float);
    }
    else if (data_type == "bfloat16")
    {
        data_size = sizeof(uint16_t);
    }
    return data_size;
}

hccl_demo_stats benchmark(hccl_demo_data& demo_data, const function<void(uint64_t)>& fn)
{
    hccl_demo_stats stat;

    // Run a single iteration for warmup to sync all the gaudis on the device.
    fn(0);

    CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.collective_stream));

    // Actual iterations
    auto start_time = Clock::now();

    for (size_t iter = 0; iter < demo_data.num_iters; ++iter)
    {
        fn(iter);
    }

    CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.collective_stream));

    auto duration             = Clock::now() - start_time;
    stat.rank_duration_in_sec = chrono::duration_cast<chrono::duration<double>>(duration).count();
    stat.rank_duration_in_sec = stat.rank_duration_in_sec / demo_data.num_iters;

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

hcclDataType_t get_demo_hccl_data_type()
{
    static bool           is_cached      = false;
    static hcclDataType_t hccl_data_type = hcclFloat32;
    if (!is_cached)
    {
        char* env_value = getenv("HCCL_DATA_TYPE");
        if (env_value == nullptr || string(env_value) == "float")
        {
            hccl_data_type = hcclFloat32;
        }
        else if (string(env_value) == "bfloat16")
        {
            hccl_data_type = hcclBfloat16;
        }
        is_cached = true;
    }
    return hccl_data_type;
}

std::string get_demo_str_data_type()
{
    static const string default_data_type = "float";
    char*               env_value         = getenv("HCCL_DATA_TYPE");
    return (env_value != nullptr) ? string(env_value) : default_data_type;
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

uint64_t get_demo_size_min()
{
    static bool     is_cached     = false;
    static uint64_t test_size_min = DEFAULT_TEST_SIZE_RANGE_MIN;
    if (!is_cached)
    {
        char* env_value = getenv("HCCL_SIZE_RANGE_MIN");
        test_size_min   = (env_value != nullptr) ? strtoull(env_value, NULL, 0) : test_size_min;
        is_cached       = true;
    }
    return test_size_min;
}

uint64_t get_demo_size_max()
{
    static bool     is_cached     = false;
    static uint64_t test_size_max = DEFAULT_TEST_SIZE_RANGE_MAX;
    if (!is_cached)
    {
        char* env_value = getenv("HCCL_SIZE_RANGE_MAX");
        test_size_max   = (env_value != nullptr) ? strtoull(env_value, NULL, 0) : test_size_max;
        is_cached       = true;
    }
    return test_size_max;
}

uint64_t get_demo_size_inc()
{
    static bool     is_cached     = false;
    static uint64_t test_size_inc = DEFAULT_TEST_SIZE_RANGE_INC;
    if (!is_cached)
    {
        char* env_value = getenv("HCCL_SIZE_RANGE_INC");
        test_size_inc   = (env_value != nullptr) ? strtoull(env_value, NULL, 0) : test_size_inc;
        is_cached       = true;
    }
    return test_size_inc;
}

bool is_master_rank(const int hccl_rank)
{
    if (hccl_rank == master_mpi_rank) return true;
    return false;
}

bool is_master_rank_unique_id(hccl_demo_data demo_data, const int hccl_rank)
{
    if (hccl_rank == demo_data.mpi_root_rank) return true;
    return false;
}

bool should_write_report(const int hccl_rank)
{
    static bool is_cached   = false;
    static bool should_write_report = false;
    if (!is_cached)
    {
        should_write_report = is_master_rank(hccl_rank) && get_demo_size_min() > 0 && get_demo_size_max() > 0;
        is_cached   = true;
    }

    return should_write_report;
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

string get_demo_custom_comm()
{
    static string custom_comm = string {""};
    char*         env_value   = getenv("HCCL_DEMO_CUSTOM_COMM");
    custom_comm               = (env_value != nullptr) ? string(env_value) : custom_comm;
    return custom_comm;
}

inline void ParseCustomCommEnv(string rank_list, vector<int>& parsed_rank_list)
{
    string delimiter = ",";
    size_t pos       = 0;

    while ((pos = rank_list.find(delimiter)) != string::npos)
    {
        parsed_rank_list.push_back(stoi(rank_list.substr(0, pos)));
        rank_list.erase(0, pos + delimiter.length());
    }

    if (!rank_list.empty())
    {
        parsed_rank_list.push_back(stoi(rank_list));
    }
    sort(parsed_rank_list.begin(), parsed_rank_list.end());
    return;
}

bool buildCustomComm(hccl_demo_data* demo_data)
{
    string rank_list = get_demo_custom_comm();
    if (rank_list.size() == 0)
    {
        // Generate HCCL comm world
        return true;
    }

    vector<int> peers;
    ParseCustomCommEnv(rank_list, peers);

    if (find(peers.begin(), peers.end(), demo_data->hccl_rank) != peers.end())
    {
        vector<int>::iterator it = find(peers.begin(), peers.end(), demo_data->hccl_rank);

        // Override params to match new custom comm
        demo_data->hccl_rank     = distance(peers.begin(), it);
        demo_data->nranks        = peers.size();
        demo_data->mpi_root_rank = *peers.begin();

        return true;
    }

    return false;
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

uint64_t get_usable_memory(const synModuleId device_module_id)
{
    uint64_t free_memory = 0;
    uint64_t total_memory = 0;  // Value is required by synDeviceGetMemoryInfo but not used
    CHECK_SYNAPSE_STATUS(
        synDeviceGetMemoryInfo(device_module_id, &free_memory, &total_memory));
    return free_memory;
}

void print_report(const string& collective_op, const size_t num_iters)
{
    constexpr size_t column_width = 14;
    const bool       is_reduction_op =
        collective_op.find("reduce") != std::string::npos;  // reduce, all_reduce, reduce_scatter

    const static vector<string> header = {"size", "count", "type", "redop", "time", "algo_bw", "nw_bw"};
    const static vector<string> units  = {"(B)", "(elements)", "", "", "(us)", "(GB/s)", "(GB/s)"};

    stringstream ss;
    const string summary   = "[SUMMARY REPORT]";
    const string stat_name = "(src!=dst, collective=" + collective_op + ", iterations=" + to_string(num_iters) + ")";
    size_t       delimiter_size = stat_name.length() + 1;
    ss << '\n' << get_print_delimiter(delimiter_size, '#') << endl;
    ss << summary << '\n' << stat_name << '\n' << endl;
    ss << left;

    // print header
    for (size_t i = 0; i < header.size(); ++i)
    {
        if (!is_reduction_op && header[i] == "redop") continue;
        ss << setw(column_width) << header[i];
    }
    ss << endl;

    // print units
    for (size_t i = 0; i < units.size(); ++i)
    {
        if (!is_reduction_op && header[i] == "redop") continue;
        ss << setw(column_width) << units[i];
    }
    ss << endl;

    // print stats for each data size
    for (const auto& entry : report_entry_vec)
    {
        ss << setw(column_width) << to_string(entry.data_size) << setw(column_width) << to_string(entry.count)
           << setw(column_width) << entry.data_type;
        if (is_reduction_op )
        {
            ss << setw(column_width) << entry.reduction_op;
        }
        ss << setw(column_width) << fixed << setprecision(3) << entry.time * 1000 << setw(column_width) << fixed
           << setprecision(6) << entry.algo_bw / 1e9 << setw(column_width) << fixed << setprecision(6)
           << entry.avg_bw / 1e9 << endl;
    }
    log() << ss.str();
}

void describe_stat(const string&          stat_name,
                   const hccl_demo_stats& stats,
                   size_t                 data_size,
                   double                 factor,
                   int                    hccl_rank,
                   int                    loop,
                   const string&          test_type,
                   const string&          data_type,
                   const string&          reduction_op,
                   const bool             reportingRank)
{
    auto algo_bandwidth = (double) data_size / stats.rank_duration_in_sec;
    auto nw_bandwidth   = algo_bandwidth * factor;
    bool write_report   = should_write_report(hccl_rank);

    if (write_report)
    {
        log() << "Processing data_size " << data_size << endl;
    }
    else if (reportingRank)
    {
        stringstream ss;
        sleep(1);
        size_t delimiter_size = stat_name.length() + string {"[BENCHMARK]"}.length() + 1;
        ss << get_print_delimiter(delimiter_size, '#') << '\n';
        ss << "[BENCHMARK] " << stat_name << '\n';
        ss << "[BENCHMARK]     NW Bandwidth   : " << format_bw(nw_bandwidth) << '\n';
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
        output << test_type << "," << hccl_rank << "," << data_type << "," << data_size << "," << loop << ","
               << format_bw(nw_bandwidth) << endl;
        output.close();
    }

    // keep the entry for the report
    if (write_report)
    {
        hccl_demo_report_entry report_entry = {data_size,
                                               (uint64_t) (data_size / get_data_type_size(data_type)),
                                               stats.rank_duration_in_sec,
                                               algo_bandwidth,
                                               nw_bandwidth,
                                               data_type,
                                               reduction_op};
        report_entry_vec.push_back(report_entry);
    }
}

hcclResult_t send_recv_test(void*                 out_dev_ptr,
                            const void*           input_dev_ptr,
                            const size_t          count,
                            hcclComm_t            comm,
                            const synStreamHandle stream,
                            const int             recvFromRank,
                            const int             sendToRank,
                            hcclDataType_t        hccl_data_type)
{
    hcclGroupStart();

    CHECK_HCCL_STATUS(hcclSend((const void*) input_dev_ptr, count, hccl_data_type, sendToRank, comm, stream));
    CHECK_HCCL_STATUS(hcclRecv((void*) out_dev_ptr, count, hccl_data_type, recvFromRank, comm, stream));

    hcclGroupEnd();

    return hcclSuccess;
}

using RanksVector = std::vector<int>;

static hcclResult_t send_recv_ranks_test(uint64_t                     iter,
                                         const std::vector<uint64_t>& output_dev_ptrs,
                                         const void*                  input_dev_ptr,
                                         const size_t                 count,
                                         hcclComm_t                   comm,
                                         const synStreamHandle        stream,
                                         const RanksVector&           recvRanks,
                                         const RanksVector&           sendRanks,
                                         hcclDataType_t               hccl_data_type)
{
    hcclGroupStart();

    for (const int sendRank : sendRanks)
    {
        CHECK_HCCL_STATUS(hcclSend((const void*) input_dev_ptr, count, hccl_data_type, sendRank, comm, stream));
    }

    uint64_t outputBufferIndex = (recvRanks.size() * iter) % output_dev_ptrs.size();
    for (const int recvRank : recvRanks)
    {
        CHECK_HCCL_STATUS(
            hcclRecv((void*) output_dev_ptrs[outputBufferIndex], count, hccl_data_type, recvRank, comm, stream));
        outputBufferIndex = (outputBufferIndex + 1 == output_dev_ptrs.size()) ? 0 : outputBufferIndex + 1;
    }

    hcclGroupEnd();

    return hcclSuccess;
}

static void send_recv_test_driver(hccl_demo_data&             demo_data,
                                  const std::string&          test_type,
                                  const int                   hccl_rank,
                                  const uint64_t              data_size,
                                  const uint64_t              count,
                                  const std::vector<uint64_t> input_dev_ptrs,
                                  const std::vector<uint64_t> output_dev_ptrs)
{
    //
    // This test does the following whether it's a single box or scale-out.
    // For single box, exchange buffer with adjacent rank. If odd number of ranks then last rank does self send/recv.
    // For scale-out test, exchange buffer with next peer rank in ring manner.
    //
    // Example:
    // 4 boxes: R0 -> R8 & R0 <- R24, R8 <- R0 & R8 -> R16, R16 <- R8 & R16 -> R24, R24 <- R16 & R24 ->R0 etc.
    // 2 boxes: R0 <> R8, R1 <> R9, etc.
    //
    // In both cases, each rank does 1 send and 1 recv from another (same) rank.

    const double       send_recv_factor = 1;
    const unsigned int boxSize          = static_cast<unsigned>(get_demo_box_size());
    const unsigned int numOfRanks       = demo_data.nranks;
    unsigned int       numOfBoxes       = numOfRanks / boxSize;
    if (numOfRanks % boxSize > 0)
    {
        numOfBoxes++;
    }
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

    auto stat = benchmark(
        demo_data,
        [&](uint64_t iter) {
            uint64_t index = iter % input_dev_ptrs.size();
            CHECK_HCCL_STATUS(send_recv_test((void*) output_dev_ptrs[index],
                                             (const void*) input_dev_ptrs[index],
                                             count,
                                             demo_data.hccl_comm,
                                             demo_data.collective_stream,
                                             recvFromRank,
                                             sendToRank,
                                             demo_data.hccl_data_type));
        });

    describe_stat("hcclSendRecv(src!=dst, data_size=" + to_string(data_size) + ", count=" + to_string(count) +
                      ", dtype=" + demo_data.str_data_type + ", iterations=" + to_string(demo_data.num_iters) + ")",
                  stat,
                  data_size,
                  send_recv_factor,
                  hccl_rank,
                  demo_data.num_iters,
                  test_type,
                  demo_data.str_data_type,
                  "",
                  should_report_stat(hccl_rank));
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

static void send_recv_ranks_test_driver(hccl_demo_data&             demo_data,
                                        const std::string&          test_type,
                                        const int                   hccl_rank,
                                        const uint64_t              data_size,
                                        const uint64_t              count,
                                        const std::vector<uint64_t> input_dev_ptrs,
                                        std::vector<uint64_t>       output_dev_ptrs)
{
    //
    // This test performs send_recv from/to specific ranks given as a list
    // A single rank can send to one or many ranks and can also recv from one or many ranks.
    // It supports both scale-up and scale-out send/recv.
    // It reports adjusted B/W according to number of receives.

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
              << ", recvFromRanks.size()=" << recvFromRanks.size() << std::endl;
    }

    if (((recvFromRanks.size() + input_dev_ptrs.size()) * data_size) > demo_data.usable_memory)
    {
        throw runtime_error("Insufficient memory for test. Required " +
                            std::to_string(recvFromRanks.size() + input_dev_ptrs.size()) + " chunks of size " +
                            std::to_string(data_size) + " bytes but only " +
                            std::to_string(demo_data.usable_memory / data_size) + " are available.");
    }

    uint64_t additional_output_dev_ptr = 0;

    if (output_dev_ptrs.size() < recvFromRanks.size())
    {
        // Allocate additional receive buffers
        uint64_t additional_buffers = recvFromRanks.size() - output_dev_ptrs.size();
        CHECK_SYNAPSE_STATUS(
            synDeviceMalloc(demo_data.device_handle, data_size * additional_buffers, 0, 0, &additional_output_dev_ptr));

        for (uint64_t index = 0; index < additional_buffers; index++)
        {
            output_dev_ptrs.push_back(additional_output_dev_ptr + (index * data_size));
        }
    }

    auto stat = benchmark(demo_data, [&](uint64_t iter) {
        const uint64_t index = iter % input_dev_ptrs.size();
        CHECK_HCCL_STATUS(send_recv_ranks_test(iter,
                                               output_dev_ptrs,
                                               (const void*) input_dev_ptrs[index],
                                               count,
                                               demo_data.hccl_comm,
                                               demo_data.collective_stream,
                                               recvFromRanks,
                                               sendToRanks,
                                               demo_data.hccl_data_type));
    });

    if (additional_output_dev_ptr)
    {
        CHECK_SYNAPSE_STATUS(synDeviceFree(demo_data.device_handle, additional_output_dev_ptr, 0));
    }

    describe_stat("hcclSendRecv(src!=dst, data_size=" + to_string(data_size) + ", count=" + to_string(count) +
                      ", dtype=" + demo_data.str_data_type + ", iterations=" + to_string(demo_data.num_iters) + ")",
                  stat,
                  data_size,
                  send_recv_ranks_factor,
                  hccl_rank,
                  demo_data.num_iters,
                  test_type,
                  demo_data.str_data_type,
                  "",
                  ((hccl_rank == reportingReceiverRank) || (hccl_rank == reportingSenderRank)));
}

int main()
{
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
        demo_data.nranks         = get_nranks();
        demo_data.num_iters      = get_demo_test_loop();
        demo_data.ranks_list     = get_ranks_list();
        demo_data.hccl_data_type = get_demo_hccl_data_type();
        demo_data.str_data_type  = get_demo_str_data_type();
        demo_data.hccl_rank      = get_hccl_rank();
        demo_data.mpi_root_rank  = 0;

        if (buildCustomComm(&demo_data) == false)
        {
            log() << "HCCL demo process id (" << demo_data.hccl_rank
                  << ") will not participate in the custom communicator" << endl;
#if MPI_ENABLED
            CHECK_MPI_STATUS(MPI_Finalize());
#endif
            return 0;
        }

        // Initialize Synapse API context
        CHECK_SYNAPSE_STATUS(synInitialize());
        // Acquire device
        const synModuleId device_module_id = get_hccl_rank() % get_demo_box_size();
        CHECK_SYNAPSE_STATUS(synDeviceAcquireByModuleId(&demo_data.device_handle, device_module_id));

#if AFFINITY_ENABLED
        if (setupAffinity(device_module_id) != 0)
        {
            throw runtime_error {"Affinity setting for HCCL demo failed."};
        }
#endif
        // Create Streams
        CHECK_SYNAPSE_STATUS(synStreamCreateGeneric(&demo_data.collective_stream, demo_data.device_handle, 0));
        CHECK_SYNAPSE_STATUS(synStreamCreateGeneric(&demo_data.device_to_host_stream, demo_data.device_handle, 0));
        CHECK_SYNAPSE_STATUS(synStreamCreateGeneric(&demo_data.host_to_device_stream, demo_data.device_handle, 0));

        // Generate unique id
        hcclUniqueId unique_id {};
        if (is_master_rank_unique_id(demo_data, get_hccl_rank()))
        {
            CHECK_HCCL_STATUS(hcclGetUniqueId(&unique_id));
        }

#if MPI_ENABLED
        CHECK_MPI_STATUS(MPI_Bcast(&unique_id, sizeof(unique_id), MPI_BYTE, demo_data.mpi_root_rank, MPI_COMM_WORLD));
#endif  // MPI_ENABLED

        // Create new HCCL communicator
        CHECK_HCCL_STATUS(hcclCommInitRank(&demo_data.hccl_comm, demo_data.nranks, unique_id, demo_data.hccl_rank));

        uint64_t              input_dev_ptr {};
        uint64_t              output_dev_ptr {};
        std::vector<uint64_t> input_dev_ptrs;
        std::vector<uint64_t> output_dev_ptrs;

        const uint64_t test_size     = get_demo_test_size();
        const uint64_t size_min      = get_demo_size_min() ? get_demo_size_min() : test_size;
        const uint64_t size_max      = get_demo_size_max() ? get_demo_size_max() : test_size;
        const uint64_t data_size_inc = get_demo_size_inc();

        // Allocate buffers on the HPU device
        string test_type = get_demo_test_type();
        // clear report vector
        report_entry_vec.clear();
        for (double size = size_min; size <= size_max; size = size * pow(2, data_size_inc))
        {
            const uint64_t data_size = (uint64_t) size;

            const uint64_t count           = data_size / get_data_type_size(demo_data.str_data_type);
            const uint64_t output_size     = (test_type == "all_gather") ? data_size * demo_data.nranks : data_size;
            const uint64_t max_buffer_size = std::max(data_size, output_size);

            demo_data.usable_memory    = get_usable_memory(device_module_id);
            uint64_t number_of_buffers = 2;
            if (max_buffer_size <= ALLOCATED_HBM_SIZE)
            {
                number_of_buffers = (ALLOCATED_HBM_SIZE / max_buffer_size) <= 2 ? AMOUNT_JUMBO_BUFFERS
                                                                                : ALLOCATED_HBM_SIZE / max_buffer_size;
            }

            CHECK_SYNAPSE_STATUS(
                synDeviceMalloc(demo_data.device_handle, data_size * number_of_buffers, 0, 0, &input_dev_ptr));
            CHECK_SYNAPSE_STATUS(
                synDeviceMalloc(demo_data.device_handle, output_size * number_of_buffers, 0, 0, &output_dev_ptr));

            for (uint64_t index = 0; index < number_of_buffers; index++)
            {
                input_dev_ptrs.push_back(input_dev_ptr + (index * data_size));
                output_dev_ptrs.push_back(output_dev_ptr + (index * output_size));
            }
            const bool is_root_rank = should_report_stat(demo_data.hccl_rank);
            if (test_type == "broadcast")
            {
                double broadcast_factor = 1;
                int    root             = get_demo_test_root();

                // Run HCCL Broadcast collective
                auto stat = benchmark(demo_data, [&](uint64_t iter) {
                    uint64_t index = iter % number_of_buffers;
                    CHECK_HCCL_STATUS(hcclBroadcast((const void*) input_dev_ptrs[index],
                                                    (void*) output_dev_ptrs[index],
                                                    count,
                                                    demo_data.hccl_data_type,
                                                    root,
                                                    demo_data.hccl_comm,
                                                    demo_data.collective_stream));
                });

                describe_stat("Broadcast(data_size=" + to_string(data_size) + ", count=" + to_string(count) +
                                  ", dtype=" + demo_data.str_data_type +
                                  ", iterations=" + to_string(demo_data.num_iters) + ")",
                              stat,
                              data_size,
                              broadcast_factor,
                              demo_data.hccl_rank,
                              demo_data.num_iters,
                              test_type,
                              demo_data.str_data_type,
                              "",
                              is_root_rank);
            }
            else if (test_type == "all_reduce")
            {
                double allreduce_factor = ((double) (2 * (demo_data.nranks - 1))) / ((double) demo_data.nranks);

                // Run HCCL AllReduce collective
                auto stat = benchmark(demo_data, [&](uint64_t iter) {
                    uint64_t index = iter % number_of_buffers;
                    CHECK_HCCL_STATUS(hcclAllReduce((const void*) input_dev_ptrs[index],
                                                    (void*) output_dev_ptrs[index],
                                                    count,
                                                    demo_data.hccl_data_type,
                                                    hcclSum,
                                                    demo_data.hccl_comm,
                                                    demo_data.collective_stream));
                });

                describe_stat("hcclAllReduce(src!=dst, data_size=" + to_string(data_size) +
                                  ", count=" + to_string(count) + ", dtype=" + demo_data.str_data_type +
                                  ", iterations=" + to_string(demo_data.num_iters) + ")",
                              stat,
                              data_size,
                              allreduce_factor,
                              demo_data.hccl_rank,
                              demo_data.num_iters,
                              test_type,
                              demo_data.str_data_type,
                              "sum",
                              is_root_rank);
            }
            else if (test_type == "reduce_scatter")
            {
                double reduce_scatter_factor = ((double) (demo_data.nranks - 1)) / ((double) demo_data.nranks);

                // Run HCCL ReduceScatter collective
                auto stat = benchmark(demo_data, [&](uint64_t iter) {
                    uint64_t index = iter % number_of_buffers;
                    CHECK_HCCL_STATUS(hcclReduceScatter((const void*) input_dev_ptrs[index],
                                                        (void*) output_dev_ptrs[index],
                                                        count / demo_data.nranks,
                                                        demo_data.hccl_data_type,
                                                        hcclSum,
                                                        demo_data.hccl_comm,
                                                        demo_data.collective_stream));
                });

                describe_stat("hcclReduceScatter(src!=dst, data_size=" + to_string(data_size) +
                                  ", count=" + to_string(count) + ", dtype=" + demo_data.str_data_type +
                                  ", iterations=" + to_string(demo_data.num_iters) + ")",
                              stat,
                              data_size,
                              reduce_scatter_factor,
                              demo_data.hccl_rank,
                              demo_data.num_iters,
                              test_type,
                              demo_data.str_data_type,
                              "sum",
                              is_root_rank);
            }
            else if (test_type == "all_gather")
            {
                double all_gather_factor = ((double) (demo_data.nranks - 1));

                // Run HCCL AllGather collective
                auto stat = benchmark(demo_data, [&](uint64_t iter) {
                    uint64_t index = iter % number_of_buffers;
                    CHECK_HCCL_STATUS(hcclAllGather((const void*) input_dev_ptrs[index],
                                                    (void*) output_dev_ptrs[index],
                                                    count,
                                                    demo_data.hccl_data_type,
                                                    demo_data.hccl_comm,
                                                    demo_data.collective_stream));
                });

                describe_stat("hcclAllGather(src!=dst, data_size=" + to_string(data_size) +
                                  ", count=" + to_string(count) + ", dtype=" + demo_data.str_data_type +
                                  ", iterations=" + to_string(demo_data.num_iters) + ")",
                              stat,
                              data_size,
                              all_gather_factor,
                              demo_data.hccl_rank,
                              demo_data.num_iters,
                              test_type,
                              demo_data.str_data_type,
                              "",
                              is_root_rank);
            }
            else if (test_type == "all2all")
            {
                double all2all_factor = ((double) (demo_data.nranks - 1)) / ((double) demo_data.nranks);

                // Run HCCL AlltoAll collective
                auto stat = benchmark(demo_data, [&](uint64_t iter) {
                    uint64_t index = iter % number_of_buffers;
                    CHECK_HCCL_STATUS(hcclAlltoAll((const void*) input_dev_ptrs[index],
                                                   (void*) output_dev_ptrs[index],
                                                   count,
                                                   demo_data.hccl_data_type,
                                                   demo_data.hccl_comm,
                                                   demo_data.collective_stream));
                });

                describe_stat("hcclAlltoAll(src!=dst, data_size=" + to_string(data_size) +
                                  ", count=" + to_string(count) + ", dtype=" + demo_data.str_data_type +
                                  ", iterations=" + to_string(demo_data.num_iters) + ")",
                              stat,
                              data_size,
                              all2all_factor,
                              demo_data.hccl_rank,
                              demo_data.num_iters,
                              test_type,
                              demo_data.str_data_type,
                              "",
                              is_root_rank);
            }
            else if (test_type == "send_recv")
            {
                if (demo_data.ranks_list.length() > 0)
                {
                    if (demo_data.hccl_rank == get_demo_test_root())
                    {
                        log() << "Will perform ranks send_recv test with list: " << get_ranks_list() << std::endl;
                    }
                    send_recv_ranks_test_driver(demo_data,
                                                test_type,
                                                demo_data.hccl_rank,
                                                data_size,
                                                count,
                                                input_dev_ptrs,
                                                output_dev_ptrs);
                }
                else
                {
                    send_recv_test_driver(demo_data,
                                          test_type,
                                          demo_data.hccl_rank,
                                          data_size,
                                          count,
                                          input_dev_ptrs,
                                          output_dev_ptrs);
                }
            }
            else if (test_type == "reduce")
            {
                double reduce_factor = 1;
                int    root          = get_demo_test_root();

                // Run HCCL Reduce collective
                auto stat = benchmark(demo_data, [&](uint64_t iter) {
                    uint64_t index = iter % number_of_buffers;
                    CHECK_HCCL_STATUS(hcclReduce((const void*) input_dev_ptrs[index],
                                                 (void*) output_dev_ptrs[index],
                                                 count,
                                                 demo_data.hccl_data_type,
                                                 hcclSum,
                                                 root,
                                                 demo_data.hccl_comm,
                                                 demo_data.collective_stream));
                });

                describe_stat("Reduce(data_size=" + to_string(data_size) + ", count=" + std::to_string(count) +
                                  ", dtype=" + demo_data.str_data_type +
                                  ", iterations=" + std::to_string(demo_data.num_iters) + ")",
                              stat,
                              data_size,
                              reduce_factor,
                              demo_data.hccl_rank,
                              demo_data.num_iters,
                              test_type,
                              demo_data.str_data_type,
                              "sum",
                              is_root_rank);
            }
            else
            {
                log() << "Unknown test type (" << test_type << ")" << endl;
                return -1;
            }

            CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.collective_stream));

            CHECK_SYNAPSE_STATUS(synDeviceFree(demo_data.device_handle, input_dev_ptr, 0));
            CHECK_SYNAPSE_STATUS(synDeviceFree(demo_data.device_handle, output_dev_ptr, 0));
            input_dev_ptrs.clear();
            output_dev_ptrs.clear();
        }

        if (should_write_report(demo_data.hccl_rank))
        {
            print_report(test_type, demo_data.num_iters);
        }

        // Destroy HCCL communicator
        CHECK_HCCL_STATUS(hcclCommDestroy(demo_data.hccl_comm));

        // destroy streams
        CHECK_SYNAPSE_STATUS(synStreamDestroy(demo_data.collective_stream));
        CHECK_SYNAPSE_STATUS(synStreamDestroy(demo_data.device_to_host_stream));
        CHECK_SYNAPSE_STATUS(synStreamDestroy(demo_data.host_to_device_stream));

        // Clean up HCCL
        CHECK_SYNAPSE_STATUS(synDeviceRelease(demo_data.device_handle));

        // Destroy synapse api context
        CHECK_SYNAPSE_STATUS(synDestroy());

#if MPI_ENABLED
        CHECK_MPI_STATUS(MPI_Finalize());
#endif  // MPI_ENABLED
    }
    catch (const exception& ex)
    {
        log() << "HCCL demo error: " << ex.what() << endl;
        return -1;
    }
    return 0;
}
