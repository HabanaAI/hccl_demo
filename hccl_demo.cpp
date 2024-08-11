/******************************************************************************
# Copyright (c) 2022 Habana Labs, Ltd.
# SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

// C++ Standard Libraries
#include <algorithm>
#include <cstdint>
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
#include <cstring> // for std::memcopy

// HCCL :: Habana Collective Communications Library
#include <hccl.h>

// Synapse :: Habana Synapse training API
#include <synapse_api.h>

#if AFFINITY_ENABLED
#include "affinity.h"
#endif

#define DATA_ELEMENTS_MAX           13
#define DEFAULT_TEST_SIZE           33554432
#define DEFAULT_TEST_SIZE_RANGE_MIN 0
#define DEFAULT_TEST_SIZE_RANGE_MAX 0
#define DEFAULT_TEST_SIZE_RANGE_INC 1
#define DEFAULT_TEST_LOOP           10
#define DEFAULT_BOX_SIZE            8
#define ALLOCATED_HBM_SIZE          (2UL * 1024 * 1024 * 1024)  // 2G
#define AMOUNT_JUMBO_BUFFERS        (2)
#define MAX_BUFFER_COUNT            (33UL)

constexpr int      INVALID_RANK      = -1;
constexpr int      master_mpi_rank   = 0;

// scale_test specific constants
constexpr float SCALE_VALIDATION_MARGIN = 0.05;  // fraction of expected BW

enum class CONTROL_TYPE : uint8_t
{
    SEND = 1,
    RECEIVE,
    END,
};

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

#define ASSERT(x)                                                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(x)) throw runtime_error {"In function " + string {__FUNCTION__} + " assertion failed"};                  \
    } while (false)

inline uint16_t floatToBf16(const float f)
{
    return ((*(const uint32_t*) &f) >> 16) & 0xffff;
}

inline float bf16ToFloat(const uint16_t a)
{
    float          val_fp32;
    const uint32_t val_32b = ((uint32_t) a) << 16;
    static_assert(sizeof(float) == sizeof(val_32b), "`float` size is incompatible!");
    std::memcpy(&val_fp32, &val_32b, sizeof(float));
    return val_fp32;
}

inline float bf16AccuracyCoefficient(unsigned numberOfRanks)
{
    numberOfRanks = (numberOfRanks > 8) ? 8 : numberOfRanks;
    return (numberOfRanks > 1) ? (float)numberOfRanks / 256.0 : 0.0;  // For 1 rank, tolerance should be 0
}

template<class T>
float get_float(T value);

template<>
float get_float<float>(float value){
    return value;
}

template<>
float get_float<uint16_t>(uint16_t value){
    return bf16ToFloat(value);
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

bool correctness_check_function(hccl_demo_data demo_data, float expected, float out_value, int i)
{
    const float accuracyCoefficient = bf16AccuracyCoefficient(demo_data.nranks);
    const float tolerance           = fabs(out_value) * accuracyCoefficient;
    const float difference          = fabs(out_value - expected);

    if (difference > tolerance)
    {
        log() << "index=" << i << ", expectedValue=" << expected << ", value=" << out_value
              << ", thisRankId=" << demo_data.hccl_rank << ", tolerance=" << tolerance << ", difference=" << difference
              << ", accuracyCoefficient=" << accuracyCoefficient << ", m_numberOfRanks=" << demo_data.nranks
              << std::endl;
        return false;
    }
    return true;
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

hccl_demo_stats benchmark(hccl_demo_data& demo_data, const function<void(uint64_t)>& fn, const std::function<void()>& fn_correctness){
    hccl_demo_stats stat;

    // Run a single iteration for warmup to sync all the gaudis on the device.
    fn(0);

    CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.collective_stream));

    // Actual iterations
    auto start_time = Clock::now();

    for (size_t iter = 1; iter < demo_data.num_iters; ++iter)
    {
        fn(iter);
    }

    // Correctness run on separate device output buffer
    fn_correctness();

    CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.collective_stream));

    auto duration             = Clock::now() - start_time;
    stat.rank_duration_in_sec = chrono::duration_cast<chrono::duration<double>>(duration).count();
    stat.rank_duration_in_sec = stat.rank_duration_in_sec / demo_data.num_iters;

    return stat;
}

bool check_correctness()
{
    static const auto default_hccl_check_correctness = true;
    const char*       env_value                      = getenv("HCCL_DEMO_CHECK_CORRECTNESS");
    return (env_value != nullptr) ? atoi(env_value) : default_hccl_check_correctness;
}

bool should_report_stat(int rank)
{
    return rank == 0;
}

inline string format_bw(const double bytes_per_sec)
{
    stringstream ss;
    ss << fixed << setprecision(6) << bytes_per_sec / 1e9 << " GB/s";
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

int get_demo_ranks_per_node()
{
    static bool is_cached = false;
    static auto ranks_per_node = DEFAULT_BOX_SIZE;
    if (!is_cached)
    {
#if MPI_ENABLED
        char* env_value = getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
#else
        char* env_value = getenv("HCCL_RANKS_PER_NODE");
#endif
        ranks_per_node  = (env_value != nullptr) ? atoi(env_value) : ranks_per_node;
        is_cached       = true;
    }
    return ranks_per_node;
}

int get_demo_scaleup_group_size()
{
    static bool is_cached          = false;
    static auto scaleup_group_size = 0;
    if (!is_cached)
    {
        char* env_value = getenv("HCCL_SCALEUP_GROUP_SIZE");  // Allow override for both MPI and non-MPI
        if (env_value == nullptr)
        {
            // get default value from MPI if possible or use ranks_per_node as default
            env_value = getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
        }
        scaleup_group_size = (env_value != nullptr) ? atoi(env_value) : get_demo_ranks_per_node();
        is_cached          = true;
    }
    return scaleup_group_size;
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
    return hccl_rank == master_mpi_rank;
}

bool is_master_rank_unique_id(hccl_demo_data demo_data, const int hccl_rank)
{
    return hccl_rank == demo_data.mpi_root_rank;
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

synDeviceType get_device_type(const synDeviceId deviceId)
{
    synDeviceInfoV2 deviceInfo;
    CHECK_SYNAPSE_STATUS(synDeviceGetInfoV2(deviceId, &deviceInfo));
    return deviceInfo.deviceType;
}

uint64_t get_demo_expected_scaleup_bw(const synDeviceId deviceId)
{
    synDeviceType deviceType = get_device_type(deviceId);
    switch (deviceType)
    {
        case synDeviceGaudi: return 12.5e9;
        case synDeviceGaudi2: return 37.5e9;
        case synDeviceGaudi3: return 75e9;
        default:
            log() << "Unknown device, setting expected scaleup bandwidth to 37.5GB" << endl;
            return 37.5e9;
    }
}

uint64_t get_demo_expected_scaleout_bw()
{
    static bool     is_cached        = false;
    static uint64_t test_expected_bw = 0;
    if (!is_cached)
    {
        const char* const env_value = getenv("HCCL_EXPECTED_SCALEOUT_BW");
        if (env_value == nullptr)
        {
            throw std::runtime_error {"missing mandatory argument for scale validation: --scaleout_bw"};
        }
        test_expected_bw = strtoull(env_value, NULL, 0);
        is_cached        = true;
    }
    return test_expected_bw;
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
    demo_data->mpi_root_rank = *peers.begin();

    if (find(peers.begin(), peers.end(), demo_data->hccl_rank) != peers.end())
    {
        vector<int>::iterator it = find(peers.begin(), peers.end(), demo_data->hccl_rank);

        // Override params to match new custom comm
        demo_data->hccl_rank     = distance(peers.begin(), it);
        demo_data->nranks        = peers.size();

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

std::string get_reduction_op_str()
{
    const char* env_value = getenv("HCCL_REDUCTION_OP");
    if (env_value != nullptr) return string(env_value);
    throw std::runtime_error {" Unknown reduction op."};
}

hcclRedOp_t get_reduction_op()
{
    const char* env_value = getenv("HCCL_REDUCTION_OP");

    if (0 == strcmp(env_value, "sum")) return hcclSum;
    if (0 == strcmp(env_value, "min")) return hcclMin;
    if (0 == strcmp(env_value, "max")) return hcclMax;

    throw std::runtime_error {" Unknown reduction op."};
}

uint64_t get_usable_memory(const synDeviceId device_id)
{
    uint64_t free_memory = 0;
    uint64_t total_memory = 0;  // Value is required by synDeviceGetMemoryInfo but not used
    CHECK_SYNAPSE_STATUS(synDeviceGetMemoryInfo(device_id, &free_memory, &total_memory));
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

string create_data_csv_file(hccl_demo_data demo_data,string test_type, string type){
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream datetime;
    datetime << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H%M");
    string file_name = "HCCL_demo_" + test_type + "_" + type + "_" + to_string(demo_data.hccl_rank) + "_" + datetime.str() + ".csv";
    fstream fout;
    fout.open(file_name, ios::out | ios::app);
    fout.close();
    return file_name;
}

template<class T>
void copy_data_to_csv_report(vector<T> host_data,int count, string data_csv_path, bool bf16_convert){
    fstream fout;
    fout.open(data_csv_path, ios::out | ios::app);
    for (int i = 0; i < count; i++)
    {
        if (bf16_convert) fout << bf16ToFloat(host_data[i]) << "\n";
        else
            fout << host_data[i] << "\n";
    }
    fout.close();
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

template<class T>
static bool send_recv_test_driver(hccl_demo_data&             demo_data,
                                  const std::string&          test_type,
                                  const int                   hccl_rank,
                                  const uint64_t              data_size,
                                  const uint64_t              count,
                                  const std::vector<T>&       input_host_data,
                                  const std::vector<uint64_t> input_dev_ptrs,
                                  const std::vector<uint64_t> output_dev_ptrs,
                                  uint64_t                    correctness_dev_ptr,
                                  bool                        bf16_convert,
                                  bool                        data_csv_enabled,
                                  string                      data_csv_path_output)
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

    bool is_ok = true;
    const double       send_recv_factor = 1;
    // const unsigned int boxSize          = static_cast<unsigned>(get_demo_ranks_per_node());
    const unsigned int scaleupGroupSize = get_demo_scaleup_group_size();
    const unsigned int numOfRanks       = demo_data.nranks;
    unsigned int       numOfBoxes       = numOfRanks / scaleupGroupSize;
    if (numOfRanks % scaleupGroupSize > 0)
    {
        numOfBoxes++;
    }
    const unsigned int ranksPerBox = numOfRanks / numOfBoxes;

    const unsigned myRank   = static_cast<unsigned>(hccl_rank);
    const unsigned myBoxNum = myRank / scaleupGroupSize;

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
    auto stat = benchmark(
        demo_data,
        [&](uint64_t iter) {
            uint64_t index = iter % input_dev_ptrs.size();
            CHECK_HCCL_STATUS(send_recv_test((void*) output_dev_ptrs[index],
                                             (const void*) input_dev_ptrs[index],
                                             (uint64_t) count,
                                             demo_data.hccl_comm,
                                             demo_data.collective_stream,
                                             recvFromRank,
                                             sendToRank,
                                             demo_data.hccl_data_type));
        },
        [&]() {
            CHECK_HCCL_STATUS(send_recv_test((void*) correctness_dev_ptr,
                                             (const void*) input_dev_ptrs[0],
                                             count,
                                             demo_data.hccl_comm,
                                             demo_data.collective_stream,
                                             recvFromRank,
                                             sendToRank,
                                             demo_data.hccl_data_type));
        });

    // Correctness check
    auto output_host_data = vector<T>(input_host_data.size());
    const void* output_host_data_ptr = reinterpret_cast<void*>(output_host_data.data());
    CHECK_SYNAPSE_STATUS(synHostMap(demo_data.device_handle, data_size, output_host_data_ptr));

    CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.device_to_host_stream,
                                         correctness_dev_ptr,
                                         data_size,
                                         (uint64_t) output_host_data_ptr,
                                         DRAM_TO_HOST));
    CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.device_to_host_stream));

    if (check_correctness())
    {
        size_t loop_size = data_size / get_demo_test_size();
        if (bf16_convert)
        {
            for (size_t i = 0; i < loop_size; ++i)
            {
                is_ok = correctness_check_function(demo_data, recvFromRank + 1, bf16ToFloat(output_host_data[i]), i);
            }
        }
        else
        {
            for (size_t i = 0; i < loop_size; ++i)
            {
                if (abs(output_host_data[i] - (float) (recvFromRank + 1)) != 0)
                {
                    is_ok = false;
                }
            }
        }
        log() << "SendRecv hccl_rank=" << demo_data.hccl_rank << " size=" << data_size << " <"
              << demo_data.str_data_type << ">"
              << " Input Buffer [" << get_float(input_host_data[0]) << " " << get_float(input_host_data[1]) << " "
              << get_float(input_host_data[2]) << " " << get_float(input_host_data[3]) << " ...]"
              << " reduced to Output Buffer [" << get_float(output_host_data[0]) << " "
              << get_float(output_host_data[1]) << " " << get_float(output_host_data[2]) << " "
              << get_float(output_host_data[3]) << " ...]"
              << " which is " << (is_ok ? "fine." : "bad.") << endl;
    }

    if (data_csv_enabled)
    {
        copy_data_to_csv_report(output_host_data, count, data_csv_path_output, bf16_convert);
    }
    CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, input_host_data_ptr));
    CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, output_host_data_ptr));

    // End of correctness check
    describe_stat("hcclSendRecv(src!=dst, data_size=" + to_string(data_size) + ", count=" + to_string(input_host_data.size()) +
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

template<class T>
static bool send_recv_ranks_test_driver(hccl_demo_data&             demo_data,
                                        const std::string&          test_type,
                                        const uint64_t              data_size,
                                        const uint64_t              count,
                                        const std::vector<T>&       input_host_data,
                                        const std::vector<uint64_t> input_dev_ptrs,
                                        std::vector<uint64_t>       output_dev_ptrs,
                                        bool                        bf16_convert)

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
        if (demo_data.hccl_rank == sendingFromRank)
        {
            reportingSenderRank = sendingFromRank;
            sendToRanks.push_back(receivingInRank);
            if (true)
            {
                log() << "Rank " << demo_data.hccl_rank << ", Going to send to rank " << receivingInRank << std::endl;
            }
        }
        else if (demo_data.hccl_rank == receivingInRank)
        {
            reportingReceiverRank = receivingInRank;
            recvFromRanks.push_back(sendingFromRank);
            if (true)
            {
                log() << "Rank " << demo_data.hccl_rank << ", Going to receive from rank " << sendingFromRank << std::endl;
            }
        }
    }

    if (demo_data.hccl_rank == reportingReceiverRank)
    {
        send_recv_ranks_factor = (float) (recvFromRanks.size());
        log() << "hccl_rank=" << demo_data.hccl_rank << ", numberOfSenders=" << numberOfSenders
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

     if (output_dev_ptrs.size() < recvFromRanks.size())
    {
        throw runtime_error {"Number of allocated receive buffers isn't sufficient to fulfill number of receives"};
    }

    auto stat = benchmark(
        demo_data,
        [&](uint64_t iter) {
            uint64_t index = iter % input_dev_ptrs.size();
            CHECK_HCCL_STATUS(send_recv_ranks_test(iter,
                                                   output_dev_ptrs,
                                                   (const void*) input_dev_ptrs[index],
                                                   count,
                                                   demo_data.hccl_comm,
                                                   demo_data.collective_stream,
                                                   recvFromRanks,
                                                   sendToRanks,
                                                   demo_data.hccl_data_type));
        },
        [&]() -> void {
            CHECK_HCCL_STATUS(send_recv_ranks_test(0,
                                                   output_dev_ptrs,
                                                   (const void*) input_dev_ptrs[0],
                                                   count,
                                                   demo_data.hccl_comm,
                                                   demo_data.collective_stream,
                                                   recvFromRanks,
                                                   sendToRanks,
                                                   demo_data.hccl_data_type));
        });

    describe_stat("hcclSendRecv(src!=dst, data_size=" + to_string(data_size) +
                      ", count=" + to_string(input_host_data.size()) + ", dtype=" + demo_data.str_data_type +
                      ", iterations=" + to_string(demo_data.num_iters) + ")",
                  stat,
                  data_size,
                  send_recv_ranks_factor,
                  demo_data.hccl_rank,
                  demo_data.num_iters,
                  test_type,
                  demo_data.str_data_type,
                  "",
                  ((demo_data.hccl_rank == reportingReceiverRank) || (demo_data.hccl_rank == reportingSenderRank)));
    return true;
}

#ifdef MPI_ENABLED
static void scaleup_pairs(const int nranksGroupSize, std::vector<RanksPairSendRecv>& ranksList)
{
    const int scaleupGroupSize = get_demo_scaleup_group_size();

    for (int sender = 0; sender < nranksGroupSize; sender++)
    {
        int boxNum = sender / scaleupGroupSize;
        for (int receiver = boxNum * scaleupGroupSize; receiver < (boxNum + 1) * scaleupGroupSize; receiver++)
        {
            if (sender == receiver) continue;
            ranksList.push_back({sender, receiver});
        }
    }
}

static void scaleout_pairs(const int nranksGroupSize, std::vector<RanksPairSendRecv>& ranksList)
{
    const int scaleupGroupSize = get_demo_scaleup_group_size();

    for (int sender = 0; sender < nranksGroupSize; sender++)
    {
        for (int receiver = sender % scaleupGroupSize; receiver < nranksGroupSize; receiver += scaleupGroupSize)
        {
            if (sender == receiver) continue;
            ranksList.push_back({sender, receiver});
        }
    }
}

static void scale_test_send(hccl_demo_data& demo_data,
                            int             receiver,
                            uint64_t        count,
                            uint64_t        data_size,
                            uint64_t        input_dev_ptr,
                            uint64_t&       result)
{
    // run single iteration as warmup
    CHECK_HCCL_STATUS(hcclSend((void*) input_dev_ptr,
                               count,
                               demo_data.hccl_data_type,
                               receiver,
                               demo_data.hccl_comm,
                               demo_data.collective_stream));

    CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.collective_stream));

    auto start_time = Clock::now();
    for (size_t i = 0; i < demo_data.num_iters; ++i)
    {
        CHECK_HCCL_STATUS(hcclSend((void*) input_dev_ptr,
                                   count,
                                   demo_data.hccl_data_type,
                                   receiver,
                                   demo_data.hccl_comm,
                                   demo_data.collective_stream));
    }

    // calculate result
    CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.collective_stream));

    auto duration             = Clock::now() - start_time;
    auto rank_duration_in_sec = chrono::duration_cast<chrono::duration<double>>(duration).count();
    rank_duration_in_sec      = rank_duration_in_sec / demo_data.num_iters;
    result                    = data_size / rank_duration_in_sec;
}

static void scale_test_receive(hccl_demo_data& demo_data, int sender, uint64_t count, uint64_t output_dev_ptr)
{
    // run single iteration as warmup
    CHECK_HCCL_STATUS(hcclRecv((void*) output_dev_ptr,
                               count,
                               demo_data.hccl_data_type,
                               sender,
                               demo_data.hccl_comm,
                               demo_data.collective_stream));

    for (size_t i = 0; i < demo_data.num_iters; ++i)
    {
        CHECK_HCCL_STATUS(hcclRecv((void*) output_dev_ptr,
                                   count,
                                   demo_data.hccl_data_type,
                                   sender,
                                   demo_data.hccl_comm,
                                   demo_data.collective_stream));
    }
}

static void scale_test_end(const int nranksGroupSize)
{
    for (int rank = 0; rank < nranksGroupSize; rank++)
    {
        if (!is_master_rank(rank))
        {
            CONTROL_TYPE control = CONTROL_TYPE::END;
            CHECK_MPI_STATUS(MPI_Send((void*) &control, 1, MPI_UINT8_T, rank, 0, MPI_COMM_WORLD));
        }
    }
}

static void scale_test_server_step(hccl_demo_data& demo_data,
                                   int             sender,
                                   int             receiver,
                                   uint64_t        count,
                                   uint64_t        data_size,
                                   uint64_t        input_dev_ptr,
                                   uint64_t        output_dev_ptr)
{
    CONTROL_TYPE control;
    MPI_Status   status;

    // trigger receiver
    if (demo_data.hccl_rank != receiver)  // do server receive only after sending send request
    {
        control = CONTROL_TYPE::RECEIVE;
        CHECK_MPI_STATUS(MPI_Send((void*) &control, 1, MPI_UINT8_T, receiver, 0, MPI_COMM_WORLD));
        CHECK_MPI_STATUS(MPI_Send((void*) &sender, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD));
    }

    // trigger sender
    if (demo_data.hccl_rank != sender)  // server send at the result stage
    {
        control = CONTROL_TYPE::SEND;
        CHECK_MPI_STATUS(MPI_Send((void*) &control, 1, MPI_UINT8_T, sender, 0, MPI_COMM_WORLD));
        CHECK_MPI_STATUS(MPI_Send((void*) &receiver, 1, MPI_INT, sender, 0, MPI_COMM_WORLD));
    }

    // server receive
    if (demo_data.hccl_rank == receiver)
    {
        scale_test_receive(demo_data, sender, count, output_dev_ptr);
    }

    // get result
    uint64_t result;
    if (demo_data.hccl_rank == sender)
    {
        scale_test_send(demo_data, receiver, count, data_size, input_dev_ptr, result);
    }
    else
    {
        // wait for result
        CHECK_MPI_STATUS(MPI_Recv((void*) &result, 1, MPI_UINT64_T, sender, 0, MPI_COMM_WORLD, &status))
    }

    // log
    const int scaleupGroupSize = get_demo_scaleup_group_size();
    uint64_t  expected_result  = (sender / scaleupGroupSize) == (receiver / scaleupGroupSize)
                                     ? get_demo_expected_scaleup_bw(demo_data.device_handle)
                                     : get_demo_expected_scaleout_bw();

    if (result < (expected_result - SCALE_VALIDATION_MARGIN * expected_result))
    {
        log() << sender << "-->" << receiver << ": " << format_bw(result) << " (" << result * 100 / expected_result
              << "%)" << endl;
    }
}

static void scale_test_client(
    hccl_demo_data& demo_data, uint64_t count, uint64_t data_size, uint64_t input_dev_ptr, uint64_t output_dev_ptr)
{
    while (true)
    {
        MPI_Status   status;
        CONTROL_TYPE control;
        CHECK_MPI_STATUS(MPI_Recv((void*) &control, 1, MPI_UINT8_T, master_mpi_rank, 0, MPI_COMM_WORLD, &status))

        switch (control)
        {
            case CONTROL_TYPE::END:
                return;
            case CONTROL_TYPE::SEND:
                uint64_t result;
                int      receiver;
                CHECK_MPI_STATUS(MPI_Recv((void*) &receiver, 1, MPI_INT, master_mpi_rank, 0, MPI_COMM_WORLD, &status))
                scale_test_send(demo_data, receiver, count, data_size, input_dev_ptr, result);

                // send result to server
                CHECK_MPI_STATUS(MPI_Send((void*) &result, 1, MPI_UINT64_T, master_mpi_rank, 0, MPI_COMM_WORLD));
                break;
            case CONTROL_TYPE::RECEIVE:
                int sender;
                CHECK_MPI_STATUS(MPI_Recv((void*) &sender, 1, MPI_INT, master_mpi_rank, 0, MPI_COMM_WORLD, &status))
                scale_test_receive(demo_data, sender, count, output_dev_ptr);
                break;
            default:
                throw std::runtime_error {" Unexpected control message type"};
        }
    }
}

static void scale_test_common_driver(hccl_demo_data&                 demo_data,
                                     uint64_t                        count,
                                     uint64_t                        data_size,
                                     uint64_t                        input_dev_ptr,
                                     uint64_t                        output_dev_ptr,
                                     std::vector<RanksPairSendRecv>& ranksPairsList)
{
    if (is_master_rank(demo_data.hccl_rank))
    {
        for (RanksPairSendRecv pair : ranksPairsList)
        {
            scale_test_server_step(demo_data,
                                   pair.sendFromRank,
                                   pair.recvInRank,
                                   count,
                                   data_size,
                                   input_dev_ptr,
                                   output_dev_ptr);
        }
        scale_test_end(demo_data.nranks);
    }
    else
    {
        scale_test_client(demo_data, count, data_size, input_dev_ptr, output_dev_ptr);
    }
}

static void scale_test_driver(hccl_demo_data& demo_data,
                              const uint64_t  count,
                              const uint64_t  data_size,
                              const uint64_t  input_dev_ptr,
                              uint64_t        output_dev_ptr)
{
    // scaleup
    if (is_master_rank(demo_data.hccl_rank))
    {
        log() << "ScaleUp - Expected " << format_bw(get_demo_expected_scaleup_bw(demo_data.device_handle)) << endl;
    }
    std::vector<RanksPairSendRecv> scaleupPairsList;
    scaleup_pairs(demo_data.nranks, scaleupPairsList);
    scale_test_common_driver(demo_data, count, data_size, input_dev_ptr, output_dev_ptr, scaleupPairsList);

    // scaleout
    if (demo_data.nranks > (size_t) get_demo_ranks_per_node())
    {
        if (is_master_rank(demo_data.hccl_rank))
        {
            log() << "ScaleOut - Expected " << format_bw(get_demo_expected_scaleout_bw()) << endl;
        }
        std::vector<RanksPairSendRecv> scaleoutPairsList;
        scaleout_pairs(demo_data.nranks, scaleoutPairsList);
        scale_test_common_driver(demo_data, count, data_size, input_dev_ptr, output_dev_ptr, scaleoutPairsList);
    }
}
#endif  //MPI_ENABLED

template<class T>
T calc_expected_reduction(std::vector<T>& args, hcclRedOp_t reduction_op)
{
    switch (reduction_op)
    {
        case hcclSum: return std::accumulate(args.cbegin(), args.cend(), 0);
        case hcclMin: return *std::min_element(args.cbegin(), args.cend());
        case hcclMax: return *std::max_element(args.cbegin(), args.cend());
        default: throw std::runtime_error {" Unknown reduction op."};
    }
}

template<class T>
int run_test(hccl_demo_data demo_data, bool bf16_convert, const synModuleId device_module_id)
{
    bool                  is_ok = true;
    uint64_t              input_dev_ptr       = 0;
    uint64_t              output_dev_ptr      = 0;
    uint64_t              correctness_dev_ptr = 0;
    std::vector<uint64_t> input_dev_ptrs;
    std::vector<uint64_t> output_dev_ptrs;

    const uint64_t test_size     = get_demo_test_size();
    const uint64_t size_min      = get_demo_size_min() ? get_demo_size_min() : test_size;
    const uint64_t size_max      = get_demo_size_max() ? get_demo_size_max() : test_size;
    const uint64_t data_size_inc = get_demo_size_inc();

    // Allocate buffers on the HPU device
    const uint64_t data_type_size = get_data_type_size(demo_data.str_data_type);
    string         test_type      = get_demo_test_type();
    vector<T>      input_host_data;
    void*          input_host_data_ptr;
    // clear report vector
    report_entry_vec.clear();
    // Create csv files for data csv flag if enabled
    char*      env_value            = getenv("HCCL_DEMO_DATA_CSV");
    const bool data_csv_enabled     = (env_value != nullptr) ? string(env_value).compare("True") == 0 : false;
    string     data_csv_path_input  = "";
    string     data_csv_path_output = "";
    if (data_csv_enabled)
    {
        data_csv_path_input  = create_data_csv_file(demo_data, test_type, "input");
        data_csv_path_output = create_data_csv_file(demo_data, test_type, "output");
    }

    for (double size = size_min; size <= size_max; size = size * pow(2, data_size_inc))
    {
        const uint64_t data_size = (uint64_t) size;
        uint64_t       count     = data_size / data_type_size;
        input_host_data          = vector<T>(count);
        input_host_data_ptr      = reinterpret_cast<void*>(input_host_data.data());

        const uint64_t output_size     = (test_type == "all_gather") ? data_size * demo_data.nranks : data_size;
        const uint64_t max_buffer_size = std::max(data_size, output_size);
        demo_data.usable_memory        = get_usable_memory(demo_data.device_handle);
        uint64_t number_of_buffers     = 2;
        if (max_buffer_size <= ALLOCATED_HBM_SIZE)
        {
            number_of_buffers = (ALLOCATED_HBM_SIZE / max_buffer_size) <= 2 ? AMOUNT_JUMBO_BUFFERS
                                                                            : ALLOCATED_HBM_SIZE / max_buffer_size;
        }
        number_of_buffers = std::min(number_of_buffers, MAX_BUFFER_COUNT);

        CHECK_SYNAPSE_STATUS(
            synDeviceMalloc(demo_data.device_handle, data_size * number_of_buffers, 0, 0, &input_dev_ptr));
        CHECK_SYNAPSE_STATUS(
            synDeviceMalloc(demo_data.device_handle, output_size * number_of_buffers, 0, 0, &output_dev_ptr));
        CHECK_SYNAPSE_STATUS(synDeviceMalloc(demo_data.device_handle, output_size, 0, 0, &correctness_dev_ptr));
        CHECK_SYNAPSE_STATUS(synHostMap(demo_data.device_handle, data_size, input_host_data_ptr));
        CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.host_to_device_stream,
                                             (uint64_t) input_host_data_ptr,
                                             data_size,
                                             input_dev_ptr,
                                             HOST_TO_DRAM));

        for (uint64_t index = 0; index < number_of_buffers; index++)
        {
            input_dev_ptrs.push_back(input_dev_ptr + (index * data_size));
            output_dev_ptrs.push_back(output_dev_ptr + (index * output_size));
        }
        CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.host_to_device_stream));
        const bool is_root_rank = should_report_stat(demo_data.hccl_rank);

        if (test_type == "broadcast")
        {
            double broadcast_factor = 1;
            int    root             = get_demo_test_root();
            for (uint64_t i = 0; i < count; ++i)
            {
                float temp = i + demo_data.hccl_rank;
                if (bf16_convert) input_host_data[i] = floatToBf16(temp);
                else
                    input_host_data[i] = temp;
            }
            if (data_csv_enabled)
            {
                copy_data_to_csv_report(input_host_data, count, data_csv_path_input, bf16_convert);
            }

            // Copy from input_host_data_ptr to input_dev_ptr (to be used in benchmark)
            CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.host_to_device_stream,
                                                 (uint64_t) input_host_data_ptr,
                                                 data_size,
                                                 input_dev_ptr,
                                                 HOST_TO_DRAM));
            CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.host_to_device_stream));
            // Run HCCL Broadcast collective
            auto stat = benchmark(
                demo_data,
                [&](uint64_t iter) {
                    uint64_t index = iter % number_of_buffers;
                    CHECK_HCCL_STATUS(hcclBroadcast((const void*) input_dev_ptrs[index],
                                                    (void*) output_dev_ptrs[index],
                                                    input_host_data.size(),
                                                    get_demo_hccl_data_type(),
                                                    root,
                                                    demo_data.hccl_comm,
                                                    demo_data.collective_stream));
                },
                [&]() {
                    CHECK_HCCL_STATUS(hcclBroadcast((const void*) input_dev_ptrs[0],
                                                    (void*) correctness_dev_ptr,
                                                    input_host_data.size(),
                                                    get_demo_hccl_data_type(),
                                                    root,
                                                    demo_data.hccl_comm,
                                                    demo_data.collective_stream));
                });
            // Correctness check
            auto        output_host_data     = vector<T>(input_host_data.size());
            const void* output_host_data_ptr = reinterpret_cast<void*>(output_host_data.data());

            CHECK_SYNAPSE_STATUS(synHostMap(demo_data.device_handle, data_size, output_host_data_ptr));
            CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.device_to_host_stream,
                                                 correctness_dev_ptr,
                                                 data_size,
                                                 (uint64_t) output_host_data_ptr,
                                                 DRAM_TO_HOST));
            CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.device_to_host_stream));
            size_t loop_size = output_size / data_type_size;
            if (check_correctness())
            {
                if (bf16_convert)
                {
                    for (size_t i = 0; i < loop_size; ++i)
                    {
                        is_ok = correctness_check_function(demo_data, i, bf16ToFloat(output_host_data[i]), i);
                    }
                }
                else
                {
                    for (size_t i = 0; i < loop_size; ++i)
                    {
                        if (abs(output_host_data[i] - (float) (i + root)) != 0)
                        {
                            is_ok = false;
                        }
                    }
                }
                log() << "Broadcast hccl_rank=" << demo_data.hccl_rank << " size=" << data_size << " <"
                      << demo_data.str_data_type << ">"
                      << " Input Buffer [" << get_float(input_host_data[0]) << " " << get_float(input_host_data[1])
                      << " " << get_float(input_host_data[2]) << " " << get_float(input_host_data[3]) << " ...]"
                      << " reduced to Output Buffer [" << get_float(output_host_data[0]) << " "
                      << get_float(output_host_data[1]) << " " << get_float(output_host_data[2]) << " "
                      << get_float(output_host_data[3]) << " ...]"
                      << " which is " << (is_ok ? "fine." : "bad.") << endl;
            }
            if (data_csv_enabled)
            {
                copy_data_to_csv_report(output_host_data, loop_size, data_csv_path_output, bf16_convert);
            }
            CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, input_host_data_ptr));
            CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, output_host_data_ptr));

            // End of correctness check
            describe_stat("Broadcast(data_size=" + to_string(data_size) +
                              ", count=" + to_string(input_host_data.size()) + ", dtype=" + demo_data.str_data_type +
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
                float temp = demo_data.hccl_rank + (demo_data.nranks * (i % DATA_ELEMENTS_MAX));
                if (bf16_convert) input_host_data[i] = floatToBf16(temp);
                else
                    input_host_data[i] = temp;
            }
            if (data_csv_enabled)
            {
                copy_data_to_csv_report(input_host_data, count, data_csv_path_input, bf16_convert);
            }

            //Copy from input_host_data_ptr to input_dev_ptr (to be used in benchmark)
            CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.host_to_device_stream,
                                                 (uint64_t) input_host_data_ptr,
                                                 data_size,
                                                 input_dev_ptr,
                                                 HOST_TO_DRAM));
            CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.host_to_device_stream));
            // Run HCCL AllReduce collective
            auto stat = benchmark(
                demo_data,
                [&](uint64_t iter) {
                    uint64_t index = iter % number_of_buffers;
                    CHECK_HCCL_STATUS(hcclAllReduce((const void*) input_dev_ptrs[index],
                                                    (void*) output_dev_ptrs[index],
                                                    input_host_data.size(),
                                                    get_demo_hccl_data_type(),
                                                    get_reduction_op(),
                                                    demo_data.hccl_comm,
                                                    demo_data.collective_stream));
                },
                [&]() {
                    CHECK_HCCL_STATUS(hcclAllReduce((const void*) input_dev_ptrs[0],
                                                    (void*) correctness_dev_ptr,
                                                    input_host_data.size(),
                                                    get_demo_hccl_data_type(),
                                                    get_reduction_op(),
                                                    demo_data.hccl_comm,
                                                    demo_data.collective_stream));
                });

            // Correctness check
            auto        output_host_data     = vector<T>(input_host_data.size());
            const void* output_host_data_ptr = reinterpret_cast<void*>(output_host_data.data());

            CHECK_SYNAPSE_STATUS(synHostMap(demo_data.device_handle, data_size, output_host_data_ptr));
            CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.device_to_host_stream,
                                                 correctness_dev_ptr,
                                                 data_size,
                                                 (uint64_t) output_host_data_ptr,
                                                 DRAM_TO_HOST));
            CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.device_to_host_stream));

            size_t loop_size = output_size / data_type_size;
            if (check_correctness())
            {
                for (size_t i = 0; i < loop_size; ++i)
                {
                    vector<T> vec(demo_data.nranks);
                    size_t    addCommSize = demo_data.nranks * (i % DATA_ELEMENTS_MAX);
                    // Arithmetic progression
                    for (size_t rank = 0; rank < demo_data.nranks; ++rank)
                    {
                        vec[rank] = rank + addCommSize;
                    }
                    T expected = calc_expected_reduction<T>(vec, get_reduction_op());

                    if (bf16_convert)
                    {
                        is_ok = correctness_check_function(demo_data, expected, bf16ToFloat(output_host_data[i]), i);
                    }
                    else
                    {
                        if (std::abs((float) output_host_data[i] - expected) != 0)
                        {
                            is_ok = false;
                        }
                    }
                }
                log() << "Allreduce hccl_rank=" << demo_data.hccl_rank << " size=" << data_size << " <"
                      << demo_data.str_data_type << ">"
                      << " Input Buffer [" << get_float(input_host_data[0]) << " " << get_float(input_host_data[1])
                      << " " << get_float(input_host_data[2]) << " " << get_float(input_host_data[3]) << " ...]"
                      << " reduced to Output Buffer [" << get_float(output_host_data[0]) << " "
                      << get_float(output_host_data[1]) << " " << get_float(output_host_data[2]) << " "
                      << get_float(output_host_data[3]) << " ...]"
                      << " which is " << (is_ok ? "fine." : "bad.") << endl;
            }
            if (data_csv_enabled)
            {
                copy_data_to_csv_report(output_host_data, loop_size, data_csv_path_output, bf16_convert);
            }
            CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, input_host_data_ptr));
            CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, output_host_data_ptr));

            // End of correctness check
            describe_stat("hcclAllReduce(src!=dst, data_size=" + to_string(data_size) +
                              ", count=" + to_string(input_host_data.size()) + ", dtype=" + demo_data.str_data_type +
                              ", iterations=" + to_string(demo_data.num_iters) + ")",
                          stat,
                          data_size,
                          allreduce_factor,
                          demo_data.hccl_rank,
                          demo_data.num_iters,
                          test_type,
                          demo_data.str_data_type,
                          get_reduction_op_str(),
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
                float temp = demo_data.hccl_rank + (demo_data.nranks * (i % DATA_ELEMENTS_MAX));
                if (bf16_convert) input_host_data[i] = floatToBf16(temp);
                else
                    input_host_data[i] = temp;
            }
            if (data_csv_enabled)
            {
                copy_data_to_csv_report(input_host_data, count, data_csv_path_input, bf16_convert);
            }

            //Copy from input_host_data_ptr to input_dev_ptr (to be used in benchmark)
            CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.host_to_device_stream,
                                                 (uint64_t) input_host_data_ptr,
                                                 data_size,
                                                 input_dev_ptr,
                                                 HOST_TO_DRAM));
            CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.host_to_device_stream));
            // Run HCCL ReduceScatter collective
            auto stat = benchmark(
                demo_data,
                [&](uint64_t iter) {
                    uint64_t index = iter % number_of_buffers;
                    CHECK_HCCL_STATUS(hcclReduceScatter((const void*) input_dev_ptrs[index],
                                                        (void*) output_dev_ptrs[index],
                                                        input_host_data.size() / demo_data.nranks,
                                                        get_demo_hccl_data_type(),
                                                        get_reduction_op(),
                                                        demo_data.hccl_comm,
                                                        demo_data.collective_stream));
                },
                [&]() {
                    CHECK_HCCL_STATUS(hcclReduceScatter((const void*) input_dev_ptrs[0],
                                                        (void*) correctness_dev_ptr,
                                                        input_host_data.size() / demo_data.nranks,
                                                        get_demo_hccl_data_type(),
                                                        get_reduction_op(),
                                                        demo_data.hccl_comm,
                                                        demo_data.collective_stream));
                });
            // Correctness check
            auto        output_host_data     = vector<T>(input_host_data.size());
            const void* output_host_data_ptr = reinterpret_cast<void*>(output_host_data.data());

            CHECK_SYNAPSE_STATUS(
                synHostMap(demo_data.device_handle, data_size / demo_data.nranks, output_host_data_ptr));
            CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.device_to_host_stream,
                                                 correctness_dev_ptr,
                                                 data_size / demo_data.nranks,
                                                 (uint64_t) output_host_data_ptr,
                                                 DRAM_TO_HOST));
            CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.device_to_host_stream));

            size_t data_size_loop = output_host_data.size() / demo_data.nranks;
            if (check_correctness())
            {
                size_t start = demo_data.hccl_rank * data_size_loop;
                for (size_t i = 0; i < data_size_loop; ++i)
                {
                    vector<T> vec(demo_data.nranks);
                    size_t    addCommSize = demo_data.nranks * ((start + i) % DATA_ELEMENTS_MAX);
                    // Arithmetic progression
                    for (size_t rank = 0; rank < demo_data.nranks; ++rank)
                    {
                        vec[rank] = rank + addCommSize;
                    }
                    T expected = calc_expected_reduction<T>(vec, get_reduction_op());

                    if (bf16_convert)
                    {
                        is_ok = correctness_check_function(demo_data, expected, bf16ToFloat(output_host_data[i]), i);
                    }
                    else
                    {
                        if (std::abs((float) output_host_data[i] - expected) != 0)
                        {
                            is_ok = false;
                        }
                    }
                }
                log() << "ReduceScatter hccl_rank=" << demo_data.hccl_rank << " size=" << data_size << " <"
                      << demo_data.str_data_type << ">"
                      << " Input Buffer [" << get_float(input_host_data[0]) << " " << get_float(input_host_data[1])
                      << " " << get_float(input_host_data[2]) << " " << get_float(input_host_data[3]) << " ...]"
                      << " reduced to Output Buffer [" << get_float(output_host_data[0]) << " "
                      << get_float(output_host_data[1]) << " " << get_float(output_host_data[2]) << " "
                      << get_float(output_host_data[3]) << " ...]"
                      << " which is " << (is_ok ? "fine." : "bad.") << endl;
            }
            if (data_csv_enabled)
            {
                copy_data_to_csv_report(output_host_data, data_size_loop, data_csv_path_output, bf16_convert);
            }
            CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, input_host_data_ptr));
            CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, output_host_data_ptr));

            // End of correctness

            describe_stat("hcclReduceScatter(src!=dst, data_size=" + to_string(data_size) +
                              ", count=" + to_string(input_host_data.size()) + ", dtype=" + demo_data.str_data_type +
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
                float temp = demo_data.hccl_rank * count + i;
                if (bf16_convert) input_host_data[i] = floatToBf16(temp);
                else
                    input_host_data[i] = temp;
            }
            if (data_csv_enabled)
            {
                copy_data_to_csv_report(input_host_data, count, data_csv_path_input, bf16_convert);
            }

            //Copy from input_host_data_ptr to input_dev_ptr (to be used in benchmark)
            CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.host_to_device_stream,
                                                 (uint64_t) input_host_data_ptr,
                                                 data_size,
                                                 input_dev_ptr,
                                                 HOST_TO_DRAM));
            CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.host_to_device_stream));
            // Run HCCL AllGather collective
            auto stat = benchmark(
                demo_data,
                [&](uint64_t iter) {
                    uint64_t index = iter % number_of_buffers;
                    CHECK_HCCL_STATUS(hcclAllGather((const void*) input_dev_ptrs[index],
                                                    (void*) output_dev_ptrs[index],
                                                    input_host_data.size(),
                                                    get_demo_hccl_data_type(),
                                                    demo_data.hccl_comm,
                                                    demo_data.collective_stream));
                },
                [&]() {
                    CHECK_HCCL_STATUS(hcclAllGather((const void*) input_dev_ptrs[0],
                                                    (void*) correctness_dev_ptr,
                                                    input_host_data.size(),
                                                    get_demo_hccl_data_type(),
                                                    demo_data.hccl_comm,
                                                    demo_data.collective_stream));
                });
            // Correctness check
            size_t output_count = output_size / data_type_size;
            auto        output_host_data     = vector<T>(output_count,0);
            const void* output_host_data_ptr = reinterpret_cast<void*>(output_host_data.data());
            CHECK_SYNAPSE_STATUS(synHostMap(demo_data.device_handle, output_size, output_host_data_ptr));
            CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.device_to_host_stream,
                                                 correctness_dev_ptr,
                                                 output_size,
                                                 (uint64_t) output_host_data_ptr,
                                                 DRAM_TO_HOST));
            CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.device_to_host_stream));
            if (check_correctness())
            {
                if (bf16_convert)
                {
                    for (size_t i = 0; i < output_count; ++i)
                    {
                        is_ok = correctness_check_function(demo_data, i, bf16ToFloat(output_host_data[i]), i);
                    }
                }
                else
                {
                    for (size_t i = 0; i < output_count; ++i)
                    {
                        if (output_host_data[i] != i)
                        {
                            is_ok = false;
                        }
                    }
                }
                log() << "AllGather hccl_rank=" << demo_data.hccl_rank << " size=" << data_size << " <"
                      << demo_data.str_data_type << ">"
                      << " Input Buffer [" << get_float(input_host_data[0]) << " " << get_float(input_host_data[1])
                      << " " << get_float(input_host_data[2]) << " " << get_float(input_host_data[3]) << " ...]"
                      << " reduced to Output Buffer [" << get_float(output_host_data[0]) << " "
                      << get_float(output_host_data[1]) << " " << get_float(output_host_data[2]) << " "
                      << get_float(output_host_data[3]) << " ...]"
                      << " which is " << (is_ok ? "fine." : "bad.") << endl;
            }
            if (data_csv_enabled)
            {
                copy_data_to_csv_report(output_host_data, output_count, data_csv_path_output, bf16_convert);
            }
            CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, input_host_data_ptr));
            CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, output_host_data_ptr));
            // End of correctness check

            describe_stat("hcclAllGather(src!=dst, data_size=" + to_string(data_size) +
                              ", count=" + to_string(input_host_data.size()) + ", dtype=" + demo_data.str_data_type +
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
                    int   val  = demo_data.hccl_rank * chunkSize + j + demo_data.nranks * i;
                    float temp = (val % DATA_ELEMENTS_MAX);
                    if (bf16_convert) input_host_data[i * chunkSize + j] = floatToBf16(temp);
                    else
                        input_host_data[i * chunkSize + j] = temp;
                }
            }
            if (data_csv_enabled)
            {
                copy_data_to_csv_report(input_host_data, count, data_csv_path_input, bf16_convert);
            }

            // Copy from input_host_data_ptr to input_dev_ptr (to be used in benchmark)
            CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.host_to_device_stream,
                                                 (uint64_t) input_host_data_ptr,
                                                 data_size,
                                                 input_dev_ptr,
                                                 HOST_TO_DRAM));
            CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.host_to_device_stream));

            // Run HCCL AlltoAll collective
            auto stat = benchmark(
                demo_data,
                [&](uint64_t iter) {
                    uint64_t index = iter % number_of_buffers;
                    CHECK_HCCL_STATUS(hcclAlltoAll((const void*) input_dev_ptrs[index],
                                                   (void*) output_dev_ptrs[index],
                                                   input_host_data.size(),
                                                   get_demo_hccl_data_type(),
                                                   demo_data.hccl_comm,
                                                   demo_data.collective_stream));
                },
                [&]() {
                    CHECK_HCCL_STATUS(hcclAlltoAll((const void*) input_dev_ptrs[0],
                                                   (void*) correctness_dev_ptr,
                                                   input_host_data.size(),
                                                   get_demo_hccl_data_type(),
                                                   demo_data.hccl_comm,
                                                   demo_data.collective_stream));
                });

            // Correctness check
            auto        output_host_data     = vector<T>(input_host_data.size());
            const void* output_host_data_ptr = reinterpret_cast<void*>(output_host_data.data());

            CHECK_SYNAPSE_STATUS(synHostMap(demo_data.device_handle, output_size, output_host_data_ptr));
            CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.device_to_host_stream,
                                                 correctness_dev_ptr,
                                                 output_size,
                                                 (uint64_t) output_host_data_ptr,
                                                 DRAM_TO_HOST));
            CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.device_to_host_stream));

            size_t loop_size = output_size / data_type_size;
            if (check_correctness())
            {
                int    start = demo_data.hccl_rank * (count / chunkSize);
                int    expected;
                if (bf16_convert)
                {
                    for (size_t i = 0; i < loop_size; ++i)
                    {
                        expected = ((start + i) % DATA_ELEMENTS_MAX);
                        is_ok    = correctness_check_function(demo_data, expected, bf16ToFloat(output_host_data[i]), i);
                    }
                }
                else
                {
                    for (size_t i = 0; i < loop_size; ++i)
                    {
                        expected = ((start + i) % DATA_ELEMENTS_MAX);
                        if ((float) output_host_data[i] != (float) expected)
                        {
                            is_ok = false;
                        }
                    }
                }
                log() << "All2All hccl_rank=" << demo_data.hccl_rank << " size=" << data_size << " <"
                      << demo_data.str_data_type << ">"
                      << " Input Buffer [" << get_float(input_host_data[0]) << " " << get_float(input_host_data[1])
                      << " " << get_float(input_host_data[2]) << " " << get_float(input_host_data[3]) << " ...]"
                      << " Output Buffer [" << get_float(output_host_data[0]) << " " << get_float(output_host_data[1])
                      << " " << get_float(output_host_data[2]) << " " << get_float(output_host_data[3]) << " ...]"
                      << " which is " << (is_ok ? "fine." : "bad.") << endl;
            }
            if (data_csv_enabled)
            {
                copy_data_to_csv_report(output_host_data, loop_size, data_csv_path_output, bf16_convert);
            }
            CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, input_host_data_ptr));
            CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, output_host_data_ptr));

            // End of correctness check

            describe_stat("hcclAlltoAll(src!=dst, data_size=" + to_string(data_size) +
                              ", count=" + to_string(input_host_data.size()) + ", dtype=" + demo_data.str_data_type +
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
            for (uint64_t i = 0; i < count; ++i)
            {
                float temp = demo_data.hccl_rank + 1;
                if (bf16_convert) input_host_data[i] = floatToBf16(temp);
                else
                    input_host_data[i] = temp;
            }
            if (data_csv_enabled)
            {
                copy_data_to_csv_report(input_host_data, count, data_csv_path_input, bf16_convert);
            }
            CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.host_to_device_stream,
                                             (uint64_t) input_host_data_ptr,
                                             data_size,
                                             input_dev_ptr,
                                             HOST_TO_DRAM));
            if (demo_data.ranks_list.length() > 0)
            {
                if (demo_data.hccl_rank == get_demo_test_root())
                {
                    log() << "Will perform ranks send_recv test with list: " << get_ranks_list() << std::endl;
                }
                is_ok = send_recv_ranks_test_driver(demo_data,
                                                    test_type,
                                                    data_size,
                                                    count,
                                                    input_host_data,
                                                    input_dev_ptrs,
                                                    output_dev_ptrs,
                                                    bf16_convert);
            }
            else
            {
                is_ok = send_recv_test_driver(demo_data,
                                              test_type,
                                              demo_data.hccl_rank,
                                              data_size,
                                              count,
                                              input_host_data,
                                              input_dev_ptrs,
                                              output_dev_ptrs,
                                              correctness_dev_ptr,
                                              bf16_convert,
                                              data_csv_enabled,
                                              data_csv_path_output);
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
                float temp = demo_data.hccl_rank + (demo_data.nranks * (i % DATA_ELEMENTS_MAX));
                if (bf16_convert) input_host_data[i] = floatToBf16(temp);
                else
                    input_host_data[i] = temp;
            }
            if (data_csv_enabled)
            {
                copy_data_to_csv_report(input_host_data, count, data_csv_path_input, bf16_convert);
            }
            // Copy from input_host_data_ptr to input_dev_ptr (to be used in benchmark)
            CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.host_to_device_stream,
                                                 (uint64_t) input_host_data_ptr,
                                                 data_size,
                                                 input_dev_ptr,
                                                 HOST_TO_DRAM));

            CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.host_to_device_stream));
            // Run HCCL Reduce collective

            auto stat = benchmark(
                demo_data,
                [&](uint64_t iter) {
                    uint64_t index = iter % number_of_buffers;
                    CHECK_HCCL_STATUS(hcclReduce((const void*) input_dev_ptrs[index],
                                                 (void*) output_dev_ptrs[index],
                                                 input_host_data.size(),
                                                 get_demo_hccl_data_type(),
                                                 get_reduction_op(),
                                                 root,
                                                 demo_data.hccl_comm,
                                                 demo_data.collective_stream));
                },
                [&]() {
                    CHECK_HCCL_STATUS(hcclReduce((const void*) input_dev_ptrs[0],
                                                 (void*) correctness_dev_ptr,
                                                 input_host_data.size(),
                                                 get_demo_hccl_data_type(),
                                                 get_reduction_op(),
                                                 root,
                                                 demo_data.hccl_comm,
                                                 demo_data.collective_stream));
                });

            // Correctness check
            auto        output_host_data     = vector<T>(input_host_data.size());
            const void* output_host_data_ptr = reinterpret_cast<void*>(output_host_data.data());
            size_t      loop_size            = output_size / data_type_size;
            // The correctness check is relevant for the root's output buffer only
            if (check_correctness())
            {
                log() << "Reduce hccl_rank=" << demo_data.hccl_rank << " size=" << data_size << " <"
                      << demo_data.str_data_type << ">"
                      << " Input Buffer [" << get_float(input_host_data[0]) << " " << get_float(input_host_data[1])
                      << " " << get_float(input_host_data[2]) << " " << get_float(input_host_data[3]) << " ...]"
                      << endl;
                if (demo_data.hccl_rank == root)
                {
                    CHECK_SYNAPSE_STATUS(synHostMap(demo_data.device_handle, data_size, output_host_data_ptr));
                    CHECK_SYNAPSE_STATUS(synMemCopyAsync(demo_data.device_to_host_stream,
                                                         correctness_dev_ptr,
                                                         data_size,
                                                         (uint64_t) output_host_data_ptr,
                                                         DRAM_TO_HOST));
                    CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.device_to_host_stream));

                    for (size_t i = 0; i < loop_size; ++i)
                    {
                        vector<T> vec(demo_data.nranks);
                        size_t addCommSize = demo_data.nranks * (i % DATA_ELEMENTS_MAX);
                        // Arithmetic progression
                        for (size_t rank = 0; rank < demo_data.nranks; ++rank)
                        {
                            vec[rank] = rank + addCommSize;
                        }
                        T expected = calc_expected_reduction<T>(vec, get_reduction_op());

                        if (bf16_convert)
                        {
                            is_ok =
                                correctness_check_function(demo_data, expected, bf16ToFloat(output_host_data[i]), i);
                        }
                        else
                        {
                            if (std::abs((float) output_host_data[i] - expected) != 0)
                            {
                                is_ok = false;
                            }
                        }
                    }
                }
                log() << " Output Buffer [" << get_float(output_host_data[0]) << " " << get_float(output_host_data[1]) << " "
                      << get_float(output_host_data[2]) << " " << get_float(output_host_data[3]) << " ...]"
                      << " which is " << (is_ok ? "fine." : "bad.") << endl;
            }
            if (demo_data.hccl_rank == root)
            {
                if (data_csv_enabled)
                {
                    copy_data_to_csv_report(output_host_data, loop_size, data_csv_path_output, bf16_convert);
                }
            }
            CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, input_host_data_ptr));
            CHECK_SYNAPSE_STATUS(synHostUnmap(demo_data.device_handle, output_host_data_ptr));

            // End of correctness check
            describe_stat("Reduce(count=" + std::to_string(input_host_data.size()) +
                              "data_size=" + to_string(data_size) + ", count=" + std::to_string(count) + ", dtype=" +
                              demo_data.str_data_type + ", iterations=" + std::to_string(demo_data.num_iters) + ")",
                          stat,
                          data_size,
                          reduce_factor,
                          demo_data.hccl_rank,
                          demo_data.num_iters,
                          test_type,
                          demo_data.str_data_type,
                          get_reduction_op_str(),
                          is_root_rank);
        }
        else if (test_type == "scale_validation")
        {
#ifdef MPI_ENABLED
            // since there is no correctness check here the data can be random (no need to initialize)
            scale_test_driver(demo_data, count, data_size, input_dev_ptr, output_dev_ptr);
#else
            throw std::runtime_error {"MPI must be enabled for scale validation test"};
#endif  //MPI_ENABLED
        }
        else
        {
            log() << "Unknown test type (" << test_type << ")" << endl;
            return -1;
        }

        CHECK_SYNAPSE_STATUS(synStreamSynchronize(demo_data.collective_stream));

        CHECK_SYNAPSE_STATUS(synDeviceFree(demo_data.device_handle, input_dev_ptr, 0));
        CHECK_SYNAPSE_STATUS(synDeviceFree(demo_data.device_handle, output_dev_ptr, 0));
        CHECK_SYNAPSE_STATUS(synDeviceFree(demo_data.device_handle, correctness_dev_ptr, 0));
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
    if (!is_ok)
    {
        throw runtime_error {"Collective operation has failed on correctness."};
    }
    return 0;
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
        bool bf16_convert        = (demo_data.hccl_data_type == hcclBfloat16);

        if (buildCustomComm(&demo_data) == false)
        {
            log() << "HCCL demo process id (" << demo_data.hccl_rank
                  << ") will not participate in the custom communicator" << endl;
#if MPI_ENABLED
            hcclUniqueId unique_id {};
            CHECK_MPI_STATUS(MPI_Bcast(&unique_id, sizeof(unique_id), MPI_BYTE, demo_data.mpi_root_rank, MPI_COMM_WORLD));
            CHECK_MPI_STATUS(MPI_Finalize());
#endif
            return 0;
        }

        // Initialize Synapse API context
        CHECK_SYNAPSE_STATUS(synInitialize());
        // Acquire device
        synModuleId device_module_id = get_hccl_rank() % get_demo_ranks_per_node();
        synStatus   rc               = synDeviceAcquireByModuleId(&demo_data.device_handle, device_module_id);
        if (rc != synSuccess)
        {
            device_module_id = INVALID_MODULE_ID;
            CHECK_SYNAPSE_STATUS(synDeviceAcquire(&demo_data.device_handle, nullptr));
        }

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

        if (bf16_convert)
        {
            return run_test<uint16_t>(demo_data,  bf16_convert, device_module_id);
        }
        else
        {
            return run_test<float>(demo_data, bf16_convert, device_module_id);
        }
    }
    catch (const exception& ex)
    {
        log() << "HCCL demo error: " << ex.what() << endl;
        return -1;
    }
    return 0;
}
