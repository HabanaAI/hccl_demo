/******************************************************************************
# Copyright (c) 2022 Habana Labs, Ltd.
# SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "common.h"

#include <algorithm>  // for sort
#include <unistd.h>   // for linux specifics (getenv)

// Constants
static constexpr HCL_Rank DEFAULT_ROOT_RANK      = 0;
static constexpr char     DEFAULT_TEST_TYPE[]    = "broadcast";
static constexpr char     DEFAULT_MEASURE_TYPE[] = "bw";
static constexpr char     DEFAULT_DATA_TYPE[]    = "float";

static constexpr uint64_t DEFAULT_TEST_SIZE           = 33554432;
static constexpr uint64_t DEFAULT_TEST_SIZE_RANGE_MIN = 0;
static constexpr uint64_t DEFAULT_TEST_SIZE_RANGE_MAX = 0;
static constexpr uint64_t DEFAULT_TEST_SIZE_RANGE_INC = 1;

static constexpr char   DEFAULT_REDOP[]           = "sum";
static constexpr size_t DEFAULT_TEST_LOOP         = 1000;
static constexpr bool   DEFAULT_CHECK_CORRECTNESS = true;
static constexpr bool   DEFAULT_SAME_BUFFERS      = false;

static constexpr size_t DEFAULT_BOX_SIZE = 8;
static constexpr size_t DEFAULT_NRANKS   = 0;

static std::string getEnvOrDefaultValue(const char* envName, std::string defaultValue)
{
    char* envValue = getenv(envName);
    return (envValue != nullptr) ? std::string(envValue) : defaultValue;
}

static uint64_t getEnvOrDefaultValue(const char* envName, uint64_t defaultValue)
{
    const char* envValue = getenv(envName);
    return (envValue != nullptr) ? strtoull(envValue, NULL, 0) : defaultValue;
}

static void checkRankValue(HCL_Rank rank, size_t nranks)
{
    if (rank >= nranks)
    {
        throw std::runtime_error {"Invalid rank number " + std::to_string(rank) + ", ranks can be in range [0," +
                                  std::to_string(nranks - 1) + "] in custom_comm"};
    }
}

static void parseCustomComm(std::string rankList, std::vector<HCL_Rank>& parsedRankList, size_t nranks)
{
    std::string delimiter = ",";
    size_t      pos       = 0;

    while ((pos = rankList.find(delimiter)) != std::string::npos)
    {
        parsedRankList.push_back(stoi(rankList.substr(0, pos)));
        checkRankValue(parsedRankList.back(), nranks);
        rankList.erase(0, pos + delimiter.length());
    }
    if (!rankList.empty())
    {
        parsedRankList.push_back(stoi(rankList));
        checkRankValue(parsedRankList.back(), nranks);
    }
    sort(parsedRankList.begin(), parsedRankList.end());
    return;
}

EnvData getenvData()
{
    EnvData envData;
    envData.root        = getEnvOrDefaultValue("HCCL_DEMO_TEST_ROOT", DEFAULT_ROOT_RANK);
    envData.testType    = getEnvOrDefaultValue("HCCL_DEMO_TEST", DEFAULT_TEST_TYPE);
    envData.measureType = getEnvOrDefaultValue("HCCL_DEMO_MEASURE", DEFAULT_MEASURE_TYPE);
    envData.dataType    = getEnvOrDefaultValue("HCCL_DATA_TYPE", DEFAULT_DATA_TYPE);

    uint64_t size    = getEnvOrDefaultValue("HCCL_DEMO_TEST_SIZE", DEFAULT_TEST_SIZE);
    uint64_t sizeMin = getEnvOrDefaultValue("HCCL_SIZE_RANGE_MIN", DEFAULT_TEST_SIZE_RANGE_MIN);
    uint64_t sizeMax = getEnvOrDefaultValue("HCCL_SIZE_RANGE_MAX", DEFAULT_TEST_SIZE_RANGE_MAX);
    envData.sizeInc  = getEnvOrDefaultValue("HCCL_SIZE_RANGE_INC", DEFAULT_TEST_SIZE_RANGE_INC);
    envData.sizeMin  = sizeMin > 0 ? sizeMin : size;
    envData.sizeMax  = sizeMax > 0 ? sizeMax : size;

    envData.redop                  = getEnvOrDefaultValue("HCCL_REDUCTION_OP", DEFAULT_REDOP);
    envData.numIters               = getEnvOrDefaultValue("HCCL_DEMO_TEST_LOOP", DEFAULT_TEST_LOOP);
    envData.useSameBuffers         = getEnvOrDefaultValue("HCCL_DEMO_SAME_BUFFERS", DEFAULT_SAME_BUFFERS);
    envData.shouldCheckCorrectness = getEnvOrDefaultValue("HCCL_DEMO_CHECK_CORRECTNESS", DEFAULT_CHECK_CORRECTNESS);
    envData.dataCSVPath            = getEnvOrDefaultValue("HCCL_DEMO_DATA_CSV", "");
    envData.resultsCSVPath         = getEnvOrDefaultValue("HCCL_DEMO_RESULT_CSV", "");

    envData.ranksList          = getEnvOrDefaultValue("HCCL_RANKS_LIST", "");
    envData.expectedScaleoutBW = getEnvOrDefaultValue("HCCL_EXPECTED_SCALEOUT_BW", 0);

    bool mpiEnabled = false;
#if MPI_ENABLED
    mpiEnabled  = true;
    int mpiRank = HCL_INVALID_RANK;
    int mpiSize = 0;
    CHECK_MPI_STATUS(MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank));
    CHECK_MPI_STATUS(MPI_Comm_size(MPI_COMM_WORLD, &mpiSize));
    envData.rank         = mpiRank;
    envData.nranks       = mpiSize;
    envData.ranksPerNode = getEnvOrDefaultValue("OMPI_COMM_WORLD_LOCAL_SIZE", DEFAULT_BOX_SIZE);
#else
    envData.rank         = getEnvOrDefaultValue("HCCL_RANK", HCL_INVALID_RANK);
    envData.nranks       = getEnvOrDefaultValue("HCCL_NRANKS", DEFAULT_NRANKS);
    envData.ranksPerNode = getEnvOrDefaultValue("HCCL_RANKS_PER_NODE", DEFAULT_BOX_SIZE);
#endif  // MPI_ENABLED

    // verify MPI configuration
    if (getEnvOrDefaultValue("HCCL_DEMO_MPI_REQUESTED", false) != mpiEnabled)
    {
        throw std::runtime_error {
            "HCCL demo compilation and user instruction regarding run type (MPI/pure) are non compatible. \nPlease "
            "consider building the demo with the correct instructions or run with -clean"};
    }

    envData.scaleupGroupSize = getEnvOrDefaultValue("HCCL_SCALEUP_GROUP_SIZE", envData.ranksPerNode);

    std::string customCommStr = getEnvOrDefaultValue("HCCL_DEMO_CUSTOM_COMM", "");
    parseCustomComm(customCommStr, envData.customComm, envData.nranks);

    return envData;
}
