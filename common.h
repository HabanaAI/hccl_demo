/******************************************************************************
# Copyright (c) 2022 Habana Labs, Ltd.
# SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#pragma once

// C++ Standard Libraries
#include <cstdint>        // for uint types
#include <cstring>        // for std::memcopy
#include <functional>     //for std::function
#include <iostream>       // for io stream
#include <sstream>        // for std::stringstream
#include <iomanip>        // for std::setprecision, std::put_time
#include <string>         // for std::string
#include <vector>         // for std::vector
#include <unordered_map>  // for std::unordered_map

#include "hcl_inc.h"  // for HCL_Rank

// HCCL :: Habana Collective Communications Library
#include <hccl.h>

// Synapse :: Habana Synapse training API
#include <synapse_api.h>

#if AFFINITY_ENABLED
#include "affinity.h"
#endif

// MPI handling
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

// Error handling
#define CHECK_HCCL_STATUS(x)                                                                                           \
    {                                                                                                                  \
        const auto _res = (x);                                                                                         \
        if (_res != hcclSuccess)                                                                                       \
            throw std::runtime_error {"In function " + std::string {__FUNCTION__} +                                    \
                                      "(): " #x " failed: " + hcclGetErrorString(_res)};                               \
    }

#define CHECK_SYNAPSE_STATUS(x)                                                                                        \
    {                                                                                                                  \
        const auto _res = (x);                                                                                         \
        if (_res != synSuccess)                                                                                        \
            throw std::runtime_error {"In function " + std::string {__FUNCTION__} +                                    \
                                      "(): " #x " failed with synapse error: " + std::to_string((_res))};              \
    }

#define ASSERT(x)                                                                                                      \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(x)) throw std::runtime_error {"In function " + std::string {__FUNCTION__} + " assertion failed"};        \
    } while (false)

// Constants
static constexpr int      DATA_ELEMENTS_MAX    = 13;
static constexpr uint64_t ALLOCATED_HBM_SIZE   = (2UL * 1024 * 1024 * 1024);  // 2GB
static constexpr uint64_t AMOUNT_JUMBO_BUFFERS = (2);
static constexpr uint64_t MAX_BUFFER_COUNT     = (33UL);

// demo structures
struct EnvData
{
    HCL_Rank              root;
    std::string           testType;
    std::string           dataType;
    uint64_t              sizeMin;
    uint64_t              sizeMax;
    uint64_t              sizeInc;
    std::string           redop;
    size_t                numIters;
    bool                  shouldCheckCorrectness;
    std::string           dataCSVPath;
    std::string           resultsCSVPath;
    std::string           ranksList;
    uint64_t              expectedScaleoutBW;
    HCL_Rank              rank;
    size_t                nranks;
    size_t                ranksPerNode;
    size_t                scaleupGroupSize;
    std::vector<HCL_Rank> customComm;
};

struct DeviceResources
{
    synDeviceId     deviceHandle;
    hcclComm_t      comm;
    HCL_Rank        commRoot;
    synStreamHandle collectiveStream;
    synStreamHandle deviceToHostStream;
    synStreamHandle hostToDeviceStream;
};

struct Buffers
{
    uint64_t              inputSize;
    uint64_t              outputSize;
    std::vector<uint64_t> inputDevPtrs;
    std::vector<uint64_t> outputDevPtrs;
    uint64_t              correctnessDevPtr;
};

struct Stats
{
    bool               isDescribing = false;
    std::string        statName;
    double             factor;
    double             rankDurationInSec;
    std::vector<float> expectedOutputs;
};

struct ReportEntry
{
    uint64_t size;
    uint64_t count;
    double   time;
    double   algoBW;
    double   avgBW;
};

struct RanksPairSendRecv
{
    HCL_Rank sendFromRank;
    HCL_Rank recvInRank;
};

inline std::ostream& log()
{
    return std::cout;
}

inline bool isBfloat16(const EnvData& envData)
{
    return envData.dataType == "bfloat16";
}

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

inline float bf16AccuracyCoefficient(size_t numberOfRanks)
{
    numberOfRanks = (numberOfRanks > 8) ? 8 : numberOfRanks;
    return (numberOfRanks > 1) ? (float) numberOfRanks / 256.0 : 0.0;  // For 1 rank, tolerance should be 0
}

inline bool isRoot(const EnvData& envData)
{
    return envData.rank == envData.root;
}

inline hcclDataType_t getDataType(const EnvData& envData)
{
    static const std::unordered_map<std::string, hcclDataType_t> dataTypeMap = {
        {"float", hcclFloat32},
        {"bfloat16", hcclBfloat16},
    };

    auto it = dataTypeMap.find(envData.dataType);
    if (it != dataTypeMap.end())
    {
        return it->second;
    }
    else
    {
        throw std::runtime_error("Unknown data type.");
    }
}

inline uint64_t getDataTypeSize(const EnvData& envData)
{
    static const std::unordered_map<std::string, uint64_t> dataTypeMap = {
        {"float", sizeof(float)},
        {"bfloat16", sizeof(uint16_t)},
    };

    auto it = dataTypeMap.find(envData.dataType);
    if (it != dataTypeMap.end())
    {
        return it->second;
    }
    else
    {
        throw std::runtime_error("Unknown data type.");
    }
}

inline hcclRedOp_t getReductionOp(const EnvData& envData)
{
    static const std::unordered_map<std::string, hcclRedOp_t> redopMap = {
        {"sum", hcclSum},
        {"min", hcclMin},
        {"max", hcclMax},
    };

    auto it = redopMap.find(envData.redop);
    if (it != redopMap.end())
    {
        return it->second;
    }
    else
    {
        throw std::runtime_error("Unknown reduction op.");
    }
}

inline std::string formatBW(const double bytesPerSec)
{
    std::stringstream ss;
    ss << std::fixed << std::setprecision(6) << bytesPerSec / 1e9 << " GB/s";
    return ss.str();
}

inline std::string getPrintDelimiter(size_t length, char delimiter)
{
    std::stringstream ss;

    for (size_t i = 0; i < length; i++)
    {
        ss << delimiter;
    }
    return ss.str();
}

template<class T>
float getFloat(T value);

template<>
inline float getFloat<float>(float value)
{
    return value;
}

template<>
inline float getFloat<uint16_t>(uint16_t value)
{
    return bf16ToFloat(value);
}

inline float getInput(const HCL_Rank rank, const size_t nranks, const uint64_t i)
{
    // We want to make sure we use different values on each cell and between ranks,
    // but we don't want the summation to get too big, that is why we modulo by DATA_ELEMENTS_MAX.
    return rank + nranks * (i % DATA_ELEMENTS_MAX);
}

// demo infrastructure functions
double benchmark(const EnvData&                       envData,
                 const DeviceResources&               resources,
                 const std::function<void(uint64_t)>& fn,
                 const std::function<void()>&         fnCorrectness);

// demo environmental variables
EnvData getenvData();

// send-recv interface
void sendRecvTestDriver(
    EnvData& envData, const DeviceResources& resources, Buffers& buffers, const uint64_t size, Stats& stats);

// scale validation interface
#ifdef MPI_ENABLED
void scaleValidationTestDriver(EnvData&               envData,
                               const DeviceResources& resources,
                               const Buffers&         buffers,
                               const uint64_t         size);
#endif  // MPI_ENABLED
