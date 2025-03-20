/******************************************************************************
# Copyright (c) 2022 Habana Labs, Ltd.
# SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "common.h"

#include <algorithm>  // for std::min_element, std::max_element
#include <chrono>     // for Bandwidth calculation
#include <cmath>      // for pow
#include <cstddef>    // for std::to_string
#include <fstream>    // for file stream
#include <numeric>    // for std::accumulate
#include <stdexcept>  // for std::runtime_error
#include <unistd.h>   // for sleep

#define RST "\033[0m"          // reset (white)
#define GRN "\033[32m"         // Green
#define RED "\033[1m\033[31m"  // Bold Red

enum class AffinityEnum : uint32_t
{
    PHYSICAL  = 0x1,
    NUMA      = 0x10,
    ISOLATION = 0x100
};

std::stringstream get_affinity_message(int value, AffinityEnum level, const std::string& feature)
{
    std::stringstream ss;
    const bool        is_done = value & static_cast<int>(level);
    std::stringstream status;
    status << (is_done ? "enabled" : "not enabled");

    std::stringstream color;
    color << (is_done ? GRN : RED);

    ss << color.str() << feature << " - " << status.str() << RST << "\n";
    return ss;
}

// Function to get and evaluate the affinity level
std::stringstream get_affinity_level()
{
    auto              numa_mapping_dir = "";
    const char* const env_value        = getenv("NUMA_MAPPING_DIR");
    numa_mapping_dir                   = (env_value != nullptr) ? env_value : numa_mapping_dir;

    std::stringstream file_final_class_output;
    file_final_class_output << numa_mapping_dir << "/.habana_module_affinity_classification";
    std::ifstream fin(file_final_class_output.str());

    if (!fin.is_open())
    {
        std::cout << RED << "Warning: Unable to open file " << file_final_class_output.str() << RST << std::endl;
        std::stringstream ss;
        ss << "No core affinity optimization level could be determined.";
        return ss;
    }

    std::string first_line;
    getline(fin, first_line);  // Read the first line of the file
    fin.close();

    // Convert the first line to an integer (assuming it's a hexadecimal value)
    int affinity_value = 0;
    try
    {
        affinity_value = stoi(first_line, nullptr, 16);  // Parse as hex
    }
    catch (const std::invalid_argument& e)
    {
        std::cout << RED << "Error: Invalid affinity value found in the file." << RST << std::endl;
        std::stringstream ss;
        ss << "Failed to interpret core affinity optimization level.";
        return ss;
    }

    // Generate the result messages
    std::stringstream result;
    result << get_affinity_message(affinity_value, AffinityEnum::PHYSICAL, "Physical core ").str();
    result << get_affinity_message(affinity_value, AffinityEnum::NUMA, "NUMA          ").str();
    result << get_affinity_message(affinity_value, AffinityEnum::ISOLATION, "Core isolation").str();

    return result;
}

// Constants
static constexpr size_t MAX_PRINTED_BUFFER_ELEMENTS = 4;

double benchmark(const EnvData&                       envData,
                 const DeviceResources&               resources,
                 const std::function<void(uint64_t)>& fn,
                 const std::function<void()>&         fnCorrectness)
{
    float rankDurationInSec;

    // Run warmup iterations to sync all the gaudis on the device.
    size_t iterations = 1;
    char   default_gdr_input[128];
    synConfigurationGet("HCCL_GAUDI_DIRECT", default_gdr_input, sizeof(default_gdr_input));
    if (default_gdr_input[0] == '1')
    {
        iterations = MAX_BUFFER_COUNT;
    }
    for (size_t warmup_iter = 0; warmup_iter < iterations; ++warmup_iter)
    {
        fn(warmup_iter);
    }

    CHECK_SYNAPSE_STATUS(synStreamSynchronize(resources.collectiveStream));

    // Actual iterations
    auto startTime = std::chrono::high_resolution_clock::now();

    for (size_t iter = 1; iter < envData.numIters; ++iter)
    {
        fn(iter);
    }

    // Correctness run on separate device output buffer
    fnCorrectness();

    CHECK_SYNAPSE_STATUS(synStreamSynchronize(resources.collectiveStream));

    const auto duration = std::chrono::high_resolution_clock::now() - startTime;
    rankDurationInSec   = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
    rankDurationInSec   = rankDurationInSec / envData.numIters;

    return rankDurationInSec;
}

static void initMPI()
{
#if MPI_ENABLED
    // Initialize the Open MPI execution context.
    CHECK_MPI_STATUS(MPI_Init(NULL, NULL));
#endif  // MPI_ENABLED
}

static HCL_Rank handleCustomComm(EnvData& envData, DeviceResources& resources)
{
    // Custom comm is not supported for send_recv and scale_validation
    if (envData.testType == "send_recv" || envData.testType == "scale_validation")
    {
        throw std::runtime_error {"Custom comm is not supported for this test type"};
    }

    std::vector<HCL_Rank> peers = envData.customComm;

    // Choosing new root rank if it is not part of the custom comm.
    std::vector<HCL_Rank>::iterator rootIt = find(peers.begin(), peers.end(), envData.root);
    if (rootIt == peers.end())
    {
        rootIt       = peers.begin();
        envData.root = *peers.begin();
        if (isRoot(envData))
        {
            log() << "While building a new custom communicator, the root rank is automatically set to "
                  << *peers.begin() << "." << std::endl;
        }
    }

    // Check if the current rank is part of the custom comm
    std::vector<HCL_Rank>::iterator rankIt = find(peers.begin(), peers.end(), envData.rank);
    if (rankIt == peers.end())
    {
        log() << "HCCL demo process id (" << envData.rank << ") will not participate in the custom communicator"
              << std::endl;
#if MPI_ENABLED
        hcclUniqueId uniqueID {};
        CHECK_MPI_STATUS(MPI_Bcast(&uniqueID, sizeof(uniqueID), MPI_BYTE, envData.root, MPI_COMM_WORLD));
        CHECK_MPI_STATUS(MPI_Finalize());
#endif
        exit(0);
    }

    // In the custom comm - override params to match new custom comm
    envData.nranks     = peers.size();
    resources.commRoot = distance(peers.begin(), rootIt);

    return distance(peers.begin(), rankIt);
}

static void initDevice(EnvData& envData, DeviceResources& resources)
{
    HCL_Rank commRank = envData.rank;
    if (envData.customComm.size() == 0)
    {
        // Generate HCCL comm world
        resources.commRoot = envData.root;
        for (HCL_Rank i = 0; i < envData.nranks; i++)
        {
            envData.customComm.push_back(i);
        }
    }
    else
    {
        commRank = handleCustomComm(envData, resources);
    }

    // Initialize Synapse API context
    CHECK_SYNAPSE_STATUS(synInitialize());

    // Acquire device
    synModuleId deviceModuleID = envData.rank % envData.ranksPerNode;
    synStatus   rc             = synDeviceAcquireByModuleId(&resources.deviceHandle, deviceModuleID);
    if (rc != synSuccess)
    {
        deviceModuleID = INVALID_MODULE_ID;
        CHECK_SYNAPSE_STATUS(synDeviceAcquire(&resources.deviceHandle, nullptr));
    }

#if AFFINITY_ENABLED
    if (setupAffinity(deviceModuleID) != 0)
    {
        throw std::runtime_error {"Affinity setting for HCCL demo failed."};
    }
#endif

    // Generate unique id
    hcclUniqueId uniqueID {};
    if (isRoot(envData))
    {
        CHECK_HCCL_STATUS(hcclGetUniqueId(&uniqueID));
    }

#if MPI_ENABLED
    CHECK_MPI_STATUS(MPI_Bcast(&uniqueID, sizeof(uniqueID), MPI_BYTE, envData.root, MPI_COMM_WORLD));
#endif  // MPI_ENABLED

    // Create new HCCL communicator
    CHECK_HCCL_STATUS(hcclCommInitRank(&resources.comm, envData.nranks, uniqueID, commRank));

    // Create Streams
    CHECK_SYNAPSE_STATUS(synStreamCreateGeneric(&resources.collectiveStream, resources.deviceHandle, 0));
    CHECK_SYNAPSE_STATUS(synStreamCreateGeneric(&resources.deviceToHostStream, resources.deviceHandle, 0));
    CHECK_SYNAPSE_STATUS(synStreamCreateGeneric(&resources.hostToDeviceStream, resources.deviceHandle, 0));
}

static void destroyDevice(DeviceResources& resources)
{
    // destroy streams
    CHECK_SYNAPSE_STATUS(synStreamDestroy(resources.collectiveStream));
    CHECK_SYNAPSE_STATUS(synStreamDestroy(resources.deviceToHostStream));
    CHECK_SYNAPSE_STATUS(synStreamDestroy(resources.hostToDeviceStream));

    // Destroy HCCL communicator
    CHECK_HCCL_STATUS(hcclCommDestroy(resources.comm));

    // Clean up HCCL
    CHECK_SYNAPSE_STATUS(synDeviceRelease(resources.deviceHandle));

    // Destroy synapse api context
    CHECK_SYNAPSE_STATUS(synDestroy());
}

template<class T>
static void createCSVReport(const EnvData& envData, const std::string& type, std::vector<T> data, int count)
{
    auto              now       = std::chrono::system_clock::now();
    auto              in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream datetime;
    datetime << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H%M");
    auto csvPath = envData.dataCSVPath + "/" + "HCCL_demo_" + envData.testType + "_" + type + "_" +
                   std::to_string(envData.rank) + "_" + datetime.str() + ".csv";

    std::fstream fout;
    fout.open(csvPath, std::ios::out | std::ios::app);
    for (int i = 0; i < count; i++)
    {
        if (isBfloat16(envData))
        {
            fout << bf16ToFloat(data[i]) << "\n";
        }
        else
        {
            fout << data[i] << "\n";
        }
    }
    fout.close();
}

static float calcExpectedReduction(std::vector<float>& args, hcclRedOp_t redop)
{
    switch (redop)
    {
        case hcclSum:
            return std::accumulate(args.cbegin(), args.cend(), 0);
        case hcclMin:
            return *std::min_element(args.cbegin(), args.cend());
        case hcclMax:
            return *std::max_element(args.cbegin(), args.cend());
        default:
            throw std::runtime_error {" Unknown reduction op."};
    }
}

static void getExpectedOutputs(const EnvData& envData, const Buffers& buffers, std::vector<float>& expectedOutputs)
{
    size_t   inputCount  = buffers.inputSize / getDataTypeSize(envData);
    size_t   outputCount = buffers.outputSize / getDataTypeSize(envData);
    HCL_Rank inputRank   = 0;
    size_t   inputIdx    = 0;

    for (size_t i = 0; i < outputCount; i++)
    {
        if (envData.testType == "broadcast")
        {
            // Output is not defined for root rank
            if (isRoot(envData))
            {
                return;
            }

            // Every rank gets root input
            inputRank = envData.root;
            inputIdx  = i;
        }
        else if (envData.testType == "all_reduce")
        {
            // Fill input data, example:
            // Input        |   Output
            // G0 G1 G2 G3      G0 G1 G2 G3
            // 0  1  2  3   =>  6  6  6  6
            // 4  5  6  7       22 22 22 22
            // 8  9  10 11      38 38 38 38
            // 12 13 14 15      54 54 54 54
            inputIdx = i;
        }
        else if (envData.testType == "reduce_scatter")
        {
            // Fill input data, example:
            // Input        |   Output
            // G0 G1 G2 G3      G0 G1 G2 G3
            // 0  1  2  3   =>  6  22 38 54
            // 4  5  6  7
            // 8  9  10 11
            // 12 13 14 15
            inputIdx = i + envData.rank * (inputCount / envData.nranks);
        }
        else if (envData.testType == "all_gather")
        {
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
            inputRank = i / inputCount;
            inputIdx  = i % inputCount;
        }
        else if (envData.testType == "all2all")
        {
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
            inputRank = i / (inputCount / envData.nranks);
            inputIdx  = i % (inputCount / envData.nranks) + envData.rank * (inputCount / envData.nranks);
        }
        else if (envData.testType == "reduce")
        {
            // Output is not defined for non-root ranks
            if (envData.rank != envData.root)
            {
                return;
            }

            // Fill input data, example:
            // root = G1
            // Input        |   Output
            // G0 G1 G2 G3      G0 G1 G2 G3
            // 0  1  2  3   =>      6
            // 4  5  6  7          22
            // 8  9  10 11         38
            // 12 13 14 15         54
            inputIdx = i;
        }

        if (envData.testType.find("reduce") != std::string::npos)
        {
            std::vector<float> inputs(envData.nranks);
            for (HCL_Rank rank = 0; rank < envData.nranks; rank++)
            {
                inputs[rank] = getInput(envData.customComm[rank], envData.nranks, inputIdx);
            }
            expectedOutputs.push_back(calcExpectedReduction(inputs, getReductionOp(envData)));
        }
        else
        {
            expectedOutputs.push_back(getInput(envData.customComm[inputRank], envData.nranks, inputIdx));
        }
    }
}

static void
prepareBuffers(const EnvData& envData, const DeviceResources& resources, const uint64_t size, Buffers& buffers)
{
    // Calculate buffers sizes
    buffers.inputSize = size;
    if (envData.testType == "all_gather")
    {
        buffers.outputSize = size * envData.nranks;
    }
    else if (envData.testType == "reduce_scatter")
    {
        buffers.outputSize = size / envData.nranks;
    }
    else
    {
        buffers.outputSize = size;
    }

    // Validate calculated buffer size.
    if (buffers.inputSize < getDataTypeSize(envData) || buffers.outputSize < getDataTypeSize(envData))
    {
        throw std::runtime_error {"Invalid buffer size"};
    }

    // Calculate number of buffers
    const uint64_t maxBufferSize   = std::max(buffers.inputSize, buffers.outputSize);
    uint64_t       numberOfBuffers = 2;
    if (maxBufferSize <= ALLOCATED_HBM_SIZE)
    {
        numberOfBuffers = (ALLOCATED_HBM_SIZE / maxBufferSize) <= AMOUNT_JUMBO_BUFFERS
                              ? AMOUNT_JUMBO_BUFFERS
                              : ALLOCATED_HBM_SIZE / maxBufferSize;
    }
    numberOfBuffers = std::min(numberOfBuffers, MAX_BUFFER_COUNT);

    // Allocate buffers on the device
    uint64_t inputDevPtr      = 0;
    uint64_t outputDevPtr     = 0;
    buffers.correctnessDevPtr = 0;
    CHECK_SYNAPSE_STATUS(
        synDeviceMalloc(resources.deviceHandle, buffers.inputSize * numberOfBuffers, 0, 0, &inputDevPtr));
    CHECK_SYNAPSE_STATUS(
        synDeviceMalloc(resources.deviceHandle, buffers.outputSize * numberOfBuffers, 0, 0, &outputDevPtr));

    for (uint64_t index = 0; index < numberOfBuffers; index++)
    {
        buffers.inputDevPtrs.push_back(inputDevPtr + (index * buffers.inputSize));
        buffers.outputDevPtrs.push_back(outputDevPtr + (index * buffers.outputSize));
    }

    // Set default correctness buffer on the device
    buffers.correctnessDevPtr = buffers.outputDevPtrs[0];
}

template<class T>
static void
generateInputs(const EnvData& envData, const DeviceResources& resources, const uint64_t size, Buffers& buffers)
{
    // Allocate correctness buffer on the device
    CHECK_SYNAPSE_STATUS(synDeviceMalloc(resources.deviceHandle, buffers.outputSize, 0, 0, &buffers.correctnessDevPtr));

    // Allocate temp host buffer
    std::vector<T> inputHostData(buffers.inputSize / getDataTypeSize(envData));
    void*          inputHostDataPtr = reinterpret_cast<void*>(inputHostData.data());

    // Create inputs
    for (size_t i = 0; i < inputHostData.size(); i++)
    {
        float value = getInput(envData.rank, envData.nranks, i);
        if (isBfloat16(envData))
        {
            inputHostData[i] = floatToBf16(value);
        }
        else
        {
            inputHostData[i] = value;
        }
    }

    // Copy from inputHostDataPtr to inputDevPtr (to be used in benchmark)
    CHECK_SYNAPSE_STATUS(synHostMap(resources.deviceHandle, buffers.inputSize, inputHostDataPtr));
    CHECK_SYNAPSE_STATUS(synMemCopyAsync(resources.hostToDeviceStream,
                                         (uint64_t)inputHostDataPtr,
                                         buffers.inputSize,
                                         buffers.inputDevPtrs[0],
                                         HOST_TO_DRAM));
    CHECK_SYNAPSE_STATUS(synStreamSynchronize(resources.hostToDeviceStream));
    CHECK_SYNAPSE_STATUS(synHostUnmap(resources.deviceHandle, inputHostDataPtr));

    // Save the input data to a CSV file if requested
    if (!envData.dataCSVPath.empty())
    {
        createCSVReport(envData, "input", inputHostData, buffers.inputSize / getDataTypeSize(envData));
    }
}

static void clearBuffers(const DeviceResources& resources, Buffers& buffers)
{
    CHECK_SYNAPSE_STATUS(synDeviceFree(resources.deviceHandle, buffers.inputDevPtrs[0], 0));
    CHECK_SYNAPSE_STATUS(synDeviceFree(resources.deviceHandle, buffers.outputDevPtrs[0], 0));
    if (buffers.correctnessDevPtr != buffers.outputDevPtrs[0])
    {
        CHECK_SYNAPSE_STATUS(synDeviceFree(resources.deviceHandle, buffers.correctnessDevPtr, 0));
    }
}

static bool compareBf16(EnvData envData, float expected, float outValue, int i)
{
    const float accuracyCoefficient = bf16AccuracyCoefficient(envData.nranks);
    const float tolerance           = fabs(outValue) * accuracyCoefficient;
    const float difference          = fabs(outValue - expected);

    if (difference > tolerance)
    {
        log() << "index=" << i << ", expectedValue=" << expected << ", value=" << outValue
              << ", thisRankId=" << envData.rank << ", tolerance=" << tolerance << ", difference=" << difference
              << ", accuracyCoefficient=" << accuracyCoefficient << ", m_numberOfRanks=" << envData.nranks << std::endl;
        return false;
    }
    return true;
}

template<class T>
static bool checkCorrectness(const EnvData&         envData,
                             const DeviceResources& resources,
                             const Buffers&         buffers,
                             std::vector<float>&    expectedOutputs)
{
    bool isOK = true;

    auto        outputHostData    = std::vector<T>(buffers.outputSize / getDataTypeSize(envData));
    const void* outputHostDataPtr = reinterpret_cast<void*>(outputHostData.data());

    CHECK_SYNAPSE_STATUS(synHostMap(resources.deviceHandle, buffers.outputSize, outputHostDataPtr));
    CHECK_SYNAPSE_STATUS(synMemCopyAsync(resources.deviceToHostStream,
                                         buffers.correctnessDevPtr,
                                         buffers.outputSize,
                                         (uint64_t)outputHostDataPtr,
                                         DRAM_TO_HOST));
    CHECK_SYNAPSE_STATUS(synStreamSynchronize(resources.deviceToHostStream));
    CHECK_SYNAPSE_STATUS(synHostUnmap(resources.deviceHandle, outputHostDataPtr));

    for (size_t i = 0; i < outputHostData.size(); i++)
    {
        if (i < expectedOutputs.size())
        {
            if (isBfloat16(envData))
            {
                isOK &= compareBf16(envData, expectedOutputs[i], bf16ToFloat(outputHostData[i]), i);
            }
            else
            {
                if (std::abs((float)outputHostData[i] - expectedOutputs[i]) != 0)
                {
                    isOK = false;
                }
            }
        }
    }

    if (expectedOutputs.size() > 0)
    {
        log() << "rank=" << envData.rank << " size=" << buffers.inputSize << " <" << envData.dataType << ">";

        // Print input buffer.
        log() << " Input Buffer [";
        for (size_t i = 0; i < std::min(MAX_PRINTED_BUFFER_ELEMENTS, buffers.inputSize / getDataTypeSize(envData)); i++)
        {
            log() << getInput(envData.rank, envData.nranks, i) << " ";
        }
        if (buffers.inputSize / getDataTypeSize(envData) > MAX_PRINTED_BUFFER_ELEMENTS)
        {
            log() << "...";
        }
        log() << "]";

        // Print output buffer.
        log() << " Output Buffer [";
        for (size_t i = 0; i < std::min(MAX_PRINTED_BUFFER_ELEMENTS, outputHostData.size()); i++)
        {
            log() << getFloat(outputHostData[i]) << " ";
        }
        if (outputHostData.size() > MAX_PRINTED_BUFFER_ELEMENTS)
        {
            log() << "...";
        }
        log() << "]";

        // Print correctness check result.
        log() << " which is " << (isOK ? "fine." : "bad.") << std::endl;
    }

    if (!envData.dataCSVPath.empty())
    {
        createCSVReport(envData, "output", outputHostData, buffers.outputSize / getDataTypeSize(envData));
    }

    return isOK;
}

static void
describeStat(const EnvData& envData, const Buffers& buffers, const Stats& stats, std::vector<ReportEntry>& reportVec)
{
    auto algoBW = (double)buffers.inputSize / stats.rankDurationInSec;
    auto nwBW   = algoBW * stats.factor;

    if (envData.sizeMax > envData.sizeMin)
    {
        log() << "Processing data_size " << buffers.inputSize << std::endl;
        ReportEntry report_entry = {buffers.inputSize,
                                    (uint64_t)(buffers.inputSize / getDataTypeSize(envData)),
                                    stats.rankDurationInSec,
                                    algoBW,
                                    nwBW};
        reportVec.push_back(report_entry);
    }
    else
    {
        // Sleep in order to describe stats after all correctness logs.
        sleep(1);

        std::string statHeadline = stats.statName + "(dataSize=" + std::to_string(buffers.inputSize) +
                                   ", count=" + std::to_string(buffers.inputSize / getDataTypeSize(envData)) +
                                   ", dtype=" + envData.dataType + ", iterations=" + std::to_string(envData.numIters) +
                                   ")";
        size_t delimiterSize = statHeadline.length() + std::string {"[BENCHMARK]"}.length() + 1;

        log() << getPrintDelimiter(delimiterSize, '#') << '\n';
        log() << "[BENCHMARK] " << statHeadline << '\n';
        log() << "[BENCHMARK]     NW Bandwidth   : " << formatBW(nwBW) << '\n';
        log() << "[BENCHMARK]     Algo Bandwidth : " << formatBW(algoBW);
        log() << '\n' << getPrintDelimiter(delimiterSize, '#') << '\n';
        log() << "Core affinity optimization result: " << '\n' << get_affinity_level().str() << std::endl;
    }

    // Write results to csv file
    auto csvPath = envData.resultsCSVPath;
    if (!csvPath.empty())
    {
        std::ofstream output;
        output.open(csvPath, std::ofstream::out | std::ofstream::app);
        output << envData.testType << "," << envData.rank << "," << envData.dataType << "," << buffers.inputSize << ","
               << envData.numIters << "," << formatBW(nwBW) << std::endl;
        output.close();
    }
}

using CollectiveWrapper = std::function<void(const EnvData&         envData,
                                             const DeviceResources& resources,
                                             const void*            sendbuff,
                                             void*                  recvbuff,
                                             size_t                 recvcount)>;

static void hcclBroadcastWrapper(const EnvData&         envData,
                                 const DeviceResources& resources,
                                 const void*            sendbuff,
                                 void*                  recvbuff,
                                 size_t                 recvcount)
{
    CHECK_HCCL_STATUS(hcclBroadcast(sendbuff,
                                    recvbuff,
                                    recvcount,
                                    getDataType(envData),
                                    resources.commRoot,
                                    resources.comm,
                                    resources.collectiveStream));
}

static void hcclAllReduceWrapper(const EnvData&         envData,
                                 const DeviceResources& resources,
                                 const void*            sendbuff,
                                 void*                  recvbuff,
                                 size_t                 recvcount)
{
    CHECK_HCCL_STATUS(hcclAllReduce(sendbuff,
                                    recvbuff,
                                    recvcount,
                                    getDataType(envData),
                                    getReductionOp(envData),
                                    resources.comm,
                                    resources.collectiveStream));
}

static void hcclReduceScatterWrapper(const EnvData&         envData,
                                     const DeviceResources& resources,
                                     const void*            sendbuff,
                                     void*                  recvbuff,
                                     size_t                 recvcount)
{
    CHECK_HCCL_STATUS(hcclReduceScatter(sendbuff,
                                        recvbuff,
                                        recvcount / envData.nranks,
                                        getDataType(envData),
                                        getReductionOp(envData),
                                        resources.comm,
                                        resources.collectiveStream));
}

static void hcclAllGatherWrapper(const EnvData&         envData,
                                 const DeviceResources& resources,
                                 const void*            sendbuff,
                                 void*                  recvbuff,
                                 size_t                 recvcount)
{
    CHECK_HCCL_STATUS(
        hcclAllGather(sendbuff, recvbuff, recvcount, getDataType(envData), resources.comm, resources.collectiveStream));
}

static void hcclAlltoAllWrapper(const EnvData&         envData,
                                const DeviceResources& resources,
                                const void*            sendbuff,
                                void*                  recvbuff,
                                size_t                 recvcount)
{
    CHECK_HCCL_STATUS(
        hcclAlltoAll(sendbuff, recvbuff, recvcount, getDataType(envData), resources.comm, resources.collectiveStream));
}

static void hcclReduceWrapper(const EnvData&         envData,
                              const DeviceResources& resources,
                              const void*            sendbuff,
                              void*                  recvbuff,
                              size_t                 recvcount)
{
    CHECK_HCCL_STATUS(hcclReduce(sendbuff,
                                 recvbuff,
                                 recvcount,
                                 getDataType(envData),
                                 getReductionOp(envData),
                                 resources.commRoot,
                                 resources.comm,
                                 resources.collectiveStream));
}

static void collectiveTestDriver(const EnvData&         envData,
                                 const DeviceResources& resources,
                                 const Buffers&         buffers,
                                 const uint64_t         size,
                                 Stats&                 stats)
{
    CollectiveWrapper collective;

    if (envData.testType == "broadcast")
    {
        stats.statName = "hcclBroadcast";
        stats.factor   = 1;
        collective     = hcclBroadcastWrapper;
    }
    else if (envData.testType == "all_reduce")
    {
        stats.statName = "hcclAllReduce";
        stats.factor   = ((double)(2 * (envData.nranks - 1))) / ((double)envData.nranks);
        collective     = hcclAllReduceWrapper;
    }
    else if (envData.testType == "reduce_scatter")
    {
        stats.statName = "hcclReduceScatter";
        stats.factor   = ((double)(envData.nranks - 1)) / ((double)envData.nranks);
        collective     = hcclReduceScatterWrapper;
    }
    else if (envData.testType == "all_gather")
    {
        stats.statName = "hcclAllGather";
        stats.factor   = ((double)(envData.nranks - 1));
        collective     = hcclAllGatherWrapper;
    }
    else if (envData.testType == "all2all")
    {
        stats.statName = "hcclAlltoAll";
        stats.factor   = ((double)(envData.nranks - 1)) / ((double)envData.nranks);
        collective     = hcclAlltoAllWrapper;
    }
    else if (envData.testType == "reduce")
    {
        stats.statName = "hcclReduce";
        stats.factor   = 1;
        collective     = hcclReduceWrapper;
    }
    else
    {
        log() << "Unknown test type (" << envData.testType << ")" << std::endl;
        return;
    }

    // Run HCCL collective
    stats.rankDurationInSec = benchmark(
        envData,
        resources,
        [&](uint64_t iter) {
            uint64_t index = iter % buffers.inputDevPtrs.size();
            collective(envData,
                       resources,
                       (const void*)buffers.inputDevPtrs[index],
                       (void*)buffers.outputDevPtrs[index],
                       buffers.inputSize / getDataTypeSize(envData));
        },
        [&]() {
            collective(envData,
                       resources,
                       (const void*)buffers.inputDevPtrs[0],
                       (void*)buffers.correctnessDevPtr,
                       buffers.inputSize / getDataTypeSize(envData));
        });

    if (envData.shouldCheckCorrectness)
    {
        getExpectedOutputs(envData, buffers, stats.expectedOutputs);
    }

    if (isRoot(envData))
    {
        stats.isDescribing = true;
    }
}

static void printReport(const EnvData& envData, const std::vector<ReportEntry>& reportVec)
{
    constexpr size_t columnWidth = 14;

    const static std::vector<std::string> header = {"size", "count", "type", "redop", "time", "algoBW", "nw_bw"};
    const static std::vector<std::string> units  = {"(B)", "(elements)", "", "", "(ms)", "(GB/s)", "(GB/s)"};

    std::stringstream ss;
    const std::string summary = "[SUMMARY REPORT]";
    const std::string statName =
        "(src!=dst, collective=" + envData.testType + ", iterations=" + std::to_string(envData.numIters) + ")";
    size_t delimiterSize = statName.length() + 1;
    ss << '\n' << getPrintDelimiter(delimiterSize, '#') << std::endl;
    ss << summary << '\n' << statName << '\n' << std::endl;
    ss << std::left;

    // print header
    for (size_t i = 0; i < header.size(); ++i)
    {
        if (header[i] == "redop" && envData.redop == "") continue;
        ss << std::setw(columnWidth) << header[i];
    }
    ss << std::endl;

    // print units
    for (size_t i = 0; i < units.size(); ++i)
    {
        if (header[i] == "redop" && envData.redop == "") continue;
        ss << std::setw(columnWidth) << units[i];
    }
    ss << std::endl;

    // print stats for each data size
    for (const auto& entry : reportVec)
    {
        ss << std::setw(columnWidth) << std::to_string(entry.size) << std::setw(columnWidth)
           << std::to_string(entry.count) << std::setw(columnWidth) << envData.dataType;
        if (envData.redop != "")
        {
            ss << std::setw(columnWidth) << envData.redop;
        }
        ss << std::setw(columnWidth) << std::fixed << std::setprecision(3) << entry.time * 1000
           << std::setw(columnWidth) << std::fixed << std::setprecision(6) << entry.algoBW / 1e9
           << std::setw(columnWidth) << std::fixed << std::setprecision(6) << entry.avgBW / 1e9 << std::endl;
    }
    log() << ss.str();
    log() << "Core affinity optimization result: " << '\n' << get_affinity_level().str() << std::endl;
}

template<class T>
static void runTest(EnvData& envData, const DeviceResources& resources)
{
    bool isOK = true;

    std::vector<ReportEntry> reportVec;
    for (double size = envData.sizeMin; size <= envData.sizeMax; size *= pow(2, envData.sizeInc))
    {
        Buffers buffers;
        prepareBuffers(envData, resources, size, buffers);
        if (envData.shouldCheckCorrectness)
        {
            generateInputs<T>(envData, resources, size, buffers);
        }

        // Perform test
        Stats stats;
        if (envData.testType == "send_recv")
        {
            sendRecvTestDriver(envData, resources, buffers, static_cast<uint64_t>(size), stats);
        }
        else if (envData.testType == "scale_validation")
        {
#ifdef MPI_ENABLED
            // since there is no correctness check here the data can be random (no need to initialize)
            scaleValidationTestDriver(envData, resources, buffers, static_cast<uint64_t>(size));
#else
            throw std::runtime_error {"MPI must be enabled for scale validation test"};
#endif  // MPI_ENABLED
        }
        else
        {
            collectiveTestDriver(envData, resources, buffers, static_cast<uint64_t>(size), stats);
        }

        if (envData.shouldCheckCorrectness && !checkCorrectness<T>(envData, resources, buffers, stats.expectedOutputs))
        {
            isOK = false;
        }

        if (stats.isDescribing)
        {
            describeStat(envData, buffers, stats, reportVec);
        }

        clearBuffers(resources, buffers);
    }

    if (reportVec.size() > 0)
    {
        printReport(envData, reportVec);
    }

    if (!isOK) throw std::runtime_error {"Collective operation has failed on correctness."};
}

int main()
{
    try
    {
        initMPI();

        auto envData = getenvData();
        if (isRoot(envData))
        {
#if MPI_ENABLED
            log() << "MPI enabled. Make sure that HCCL demo is launched with mpirun." << std::endl;
#endif  // MPI_ENABLED
            log() << "Running HCCL Demo :: A simple program demonstrating HCCL usage from C++" << std::endl;
        }

        // Init device
        DeviceResources resources;
        initDevice(envData, resources);

        if (isBfloat16(envData))
        {
            runTest<uint16_t>(envData, resources);
        }
        else
        {
            runTest<float>(envData, resources);
        }

        destroyDevice(resources);

#if MPI_ENABLED
        CHECK_MPI_STATUS(MPI_Finalize());
#endif  // MPI_ENABLED
    }
    catch (const std::exception& ex)
    {
        log() << "HCCL demo error: " << ex.what() << std::endl;
        return -1;
    }
    return 0;
}
