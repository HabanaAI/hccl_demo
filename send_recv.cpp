/******************************************************************************
# Copyright (c) 2022 Habana Labs, Ltd.
# SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "common.h"

#include <sstream>  // for std::stringstream

uint64_t getUsableMemory(const synDeviceId deviceID)
{
    uint64_t freeMemory  = 0;
    uint64_t totalMemory = 0;  // Value is required by synDeviceGetMemoryInfo but not used
    CHECK_SYNAPSE_STATUS(synDeviceGetMemoryInfo(deviceID, &freeMemory, &totalMemory));
    return freeMemory;
}

static std::vector<RanksPairSendRecv> parseRanksList(const std::string& ranksListSt, const HCL_Rank maxRankNumber)
{
    std::vector<RanksPairSendRecv> ranksList;
    std::stringstream              ss(ranksListSt);

    std::vector<HCL_Rank> tempRanksVector;
    std::string           token;

    while (std::getline(ss, token, ','))
    {
        const HCL_Rank rankNum = std::stoi(token);
        if (rankNum <= maxRankNumber)
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
    else if (tempRanksVector.size() > 0)
    {
        const size_t pairsNum = tempRanksVector.size() / 2;
        for (size_t count = 0; count < pairsNum; count++)
        {
            HCL_Rank sendFromRank = tempRanksVector[count * 2];
            HCL_Rank recvInRank   = tempRanksVector[count * 2 + 1];
            ranksList.push_back({sendFromRank, recvInRank});
        }
    }

    return ranksList;
}

static hcclResult_t sendRecvTest(const EnvData&         envData,
                                 const DeviceResources& resources,
                                 const HCL_Rank         recvFromRank,
                                 const HCL_Rank         sendToRank,
                                 const size_t           count,
                                 const void*            sendbuff,
                                 void*                  recvbuff)
{
    hcclGroupStart();

    CHECK_HCCL_STATUS(
        hcclSend(sendbuff, count, getDataType(envData), sendToRank, resources.comm, resources.collectiveStream));
    CHECK_HCCL_STATUS(
        hcclRecv(recvbuff, count, getDataType(envData), recvFromRank, resources.comm, resources.collectiveStream));

    hcclGroupEnd();

    return hcclSuccess;
}

static hcclResult_t sendRecvRanksTest(uint64_t                     iter,
                                      const EnvData&               envData,
                                      const DeviceResources&       resources,
                                      const std::vector<HCL_Rank>& recvRanks,
                                      const std::vector<HCL_Rank>& sendRanks,
                                      const size_t                 count,
                                      const void*                  sendbuff,
                                      const std::vector<uint64_t>& recvbuffs)
{
    hcclGroupStart();

    for (const HCL_Rank sendRank : sendRanks)
    {
        CHECK_HCCL_STATUS(
            hcclSend(sendbuff, count, getDataType(envData), sendRank, resources.comm, resources.collectiveStream));
    }

    uint64_t recvBufferIndex = (recvRanks.size() * iter) % recvbuffs.size();
    for (const HCL_Rank recvRank : recvRanks)
    {
        CHECK_HCCL_STATUS(hcclRecv((void*)recvbuffs[recvBufferIndex],
                                   count,
                                   getDataType(envData),
                                   recvRank,
                                   resources.comm,
                                   resources.collectiveStream));
        recvBufferIndex = (recvBufferIndex + 1 == recvbuffs.size()) ? 0 : recvBufferIndex + 1;
    }

    hcclGroupEnd();

    return hcclSuccess;
}

void sendRecvTestDefaultDriver(const EnvData&         envData,
                               const DeviceResources& resources,
                               Buffers&               buffers,
                               const uint64_t         size,
                               Stats&                 stats)
{
    // The flow of the test is as follows:
    // For single box, exchange buffer with adjacent rank. If odd number of ranks then last rank does self send/recv.
    // For scale-out test, exchange buffer with next peer rank in ring manner.
    //
    // Example:
    // 4 boxes: R0 -> R8 & R0 <- R24, R8 <- R0 & R8 -> R16, R16 <- R8 & R16 -> R24, R24 <- R16 & R24 ->R0 etc.
    // 2 boxes: R0 <> R8, R1 <> R9, etc.
    //
    // In both cases, each rank does 1 send and 1 recv from another (same) rank.
    const size_t scaleupGroupSize = envData.scaleupGroupSize;
    const size_t numOfRanks       = envData.nranks;
    size_t       numOfBoxes       = envData.nranks / envData.scaleupGroupSize;
    if (numOfRanks % scaleupGroupSize > 0)
    {
        numOfBoxes++;
    }
    const size_t ranksPerBox = numOfRanks / numOfBoxes;

    const HCL_Rank myRank   = envData.rank;
    const size_t   myBoxNum = myRank / scaleupGroupSize;

    HCL_Rank sendToRank;
    HCL_Rank recvFromRank;
    if (numOfBoxes > 1)
    // scaleout
    {
        // Do ring with adjacent boxes
        const size_t targetSendBox = myBoxNum == numOfBoxes - 1 ? 0 : myBoxNum + 1;
        sendToRank                 = targetSendBox * ranksPerBox + (myRank % ranksPerBox);
        const size_t targetRecvBox = myBoxNum == 0 ? numOfBoxes - 1 : myBoxNum - 1;
        recvFromRank               = targetRecvBox * ranksPerBox + (myRank % ranksPerBox);
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

    // Choose benchmark function based on latency benchmark setting
    bool useLatencyBenchmark = false;
    const char* latencyEnv = getenv("HCCL_DEMO_LATENCY_BENCHMARK");
    if (latencyEnv && latencyEnv[0] == '1')
    {
        useLatencyBenchmark = true;
    }

    // Run send/recv with appropriate benchmark function
    if (useLatencyBenchmark)
    {
        stats.rankDurationInSec = benchmark_latency(
            envData,
            resources,
            [&](uint64_t iter) {
                uint64_t index = iter % buffers.inputDevPtrs.size();
                CHECK_HCCL_STATUS(sendRecvTest(envData,
                                               resources,
                                               recvFromRank,
                                               sendToRank,
                                               buffers.inputSize / getDataTypeSize(envData),
                                               (const void*)buffers.inputDevPtrs[index],
                                               (void*)buffers.outputDevPtrs[index]));
            },
            [&]() {
                CHECK_HCCL_STATUS(sendRecvTest(envData,
                                               resources,
                                               recvFromRank,
                                               sendToRank,
                                               buffers.inputSize / getDataTypeSize(envData),
                                               (const void*)buffers.inputDevPtrs[0],
                                               (void*)buffers.correctnessDevPtr));
            });
    }
    else
    {
        stats.rankDurationInSec = benchmark(
            envData,
            resources,
            [&](uint64_t iter) {
                uint64_t index = iter % buffers.inputDevPtrs.size();
                CHECK_HCCL_STATUS(sendRecvTest(envData,
                                               resources,
                                               recvFromRank,
                                               sendToRank,
                                               buffers.inputSize / getDataTypeSize(envData),
                                               (const void*)buffers.inputDevPtrs[index],
                                               (void*)buffers.outputDevPtrs[index]));
            },
            [&]() {
                CHECK_HCCL_STATUS(sendRecvTest(envData,
                                               resources,
                                               recvFromRank,
                                               sendToRank,
                                               buffers.inputSize / getDataTypeSize(envData),
                                               (const void*)buffers.inputDevPtrs[0],
                                               (void*)buffers.correctnessDevPtr));
            });
    }

    // Calculate expected results for correctness check
    if (envData.shouldCheckCorrectness)
    {
        for (size_t i = 0; i < buffers.outputSize / getDataTypeSize(envData); i++)
        {
            stats.expectedOutputs.push_back(getInput(recvFromRank, envData.nranks, i));
        }
    }

    stats.isDescribing = true;
}

void sendRecvRanksTestDriver(const EnvData&         envData,
                             const DeviceResources& resources,
                             Buffers&               buffers,
                             const uint64_t         size,
                             Stats&                 stats)
{
    // This test performs send_recv from/to specific ranks given as a list
    // A single rank can send to one or many ranks and can also recv from one or many ranks.
    // It supports both scale-up and scale-out send/recv.
    // It reports adjusted B/W according to number of receives.
    const size_t numOfRanks = envData.nranks;

    const std::string                    ranksListStr   = envData.ranksList;
    const std::vector<RanksPairSendRecv> ranksPairsList = parseRanksList(ranksListStr, numOfRanks - 1);

    std::vector<HCL_Rank> sendToRanks;
    std::vector<HCL_Rank> recvFromRanks;
    for (const auto ranksPair : ranksPairsList)
    {
        HCL_Rank sendingFromRank = ranksPair.sendFromRank;
        HCL_Rank receivingInRank = ranksPair.recvInRank;
        if (envData.rank == sendingFromRank)
        {
            sendToRanks.push_back(receivingInRank);
        }
        else if (envData.rank == receivingInRank)
        {
            recvFromRanks.push_back(sendingFromRank);
        }
    }

    uint64_t usableMemory = getUsableMemory(resources.deviceHandle);
    if (((recvFromRanks.size() + buffers.inputDevPtrs.size()) * size) > usableMemory)
    {
        throw std::runtime_error("Insufficient memory for test. Required " +
                                 std::to_string(recvFromRanks.size() + buffers.inputDevPtrs.size()) +
                                 " chunks of size " + std::to_string(size) + " bytes but only " +
                                 std::to_string(usableMemory / size) + " are available.");
    }

    uint64_t additionalOutputDevPtr = 0;

    if (buffers.outputDevPtrs.size() < recvFromRanks.size())
    {
        // Allocate additional receive buffers
        uint64_t additionalBuffers = recvFromRanks.size() - buffers.outputDevPtrs.size();
        CHECK_SYNAPSE_STATUS(
            synDeviceMalloc(resources.deviceHandle, size * additionalBuffers, 0, 0, &additionalOutputDevPtr));

        for (uint64_t index = 0; index < additionalBuffers; index++)
        {
            buffers.outputDevPtrs.push_back(additionalOutputDevPtr + (index * size));
        }
    }

    if (buffers.outputDevPtrs.size() < recvFromRanks.size())
    {
        throw std::runtime_error {"Number of allocated receive buffers isn't sufficient to fulfill number of receives"};
    }

    // Choose benchmark function based on latency benchmark setting
    bool useLatencyBenchmark = false;
    const char* latencyEnv = getenv("HCCL_DEMO_LATENCY_BENCHMARK");
    if (latencyEnv && latencyEnv[0] == '1')
    {
        useLatencyBenchmark = true;
    }

    // Run send/recv with appropriate benchmark function
    if (useLatencyBenchmark)
    {
        stats.rankDurationInSec = benchmark_latency(
            envData,
            resources,
            [&](uint64_t iter) {
                uint64_t index = iter % buffers.inputDevPtrs.size();
                CHECK_HCCL_STATUS(sendRecvRanksTest(iter,
                                                    envData,
                                                    resources,
                                                    recvFromRanks,
                                                    sendToRanks,
                                                    size / getDataTypeSize(envData),
                                                    (const void*)buffers.inputDevPtrs[index],
                                                    buffers.outputDevPtrs));
            },
            [&]() -> void {
                CHECK_HCCL_STATUS(sendRecvRanksTest(0,
                                                    envData,
                                                    resources,
                                                    recvFromRanks,
                                                    sendToRanks,
                                                    size / getDataTypeSize(envData),
                                                    (const void*)buffers.inputDevPtrs[0],
                                                    buffers.outputDevPtrs));
            });
    }
    else
    {
        stats.rankDurationInSec = benchmark(
            envData,
            resources,
            [&](uint64_t iter) {
                uint64_t index = iter % buffers.inputDevPtrs.size();
                CHECK_HCCL_STATUS(sendRecvRanksTest(iter,
                                                    envData,
                                                    resources,
                                                    recvFromRanks,
                                                    sendToRanks,
                                                    size / getDataTypeSize(envData),
                                                    (const void*)buffers.inputDevPtrs[index],
                                                    buffers.outputDevPtrs));
            },
            [&]() -> void {
                CHECK_HCCL_STATUS(sendRecvRanksTest(0,
                                                    envData,
                                                    resources,
                                                    recvFromRanks,
                                                    sendToRanks,
                                                    size / getDataTypeSize(envData),
                                                    (const void*)buffers.inputDevPtrs[0],
                                                    buffers.outputDevPtrs));
            });
    }

    if (recvFromRanks.size() > 0)
    {
        stats.isDescribing = true;
        stats.statName.append("_rank_" + std::to_string(envData.rank));
        stats.factor = recvFromRanks.size();
    }
}

void sendRecvTestDriver(EnvData&               envData,
                        const DeviceResources& resources,
                        Buffers&               buffers,
                        const uint64_t         size,
                        Stats&                 stats)
{
    // The flow of the test is as follows:
    // For single box, exchange buffer with adjacent rank. If odd number of ranks then last rank does self send/recv.
    // For scale-out test, exchange buffer with next peer rank in ring manner.
    //
    // Example:
    // 4 boxes: R0 -> R8 & R0 <- R24, R8 <- R0 & R8 -> R16, R16 <- R8 & R16 -> R24, R24 <- R16 & R24 ->R0 etc.
    // 2 boxes: R0 <> R8, R1 <> R9, etc.
    //
    // In both cases, each rank does 1 send and 1 recv from another (same) rank.
    stats.statName = "hcclSendRecv";
    stats.factor   = 1;
    if (envData.ranksList.length() > 0)
    {
        if (isRoot(envData))
        {
            log() << "Will perform ranks send_recv test with list: " << envData.ranksList << std::endl;
            log().flush();
        }

        sendRecvRanksTestDriver(envData, resources, buffers, size, stats);
        envData.shouldCheckCorrectness = false;
    }
    else
    {
        sendRecvTestDefaultDriver(envData, resources, buffers, size, stats);
    }
}
