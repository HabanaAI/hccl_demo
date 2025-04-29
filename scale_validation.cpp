/******************************************************************************
# Copyright (c) 2022 Habana Labs, Ltd.
# SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "common.h"

#include <algorithm>                                    // for std::for_each
#include <chrono>                                       // for Bandwidth calculation
#include <iostream>                                     // for io stream
static constexpr float SCALE_VALIDATION_MARGIN = 0.05;  // fraction of expected BW

enum class ControlType : uint8_t
{
    SEND = 1,
    RECEIVE,
    END,
};

#ifdef MPI_ENABLED
static synDeviceType getDeviceType(const synDeviceId deviceId)
{
    synDeviceInfoV2 deviceInfo;
    CHECK_SYNAPSE_STATUS(synDeviceGetInfoV2(deviceId, &deviceInfo));
    return deviceInfo.deviceType;
}

static uint64_t getExpectedScaleupBW(const DeviceResources& resources)
{
    synDeviceType deviceType = getDeviceType(resources.deviceHandle);
    switch (deviceType)
    {
        case synDeviceGaudi:
            return 12.5e9;
        case synDeviceGaudi2:
            return 37.5e9;
        case synDeviceGaudi3:
            return 75e9;
        default:
            log() << "Unknown device, setting expected scaleup bandwidth to 37.5GB" << std::endl;
            return 37.5e9;
    }
}

static void getScaleupPairs(const EnvData& envData, std::vector<RanksPairSendRecv>& ranksList)
{
    for (HCL_Rank sender = 0; sender < envData.nranks; sender++)
    {
        size_t boxNum = sender / envData.scaleupGroupSize;
        for (HCL_Rank receiver = boxNum * envData.scaleupGroupSize; receiver < (boxNum + 1) * envData.scaleupGroupSize;
             receiver++)
        {
            if (sender == receiver) continue;
            ranksList.push_back({sender, receiver});
        }
    }
}

void getScaleoutPairs(const EnvData& envData, std::vector<RanksPairSendRecv>& ranksList)
{
    for (HCL_Rank sender = 0; sender < envData.nranks; sender++)
    {
        for (HCL_Rank receiver = sender % envData.scaleupGroupSize; receiver < envData.nranks;
             receiver += envData.scaleupGroupSize)
        {
            if (sender == receiver) continue;
            ranksList.push_back({sender, receiver});
        }
    }
}

static void scaleValidationSend(const EnvData&         envData,
                                const DeviceResources& resources,
                                const Buffers&         buffers,
                                const HCL_Rank         receiver,
                                uint64_t&              result)
{
    // run single iteration as warmup
    CHECK_HCCL_STATUS(hcclSend((void*)buffers.inputDevPtrs[0],
                               buffers.inputSize / getDataTypeSize(envData),
                               getDataType(envData),
                               receiver,
                               resources.comm,
                               resources.collectiveStream));

    CHECK_SYNAPSE_STATUS(synStreamSynchronize(resources.collectiveStream));

    auto startTime = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < envData.numIters; ++i)
    {
        uint64_t index = i % buffers.inputDevPtrs.size();
        CHECK_HCCL_STATUS(hcclSend((void*)buffers.inputDevPtrs[index],
                                   buffers.inputSize / getDataTypeSize(envData),
                                   getDataType(envData),
                                   receiver,
                                   resources.comm,
                                   resources.collectiveStream));
    }

    // calculate result
    CHECK_SYNAPSE_STATUS(synStreamSynchronize(resources.collectiveStream));

    auto duration          = std::chrono::high_resolution_clock::now() - startTime;
    auto rankDurationInSec = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
    rankDurationInSec      = rankDurationInSec / envData.numIters;
    result                 = buffers.inputSize / rankDurationInSec;
}

static void scaleValidationReceive(const EnvData&         envData,
                                   const DeviceResources& resources,
                                   const Buffers&         buffers,
                                   const HCL_Rank         sender)
{
    // run single iteration as warmup
    CHECK_HCCL_STATUS(hcclRecv((void*)buffers.outputDevPtrs[0],
                               buffers.inputSize / getDataTypeSize(envData),
                               getDataType(envData),
                               sender,
                               resources.comm,
                               resources.collectiveStream));

    for (size_t i = 0; i < envData.numIters; ++i)
    {
        uint64_t index = i % buffers.inputDevPtrs.size();
        CHECK_HCCL_STATUS(hcclRecv((void*)buffers.outputDevPtrs[index],
                                   buffers.inputSize / getDataTypeSize(envData),
                                   getDataType(envData),
                                   sender,
                                   resources.comm,
                                   resources.collectiveStream));
    }
}

static void scaleValidationEnd(const EnvData& envData)
{
    for (HCL_Rank rank = 0; rank < envData.nranks; rank++)
    {
        if (rank != envData.root)
        {
            ControlType control = ControlType::END;
            CHECK_MPI_STATUS(MPI_Send((void*)&control, 1, MPI_UINT8_T, rank, 0, MPI_COMM_WORLD));
        }
    }
}

static bool scaleValidationServerStep(const EnvData&         envData,
                                      const DeviceResources& resources,
                                      const Buffers&         buffers,
                                      const HCL_Rank         sender,
                                      const HCL_Rank         receiver,
                                      const uint64_t         expectedBW)
{
    ControlType control;
    MPI_Status  status;

    // trigger receiver
    if (envData.rank != receiver)  // do server receive only after sending send request
    {
        control = ControlType::RECEIVE;
        CHECK_MPI_STATUS(MPI_Send((void*)&control, 1, MPI_UINT8_T, receiver, 0, MPI_COMM_WORLD));
        CHECK_MPI_STATUS(MPI_Send((void*)&sender, 1, MPI_INT, receiver, 0, MPI_COMM_WORLD));
    }

    // trigger sender
    if (envData.rank != sender)  // server send at the result stage
    {
        control = ControlType::SEND;
        CHECK_MPI_STATUS(MPI_Send((void*)&control, 1, MPI_UINT8_T, sender, 0, MPI_COMM_WORLD));
        CHECK_MPI_STATUS(MPI_Send((void*)&receiver, 1, MPI_INT, sender, 0, MPI_COMM_WORLD));
    }

    // server receive
    if (envData.rank == receiver)
    {
        scaleValidationReceive(envData, resources, buffers, sender);
    }

    // get result
    uint64_t result;
    if (envData.rank == sender)
    {
        scaleValidationSend(envData, resources, buffers, receiver, result);
    }
    else
    {
        // wait for result
        CHECK_MPI_STATUS(MPI_Recv((void*)&result, 1, MPI_UINT64_T, sender, 0, MPI_COMM_WORLD, &status))
    }

    // log
    if (result < (expectedBW - SCALE_VALIDATION_MARGIN * expectedBW))
    {
        log() << sender << "-->" << receiver << ": Test failed, Actual BW does not meet the expectation" << " ("
              << formatBW(result) << " - " << result * 100 / expectedBW << "% of expected BW)" << std::endl;

        return false;
    }

    return true;
}

static void scaleValidationClient(const EnvData& envData, const DeviceResources& resources, const Buffers& buffers)
{
    while (true)
    {
        MPI_Status  status;
        ControlType control;
        CHECK_MPI_STATUS(MPI_Recv((void*)&control, 1, MPI_UINT8_T, envData.root, 0, MPI_COMM_WORLD, &status))

        switch (control)
        {
            case ControlType::END:
                return;
            case ControlType::SEND:
                uint64_t result;
                int      receiver;
                CHECK_MPI_STATUS(MPI_Recv((void*)&receiver, 1, MPI_INT, envData.root, 0, MPI_COMM_WORLD, &status))
                scaleValidationSend(envData, resources, buffers, receiver, result);

                // send result to server
                CHECK_MPI_STATUS(MPI_Send((void*)&result, 1, MPI_UINT64_T, envData.root, 0, MPI_COMM_WORLD));
                break;
            case ControlType::RECEIVE:
                int sender;
                CHECK_MPI_STATUS(MPI_Recv((void*)&sender, 1, MPI_INT, envData.root, 0, MPI_COMM_WORLD, &status))
                scaleValidationReceive(envData, resources, buffers, sender);
                break;
            default:
                throw std::runtime_error {" Unexpected control message type"};
        }
    }
}

static bool scaleValidationCommonDriver(const EnvData&                        envData,
                                        const DeviceResources&                resources,
                                        const Buffers&                        buffers,
                                        const std::vector<RanksPairSendRecv>& ranksPairsList,
                                        const uint64_t                        expectedBW)
{
    bool isOK = true;

    if (isRoot(envData))
    {
        std::for_each(ranksPairsList.cbegin(), ranksPairsList.cend(), [&](RanksPairSendRecv pair) {
            if (!scaleValidationServerStep(envData, resources, buffers, pair.sendFromRank, pair.recvInRank, expectedBW))
                isOK = false;
        });
        scaleValidationEnd(envData);
    }
    else
    {
        scaleValidationClient(envData, resources, buffers);
    }

    return isOK;
}

void scaleValidationTestDriver(EnvData&               envData,
                               const DeviceResources& resources,
                               const Buffers&         buffers,
                               const uint64_t         size)
{
    bool isOK = false;

    // scaleup
    uint64_t expectedBW = getExpectedScaleupBW(resources);
    if (isRoot(envData))
    {
        log() << "ScaleUp - Expected " << formatBW(expectedBW) << std::endl;
    }
    std::vector<RanksPairSendRecv> scaleupPairsList;
    getScaleupPairs(envData, scaleupPairsList);
    isOK = scaleValidationCommonDriver(envData, resources, buffers, scaleupPairsList, expectedBW);
    if (isOK && isRoot(envData))
    {
        log() << "Test passed. The BW between all nodes meets the expectations" << std::endl;
    }

    // scaleout
    if (envData.nranks > envData.ranksPerNode)
    {
        expectedBW = envData.expectedScaleoutBW;
        if (expectedBW == 0)
        {
            throw std::runtime_error {"missing mandatory argument for scale validation: --scaleout_bw"};
        }

        if (isRoot(envData))
        {
            log() << "ScaleOut - Expected " << formatBW(expectedBW) << std::endl;
        }
        std::vector<RanksPairSendRecv> scaleoutPairsList;
        getScaleoutPairs(envData, scaleoutPairsList);
        isOK = scaleValidationCommonDriver(envData, resources, buffers, scaleoutPairsList, expectedBW);
        if (isOK && isRoot(envData))
        {
            log() << "Test passed. The BW between all nodes meets the expectations" << std::endl;
        }
    }

    envData.shouldCheckCorrectness = false;
}

#endif  // MPI_ENABLED
