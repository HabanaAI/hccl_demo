/******************************************************************************
# Copyright (c) 2022 Habana Labs, Ltd.
# SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#pragma once

// Affinity setup functions
int  setAutoAffinity(int moduleID);
int  setCustomAffinity(int moduleID, int numSockets, int numCoresPerSocket, int numHT);
int  setBestEffortAffinity(int moduleID);
void printAffinity(int moduleID);
int  setupAffinity(int moduleID);