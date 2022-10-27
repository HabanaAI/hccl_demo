#!/bin/bash

# ******************************************************************************
# Copyright (c) 2022 Habana Labs, Ltd.
# SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

if [ "$MPI_ENABLED" == "1" ]; then
   if [ "$OMPI_COMM_WORLD_LOCAL_RANK" == "0" ]; then
      echo "Building HCCL demo with MPI"
      MPI=1 make $1
   fi
else
   echo "Building HCCL demo"
   make $1
fi
exit $?