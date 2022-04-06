#!/bin/bash

# ******************************************************************************
# Copyright (c) 2022 Habana Labs, Ltd.
# SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

if [ "$MPI_ENABLED" == "1" ]; then
   if [ "$OMPI_COMM_WORLD_LOCAL_RANK" == "0" ]; then
      MPI=1 make $1
      echo "Build script with MPI has finished successfully"
   fi
else
   make $1
   echo "Build script has finished successfully"
fi
exit 0