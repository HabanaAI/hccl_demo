# HCCL Demo
HCCL demo is a program demostrating HCCL usage.<br />
HCCL demo supports communication via Gaudi scaleout or host scaleout (TCP and OFI).<br />
Supported collective communication:
1. All_reduce
2. All_gather
3. Reduce_scatter
4. Broadcast

Supported point to point communication:
1. Send/Recv

## Contents
1. C++ project with all tests and makefile
2. Python wrapper which builds and runs the tests on multiple processes according to the number of devices

## Licensing
Copyright (c) 2022 Habana Labs, Ltd.<br />
SPDX-License-Identifier: Apache-2.0

## Build
Building and cleaning of the project is handled by the Python wrapper.<br />
Alternatively, it can also be built by running the 'make' command.

## Python wrapper arguments
    --nranks         - int, Number of ranks participating in the demo
    --ranks_per_node - int, Number of ranks participating in the demo for current node
    --node_id        - int, ID of the running host. Each host should have unique id between 0-num_nodes
    --test           - str, Which hccl test to run (for example: broadcast/all_reduce) (default: broadcast)
    --size           - str, Data size in units of G,M,K,B or no unit (default: 33554432)
    --loop           - int, Number of iterations (default: 10)
    --broadcast_root - int, Index of root rank for broadcast test
    --csv_path       - str, Path to a file for results output
    -clean           - Clear old executable and compile a new one
    -l               - Display a list of available tests

## Environment variables
    HCCL_COMM_ID     - IP of node_id=0 host and an available port, in the format <IP:PORT>
    SOCKET_NTHREADS  - Number of threads to manage TCP sockets
    NSOCK_PERTHREAD  - Number of sockets per thread
    HCCL_OVER_TCP    - 1 to use TCP between servers, 0 to use scaleout nics
    HCCL_OVER_OFI    - 1 to use OFI between servers, 0 to use scaleout nics

## Run
    HCCL_COMM_ID=<IP:PORT> HCCL_OVER_TCP={0,1} ./run_hcl_demo.py [options]

## Results
Results are printed to the display<br />
Results can also be printed to output file by using --csv_path <path_to_file>

## Examples
Run on 1 server (8 Gaudi devices)

    HCCL_COMM_ID=127.0.0.1:5555 HCCL_OVER_TCP=1 python3 run_hccl_demo.py --nranks 8 --node_id 0

Run on 1 server (8 Gaudi devices) with size 32 MB

    HCCL_COMM_ID=127.0.0.1:5555 HCCL_OVER_TCP=1 python3 run_hccl_demo.py --nranks 8 --node_id 0 --size 32m --test all_reduce
    HCCL_COMM_ID=127.0.0.1:5555 HCCL_OVER_TCP=1 python3 run_hccl_demo.py --nranks 8 --node_id 0 --size 32M --test all_reduce
    HCCL_COMM_ID=127.0.0.1:5555 HCCL_OVER_TCP=1 python3 run_hccl_demo.py --nranks 8 --node_id 0 --size 33554432 --test all_reduce

Run on 2 servers (16 Gaudi devices)

Server 1:

    HCCL_COMM_ID=10.128.11.84:9696 HCCL_OVER_TCP=0 python3 run_hccl_demo.py --nranks 16 --ranks_per_node 8 --node_id 0
    
Server 2:

    HCCL_COMM_ID=10.128.11.84:9696 HCCL_OVER_TCP=1 python3 run_hccl_demo.py --nranks 16 --ranks_per_node 8 --node_id 1

## Results Output Examples

### One Server run

Command:

    HCCL_COMM_ID=10.127.0.0.1:5555 HCCL_OVER_TCP=0 python3 run_hccl_demo.py --test all_reduce --nranks 4 --loop 1000 --node_id 0 --size 536870912 --ranks_per_node 4

Outputs:

    Allreduce hccl_rank=2 size=536870912 <float> Input Buffer [2 6 10 14 ...] reduced to Output Buffer [6 22 38 54 ...] which is fine.
    Allreduce hccl_rank=0 size=536870912 <float> Input Buffer [0 4 8 12 ...] reduced to Output Buffer [6 22 38 54 ...] which is fine.
    Allreduce hccl_rank=1 size=536870912 <float> Input Buffer [1 5 9 13 ...] reduced to Output Buffer [6 22 38 54 ...] which is fine.
    Allreduce hccl_rank=3 size=536870912 <float> Input Buffer [3 7 11 15 ...] reduced to Output Buffer [6 22 38 54 ...] which is fine.
    #################################################################################
    [BENCHMARK] hcclAllReduce(src!=dst, count=134217728, dtype=fp32, iterations=1000)
    [BENCHMARK]     Bandwidth     : 69214.565 MB/s
    #################################################################################

### Two servers run

First server command:
    
    HCCL_COMM_ID=10.127.0.0.1:5555 HCCL_OVER_TCP=0 python3 run_hccl_demo.py --test reduce_scatter --nranks 8 --loop 1000 --node_id 0 --size 33554432 --ranks_per_node 4
   
Second server command:

    HCCL_COMM_ID=10.127.0.0.1:5555 HCCL_OVER_TCP=0 python3 run_hccl_demo.py --test reduce_scatter --nranks 8 --loop 1000 --node_id 0 --size 33554432 --ranks_per_node 4

First server output:
    
    ReduceScatter hccl_rank=2 size=33554432 <float> Input Buffer [2 10 18 26 ...] reduced to Output Buffer [348 412 476 540 ...] which is fine.
    ReduceScatter hccl_rank=1 size=33554432 <float> Input Buffer [1 9 17 25 ...] reduced to Output Buffer [604 668 732 796 ...] which is fine.
    ReduceScatter hccl_rank=3 size=33554432 <float> Input Buffer [3 11 19 27 ...] reduced to Output Buffer [92 156 220 284 ...] which is fine.
    ReduceScatter hccl_rank=0 size=33554432 <float> Input Buffer [0 8 16 24 ...] reduced to Output Buffer [28 92 156 220 ...] which is fine.
    ###################################################################################
    [BENCHMARK] hcclReduceScatter(src!=dst, count=8388608, dtype=fp32, iterations=1000)
    [BENCHMARK]     Bandwidth     : 49980.655 MB/s
    ###################################################################################

Second server output:
    
    ReduceScatter hccl_rank=6 size=33554432 <float> Input Buffer [6 14 22 30 ...] reduced to Output Buffer [156 220 284 348 ...] which is fine.
    ReduceScatter hccl_rank=5 size=33554432 <float> Input Buffer [5 13 21 29 ...] reduced to Output Buffer [412 476 540 604 ...] which is fine.
    ReduceScatter hccl_rank=7 size=33554432 <float> Input Buffer [7 15 23 31 ...] reduced to Output Buffer [732 796 28 92 ...] which is fine.
    ReduceScatter hccl_rank=4 size=33554432 <float> Input Buffer [4 12 20 28 ...] reduced to Output Buffer [668 732 796 28 ...] which is fine.
