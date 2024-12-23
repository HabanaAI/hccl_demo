# HCCL Demo
HCCL demo is a program that demonstrates HCCL usage and supports communication via Gaudi<br />
based scale-out or Host NIC scale-out.

This README provides HCCL demo setup and usage as well as example run commands. In<br />
addition, it provides further setup steps required when using Host NIC Scale out.<br />
Host NIC Scale out is achieved using OFI. [Host NIC Scale-Out Setup](#Host-NIC-Scale-Out-Setup)<br />
section details the steps required to download, install and build OFI. It also provides<br />
the required environment variables to run Host NIC scale-out with Gaudi Direct. <br />

## Supported Collective Operations
The following lists the supported collective operations:
1. All_reduce
2. All_gather
3. All2All
4. Reduce_scatter
5. Broadcast
6. Reduce

Send/Recv is the supported point to point communication. It illustrates exchanging data between pairs of Gaudis in same box or an outer box, via Gaudi based scale-out or Host NIC scale-out


## Contents
1. C++ project which includes all tests and a makefile
2. Python wrapper which builds and runs the tests on multiple processes according to the number of devices

## Licensing
Copyright (c) 2022 Habana Labs, Ltd.<br />
SPDX-License-Identifier: Apache-2.0

## Build
The Python wrapper builds and cleans the project (for cleaning please use "-clean").<br />
Alternatively, the project can be built using the following command:<br />
```
make
```
For building the project with MPI:<br />
```
MPI=1 make
```
By default, the demo is built with affinity configuration.<br />
When switching between MPI and non MPI modes, please remember to run with "-clean".

## Host NIC Scale-Out Setup
### Download and Install libfabric
libfabric should be downloaded and installed in order to use it.<br />
Please follow the instructions below:<br />
1.  Define required version to be installed:
    ```
    export REQUIRED_VERSION=1.20.0
    ```
2.  Download libfabric tarball from https://github.com/ofiwg/libfabric/releases:
    ```
    wget  https://github.com/ofiwg/libfabric/releases/download/v$REQUIRED_VERSION/libfabric-$REQUIRED_VERSION.tar.bz2 -P /tmp/libfabric
    ```
3.  Store temporary download directory in stack:
    ```
    pushd /tmp/libfabric
    ```
4.  Open the file:
    ```
    tar -xf libfabric-$REQUIRED_VERSION.tar.bz2
    ```
5.  Define libfabric root location:
    ```
    export LIBFABRIC_ROOT=<libFabric library location>
    ```
6.  Create folder for libfabric:
    ```
    mkdir -p ${LIBFABRIC_ROOT}
    ```
7.  Change permissions for libfabric folder:
    ```
    chmod 777 ${LIBFABRIC_ROOT}
    ```
8.  Change directory to libfabric folder created after opening tar file:
    ```
    cd libfabric-$REQUIRED_VERSION/
    ```
9.  Configure libfabric:
    ```
    ./configure --prefix=$LIBFABRIC_ROOT --with-synapseai=/usr
    ```
10. Build and install libfabric:
    ```
    make -j 32 && make install
    ```
11. Remove temporary download directory from stack:
    ```
    popd
    ```
12. Delete temporary download directory:
    ```
    rm -rf /tmp/libfabric
    ```
13. Include LIBFABRIC_ROOT in LD_LIBRARY_PATH:
    ```
    export LD_LIBRARY_PATH=$LIBFABRIC_ROOT/lib:$LD_LIBRARY_PATH
    ```

    Installation can be verified by running: `fi_info --version`.<br />
    For more information please see: https://github.com/ofiwg/libfabric

### Build HCCL OFI wrapper
To use libfabric library, HCCL OFI wrapper should be built.<br />
Please follow the instructions below:<br />
1. Clone wrapper from https://github.com/HabanaAI/hccl_ofi_wrapper:
   ```
   git clone https://github.com/HabanaAI/hccl_ofi_wrapper.git
   ```
2. Define LIBFABRIC_ROOT:
   ```
   export LIBFABRIC_ROOT=/tmp/libfabric-1.20.0
   ```
3. Change directory to hccl_ofi_wrapper:
   ```
   cd hccl_ofi_wrapper
   ```
4. Build wrapper:
   ```
   make
   ```
5. Copy wrapper to /usr/lib/habanalabs/:
   ```
   cp libhccl_ofi_wrapper.so /usr/lib/habanalabs/libhccl_ofi_wrapper.so
   ```
6. Run ldconfig utility:
   ```
   ldconfig
   ```
7. Include libhccl_ofi_wrapper.so location in LD_LIBRARY_PATH:
   ```
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/habanalabs/
   ```

### Gaudi Direct
Gaudi direct (GDR) enables direct fabric access to Gaudi memory.
This mode is supported with Verbs or EFA provider if the following conditions are met:
1. OFI version 1.16.0 (or higher) for EFA and 1.20.0 (or higher) for Verbs
2. Kernel version 5.12 (or higher)
3. The following environment variables are set:
   FI_EFA_USE_DEVICE_RDMA=1 (For AWS EFA)
   RDMAV_FORK_SAFE=1
   MLX5_SCATTER_TO_CQE=0 (For MLX Verbs)
4. PCIe ACS (Access Control) should be disabled

## Python Wrapper Arguments
### General flags
    -h, --help               Show this help message and exit.
    --clean, -clean          Clean previous artifacts including logs, recipe and csv results.
    -list, --list_tests      Display a list of available tests.
    --doc                    Display detailed help for HCCL demo in a form of docstring.
### Setup configuration flags
    --nranks NRANKS          Number of ranks in the communicator.
    --ranks_per_node RANKS_PER_NODE
                             Number of ranks per node (default read from h/w or set by MPI)
    --scaleup_group_size     The scaleup group size per node (default is ranks_per_node)
    --node_id NODE_ID        Box index. Value in the range of (0, NUM_BOXES).
    --mpi, -mpi              Use MPI for managing execution.
### Test control flags
    --test TEST              Specify test (use '-l' option for test list).
    --size N                 Data size in units of G,M,K,B or no unit. Default is Bytes.
    --size_range MIN MAX     Test will run from MIN to MAX, units of G,M,K,B or no unit. Default is Bytes. E.g. --size_range 32B 1M.
    --size_range_inc M       Test will run on all multiplies by 2^size_range_inc from MIN to MAX.
    --loop LOOP              Number of loop iterations.
    --test_root TEST_ROOT
                             Index of root rank for broadcast and reduce tests (optional).
    --ranks_list RANKS_LIST  List of pairs of ranks for send_recv ranks scaleout. E.g. 0,8,1,8 (optional).
    --data_type DATA_TYPE    Data type, float or bfloat16. Default is float.
    --custom_comm CUSTOM_COMM
                             List of HCCL process that will open a communicator.
    --no_correctness         Skip correctness validation.
    --reduction_op           <sum|min|max> (default=sum)
### Logging flags
    --csv_path CSV_PATH      Path to a file for results output (optional).
    --ignore_mpi_errors, -ignore_mpi_errors
                             Ignore generic MPI errors.
    --no_color, -no_color
                             Disable colored output in terminal.
    --data_csv, -data_csv
                             Creates 2 csv file for each rank, one for data input and second for data output.


## Environment Variables
    HCCL_COMM_ID     - IP of node_id=0 host and an available port, in the format <IP:PORT>


## Run HCCL Demo

Set the below when using any operating system that has Linux kernel version between 5.9.x and 5.16.x. Currently, this is applicable to Ubuntu20 and Amazon Linux AMIs:

    echo 0 > /proc/sys/kernel/numa_balancing

Run the execution command

    HCCL_COMM_ID=<IP:PORT> ./run_hccl_demo.py [options]

## Results
Results are printed to the display<br />
Results per rank can also be printed to output file by using --csv_path <path_to_file>

## Examples - without MPI

**Note**: The following examples are applicable for Gaudi based and Host NIC scale-out.

### Running HCCL on 1 server (8 Gaudi devices)

Configuration: One server with 8 ranks, 32 MB size, all_reduce collective, 1000 iterations

    HCCL_COMM_ID=127.0.0.1:5555 python3 run_hccl_demo.py --nranks 8 --node_id 0 --size 32m --test all_reduce --loop 1000 --ranks_per_node 8

Output example:

    ###############################################################################
    [BENCHMARK] hcclAllReduce(src!=dst, count=8388608, dtype=float, iterations=1000)
    [BENCHMARK]     NW Bandwidth   : <Test results> GB/s
    [BENCHMARK]     Algo Bandwidth : <Test results> GB/s
    ###############################################################################

Different options for running one server with 8 ranks and size of 32 MB:

    HCCL_COMM_ID=127.0.0.1:5555 python3 run_hccl_demo.py --nranks 8 --node_id 0 --size 32m --test all_reduce
    HCCL_COMM_ID=127.0.0.1:5555 python3 run_hccl_demo.py --nranks 8 --node_id 0 --size 32M --test all_reduce
    HCCL_COMM_ID=127.0.0.1:5555 python3 run_hccl_demo.py --nranks 8 --node_id 0 --size 33554432 --test all_reduce
    HCCL_COMM_ID=127.0.0.1:5555 python3 run_hccl_demo.py --nranks 8 --node_id 0 --size 33554432b --test all_reduce
    HCCL_COMM_ID=127.0.0.1:5555 python3 run_hccl_demo.py --nranks 8 --node_id 0 --size 33554432B --test all_reduce
### Running HCCL demo on 2 servers (16 Gaudi devices)

Configuration: 16 ranks, 32 MB size, all_reduce collective, 1000 iterations

First server command:

    HCCL_COMM_ID=10.111.12.234:5555 python3 run_hccl_demo.py --test all_reduce --nranks 16 --loop 1000 --node_id 0 --size 32m --ranks_per_node 8

Second server command:

    HCCL_COMM_ID=10.111.12.234:5555 python3 run_hccl_demo.py --test all_reduce --nranks 16 --loop 1000 --node_id 1 --size 32m --ranks_per_node 8

First server output:

    ###############################################################################
    [BENCHMARK] hcclAllReduce(src!=dst, count=8388608, dtype=float, iterations=1000)
    [BENCHMARK]     NW Bandwidth     : <Test results> GB/s
    [BENCHMARK]     Algo Bandwidth   : <Test results> GB/s
    ###############################################################################

### Running HCCL with size range on 1 server (8 Gaudi devices)

Configuration: One server with 8 ranks, size range 32B to 1 MB, all_reduce collective, 1 iteration

    HCCL_COMM_ID=127.0.0.1:5555 python3 run_hccl_demo.py --nranks 8 --node_id 0 --size_range 32b 1m --test all_reduce --loop 1 --ranks_per_node 8

Output example:

    ################################################
    [SUMMARY REPORT]
    (src!=dst, collective=all_reduce, iterations=1)

    size          count         type          redop         time          algo_bw       nw_bw
    (B)           (elements)                                (us)          (GB/s)        (GB/s)
    32            8             float         sum           <time>        <bandwidth>   <bandwidth>
    64            16            float         sum           <time>        <bandwidth>   <bandwidth>
    128           32            float         sum           <time>        <bandwidth>   <bandwidth>
    256           64            float         sum           <time>        <bandwidth>   <bandwidth>
    512           128           float         sum           <time>        <bandwidth>   <bandwidth>
    1024          256           float         sum           <time>        <bandwidth>   <bandwidth>
    2048          512           float         sum           <time>        <bandwidth>   <bandwidth>
    4096          1024          float         sum           <time>        <bandwidth>   <bandwidth>
    8192          2048          float         sum           <time>        <bandwidth>   <bandwidth>
    16384         4096          float         sum           <time>        <bandwidth>   <bandwidth>
    32768         8192          float         sum           <time>        <bandwidth>   <bandwidth>
    65536         16384         float         sum           <time>        <bandwidth>   <bandwidth>
    131072        32768         float         sum           <time>        <bandwidth>   <bandwidth>
    262144        65536         float         sum           <time>        <bandwidth>   <bandwidth>
    524288        131072        float         sum           <time>        <bandwidth>   <bandwidth>
    1048576       262144        float         sum           <time>        <bandwidth>   <bandwidth>

## Examples - MPI mode

**Note**: The following examples are applicable for Gaudi based and Host NIC scale-out.

### Running HCCL on 1 server (8 Gaudi devices)

All available MPI options are supported.<br />
* For MPI different running options please refer to: https://www.open-mpi.org/faq/?category=running#mpirun

Configuration: One server with 8 ranks, 32 MB size, all_reduce collective, 1000 iterations

    python3 run_hccl_demo.py --size 32m --test all_reduce --loop 1000 -mpi -np 8

Output example:

    ###############################################################################
    [BENCHMARK] hcclAllReduce(src!=dst, count=8388608, dtype=float, iterations=1000)
    [BENCHMARK]     NW Bandwidth     : <Test results> GB/s
    [BENCHMARK]     Algo Bandwidth   : <Test results> GB/s
    ###############################################################################

### Running HCCL demo on 2 servers (16 Gaudi devices)

Configuration: 16 ranks, 32 MB size, all_reduce collective, 1000 iterations

First option using MPI hostfile:

    python3 run_hccl_demo.py --test all_reduce --loop 1000 --size 32m -mpi --hostfile hostfile.txt

* For MPI --hostfile option, please refer to: https://www.open-mpi.org/faq/?category=running#mpirun-hostfile

Second option using MPI host:

    python3 run_hccl_demo.py --test all_reduce --loop 1000 --size 32m -mpi --host 10.111.12.234:8,10.111.12.235:8

* For MPI --host option, please refer to: https://www.open-mpi.org/faq/?category=running#mpirun-host

First server output:

    ###############################################################################
    [BENCHMARK] hcclAllReduce(src!=dst, count=8388608, dtype=float, iterations=1000)
    [BENCHMARK]     NW Bandwidth     : <Test results> GB/s
    [BENCHMARK]     Algo Bandwidth   : <Test results> GB/s
    ###############################################################################

### Running HCCL demo with custom communicator:

Running on 1 server:

    Configuration: One server with 8 ranks, 32 MB size, all_reduce collective, 1000 iterations, communicator includes only ranks 0 and 1:

        HCCL_COMM_ID=127.0.0.1:5555 python3 run_hccl_demo.py --nranks 8 --node_id 0 --size 32m --test all_reduce --loop 1 --ranks_per_node 8 --custom_comm 0,1

Running on 2 servers with MPI (16 Gaudi devices):

    * Note: When defining custom communicator, for each rank in the communicator we should have at least one more rank included that is a peer to the first one.
    * In the following examaples we used MPI hostfile, using MPI host is good as well.

    Configuration: 16 ranks, 32 MB size, all_reduce collective, 1000 iterations, communicator includes only ranks 0 and 8:

        python3 run_hccl_demo.py --test all_reduce --loop 1000 --size 32m --custom_comm 0,8 -mpi --hostfile hostfile.txt

Running on 2 servers without MPI (16 Gaudi devices):

    Configuration: 16 ranks, 32 MB size, all_reduce collective, 1000 iterations, communicator includes only ranks 0,1,8,9:

        First node:
        HCCL_COMM_ID=10.111.12.234:5555 python3 run_hccl_demo.py --test all_reduce --nranks 16 --loop 1000 --node_id 0 --custom_comm 0,1,8,9

        Second node:
        HCCL_COMM_ID=10.111.12.234:5555 python3 run_hccl_demo.py --test all_reduce --nranks 16 --loop 1000 --node_id 1 --custom_comm 0,1,8,9
