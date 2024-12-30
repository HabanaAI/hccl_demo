CC = g++
MPI_FLAG =

ifeq ($(MPI),1)
    $(info Compiling HCCL demo with MPI)
    CC = mpic++
    MPI_FLAG = -D MPI_ENABLED=1
endif

make:
	$(CC) -std=gnu++0x $(MPI_FLAG) -I/usr/include/habanalabs -I${SPDLOG_ROOT} -Wall \
        -o hccl_demo hccl_demo.cpp affinity.cpp env.cpp send_recv.cpp scale_validation.cpp \
        -D AFFINITY_ENABLED=1 -L/usr/lib/habanalabs/ -lSynapse -lpthread

dev:
	$(CC) -std=gnu++0x $(MPI_FLAG) -I${HCL_ROOT}/include/ -I${SYNAPSE_ROOT}/include/ -I${SPDLOG_ROOT} \
        -g -Wall -o hccl_demo hccl_demo.cpp affinity.cpp env.cpp send_recv.cpp scale_validation.cpp -D AFFINITY_ENABLED=1  \
        -L${BUILD_ROOT_LATEST}/ -lSynapse -lpthread

clean:
	rm -f hccl_demo
