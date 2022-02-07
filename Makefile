CC = g++

AFFINITY =

ifeq ($(AFFINITY_ENABLED),1)
    $(info Compiling HCCL demo with affinity)
    AFFINITY = affinity.cpp -D AFFINITY_ENABLED=1
endif

make:
	$(CC) -std=gnu++0x -I/usr/include/habanalabs \
        -I${SPDLOG_ROOT} -Wall -o hccl_demo hccl_demo.cpp  \
        -L/usr/lib/habanalabs/ -lSynapse -lpthread

dev:
	$(CC) -std=gnu++0x -I${HCL_ROOT}/include/ -I${SYNAPSE_ROOT}/include/ -I${SPDLOG_ROOT}\
        -g -Wall -o hccl_demo hccl_demo.cpp $(AFFINITY)  \
        -L${BUILD_ROOT_LATEST}/ -lSynapse -lpthread

clean:
	rm -f hccl_demo
