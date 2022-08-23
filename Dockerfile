# see build_and_run.sh for details

ARG base_image
FROM ${base_image}

#Install essential apps
RUN apt-get update && apt-get install -y apt-utils build-essential
RUN apt-get install -y curl
ENV DEBIAN_FRONTEND noninteractive


#Copy test files to demo directory
RUN mkdir -p /root/tests/hccl_demo

COPY affinity.cpp /root/tests/hccl_demo
COPY affinity.h /root/tests/hccl_demo
COPY affinity.py /root/tests/hccl_demo
COPY build_demo.sh /root/tests/hccl_demo
COPY hccl_demo.cpp /root/tests/hccl_demo
COPY LICENSE /root/tests/hccl_demo
COPY list_affinity_topology.sh /root/tests/hccl_demo
COPY Makefile /root/tests/hccl_demo
COPY README.md /root/tests/hccl_demo
COPY run_hccl_demo.py /root/tests/hccl_demo
COPY vault.key /root/tests/hccl_demo

#Setup test specific environments
ENV PATH "$PATH:/root/tests/hccl_demo"
ENV PYTHONPATH "$PYTHONPATH:/root/tests/hccl_demo"

#Install Synapse runtime packages - if required
RUN echo "deb https://vault.habana.ai/artifactory/debian `lsb_release -c | awk '{print $2}'` main" > /etc/apt/sources.list.d/artifactory.list
RUN apt-key add /root/tests/hccl_demo/vault.key
RUN apt-get update

WORKDIR /root/tests/hccl_demo
