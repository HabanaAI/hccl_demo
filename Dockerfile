# we use image which already have hcl/synapse setup
# image originally taken from: artifactory-kfs.habana-labs.com/docker-local/${_synapse_version}/ubuntu${_os_version}/habanalabs/tensorflow-installer-tf-cpu-${_tf_version}:${_synapse_version}-${_synapse_build}
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

#Install Synapse runtime packages - if required
RUN echo "deb https://vault.habana.ai/artifactory/debian `lsb_release -c | awk '{print $2}'` main" > /etc/apt/sources.list.d/artifactory.list
RUN apt-key add /root/tests/hccl_demo/vault.key
RUN apt-get update

WORKDIR /root/tests/hccl_demo
