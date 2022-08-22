#!/bin/bash

cur_dir=$(pwd)
script_name=$(basename ${BASH_SOURCE[0]})
work_dir=$(realpath ${BASH_SOURCE[0]%/*})
_dockerfile_path=${work_dir}
_container_name='latest'
_prefix_image_name='lamatriz'
_interactive=0
_push=0
_run=0

function help() {
    echo
    echo " This script will build, deploy container based on current Dockerfile"
    echo "     ${script_name} [OPTIONS]"
    echo "        -df | --dockerfile_path <path to dockerfile>"
    echo "        -c  | --container <container_name>"
    echo "        -i  | --interactive"
    echo "        -r  | --run"
    echo "        -p  | --push"
    echo
}

while [ -n "$1" ];
do
    case $1 in
        -c | --container)
            shift
            _container_name=$1
            ;;
        -df | --dockerfile_path)
            shift
            _dockerfile_path=$1
            ;;
        -h  | --help)
            help
            exit 0
            ;;
        -i  | --interactive)
            shift
            _interactive=1
            ;;
        -r  | --run)
	    shift
	    _run=1
	    ;;
	-p  | --push)
	    shift
	    _push=1
	    ;;
        * )
            echo "The parameter $1 is not allowed"
            echo
            help
            exit 1
            ;;
    esac
    shift
done

# base image taken directly from artifactory and pushed to hub.docker with most apps installed from Dockerfile
_base_image="lamatriz/pytorch:1.12.0-1.6.0-439-deepspeed"
_log_dir=/tmp/hccl_demo_logs
rm -rf ${_log_dir}
mkdir ${_log_dir}

source build_docker.func

echo "Build Docker image."
echo "docker_build ${_prefix_image_name} ${_base_image} ${_log_dir} ${_dockerfile_path} ${_interactive} ${_container_name}"

docker_build ${_prefix_image_name} ${_base_image} ${_log_dir} ${_dockerfile_path} ${_interactive} ${_container_name}
printf 'hccl_demo logs: %s\n' "${_log_dir}"

if [[ ${_push} -eq 1 ]]
then
  echo "Push the container."
  docker_push ${_prefix_image_name} ${_log_dir} ${_container_name}
else
  echo "Build Only!"
fi
