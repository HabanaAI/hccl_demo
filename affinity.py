#!/usr/bin/env python3

import os, subprocess

def create_affinity_files(mpi, mpi_args):
    try:
        # In case affinity configuration is disabled, do nothing
        if 'DISABLE_PROC_AFFINITY' in os.environ and int(os.environ['DISABLE_PROC_AFFINITY']):
            return 0
        print("Affinity: Creating affinity files...")
        file_name = "list_affinity_topology.sh"

        # Return if the script does not exist
        if not os.path.isfile(file_name):
            print(f"Affinity: Could not find {file_name}")
            return

        # Set the output directory for the moduleID<->numa mapping
        output_path = os.environ.get('NUMA_MAPPING_DIR', "/tmp/affinity_topology_output")
        os.environ['NUMA_MAPPING_DIR'] = output_path

        if mpi:
            # Generate MPI command line for running bash script
            cmd = generate_mpi_cmd(mpi_args, file_name)
        else:
            cmd = f"MPI_ENABLED=0 bash {file_name}"
        print("Affinity: Running " + str(cmd))

        # Run the script
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
        return_code = process.poll()
        return return_code

    except Exception as e:
        print("Affinity: create_affinity_files function failed with exception: " + str(e))
        raise e

def generate_mpi_cmd(mpi_args, file_name):
    mpi = get_mpi_prefix()
    cmd = mpi + " -x MPI_ENABLED=1 " + ' '.join(mpi_args) + " bash " + file_name
    print("Affinity: Running " + str(file_name) + " with MPI")
    return cmd

def get_mpi_prefix():
    result = subprocess.run(['which', 'mpirun'], stdout=subprocess.PIPE)
    return str(result.stdout.decode('utf-8').strip())

