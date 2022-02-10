#!/usr/bin/env python3

import os, subprocess

def create_affinity_files():
    # In case affinity configuration is disabled, do nothing
    if 'DISABLE_PROC_AFFINITY' in os.environ and int(os.environ['DISABLE_PROC_AFFINITY']):
        return
    print("Affinity: Creating affinity files...")
    file_name = "list_affinity_topology.sh"
    run_script_cmd = f"bash {file_name}"

    # Return if the script does not exist
    if not os.path.isfile(file_name):
        print(f"Affinity: Could not find {file_name}")
        return

    # Set the output directory for the moduleID<->numa mapping
    output_path = os.environ.get('NUMA_MAPPING_DIR', "/tmp/affinity_topology_output")
    os.environ['NUMA_MAPPING_DIR'] = output_path

    # Check if the script has already been executed successfully in the past
    if not os.path.exists(os.path.join(output_path, ".habana_moduleID0")):
        print("Affinity: Script has not been executed before, going to execute...")

        # Create the output directory if it does not exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Run the script
        process = subprocess.Popen(run_script_cmd, shell=True)
        process.wait()
