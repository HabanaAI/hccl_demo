#!/bin/bash

# ******************************************************************************
# Copyright (c) 2022 Habana Labs, Ltd.
# SPDX-License-Identifier: Apache-2.0
# ******************************************************************************

# Description
# This script outputs a file for each moduleID.
# These files contain the Hthread_sequence to which the process is bound (this is a restriction and not a reservation).
# This is achieved by getting the mapping of (ModuleID, pcie_bus_id) from hl-smi
# A mapping is performed, where 2 tuple is mapped to a numa by opening the file
# /sys/bus/pci/devices/<pcie_bus_id >/numa_node
# At this point, there are 3 tuple (ModuleID, pcie_bus_id, numa_node)
# Lastly the Hthread_sequence that correspond to that numa_node is received from lscpu for achieving:
# (ModuleID, pcie_bus_id,  numa_node, Hthread_sequence)
# The Hthread_sequence is then used to bind the process to the specific threads on the numa closest to the PCIE bus.

affinity_print()
{
   echo "Affinity: "$1
}

hl_smi_check()
{
   if [[ ! -x `which hl-smi` ]];
   then
         affinity_print "hl-smi could not be found, exiting"
         exit 1
   fi
}

check_env()
{
   if [[ -z "$NUMA_MAPPING_DIR" ]];
   then
         affinity_print "Missing env variable \"NUMA_MAPPING_DIR\", exiting!"
         exit 1
   fi
}

create_temp_files()
{
   # Create a temp directory, mktemp is used to create a temp directory with a unique name
   temp_dir=$(mktemp -d)

   # Create temp files for holding output
   file_hl_smi=$temp_dir/hl_smi.txt
   file_module_id=$temp_dir/module_id.txt
   file_pcie_bus_id=$temp_dir/pcie_bus_id.txt
   file_pcie_numa=$temp_dir/pcie_numa.txt
   file_cpus=$temp_dir/cpus.txt
   file_configuration_table=$temp_dir/configuration_table.txt
   file_final_output=$NUMA_MAPPING_DIR/.habana_module_topo

   echo > $file_cpus
}

# Function to expand ranges of core isolation
expand_ranges() {
    echo "$1" | tr ',' '\n' | while read range; do
        if [[ $range == *-* ]]; then
            seq $(echo $range | awk -F'-' '{print $1, $2}')
        else
            echo "$range"
        fi
    done
}

create_configuration_table()
{
   # Save the entire hl-smi output to file
   hl-smi -L > $file_hl_smi

   # Check that the driver is up
   if [ $? -eq 1 ]; then
      affinity_print "Issue while trying to run hl-smi, aborting..."
      exit 1
   fi

   # Get module IDs (unique identifier for each Gaudi)
   grep "Module ID" $file_hl_smi > $file_module_id

   # Get bus IDs
   grep "Bus Id" $file_hl_smi > $file_pcie_bus_id

   # Get the numa for each PCIE bus
   for i in `cat $file_pcie_bus_id|awk '{print $4}'`; do
      numa_node=`cat /sys/bus/pci/devices/$i/numa_node`
      if [ $numa_node -ge 0 ]; then
         echo $numa_node >> $file_pcie_numa
      else
      for i in `hl-smi -L|grep "Bus Id"|awk '{print $4}'`; do
          affinity_print "PCIE:"$i", NUMA:"`cat /sys/bus/pci/devices/$i/numa_node`;
       done
       affinity_print "Numa mapping is not set properly, most likely you are using an unsupported VM, aborting affinity setting"
       exit 1
      fi
   done

   # Append output files
   paste $file_module_id $file_pcie_bus_id $file_pcie_numa | awk ' {print $4,$8,$9}' | sort -k1 > $file_configuration_table
}

mask_cpu()
{
    grep "gaudi2" $file_hl_smi > /dev/null
    if [ $? == 0 ]; then
        # Place cpuid hyperthread in mask file
        cpuid=$1

        siblings=$(cat /sys/devices/system/cpu/cpu${cpuid}/topology/thread_siblings_list | tr , " " | tr "-" " ")
        for siblings in $siblings; do
            echo $siblings >> $file_cpus
        done

        cat $file_cpus | sort | uniq > /tmp/tmp
        mv /tmp/tmp $file_cpus
    fi
}

create_thread_list()
{
   no_of_numa_nodes=`lscpu|grep "NUMA node(s):"|awk '{print $3}'`
   no_of_gaudis=`cat $file_configuration_table|wc -l`
   no_of_used_numa=`cat $file_pcie_numa | uniq | wc -l`

   for module_id in $(seq 0 $(($no_of_gaudis-1))); do
      # Grab one PCIE id at a time (busID)
      pcie_bus_id=`cat $file_configuration_table | awk '{print $2}' | sed -n $(($module_id+1))p`

      # Get the corresponding numa node (pcie_numa)
      numa_node=`cat /sys/bus/pci/devices/$pcie_bus_id/numa_node`
      no_of_sockets=`lscpu |grep Socket|cut -d ":" -f 2`
      no_of_numanode=`lscpu |grep NUMA|head -1|cut -d ":" -f 2`
      numa_per_socket=$((no_of_numanode/no_of_sockets))

      # Get the list of threads for the main processes
      if [ $numa_node -ge 0 ]; then
         # Read isolated cores and expand them
         isolated_cores=$(expand_ranges "$(cat /sys/devices/system/cpu/isolated)")

         # Get physical cores (excluding hyper threaded ones)
         physical_cores=$(lscpu --parse=CORE,CPU,SOCKET,NODE | grep -v '^#' | awk -F',' '!seen[$1]++ {print $2}')

         cpulist_vector=`lscpu --parse | grep ",$numa_node,,"|awk -F"," '{print $1}'`
         # check if sub numa clustering is enabled
         if [ $no_of_numanode -gt $no_of_sockets ]; then
                 numa_node=$((numa_node/numa_per_socket))
                 cpulist_vector=`lscpu -e=CPU,SOCKET| grep -v SOCKET| awk ' { printf("%d,%d,%d,%d,,\n",$1,$1,$2,$2); }'| grep ",$numa_node,,"`
         fi

         # Filter NUMA node cores to keep only physical cores that are isolated
         filtered_vector=$(for core in $cpulist_vector; do
            # Check if isolated_cores is empty
            if [[ -z "$isolated_cores" ]]; then
               # If isolated_cores is empty, consider all cores as isolated
               if echo "$physical_cores" | grep -q "^$core$"; then
                     echo "$core"
               fi
            else
               # Proceed with normal filtering if isolated_cores is not empty
               if echo "$isolated_cores" | grep -q "^$core$" && echo "$physical_cores" | grep -q "^$core$"; then
                     echo "$core"
               fi
            fi
         done)

         # Convert filtered_vector into an array
         filtered_array=($(echo "$filtered_vector"))

         # Calculate the key_factor
         num_cores=${#filtered_array[@]}
         key_factor=$((num_cores / ((no_of_gaudis)/no_of_numa_nodes)))
         start_index=$(( ((module_id % ((no_of_gaudis)/no_of_numa_nodes)) - 1) * key_factor ))
         end_index=$(( start_index + key_factor - 1 ))

         # Extract cores for process_x
         process_cores=("${filtered_array[@]:$start_index:$key_factor}")
         echo ${process_cores[@]} >> $temp_dir/.habana_moduleID$module_id
         echo ${process_cores[@]} >> $temp_dir/.module

         if [ -f $temp_dir/.habana_moduleID$module_id ]; then
             cat $temp_dir/.habana_moduleID$module_id | tr '\n' ' ' > $NUMA_MAPPING_DIR/.habana_moduleID$module_id
         fi
      fi
   done
}

add_thread_list_to_config_table()
{
   # Combine output
   echo "ModID   BusID  NUMA   CPUs: " > $file_final_output
   echo "=====   =====  =====  ===== " >> $file_final_output
   paste $file_configuration_table $temp_dir/.module >> $file_final_output
}

clean_up()
{
   # Remove temporary directory
   if [ ! -z "$temp_dir" ]; then
      rm -fr $temp_dir
   fi
}

main()
{
   export HWLOC_HIDE_ERRORS=1
   if [ "$OMPI_COMM_WORLD_LOCAL_RANK" == "0" ] || [ "$MPI_ENABLED" == "0" ];
   then
      if [[ -z "${NUMA_MAPPING_DIR}" ]];
      then
         output_path="/tmp/affinity_topology_output"
         export NUMA_MAPPING_DIR="/tmp/affinity_topology_output"
      else
         affinity_print "Numa mapping directory: ${NUMA_MAPPING_DIR}"
         output_path="${NUMA_MAPPING_DIR}"
      fi
      if [ ! -d $output_path ];
      then
         mkdir -p -m 777 $output_path
      fi
      output_file="${output_path}/.habana_moduleID0"
      if [ ! -f $output_file ];
      then
         echo "Affinity: Script has not been executed before, going to execute..."
         hl_smi_check
         create_temp_files
         create_configuration_table
         create_thread_list
         add_thread_list_to_config_table
         clean_up
         affinity_print "Script has finished successfully"
      else
         echo "Affinity: Script has been executed before."
      fi
   fi
   exit 0
}

main