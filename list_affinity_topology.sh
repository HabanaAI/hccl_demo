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
         affinity_print "hl-smi not found, exiting"
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
   file_hl_smi=$temp_dir/hl_smi.txt
   file_configuration_table=$temp_dir/configuration_table.txt
   file_final_output=$NUMA_MAPPING_DIR/.habana_module_topo
}

create_configuration_table()
{
   # Save the entire hl-smi output to file
   hl-smi -L > $file_hl_smi

   # Check that the driver
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
         echo $numa_node | tee -a $file_pcie_numa > /dev/null
      else
         for i in `hl-smi -L|grep "Bus Id"|awk '{print $4}'`; do
            affinity_print "PCIE:"$i", NUMA:"`cat /sys/bus/pci/devices/$i/numa_node`;
         done
         affinity_print "Failed to read numa to PCIe device mapping, aborting"
         exit 1
      fi
   done

   # Append output files
   paste $file_module_id $file_pcie_bus_id $file_pcie_numa | awk ' {print $4,$8,$9}' | sort -k1 > $file_configuration_table
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

      # Get the list of threads
      if [ $numa_node -ge 0 ]; then
         vector=`lscpu --parse | grep ",$numa_node,,"|awk -F"," '{print $1}'`
         echo $vector | tee $NUMA_MAPPING_DIR/.habana_moduleID$module_id > /dev/null
         echo $vector | tee -a $temp_dir/.module > /dev/null
      fi
   done
}

add_thread_list_to_config_table()
{
   # Combine output
   echo "ModID   BusID  NUMA   CPUs: " | tee $file_final_output > /dev/null
   echo "=====   =====  =====  ===== " | tee -a $file_final_output > /dev/null
   paste $file_configuration_table $temp_dir/.module | tee -a $file_final_output > /dev/null
}

clean_up()
{
   # Remove temp dir
   if [ ! -z "$temp_dir" ]; then
      rm -fr $temp_dir
   fi
}

main()
{
   if [ "$OMPI_COMM_WORLD_LOCAL_RANK" == "0" ] || [ "$MPI_ENABLED" == "0" ];
   then
      if [[ -z "${NUMA_MAPPING_DIR}" ]];
      then
         output_path="/tmp/affinity_topology_output"
         export NUMA_MAPPING_DIR="/tmp/affinity_topology_output"
      else
         output_path="${NUMA_MAPPING_DIR}"
      fi
      if [ ! -d $output_path ];
      then
         mkdir -m 777 $output_path
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