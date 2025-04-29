#!/bin/bash



#Description
#This script outputs a file for each moduleID.
#These files contain the Hthread_sequence on which the process is bound too (this is a restriction and not a reservation).
#We do this by getting the mapping of (ModuleID, pcie_bus_id) from hl-smi
#Then we map the 2 tuple to a numa by opening the file
#/sys/bus/pci/devices/<pcie_bus_id >/numa_node
#Now we have a 3 tuple (ModuleID, pcie_bus_id,  numa_node)
#Lastly we get the Hthread_sequence that correspond to that numa_node from lscpu so now we have
#(ModuleID, pcie_bus_id,  numa_node, Hthread_sequence )
#The Hthread_sequence is then used to bind the process to the specific threads on the numa closest to the PCIE bus.
#set -x
declare -A AFFINITY_ENUM=(
    [PHYSICAL]=0x1
    [NUMAS]=0x10
    [ISOLATION]=0x100
)
export AFFINITY_LEVEL
AFFINITY_LEVEL=0
if [[ -z "$MOCKUP_FILES_LOCATION" ]];
then
    MOCKUP_FILES_LOCATION=$HCL_ROOT/tests/affinity
fi
HL_SMI_CMD="hl-smi -L"
LSCPU_CMD="lscpu"
LSCPU_PARSE_CMD="lscpu --parse"
LSCPU_PARSE_EQ_CMD="lscpu --parse=CORE,CPU,SOCKET,NODE"
ISOLATED_CMD="cat /sys/devices/system/cpu/isolated"
if [[ $USE_G2_MOCKUP_FILES == 1 ]]; then
    NUMA_MAPPING_DIR=$NUMA_MAPPING_DIR/MOCKUP_RESULT/g2
    HL_SMI_CMD="cat $MOCKUP_FILES_LOCATION/mockup_files/g2/hl_smi_g2.txt"
    LSCPU_CMD="cat $MOCKUP_FILES_LOCATION/mockup_files/g2/lscpu_g2.txt"
    LSCPU_PARSE_CMD="cat $MOCKUP_FILES_LOCATION/mockup_files/g2/lscpu_parse_g2.txt"
    LSCPU_PARSE_EQ_CMD="cat $MOCKUP_FILES_LOCATION/mockup_files/g2/lscpu_parse_eq_g2.txt"
    ISOLATED_CMD="cat $MOCKUP_FILES_LOCATION/mockup_files/g2/isolated_g2.txt"
    # Mockup files for G2
fi
if [[ $USE_G3_MOCKUP_FILES == 1 ]]; then
    NUMA_MAPPING_DIR=$NUMA_MAPPING_DIR/MOCKUP_RESULT/g3
    HL_SMI_CMD="cat $MOCKUP_FILES_LOCATION/mockup_files/g3/hl_smi_g3.txt"
    LSCPU_CMD="cat $MOCKUP_FILES_LOCATION/mockup_files/g3/lscpu_g3.txt"
    LSCPU_PARSE_CMD="cat $MOCKUP_FILES_LOCATION/mockup_files/g3/lscpu_parse_g3.txt"
    LSCPU_PARSE_EQ_CMD="cat $MOCKUP_FILES_LOCATION/mockup_files/g3/lscpu_parse_eq_g3.txt"
    ISOLATED_CMD="cat $MOCKUP_FILES_LOCATION/mockup_files/g3/isolated_g3.txt"
    # Mockup files for G3
fi

affinity_print()
{
   echo "Affinity: "$1
}

hl_smi_check()
{
   if [[ ! -x `which hl-smi` && $USE_G3_MOCKUP_FILES -ne 1 && $USE_G2_MOCKUP_FILES -ne 1 ]]; then
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
   elif [[ ! -d "$NUMA_MAPPING_DIR" ]];
   then
         affinity_print "NUMA_MAPPING_DIR does not exist! creating..."
         mkdir -p $NUMA_MAPPING_DIR
         if [[ $? -ne 0 ]];
         then
            affinity_print "Failed to create NUMA_MAPPING_DIR, exiting!"
            exit 1
         fi
   fi
}

create_temp_files()
{
   # create a temp directory, mktemp is used to create a temp directory with a unique name
   temp_dir=$(mktemp -d)

   # create temp files for holding outputs
   file_hl_smi=$temp_dir/hl_smi.txt
   file_module_id=$temp_dir/module_id.txt
   file_pcie_bus_id=$temp_dir/pcie_bus_id.txt
   file_pcie_numa=$temp_dir/pcie_numa.txt
   file_hl_smi=$temp_dir/hl_smi.txt
   file_configuration_table=$temp_dir/configuration_table.txt
   file_final_output=$NUMA_MAPPING_DIR/.habana_module_topo
   file_final_class_output=$NUMA_MAPPING_DIR/.habana_module_affinity_classification
}

create_configuration_table()
{
   # save the entire hl-smi output to file
   $HL_SMI_CMD > $file_hl_smi

   #check that the driver is up
   if [ $? -eq 1 ]; then
      affinity_print "Issue while trying to run hl-smi, aborting..."
      exit 1
   fi

   # get the module IDs (unique identifier for each gaudi)
   grep "Module ID" $file_hl_smi > $file_module_id

   # get the bus IDs
   grep "Bus Id" $file_hl_smi > $file_pcie_bus_id

   # Get the numa for each PCIE bus
index=0
for i in $(awk '{print $4}' "$file_pcie_bus_id"); do
    if [ -f "/sys/bus/pci/devices/$i/numa_node" ]; then
        numa_node=$(cat /sys/bus/pci/devices/$i/numa_node)
    else
        numa_node=""
    fi

    if [[ -n "$numa_node" && $numa_node -ge 0 ]]; then
        echo "$numa_node" >> "$file_pcie_numa"
    elif [[ $USE_G3_MOCKUP_FILES == 1 || $USE_G2_MOCKUP_FILES == 1 ]]; then
        echo $((index % 2)) >> "$file_pcie_numa"
    else
        for bus in $($HL_SMI_CMD | grep "Bus Id" | awk '{print $4}'); do
            # Redirect errors if the file does not exist
            affinity_print "PCIE:$bus, NUMA:"$(cat /sys/bus/pci/devices/$bus/numa_node 2>/dev/null)
        done
        affinity_print "Numa mapping isn't set properly, you are most likely running on an unsupported VM, aborting..."
        exit 1
    fi
    index=$((index + 1))
done


   #append output files horizontally
   paste $file_module_id $file_pcie_bus_id $file_pcie_numa | awk ' {print $4,$8,$9}' | sort -k1 > $file_configuration_table
}

# Function to expand ranges
expand_ranges() {
    echo "$1" | tr ',' '\n' | while read range; do
        if [[ $range == *-* ]]; then
            seq $(echo $range | awk -F'-' '{print $1, $2}')
        else
            echo "$range"
        fi
    done
}

create_thread_list()
{
   no_of_numa_nodes=`$LSCPU_CMD|grep "NUMA node(s):"|awk '{print $3}'`
   no_of_gaudis=`cat $file_configuration_table|wc -l`
   no_of_used_numa=`cat $file_pcie_numa | uniq | wc -l`

   if [ $no_of_numa_nodes -ge 1 ]; then
      AFFINITY_LEVEL=$((AFFINITY_LEVEL | ENUM[NUMAS]))
   fi

   for module_id in $(seq 0 $(($no_of_gaudis-1))); do
      #grab one pcie-id at a time (busID)
      pcie_bus_id=`cat $file_configuration_table | awk '{print $2}' | sed -n $(($module_id+1))p`

      #get the corresponding numanode (pcie_numa)
      if [ -f "/sys/bus/pci/devices/$pcie_bus_id/numa_node" ]; then
         numa_node=`cat /sys/bus/pci/devices/$pcie_bus_id/numa_node`
      elif [[ $USE_G3_MOCKUP_FILES == 1 || $USE_G2_MOCKUP_FILES == 1 ]]; then
         numa_node=$((index % 2))
      fi


      #special barcelona configuration where two sockets are configured to be 4 virtual numa nodes
      if [[ $no_of_used_numa -eq 2 && $no_of_numa_nodes -eq 4 ]]; then
         #get current node (moduleID)
         curr_node=`cat $file_configuration_table | awk '{print ","$3,$1}'| grep ",$numa_node" | awk '{print $2}'|head -1`
         if [ $module_id -eq $curr_node ]; then
            numa_node=$(($numa_node-1))
         fi
      fi

      #get the list of threads
      if [ $numa_node -ge 0 ]; then
         # Read isolated cores and expand them
         isolated_cores=$(expand_ranges "$($ISOLATED_CMD)")
         if [[ -z "$isolated_cores" ]]; then
            echo "If isolated_cores is empty, consider all cores as isolated. It will impact affinity level."
         else
            # Perform bitwise OR and reassign to AFFINITY_LEVEL
            AFFINITY_LEVEL=$((AFFINITY_LEVEL | AFFINITY_ENUM[ISOLATION]))
         fi

         # Get physical cores (excluding hyper-threaded ones)
         physical_cores=$($LSCPU_PARSE_EQ_CMD | grep -v '^#' | awk -F',' '!seen[$1]++ {print $2}')

         vector=`$LSCPU_PARSE_CMD | grep ",$numa_node,,"|awk -F"," '{print $1}'`

         AFFINITY_LEVEL=$((AFFINITY_LEVEL | AFFINITY_ENUM[NUMAS]))

         # Filter NUMA node cores to keep only physical cores that are isolated
         filtered_vector=$(for core in $vector; do
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
         AFFINITY_LEVEL=$((AFFINITY_LEVEL | AFFINITY_ENUM[PHYSICAL]))

         # Convert filtered_vector into an array
         filtered_array=($(echo "$filtered_vector"))

         # Calculate the key_factor
         num_cores=${#filtered_array[@]}
         key_factor=$((num_cores / ((no_of_gaudis)/no_of_numa_nodes)))

         # Calculate the core range for process_x
         start_index=$(( ((module_id % ((no_of_gaudis)/no_of_numa_nodes)) - 1) * key_factor ))
         end_index=$(( start_index + key_factor - 1 ))

         # Extract cores for process_x
         process_cores=("${filtered_array[@]:$start_index:$key_factor}")


         echo ${process_cores[@]} > $NUMA_MAPPING_DIR/.habana_moduleID$module_id
         echo ${process_cores[@]} >> $temp_dir/.module
      fi
   done
}

add_thread_list_to_config_table()
{
   #put it all together
   echo "ModID   BusID  NUMA   CPUs: " > $file_final_output
   echo "=====   =====  =====  ===== " >> $file_final_output
   paste $file_configuration_table $temp_dir/.module >> $file_final_output
}

clean_up()
{
   #remove the temp dir
   if [ ! -z "$temp_dir" ]; then
      rm -fr $temp_dir
   fi
}

main()
{
   check_env
   hl_smi_check
   create_temp_files
   create_configuration_table
   create_thread_list
   add_thread_list_to_config_table
   clean_up
   affinity_print "Script finished successfully"
   echo "AFFINITY_LEVEL exported with value: 0x$(printf "%x" "$AFFINITY_LEVEL")"
   echo "0x$(printf "%x" "$AFFINITY_LEVEL")" > $file_final_class_output
   exit 0
}

main