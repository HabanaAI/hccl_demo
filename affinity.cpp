/******************************************************************************
# Copyright (c) 2022 Habana Labs, Ltd.
# SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

// C++ Standard Libraries
#include <iostream>
#include <unistd.h>
#include <sstream>
#include <fstream>
#include <synapse_api.h>

using namespace std;

#define DEFAULT_NUM_HT               0
#define DEFAULT_NUM_SOCKETS          0
#define DEFAULT_NUM_CORES_PER_SOCKET 0
#define DEFAULT_ENFORCE_AFFINITY     0
#define DEFAULT_DISABLE_AFFINITY     0
#define DEFAULT_BEST_EFFORT_AFFINITY 0
#define NUM_GAUDI_DEVICES_PER_HLS1   8

int get_num_sockets()
{
    static bool is_cached   = false;
    static auto num_sockets = DEFAULT_NUM_SOCKETS;
    if (!is_cached)
    {
        char* env_value = getenv("NUM_SOCKETS");
        num_sockets     = (env_value != nullptr) ? atoi(env_value) : num_sockets;
        is_cached       = true;
    }
    return num_sockets;
}

int get_num_cores_per_socket()
{
    static bool is_cached            = false;
    static auto num_cores_per_socket = DEFAULT_NUM_CORES_PER_SOCKET;
    if (!is_cached)
    {
        char* env_value      = getenv("NUM_CORES_PER_SOCKET");
        num_cores_per_socket = (env_value != nullptr) ? atoi(env_value) : num_cores_per_socket;
        is_cached            = true;
    }
    return num_cores_per_socket;
}

int get_num_ht()
{
    static bool is_cached = false;
    static auto num_ht    = DEFAULT_NUM_HT;
    if (!is_cached)
    {
        char* env_value = getenv("NUM_HT");
        num_ht          = (env_value != nullptr) ? atoi(env_value) : num_ht;
        is_cached       = true;
    }
    return num_ht;
}

bool get_disable_proc_affinity_env()
{
    static bool is_cached        = false;
    static auto disable_affinity = DEFAULT_DISABLE_AFFINITY;
    if (!is_cached)
    {
        char* env_value  = getenv("DISABLE_PROC_AFFINITY");
        disable_affinity = (env_value != nullptr) ? atoi(env_value) : disable_affinity;
        is_cached        = true;
    }
    return disable_affinity;
}

bool get_enforce_proc_affinity_env()
{
    static bool is_cached        = false;
    static auto enforce_affinity = DEFAULT_ENFORCE_AFFINITY;
    if (!is_cached)
    {
        char* env_value  = getenv("ENFORCE_PROC_AFFINITY");
        enforce_affinity = (env_value != nullptr) ? atoi(env_value) : enforce_affinity;
        is_cached        = true;
    }
    return enforce_affinity;
}

bool get_best_effort_proc_affinity_env()
{
    static bool is_cached            = false;
    static auto best_effort_affinity = DEFAULT_BEST_EFFORT_AFFINITY;
    if (!is_cached)
    {
        char* env_value      = getenv("BEST_EFFORT_AFFINITY");
        best_effort_affinity = (env_value != nullptr) ? atoi(env_value) : best_effort_affinity;
        is_cached            = true;
    }
    return best_effort_affinity;
}

string get_numa_mapping_dir_env()
{
    static bool is_cached        = false;
    static auto numa_mapping_dir = "";
    if (!is_cached)
    {
        char* env_value  = getenv("NUMA_MAPPING_DIR");
        numa_mapping_dir = (env_value != nullptr) ? env_value : numa_mapping_dir;
        is_cached        = true;
    }
    return numa_mapping_dir;
}

int setAutoAffinity(int moduleID)
{
    ostringstream filename;
    cpu_set_t     mask;
    int           thread_num;
    ifstream      ht_file;

    CPU_ZERO(&mask);

    filename << get_numa_mapping_dir_env() + "/.habana_moduleID" << moduleID;
    cout << "filename = " << filename.str() << endl;
    ht_file.open(filename.str());

    if (!ht_file.good())
    {
        cout << "Info: not enough information to set affinity correctly" << endl;
        cout << "       missing moduleId<->HThread mapping file (" + filename.str() + ")" << endl;
        return 1;
    }

    while (ht_file >> thread_num)
    {
        cout << thread_num << " ";
        CPU_SET(thread_num, &mask);
    }
    cout << endl;
    ht_file.close();

    if (sched_setaffinity(getpid(), sizeof(cpu_set_t), &mask) < 0)
    {
        cout << "sched_setaffinity() failed" << endl;
        return 1;
    }
    return 0;
}

int setCustomAffinity(int moduleID, int numSockets, int numCoresPerSocket, int numHT)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    int pid = getpid();

    if (numSockets > NUM_GAUDI_DEVICES_PER_HLS1)
    {
        cout << "Can't set process affinity for more than " << NUM_GAUDI_DEVICES_PER_HLS1 << " sockets" << endl;
        return 1;
    }

    int numRanksPerSocket = NUM_GAUDI_DEVICES_PER_HLS1 / numSockets;
    int hclCoresPerRank   = (numCoresPerSocket / numRanksPerSocket) - 1;  // one core for the app
    int chosenSocketNum   = moduleID / numRanksPerSocket;
    int chosenCPUNum      = numCoresPerSocket * chosenSocketNum + hclCoresPerRank * (moduleID % numRanksPerSocket);

    for (int i = 0; i < hclCoresPerRank; i++)
    {
        for (int hyperThread = 0; hyperThread < numHT; hyperThread++)
        {
            CPU_SET(chosenCPUNum + i + hyperThread * numCoresPerSocket * numSockets, &set);
        }
    }

    if (sched_setaffinity(pid, sizeof(cpu_set_t), &set) < 0)
    {
        cout << "sched_setaffinity() failed" << endl;
        return 1;
    }
    return 0;
}

int setBestEffortAffinity(int moduleID)
{
    int numSockets, numCoresPerSocket, numHT;

    string file_name = "setBestEffortAffinity" + to_string(moduleID) + ".txt";
    string cmd       = "lscpu|grep -E ' per core:| per socket:|Socket'|awk '{print $NF}' > " + file_name;

    system(cmd.c_str());
    ifstream affinity_output_file;
    affinity_output_file.open(file_name);
    if (affinity_output_file.is_open())
    {
        while (affinity_output_file >> numHT >> numCoresPerSocket >> numSockets)
        {
        }
        affinity_output_file.close();
    }
    else
    {
        cout << "Error opening file " << file_name << endl;
        return 1;
    }
    if (remove(file_name.c_str()) != 0)
    {
        cout << "Error removing file " << file_name << endl;
        return 1;
    }

    if (numSockets == 0 || numCoresPerSocket == 0 || numHT == 0)
    {
        cout << "The settings for best effort affinity are incorrect. Number of sockets: " << numSockets
             << ",  Number of cores per socket: " << numCoresPerSocket << ", Number of hyper threads: " << numHT
             << ". Please use custom affinity." << endl;
        return 1;
    }
    return setCustomAffinity(moduleID, numSockets, numCoresPerSocket, numHT);
}

void printAffinity(int moduleID)
{
    cpu_set_t mask;
    long      nproc, i;

    if (sched_getaffinity(getpid(), sizeof(cpu_set_t), &mask) < 0)
    {
        cout << "sched_getaffinity() failed" << endl;
    }
    nproc = sysconf(_SC_NPROCESSORS_ONLN);
    cout << "moduleID=" << moduleID << " affinity set to = ";
    for (i = 0; i < nproc; i++)
    {
        cout << CPU_ISSET(i, &mask);
    }
    cout << endl;
}

int setupAffinity(int device_module_id)
{
    if (!get_disable_proc_affinity_env())
    {
        if((uint32_t)device_module_id == INVALID_MODULE_ID)
        {
                cout << "Failed to set affinty, unknown module ID" << endl;
                return get_enforce_proc_affinity_env();
        }
        else if (get_num_sockets() && get_num_cores_per_socket() && get_num_ht())
        {
            cout << "Setting custom affinity" << endl;
            if (setCustomAffinity(device_module_id, get_num_sockets(), get_num_cores_per_socket(), get_num_ht()) != 0)
            {
                cout << "Failed to set custom affinty" << endl;
                return get_enforce_proc_affinity_env();
            }
        }
        else if (get_best_effort_proc_affinity_env())
        {
            cout << "Setting best effort affinity" << endl;
            if (setBestEffortAffinity(device_module_id) != 0)
            {
                cout << "Failed to set best effort affinty" << endl;
                return get_enforce_proc_affinity_env();
            }
        }
        else
        {
            cout << "Setting auto-affinity" << endl;
            if (setAutoAffinity(device_module_id) != 0)
            {
                if (get_enforce_proc_affinity_env())
                {
                    cout << "Setting auto-affinity failed, --enforce-affinity is on, exiting!" << endl;
                    return 1;
                }
                else
                {
                    cout << "Continuing without process affinity " << endl;
                }
            }
        }
        printAffinity(device_module_id);
    }
    else
    {
        cout << "Process affinity disabled" << endl;
    }
    return 0;
}
