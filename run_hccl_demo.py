#!/usr/bin/env python3

"""
HCCL demo runner.
Usage example -
HCCL_COMM_ID=127.0.0.1:5553 HCCL_OVER_TCP=0 python3 run_hccl_demo.py --test broadcast --nranks 16 --node_id=0 --ranks_per_node 8

Args
    --nranks         - int, Number of ranks participating in the demo
    --ranks_per_node - int, Number of ranks participating in the demo for current node
    --node_id        - int, ID of the running host. Each host should have unique id between 0-num_nodes
    --test           - str, Which hccl test to run (for example: broadcast/all_reduce) (default: broadcast)
    --size           - str, Data size in units of G,M,K,B or no unit (default: 33554432)
    --loop           - int, Number of iterations (default: 10)
    --test_root      - int, Index of root rank for broadcast and reduce tests
    --csv_path       - str, Path to a file for results output
    -clean           - Clear old executable and compile a new one
    -l               - Display a list of available tests

Env variables - General
    HCCL_COMM_ID     - IP of node_id=0 host and an available port, in the format <IP:PORT>

Env variables - Host scaleout
    SOCKET_NTHREADS  - Number of threads to manage TCP sockets
    NSOCK_PERTHREAD  - Number of sockets per thread
    HCCL_OVER_TCP    - 1 to use TCP between boxes, 0 to use scaleout nics
    HCCL_OVER_OFI    - 1 to use OFI between boxes, 0 to use scaleout nics
"""

import os
import sys
from multiprocessing import Pool
import argparse
import subprocess
import signal

demo_exe = "./hccl_demo"

test_list = ('broadcast', 'all_reduce', 'reduce_scatter', 'all_gather', 'send_recv', 'reduce')
test_params = {}
print_log = lambda *x: None

def show_test_list():
    print("Test list:")
    for test in test_list:
        print(f'    {test}')

def run_command(command):
    print('Running command line {}'.format(command))
    p = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True, stderr=subprocess.PIPE)
    out, err = p.communicate()
    return out.decode('utf-8').splitlines()

def get_ranks_per_node():
    ranks_per_node = run_command("lspci | grep -c -E '(Habana|1da3)'")
    return int(ranks_per_node[0])

def read_settings():
    global num_processes
    num_processes = min(test_params["ranks_per_node"], test_params["nranks"])

def is_dev():
    if 'SYNAPSE_RELEASE_BUILD' in os.environ:
        return True
    else:
        return False

def handle_make(isClean=False):
    make_cmd = 'make'
    if isClean:
        make_cmd += ' clean'
    elif is_dev():
        print_log('Detected dev environment!')
        make_cmd += ' dev'
    run_process(make_cmd)

def clear_logs():
    rm_cmd = 'rm -rf ~/.habana_logs*'
    run_process(rm_cmd)

def clean_artifacts():
    handle_make(isClean=True)
    clear_logs()
    all_files = os.listdir(".")
    files_to_delete = [f for f in all_files if f.endswith('.recipe.used') or f.endswith('.csv')]
    for f in files_to_delete:
        try:
            os.remove(f)
            print_log(f'Cleaning: {f}')
        except:
            print(f'Failed to remove file: {f}')

def run_process(p):
    print_log(f'Running: {p}')
    return os.system(p)

def parse_size(size):
    print_log(f'Parsing size: {size}')
    units_dict = {"G": 1024*1024*1024,
                  "M": 1024*1024,
                  "K": 1024,
                  "B": 1}

    unit = size[-1].upper()
    if unit.isalpha():
        number = float(size[:-1])
        if unit in units_dict:
            unit_size = units_dict[unit]
        else:
            print("Provided unit is not supported. Please choose between G,M,K,B or no unit. Going to use Bytes as default.")
            unit_size = 1
        return str(int(number*unit_size))
    return size

def handle_affinity():
    if is_dev():
        if 'AFFINITY_ENABLED' in os.environ and int(os.environ['AFFINITY_ENABLED']):
            from affinity import create_affinity_files
            create_affinity_files()

def handle_args():
    parser = argparse.ArgumentParser(description="""Run HCL demo test""")

    parser.add_argument("--nranks", type=int,
                        help="Number of ranks in the communicator")
    parser.add_argument("--ranks_per_node", type=int,
                        help="Number of ranks in the node")
    parser.add_argument("--node_id", type=int,
                        help="Box index. Value in the range of (0, NUM_BOXES)", default=-1)
    parser.add_argument("--test", type=str,
                        help="Specify test (use '-l' option for test list)", default="broadcast")
    parser.add_argument("--size", metavar="N", type=str,
                        help="Data size in units of G,M,K,B or no unit. Default is Bytes.", default=33554432)
    parser.add_argument("--loop", type=int,
                        help="Number of loop iterations", default=10)
    parser.add_argument("--test_root", type=int,
                        help="Index of root rank for broadcast and reduce tests (optional)")
    parser.add_argument("--csv_path", type=str, default="",
                        help="Path to a file for results output (optional)")
    parser.add_argument("-clean", action="store_true",
                        help="Clean previous artifacts including logs, recipe and csv results")
    parser.add_argument("-l", "--list_tests", action="store_true",
                        help="Display a list of available tests")

    args = parser.parse_args()

    if args.clean:
        clean_artifacts()

    if args.nranks:
        test_params["nranks"] = args.nranks
    if args.ranks_per_node:
        test_params["ranks_per_node"] = args.ranks_per_node
    else:
        test_params["ranks_per_node"] = get_ranks_per_node()

    if args.node_id >= 0:
        test_params["node_id"] = args.node_id

    if args.test:
        if not args.test in test_list:
            print(f'Error: no test {args.test}. Select a test from the list:')
            show_test_list()
            sys.exit(1)
        test_params["test"] = args.test

    if args.size:
        test_params['size'] = parse_size(str(args.size))

    if args.loop:
        test_params['loop'] = args.loop

    if args.test_root:
        test_params['test_root'] = args.test_root

    if args.csv_path:
        test_params['csv_path'] = args.csv_path

    if args.list_tests:
        show_test_list()
        sys.exit(0)

def get_hccl_demo_command(id=0):
    cmd_args = []
    cmd_args.append("HCCL_DEMO_TEST=" + str(test_params['test']))
    if ('test_root' in test_params):
        cmd_args.append("HCCL_DEMO_TEST_ROOT=" + str(test_params['test_root']))
    if ('csv_path' in test_params):
        cmd_args.append("HCCL_DEMO_CSV_PATH=" + str(test_params['csv_path']))
    else:
        cmd_args.append("HCCL_DEMO_TEST_SIZE=" + str(test_params['size']))
    cmd_args.append("HCCL_DEMO_TEST_LOOP=" + str(test_params['loop']))

    rank = id + test_params["node_id"] * num_processes
    cmd_args.append("HCCL_RANK=" + str(rank))
    cmd_args.append("HCCL_NRANKS=" + str(test_params["nranks"]))
    cmd_args.append("HCCL_BOX_SIZE=" + str(test_params["ranks_per_node"]))
    cmd_args.append(demo_exe)
    cmd = " ".join(cmd_args)
    return cmd

def main():
    handle_args()
    print_log("Printing test params:")
    print_log(test_params)
    read_settings()
    handle_affinity()

    # Create the test executable if not found
    if not os.path.exists(demo_exe):
        handle_make()

    processes = []

    for i in range(num_processes):
        p = get_hccl_demo_command(i)
        processes.append(p)

    pool = Pool(processes=test_params["nranks"])

    results = pool.imap_unordered(run_process, processes)
    for res in results:
        if res != 0:
            print("One of the hccl_test processes failed, terminating hccl demo.")
            pool.close()
            pool.terminate()
            pool.join()
            os.killpg(0, signal.SIGTERM)
            sys.exit(os.EX_DATAERR)
            break

    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
