#!/usr/bin/env python3

"""
HCCL demo runner.
Usage example -
HCCL_COMM_ID=127.0.0.1:9696 python3 run_hccl_demo.py --test broadcast --nranks 8 --node_id 0 --ranks_per_node 8

Env variables - General
    HCCL_COMM_ID     - IP of node_id=0 host and an available port, in the format <IP:PORT>

Env variables - Host scaleout
    HCCL_OVER_OFI         - 1 to use OFI between boxes, 0 to use scaleout nics
    HCCL_GAUDI_DIRECT     - 1 to enable Gaudi Direct

Env variables - Host affinity settings
    NUM_SOCKETS           - Number of sockets for process affinity (default 0)
    NUM_HT                - Number of hyper threads for process affinity (default 0)
    NUM_CORES_PER_SOCKET  - Number of cores per socket for process affinity (default 0)
    ENFORCE_PROC_AFFINITY - Enforce using process affinity (default 0)
    DISABLE_PROC_AFFINITY - Disable using process affinity (default 0)
    BEST_EFFORT_AFFINITY  - Use best effort process affinity (default 0)
    NUMA_MAPPING_DIR      - Location of numa mapping file used for process affinity'''
"""

import argparse
import logging as Logger
import datetime, glob
import os, sys, subprocess, signal
from multiprocessing import Pool
import struct

class DemoTest:
    def __init__(self):
        self.nranks                   = None
        self.ranks_per_node           = None
        self.scaleup_group_size       = None
        self.node_id                  = None
        self.test                     = None
        self.size                     = None
        self.size_range               = None
        self.size_range_inc           = None
        self.loop                     = None
        self.ranks_list               = None
        self.test_root                = None
        self.mpi                      = None
        self.clean                    = None
        self.list_tests               = None
        self.dev_env                  = False
        self.doc                      = None
        self.number_of_processes      = None
        self.ignore_mpi_errors        = None
        self.no_color                 = None
        self.data_type                = None
        self.no_correctness           = False
        self.reduction_op             = None
        self.scaleout_bw              = None
        self.default_affinity_dir     = '/tmp/affinity_topology_output'
        self.cmd_list                 = []
        self.log_level                = Logger.DEBUG
        self.mpi_args                 = []
        self.ERROR                    = 1
        self.SUCCESS                  = 0
        self.csv_path                 = ""
        self.log_prefix               = "HCCL_demo_log_"
        self.demo_exe                 = "./hccl_demo"
        self.test_list                = ['broadcast',
                                         'all_reduce',
                                         'reduce_scatter',
                                         'all_gather',
                                         'send_recv',
                                         'reduce',
                                         'all2all',
                                         'scale_validation']
        self.data_type_list           = ['float',
                                         'bfloat16']
        self.optional_env_list        = ['DISABLE_PROC_AFFINITY',
                                         'ENFORCE_PROC_AFFINITY',
                                         'BEST_EFFORT_AFFINITY',
                                         'HCCL_OVER_OFI',
                                         'NUM_HT',
                                         'NUM_SOCKETS',
                                         'NUM_CORES_PER_SOCKET',
                                         'NUMA_MAPPING_DIR']
        self.default_mpi_env_list     = ['LD_LIBRARY_PATH']
        self.default_mpi_env_list_dev = ['HCL_ROOT',
                                         'SYNAPSE_ROOT',
                                         'BUILD_ROOT_LATEST',
                                         'GC_KERNEL_PATH',
                                         'ENGINES_FW_RELEASE_BUILD']
        self.default_mpi_arg_list     = ['--allow-run-as-root',
                                         '--mca btl_tcp_if_include']
        self.ignore_mpi_errors_list   = ['--mca btl_openib_warn_no_device_params_found 0']
        self.custom_comm = ""
        self.data_csv = False

        self.data_type_to_struct_format = {'float' : 'f', 'bfloat16' : 'e'}

        parser = argparse.ArgumentParser(description="""Run HCCL demo test""", allow_abbrev=False)
        # General flags
        general_group = parser.add_argument_group('General Options')
        general_group.add_argument("--clean", "-clean", action="store_true",
                            help="Clean previous artifacts including logs, recipe and csv results.")
        general_group.add_argument("-list", "--list_tests", action="store_true",
                            help="Display a list of available tests.")
        general_group.add_argument("--doc", action="store_true",
                            help="Display detailed help for HCCL demo in a form of docstring.")

        # Setup configuration flags
        setup_group = parser.add_argument_group('Setup Configuration Options')
        setup_group.add_argument("--nranks", type=int, default=-1,
                            help="Number of ranks in the communicator.")
        setup_group.add_argument("--ranks_per_node", type=int,
                            help="Number of cards per node (box), default read from h/w or set by MPI.")
        setup_group.add_argument("--scaleup_group_size", type=int,
                            help="Scaleup group size per node, default is ranks_per_node")
        setup_group.add_argument("--node_id", type=int,
                            help="Box index. Value in the range of (0, NUM_BOXES).", default=-1)
        setup_group.add_argument("--mpi", "-mpi", action="store_true",
                            help="Use MPI for managing execution.")
        # Test control flags
        test_group = parser.add_argument_group('Test Control Options')
        test_group.add_argument("--test", type=str,
                            help="Specify test (use '-l' option for test list).", default="broadcast")
        test_group.add_argument("--size", metavar="N", type=str,
                            help="Data size in units of G,M,K,B or no unit. Default is Bytes.", default=33554432)
        test_group.add_argument("--size_range", type=str, nargs=2, metavar=("MIN","MAX"),
                            help="Test will run from MIN to MAX, units of G,M,K,B or no unit. Default is Bytes. E.g. --size_range 32B 1M.")
        test_group.add_argument("--size_range_inc", metavar="M", type=int,
                            help="Test will run on all multiplies by 2^size_range_inc from MIN to MAX.", default=1)
        test_group.add_argument("--loop", type=int,
                            help="Number of loop iterations.", default=10)
        test_group.add_argument("--test_root", type=int, default=0,
                            help="Index of root rank for broadcast and reduce tests (optional).")
        test_group.add_argument("--ranks_list", type=str, help="List of pairs of ranks for send_recv ranks scaleout. E.g. 0,8,1,8 (optional).")
        test_group.add_argument("--data_type", type=str, default="float",
                            help="Data type, float or bfloat16. Default is float.")
        test_group.add_argument("--custom_comm", type=str, default="",
                            help="List of HCCL process that will open a communicator.")
        test_group.add_argument("--no_correctness", action="store_true", help="Skip correctness validation.")
        test_group.add_argument("--reduction_op", type=str, help="<sum|min|max> (default=sum)", default="sum")
        test_group.add_argument("--scaleout_bw", type=str, help="Expected scaleout BW in units of G,M,K,B or no unit. Default is Bytes/sec")
        # Logging flags
        log_group = parser.add_argument_group('Logging Options')
        log_group.add_argument("--csv_path", type=str,
                            help="Path to a file for results output (optional).")
        log_group.add_argument("--ignore_mpi_errors", "-ignore_mpi_errors", action="store_true",
                            help="Ignore generic MPI errors.")
        log_group.add_argument("--no_color", "-no_color", action="store_true",
                            help="Disable colored output in terminal.")
        log_group.add_argument("--data_csv", "-data_csv", action="store_true",
                            help="Creates 2 csv file for each rank, one for data input and second for data output.")

        self.create_logger()

        args, self.mpi_args = parser.parse_known_args()

        self.check_color(args)

        self.print_header()

        self.log_info("\nSetting HCCL demo attributes:", 'cyan')
        for arg in vars(args):
            if hasattr(self, arg):
                if arg == 'no_color' and self.no_color == True:
                    self.log_info(f'{arg.ljust(20)} = {self.no_color}', 'cyan')
                else:
                    setattr(self, arg, getattr(args, arg))
                    self.log_info(f'{arg.ljust(20)} = {getattr(args, arg)}', 'cyan')

    def validate_arguments(self):
        '''The following method is used to validate the correctness
           of the command line arguments before starting HCCL demo test.'''
        try:
            if not self.mpi:
                if not self.ranks_per_node:
                    self.get_ranks_per_node()   # Read from h/w or user must specify it
                if not self.scaleup_group_size:
                    self.get_scaleup_group_size()
                if self.node_id < 0:
                    self.exit_demo(f'[validate_arguments] Argument node_id was set to: {self.node_id}')
                if self.nranks < 1:
                    self.exit_demo(f'[validate_arguments] Argument nranks was set to: {self.nranks}')
                if self.mpi_args:
                    self.exit_demo(f'[validate_arguments] The option "--mpi" was not used, therefore the following arguments cannot be used: {self.mpi_args}')
                self.number_of_processes = min(self.ranks_per_node, self.nranks)
                self.log_debug(f'Number of processes to be used is: {self.number_of_processes}')
            else:   # MPI mode
                invalid_arguments = []
                if self.node_id >= 0:
                    invalid_arguments.append("node_id")
                if self.nranks >= 1:
                    invalid_arguments.append("nranks")
                if self.ranks_per_node:
                    invalid_arguments.append("ranks_per_node")
                if invalid_arguments:
                    self.exit_demo(f'[validate_arguments] the following command line arguments cannot be used in MPI mode: {invalid_arguments}')
            if not self.test in self.test_list:
                self.display_test_list()
                self.exit_demo(f'[validate_arguments] Chosen test: {self.test} is not part of the tests list')
            if not self.data_type in self.data_type_list:
                self.display_data_type_list()
                self.exit_demo(f'[validate_arguments] Chosen data type: {self.data_type} is not a valid data type')
            if self.loop < 1:
                self.exit_demo(f'[validate_arguments] Argument loop was set to: {self.loop}, it must be positive')
        except Exception as e:
            self.log_error(f'[validate_arguments] {e}' ,exception=True)
            raise Exception(e)

    def prepare_demo(self):
        '''The following method is used to prepare the required information
           before starting HCCL demo test.'''
        try:
            if self.doc:
                help(DemoTest)
                self.exit_demo()
            if self.list_tests:
                self.display_test_list()
                self.exit_demo()
            self.validate_arguments()
            self.get_env()
            if self.size_range:
                self.size_range[0] = self.parse_size(self.size_range[0])
                self.size_range[1] = self.parse_size(self.size_range[1])
            else:
                self.size = self.parse_size(self.size)

            self.validate_size()
            self.prepare_command()
            if self.clean:
                self.clean_artifacts()
            self.handle_affinity()
            if not os.path.exists(self.demo_exe):
                self.make_demo()
        except Exception as e:
            self.log_error(f'[prepare_demo] {e}' ,exception=True)
            raise Exception(e)

    def prepare_command(self):
        '''The following method is used to prepare the command to be used.
           Please notice that different commands are used for running
           HCCL demo in pure and mpi modes.'''
        try:
            if self.mpi:
                self.log_info("HCCL demo runs in MPI mode", 'green')
                cmd = self.get_command()
                self.log_debug(f"HCCL demo test command line: {cmd}")
                mpi_cmd = self.get_mpi_command()
                for i in cmd.split(" "):
                    mpi_cmd += " -x " + i
                mpi_cmd = self.apply_mpi_defaults(mpi_cmd)
                if self.ignore_mpi_errors:
                    self.set_env('HWLOC_HIDE_ERRORS','1')
                    for ignore_arg in self.ignore_mpi_errors_list:
                        mpi_cmd += f' {ignore_arg}'
                self.log_debug(f"HCCL demo mpi command line: {mpi_cmd}")
                self.cmd_list.append(mpi_cmd)
            else:
                self.log_info("HCCL demo runs in pure mode", 'green')
                for i in range(self.number_of_processes):
                    cmd = self.get_command(i)
                    self.cmd_list.append(cmd)
                self.log_debug("HCCL demo command line:")
                self.log_debug('\n'.join(self.cmd_list))
        except Exception as e:
            self.log_error(f'[prepare_command] {e}', exception=True)
            raise Exception(e)

    def get_command(self, id=0):
        '''The following method is used in order to determine HCCL demo command
           and translate class attributes to the corresponding env variables.'''
        try:
            cmd_args = []
            numa_output_path = os.getenv('NUMA_MAPPING_DIR', self.default_affinity_dir)
            cmd_args.append("HCCL_DEMO_TEST="          + str(self.test))
            cmd_args.append("HCCL_DATA_TYPE="          + str(self.data_type))
            if self.size_range:
                cmd_args.append("HCCL_SIZE_RANGE_MIN=" + str(self.size_range[0]))
                cmd_args.append("HCCL_SIZE_RANGE_MAX=" + str(self.size_range[1]))
                cmd_args.append("HCCL_SIZE_RANGE_INC=" + str(self.size_range_inc))
            else:
                cmd_args.append("HCCL_DEMO_TEST_SIZE=" + str(self.size))

            cmd_args.append("HCCL_DEMO_DATA_CSV="      + str(self.data_csv))

            if (self.no_correctness == True):
                cmd_args.append("HCCL_DEMO_CHECK_CORRECTNESS=0")

            cmd_args.append("HCCL_DEMO_TEST_LOOP="     + str(self.loop))

            if self.scaleout_bw:
                cmd_args.append("HCCL_EXPECTED_SCALEOUT_BW=" + self.parse_size(self.scaleout_bw, is_memory_size=False))

            cmd_args.append("HCCL_REDUCTION_OP=" + self.reduction_op)

            if self.ranks_list:
                cmd_args.append("HCCL_RANKS_LIST="        + str(self.ranks_list))
            cmd_args.append("HCCL_DEMO_TEST_ROOT="     + str(self.test_root))
            cmd_args.append("HCCL_DEMO_CSV_PATH="      + str(self.csv_path))
            cmd_args.append("HCCL_DEMO_MPI_REQUESTED=" + str(int(self.mpi)))
            cmd_args.append("MPI_ENABLED="             + str(int(self.mpi)))
            cmd_args.append("NUMA_MAPPING_DIR="        + str(numa_output_path))
            if self.custom_comm:
                cmd_args.append("HCCL_DEMO_CUSTOM_COMM=" +str(self.custom_comm))
            cmd_args.extend(self.set_optional_env())
            if not self.mpi:
                rank = id + self.node_id * self.number_of_processes
                cmd_args.append("ID=" + str(rank))
                cmd_args.append("HCCL_RANK=" + str(rank))
                cmd_args.append("HCCL_NRANKS=" + str(self.nranks))
                cmd_args.append("HCCL_RANKS_PER_NODE=" + str(self.ranks_per_node))
                cmd_args.append("HCCL_SCALEUP_GROUP_SIZE=" + str(self.scaleup_group_size))
                cmd_args.append(self.demo_exe)
            else:   # MPI
                if self.scaleup_group_size is not None:
                    cmd_args.append("HCCL_SCALEUP_GROUP_SIZE=" + str(self.scaleup_group_size))

            cmd = " ".join(cmd_args)
            return cmd
        except Exception as e:
            self.log_error(f'[get_command] {e}', exception=True)
            raise Exception(e)

    def get_mpi_command(self):
        '''# MPI helper method
           The following method is used in order to determine mpi command.
           The command will include full mpi path, mpi arguments requested by user.'''
        try:
            mpi_prefix = self.get_mpi_prefix()
            return mpi_prefix.ljust(len(mpi_prefix) + 1) + ' '.join(self.mpi_args)
        except Exception as e:
            self.log_error(f'[get_mpi_command] {e}', exception=True)
            raise Exception(e)

    def set_optional_env(self):
        '''The following method is used in order to append optional environment
           variables to the command line, in case any were requsted by the user.'''
        try:
            optional_args = []
            for env in self.optional_env_list:
                if env in os.environ:
                    optional_args.append(f'{env}={os.getenv(env).strip()}')
            return optional_args
        except Exception as e:
            self.log_error(f'[set_optional_env] {e}' ,exception=True)
            raise Exception(e)

    def set_env(self, key, value):
        '''The following method is used in order to set environment variables.'''
        try:
            os.environ[key] = value
        except Exception as e:
            self.log_error(f'[set_env] {e}' ,exception=True)
            raise Exception(e)

    def run_demo(self):
        '''The following method is used in order to trigger HCCL demo run.
           HCCL demo can be triggered in one of the following modes:
           1) Pure mode (default)
           2) MPI mode (triggered by adding -mpi)'''
        try:
            if self.mpi:
                self.run_mpi_test()
            else:
                self.run_test()
        except Exception as e:
            self.log_error(f'[run_demo] {e}' ,exception=True)
            raise Exception(e)

    def run_test(self):
        '''The following method is used in order to run HCCL demo test in pure mode.
           HCCL demo will invoke as many processes as were requested by the user.'''
        try:
            self.log_info("HCCL demo test command line:", 'green')
            self.log_info('\n\n'.join(self.cmd_list))
            pool = Pool(processes=self.nranks)
            results = pool.imap_unordered(self.run_process, self.cmd_list)
            for res in results:
                if res != 0:
                    pool.close()
                    pool.terminate()
                    pool.join()
                    self.log_error(f'[run_test] One of the hccl_demo processes failed, terminating hccl demo')
                    os.killpg(0, signal.SIGTERM)
                    self.exit_demo()
                    break
            pool.close()
            pool.join()

        except Exception as e:
            self.log_error(f'[run_test] One of the hccl_demo processes failed, terminating hccl demo, {e}, Processes: {str(self.cmd_list)}', exception=True)
            raise Exception(e)

    def run_mpi_test(self):
        '''# MPI helper method
           The following method is used in order to run HCCL demo test using MPI.'''
        mpi_cmd = self.cmd_list[0]
        mpi_cmd += " hccl_demo"
        self.run_mpi_command(mpi_cmd)

    def run_mpi_command(self, mpi_cmd):
        '''# MPI helper method
           The following method is used in order to run MPI command.'''
        try:
            self.log_info(f"HCCL mpi command line:", 'green')
            self.log_info(mpi_cmd)
            process = subprocess.Popen(mpi_cmd, shell=True)
            process.wait()
            process.communicate()
            return_code = process.poll()
            if return_code != 0:
                self.exit_demo(f'[run_mpi_command] mpi command processes failed, terminating hccl demo. command: {mpi_cmd}')
        except Exception as e:
            self.log_error(f'[run_mpi_command] {e}, command: {mpi_cmd}', exception=True)
            raise Exception(e)

    def run_command(self, command):
        '''The following method is used in order to run commands as a subprocess.'''
        try:
            self.log_debug(f'Running command line: {command}')
            p = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True, stderr=subprocess.PIPE)
            out, err = p.communicate()
            return out.decode('utf-8').splitlines()
        except Exception as e:
            self.log_error(f'[run_command] {e}' ,exception=True)
            raise Exception(e)

    def run_process(self, process):
        '''The following method is used in order to trigger system calls.'''
        try:
            self.log_debug(f'Running process: {process}')
            return os.system(process)
        except Exception as e:
            self.log_error(f'[run_process] {e}' ,exception=True)
            raise Exception(e)

    def make_demo(self, is_clean=False):
        '''The following method is used in order to build the HCCL demo.
           The build command will automatically adjust iself accordingly
           to the following:
           1) Environment type (development / release)
           2) Running mode (MPI / Pure)'''
        try:
            make_cmd = ''
            if self.mpi:
                make_cmd = self.cmd_list[0]
            make_cmd += ' bash build_demo.sh'
            if is_clean:
                make_cmd += ' clean'
            elif self.dev_env:
                self.log_debug('Detected development environment, going to build using make dev')
                make_cmd += ' dev'
            self.log_debug(f'Make command: {make_cmd}')
            result = self.run_process(make_cmd)
            if result != 0:
                self.exit_demo(f'[make_demo] The following make command has failed: {make_cmd}')
        except Exception as e:
            self.log_error(f'[make_demo] The following make command has failed: {make_cmd}, {e}', exception=True)
            raise Exception(e)

    def handle_affinity(self):
        '''The following method is used in order to set affinity for the processes.
            Affinity settings could be managed using the following environment variables:
            NUM_SOCKETS           - Number of sockets for process affinity (default 0)
            NUM_HT                - Number of hyper threads for process affinity (default 0)
            NUM_CORES_PER_SOCKET  - Number of cores per socket for process affinity (default 0)
            ENFORCE_PROC_AFFINITY - Enforce using process affinity (default 0)
            DISABLE_PROC_AFFINITY - Disable using process affinity (default 0)
            BEST_EFFORT_AFFINITY  - Use best effort process affinity (default 0)
            NUMA_MAPPING_DIR      - Location of numa mapping file used for process affinity'''
        try:
            from affinity import Affinity
            self.log_debug('Setting affinity')
            affinityObj = Affinity(self.mpi, self.cmd_list[0])
            result = affinityObj.create_affinity_files()
            if result != 0:
                self.exit_demo(f'[handle_affinity] Setting affinity has failed')
        except Exception as e:
            self.log_error(f'[handle_affinity] Setting affinity has failed, {e}', exception=True)
            raise Exception(e)

    def get_mpi_prefix(self):
        '''# MPI helper method
           The following method is used in order to determine mpi location using "which mpi" command'''
        try:
            output = subprocess.run(['which', 'mpirun'], stdout=subprocess.PIPE)
            result = str(output.stdout.decode('utf-8').strip())
            if not result:
                self.exit_demo(f'[get_mpi_prefix] MPI is not installed. Please install MPI or use HCCL demo in pure mode.')
            self.log_debug(f'MPI prefix is: {result}')
            return result
        except Exception as e:
            self.log_error(f'[get_mpi_prefix] {e}', exception=True)
            raise Exception(e)

    def apply_mpi_defaults(self, mpi_cmd):
        '''# MPI helper method
           The following method is used to add default arguments
           and environment variables to MPI command line,
           in case were not specified by user.'''
        try:
            self.log_debug(f'Setting HCCL demo MPI default environment variables:')
            if self.dev_env:
                self.default_mpi_env_list.extend(self.default_mpi_env_list_dev)
            for default_env in self.default_mpi_env_list:
                if default_env not in mpi_cmd:
                    mpi_cmd += " -x " + default_env
                    self.log_debug(f'-x {default_env}')

            self.log_debug(f'Setting HCCL demo MPI default arguments:')
            for default_arg in self.default_mpi_arg_list:
                if default_arg not in mpi_cmd:
                    if default_arg == '--mca btl_tcp_if_include':
                        interface = self.find_interface()
                        mpi_cmd += default_arg.rjust(len(default_arg) + 1) + " " + interface
                        self.log_debug(f'{default_arg} {interface}')
                    else:
                        mpi_cmd += default_arg.rjust(len(default_arg) + 1)
                        self.log_debug(f'{default_arg}')
            return mpi_cmd
        except Exception as e:
            self.log_error(f'[apply_mpi_defaults] {e}', exception=True)
            raise Exception(e)

    def parse_size(self, input_size, is_memory_size=True):
        '''The following method is used to parse the size to be sent.
           The format of the size would be <size><unit> , for example: 4G.
           One of the following sizes can be requested: G/M/K/B (not case sensitive).
           The unit is optional, if omitted the default unit <B> will be used.'''
        try:
            kilo = 1000
            if is_memory_size:
                kilo = 1024

            size = str(input_size)
            units_dict = {"G": kilo * kilo * kilo,
                          "M": kilo * kilo,
                          "K": kilo,
                          "B": 1}

            unit = size[-1].upper()
            if unit.isalpha():
                number = float(size[:-1])
                self.log_debug(f'Requested size: {number}')
                if unit in units_dict:
                    self.log_debug(f'Requested unit: {unit}')
                    unit_size = units_dict[unit]
                else:
                    self.log_error("Provided unit is not supported. Please choose between G,M,K,B or no unit. Going to use Bytes as default.")
                    unit_size = 1
                return str(int(number*unit_size))
            else:
                self.log_debug(f'Unit was not specified by user. Using Bytes as default unit.')
                return size
        except Exception as e:
            self.log_error(f'[parse_size] {e}' ,exception=True)
            raise Exception(e)

    def get_data_type_size(self):
        '''The following method calculates the size of the data type used in bytes'''
        try:
            return struct.calcsize(self.data_type_to_struct_format[self.data_type])
        except Exception as e:
            self.log_error(f'[get_data_type_size] {e}', exception=True)
            raise Exception(e)

    def validate_size(self):
        '''The following method is used to validate the size requested by the user.
           broadcast | reduce | all_reduce | all_gather | send_recv - require at least a single element.
           all2all | reduce_scatter - require at least a single element for each rank (i.e element * number of ranks)
           since HCCL demo supports float data type only, the minimum size would be:
           calculated number of elements * size of float
           when using size range the min size should be smaller than max size'''
        try:
            data_type_size = self.get_data_type_size()
            min_supported_size = self.nranks * data_type_size if (self.test == 'reduce_scatter' or self.test == 'all2all') else data_type_size
            min_size = self.size_range[0] if self.size_range else self.size
            if (min_supported_size > int(min_size)):
                self.exit_demo(f'[validate_size] Requested size: {min_size}B was less than supported minimun size: {min_supported_size}B')
            if self.size_range and int(self.size_range[0]) > int(self.size_range[1]):
                self.exit_demo(f'[validate_size] Requested size_range min size: {self.size_range[0]}B should be less than size_range max size: {self.size_range[1]}B')
        except Exception as e:
            self.log_error(f'[validate_size] {e}' ,exception=True)
            raise Exception(e)

    def display_test_list(self):
        '''The following method is used to display list of tests
           for the user upon request or error in chosen test name.'''
        try:
            self.log_info("\nTests list:", 'yellow')
            for test in self.test_list:
                self.log_info(f'{test}', 'yellow')
        except Exception as e:
            self.log_error(f'[display_test_list] {e}' ,exception=True)
            raise Exception(e)

    def display_data_type_list(self):
        '''The following method is used to display list of data types
           for the user upon request or error in chosen data type name.'''
        try:
            self.log_info("\nData type list:", 'yellow')
            for data_type in self.data_type_list:
                self.log_info(f'{data_type}', 'yellow')
        except Exception as e:
            self.log_error(f'[display_data_type_list] {e}' ,exception=True)
            raise Exception(e)

    def get_ranks_per_node(self):
        '''The following method is used to find the number of ranks
           per node using lspci command, in case the argument
           --ranks_per_node was not set by the user.'''
        try:
            ranks_per_node = self.run_command("lspci | grep -c -E '(Habana|1da3)'")
            if (ranks_per_node[0] != '0'):
                self.ranks_per_node = int(ranks_per_node[0])
                self.log_info(f'The user did not set --ranks_per_node. lscpi command found {self.ranks_per_node} ranks per node.')
            else:
                self.exit_demo(f'[get_ranks_per_node] The user did not set --ranks_per_node. lscpi command did not find number of ranks. Please provide --ranks_per_node.')
        except Exception as e:
            self.log_error(f'[get_ranks_per_node] {e}' ,exception=True)
            raise Exception(e)

    def get_scaleup_group_size(self):
        '''The following method is used to set the scaleup group size
           In case it is not supplied by user, by default its ranks_per_node'''
        try:
            self.scaleup_group_size = self.ranks_per_node
            self.log_info(f'The user did not set --scaleup_group_size. It is set to {self.ranks_per_node}')
        except Exception as e:
            self.log_error(f'[get_scaleup_group_size] {e}' ,exception=True)
            raise Exception(e)

    def get_env(self):
        '''The following method is used in order to determine environment type:
           1) Development environment
           2) Release environment'''
        try:
            if 'SYNAPSE_RELEASE_BUILD' in os.environ:
                self.dev_env = True
                self.log_debug('HCCL demo is running in development environment.')
            else:
                self.log_debug('HCCL demo is running in release environment.')
        except Exception as e:
            self.log_error(f'[get_env] {e}' ,exception=True)
            raise Exception(e)

    def import_package(self, package):
        '''The following method is used in order to import packages if needed.'''
        try:
            globals()[package] = __import__(package)
            self.log_debug(f'{package} package was successfully imported.')
        except ModuleNotFoundError:
            if package.lower() == "termcolor":
                self.no_color = True
            self.log_debug(f'{package} package could not be imported.')

    def check_color(self, args):
        '''The following method is used in order to determine whether colored output
           to the console should be supported in the used environment.'''
        if args.no_color:
            self.no_color = True
            self.log_debug('By request from the user, colors will not be used in console output.')
        else:
            self.import_package("termcolor")

    def print_colored(self, txt, color, attr=[]):
        '''The following method is used in order to determine whether
           the output to the console should be colored.'''
        if self.no_color:
            print(txt)
        else:
            termcolor.cprint(txt, color, attrs=attr)

    def print_header(self):
        '''The following method is used in order to print HCCL demo header.'''
        self.print_colored('Welcome to HCCL demo', 'magenta', ['underline'])

    def log_debug(self, msg):
        '''The following method is used in order to print in log level debug.'''
        Logger.debug(msg)

    def log_error(self, msg, exception=False):
        '''The following method is used in order to print in log level error.'''
        if exception:
            msg = f'HCCL demo exception: {msg}'
        else:
            msg = f'HCCL demo error: {msg}'
        Logger.error(msg)
        self.print_colored(msg, 'red', ['reverse'])

    def log_warning(self, msg, exception=False):
        '''The following method is used in order to print in log level warning.'''
        if exception:
            msg = f'HCCL demo exception: {msg}'
        else:
            msg = f'HCCL demo warning: {msg}'
        Logger.error(msg)
        self.print_colored(msg, 'yellow', ['reverse'])

    def log_info(self, msg, color=None):
        '''The following method is used in order to print in log level info.'''
        Logger.info(msg)
        if color:
            self.print_colored(msg, color)
        else:
            print(msg)

    def is_remote_directory(self, directory_path):
        '''The following method is used in order to check if a directory is remote'''
        try:
            # Get the device number of the specified directory
            target_device = os.stat(directory_path).st_dev

            # Get information about mounted filesystems
            mount_info = os.statvfs(directory_path)

            # Compare the device number of the directory with device numbers of mounted filesystems
            for mount_point in os.listdir('/'):
                mount_point_path = os.path.join('/', mount_point)
                if os.path.ismount(mount_point_path):
                    mount_device = os.statvfs(mount_point_path).f_fsid
                    if mount_device != mount_info.f_fsid:
                        return True
            return False
        except OSError as e:
            self.log_error(f'[should_rm_dir_using_mpi] {e}' ,exception=True)
            return None

    def clear_logs(self):
        '''The following method is used in order to clean old HCCL demo log files'''
        try:
            habana_logs_dir = os.environ.get('HABANA_LOGS')
            if(habana_logs_dir is not None and habana_logs_dir != "" and os.path.exists(habana_logs_dir)):
                rm_cmd = 'rm -rf $HABANA_LOGS'
                is_remote_dir = self.is_remote_directory(habana_logs_dir)
                if self.mpi and is_remote_dir is False:
                    mpi_prefix = self.get_mpi_command()
                    mpi_rm_cmd = mpi_prefix + ' -N 1 ' + rm_cmd
                    self.run_mpi_command(mpi_rm_cmd)
                else:
                    self.run_process(rm_cmd)
            else:
                self.log_error(f'[clear_logs] HABANA_LOGS variable is not set')
        except Exception as e:
            self.log_error(f'[clear_logs] {e}' ,exception=True)

    def remove_old_logs(self):
        '''The following method is used in order to remove old log files.
           Logs will be sorted and removed accordingly to their creation date, keeping 2 newest log files.'''
        try:
            log_files=[]
            for filename in glob.glob(self.log_prefix + "*"):
                log_files.append(filename)
            log_files.sort(key=lambda x: os.path.getctime(x))
            log_files=log_files[:-2]
            for log_file in log_files:
                self.log_debug(f'Removing old log file: {log_file}')
                os.remove(log_file)
        except Exception as e:
            self.log_error(f'[remove_old_logs] {e}' ,exception=True)

    def create_logger(self):
        '''The following method is used in order to start logger.
           Log files will be saved locally with the following prefix: HCCL_demo_log_*'''
        try:
            Logger.basicConfig(filename=self.log_prefix + str(datetime.datetime.now().strftime("%Y-%m-%d_%H%M")) + ".txt",format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level=self.log_level)
            self.log_debug("HCCL Demo - Start Logger")
            self.remove_old_logs()
        except Exception as e:
            self.log_error(f'[create_logger] {e}' ,exception=True)

    def clean_artifacts(self):
        '''The following method is used in order to clean artifacts such as:
        1) Old HCCL demo log files
        2) Old .csv files'''
        try:
            self.make_demo(is_clean=True)
            self.clear_logs()
            all_files = os.listdir(".")
            files_to_delete = [f for f in all_files if f.endswith('.csv')]
            for f in files_to_delete:
                os.remove(f)
                self.log_debug(f'Cleaning: {f}')
        except Exception as e:
            self.log_error(f'[clean_artifacts] {e}' ,exception=True)

    def exit_demo(self, error_msg="", exception=False):
        '''The following method is used in order to terminate HCCL demo test in case of
           an error on exception and display the relevant information about the termination.'''
        try:
            exit_code  = self.SUCCESS
            exit_color = 'green'
            if error_msg:
                if exception:
                    self.log_error(f'HCCL demo exception: {error_msg}')
                else:
                    self.log_error(f'HCCL demo error: {error_msg}')
                exit_code  = self.ERROR
                exit_color = 'red'
            self.log_info(f'\nExiting HCCL demo with code: {str(exit_code)}', exit_color)
            sys.exit(exit_code)
        except Exception as e:
            self.log_error(f'[exit_demo] {e}', exception=True)

    def find_interface(self):
        try:
            res = self.run_command("route")
            interface = ""
            for line in res:
                if "default" in line:
                    interface = line.split(" ")[-1]
                    break
            if interface == "":
                self.exit_demo(f'[find_interface] MPI requires --mca btl_tcp_if_include <interface>')
            return interface
        except Exception as e:
            self.exit_demo(f'[find_interface] MPI requires --mca btl_tcp_if_include <interface>')


if __name__ == '__main__':
    try:
        # Create new process group
        os.setpgrp()
        DemoTestObj = DemoTest()
        DemoTestObj.prepare_demo()
        DemoTestObj.run_demo()
    except Exception as e:
        DemoTestObj.exit_demo(f'[__main__] {e}', exception=True)