#!/usr/bin/env python3

import os, subprocess

class Affinity:
    def __init__(self, mpi, user_cmd):
        self.user_cmd    = user_cmd
        self.mpi         = mpi
        self.file_name   = 'list_affinity_topology.sh'
        self.default_dir = '/tmp/affinity_topology_output'
        self.exe         = f'bash {self.file_name}'
        self.SUCCESS     = 0
        self.ERROR       = 1
        self.return_code = self.SUCCESS

    def create_affinity_files(self):
        try:
            # ENABLE_CONSOLE env var affects bash output redirection.
            # This variable needs to deleted before running the bash script
            # and later returned.
            enable_console_val = None
            if 'ENABLE_CONSOLE' in os.environ:
                enable_console_val = os.getenv('ENABLE_CONSOLE')
                del os.environ['ENABLE_CONSOLE']

            # In case affinity configuration is disabled, exit
            if self.is_enabled_in_cmd('DISABLE_PROC_AFFINITY'):
                self.print_affinity(f'Affinity setting was disabled by user.')
                self.calculate_return_code(self.SUCCESS)
                return self.return_code

            self.print_affinity("Creating affinity files...")

            # Make sure affinity script exists
            if not os.path.isfile(self.file_name):
                self.print_affinity(f'Could not find {self.file_name}')
                self.calculate_return_code(self.ERROR)
                return self.return_code

            # Set the output directory for the moduleID <-> numa mapping
            output_path = os.getenv('NUMA_MAPPING_DIR', self.default_dir)

            # Determine correct command line (MPI/pure mode)
            if self.mpi:
                self.print_affinity('Running in MPI mode.')
                cmd = f'{self.user_cmd} -x NUMA_MAPPING_DIR={output_path} {self.exe}'
            else:
                self.print_affinity('Running in pure mode.')
                cmd = f'MPI_ENABLED=0 NUMA_MAPPING_DIR={output_path} {self.exe}'

            self.print_affinity(f'Running the following command line: {cmd}')

            # Run the script
            process = subprocess.Popen(cmd, shell=True)
            process.wait()
            return_code = process.poll()
            self.print_affinity(f'Finished with code: {return_code}')
            self.calculate_return_code(return_code)

            if enable_console_val:
                os.environ['ENABLE_CONSOLE'] = enable_console_val

            return self.return_code
        except Exception as e:
            self.print_affinity(f'[create_affinity_files] failed with exception: {e}')

    def calculate_return_code(self, status):
        try:
            if self.is_enabled_in_cmd('ENFORCE_PROC_AFFINITY') and status != self.SUCCESS:
                self.return_code = self.ERROR
        except Exception as e:
            self.print_affinity(f'[calculate_return_code] failed with exception: {e}')
            self.return_code = self.ERROR

    def print_affinity(self, msg):
        try:
            print(f'Affinity: {msg}')
        except Exception as e:
            self.print_affinity(f'[print_affinity] failed with exception: {e}')

    def is_enabled_in_cmd(self, arg):
        try:
            list_of_values = ['1', 'true']
            list_of_commands=self.user_cmd.split()
            for single_arg in list_of_commands:
                if single_arg.startswith(arg) and single_arg.split("=")[-1].lower() in list_of_values:
                    return True
            return False
        except Exception as e:
            self.print_affinity(f'[is_enabled_in_cmd] failed with exception: {e}')
            return False
