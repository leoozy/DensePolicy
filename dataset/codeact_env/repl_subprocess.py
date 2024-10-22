import ast
import os
import re
import signal
import subprocess
import textwrap
import traceback
from contextlib import contextmanager

import numpy as np
import psutil

# MEMORY_LIMIT_CODE = '''
# import psutil
# import os
# process = psutil.Process(os.getpid())
# process.rlimit(psutil.RLIMIT_AS, ({}, {}))
# '''

MEMORY_LIMIT_CODE = '''
import resource
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, ({}, hard))
'''

EXECUTION_FINISH_CODE = "\n\nprint('FINISHED')\n"

READY_CHECK_CODE = "\n\nprint('{}')\n"


def add_newlines_to_code_blocks(query):
    query_lines = query.strip().split('\n')
    query_lines = [line for line in query_lines if line.strip() != '']
    tree = ast.parse('\n'.join(query_lines))
    new_lines = []
    if len(tree.body) > 0:
        for i, node in enumerate(tree.body[:-1]):
            # print(f"Type: {type(node).__name__}, Line: {node.lineno}")
            new_lines.extend(query_lines[(node.lineno - 1):node.end_lineno])
            new_lines.append('')
        new_lines.extend(query_lines[(tree.body[-1].lineno - 1):])
    return '\n'.join(new_lines)


class CodeCheckVisitor(ast.NodeVisitor):
    def __init__(self):
        self.has_exit = False
        self.has_input = False

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "exit":
            self.has_exit = True
        elif (isinstance(node.func, ast.Attribute) and
              isinstance(node.func.value, ast.Name) and
              node.func.value.id == "sys" and
              node.func.attr == "exit"):
            self.has_exit = True
        elif isinstance(node.func, ast.Name) and node.func.id == "input":
            self.has_input = True
        self.generic_visit(node)


def check_exit_in_code(code_string):
    tree = ast.parse(code_string)
    visitor = CodeCheckVisitor()
    visitor.visit(tree)
    return {'has_exit': visitor.has_exit, 'has_input': visitor.has_input}


class PythonREPL_subprocess():
    def __init__(self, timeout=5, max_memory=4096 * 1024 * 1024):
        self.timeout = timeout
        self.max_memory = max_memory  # default: 1G memory
        self.create_python_subprocess()

    @contextmanager
    def time_limit(self, seconds):
        def signal_handler(signum, frame):
            raise TimeoutError(f"Timed out after {seconds} seconds.")

        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)  # Disable the alarm

    def create_python_subprocess(self):
        conda_env_name = "agent_env"  # Replace with your conda environment name
        # python_cmd = f"source activate {conda_env_name} && python3 -i" #conda activate agent_env && python3 -i
        # python_cmd = f"conda run -n {conda_env_name} python3 -i"
        # python_cmd = "python3 -i"
        # python_cmd =['python3', '-i']
        python_cmd = f"bash -i -c 'conda activate {conda_env_name} && python3 -i'" # bash -i -c 'conda activate agent_env && python3 -i'
        python_cmd = f"bash -i -c 'export PATH=\"/root/anaconda3/bin/:$PATH\" && source activate {conda_env_name} && python3 -i'"  # bash -i -c 'export PATH=\"/root/anaconda3/bin/:$PATH\" && source activate agent_env && python3 -i'

        self.process = subprocess.Popen(python_cmd,
                                        shell=True,
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True,
                                        preexec_fn=os.setsid)
        # self.process = subprocess.Popen(['python3', '-i'],  # -i for interactive mode
        #                                 stdin=subprocess.PIPE,
        #                                 stdout=subprocess.PIPE,
        #                                 stderr=subprocess.PIPE,
        #                                 text=True)  # or universal_newlines=True for Python < 3.7

    def stop_python_subprocess(self):
        parent = psutil.Process(self.process.pid)
        # print(self.process.pid)
        children = parent.children()
        if children:
            for child in children:
                # print(child.pid)
                os.kill(child.pid, signal.SIGINT)
        os.kill(self.process.pid, signal.SIGINT)
        os.killpg(os.getpgid(self.process.pid), signal.SIGINT)

    def close(self):
        self.stop_python_subprocess()
        self.process.stdin.close()
        self.process.terminate()
        os.killpg(os.getpgid(self.process.pid), signal.SIGINT)
        print('Process Closed')
        self.process.wait(timeout=10)

    def clean(self):
        clean_time = 1
        print('Cleaning up...')
        while True:
            try:
                with self.time_limit(clean_time):
                    # Different from code execution output read: stop criteria do not have "FINISHED"
                    output = ''
                    while True:
                        line = self.process.stdout.readline()
                        print('Cleaning Line: ' + str(line))
                        if not line:
                            break
                        output += line
                    # print('Output:\n' + output)
            except TimeoutError as e:
                pass
            # Read the error
            try:
                with self.time_limit(clean_time):
                    error = ''
                    while True:
                        line = self.process.stderr.readline()
                        print('Cleaning Line: ' + str(line))
                        if not line:
                            break
                        error += line
            except TimeoutError as e:
                pass
            if output == '' and error == '':
                break

    @staticmethod
    def forbidden_code_check(query):
        return check_exit_in_code(query)

    @staticmethod
    def code_legal_check(query):
        try:
            tree = ast.parse(query)
            return True, None
        except Exception as e:
            formatted_exception = traceback.format_exception(type(e), e, e.__traceback__)
            # Filter out the lines containing file paths
            filtered_exception = [line for line in formatted_exception if
                                  not (line.strip().startswith("File") or line.strip().startswith("Traceback"))]
            output = "".join(filtered_exception)
            success = False
            return success, output

    def execute_code(self, query):
        try:
            query_with_newlines = add_newlines_to_code_blocks(query)
        except SyntaxError as e:
            query_with_newlines = query

        query_with_finish = MEMORY_LIMIT_CODE.format(
            self.max_memory) + '\n' + query_with_newlines + '\n' + EXECUTION_FINISH_CODE
        print('Code to execute:\n' + query_with_finish + '\n')

        self.process.stdin.write(query_with_finish)
        self.process.stdin.flush()

    def read_output(self, timeout):
        # Code execution is expected to finish by 'FINISHED' flag
        try:
            with self.time_limit(timeout):
                output = ''
                finished = False
                while True:
                    line = self.process.stdout.readline()
                    # print('New Line: ' + str(line))
                    if not line or 'FINISHED' in line:
                        finished = True
                        break
                    output += line
            return output, finished
        except TimeoutError as e:
            return output, finished

    def read_error(self, timeout):
        # Error read can only be finished with timeout
        try:
            with self.time_limit(timeout):
                error = ''
                while True:
                    line = self.process.stderr.readline()
                    # print('New Line: ' + str(line))
                    if not line:
                        break
                    error += line
        except TimeoutError as e:
            pass
        return error

    def check_ready(self):
        print('Checking ready for execution...')
        ready_check_time = 1
        key = str(np.random.rand())
        ready_check_code = READY_CHECK_CODE.format(key)
        self.clean()
        self.execute_code(ready_check_code)

        res, finished = self.read_output(ready_check_time)
        if not finished:
            print('Previous subprocess not finished.\n{}\n{}'.format(key, res))
            return False

        if key == res.strip():
            print('Ready check success.\n{}\n{}'.format(key, res))
            return True
        else:
            print('Ready check failed.\n{}\n{}'.format(key, res))
            return False

    def stop_timeout_subprocess(self):
        print('Stopping Timeout Process...')
        wait_time = 200
        try:
            with self.time_limit(wait_time):
                while True:
                    self.stop_python_subprocess()
                    if self.check_ready():
                        break
        except TimeoutError as e:
            # todo directly kill the subprocess, and create a new subprocess
            pass

    def output_wrap(self, output):
        output = output.strip()
        if len(output) > 2000:
            output = output[:2000] + "...\n[Output Truncated]"
        if output == "":
            output = "Executed Successfully with No Output, Did you forget to print?"
        return output.strip() + '\n'

    def __call__(self, query: str) -> str:
        legal, feedback = self.code_legal_check(query)
        if not legal:
            return feedback
        check_res= self.forbidden_code_check(query)
        if check_res['has_exit']:
            return 'Do not use exit() or sys.exit() in your code, as they will close the Interactive Python.'
        if check_res['has_input']:
            return 'Do not use input() in your code; instead, set the value directly within your code.'

        self.clean()
        self.execute_code(query)

        output, finished = self.read_output(self.timeout)
        # print('Output:\n' + output)
        if not finished:
            print('TIME OUT')
            self.stop_timeout_subprocess()
            error_msg = f"TimeoutError: Timed out after {self.timeout} seconds."
            return error_msg.strip() + '\n'

        # Read the error
        error = self.read_error(1)
        # print('Error:\n' + output)

        return self.output_wrap((output.strip() + '\n\n' + error.strip()).strip())

    def __del__(self):
        self.close()
