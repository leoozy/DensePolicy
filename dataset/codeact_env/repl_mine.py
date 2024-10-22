import io
import os
import traceback
from contextlib import contextmanager, redirect_stdout
import signal
import re
from typing import Any, Mapping

import dill

from sup_func.sup_func import execute, PrintAllExpressions, execute_basic, run_with_timeout, execute_with_IS

import ast

from contextlib import contextmanager
from multiprocessing import Process, Pipe
from typing import Any, Mapping
import signal
import psutil


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds} seconds.")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)  # Disable the alarm


class LimitedMemoryProcess(Process):
    def __init__(self, max_memory, timeout, code, conn, user_ns):
        super().__init__()
        self.max_memory = max_memory
        self.timeout = timeout
        self.code = code
        self.conn = conn
        self.user_ns = user_ns

    def run(self):
        # Capture all output
        # success = False
        captured = io.StringIO()
        with redirect_stdout(captured):
            try:
                with time_limit(self.timeout):
                    # Set memory limit
                    process = psutil.Process(os.getpid())
                    process.rlimit(psutil.RLIMIT_AS, (self.max_memory, self.max_memory))
                    # Execute the code
                    # user_ns = execute(self.code, time_limit_query=self.timeout, global_env=self.user_ns)
                    # user_ns = execute_basic(self.code, global_env=self.user_ns)
                    user_ns = execute_with_IS(self.code, global_env=self.user_ns)
                    # serialized_user_ns = dill.dumps(user_ns)
                    output = captured.getvalue()
                success = True
            except Exception as e:
                formatted_exception = traceback.format_exception(type(e), e, e.__traceback__)
                # Filter out the lines containing file paths
                filtered_exception = [line for line in formatted_exception if
                                      not (line.strip().startswith("File") or line.strip().startswith("Traceback"))]
                output = "".join(filtered_exception)
                success = False
                # serialized_user_ns = dill.dumps(self.user_ns)

        output = re.sub('^None\n', '', output, flags=re.MULTILINE)
        if len(output) > 2000:
            output = output[:2000] + "...\n[Output Truncated]"

        # finally:
        # Send the result back through the pipe
        # if success:
        #     self.conn.send((serialized_user_ns, output))
        # else:
        #     self.conn.send(output)
        # print(output)
        self.conn.send((success, output))
        self.conn.close()


class PythonExec:
    """A tool for running python code in a REPL."""

    name = "PythonREPL"
    signature = "NOT_USED"
    description = "NOT_USED"

    def __init__(
            self,
            user_ns: Mapping[str, Any],
            timeout: int = 10,
    ) -> None:
        super().__init__()
        self.init_user_ns = user_ns
        self.timeout = timeout
        self.max_memory = 500 * 1024 * 1024  # 500 MB
        self.reset()

    def reset(self) -> None:
        self.user_ns = dict(self.init_user_ns)

    def wrap_code_with_print(self, query):
        # Wrap all expressions in print() statements
        # try:
        #     tree = ast.parse(query)
        #     _ = compile(tree, filename="<ast>", mode="exec")
        #     code_cor = True
        # except:
        #     code_cor = False
        # if code_cor:
        #     tree = PrintAllExpressions().visit(tree)
        #     tree = ast.fix_missing_locations(tree)
        #     query_p = compile(tree, filename="<ast>", mode="exec")
        # else:
        #     query_p = query

        query_p = query
        return query_p

    def basic_run(self, code):
        captured = io.StringIO()
        with redirect_stdout(captured):
            try:
                with time_limit(self.timeout):
                    # Execute the code
                    # user_ns = execute(self.code, time_limit_query=self.timeout, global_env=self.user_ns)
                    # user_ns = execute_basic(self.code, global_env=self.user_ns)
                    user_ns = execute_with_IS(code, global_env=self.user_ns)
                    output = captured.getvalue()
                success = True
            except Exception as e:
                formatted_exception = traceback.format_exception(type(e), e, e.__traceback__)
                # Filter out the lines containing file paths
                filtered_exception = [line for line in formatted_exception if
                                      not (line.strip().startswith("File") or line.strip().startswith("Traceback"))]
                output = "".join(filtered_exception)
                success = False
                # serialized_user_ns = dill.dumps(self.user_ns)

        output = re.sub('^None\n', '', output, flags=re.MULTILINE)
        if len(output) > 2000:
            output = output[:2000] + "...\n[Output Truncated]"
        return success, output

    def run_code_with_LimitedMemoryProcess(self, query_p):
        try:
            with time_limit(self.timeout):
                # Create a pipe to communicate between processes
                parent_conn, child_conn = Pipe()
                process = LimitedMemoryProcess(self.max_memory, self.timeout, query_p, child_conn, self.user_ns)

                # print('pos 2')
                # Start the process and wait for it to finish
                process.start()
                process.join()

                # print('pos 3')
                # Get the result from the pipe
                # if parent_conn.poll():
                # result = parent_conn.recv()
                # if isinstance(result, str):  # An exception occurred
                #     output = result
                # else:  # The code executed successfully
                #     serialized_user_ns, output = result
                #     # self.user_ns = dill.loads(serialized_user_ns)
                # success, output = run_with_timeout(parent_conn.recv, timeout=self.timeout)
                success, output = parent_conn.recv()
        except Exception as e:
            formatted_exception = traceback.format_exception(type(e), e, e.__traceback__)
            # Filter out the lines containing file paths
            filtered_exception = [line for line in formatted_exception if
                                  not (line.strip().startswith("File") or line.strip().startswith("Traceback"))]
            output = "".join(filtered_exception)
            success = False
        return success, output

    def update_ns(self, query_p):
        try:
            with time_limit(self.timeout):
                captured = io.StringIO()
                with redirect_stdout(captured):
                    # self.user_ns = execute(query_p, time_limit_query=self.timeout, global_env=self.user_ns)
                    # self.user_ns = execute_basic(query_p, global_env=self.user_ns)
                    self.user_ns = execute_with_IS(query_p, global_env=self.user_ns)
        except Exception as e:
            # print('Code local executing Error: {}'.format(e))
            pass
        # print('Local Executed')

    def __call__(self, query: str) -> str:

        query_p = self.wrap_code_with_print(query)

        # Test whether the code can be executed within time and space limitation
        # success, output = self.run_code_with_LimitedMemoryProcess(query_p)
        success, output = self.basic_run(query_p)

        # if success:
        #     self.update_ns(query_p)

        # # Replace potentially sensitive file paths
        # output = re.sub(
        #     r"File (.*)mint/tools/python_tool.py:(\d+)",
        #     r"File <hidden_filepath>:\1",
        #     output,
        # )

        if output == "":
            output = "Executed Successfully with No Output, Did you forget to print?"

        return output.strip()
