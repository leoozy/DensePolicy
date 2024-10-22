import os
import signal
import subprocess
from contextlib import contextmanager


class Eurus_PythonREPL():
    def __init__(self, timeout=5, tmp_file="cache/tmp"):
        self.timeout = timeout
        self.tmp_file = tmp_file
        os.system(f"touch {self.tmp_file}.py")

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

    def __call__(self, query: str) -> str:
        query = query.strip().split("\n")
        # if "print(" not in query[-1]:
        # query[-1] = "print(" + query[-1] + ")"
        query = "\n".join(query)

        with open(f'{self.tmp_file}.py', "w") as f:
            f.write(query)

        try:
            with self.time_limit(self.timeout):
                result = subprocess.run(
                    ['python3', f'{self.tmp_file}.py'], capture_output=True, check=False, text=True,
                    timeout=self.timeout)

                if result.returncode == 0:
                    output = result.stdout
                    return output.strip()
                else:
                    error_msg = result.stderr.strip()
                    msgs = error_msg.split("\n")
                    new_msgs = []
                    want_next = False
                    for m in msgs:
                        if "Traceback" in m:
                            new_msgs.append(m)
                        elif m == msgs[-1]:
                            new_msgs.append(m)
                        elif self.tmp_file in m:
                            st = m.index('"/') + 1 if '"/' in m else 0
                            ed = m.index(f'/{self.tmp_file}.py') + 1 if f'/{self.tmp_file}.py' in m else None
                            clr = m[st:ed] if not ed else m[st:]
                            m = m.replace(clr, "")
                            new_msgs.append(m)
                            want_next = True
                        elif want_next:
                            new_msgs.append(m)
                            want_next = False
                    error_msg = "\n".join(new_msgs)
                    return error_msg.strip()
        except TimeoutError as e:
            error_msg = f"Timed out after {self.timeout} seconds."
            return error_msg.strip()
