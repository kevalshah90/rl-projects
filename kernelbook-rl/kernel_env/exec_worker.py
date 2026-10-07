"""Subprocess worker for F5: run exec_check() on a batch of rows, one JSON result per line.

Runs on: the Mac, started by scripts/s02_filter.py as  python -m kernel_env.exec_worker
Input  (stdin):  a JSON list of {"uuid", "python_code", "module_name"}
Output (stdout): one line per finished row: {"uuid": ..., "reason": null | "..."}

Why a separate process: exec_check() runs untrusted dataset code. In a child process, a
crash, a hang or a row that pollutes global state only takes down this worker, never
the filter script. The parent re-runs any row that didn't report back, alone.
This contains accidents; it is NOT a security sandbox (no network or file isolation).
"""

import json
import signal
import socket
import sys

from kernel_env.data import exec_check

ROW_TIMEOUT_SEC = 30          # a row still running after 30 s is treated as hung


class RowTimeout(BaseException):
    """BaseException, not Exception: exec_check's `except Exception` must NOT swallow it,
    otherwise a hung row would be reported as an ordinary crash."""


def _on_alarm(signum, frame):
    raise RowTimeout()


def _no_network(*args, **kwargs):
    # Task containers have no network, so a row that downloads (e.g. pretrained weights)
    # could never run there. Fail it here, on purpose, with a recognizable message.
    raise ConnectionRefusedError("network disabled in exec_check")


def main() -> None:
    rows = json.load(sys.stdin)
    socket.socket.connect = _no_network          # every outgoing connection fails in this worker
    socket.create_connection = _no_network
    signal.signal(signal.SIGALRM, _on_alarm)     # SIGALRM interrupts long-running Python code
    for row in rows:
        signal.alarm(ROW_TIMEOUT_SEC)            # start this row's countdown
        try:
            reason = exec_check(row["python_code"], row["module_name"])
        except RowTimeout:
            reason = f"exec: timeout after {ROW_TIMEOUT_SEC} s"
        finally:
            signal.alarm(0)                      # cancel the countdown
        # One line per row, flushed immediately: if the worker dies on the NEXT row,
        # the parent still has every result reported so far.
        print(json.dumps({"uuid": row["uuid"], "reason": reason}), flush=True)


if __name__ == "__main__":
    main()
