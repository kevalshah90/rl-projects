"""The agent's tool, check(), served over MCP (stdio) inside the agent's sandbox.

Runs on: the agent's sandbox (GPU). The harness starts it from task.toml:
    python /workspace/.kernel_tools/kernel_env/mcp_tools.py --module-name SumAggregator
Advisory only: check() runs the real grader on the agent's own files, but the reward comes
from the separate grader container, which only receives /workspace/solution.py. Editing
this file or reference.py can fool check(), never the reward.
"""

import argparse
import contextlib
import sys
from pathlib import Path

# Make `import kernel_env` work however the harness starts us: this file lives at
# <tools>/kernel_env/mcp_tools.py, so <tools> must be on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.mcpserver import MCPServer   # mcp 2.x: FastMCP was renamed MCPServer

from kernel_env.grade import grade

server = MCPServer("kernel-tools")
MODULE_NAME = ""                              # set from --module-name in __main__


@server.tool()
def check() -> dict:
    """Test /workspace/solution.py exactly like the final grader does.

    Returns each gate (1 = passed): exists, guard, parse, weights, runs, uses_triton,
    correct; plus failed_gate and message (the first failure and why), max_abs_diff
    (largest difference from the reference) and kernel_launches. reward = 1 means
    your solution would pass.
    """
    # stdout IS the MCP channel: anything the solution prints would corrupt it, so
    # send prints to stderr while grading.
    with contextlib.redirect_stdout(sys.stderr):
        try:
            result = grade("/workspace/reference.py", "/workspace/solution.py", MODULE_NAME)
        except Exception as e:                   # e.g. reference.py was edited and now crashes
            result = {"reward": 0.0, "failed_gate": "grader", "message": f"{type(e).__name__}: {e}"}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module-name", required=True)
    MODULE_NAME = parser.parse_args().module_name
    server.run("stdio")
