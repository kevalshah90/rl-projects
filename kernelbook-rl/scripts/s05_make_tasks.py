"""Step 5: turn kept KernelBook rows into Harbor task folders.

Runs on: the Mac, no GPU. From the project root:
    uv run python scripts/s05_make_tasks.py --limit 20     # first 20 rows, to inspect
    uv run python scripts/s05_make_tasks.py                # all kept rows
Reads:  data/kept.jsonl, templates/instruction.md, templates/test.sh, kernel_env/
Writes: tasks/<task-id>/ per row, and tasks/registry.json listing them as one dataset.

One task folder:
    task.toml              Harbor config (built as a dict, written with tomli_w: always valid TOML)
    instruction.md         the prompt the model reads
    environment/           uploaded into the AGENT's /workspace: reference.py + empty solution.py
    tests/                 staged only into the GRADER's fresh container at /tests:
                           reference.py (trusted copy), test.sh, kernel_env/ (the grader code)
    solution/solve.sh      oracle: writes the row's triton_code into /workspace/solution.py
"""

import argparse
import json
import shutil
from pathlib import Path
from string import Template

import tomli_w

from kernel_env.config import BASE_IMAGE, DATA, TASKS

INSTRUCTION = Template(Path("templates/instruction.md").read_text())   # uses ${class_name}
TEST_SH = Path("templates/test.sh").read_text()                        # same for every task
GRADER_FILES = ["__init__.py", "guard.py", "triton_hook.py", "grade.py"]  # all grade.py needs -> tests/
TOOL_FILES = GRADER_FILES + ["mcp_tools.py"]      # the agent's check tool calls grade() -> environment/.kernel_tools/


def task_id(row: dict) -> str:
    return f"kernelbook-{row['uuid']:05d}-{row['module_name'].lower()}"


def task_toml(row: dict) -> dict:
    """Harbor's task.toml as a plain dict. Keys follow Harbor's TaskConfig."""
    return {
        "schema_version": "1.4",
        "artifacts": ["/workspace/solution.py"],          # the only file copied to the grader
        "task": {"name": f"kernelbook/{row['uuid']:05d}-{row['module_name'].lower()}"},
        "metadata": {"source": "GPUMODE/KernelBook", "uuid": row["uuid"],
                     "module_name": row["module_name"], "license": row["licenses"][0]},
        "agent": {"timeout_sec": 900},                    # 15 min cap per episode
        "environment": {                                  # the agent's container
            "docker_image": BASE_IMAGE, "workdir": "/workspace", "network_mode": "no-network",
            "gpus": 1, "gpu_types": ["L4"],               # so the agent can run its own kernel
            # The agent's `check` tool. stdio = the harness starts it as a child process inside
            # this sandbox (no network, no extra container). The module name goes in as an
            # argument, so it doesn't depend on how the harness passes env vars.
            # The harness exposes this as the tool `kernel-tools_check` (<server name>_<tool name>);
            # templates/instruction.md names it exactly, so keep the two in sync.
            "mcp_servers": [{
                "name": "kernel-tools", "transport": "stdio", "command": "python",
                "args": ["/workspace/.kernel_tools/kernel_env/mcp_tools.py",
                         "--module-name", row["module_name"]],
            }],
        },
        "verifier": {
            "timeout_sec": 600,
            "environment_mode": "separate",               # grade in a fresh copy of the container
            "env": {"MODULE_NAME": row["module_name"]},   # read by kernel_env.grade
        },
    }


def solve_sh(triton_code: str) -> str:
    """Oracle script. Quoted heredoc = no bash expansion; the delimiter never occurs in code."""
    return ("#!/bin/bash\n"
            "cat > /workspace/solution.py <<'KERNELBOOK_ORACLE_EOF'\n"
            f"{triton_code}\nKERNELBOOK_ORACLE_EOF\n")


def make_task(row: dict) -> Path:
    task_dir = TASKS / task_id(row)
    files = {
        "task.toml": tomli_w.dumps(task_toml(row)),
        "instruction.md": INSTRUCTION.substitute(class_name=row["module_name"]),
        "environment/reference.py": row["python_code"],   # agent's copy (untrusted)
        "environment/solution.py": "",                    # the agent fills this
        "tests/reference.py": row["python_code"],         # grader's trusted copy
        "tests/test.sh": TEST_SH,
        "solution/solve.sh": solve_sh(row["triton_code"]),
    }
    for rel, text in files.items():
        path = task_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for script in ("tests/test.sh", "solution/solve.sh"):
        (task_dir / script).chmod(0o755)                  # executable, like Harbor's examples
    grader = task_dir / "tests" / "kernel_env"            # option B: grader code travels with the task
    grader.mkdir(exist_ok=True)
    for name in GRADER_FILES:
        shutil.copy(Path("kernel_env") / name, grader / name)
    tools = task_dir / "environment" / ".kernel_tools" / "kernel_env"   # uploaded to /workspace/.kernel_tools
    tools.mkdir(parents=True)
    for name in TOOL_FILES:
        shutil.copy(Path("kernel_env") / name, tools / name)
    return task_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only the first N kept rows")
    limit = parser.parse_args().limit

    rows = [json.loads(line) for line in open(DATA / "kept.jsonl")][:limit]
    # Regenerate from scratch (no stale tasks) by emptying tasks/, not deleting it: tasks/ is
    # a symlink into ~/kernelbook-rl-work, and rmtree refuses to delete through a symlink.
    TASKS.mkdir(exist_ok=True)
    for old in TASKS.iterdir():
        shutil.rmtree(old) if old.is_dir() else old.unlink()
    task_dirs = [make_task(row) for row in rows]

    # Harbor registry: one dataset; a task without git_url is loaded from its local path.
    registry = [{"name": "kernelbook", "version": "0.1",
                 "description": "Triton kernel tasks from GPUMODE/KernelBook",
                 "tasks": [{"name": d.name, "path": str(d.resolve())} for d in task_dirs]}]
    (TASKS / "registry.json").write_text(json.dumps(registry, indent=2))
    print(f"{len(task_dirs):,} tasks -> {TASKS}/  (+ registry.json)")


if __name__ == "__main__":
    main()
