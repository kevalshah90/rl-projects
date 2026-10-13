"""Settings shared by scripts and task generation. One place to change them.

Runs on: anywhere (pure constants).
"""

import os
from pathlib import Path

# Where generated data lives: work.nosync/ inside the project. iCloud skips folders ending in
# ".nosync"; otherwise it creates empty "name 2" duplicates whenever a folder is deleted and
# recreated (as s05 does to tasks/). Anchored to this file, so scripts work from any directory.
PROJECT = Path(__file__).resolve().parent.parent   # kernelbook-rl/
WORK = PROJECT / "work.nosync"
DATA = WORK / "data"          # profile report, kept/dropped rows, smoke and validation results
TASKS = WORK / "tasks"        # Harbor task folders + registry.json (step 5)
OUTPUTS = WORK / "outputs"    # Harbor job results, later checkpoints and run logs

# Your Docker Hub username: the image lives at <user>/kernel-env-base on Docker Hub.
DOCKERHUB_USER = "kevalshah90"

# The image every task container (agent and grader) starts from. Built by
# .github/workflows/kernel-env-base-image.yml. Override the whole name, e.g. to test a
# different tag, with:  KERNEL_ENV_IMAGE=user/kernel-env-base:other uv run ...
# The tag names the torch pin plus a revision (-rN). Bump the revision on every rebuild:
# Modal caches images by name, so reusing a tag would keep serving the old image.
# r2: adds gcc (Triton needs a C compiler), pins numpy and mcp.
BASE_IMAGE = os.environ.get("KERNEL_ENV_IMAGE", f"{DOCKERHUB_USER}/kernel-env-base:torch2.5.1-r2")
