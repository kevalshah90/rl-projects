"""Settings shared by scripts and task generation. One place to change them.

Runs on: anywhere (pure constants).
"""

import os

# Your Docker Hub username: the image lives at <user>/kernel-env-base on Docker Hub.
DOCKERHUB_USER = "kevalshah90"

# The image every task container (agent and grader) starts from. Built by
# .github/workflows/kernel-env-base-image.yml. Override the whole name, e.g. to test a
# different tag, with:  KERNEL_ENV_IMAGE=user/kernel-env-base:other uv run ...
BASE_IMAGE = os.environ.get("KERNEL_ENV_IMAGE", f"{DOCKERHUB_USER}/kernel-env-base:torch2.5.1")
