# Shared base image for every kernelbook-rl task: both the agent's container and the grader's.
#
# Dependencies only (option B). Our own code (kernel_env/) is copied into each task folder
# in step 5, so changing the grader never forces a rebuild of this ~4 GB image.
# Built and pushed to Docker Hub by .github/workflows/kernel-env-base-image.yml (repo root).

FROM python:3.12-slim

# torch 2.5.1: the version KernelBook's oracle Triton code was generated with (its
#   torch._inductor imports move between versions). The Linux wheel from PyPI bundles the
#   CUDA 12.4 runtime and triton 3.1.0, so no CUDA toolkit is needed here; the GPU *driver*
#   is provided by the host (Modal).
# numpy: imported by some reference modules.
# mcp:   the agent's tool server (kernel_env/mcp_tools.py, step 7).
RUN pip install --no-cache-dir torch==2.5.1 numpy mcp

# Harbor uploads each task's environment/ files here (task.toml sets workdir = /workspace).
WORKDIR /workspace
