#!/bin/bash
# Runs in the grader's fresh container. /tests holds this folder; /workspace/solution.py
# is the only file copied from the agent. kernel_env.grade always writes the reward files.
mkdir -p /logs/verifier
PYTHONPATH=/tests python -m kernel_env.grade \
  --reference /tests/reference.py \
  --solution /workspace/solution.py \
  --out-dir /logs/verifier
