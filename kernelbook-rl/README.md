# kernelbook-rl

Turns [GPUMODE/KernelBook](https://huggingface.co/datasets/GPUMODE/KernelBook) rows into Harbor-format
Triton-kernel tasks, then trains Qwen on them with GRPO — first with our own loop (`rl_scratch/`),
then with prime-rl (`rl_prime/`).

Plan: https://claude.ai/code/artifact/915e76e0-6253-4b37-b9cc-29745e1cbff2

## Steps

Each step adds one component and ends with a check. Run commands from this folder.

| Step | What it adds | Check |
| --- | --- | --- |
| 0 | Project skeleton | `uv sync` then `uv run python -c "import modal, kernel_env"` and `uv run modal token info` |
| 1 | `kernel_env/data.py` (loading), `scripts/s01_profile.py` | `uv run python scripts/s01_profile.py` writes `data/profile_report.md` |
| 2 | `kernel_env/data.py` (filters), `scripts/s02_filter.py`, `tests/test_data.py` | `uv run pytest tests/test_data.py`, then `uv run python scripts/s02_filter.py` writes `data/kept.jsonl` + `data/dropped.jsonl` (exec check runs on Modal CPU) |

Setup note: `.venv` is a symlink to `~/.venvs/kernelbook-rl`. iCloud-synced folders mark `.pth` files hidden,
which Python skips, breaking the editable install. Recreate with
`UV_PROJECT_ENVIRONMENT=~/.venvs/kernelbook-rl uv sync && ln -s ~/.venvs/kernelbook-rl .venv`.
