# kernelbook-rl

Turns [GPUMODE/KernelBook](https://huggingface.co/datasets/GPUMODE/KernelBook) rows into Harbor-format
Triton-kernel tasks, then trains Qwen on them with GRPO — first with our own loop (`rl_scratch/`),
then with prime-rl (`rl_prime/`).

Plan: https://claude.ai/code/artifact/915e76e0-6253-4b37-b9cc-29745e1cbff2

## Where things run

- **Mac CPU:** everything that needs no GPU — dataset profiling, filtering, and the F5 execution check
  (`kernel_env/exec_worker.py` runs rows in timeout-guarded, network-blocked subprocesses), plus all
  Mac-only tests.
- **Modal:** only work that needs a GPU — the base-image smoke test, grading Triton kernels, task
  validation, baselines, and the RL loops (steps 3+).

## Steps

Each step adds one component and ends with a check. Run commands from this folder.

| Step | What it adds | Runs on | Check |
| --- | --- | --- | --- |
| 0 | Project skeleton | Mac | `uv sync` then `uv run python -c "import modal, kernel_env"` and `uv run modal token info` (login check only, starts nothing) |
| 1 | `kernel_env/data.py` (loading), `scripts/s01_profile.py` | Mac | `uv run python scripts/s01_profile.py` writes `data/profile_report.md` |
| 2 | `kernel_env/data.py` (filters), `kernel_env/exec_worker.py`, `scripts/s02_filter.py`, `tests/test_data.py` | Mac CPU | `uv run pytest tests/test_data.py`, then `uv run python scripts/s02_filter.py` writes `data/kept.jsonl` + `data/dropped.jsonl` |
| 3 | `docker/base.Dockerfile`, `../.github/workflows/kernel-env-base-image.yml`, `kernel_env/config.py`, `scripts/s03_smoke_test.py` | GitHub Actions (build), Modal L4 (test) | Image `<user>/kernel-env-base:torch2.5.1-r2` on Docker Hub; `uv run modal run scripts/s03_smoke_test.py` writes `data/smoke_s03.jsonl` |
| 4 | `kernel_env/guard.py`, `kernel_env/triton_hook.py`, `kernel_env/grade.py`, `tests/test_grade_gpu.py` | Modal L4 | `uv run modal run tests/test_grade_gpu.py` ends with `7/7 as expected` |
| 5 | `templates/instruction.md`, `templates/test.sh`, `scripts/s05_make_tasks.py` | Mac | `uv run python scripts/s05_make_tasks.py --limit 20` (or no limit for all) writes `tasks/<id>/` + `tasks/registry.json` |
| 6 | `scripts/s06_validate.py` | Harbor trials on Modal L4s | `uv run python scripts/s06_validate.py` runs oracle (must score 1) and nop (must score 0) on every task; writes `data/validation.jsonl` |

## Status (2026-10-07)

Steps 0–6 done on a 20-task sample (16 tasks validated). Next: decide how many tasks to validate
(~500 recommended over all 11,162), then step 7 (MCP tools: check / benchmark / submit), step 8
(splits), step 9 (baseline pass rates via verifiers' HarborTaskset).

## Known data issues

- **Step 6 drops ~20% of tasks** (4 of the first 20). Most are a bug in KernelBook's answer keys,
  not in our grader: for modules with 2+ same-shaped weights, the oracle's `forward()` wrapper
  passes the weights to the compiled kernel in the wrong order (e.g. weight and bias swapped in
  ScalarBiasScale, uuid 25; swapping them back gives an exact match). The task is valid, but with
  no working answer key it can't be proven solvable, so it's dropped. See `scripts/s06_validate.py`.
- Weights created with `torch.Tensor(n)` / `torch.empty` are uninitialized memory (NaN or huge on a
  GPU). `kernel_env/grade.py` replaces such values with small seeded random ones before copying
  the reference's weights into the solution.

## Setup notes

- `.venv` is a symlink to `~/.venvs/kernelbook-rl`. iCloud-synced folders mark `.pth` files hidden,
  which Python skips, breaking the editable install. Recreate with
  `UV_PROJECT_ENVIRONMENT=~/.venvs/kernelbook-rl uv sync && ln -s ~/.venvs/kernelbook-rl .venv`.
- `.python-version` pins an ARM64 Python 3.12: torch 2.5.1 has no wheels for Intel-Mac Python.
