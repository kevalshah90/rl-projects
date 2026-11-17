"""Step 11 (part 1, check 1): a Modal image with prime-rl + our taskset, and a CPU smoke check of it.

prime-rl's own Dockerfile.cuda also builds UCX, NIXL and the AWS EFA stack (for multi-node /
disaggregated inference). One node with 2 H100s needs none of that, so this image keeps only:
    CUDA 13.0 base  ->  prime-rl at a pinned commit (+ its git submodules)
    ->  `uv sync --locked` with the `gpu` and `flash-attn` extras (torch 2.14 cu130, vLLM 0.33,
        flash-linear-attention for Qwen3.5's hybrid attention)
    ->  our code on top: kernel_env/, kernelbook_taskset.py, configs/, task folders.

`--frozen` means: install exactly prime-rl's uv.lock (no re-resolution). So the versions in this
image are the ones prime-rl tests with, including ITS verifiers (a submodule), not ours.

Run from the project root (no GPU is used; the check runs on CPU):
    uv run modal run scripts/s11_prime_rl_image.py
"""

from pathlib import Path

import modal

PROJECT = Path(__file__).resolve().parent.parent
PRIME_RL_REV = "389eacb"          # the commit whose loss/GRPO settings we read (NOTEBOOK 2026-11-09)
APP = Path("/app")                # prime-rl checkout
VENV = APP / ".venv"
KB = Path("/root/kb")             # our project inside the container (same layout as s10)

_skip = ["__pycache__", "**/__pycache__/**", ".DS_Store"]
image = (
    # cudnn-devel, like prime-rl's builder stage: nvcc + headers for kernels that JIT-compile
    # (FlashInfer, TileLang, Triton). add_python gives Modal the interpreter it needs.
    modal.Image.from_registry("nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "build-essential", "curl", "ninja-build", "libnuma-dev")
    .pip_install("uv==0.11.9")    # the uv prime-rl's Dockerfile.cuda uses (pyproject needs >= 0.11.1)
    .run_commands(
        # Submodule URLs are git@github.com:... (SSH); rewrite to HTTPS so no SSH key is needed.
        'git config --global url."https://github.com/".insteadOf "git@github.com:"',
        f"git clone https://github.com/PrimeIntellect-ai/prime-rl {APP}",
        f"cd {APP} && git checkout {PRIME_RL_REV} && git submodule update --init --recursive",
    )
    .env({
        "UV_PROJECT_ENVIRONMENT": str(VENV),
        "UV_PYTHON": "/usr/local/bin/python3.12",   # the add_python interpreter
        "UV_LINK_MODE": "copy",
        "UV_COMPILE_BYTECODE": "1",
    })
    .run_commands(
        # --frozen, not --locked: install exactly uv.lock without re-resolving. --locked re-resolves
        # to prove the lock is current, and prime-rl's lock uses a ROLLING cooldown
        # (exclude-newer-span = "P7D"), so a re-resolve on a later day differs and --locked fails.
        f"cd {APP} && uv sync --frozen --no-dev --extra gpu --extra flash-attn",
        # prime-rl depends on verifiers[harbor] only, without the modal extra; our orchestrator
        # creates the agent/grader sandboxes on Modal, so add the client. 1.5.1 = the version in
        # prime-rl's own uv.lock (keeps it consistent with that verifiers).
        f"uv pip install --python {VENV}/bin/python modal==1.5.1",
    )
    .env({
        "PATH": f"{VENV}/bin:/usr/local/bin:/usr/bin:/bin",   # `rl`, `vf-eval`, python -> the venv
        "PYTHONPATH": str(KB),                                 # `import kernelbook_taskset, kernel_env`
    })
    .add_local_dir(PROJECT / "kernel_env", f"{KB}/kernel_env", ignore=_skip)
    .add_local_file(PROJECT / "kernelbook_taskset.py", f"{KB}/kernelbook_taskset.py")
    .add_local_dir(PROJECT / "configs", f"{KB}/configs", ignore=_skip)
    .add_local_dir(PROJECT / "work.nosync/tasks", f"{KB}/work.nosync/tasks", ignore=_skip)
)

app = modal.App("kernelbook-prime-rl", image=image)

CHECK = r"""
import importlib, importlib.metadata as md, subprocess, sys
print("python", sys.version.split()[0], sys.executable)
for dist in ["prime-rl", "verifiers", "renderers", "torch", "vllm", "transformers",
             "flash-linear-attention", "flash-attn", "harbor", "modal"]:
    try:
        print(f"  {dist:24s} {md.version(dist)}")
    except md.PackageNotFoundError:
        print(f"  {dist:24s} MISSING")
import torch
print("torch built for CUDA", torch.version.cuda, "| GPU visible:", torch.cuda.is_available())
for mod in ["prime_rl", "verifiers", "verifiers.v1.tasksets.harbor", "renderers", "vllm", "fla"]:
    importlib.import_module(mod); print("  import ok:", mod)
"""


@app.function(cpu=2, memory=8192, timeout=30 * 60)
def check() -> None:
    import subprocess

    def sh(cmd: str) -> None:
        print(f"\n$ {cmd}", flush=True)
        r = subprocess.run(cmd, shell=True, cwd=KB, capture_output=True, text=True)
        print((r.stdout + r.stderr).strip()[-3000:], flush=True)
        print(f"-> exit {r.returncode}", flush=True)

    # 1. Versions + imports of everything the trainer/orchestrator/inference will load.
    sh(f"python -c '{CHECK}'")
    # 2. Which verifiers commit prime-rl pinned (check 2 compares it with ours, da75527).
    sh(f"git -C {APP} submodule status deps/verifiers deps/renderers")
    # 3. The entry point we will launch training with.
    sh("rl --help > /tmp/rl_help.txt; echo \"rl --help exit $?\"; head -5 /tmp/rl_help.txt")
    # 4. Preview of check 2: does our taskset import against prime-rl's verifiers?
    sh("python -c 'import kernelbook_taskset as k; print(\"kernelbook_taskset ok:\", k.__file__)'")


@app.local_entrypoint()
def main() -> None:
    check.remote()
