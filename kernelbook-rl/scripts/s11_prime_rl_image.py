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
    )
    # Our additions, in their own layer so changing them doesn't redo the ~15 GB sync above.
    #   modal 1.6.1: the client our Modal runtime plugin (modal_runtime.py) calls. prime-rl's
    #     verifiers no longer imports modal at all; 1.6.1 is what the same runtime code ran
    #     with in our baseline and E0.
    #   pytest + pytest-asyncio: verifiers' runtime test suite, run against the plugin.
    .run_commands(
        f"uv pip install --python {VENV}/bin/python modal==1.6.1 pytest pytest-asyncio pytest-timeout",
    )
    .env({
        "PATH": f"{VENV}/bin:/usr/local/bin:/usr/bin:/bin",   # `rl`, `vf-eval`, python -> the venv
        "PYTHONPATH": str(KB),                                 # `import kernelbook_taskset, kernel_env`
    })
    .add_local_dir(PROJECT / "kernel_env", f"{KB}/kernel_env", ignore=_skip)
    .add_local_file(PROJECT / "kernelbook_taskset.py", f"{KB}/kernelbook_taskset.py")
    .add_local_file(PROJECT / "modal_runtime.py", f"{KB}/modal_runtime.py")
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


TASKSET_CHECK = r"""
# Check 2: parse our tasks with prime-rl's verifiers and show what grading will do.
import json, sys
from pathlib import Path
import kernelbook_taskset as k
from verifiers.v1.tasksets.harbor import HarborConfig

print("HarborEnv exported:", k.HarborEnv is not None, "| __all__:", k.__all__)
for registry, dataset in [("registry-train.json", "kernelbook-train"), ("registry-dev.json", "kernelbook-dev")]:
    cfg = HarborConfig(registry_path=Path(sys.argv[1]) / registry, dataset=dataset, ignore_timeouts=False)
    tasks = list(k.KernelBookTaskset(cfg).load())
    print(f"{dataset}: {len(tasks)} tasks parsed")
t = tasks[0].data
v = t.verifier
print("example:", t.name)
print("  agent box:   image", t.image, "| workdir", t.workdir, "| gpu", t.resources.gpu, "| network allow", t.network.allow)
print("  mcp servers:", [s["name"] if isinstance(s, dict) else getattr(s, "name", s) for s in t.mcp_servers])
print("  scoring:     mode", t.scoring.mode, "| network", t.scoring.network)
print("  grader box:  fresh_copy", v.fresh_copy, "| network allow", v.network.allow, "| gpu", (v.resources.gpu if v.resources else None))
print("  artifacts:  ", [a.path if hasattr(a, "path") else a for a in t.artifacts])
print("  reward_key: ", tasks[0].config.reward_key)
"""


def container_registries() -> Path:
    """Registries hold Mac paths; rewrite them to the container's task folders (as s10 does)."""
    import json
    out = Path("/tmp/registries")
    out.mkdir(exist_ok=True)
    for name in ["registry-train.json", "registry-dev.json"]:
        data = json.loads((KB / "work.nosync/tasks" / name).read_text())
        for dataset in data:
            for task in dataset["tasks"]:
                task["path"] = str(KB / "work.nosync/tasks" / Path(task["path"]).name)
        (out / name).write_text(json.dumps(data))
    return out


@app.function(cpu=2, memory=8192, timeout=30 * 60)
def check_taskset() -> None:
    import subprocess

    regs = container_registries()
    for cmd in (
        ["python", "-c", TASKSET_CHECK, str(regs)],
        # Our real eval config through the new CLI (vf-eval was renamed vf-run after da75527):
        # resolve + validate only, no rollouts.
        ["vf-run", "@", "configs/baseline_train.toml", "--dry-run",
         "--env.taskset.registry_path", str(regs / "registry-train.json"),
         "-o", "/tmp/dry"],
    ):
        print("\n$", " ".join(c if len(c) < 80 else c[:40] + "..." for c in cmd), flush=True)
        r = subprocess.run(cmd, cwd=KB, capture_output=True, text=True)
        print((r.stdout + r.stderr).strip()[-4000:], flush=True)
        print(f"-> exit {r.returncode}", flush=True)
    subprocess.run("find /tmp/dry -name '*.json' | head -3", shell=True)


runs = modal.Volume.from_name("kernelbook-runs", create_if_missing=True)


@app.function(cpu=2, memory=8192, timeout=60 * 60, volumes={"/runs": runs})
def check_runtime() -> None:
    """verifiers' runtime contract tests against our Modal plugin. Each test starts real
    (CPU) Modal sandboxes from this container; declared features must work, undeclared
    ones (user switching, Compose) must be refused before anything is provisioned.
    Output streams live (-v, one line per test); a test over 300 s fails instead of hanging;
    the JUnit report lands in the Volume at checks/runtime_tests.xml."""
    import json
    import subprocess

    plugin = json.dumps({"type": "custom", "import_path": "modal_runtime.ModalBackend"})
    cmd = ["python", "-m", "pytest", "tests/unit/test_runtimes.py", "--runtime", plugin,
           "-p", "no:cacheprovider", "-v", "-rA", "--timeout", "300",
           "--junitxml", "/runs/checks/runtime_tests.xml"]
    print("$", " ".join(cmd), flush=True)
    code = subprocess.run(cmd, cwd=APP / "deps/verifiers").returncode   # not captured: live
    runs.commit()
    print(f"-> exit {code}", flush=True)


@app.local_entrypoint()
def main(which: str = "image") -> None:
    {"image": check, "taskset": check_taskset, "runtime": check_runtime}[which].remote()
