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
    # prime-kernels: the trainer imports it unconditionally (trainer/distributed/collectives.py),
    # but it sits in the `kernels` extra, which `all` leaves out (upstream's Dockerfile adds it).
    # Exact wheel pinned in uv.lock; its own layer so the sync layer above stays cached.
    .run_commands(
        f"uv pip install --python {VENV}/bin/python --no-deps "
        "'prime-kernels @ https://github.com/PrimeIntellect-ai/prime-kernels/releases/download/"
        "v0.1.0-86f0ad0/prime_kernels-0.1.0+cu130torch2.14.0-cp312-cp312-linux_x86_64.whl'",
    )
    # FlashAttention 3: the trainer's attn="auto" picks FA3 on Hopper (H100) and refuses to start
    # without it. Prebuilt wheel from PyTorch's cu130 index, as pinned (with hash) in uv.lock.
    .run_commands(
        f"uv pip install --python {VENV}/bin/python --no-deps "
        "'flash_attn_3 @ https://download.pytorch.org/whl/test/cu130/"
        "flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
        "#sha256=44a019ca428d0866fc73ca49b8af02eb4b6a37ae8a1a6b1a07917c383a187fe6'",
    )
    # vllm-router: prime-rl's inference launcher always fronts the vLLM engine with it (even on one
    # node); it ships in the `disagg` extra, which the lean image leaves out. Wheel + hash from
    # uv.lock; its deps (aiohttp, fastapi, uvicorn, ...) are already installed with vLLM.
    .run_commands(
        f"uv pip install --python {VENV}/bin/python --no-deps "
        "'vllm-router @ https://github.com/PrimeIntellect-ai/router/releases/download/v0.2.4/"
        "vllm_router-0.2.4-cp38-abi3-manylinux_2_28_x86_64.whl"
        "#sha256=c2ed887d8be68430630b187c4044131bbe6d9ed4b7d99f40e2c7cc5e73a4824c'",
        f"{VENV}/bin/vllm-router --help > /dev/null && echo 'vllm-router OK'",
    )
    # Local patch to prime-rl's pinned renderers (a9d1676; unchanged on upstream main 9f4943f).
    # Its Qwen3.5 tool-call parser strips ALL leading/trailing whitespace from every parameter
    # value (`pm.group(2).strip()`), on top of the regex already removing the template's one
    # newline per side. vLLM's qwen3_coder parser (our baseline/E0) keeps the value as written.
    # Without the patch, training-time `edit` calls lose indentation in old_str/new_str (no match,
    # or unindented code): the agent would act in a different environment than in eval.
    # Found by check 3 (s12): 10/79 calls differed. The assert fails the build if the line moved.
    .run_commands(
        f"python3 - <<'EOF'\n"
        f"p = '{APP}/deps/renderers/renderers/parsing.py'\n"
        f"s = open(p).read()\n"
        f"old = 'arg_value = pm.group(2).strip()'\n"
        f"assert s.count(old) == 1, 'renderers patch target not found exactly once'\n"
        f"open(p, 'w').write(s.replace(old, 'arg_value = pm.group(2)  # kernelbook patch: keep whitespace'))\n"
        f"EOF",
    )
    .env({
        "PATH": f"{VENV}/bin:/usr/local/bin:/usr/bin:/bin",   # `rl`, `vf-eval`, python -> the venv
        "PYTHONPATH": str(KB),                                 # `import kernelbook_taskset, kernel_env`
    })
    .add_local_dir(PROJECT / "kernel_env", f"{KB}/kernel_env", ignore=_skip)
    .add_local_file(PROJECT / "kernelbook_taskset.py", f"{KB}/kernelbook_taskset.py")
    .add_local_file(PROJECT / "modal_runtime.py", f"{KB}/modal_runtime.py")
    .add_local_file(PROJECT / "scripts/s12_renderer_check.py", f"{KB}/scripts/s12_renderer_check.py")
    .add_local_file(PROJECT / "scripts/s13_trainer_check.py", f"{KB}/scripts/s13_trainer_check.py")
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
    for name in ["registry-train.json", "registry-dev.json", "registry-trainable.json"]:
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


PORT = 8765   # interception server's fixed port, published with modal.forward (as in s10)


@app.function(
    secrets=[modal.Secret.from_name("kernelbook-vllm")],   # VLLM_API_KEY
    volumes={"/runs": runs},
    cpu=2, memory=8192, timeout=3 * 60 * 60,
)
def e2e(config: str = "configs/e2e_prime_rl.toml", dry_run: bool = False, args: str = "") -> None:
    """Check 2 end to end: vf-run (prime-rl's verifiers) on our tasks with the Modal plugin.
    Same transport as s10: the interception server runs here, and sandboxes reach it at a
    modal.forward URL (custom tunnel). Traces go to the Volume under e2e_prime_rl/."""
    import subprocess
    import threading
    import tomllib

    regs = container_registries()
    cfg = tomllib.loads((KB / config).read_text())
    registry = regs / Path(cfg["env"]["taskset"]["registry_path"]).name
    out = Path("/runs") / Path(cfg["output_dir"]).name
    cmd = ["vf-run", "@", config, "--env.taskset.registry_path", str(registry), "-o", str(out),
           *args.split()]
    if dry_run:
        cmd += ["--dry-run"]

    stop = threading.Event()

    def keep_committing() -> None:   # traces visible in the Volume while the run goes
        while not stop.wait(60):
            runs.commit()

    threading.Thread(target=keep_committing, daemon=True).start()
    with modal.forward(PORT) as tunnel:
        cmd += ["--env.interception.type", "server",
                "--env.interception.tunnel.type", "custom",
                "--env.interception.tunnel.url", tunnel.url,
                "--env.interception.tunnel.port", str(PORT)]
        print("interception server reachable at", tunnel.url, flush=True)
        print("$", " ".join(cmd), flush=True)
        code = subprocess.run(cmd, cwd=KB).returncode
    smi.terminate()
    stop.set()
    runs.commit()
    print(f"-> exit {code}", flush=True)


@app.function(cpu=4, memory=16384, timeout=30 * 60, volumes={"/runs": runs})
def check_renderer(run: str = "e2e_prime_rl/kernelbook_taskset--qwen--qwen3.5-9b--bash--960c9af1") -> None:
    """Check 3: scripts/s12_renderer_check.py on real e2e conversations (CPU; downloads the
    Qwen3.5-9B tokenizer only, not the weights)."""
    import os
    import subprocess
    code = subprocess.run(["python", "scripts/s12_renderer_check.py",
                           f"/runs/{run}/traces.jsonl"], cwd=KB).returncode
    print(f"-> exit {code}", flush=True)
    # renderers' own tests for Qwen3.5 + round trips: the whitespace patch must not break them.
    # A site-packages dependency ships a top-level `tests` package, which shadows renderers'
    # namespace package `tests`; an __init__.py (in this throwaway container only) fixes that.
    shadow = subprocess.run(["python", "-c", "import tests; print(tests.__file__)"],
                            capture_output=True, text=True, cwd="/").stdout.strip()
    print("\nshadowing `tests` package:", shadow or "none", flush=True)
    (APP / "deps/renderers/tests/__init__.py").touch()
    print("$ pytest renderers tests (qwen35 / roundtrip / parsing)", flush=True)
    code = subprocess.run(["python", "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider",
                           "-k", "qwen35 or Qwen35 or roundtrip or parse", "-x", "--timeout", "600"],
                          cwd=APP / "deps/renderers",
                          env={**os.environ, "PYTHONPATH": f"{APP}/deps/renderers/tests:{APP}/deps/renderers:{KB}"}).returncode
    print(f"-> exit {code}", flush=True)


hf_cache = modal.Volume.from_name("kernelbook-hf-cache", create_if_missing=True)   # s09's weights


@app.function(gpu="H100", cpu=8, memory=65536, timeout=90 * 60,
              volumes={"/runs": runs, "/root/.cache/huggingface": hf_cache})
def check_trainer(run: str = "e2e_prime_rl/kernelbook_taskset--qwen--qwen3.5-9b--bash--960c9af1",
                  parts: str = "abc") -> None:
    """Check 4 on one H100: (a) driver vs CUDA 13, (b) prime-rl's trainer loop on fake data,
    (c) vLLM vs trainer logprobs on real prompts (scripts/s13_trainer_check.py)."""
    import subprocess

    def sh(cmd: str) -> int:
        print(f"\n$ {cmd}", flush=True)
        code = subprocess.run(cmd, shell=True, cwd=KB).returncode
        print(f"-> exit {code}", flush=True)
        return code

    if "a" in parts:
        sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv")
        sh("python -c \"import torch; print('torch', torch.__version__, '| built for CUDA', "
           "torch.version.cuda, '| available', torch.cuda.is_available()); "
           "x = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16); "
           "print('bf16 matmul ok:', (x @ x).float().norm().item() > 0)\"")
        sh("python -c \"import fla, flash_attn; print('fla', fla.__version__, '| flash_attn', "
           "flash_attn.__version__)\"")
    tr = "torchrun --nproc-per-node 1 --master-port 29511"
    if "b" in parts:
        sh(f"{tr} -m prime_rl.trainer.rl.train @ configs/check4_trainer_fake.toml "
           "--no-monitors.prime 2>&1 | "   # standalone trainer: no `rl` launcher run to log to
           "grep -v -i 'warn' | tail -40")
    data = "/runs/checks/check4_vllm_samples.json"
    if "c" in parts:   # c = sample with vLLM, then score with the trainer
        Path("/runs/checks").mkdir(parents=True, exist_ok=True)
        if sh(f"python scripts/s13_trainer_check.py vllm --traces /runs/{run}/traces.jsonl "
              f"--out {data}") != 0:
            return
        runs.commit()
    if "c" in parts or "t" in parts:   # t = trainer only, reusing the saved vLLM samples
        sh("nvidia-smi --query-gpu=memory.used --format=csv")   # must be ~0 before training
        sh(f"{tr} scripts/s13_trainer_check.py trainer --data {data}")
        sh(f"{tr} scripts/s13_trainer_check.py trainer --gdn-lora")   # params + backward only
    hf_cache.commit()


def resolve_rl_config(config: str, tunnel_url: str) -> Path:
    """The rl config as this container needs it: Mac registry path -> container registry,
    tunnel placeholder -> the real modal.forward URL. Written next to nothing else in /tmp."""
    regs = container_registries()
    text = (KB / config).read_text()
    for name in ["registry-train.json", "registry-dev.json", "registry-trainable.json"]:
        text = text.replace(f'"work.nosync/tasks/{name}"', f'"{regs / name}"')
    assert "__TUNNEL_URL__" in text, "config has no tunnel placeholder"
    out = Path("/tmp") / Path(config).name
    out.write_text(text.replace("__TUNNEL_URL__", tunnel_url))
    return out


# Off for every rl launch: the Prime platform monitor (default on in trainer + orchestrator)
# would register the run on Prime's platform. Metrics stay local (file monitor, metrics.jsonl).
NO_PRIME = ["--no-trainer.monitors.prime", "--no-orchestrator.monitors.prime"]

# WANDB_API_KEY for [monitors.wandb]: the user's Modal secret (key re-issued 2026-11-18; never in the repo).
WANDB = modal.Secret.from_name("wandb-secret")


@app.function(cpu=2, memory=8192, timeout=20 * 60, volumes={"/runs": runs}, secrets=[WANDB])
def smoke_dry_run(config: str = "configs/rl_smoke.toml") -> None:
    """`rl --dry-run` on the smoke config (CPU only): parse + validate + resolve the full config,
    write the per-process configs (trainer / orchestrator / inference / env workers), exit.
    No model download, no GPU, no sandboxes. Resolved configs land in the Volume under
    checks/rl_smoke_dry/; the key fields are printed here."""
    import json
    import shutil
    import subprocess

    import os
    import wandb   # the key is present and accepted by W&B (read-only check; creates no run)
    print("WANDB_API_KEY present:", bool(os.environ.get("WANDB_API_KEY")), "| W&B login verified:",
          wandb.login(verify=True), flush=True)
    cfg = resolve_rl_config(config, "https://dry-run.invalid")   # no tunnel needed to validate
    out = Path("/runs/checks/rl_smoke_dry")
    shutil.rmtree(out, ignore_errors=True)
    # the same flags rl_run passes on a restart (--resume with no checkpoint = start fresh)
    cmd = ["rl", "@", str(cfg), "--dry-run", "--output-dir", str(out), "--run.name", "dry",
           "--resume", *NO_PRIME]
    print("$", " ".join(cmd), flush=True)
    code = subprocess.run(cmd, cwd=KB).returncode
    print(f"-> exit {code}", flush=True)
    for f in sorted(out.rglob("resolved/*.json")):
        c = json.loads(f.read_text())
        print(f"{f.name}: monitors={c.get('monitors')} lr={c.get('optim', {}).get('lr')} "
              f"ckpt={c.get('ckpt')} resume={c.get('resume')}", flush=True)
    runs.commit()


def ckpt_complete(step_dir: Path) -> bool:
    """A step checkpoint is usable only when both halves are fully written: the trainer's DCP
    `.metadata` is written last (after every shard), the orchestrator's progress.pt by atomic
    rename. prime-rl itself has no completion marker, and `--resume` picks the highest step_*
    dir, so a half-written one (preemption mid-save) must never reach the Volume."""
    return (step_dir / "trainer" / ".metadata").exists() and (step_dir / "orchestrator" / "progress.pt").exists()


def ckpt_steps(ckpt_dir: Path) -> list[int]:
    return sorted(int(d.name.removeprefix("step_")) for d in ckpt_dir.glob("step_*")
                  if d.name.removeprefix("step_").isdigit() and ckpt_complete(d))


@app.function(
    gpu="H100:2",                       # 1 vLLM + 1 trainer (deployment in the config)
    cpu=16, memory=131072,
    timeout=24 * 60 * 60,               # Modal's max; a run past this is relaunched with the same name
    volumes={"/runs": runs, "/root/.cache/huggingface": hf_cache},
    secrets=[WANDB],
)
def rl_run(config: str, run_name: str, args: str = "") -> None:
    """An RL run: `rl` (trainer + vLLM + orchestrator + env worker) on one 2-H100 container.
    Sandboxes reach the env worker's interception server through modal.forward (as check 2).

    Resumable. The run dir (/tmp/rl/<run_name>) stays on local disk: vLLM reads weight
    broadcasts from it, and the Volume is too slow for that. Every minute a thread mirrors it to
    /runs/rl/<run_name>: logs, configs, metrics, and each COMPLETE checkpoint (copied as
    step_N.partial, then renamed; only the newest 2 kept). When the container starts and the
    Volume holds a checkpoint, it is copied back and `rl --resume` continues from it: the trainer
    reloads LoRA + optimizer + scheduler, the orchestrator its progress and data position, and
    the trainer's startup broadcast puts that step's adapter on vLLM. This covers both a
    preemption (Modal reruns this function with the same input) and a manual relaunch with the
    same --run-name (e.g. after the 24 h timeout). Episodes in flight at the cut are lost."""
    import os
    import shutil
    import subprocess
    import threading
    import time

    out = Path("/tmp/rl")
    local = out / run_name
    mirror = Path("/runs/rl") / run_name
    wandb_dir = Path("/tmp/rl_wandb") / run_name
    sys_dir = Path("/tmp/rl_system") / run_name   # GPU samples for BOTH H100s (vLLM's included)
    skip = shutil.ignore_patterns("broadcasts", "weights", "checkpoints", "*.safetensors", "*.bin", "*.pt")

    resume = []
    runs.reload()
    saved = ckpt_steps(mirror / "checkpoints")
    if saved:
        step = saved[-1]
        print(f"resuming {run_name} from Volume checkpoint step_{step}", flush=True)
        shutil.copytree(mirror / "checkpoints" / f"step_{step}", local / "checkpoints" / f"step_{step}")
        resume = ["--resume"]
    elif mirror.exists():
        print(f"{mirror} exists but holds no complete checkpoint: starting {run_name} from scratch",
              flush=True)

    def sync_ckpts() -> None:
        src, dst = local / "checkpoints", mirror / "checkpoints"
        new = [n for n in ckpt_steps(src) if not (dst / f"step_{n}").exists()]
        for n in new[-1:]:                       # only the newest; older ones are superseded
            tmp = dst / f"step_{n}.partial"
            shutil.rmtree(tmp, ignore_errors=True)
            t0 = time.time()
            shutil.copytree(src / f"step_{n}", tmp)
            tmp.rename(dst / f"step_{n}")
            print(f"checkpoint step_{n} -> Volume in {time.time() - t0:.0f}s", flush=True)
        for n in ckpt_steps(dst)[:-2]:           # keep the newest 2 on the Volume
            shutil.rmtree(dst / f"step_{n}", ignore_errors=True)

    def sync() -> None:
        if local.exists():
            shutil.copytree(local, mirror, dirs_exist_ok=True, ignore=skip)
            sync_ckpts()
        for extra, name in [(wandb_dir, "wandb"), (sys_dir, "system")]:
            if extra.exists():
                shutil.copytree(extra, mirror / name, dirs_exist_ok=True)
        runs.commit()

    stop = threading.Event()

    def keep_syncing() -> None:
        while not stop.wait(60):
            try:
                sync()
            except Exception as e:   # a failed mirror must never kill the run
                print("mirror failed:", e, flush=True)

    threading.Thread(target=keep_syncing, daemon=True).start()
    subprocess.run("nvidia-smi --query-gpu=name,memory.used --format=csv", shell=True)
    # W&B online: trainer + orchestrator log to one shared run (prime-rl's mode under `rl`). Its local
    # files (full .wandb log of metrics + system stats) sit next to the run dir (inside it, prime-rl's
    # "run dir already used" guard would trip on a fresh start) and are mirrored to <mirror>/wandb:
    # a backup if the upload fails (`wandb sync`).
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(wandb_dir)
    # W&B's system metrics come from the processes that log to it (trainer, orchestrator); vLLM's
    # process doesn't, so its GPU may be missing there. nvidia-smi samples every GPU every 15 s.
    sys_dir.mkdir(parents=True, exist_ok=True)
    # one file per container: a preemption restart must not overwrite the previous one's samples
    gpu_log = open(sys_dir / f"gpu-{time.strftime('%Y%m%d-%H%M%S')}.csv", "a")
    smi = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,"
         "memory.total,power.draw,temperature.gpu,clocks.sm", "--format=csv,nounits", "-l", "15"],
        stdout=gpu_log, stderr=subprocess.STDOUT)
    with modal.forward(PORT) as tunnel:
        cfg = resolve_rl_config(config, tunnel.url)
        cmd = ["rl", "@", str(cfg), "--output-dir", str(out), "--run.name", run_name,
               *resume, *NO_PRIME, *args.split()]
        print("interception server reachable at", tunnel.url, flush=True)
        print("$", " ".join(cmd), flush=True)
        code = subprocess.run(cmd, cwd=KB).returncode
    smi.terminate()
    stop.set()
    sync()
    hf_cache.commit()
    print(f"-> exit {code}", flush=True)


@app.local_entrypoint()
def main(which: str = "image", dry_run: bool = False, args: str = "",
         config: str = "configs/rl_smoke.toml", run_name: str = "") -> None:
    if which in ("smoke", "rl"):
        if dry_run:
            smoke_dry_run.remote()
            return
        # The run name is the resume key: relaunching with the same name continues from its
        # newest checkpoint on the Volume (/runs/rl/<name>); a new name starts fresh.
        name = run_name or Path(config).stem.replace("_", "-")
        # spawn, not remote: a .remote() call stays tied to this local client, and when the
        # client goes away (Mac sleep, expired token) Modal cancels the input, which killed
        # smoke attempt 2 at 08:22. A spawned call runs to completion in the detached app.
        call = rl_run.spawn(config=config, run_name=name, args=args)
        print(f"spawned rl_run {name} ({config}): function call {call.object_id}", flush=True)
        return
    if which == "e2e":
        e2e.remote(dry_run=dry_run, args=args)
        return
    if which == "trainer":
        check_trainer.remote(parts=args or "abc")
        return
    {"image": check, "taskset": check_taskset, "runtime": check_runtime,
     "renderer": check_renderer}[which].remote()
