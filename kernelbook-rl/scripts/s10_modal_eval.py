"""Step 10: run vf-eval on Modal instead of the Mac (no Prime tunnel, no laptop in the loop).

Why: vf-eval's *interception server* sits between the agent and vLLM. Every model call from a
sandbox goes to it; it forwards the call to vLLM and records the turn (messages, tokens) into
the trace, which training needs. On the Mac, Modal sandboxes could only reach that server
through Prime's tunnel service, which failed twice in the train baseline (and the Mac itself
lost DNS once). Here the host process runs in a Modal container instead, and the server's
port is published with Modal's own `modal.forward`; verifiers' `custom` tunnel just uses
that URL (no tunnel is created).

    Mac (only launches)          Modal container (this file)         Modal sandboxes (A10G)
    modal run --detach  ---->    vf-eval + interception server <---- agent's model calls
                                       |  modal.forward URL            (OPENAI_BASE_URL)
                                       v
                                 vLLM (s09_serve.py, H100)

Run from the project root (vLLM must be deployed; ping it first with scripts/s09_ping.py):
    uv run modal run --detach scripts/s10_modal_eval.py --config configs/baseline_train.toml \
        --args "--select.include.names kernelbook/00456-smoothl1loss --select.skip 0 --select.limit 1 -o work.nosync/outputs/modal_smoke"
    uv run modal run --detach scripts/s10_modal_eval.py --resume RUN_DIR   (RUN_DIR as printed)
Results land in the Modal Volume `kernelbook-runs`; copy them back with:
    uv run modal volume get kernelbook-runs <run path under outputs/> work.nosync/outputs/<same path>
--detach keeps the run going if the Mac sleeps or the terminal closes.
"""

import json
import subprocess
import threading
import time
import tomllib
from pathlib import Path

import modal

PROJECT = Path(__file__).resolve().parent.parent
REMOTE = Path("/root/kb")                 # project root inside the container
OUTPUTS = REMOTE / "work.nosync/outputs"  # = the Volume: traces survive the container
PORT = 8765                               # interception server's fixed port (published below)

app = modal.App("kernelbook-runner")
runs = modal.Volume.from_name("kernelbook-runs", create_if_missing=True)

_skip = ["__pycache__", "**/__pycache__/**", ".DS_Store"]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")                   # verifiers is a git dependency
    .pip_install("uv")
    .add_local_file(PROJECT / "pyproject.toml", f"{REMOTE}/pyproject.toml", copy=True)
    .add_local_file(PROJECT / "uv.lock", f"{REMOTE}/uv.lock", copy=True)
    # The locked versions, minus torch and its CUDA wheels (~4 GB): the host side never runs a
    # kernel (agents and the grader do, in their own sandboxes with the task image).
    .run_commands(
        f"cd {REMOTE} && uv export --frozen --no-dev --no-hashes --no-emit-project"
        " | grep -vE '^(torch|nvidia-|triton)' > /tmp/req.txt"
        " && uv pip install --system -r /tmp/req.txt"
    )
    .env({"PYTHONPATH": str(REMOTE)})     # `kernelbook_taskset` and `kernel_env` importable
    .add_local_dir(PROJECT / "kernel_env", f"{REMOTE}/kernel_env", ignore=_skip)
    .add_local_file(PROJECT / "kernelbook_taskset.py", f"{REMOTE}/kernelbook_taskset.py")
    .add_local_dir(PROJECT / "configs", f"{REMOTE}/configs", ignore=_skip)
    .add_local_dir(PROJECT / "work.nosync/tasks", f"{REMOTE}/work.nosync/tasks", ignore=_skip)
)


def local_registry(registry: str) -> str:
    """Registries list absolute Mac paths; write a copy pointing at the container's tasks.
    Example: /Users/.../work.nosync/tasks/kernelbook-00002-x -> /root/kb/work.nosync/tasks/kernelbook-00002-x"""
    data = json.loads((REMOTE / registry).read_text())
    for dataset in data:
        for task in dataset["tasks"]:
            task["path"] = str(REMOTE / "work.nosync/tasks" / Path(task["path"]).name)
    out = Path("/tmp") / Path(registry).name
    out.write_text(json.dumps(data))
    return str(out)


def keep_committing(stop: threading.Event) -> None:
    """Commit the Volume every 2 min, so traces written so far are visible (and kept) even if
    the container dies; vf-eval appends to traces.jsonl as each episode finishes."""
    while not stop.wait(120):
        runs.commit()


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("kernelbook-vllm")],   # VLLM_API_KEY (the [client] key)
    volumes={str(OUTPUTS): runs},
    timeout=24 * 60 * 60,                 # Modal's maximum; a full 382-task pass took ~30 h on 4 lanes
    cpu=2, memory=4096,                   # host side is I/O: proxying, sandbox control, traces
)
def run(config: str = "", args: str = "", resume: str = "") -> None:
    if resume:   # re-run missing/errored rollouts of an existing run (its own resolved config)
        resolved = OUTPUTS / resume / "configs/resolved/eval.json"
        cmd = ["vf-eval", "@", str(resolved), "--resume"]
        registry = json.loads(resolved.read_text())["env"]["taskset"]["registry_path"]
        # A run started here recorded /tmp/registry-*.json: rebuild that file in this container.
        local_registry(f"work.nosync/tasks/{Path(registry).name}")
    else:
        cmd = ["vf-eval", "@", config, *args.split()]
        registry = tomllib.loads((REMOTE / config).read_text())["env"]["taskset"]["registry_path"]
        cmd += ["--env.taskset.registry_path", local_registry(registry)]

    stop = threading.Event()
    threading.Thread(target=keep_committing, args=(stop,), daemon=True).start()
    with modal.forward(PORT) as tunnel:   # public https URL -> this container's PORT
        if resume:
            # --resume refuses any config change, but the forward URL is new in every container.
            # It is transport only (where sandboxes send model calls), not an experimental
            # setting, so update it in the run's saved config instead of passing an override.
            saved = json.loads(resolved.read_text())
            saved["env"]["interception"]["tunnel"]["url"] = tunnel.url
            resolved.write_text(json.dumps(saved, indent=2))
        else:
            cmd += ["--env.interception.type", "server",
                    "--env.interception.tunnel.type", "custom",
                    "--env.interception.tunnel.url", tunnel.url,
                    "--env.interception.tunnel.port", str(PORT)]
        print("interception server reachable at", tunnel.url, flush=True)
        print("$", " ".join(cmd), flush=True)
        t0 = time.time()
        code = subprocess.run(cmd, cwd=REMOTE).returncode
    stop.set()
    runs.commit()
    print(f"vf-eval exited {code} after {(time.time() - t0) / 60:.0f} min", flush=True)
    if code:
        raise SystemExit(code)


@app.local_entrypoint()
def main(config: str = "", args: str = "", resume: str = "") -> None:
    if not (config or resume):
        raise SystemExit("pass --config configs/X.toml [--args '...'] or --resume RUN_DIR")
    run.remote(config=config, args=args, resume=resume)
