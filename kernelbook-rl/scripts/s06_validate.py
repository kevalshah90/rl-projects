"""Step 6: validate every task folder; drop tasks whose answer key fails.

Run from the project root (after step 5):
    uv run python scripts/s06_validate.py                  # direct mode (default): bulk, cheap
    uv run python scripts/s06_validate.py --mode harbor    # full Harbor trials: plumbing checks
Reads:  TASKS/<id>/ (every task folder; paths in kernel_env/config.py)
Writes: DATA/validation.jsonl, one line per task:
        {"task": ..., "mode": "direct", "oracle": 1.0, "nop": 0.0, "keep": true, "reason": null}

Every task is graded twice:
  oracle  the task's reference Triton code (from solution/solve.sh)  -> must score 1
  nop     an empty solution                                           -> must score 0
Keep a task only if both hold. An oracle that fails means the task's answer key is wrong,
so no model could ever score on it: training on it would be pure noise.

Two modes, same grader (kernel_env.grade), same keep/drop rule:
  direct  A Modal function grades each task's own files on an L4: at most 4 containers,
          each reused across many tasks. No agent sandbox.
  harbor  Real Harbor trials: an agent sandbox (oracle/nop agent), then a fresh grader
          sandbox, 2 L4s per trial. Proves Harbor's plumbing (uploads, artifact copy,
          separate grader, MODULE_NAME); proven on 20 tasks, so bulk runs use direct.
  Why direct for bulk: the oracle/nop agents never use the agent sandbox's GPU, and Harbor's
  plumbing is identical for every task. What varies per task (reference, answer key, module
  name, weights) is exactly what direct mode checks. About 50x cheaper.

Expected drop rate (first 20 tasks: 4 dropped). Most drops are a KernelBook bug, not ours:
in modules with 2+ same-shaped weights, the oracle's forward() wrapper passes the weights to
the compiled kernel in the wrong order, e.g. ScalarBiasScale (uuid 25):
    primals_1 = self.weight   # the compiled kernel expects the BIAS here
    primals_3 = self.bias     # ...and the WEIGHT here
A debug run confirmed it: with the two weights swapped the oracle matches the reference
exactly (max diff 0), as it does for GatAttention (uuid 17). The task itself is valid, but
without a working answer key we can't prove it's solvable, so we drop it (option A). Repairing
the wrappers by trying weight orders (option B) could recover these tasks later.
"""

import argparse
import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import modal

from kernel_env.config import BASE_IMAGE, DATA, OUTPUTS, TASKS

JOBS = OUTPUTS / "harbor"
OUT = DATA / "validation.jsonl"
CONCURRENT = 4               # harbor mode: trials at once (each uses an agent L4, then a grader L4)
SANDBOX_TIMEOUT = 1800       # harbor mode: Harbor's default sandbox lifetime is 24 h, cap it at 30 min
MAX_CONTAINERS = 4           # direct mode: L4 containers at once, each reused across many tasks


# ---------- direct mode: grade each task's own files with a Modal function ----------

app = modal.App("kernelbook-s06-validate")
image = modal.Image.from_registry(BASE_IMAGE).add_local_python_source("kernel_env")
ORACLE_START = "<<'KERNELBOOK_ORACLE_EOF'\n"            # solve.sh wraps the oracle in this heredoc
ORACLE_END = "\nKERNELBOOK_ORACLE_EOF"


def read_task(task_dir: Path) -> dict:
    """The three things grade() needs, read from the task folder exactly as Harbor would see them."""
    solve = (task_dir / "solution" / "solve.sh").read_text()
    return {
        "task": task_dir.name,
        "reference": (task_dir / "tests" / "reference.py").read_text(),
        "oracle": solve.split(ORACLE_START, 1)[1].rsplit(ORACLE_END, 1)[0],
        "module_name": tomllib.loads((task_dir / "task.toml").read_text())["verifier"]["env"]["MODULE_NAME"],
    }


@app.function(image=image, gpu="L4", timeout=600, max_containers=MAX_CONTAINERS)
def grade_task(task: dict) -> dict:
    """On an L4: grade the task's oracle (should be 1) and an empty solution (should be 0)."""
    import tempfile

    from kernel_env.grade import grade

    folder = Path(tempfile.mkdtemp())                     # real files: @triton.jit needs one
    (folder / "reference.py").write_text(task["reference"])
    results = {}
    for agent, code in [("oracle", task["oracle"]), ("nop", "")]:
        (folder / f"{agent}.py").write_text(code)
        try:
            r = grade(str(folder / "reference.py"), str(folder / f"{agent}.py"), task["module_name"])
        except Exception as e:                            # reference itself crashed: like test.sh, reward 0
            r = {"reward": 0.0, "failed_gate": "grader", "message": f"{type(e).__name__}: {e}"}
        results[agent] = {k: r.get(k) for k in ("reward", "failed_gate", "message")}
    return {"task": task["task"], **results}


def run_direct(task_names: list[str]) -> tuple[dict, dict]:
    """Grade every task on Modal. Returns {task: {reward, failed_gate, message}} for oracle and nop."""
    payloads = [read_task(TASKS / name) for name in task_names]
    oracle, nop = {}, {}
    with modal.enable_output(), app.run():
        results = grade_task.map(payloads, order_outputs=False, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):                  # container-level failure: logged, task dropped
                print(f"  container error: {r!r}"[:200])
                continue
            oracle[r["task"]], nop[r["task"]] = r["oracle"], r["nop"]
    return oracle, nop


# ---------- harbor mode: real Harbor trials on Modal sandboxes ----------

def run_harbor(agent: str) -> Path:
    """One Harbor job: `agent` on every task folder, on Modal. Returns the job folder."""
    import certifi                                        # local only: not needed in Modal containers

    job = JOBS / f"validate-{agent}"
    shutil.rmtree(job, ignore_errors=True)               # fresh results, no resumed job
    # SSL_CERT_FILE: this python.org Python has no CA bundle of its own (step 6 finding),
    # so point it at certifi's; otherwise Modal's sandbox channel fails SSL verification.
    env = {**os.environ, "SSL_CERT_FILE": certifi.where()}
    subprocess.run(
        ["harbor", "run", "--path", str(TASKS), "--agent", agent, "--env", "modal",
         "--n-concurrent", str(CONCURRENT), "--ek", f"sandbox_timeout_secs={SANDBOX_TIMEOUT}",
         "--jobs-dir", str(JOBS), "--job-name", job.name, "--yes"],
        env=env, check=False)                            # failed trials are data, not a crash
    return job


def read_trials(job: Path) -> dict[str, dict]:
    """task folder name -> {"reward", "failed_gate", "message"} for each trial in a Harbor job."""
    trials = {}
    for result_file in job.glob("*/result.json"):
        result = json.loads(result_file.read_text())
        # task_name "kernelbook/00016-fullyconnectednet" -> folder "kernelbook-00016-fullyconnectednet".
        # Not trial_name: Harbor shortens it ("kernelbook-00016-fullyconnectedn__YeVyHZh").
        task = result["task_name"].replace("kernelbook/", "kernelbook-", 1)
        rewards = (result.get("verifier_result") or {}).get("rewards") or {}
        details_file = result_file.parent / "verifier" / "details.json"
        details = json.loads(details_file.read_text()) if details_file.exists() else {}
        error = result.get("exception_info")             # Harbor-level failure (sandbox, timeout, ...)
        trials[task] = {
            "reward": rewards.get("reward"),             # None if the grader never ran
            "failed_gate": details.get("failed_gate") or ("harbor" if error else None),
            "message": details.get("message") or (str(error)[:200] if error else None),
        }
    return trials


def leftover_containers() -> int:
    """Modal containers still running on the account after the run (they would keep billing)."""
    listing = subprocess.run(["modal", "container", "list", "--json"],
                             capture_output=True, text=True).stdout
    return len(json.loads(listing or "[]"))


# ---------- main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["direct", "harbor"], default="direct",
                        help="direct: grader only, cheap (bulk). harbor: full Harbor trials (plumbing checks).")
    mode = parser.parse_args().mode

    # Task folder names never contain spaces; skip any that do (e.g. iCloud "name 2" copies).
    tasks = sorted(d.name for d in TASKS.iterdir() if d.is_dir() and " " not in d.name)
    print(f"Validating {len(tasks)} tasks in {mode} mode")
    if mode == "direct":
        oracle, nop = run_direct(tasks)
    else:
        oracle, nop = read_trials(run_harbor("oracle")), read_trials(run_harbor("nop"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    kept, drops = 0, []
    with OUT.open("w") as out:
        for task in tasks:
            o, n = oracle.get(task, {}), nop.get(task, {})
            if o.get("reward") != 1.0:
                reason = f"oracle failed at {o.get('failed_gate')}: {o.get('message')}"
            elif n.get("reward") != 0.0:
                reason = f"nop scored {n.get('reward')}: empty submission not rejected"
            else:
                reason = None
            kept += reason is None
            if reason:
                drops.append(reason)
            out.write(json.dumps({"task": task, "mode": mode, "oracle": o.get("reward"),
                                  "nop": n.get("reward"), "keep": reason is None,
                                  "reason": reason}) + "\n")

    print(f"\nKept {kept}/{len(tasks)} tasks -> {OUT}")
    by_gate = {}
    for reason in drops:                                  # e.g. "oracle failed at correct: ..."
        gate = reason.split(":")[0]
        by_gate[gate] = by_gate.get(gate, 0) + 1
    for gate, count in sorted(by_gate.items(), key=lambda kv: -kv[1]):
        print(f"  dropped {count:>4}  {gate}")
    if (left := leftover_containers()):
        print(f"WARNING: {left} Modal container(s) still running. "
              f"Stop them: modal container list, then modal container stop -y <id>")


if __name__ == "__main__":
    main()
