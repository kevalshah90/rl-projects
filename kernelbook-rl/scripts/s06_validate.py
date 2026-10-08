"""Step 6: validate every task folder through Harbor on Modal; drop tasks whose answer key fails.

Run from the project root (after step 5):  uv run python scripts/s06_validate.py
Reads:  tasks/<id>/ (every task folder)
Writes: data/validation.jsonl, one line per task:
        {"task": ..., "oracle": 1.0, "nop": 0.0, "keep": true, "reason": null}
        Harbor's own job folders: outputs/harbor/validate-oracle/, outputs/harbor/validate-nop/

Two Harbor runs over the same tasks, each a real trial (agent container, then a fresh grader):
  oracle  writes the task's reference Triton code (solution/solve.sh)  -> must score 1
  nop     writes nothing (solution.py stays empty)                       -> must score 0
Keep a task only if both hold. An oracle that fails means the task's answer key is wrong,
so no model could ever score on it: training on it would be pure noise.

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

import json
import os
import shutil
import subprocess
from pathlib import Path

import certifi

TASKS = Path("tasks")
JOBS = Path("outputs/harbor")
OUT = Path("data/validation.jsonl")
CONCURRENT = 4               # trials at once (each uses an agent L4, then a grader L4)
SANDBOX_TIMEOUT = 1800       # Harbor's default sandbox lifetime is 24 h: cap it at 30 min


def run_harbor(agent: str) -> Path:
    """One Harbor job: `agent` on every task folder, on Modal. Returns the job folder."""
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
    """Harbor sandboxes still running on Modal (they would keep billing)."""
    listing = subprocess.run(["modal", "container", "list", "--json"],
                             capture_output=True, text=True).stdout
    return sum(c.get("App Name") == "__harbor__" for c in json.loads(listing or "[]"))


def main() -> None:
    tasks = sorted(d.name for d in TASKS.iterdir() if d.is_dir())
    print(f"Validating {len(tasks)} tasks: oracle, then nop ({CONCURRENT} trials at a time)")
    oracle, nop = read_trials(run_harbor("oracle")), read_trials(run_harbor("nop"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
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
            out.write(json.dumps({"task": task, "oracle": o.get("reward"), "nop": n.get("reward"),
                                  "keep": reason is None, "reason": reason}) + "\n")
            if reason:
                print(f"  DROP {task}: {reason[:120]}")

    print(f"\nKept {kept}/{len(tasks)} tasks -> {OUT}")
    if (left := leftover_containers()):
        print(f"WARNING: {left} Harbor sandbox(es) still running on Modal. "
              f"Stop them: modal container list, then modal container stop -y <id>")


if __name__ == "__main__":
    main()
