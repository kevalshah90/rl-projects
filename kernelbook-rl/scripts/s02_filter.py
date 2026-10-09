"""Step 2: decide which KernelBook rows become tasks.

Runs on: the Mac, from the project root:  uv run python scripts/s02_filter.py
Writes:  data/kept.jsonl     one full Row per line (the input to step 5)
         data/dropped.jsonl  {uuid, module_name, reason} per line, so every drop is explainable

Filters run cheapest first; a row stops at its first failure:
  F2-F4  static_reason   license, prompt length, input size       (Mac, parse only)
  F1     duplicates      same code ignoring comments/formatting   (Mac, parse only)
  F6     KernelBench     same module class as a held-out problem  (Mac, parse only)
  F5     exec_check      runs, single finite tensor, deterministic (Mac CPU, subprocesses)

F5 executes untrusted code from GitHub. It runs in child processes (kernel_env/exec_worker.py)
with a per-row timeout, inside a temporary directory, so crashes, hangs and stray files
can't affect this script or the project. That contains accidents; it is not a security
sandbox. Modal is reserved for steps that need a GPU.
"""

import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

from kernel_env.config import DATA
from kernel_env.data import Row, load_rows, source_key, static_reason, structure_key
from kernel_env.exec_worker import ROW_TIMEOUT_SEC

KEPT = DATA / "kept.jsonl"
DROPPED = DATA / "dropped.jsonl"
KERNELBENCH_LEVELS = ("level_1", "level_2", "level_3", "level_4")

# ---------- F5: run rows in worker subprocesses ----------

WORKERS = max(1, (os.cpu_count() or 2) - 2)   # leave two cores free for the Mac
BATCH = 25                                    # rows per subprocess: amortizes the ~1 s torch import


def run_batch(batch: list[Row]) -> dict[int, str | None]:
    """Run one batch in a fresh worker process. Returns {uuid: reason} for rows that reported."""
    payload = json.dumps([{"uuid": r.uuid, "python_code": r.python_code,
                           "module_name": r.module_name} for r in batch])
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}  # 1 thread per worker
    with tempfile.TemporaryDirectory() as scratch:          # rows that write files write here
        try:
            proc = subprocess.run([sys.executable, "-m", "kernel_env.exec_worker"],
                                  input=payload, capture_output=True, text=True, cwd=scratch,
                                  env=env, timeout=len(batch) * ROW_TIMEOUT_SEC + 60)
            stdout = proc.stdout
        except subprocess.TimeoutExpired as e:               # the whole worker hung
            stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
    results = {}
    for line in stdout.splitlines():                          # rows may print too; skip non-results
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and "uuid" in record:
            results[record["uuid"]] = record["reason"]
    return results


def exec_check_all(rows: list[Row]) -> dict[int, str | None]:
    """F5 for every row, in parallel batches. A row that never reported (its worker crashed
    or hung on it or an earlier row) is retried ALONE, so one bad row can't sink its batch."""
    batches = [rows[i:i + BATCH] for i in range(0, len(rows), BATCH)]
    results: dict[int, str | None] = {}
    with ThreadPoolExecutor(WORKERS) as pool:                 # threads just wait on subprocesses
        for done, batch_results in enumerate(pool.map(run_batch, batches), 1):
            results.update(batch_results)
            print(f"\r  exec_check: {done}/{len(batches)} batches", end="", flush=True)
    missing = [r for r in rows if r.uuid not in results]
    print(f"\n  retrying {len(missing)} unreported rows one at a time")
    with ThreadPoolExecutor(WORKERS) as pool:
        for row, single in zip(missing, pool.map(lambda r: run_batch([r]), missing)):
            results[row.uuid] = single.get(row.uuid, "exec: worker crashed or hung")
    return results

# ---------- F6 helper ----------

def kernelbench_keys() -> set[str]:
    """structure_key of every KernelBench problem (its class is always named `Model`)."""
    from datasets import load_dataset
    keys = set()
    for level in KERNELBENCH_LEVELS:
        for record in load_dataset("ScalingIntelligence/KernelBench", split=level):
            if key := structure_key(record["code"], "Model"):
                keys.add(key)
    return keys

# ---------- main ----------

def main() -> None:
    rows = sorted(load_rows(), key=lambda r: r.uuid)   # lowest uuid first: it wins duplicates
    dropped: dict[int, str] = {}                      # uuid -> reason

    # F2-F4: license, length, input size
    for row in rows:
        if reason := static_reason(row):
            dropped[row.uuid] = reason

    # F1: duplicates. Only rows still kept compete, so a dropped row can't "claim" a key.
    first_seen: dict[str, int] = {}
    for row in rows:
        if row.uuid in dropped:
            continue
        key = source_key(row.python_code)
        if key in first_seen:
            dropped[row.uuid] = f"duplicate: same code as uuid {first_seen[key]}"
        else:
            first_seen[key] = row.uuid

    # F6: overlap with KernelBench, our held-out eval set
    held_out = kernelbench_keys()
    for row in rows:
        if row.uuid not in dropped and structure_key(row.python_code, row.module_name) in held_out:
            dropped[row.uuid] = "kernelbench: same module as a KernelBench problem"

    # F5: execute the survivors locally, in worker subprocesses
    todo = [row for row in rows if row.uuid not in dropped]
    print(f"Static filters done: {len(todo):,} rows left. Running exec_check "
          f"with {WORKERS} workers...")
    for uuid, reason in exec_check_all(todo).items():
        if reason is not None:
            dropped[uuid] = reason

    # Write both files, then print a summary grouped by reason category
    KEPT.parent.mkdir(parents=True, exist_ok=True)
    with KEPT.open("w") as kept_file, DROPPED.open("w") as dropped_file:
        for row in rows:
            if row.uuid in dropped:
                record = {"uuid": row.uuid, "module_name": row.module_name, "reason": dropped[row.uuid]}
                dropped_file.write(json.dumps(record) + "\n")
            else:
                kept_file.write(json.dumps(asdict(row)) + "\n")

    by_category = Counter(reason.split(":")[0] for reason in dropped.values())
    print(f"\nKept {len(rows) - len(dropped):,} of {len(rows):,} rows -> {KEPT}")
    for category, count in by_category.most_common():
        print(f"  dropped {count:>6,}  {category}")


if __name__ == "__main__":
    main()
