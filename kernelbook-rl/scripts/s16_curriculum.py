"""Step 16: curriculum task list for RL run 2: trainable tasks most likely to give mixed groups.

Run from the project root, with rl-main-1's trace stream downloaded from the Volume:
    uv run modal volume get kernelbook-runs rl/rl-main-1/monitors/file/traces/stream/00000.jsonl /tmp/
    uv run python scripts/s16_curriculum.py /tmp/00000.jsonl
Reads:  DATA/splits/trainable.txt   the 138 tasks rl-main-1 trained on (s09/s14)
        DATA/baseline.jsonl         baseline: 4 attempts per task (s09)
        <stream>                    rl-main-1: ~8 more attempts per task (2 groups of 4)
Writes: DATA/curriculum.jsonl       per task: baseline + run passes, pooled pass rate p, P(mixed)
        DATA/splits/curriculum.txt  the kept tasks (taskset names)
        TASKS/registry-curriculum.json  dataset "kernelbook-curriculum" (s08 format)

Why: GRPO learns only from mixed groups (some pass, some fail). In rl-main-1, 101 of 267 groups
(38%) were all-pass or all-fail and were dropped: their sandboxes and GPU time taught nothing.
With pass rate p, a group of 4 is mixed with probability 1 - p^4 - (1-p)^4. Keeping tasks with
P(mixed) >= 0.6 (p ~ 0.25..0.75, pooled over ~12 attempts) raises the expected mixed share from
~0.64 to ~0.78, and gives each kept task ~4 visits over 40 steps instead of ~2.
"""

import json
import sys
from collections import defaultdict

from kernel_env.config import DATA, TASKS
from s08_split import write_registry

MIN_P_MIXED = 0.6


def run_attempts(stream: str) -> dict[str, list[float]]:
    out = defaultdict(list)
    with open(stream) as f:
        for line in f:
            r = json.loads(line)
            if not (r.get("ok") and r.get("traces")):
                continue
            t = r["traces"][0]
            reward = t.get("reward")
            if reward is None:
                reward = t["rewards"]["reward"]["score"]
            out[r["task"]["data"]["name"]].append(float(reward))
    return out


def main() -> None:
    trainable = [line.strip() for line in open(DATA / "splits" / "trainable.txt") if line.strip()]
    base = {r["task"]: r for r in map(json.loads, open(DATA / "baseline.jsonl"))}
    run = run_attempts(sys.argv[1])
    rows = []
    for task in trainable:
        b, rr = base[task], run.get(task, [])
        n = b["attempts"] + len(rr)
        p = (b["passes"] + sum(rr)) / n
        rows.append({"task": task, "base_passes": b["passes"], "base_attempts": b["attempts"],
                     "run_passes": int(sum(rr)), "run_attempts": len(rr), "p": round(p, 3),
                     "p_mixed": round(1 - p ** 4 - (1 - p) ** 4, 3)})
    keep = [r for r in rows if r["p_mixed"] >= MIN_P_MIXED]
    (DATA / "curriculum.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (DATA / "splits" / "curriculum.txt").write_text("".join(r["task"] + "\n" for r in keep))
    write_registry("curriculum", sorted(r["task"].replace("kernelbook/", "kernelbook-", 1) for r in keep))
    mean = lambda xs: sum(xs) / len(xs)
    print(f"{len(keep)} of {len(rows)} trainable tasks kept (P(mixed) >= {MIN_P_MIXED}) -> "
          f"{TASKS / 'registry-curriculum.json'} (dataset kernelbook-curriculum)")
    print(f"expected mixed-group share: all {mean([r['p_mixed'] for r in rows]):.2f} -> "
          f"kept {mean([r['p_mixed'] for r in keep]):.2f}")
    print(f"pooled pass rate of kept tasks: {min(r['p'] for r in keep):.2f}..{max(r['p'] for r in keep):.2f}, "
          f"mean {mean([r['p'] for r in keep]):.2f}")


if __name__ == "__main__":
    main()
