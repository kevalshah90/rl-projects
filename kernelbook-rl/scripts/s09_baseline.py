"""Step 9: summarize a vf-eval run into per-task pass rates.

Run from the project root (after `vf-eval @ configs/baseline_train.toml` or `configs/eval_dev.toml`):
    uv run python scripts/s09_baseline.py [RUN_DIR]     # default: newest run under outputs/baseline_train
Reads:  RUN_DIR/traces.jsonl        one JSON trace per episode, written by vf-eval
Writes: DATA/baseline.jsonl         one line per task: attempts, passes, pass_rate, turns, tokens
        DATA/splits/trainable.txt   TRAIN tasks with 0 < pass_rate < 1 (dev tasks never go here)

Why "trainable": GRPO learns from the DIFFERENCE between attempts at the same task.
If all attempts pass (or all fail), every advantage is 0 and the task teaches nothing.

Example line in baseline.jsonl:
    {"task": "kernelbook/00042-...", "attempts": 2, "passes": 1, "pass_rate": 0.5,
     "errors": 0, "mean_turns": 7.5, "mean_output_tokens": 9120, "mean_seconds": 312.4}
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from kernel_env.config import DATA, OUTPUTS

PASS = 1.0                       # grade.py gives exactly 1.0 only when all 7 gates pass


def summarize(episode: dict) -> dict:
    """The few numbers we need from one episode (the full trace has every message and token).

    Each traces.jsonl line is an EPISODE: {"task", "ok", "errors", "traces": [trace], ...}.
    Our env has one agent, so `traces` holds exactly one trace (the agent's).
    Example trace fields:  task.data.name, rewards {"solved": {"score": 1.0, "weight": 1.0}},
                           ok, stop_condition, nodes (sampled = model turns), timing
    Example output:        {"task": "kernelbook/00042-...", "reward": 1.0, "ok": True,
                            "stop": None, "turns": 8, "output_tokens": 9120, "seconds": 301.2}
    """
    (trace,) = episode["traces"]
    rewards = [r["score"] * r["weight"] for r in trace["rewards"].values() if r is not None]
    timing = trace["timing"]
    # Last phase that actually ran (an errored episode never reaches scoring; unrun phases are 0).
    end = max(timing[p]["end"] for p in ("boot", "setup", "agent", "finalize", "scoring"))
    return {
        "task": trace["task"]["data"]["name"],
        "reward": sum(rewards),                      # scoring never ran (error) -> 0
        "ok": episode["ok"],                         # False: infra or harness error, not a fail
        "stop": trace["stop_condition"],             # why it ended, e.g. "max_turns", "sandbox_error"
        "turns": sum(1 for n in trace["nodes"] if n.get("sampled")),
        "output_tokens": episode["num_output_tokens"],
        "seconds": end - timing["start"],
    }


def main() -> None:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else max(
        (OUTPUTS / "baseline_train").glob("*/traces.jsonl"), key=lambda p: p.stat().st_mtime).parent
    episodes = [summarize(json.loads(line)) for line in open(run / "traces.jsonl")]

    by_task = defaultdict(list)
    for e in episodes:
        by_task[e["task"]].append(e)

    rows = []
    for task, all_eps in sorted(by_task.items()):
        # Errored episodes (sandbox, tunnel, ...) never got a fair attempt: not counted as fails.
        eps = [e for e in all_eps if e["ok"]]
        if not eps:
            continue
        n = len(eps)
        passes = sum(e["reward"] >= PASS for e in eps)
        rows.append({"task": task, "attempts": n, "passes": passes, "pass_rate": passes / n,
                     "errors": len(all_eps) - n,
                     "mean_turns": sum(e["turns"] for e in eps) / n,
                     "mean_output_tokens": sum(e["output_tokens"] for e in eps) / n,
                     "mean_seconds": sum(e["seconds"] for e in eps) / n})

    (DATA / "baseline.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    # Only train tasks may be selected for training: a dev (eval) run must never feed this list.
    # Trace names are "kernelbook/00034-simplemlp"; split files hold folder names "kernelbook-00034-...".
    train = set((DATA / "splits" / "train.txt").read_text().split())
    trainable = [r["task"] for r in rows
                 if 0 < r["pass_rate"] < 1 and r["task"].replace("/", "-") in train]
    if all(r["task"].replace("/", "-") in train for r in rows):
        (DATA / "splits" / "trainable.txt").write_text("".join(t + "\n" for t in trainable))
    else:
        print("not a train-only run: trainable.txt left unchanged")

    print(f"run: {run}\n{len(episodes)} episodes; {len(rows)} of {len(by_task)} tasks "
          f"have at least one error-free attempt")
    for label, keep in [("always fail (p=0)", lambda p: p == 0),
                        ("mixed (0<p<1)  <- trainable", lambda p: 0 < p < 1),
                        ("always pass (p=1)", lambda p: p == 1)]:
        print(f"  {label:<30} {sum(keep(r['pass_rate']) for r in rows)}")
    print(f"errored episodes: {sum(not e['ok'] for e in episodes)}  "
          f"stop reasons: {dict(Counter(e['stop'] for e in episodes))}")
    print(f"mean per episode: {sum(e['turns'] for e in episodes) / len(episodes):.1f} turns, "
          f"{sum(e['output_tokens'] for e in episodes) / len(episodes):.0f} output tokens, "
          f"{sum(e['seconds'] for e in episodes) / len(episodes):.0f} s")


if __name__ == "__main__":
    main()
