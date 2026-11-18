"""Step 14: a Harbor registry of the trainable tasks, for the RL runs.

Run from the project root (after step 9's train baseline):  uv run python scripts/s14_trainable_registry.py
Reads:  DATA/splits/trainable.txt   train tasks with 0 < baseline pass rate < 1 (s09), as
                                    taskset names ("kernelbook/00000-sumaggregator")
Writes: TASKS/registry-trainable.json  dataset "kernelbook-trainable", same format as s08's

Only these tasks can give GRPO a learning signal at the start: a group of 4 rollouts that all
pass or all fail has zero advantage. Dev tasks never appear here (s09 only reads train runs).
"""

from kernel_env.config import DATA, TASKS
from s08_split import write_registry


def main() -> None:
    names = [line.strip() for line in open(DATA / "splits" / "trainable.txt") if line.strip()]
    tasks = [n.replace("kernelbook/", "kernelbook-", 1) for n in names]   # taskset name -> task dir
    missing = [t for t in tasks if not (TASKS / t / "task.toml").exists()]
    assert not missing, f"no task dir for {missing[:5]}"
    train = set(open(DATA / "splits" / "train.txt").read().split())
    assert set(tasks) <= train, f"not in the train split: {sorted(set(tasks) - train)[:5]}"
    write_registry("trainable", sorted(tasks))
    print(f"{len(tasks)} trainable tasks -> {TASKS / 'registry-trainable.json'} (dataset kernelbook-trainable)")


if __name__ == "__main__":
    main()
