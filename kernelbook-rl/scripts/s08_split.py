"""Step 8: split the validated tasks into train and dev.

Run from the project root (after step 6):  uv run python scripts/s08_split.py
Reads:  DATA/validation.jsonl (tasks with keep: true), each task's task.toml (module name)
Writes: DATA/splits/train.txt, DATA/splits/dev.txt          one task id per line
        TASKS/registry-train.json, TASKS/registry-dev.json  Harbor registries (one dataset each),
                                                            loaded later by verifiers / prime-rl

Split by GROUP, not by task: a group is the module name, lowercased. KernelBook often has
several rows of the same module from different repos (e.g. tasks 3 and 4 are both LayerNorm);
keeping them together means dev measures modules the model never trained on.

Each group's split comes from a hash of its name:  dev if sha256(name) % 10 == 0, else train.
The hash is effectively a fair 10-sided die that always lands the same way for the same name,
so ~10% of groups go to dev, the split is identical on every run, and groups added later
never move an existing group (no silent dev -> train leakage).

Not done here: filtering train by difficulty (needs step 9's baseline pass rates) and a
KernelBench eval set (skipped for this toy project; dev is the held-out set).
"""

import hashlib
import json
import tomllib
from collections import Counter

from kernel_env.config import DATA, TASKS

DEV_BUCKETS = 10                 # 1 bucket out of 10 -> ~10% of groups in dev
SPLITS = DATA / "splits"


def group_of(task: str) -> str:
    """The module name of a task, lowercased: all tasks of the same module share a group."""
    meta = tomllib.loads((TASKS / task / "task.toml").read_text())["metadata"]
    return meta["module_name"].lower()


def split_of(group: str) -> str:
    """dev if the name's sha256, read as a number, is divisible by DEV_BUCKETS; else train."""
    return "dev" if int(hashlib.sha256(group.encode()).hexdigest(), 16) % DEV_BUCKETS == 0 else "train"


def write_registry(name: str, tasks: list[str]) -> None:
    """A Harbor registry with one dataset, same format as s05's registry.json."""
    registry = [{"name": f"kernelbook-{name}", "version": "0.1",
                 "description": f"KernelBook Triton tasks, {name} split",
                 "tasks": [{"name": t, "path": str((TASKS / t).resolve())} for t in tasks]}]
    (TASKS / f"registry-{name}.json").write_text(json.dumps(registry, indent=2))


def main() -> None:
    kept = [r["task"] for r in map(json.loads, open(DATA / "validation.jsonl")) if r["keep"]]
    groups = {task: group_of(task) for task in kept}
    splits = {"train": [], "dev": []}
    for task in sorted(kept):
        splits[split_of(groups[task])].append(task)

    # No module group may appear in both splits: that is the whole point of grouping.
    train_groups = {groups[t] for t in splits["train"]}
    dev_groups = {groups[t] for t in splits["dev"]}
    assert not train_groups & dev_groups, f"groups in both splits: {train_groups & dev_groups}"
    assert len(splits["train"]) + len(splits["dev"]) == len(kept)

    SPLITS.mkdir(parents=True, exist_ok=True)
    for name, tasks in splits.items():
        (SPLITS / f"{name}.txt").write_text("".join(t + "\n" for t in tasks))
        write_registry(name, tasks)

    print(f"{len(kept)} kept tasks in {len(set(groups.values()))} groups")
    for name, tasks in splits.items():
        n_groups = len(train_groups if name == "train" else dev_groups)
        print(f"  {name:<5} {len(tasks):>4} tasks ({100 * len(tasks) / len(kept):.1f}%) "
              f"in {n_groups} groups -> {SPLITS / (name + '.txt')}, registry-{name}.json")
    print("largest groups:")
    for group, count in Counter(groups.values()).most_common(5):
        print(f"  {group:<24} {count} tasks -> {split_of(group)}")


if __name__ == "__main__":
    main()
