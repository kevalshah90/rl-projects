"""Step 9: a verifiers taskset that reads our Harbor tasks from local folders.

Why it exists: verifiers' built-in `harbor` taskset always runs `harbor download --export`
and reads the exported copy. For a registry whose tasks are LOCAL paths (ours, from
s05/s08), Harbor uses the folders in place and exports nothing, so verifiers fails
renaming an export folder that was never created.
This subclass changes only that: it reads the task folders listed in the registry.
Parsing each task (task.toml, MCP servers, GPU, timeouts) and grading (HarborEnv's
separate verifier box) are verifiers' own code, unchanged.

Use:  [env.taskset] id = "kernelbook_taskset"   (verifiers imports this module by that name;
      it must be top-level, and importable because the editable install puts the project
      root on sys.path)
Example: registry-dev.json, dataset "kernelbook-dev" -> 54 HarborTasks, the first is
         "kernelbook/00002-customizelayer" (image kernel-env-base, L4, check tool).
"""

import json
from pathlib import Path

from verifiers.v1.tasksets.harbor import HarborEnv, HarborTask, HarborTaskset
from verifiers.v1.tasksets.harbor.taskset import parse_task


class KernelBookTaskset(HarborTaskset):
    def load(self):
        """Yield one HarborTask per task folder in the registry's dataset (same order)."""
        registry = json.loads(self.config.registry_path.read_text())
        (dataset,) = [d for d in registry if d["name"] == self.config.dataset]
        for idx, entry in enumerate(dataset["tasks"]):
            yield HarborTask(parse_task(Path(entry["path"]), idx, self.config), self.config.task)


# verifiers finds the taskset and its env through __all__. Exporting HarborEnv keeps
# separate grading: without it, verifiers would fall back to a single-box env.
__all__ = ["KernelBookTaskset", "HarborEnv"]
