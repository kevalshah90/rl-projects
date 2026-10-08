"""Grade one submission against its reference. Always writes reward.json + details.json.

Runs on: GPU, in the grader's container. Each task's tests/test.sh runs:
    python -m kernel_env.grade --reference /tests/reference.py \
        --solution /workspace/solution.py --out-dir /logs/verifier
The module name comes from $MODULE_NAME, set per task in task.toml's [verifier.env].

Gates, in order; the first failure stops grading. reward = 1 only if all seven pass.
  1 exists       solution.py is there and not empty
  2 guard        no banned pattern in the source text            (guard.py)
  3 parse        solution imports and defines <Module>New
  4 weights      the reference's weights load into it by name    (strict)
  5 runs         forward() completes without raising
  6 uses_triton  forward() launches at least one Triton kernel   (triton_hook.py)
  7 correct      same shape, finite, allclose on 3 input seeds
Separate gates show WHERE a model is stuck during RL: crashing kernels (runs) need
syntax/indexing fixes; running-but-wrong kernels (correct) need logic fixes.
"""

import argparse
import importlib.util
import json
import os
import traceback
from pathlib import Path

import torch

from kernel_env.guard import scan
from kernel_env.triton_hook import count_launches

GATES = ("exists", "guard", "parse", "weights", "runs", "uses_triton", "correct")
SEEDS = (0, 1, 2)            # 3 different random inputs per submission
RTOL, ATOL = 1e-3, 1e-5      # step 3: correct kernels differ by <= 3e-7, lots of headroom


def load_module(path: str, name: str):
    """Import a .py file by path. Must be a real file: @triton.jit reads its source from it."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gpu(xs):    # move tensors to the GPU, leave other arguments alone
    return [x.cuda() if torch.is_tensor(x) else x for x in xs]


def clone(xs):  # fresh copies: a forward() may modify its inputs in place
    return [x.clone() if torch.is_tensor(x) else x for x in xs]


def grade(reference_path: str, solution_path: str, module_name: str) -> dict:
    result = {gate: 0 for gate in GATES} | {
        "reward": 0.0, "kernel_launches": 0, "max_abs_diff": -1.0,
        "failed_gate": None, "message": None}

    def fail(gate: str, message: str) -> dict:
        result.update(failed_gate=gate, message=message)
        return result

    # 1. exists
    source = Path(solution_path).read_text() if Path(solution_path).exists() else ""
    if not source.strip():
        return fail("exists", "solution.py is missing or empty")
    result["exists"] = 1

    # 2. guard: before any solution code runs
    if reason := scan(source):
        return fail("guard", reason)
    result["guard"] = 1

    # Build the reference (trusted code) once.
    ref = load_module(reference_path, "reference")
    args, kwargs = ref.get_init_inputs()                 # KernelBook format: [[args], {kwargs}]
    torch.manual_seed(0)
    ref_model = getattr(ref, module_name)(*args, **kwargs).cuda().eval()
    # Weights made with torch.empty / torch.Tensor(n) are uninitialized memory: on a GPU they
    # can be NaN or 1e30, and NaN != NaN fails allclose even for a perfect kernel (step 6).
    # The grader only needs both sides to share sane weights (a correct kernel works for any
    # weights), so replace those with small seeded random values before copying them over.
    for t in [*ref_model.parameters(), *ref_model.buffers()]:
        if t.is_floating_point() and t.numel() and (not torch.isfinite(t).all() or t.abs().max() > 1e3):
            t.data = torch.randn_like(t) * 0.1

    # 3. parse: importing the solution runs its code; untrusted from here on.
    try:
        sol = load_module(solution_path, "solution")
        sol_model = getattr(sol, module_name + "New")(*args, **kwargs).cuda().eval()
    except Exception as e:
        return fail("parse", f"{type(e).__name__}: {e}")
    result["parse"] = 1

    # 4. weights: identical weights, so only the computation can differ (see step 3).
    try:
        sol_model.load_state_dict(ref_model.state_dict(), strict=True)
    except Exception as e:
        return fail("weights", f"parameter names must match the reference: {e}")
    result["weights"] = 1

    # 5-7, for each input seed
    for seed in SEEDS:
        torch.manual_seed(seed)
        inputs = gpu(ref.get_inputs())
        with torch.no_grad():
            expected = ref_model(*clone(inputs))
            try:                                                   # 5. runs
                with count_launches() as launches:
                    got = sol_model(*clone(inputs))
            except Exception as e:
                return fail("runs", f"forward() crashed at seed {seed}: {type(e).__name__}: {e}")
        result["runs"] = 1

        result["kernel_launches"] = launches["n"]
        if launches["n"] == 0:                                     # 6. uses_triton
            return fail("uses_triton", "forward() launched no Triton kernel")
        result["uses_triton"] = 1

        if not torch.is_tensor(got) or got.shape != expected.shape:   # 7. correct
            shape = tuple(got.shape) if torch.is_tensor(got) else type(got).__name__
            return fail("correct", f"output {shape} != expected {tuple(expected.shape)}")
        diff = (got.float() - expected.float()).abs().max().item()
        result["max_abs_diff"] = max(result["max_abs_diff"], diff)
        if not torch.isfinite(got).all() or not torch.allclose(got, expected, rtol=RTOL, atol=ATOL):
            return fail("correct", f"max difference {diff:.3g} at seed {seed}")

    result.update(correct=1, reward=1.0)
    return result


def write_outputs(result: dict, out_dir: str) -> None:
    """reward.json: numbers only (verifiers rejects text). details.json: everything."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    numbers = {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
    (out / "reward.json").write_text(json.dumps(numbers))
    (out / "details.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--solution", required=True)
    p.add_argument("--module-name", default=os.environ.get("MODULE_NAME"))
    p.add_argument("--out-dir", default="/logs/verifier")
    a = p.parse_args()
    try:
        result = grade(a.reference, a.solution, a.module_name)
    except Exception:     # a grader crash must still produce a reward file: 0, not "missing"
        result = {"reward": 0.0, "failed_gate": "grader", "message": traceback.format_exc()[-500:]}
    write_outputs(result, a.out_dir)
