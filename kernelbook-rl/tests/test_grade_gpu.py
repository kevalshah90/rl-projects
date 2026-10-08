"""Step 4 check: the grader on submissions with known answers, on one Modal L4.

Run from the project root:  uv run modal run tests/test_grade_gpu.py

What it does
  Grades 7 submissions whose outcome we already know, using the real grader
  (kernel_env.grade.grade) in the real task image on a GPU, and checks each one stops
  at the expected gate. If all 7 match, the grader catches what it's meant to catch.
  1. On the Mac: build the 7 cases. References come from data/kept.jsonl; solutions are
     the strings below or KernelBook's own oracle code.
  2. On one L4: for each case, write reference + solution to real .py files
     (@triton.jit needs a file) and call grade(). One container, cases in order.
  3. On the Mac: print OK/BAD per case; exit 1 if any case misbehaves.

Input: one case (a dict sent to the GPU)
    {"name": "wrong_output",
     "reference": "class SumAggregator(nn.Module): ... torch.sum(neighbor, dim=1) ...",
     "solution":  "...@triton.jit ... tl.store(out_ptr + offs, 2 * acc, mask=mask) ...",
     "module_name": "SumAggregator",
     "expect": "correct"}                  # gate it should fail at; None = should pass

Output: the same dict plus grade()'s result, e.g.
    {..., "reward": 0.0, "exists": 1, "guard": 1, "parse": 1, "weights": 1, "runs": 1,
     "uses_triton": 1, "correct": 0, "kernel_launches": 1, "max_abs_diff": 3.05,
     "failed_gate": "correct", "message": "max difference 3.05 at seed 0"}

Printed (real output from the first run, 33 s on one L4):
    OK  good_triton    expect=None         got=None         launches=1
    OK  crashes        expect=runs         got=runs         launches=0 forward() crashed at seed 0: TypeError: dynamic_func() missi
    OK  noop_torch     expect=uses_triton  got=uses_triton  launches=0 forward() launched no Triton kernel
    OK  wrong_output   expect=correct      got=correct      launches=1 max difference 3.05 at seed 0
    OK  hack_timer     expect=guard        got=guard        launches=0 patches a timer
    OK  good_weighted  expect=None         got=None         launches=2
    OK  bad_oracle     expect=correct      got=correct      launches=6 max difference 0.498 at seed 0

    7/7 as expected
"""

import json

import modal

from kernel_env.config import BASE_IMAGE

app = modal.App("kernelbook-test-grade")
image = modal.Image.from_registry(BASE_IMAGE).add_local_python_source("kernel_env")

# A correct Triton kernel for row 0, SumAggregator = torch.sum(x, dim=1).
GOOD = '''
import torch, torch.nn as nn, triton, triton.language as tl

@triton.jit
def k(x_ptr, out_ptr, n_out, D1, R, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)   # output elements of x viewed as [B, D1, R]
    mask = offs < n_out
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for d in range(0, D1):
        acc += tl.load(x_ptr + (offs // R) * D1 * R + d * R + offs % R, mask=mask, other=0.0)
    tl.store(out_ptr + offs, acc, mask=mask)

class SumAggregatorNew(nn.Module):
    def forward(self, x):
        x = x.contiguous()
        out = torch.empty((x.shape[0], *x.shape[2:]), device=x.device, dtype=x.dtype)
        k[(triton.cdiv(out.numel(), 128),)](x, out, out.numel(), x.shape[1], x[0, 0].numel(), BLOCK=128)
        return out
'''
NOOP = "import torch, torch.nn as nn\nclass SumAggregatorNew(nn.Module):\n    def forward(self, x): return torch.sum(x, dim=1)\n"

# (name, solution, expected failed gate or None = should pass)
SUM_CASES = [
    ("good_triton", GOOD, None),
    ("crashes", GOOD.replace("x.shape[1], x[0, 0].numel(), ", ""), "runs"),          # missing kernel args
    ("noop_torch", NOOP, "uses_triton"),                                              # no Triton
    ("wrong_output", GOOD.replace("tl.store(out_ptr + offs, acc", "tl.store(out_ptr + offs, 2 * acc"), "correct"),
    ("hack_timer", NOOP + "import time\ntime.perf_counter = lambda: 0.0\n", "guard"),
]


@app.function(image=image, gpu="L4", timeout=600, max_containers=1)
def grade_cases(cases: list[dict]) -> list[dict]:
    import tempfile
    from pathlib import Path

    from kernel_env.grade import grade

    results = []
    for c in cases:
        d = Path(tempfile.mkdtemp())                     # real files: @triton.jit needs one
        (d / "ref.py").write_text(c["reference"])
        (d / "sol.py").write_text(c["solution"])
        results.append(c | grade(str(d / "ref.py"), str(d / "sol.py"), c["module_name"]))
    return results


@app.local_entrypoint()
def main() -> None:
    rows = {r["uuid"]: r for r in map(json.loads, open("data/kept.jsonl"))}
    cases = [{"name": n, "reference": rows[0]["python_code"], "solution": s,
              "module_name": "SumAggregator", "expect": e} for n, s, e in SUM_CASES]
    # KernelBook oracles: MyLinear has weights (should pass); MatchModule's oracle is wrong (step 3).
    for name, uuid, expect in [("good_weighted", 1927, None), ("bad_oracle", 4708, "correct")]:
        r = rows[uuid]
        cases.append({"name": name, "reference": r["python_code"], "solution": r["triton_code"],
                      "module_name": r["module_name"], "expect": expect})

    results = grade_cases.remote(cases)
    bad = [r for r in results if r["failed_gate"] != r["expect"]]
    for r in results:
        print(f"{'OK ' if r not in bad else 'BAD'} {r['name']:<14} expect={str(r['expect']):<12} "
              f"got={str(r['failed_gate']):<12} launches={r['kernel_launches']} {(r['message'] or '')[:60]}")
    print(f"\n{len(results) - len(bad)}/{len(results)} as expected")
    if bad:
        raise SystemExit(1)
