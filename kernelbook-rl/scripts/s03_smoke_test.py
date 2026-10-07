"""Step 3: smoke-test the shared base image on a real GPU.

Runs on: Modal, one L4 GPU, using the exact image tasks will use (pulled from Docker Hub).
Run from the project root:  uv run modal run scripts/s03_smoke_test.py
Writes: data/smoke_s03.jsonl, one result per checked row.

Checks, in order:
  1. versions   CUDA visible, torch 2.5.1, triton 3.1.0
  2. row 0      KernelBook's oracle (SumAggregatorNew) matches the reference on the GPU
  3. sample     the same check on 20 random kept rows (fixed seed: reproducible)

The oracle check copies the reference's weights into the oracle module (load_state_dict),
because step 2 showed that rebuilding a module with the same seed does NOT reproduce
weights made from uninitialized memory. The grader (step 4) will do the same.
"""

import json
import random
from pathlib import Path

import modal

from kernel_env.config import BASE_IMAGE

SAMPLE_SIZE = 20
SAMPLE_SEED = 0
RTOL, ATOL = 1e-3, 1e-5          # the tolerances the grader will use
OUT = Path("data/smoke_s03.jsonl")

app = modal.App("kernelbook-s03-smoke")
image = modal.Image.from_registry(BASE_IMAGE)   # the same image every task container uses


@app.function(image=image, gpu="L4", timeout=300)
def check_versions() -> dict:
    import torch
    import triton
    return {"cuda": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "triton": triton.__version__}


@app.function(image=image, gpu="L4", timeout=300)
def check_oracle(row: dict) -> dict:
    """Run one row's reference module and its Inductor oracle on the GPU; compare outputs."""
    import importlib.util
    import tempfile
    import traceback

    import torch

    result = {"uuid": row["uuid"], "module_name": row["module_name"], "ok": False,
              "max_abs_diff": None, "missing_keys": [], "unexpected_keys": [], "error": None}

    def load(code: str, name: str):
        # From a real file, not exec(): @triton.jit reads the kernel's source from its file.
        path = Path(tempfile.mkdtemp()) / f"{name}.py"
        path.write_text(code)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    try:
        ref = load(row["python_code"], "reference")
        oracle = load(row["triton_code"], "oracle")
        args, kwargs = ref.get_init_inputs()                 # KernelBook format: [[args], {kwargs}]
        torch.manual_seed(0)
        ref_model = getattr(ref, row["module_name"])(*args, **kwargs).cuda().eval()
        oracle_model = getattr(oracle, row["module_name"] + "New")(*args, **kwargs).cuda().eval()
        # Copy weights reference -> oracle. strict=False: report name mismatches instead of
        # crashing, since whether the names line up is exactly what we want to learn here.
        keys = oracle_model.load_state_dict(ref_model.state_dict(), strict=False)
        result["missing_keys"], result["unexpected_keys"] = list(keys.missing_keys), list(keys.unexpected_keys)

        torch.manual_seed(1)
        inputs = [x.cuda() if torch.is_tensor(x) else x for x in ref.get_inputs()]
        with torch.no_grad():                                # clone: forward may edit inputs in place
            expected = ref_model(*[x.clone() if torch.is_tensor(x) else x for x in inputs])
            got = oracle_model(*[x.clone() if torch.is_tensor(x) else x for x in inputs])
        if got.shape != expected.shape:
            result["error"] = f"shape {tuple(got.shape)} != {tuple(expected.shape)}"
            return result
        result["max_abs_diff"] = (got.float() - expected.float()).abs().max().item()
        result["ok"] = torch.allclose(got, expected, rtol=RTOL, atol=ATOL)
    except Exception:
        result["error"] = traceback.format_exc(limit=2)[-300:]   # the end has the actual error
    return result


@app.local_entrypoint()
def main() -> None:
    print("1. versions:", check_versions.remote())

    rows = [json.loads(line) for line in open("data/kept.jsonl")]
    row0 = next(r for r in rows if r["uuid"] == 0)
    sample = random.Random(SAMPLE_SEED).sample([r for r in rows if r["uuid"] != 0], SAMPLE_SIZE)

    # 2 + 3: row 0 first, then the sample, all in parallel on L4s
    results = list(check_oracle.map([row0, *sample], return_exceptions=True))
    OUT.write_text("".join(json.dumps(r if isinstance(r, dict) else {"error": repr(r)}) + "\n"
                           for r in results))

    print(f"2-3. oracle vs reference ({len(results)} rows; row 0 first):")
    for r in results:
        if not isinstance(r, dict):
            print(f"   CRASH  {r!r}")
            continue
        flag = "PASS " if r["ok"] else "FAIL "
        keys = f" missing={len(r['missing_keys'])} unexpected={len(r['unexpected_keys'])}" \
            if r["missing_keys"] or r["unexpected_keys"] else ""
        err = f"  {r['error'].strip().splitlines()[-1]}" if r["error"] else ""
        print(f"   {flag} uuid {r['uuid']:>5} {r['module_name']:<28} diff={r['max_abs_diff']}{keys}{err}")
    passed = sum(isinstance(r, dict) and r["ok"] for r in results)
    print(f"\n{passed}/{len(results)} oracles match their reference -> {OUT}")
