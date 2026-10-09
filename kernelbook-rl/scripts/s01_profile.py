"""Step 1: profile KernelBook and write data/profile_report.md.

Runs on: the Mac, from the project root:  uv run python scripts/s01_profile.py

Never executes dataset code: every row's python_code is untrusted code from GitHub, so we
only PARSE it with Python's `ast` module (turn text into a syntax tree) and read the tree.
Executing rows safely (subprocess + timeout) is step 2's job.

The report has eight sections, one function each. Every function takes the rows (and the
parsed trees) and returns a markdown string, so sections are easy to read and to change.
"""

import ast
import hashlib
import math
from collections import Counter

from kernel_env.config import DATA
from kernel_env.data import Row, dotted, input_shapes, load_rows, parse, top_level

OUT = DATA / "profile_report.md"


# ---------- small helpers (the code-parsing ones live in kernel_env/data.py) ----------

def module_ops(tree: ast.Module, class_name: str) -> set[str]:
    """torch / F / nn names called inside the module class (both __init__ and forward).
    Limitation: method calls on tensors (x.sum(), x.view()) are not counted."""
    cls = top_level(tree, ast.ClassDef, class_name)
    if cls is None:
        return set()
    names = {dotted(n.func) for n in ast.walk(cls) if isinstance(n, ast.Call)}
    return {n for n in names if n and n.split(".")[0] in {"torch", "F", "nn"}}


def pct(values: list[float], q: float) -> float:
    """q-th percentile (0-100) by nearest rank. Good enough for a profile."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q / 100 * len(ordered)))] if ordered else float("nan")


def table(headers: list[str], rows: list[list]) -> str:
    """A markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def share(n: int, total: int) -> str:
    return f"{n:,} ({100 * n / total:.1f}%)"


# ---------- the eight sections ----------

def size_section(rows: list[Row]) -> str:
    by_hash = Counter(hashlib.sha256(r.python_code.encode()).hexdigest() for r in rows)
    dup_rows = sum(c for c in by_hash.values() if c > 1)
    return "## 1. Size and duplicates\n\n" + table(["Measure", "Value"], [
        ["Rows", f"{len(rows):,}"],
        ["Unique uuids", f"{len({r.uuid for r in rows}):,}"],
        ["Rows whose python_code is identical to another row's", share(dup_rows, len(rows))],
    ])


def provenance_section(rows: list[Row]) -> str:
    n = len(rows)
    licenses = Counter(lic for r in rows for lic in r.licenses)   # a row can list several
    out = "## 2. Provenance\n\n" + table(["Measure", "Value"], [
        ["synthetic = True", share(sum(r.synthetic for r in rows), n)],
        ["No license listed", share(sum(not r.licenses for r in rows), n)],
    ])
    return out + "\n\nTop 15 licenses (rows listing each):\n\n" + table(
        ["License", "Rows"], [[lic, share(c, n)] for lic, c in licenses.most_common(15)])


def interface_section(rows: list[Row], trees: dict) -> str:
    n = len(rows)
    checks = {
        "python_code parses": lambda r, t: t is not None,
        "has get_inputs()": lambda r, t: t is not None and top_level(t, ast.FunctionDef, "get_inputs") is not None,
        "has get_init_inputs()": lambda r, t: t is not None and top_level(t, ast.FunctionDef, "get_init_inputs") is not None,
        "class named module_name exists": lambda r, t: t is not None and top_level(t, ast.ClassDef, r.module_name) is not None,
        "triton_code defines <module_name>New": lambda r, t: f"class {r.module_name}New" in r.triton_code,
        "entry_point == module_name": lambda r, t: r.entry_point == r.module_name,
    }
    counts = {name: sum(check(r, trees[r.uuid]) for r in rows) for name, check in checks.items()}
    all_ok = sum(all(check(r, trees[r.uuid]) for check in checks.values()) for r in rows)
    body = [[name, share(c, n)] for name, c in counts.items()] + [["**All of the above**", share(all_ok, n)]]
    return "## 3. Task interface\n\n" + table(["Check", "Rows passing"], body)


def inputs_section(rows: list[Row], trees: dict) -> str:
    shapes = [input_shapes(t) if t else None for t in (trees[r.uuid] for r in rows)]
    readable = [s for s in shapes if s is not None]
    elements = [sum(math.prod(s) for s in shp) for shp in readable]       # total floats per row
    tensors = Counter(min(len(s), 4) for s in readable)                   # 4 means "4 or more"
    out = f"## 4. Input sizes (from get_inputs, read statically)\n\nReadable: {share(len(readable), len(rows))}\n\n"
    out += table(["Total elements per row", "Value"], [
        [f"p{q}", f"{pct(elements, q):,.0f}"] for q in (10, 50, 90, 99)] + [
        ["max", f"{max(elements, default=0):,}"],
        ["<= 1,024 (tiny)", share(sum(e <= 1024 for e in elements), len(elements))],
        ["> 1,000,000", share(sum(e > 1_000_000 for e in elements), len(elements))],
    ])
    return out + "\n\n" + table(["Input tensors", "Rows"], [
        [("4+" if k == 4 else k), share(c, len(readable))] for k, c in sorted(tensors.items())])


def ops_section(rows: list[Row], trees: dict) -> tuple[str, Counter]:
    per_row = [module_ops(trees[r.uuid], r.module_name) if trees[r.uuid] else set() for r in rows]
    counts = Counter(op for ops in per_row for op in ops)
    sizes = Counter(min(len(ops), 4) for ops in per_row)
    out = "## 5. Operations (torch / F / nn calls in the module class)\n\n" + table(
        ["Distinct ops per row", "Rows"],
        [[("4+" if k == 4 else k), share(c, len(rows))] for k, c in sorted(sizes.items())])
    out += "\n\nTop 30 ops:\n\n" + table(["Op", "Rows using it"],
                                          [[op, share(c, len(rows))] for op, c in counts.most_common(30)])
    return out, counts


def code_size_section(rows: list[Row]) -> str:
    py_lines = [r.python_code.count("\n") + 1 for r in rows]
    tr_lines = [r.triton_code.count("\n") + 1 for r in rows]
    kernels = [r.triton_code.count("@triton.jit") for r in rows]
    return "## 6. Code size\n\n" + table(["Measure", "p50", "p90", "p99", "max"], [
        [name, *(f"{pct(v, q):,.0f}" for q in (50, 90, 99)), f"{max(v):,}"]
        for name, v in [("python_code lines", py_lines), ("triton_code lines", tr_lines),
                        ("@triton.jit kernels in oracle", kernels)]])


def oracle_risk_section(rows: list[Row]) -> str:
    # Import lines in the oracle that reach into torch internals; these break across torch versions.
    imports = Counter(line.strip() for r in rows for line in r.triton_code.splitlines()
                      if line.lstrip().startswith(("from torch._", "import torch._")))
    uses_inductor = sum("torch._inductor" in r.triton_code for r in rows)
    out = "## 7. Oracle risk (torch internals imported by triton_code)\n\n"
    out += f"Oracles mentioning torch._inductor: {share(uses_inductor, len(rows))}\n\n"
    return out + table(["Import line", "Rows"], [[f"`{line}`", share(c, len(rows))]
                                                for line, c in imports.most_common(10)])


def examples_section(rows: list[Row], trees: dict, op_counts: Counter) -> str:
    out = "## 8. Examples (first single-op row for each of the 3 most common ops)\n"
    for op, _ in op_counts.most_common(3):
        row = next((r for r in rows if trees[r.uuid]
                    and module_ops(trees[r.uuid], r.module_name) == {op}), None)
        if row is None:
            continue
        excerpt = "\n".join(row.python_code.strip().splitlines()[:20])   # first 20 lines only
        out += f"\n### `{op}` — uuid {row.uuid}, {row.module_name}\n\n```python\n{excerpt}\n```\n"
    return out


# ---------- main ----------

def main() -> None:
    rows = load_rows()
    # Sections look up each row's tree by uuid, so uuids must be unique. Fail loudly if not.
    assert len({r.uuid for r in rows}) == len(rows), "duplicate uuids: trees keyed by uuid would collide"
    trees = {r.uuid: parse(r.python_code) for r in rows}     # parse once, reuse in every section
    ops_md, op_counts = ops_section(rows, trees)
    sections = [
        "# KernelBook profile\n\nGenerated by scripts/s01_profile.py. Static analysis only.",
        size_section(rows),
        provenance_section(rows),
        interface_section(rows, trees),
        inputs_section(rows, trees),
        ops_md,
        code_size_section(rows),
        oracle_risk_section(rows),
        examples_section(rows, trees, op_counts),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n\n".join(sections) + "\n")
    print(f"{len(rows):,} rows profiled -> {OUT}")


if __name__ == "__main__":
    main()
