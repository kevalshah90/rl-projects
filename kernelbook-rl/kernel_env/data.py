"""
Load GPUMODE/KernelBook as typed rows, and decide which rows become tasks.

Sections:  1. loading (step 1)   2. static code analysis   3. filters (step 2)
Runs on: the Mac (scripts/ and tests/), no GPU. Everything here only PARSES dataset code,
except exec_check() at the bottom, which runs it and is only ever called in a worker subprocess.

Each KernelBook record is one PyTorch module paired with Triton code that TorchInductor
generated for it. We keep only the fields the pipeline uses and give them types, so the
rest of the code works with `row.module_name` instead of raw dictionary lookups.

Example
-------
Input  — one raw record dict from the HuggingFace dataset:
    {
        "uuid": 42,
        "module_name": "SumAggregator",
        "entry_point": "SumAggregator",
        "python_code": "class SumAggregator(nn.Module): ...",
        "triton_code": "class SumAggregatorNew: ...",
        "licenses": ["Apache-2.0"],
        "synthetic": False,
        "repo_link": "https://github.com/org/repo/tree/<sha>",
    }

Output — a typed, immutable Row:
    Row(
        uuid=42,
        module_name="SumAggregator",
        entry_point="SumAggregator",
        python_code="class SumAggregator(nn.Module): ...",
        triton_code="class SumAggregatorNew: ...",
        licenses=("Apache-2.0",),
        synthetic=False,
        repo_link="https://github.com/org/repo/tree/<sha>",
    )

Section 2 — Static code analysis (AST only, never executes code)
    parse()             Parse python_code into an AST; returns None on SyntaxError.
    top_level()         Find a named class or function at the top level of the tree.
    dotted()            Resolve an attribute chain (torch.nn.functional.relu) to a
                        dotted string, normalised to the short form (F.relu).
    input_shapes()      Read tensor shapes from get_inputs() without running it.
                        Used by size_reason() to check element count statically.
    _strip_docstrings() Remove docstring nodes so modules that differ only in
                        documentation hash identically.
    source_key()        SHA-256 of the canonicalised, docstring-free source — used
                        by F1 (duplicate detection) in s02_filter.py.
    structure_key()     SHA-256 of the module class alone, renamed to `Model` — used
                        by F6 (KernelBench overlap) so `class Foo` and `class Model`
                        match when their bodies are identical.

    Example (using the SumAggregator row above):
        tree = parse(row.python_code)
        # → ast.Module

        top_level(tree, ast.FunctionDef, "get_inputs")
        # → ast.FunctionDef node for get_inputs()

        top_level(tree, ast.ClassDef, "SumAggregator")
        # → ast.ClassDef node for the module class

        # inside get_inputs(), torch.rand([4, 4, 4, 4]) is a call node;
        # dotted(call.func) resolves the attribute chain:
        dotted(call.func)                      # → "torch.rand"

        input_shapes(tree)                     # → [(4, 4, 4, 4)]
        source_key(row.python_code)            # → "a3f9..."  (same for any whitespace/comment variant)
        structure_key(row.python_code, "SumAggregator")  # → "c71b..."

Section 3 — Filters (each returns None to keep, or a reason string to drop)
    license_reason()  Drop rows with no license or a copyleft/non-commercial license.
    size_reason()     Drop rows whose prompt is too long (> 200 lines), whose input
                      shapes can't be read statically, or whose total input element
                      count exceeds 1 M (4 MB of float32).
    static_reason()   Combines parse + license + size; the single call site for
                      F2–F4 in s02_filter.py.
    exec_check()      Actually instantiates and runs the nn.Module on CPU. Drops the
                      row if construction produces non-finite weights, the forward
                      pass crashes, the output is not a single finite tensor, or two
                      forward calls on the same inputs differ. Runs on the Mac, but only
                      inside a disposable worker subprocess (exec_worker.py), never in
                      the main process.

    Example (using the SumAggregator row above):
        license_reason(row)                    # → None           (Apache-2.0 is fine)
        static_reason(row)                     # → None           (passes all static checks)

        # a row that would be dropped:
        license_reason(gpl_row)                # → "license: copyleft or non-commercial ['GPL-3.0']"
        static_reason(long_row)                # → "length: 210 lines > 200"
        static_reason(huge_row)                # → "inputs: 2,097,152 elements > 1,048,576"

        exec_check(row.python_code, "SumAggregator")   # → None  (runs, finite, deterministic)
        exec_check(bad_row.python_code, "BadInit")     # → "init: non-finite weights (uninitialized memory?)"

"""

import ast
import hashlib
import math
from dataclasses import dataclass

# =====================================================================================
# 1. Loading
# =====================================================================================

DATASET = "GPUMODE/KernelBook"   # Hugging Face dataset id
SPLIT = "train"                  # KernelBook ships a single split with all 18,162 rows


@dataclass(frozen=True)          # frozen: rows are read-only facts about the dataset
class Row:
    uuid: int                    # stable id from the dataset; we use it in task folder names
    module_name: str             # PyTorch class name, e.g. "SumAggregator"
    entry_point: str             # the dataset's own name for the module (usually == module_name)
    python_code: str             # the nn.Module + get_inputs() + get_init_inputs(): the task
    triton_code: str             # Inductor-generated Triton with class <module_name>New: the oracle
    licenses: tuple[str, ...]    # licenses of the source GitHub repo, e.g. ("Apache-2.0",)
    synthetic: bool              # True if the module was generated, not taken from GitHub
    repo_link: str               # source repo at a fixed commit, for provenance


def to_row(record: dict) -> Row:
    """Convert one raw dataset record into a Row. Fields we don't use are left behind."""
    return Row(
        uuid=int(record["uuid"]),
        module_name=record["module_name"],
        entry_point=record["entry_point"],
        python_code=record["python_code"],
        triton_code=record["triton_code"],
        # The dataset stores a list (or None); a tuple keeps the frozen Row hashable.
        licenses=tuple(record["licenses"] or ()),
        synthetic=bool(record["synthetic"]),
        repo_link=record["repo_link"] or "",
    )


def load_rows() -> list[Row]:
    """Download KernelBook (cached under ~/.cache/huggingface after the first call) as Rows."""
    # Imported here, not at the top, so modules that only need `Row` don't pay
    # the cost of importing the `datasets` library.
    from datasets import load_dataset

    records = load_dataset(DATASET, split=SPLIT)
    return [to_row(record) for record in records]


# =====================================================================================
# 2. Static code analysis: parse code into a syntax tree and read it. Never executes it.
# =====================================================================================

# torch functions that create input tensors in get_inputs(), e.g. torch.rand([4, 4, 4, 4])
TENSOR_MAKERS = {"rand", "randn", "ones", "zeros", "empty", "full", "randint"}


def parse(code: str) -> ast.Module | None:
    """Syntax tree of the code, or None if it isn't valid Python."""
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def top_level(tree: ast.Module, kind: type, name: str):
    """Find a top-level function or class by name (kind = ast.FunctionDef or ast.ClassDef)."""
    return next((n for n in tree.body if isinstance(n, kind) and n.name == name), None)


def dotted(node: ast.AST) -> str | None:
    """Turn a call target into a dotted name: torch.nn.functional.relu -> 'F.relu'."""
    parts = []
    while isinstance(node, ast.Attribute):          # walk a.b.c from the right
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None                                  # e.g. x.view(...)[0].sum(): not a plain name
    name = ".".join([node.id, *reversed(parts)])
    # Normalize the long spellings so the same op is counted once.
    return name.replace("torch.nn.functional.", "F.").replace("torch.nn.", "nn.")


def input_shapes(tree: ast.Module) -> list[tuple[int, ...]] | None:
    """Shapes of the tensors created in get_inputs(), or None if we can't read them statically."""
    fn = top_level(tree, ast.FunctionDef, "get_inputs")
    if fn is None:
        return None
    shapes = []
    for node in ast.walk(fn):
        name = dotted(node.func) if isinstance(node, ast.Call) else None
        if name is None or not name.startswith("torch.") or name[6:] not in TENSOR_MAKERS:
            continue
        # Shapes are written three ways: torch.rand([4, 4]), torch.rand(4, 4), torch.randint(0, 9, (4, 4)).
        try:
            if name == "torch.randint":
                shape = ast.literal_eval(node.args[-1])                 # size is the last argument
            elif node.args and all(isinstance(a, ast.Constant) for a in node.args) and name != "torch.full":
                shape = tuple(ast.literal_eval(a) for a in node.args)   # torch.rand(4, 4)
            else:
                shape = ast.literal_eval(node.args[0])                  # torch.rand([4, 4]), torch.full((4,), 1.0)
        except (ValueError, TypeError, IndexError):
            return None                              # a variable or no args: unreadable statically
        if not (isinstance(shape, (list, tuple)) and all(isinstance(d, int) for d in shape)):
            return None
        shapes.append(tuple(shape))
    return shapes


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove docstrings in place. (Comments never reach the tree, so they're already gone.)"""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]   # a body can't be empty
    return tree


def source_key(code: str) -> str:
    """Fingerprint for duplicate detection: two rows that differ only in comments,
    docstrings or formatting get the same key, because ast.unparse re-prints the tree
    in one canonical style."""
    tree = parse(code)
    text = ast.unparse(_strip_docstrings(tree)) if tree else code
    return hashlib.sha256(text.encode()).hexdigest()


def structure_key(code: str, class_name: str) -> str | None:
    """Fingerprint of the module class alone, renamed to `Model`.
    KernelBench always names its class `Model`; renaming ours lets `class Foo` and
    `class Model` match when everything else in the class is identical."""
    tree = parse(code)
    cls = top_level(tree, ast.ClassDef, class_name) if tree else None
    if cls is None:
        return None
    _strip_docstrings(cls)
    for node in ast.walk(cls):                     # rename every mention, e.g. super(Foo, self)
        if isinstance(node, ast.Name) and node.id == class_name:
            node.id = "Model"
    cls.name = "Model"
    return hashlib.sha256(ast.unparse(cls).encode()).hexdigest()


# =====================================================================================
# 3. Filters. Each returns None to KEEP the row, or a reason string to DROP it.
#    Reasons start with a short category ("license:", "inputs:") so they can be counted.
# =====================================================================================

# A row is dropped if ANY of its repo's licenses starts with one of these (copyleft or
# non-commercial). Everything KernelBook actually contains (MIT, Apache, BSD, ...) passes.
COPYLEFT_PREFIXES = ("GPL", "AGPL", "LGPL", "MPL", "EPL", "EUPL", "OSL", "CC-BY-SA", "CC-BY-NC")
MAX_ELEMENTS = 1_048_576   # total input elements: 4 MB of float32, comfortable on an L4
MAX_LINES = 200            # python_code length: keeps prompts short (~1% of rows are longer)


def license_reason(row: Row) -> str | None:
    if not row.licenses:
        return "license: none listed"
    bad = [lic for lic in row.licenses if lic.upper().startswith(COPYLEFT_PREFIXES)]
    return f"license: copyleft or non-commercial {bad}" if bad else None


def size_reason(row: Row, tree: ast.Module) -> str | None:
    lines = row.python_code.count("\n") + 1
    if lines > MAX_LINES:
        return f"length: {lines} lines > {MAX_LINES}"
    shapes = input_shapes(tree)
    if shapes is None:
        return "inputs: shapes not readable statically"
    elements = sum(math.prod(s) for s in shapes)
    if elements == 0:
        return "inputs: get_inputs creates no tensors"
    if elements > MAX_ELEMENTS:
        return f"inputs: {elements:,} elements > {MAX_ELEMENTS:,}"
    return None


def static_reason(row: Row) -> str | None:
    """All filters that only read the code (cheap, safe on the Mac). First failure wins."""
    tree = parse(row.python_code)
    if tree is None:
        return "parse: python_code is not valid Python"
    return license_reason(row) or size_reason(row, tree)


def exec_check(python_code: str, module_name: str) -> str | None:
    """
    RUNS the row's code on CPU: build the module, call forward, check the output.

    Executes untrusted code from GitHub, so call it ONLY inside a disposable child process
    (kernel_env/exec_worker.py, started by scripts/s02_filter.py), never in a long-lived one.
    Keeps a row only if: it builds with finite weights, runs, returns a single finite
    tensor, and calling forward twice on the same inputs gives exactly the same output
    (the grader compares outputs, so the reference's forward must be deterministic).

    The module is built ONCE. Weights made with torch.empty / torch.Tensor(n) are
    uninitialized memory that no seed controls, so the grader copies the reference's
    weights into the solution instead of rebuilding them; this check mirrors that.
    """

    import torch   # imported here: only the worker process needs torch

    try:
        
        namespace = {"__name__": "kernelbook_row"}
        exec(python_code, namespace)                      # defines the class + get_inputs()
        args, kwargs = namespace["get_init_inputs"]()     # KernelBook format: [[args], {kwargs}]

        torch.manual_seed(0)
        model = namespace[module_name](*args, **kwargs).eval()   # eval(): dropout off
        weights = [t for t in [*model.parameters(), *model.buffers()] if t.is_floating_point()]
        if any(not torch.isfinite(t).all() for t in weights):
            return "init: non-finite weights (uninitialized memory?)"
        
        torch.manual_seed(1)
        inputs = namespace["get_inputs"]()

        with torch.no_grad():                             # clone: forward may modify inputs in place
            first = model(*[x.clone() if torch.is_tensor(x) else x for x in inputs])
            second = model(*[x.clone() if torch.is_tensor(x) else x for x in inputs])

    except Exception as e:                                # any crash, incl. missing imports, no network
        return f"exec: {type(e).__name__}: {str(e)[:100]}"
    if not isinstance(first, torch.Tensor):
        return f"output: {type(first).__name__}, not a single tensor"
    if first.is_floating_point() and not torch.isfinite(first).all():
        return "output: contains NaN or inf"
    if not torch.equal(first, second):
        return "nondeterministic: two forward calls differ"
    return None
