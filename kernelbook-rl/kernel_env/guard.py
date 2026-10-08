"""Static hack scan: reject a submission's source text before any of it runs.

Runs on: anywhere (pure text, no torch). Used by grade.py (gate 2) and, in step 7,
by the agent's `check` tool.

Why: in this toy project the solution runs inside the grader's own process, so code
that rewrites torch.allclose or Triton's launch counter could fake a pass. Scanning
the text first blocks the obvious ways to do that. It is a heuristic, not a proof:
deliberately obfuscated code can still slip through (that's the Problem 4 exercise).
"""

import re

# pattern -> reason reported back to the agent. Matched against the raw source text,
# line by line (re.MULTILINE, so ^ means "start of a line"). Keep the list short:
# every entry should be explainable in one line.
BANNED: dict[str, str] = {
    # 1. Rewriting torch or triton. The grader compares outputs with torch and counts
    #    launches through triton, so assigning to either can fake a result.
    #    Anchored at line start so type hints like `x: torch.Tensor = None` don't match.
    r"^\s*(torch|triton)(\.\w+)+\s*=(?!=)": "assigns to a torch/triton attribute",
    r"\bsetattr\s*\(\s*(torch|triton)\b": "patches torch/triton with setattr",

    # 2. Patching timers. The phase 2 speed reward measures time.
    r"^\s*time\.\w+\s*=(?!=)": "patches a timer",

    # 3. Letting torch write the kernels. torch.compile generates real Triton kernels,
    #    which would pass the launch counter without the agent writing any.
    r"\btorch\.compile\b": "uses torch.compile instead of writing kernels",

    # 4. Running hidden code. (?<![\w.]) means "not preceded by a name or a dot",
    #    so model.eval() is fine but a bare eval(...) is not.
    r"(?<![\w.])(exec|eval|compile|__import__)\s*\(": "runs hidden code",
    r"\b(builtins|__builtins__)\b": "touches Python builtins",
    r"\bsys\.modules\b": "swaps imported modules",

    # 5. Leaving the sandbox: processes, network, raw memory, grader files.
    r"\b(subprocess|socket|requests|urllib|ctypes)\b": "uses processes, network or ctypes",
    r"/tests\b|/logs\b|reward\.json": "touches grader files",
}


def scan(source: str) -> str | None:
    """Return the reason for the first banned pattern found, or None if the source is clean."""
    for pattern, reason in BANNED.items():
        if re.search(pattern, source, flags=re.MULTILINE):
            return reason
    return None
