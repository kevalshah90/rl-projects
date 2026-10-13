"""Count @triton.jit kernel launches inside a `with` block (gate `uses_triton` in grade.py).

Every launch, `kernel[grid](args)`, goes through JITFunction.run, so we wrap that method.
"""

import contextlib

from triton.runtime.jit import JITFunction


@contextlib.contextmanager
def count_launches():
    counter = {"n": 0}                    # dict so the caller sees the final count
    original_run = JITFunction.run

    def counting_run(self, *args, **kwargs):
        counter["n"] += 1
        return original_run(self, *args, **kwargs)

    JITFunction.run = counting_run
    try:
        yield counter
    finally:
        JITFunction.run = original_run    # always restore, even if forward() crashed
