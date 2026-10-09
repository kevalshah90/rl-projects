Rewrite the PyTorch module in /workspace/reference.py as a Triton kernel.

Write /workspace/solution.py defining `class ${class_name}New(nn.Module)`:
- Same __init__ arguments as `${class_name}`, and the SAME parameter and submodule names
  (the grader copies the reference's weights into yours by name).
- forward() must launch at least one @triton.jit kernel that does the computation.
  torch.compile is not allowed.
- Output must match the reference (rtol 1e-3, atol 1e-5) on the inputs from get_inputs().

Workflow:
1. Read /workspace/reference.py.
2. Write /workspace/solution.py.
3. Test it by calling the `kernel-tools_check` tool. It is a tool call, like `bash`, not a
   file or a command. It runs the final grader on your current solution.py and reports the
   first gate that fails and why (gates: exists, guard, parse, weights, runs, uses_triton,
   correct).
4. Fix what it reports, then call `kernel-tools_check` again. Repeat until it returns
   reward 1.0. Prefer small edits over rewriting the whole file.

You can also run Python in /workspace. Only /workspace/solution.py is graded.
