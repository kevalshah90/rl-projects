Rewrite the PyTorch module in /workspace/reference.py as a Triton kernel.

Write /workspace/solution.py defining `class ${class_name}New(nn.Module)`:
- Same __init__ arguments as `${class_name}`, and the SAME parameter and submodule names
  (the grader copies the reference's weights into yours by name).
- forward() must launch at least one @triton.jit kernel that does the computation.
  torch.compile is not allowed.
- Output must match the reference (rtol 1e-3, atol 1e-5) on the inputs from get_inputs().

You can run Python in /workspace to test your kernel. Only /workspace/solution.py is graded.
