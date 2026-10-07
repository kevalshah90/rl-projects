"""kernel_env: the core library.

Used in two places:
  - on the Mac, by scripts/ and tests/ (loading data, filtering, building tasks)
  - inside the base Docker image, by the grader (tests/test.sh) and the agent's MCP tools

Modules are added one build step at a time; see README.md.
"""
