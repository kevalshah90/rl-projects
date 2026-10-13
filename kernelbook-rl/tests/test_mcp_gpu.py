"""Step 7 check: the agent's MCP tool, started exactly as task.toml specifies, on one Modal L4.

Run from the project root (after s05):  uv run modal run tests/test_mcp_gpu.py

What it does
  1. On the Mac: read row 0's task folder (task.toml's mcp_servers entry, the files in
     environment/, the oracle from solution/solve.sh).
  2. On one L4: recreate /workspace with those files, start the MCP server with the exact
     command + args from task.toml, then as an MCP client:
       list_tools                          -> expect exactly one tool, "check"
       call check, solution.py = oracle    -> expect reward 1.0
       call check, solution.py = ""        -> expect failed_gate "exists"
     Both calls share one session, like a real multi-turn episode, which also proves the
     second call grades the NEW file rather than a cached result.
  3. On the Mac: print OK/BAD per expectation; exit 1 if any fails.
The SERVER must find the grader in the task's own /workspace/.kernel_tools/, exactly as in
a real episode: it starts in /workspace with only PATH in its environment, so it cannot see
the kernel_env copy Modal ships for this test script itself (needed because Modal starts the
container by re-importing this file, which imports kernel_env.config).

Output (illustrative):
    OK  tools listed: ['check']
    OK  oracle -> reward=1.0 failed_gate=None
    OK  empty  -> reward=0.0 failed_gate=exists
    3/3 as expected
"""

import tomllib

import modal

from kernel_env.config import BASE_IMAGE, TASKS

app = modal.App("kernelbook-test-mcp")
# kernel_env shipped only so this script can be re-imported in the container (see docstring).
image = modal.Image.from_registry(BASE_IMAGE).add_local_python_source("kernel_env")
TASK = TASKS / "kernelbook-00000-sumaggregator"


@app.function(image=image, gpu="L4", timeout=600)
def run_tool_calls(files: dict[str, str], server: dict, oracle: str) -> list[str]:
    """Rebuild /workspace from the task's files, start the server, call its tool as a client."""
    import asyncio
    import json
    import os
    from pathlib import Path

    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    for rel, text in files.items():                    # environment/* -> /workspace/*, as Harbor uploads
        path = Path("/workspace") / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    async def calls() -> list[str]:
        out = []
        # cwd + minimal env: the server can only import the grader from /workspace/.kernel_tools.
        params = StdioServerParameters(command=server["command"], args=server["args"],
                                       cwd="/workspace", env={"PATH": os.environ["PATH"]})
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            tools = [t.name for t in (await session.list_tools()).tools]
            out.append(f"{'OK ' if tools == ['check'] else 'BAD'} tools listed: {tools}")
            for name, code, expect_reward, expect_gate in [("oracle", oracle, 1.0, None),
                                                           ("empty", "", 0.0, "exists")]:
                Path("/workspace/solution.py").write_text(code)
                text = (await session.call_tool("check", {})).content[0].text
                r = json.loads(text)
                ok = r["reward"] == expect_reward and r["failed_gate"] == expect_gate
                out.append(f"{'OK ' if ok else 'BAD'} {name:<6} -> reward={r['reward']} "
                           f"failed_gate={r['failed_gate']} {(r.get('message') or '')[:60]}")
        return out

    # 300 s cap: a server that never answers fails loudly instead of hanging the test.
    return asyncio.run(asyncio.wait_for(calls(), timeout=300))


@app.local_entrypoint()
def main() -> None:
    server = tomllib.loads((TASK / "task.toml").read_text())["environment"]["mcp_servers"][0]
    files = {str(p.relative_to(TASK / "environment")): p.read_text()
             for p in (TASK / "environment").rglob("*") if p.is_file()}
    solve = (TASK / "solution" / "solve.sh").read_text()
    oracle = solve.split("<<'KERNELBOOK_ORACLE_EOF'\n", 1)[1].rsplit("\nKERNELBOOK_ORACLE_EOF", 1)[0]

    lines = run_tool_calls.remote(files, server, oracle)
    print("\n".join(lines))
    bad = sum(line.startswith("BAD") for line in lines)
    print(f"\n{len(lines) - bad}/{len(lines)} as expected")
    if bad:
        raise SystemExit(1)
