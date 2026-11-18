"""Step 12 (part 1, check 3): does prime-rl's qwen3.5 renderer agree with what we ran?

Runs inside the s11 image (renderers, transformers and vLLM installed); launch with
    uv run modal run --detach scripts/s11_prime_rl_image.py --which renderer
Input: real conversations from the e2e run (traces.jsonl, read from the Volume).

Training builds token ids with the renderer (orchestrator side) instead of letting vLLM apply
the chat template, so the trainer scores exactly the tokens vLLM sampled. Our baseline/E0 ran
through vLLM's chat endpoint: HF chat template + the `qwen3_coder` tool parser. Checks, for
every branch of every episode (a compacted episode has 2 branches):
  1. render_ids(messages, tools) == tokenizer.apply_chat_template(messages, tools)
  2. each assistant turn: the generation prompt it was sampled from is an exact prefix of the
     full render; the tokens after it are that turn's completion
  3. that completion parses back to the same tool calls with BOTH the renderer
     (parse_response) and vLLM's qwen3_coder parser (what produced the calls in our runs)
"""

import json
import sys
from collections import Counter

MODEL = "Qwen/Qwen3.5-9B"


def load_branches(path):
    """Each episode's branches as message lists, via verifiers' own trace model."""
    from verifiers.v1 import graph
    from verifiers.v1.trace import Trace

    out = []
    for line in open(path):
        ep = json.loads(line)
        raw = ep["traces"][0]
        trace = Trace.model_validate(raw)
        leaves = graph.leaves(trace)
        branches = [[dict(m) if isinstance(m, dict) else m.model_dump(exclude_none=True)
                     for m in graph.path(trace, leaf)] for leaf in leaves]
        out.append((ep["id"][:8], raw["tools"], branches))
    return out


def hf_message(m):
    """Our trace message -> the shape HF's Qwen3.5 template reads (nested function, dict args)."""
    m = {k: v for k, v in m.items() if v is not None}
    if m.get("tool_calls"):
        m["tool_calls"] = [{"type": "function", "id": tc.get("id"),
                            "function": {"name": tc["name"], "arguments": json.loads(tc["arguments"])}}
                           for tc in m["tool_calls"]]
    return m


def hf_tool(t):
    return {"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters")}}


def calls(tool_calls):
    """Comparable form: [(name, args dict)]."""
    out = []
    for tc in tool_calls or []:
        f = tc.get("function", tc) if isinstance(tc, dict) else tc
        name = f["name"] if isinstance(f, dict) else getattr(f, "name")
        args = f["arguments"] if isinstance(f, dict) else getattr(f, "arguments")
        out.append((name, json.loads(args) if isinstance(args, str) else args))
    return out


def vllm_parser(tokenizer, tools):
    """vLLM's qwen3_coder tool parser and a request carrying our tool schemas."""
    try:
        from vllm.tool_parsers import ToolParserManager
    except ImportError:
        from vllm.entrypoints.openai.tool_parsers import ToolParserManager
    try:
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    except ImportError:
        from vllm.entrypoints.openai.protocol import ChatCompletionRequest
    parser = ToolParserManager.get_tool_parser("qwen3_coder")(tokenizer)
    request = ChatCompletionRequest(model=MODEL, messages=[{"role": "user", "content": "x"}],
                                    tools=[hf_tool(t) for t in tools])
    return parser, request


def main(path):
    from renderers import create_renderer
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    r = create_renderer(tok, renderer="qwen3.5") if "renderer" in create_renderer.__code__.co_varnames \
        else create_renderer(tok)
    print(f"renderer: {type(r).__name__} | is_prefix_stable={r.is_prefix_stable}")
    for t in ("<think>", "</think>", "<tool_call>", "</tool_call>", "<|im_end|>"):
        print(f"  {t:14s} id {tok.convert_tokens_to_ids(t)}")
    print("  stop ids:", r.get_stop_token_ids())

    stats = Counter()
    first_diffs = []
    for ep_id, tools, branches in load_branches(path):
        parser, request = vllm_parser(tok, tools)
        for b, msgs in enumerate(branches):
            tag = f"{ep_id} branch {b + 1}/{len(branches)}"
            # 1. full render vs HF template
            wire = [hf_tool(t) for t in tools]   # the training client's tool_to_wire form
            ids_r = r.render_ids(msgs, tools=wire)
            ids_hf = tok.apply_chat_template([hf_message(m) for m in msgs],
                                             tools=[hf_tool(t) for t in tools], tokenize=True)
            if hasattr(ids_hf, "input_ids"):
                ids_hf = ids_hf["input_ids"]
            same = ids_r == list(ids_hf)
            stats["branch_render_match" if same else "branch_render_DIFF"] += 1
            if not same:
                k = next((i for i, (a, c) in enumerate(zip(ids_r, ids_hf)) if a != c),
                         min(len(ids_r), len(ids_hf)))
                first_diffs.append((tag, len(ids_r), len(ids_hf), k,
                                    tok.decode(ids_r[max(0, k - 30):k + 30]),
                                    tok.decode(list(ids_hf)[max(0, k - 30):k + 30])))
            # 2 + 3. per assistant turn
            for i, m in enumerate(msgs):
                if m["role"] != "assistant":
                    continue
                prompt = r.render_ids(msgs[:i], tools=wire, add_generation_prompt=True)
                full = r.render_ids(msgs[:i + 1], tools=wire)
                if full[:len(prompt)] != prompt:
                    stats["turn_prefix_BROKEN"] += 1
                    continue
                stats["turn_prefix_ok"] += 1
                completion = full[len(prompt):]
                want = calls(m.get("tool_calls"))
                got_r = calls(r.parse_response(completion, prompt_ids=prompt, tools=wire).tool_calls)
                text = tok.decode(completion, skip_special_tokens=False)
                info = parser.extract_tool_calls(text.split("</think>")[-1], request)
                got_v = calls([{"name": tc.function.name, "arguments": tc.function.arguments}
                               for tc in info.tool_calls]) if info.tools_called else []
                stats["calls_renderer_" + ("ok" if got_r == want else "DIFF")] += 1
                stats["calls_vllm_" + ("ok" if got_v == want else "DIFF")] += 1
                if got_r != want or got_v != want:
                    print(f"  call mismatch {tag} turn {i}: want {str(want)[:150]} | "
                          f"renderer {str(got_r)[:150]} | vllm {str(got_v)[:150]}")
                stats["tool_calls"] += len(want)

    print("\nresults:")
    for k, v in sorted(stats.items()):
        print(f"  {k:24s} {v}")
    for tag, n_r, n_hf, k, a, c in first_diffs[:4]:
        print(f"\n  render diff {tag}: renderer {n_r} vs HF {n_hf} tokens; first diff at {k}")
        print(f"    renderer: {a!r}\n    hf:       {c!r}")


if __name__ == "__main__":
    main(sys.argv[1])
