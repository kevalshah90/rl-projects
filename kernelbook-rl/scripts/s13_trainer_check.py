"""Step 13 (part 1, check 4): Qwen3.5-9B in prime-rl's trainer, and how far its logprobs sit
from vLLM's (the "engine mismatch" source of off-policy data in docs/NOTEBOOK.md).

Runs inside the s11 image on one H100; launch with
    uv run modal run --detach scripts/s11_prime_rl_image.py --which trainer

Two phases, each its own process so the GPU is free in between:

  vllm     Prompts = every assistant turn of the e2e conversations, rendered by the qwen3.5
           renderer (exactly what training sends). vLLM (prime-rl's pinned 0.33) samples a
           completion at temperature 1 and returns the logprob of each sampled token. Those come
           from the decode path, like the inference_logprobs the trainer receives in RL.

  trainer  (under torchrun, 1 process) prime-rl's own model stack: setup_model() with LoRA,
           FSDP, activation checkpointing, compile, as the trainer builds it. Fresh LoRA has B = 0,
           so the policy equals the base model. Scores prompt + completion in one forward and
           compares per sampled token with vLLM:
             mismatch_kl  prime-rl's metric, exp(r) - 1 - r with r = log p_train - log p_vllm
             masked       share of tokens the IPO loss masks, |p_train - p_vllm| > 0.3
           Then one backward pass on the longest sequence: LoRA grads flow, peak memory.
"""

import json
import sys
import time
from pathlib import Path

MODEL = "Qwen/Qwen3.5-9B"
MAX_LEN = 32768            # the served context; prompt + completion must fit
MAX_NEW = 2048             # tokens sampled per prompt (enough for statistics, keeps it quick)


# ---------------------------------------------------------------- phase 1: vLLM
def phase_vllm(traces: str, out: str) -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    from s12_renderer_check import hf_tool, load_branches
    from renderers import create_renderer
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(MODEL)
    r = create_renderer(tok)   # resolves to Qwen35Renderer from the model name (check 3)
    print(f"renderer: {type(r).__name__}", flush=True)
    prompts, seen = [], set()
    for ep_id, tools, branches in load_branches(traces):
        wire = [hf_tool(t) for t in tools]
        for msgs in branches:
            for i, m in enumerate(msgs):
                if m["role"] != "assistant":
                    continue
                ids = r.render_ids(msgs[:i], tools=wire, add_generation_prompt=True)
                key = (len(ids), tuple(ids[-64:]))
                if key in seen or len(ids) + MAX_NEW > MAX_LEN:   # turns shared by 2 branches
                    continue
                seen.add(key)
                prompts.append({"ep": ep_id, "turn": i, "prompt_ids": ids})
    print(f"{len(prompts)} prompts | lengths {min(len(p['prompt_ids']) for p in prompts)}"
          f"..{max(len(p['prompt_ids']) for p in prompts)} tokens", flush=True)

    llm = LLM(MODEL, max_model_len=MAX_LEN, gpu_memory_utilization=0.85,
              language_model_only=True, seed=0)
    sp = SamplingParams(temperature=1.0, max_tokens=MAX_NEW, logprobs=0, seed=0,
                        stop_token_ids=r.get_stop_token_ids())
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p["prompt_ids"]) for p in prompts], sp)
    print(f"generated in {time.time() - t0:.0f}s", flush=True)
    for p, o in zip(prompts, outs):
        c = o.outputs[0]
        p["completion_ids"] = list(c.token_ids)
        p["vllm_logprobs"] = [lp[t].logprob for t, lp in zip(c.token_ids, c.logprobs)]
    Path(out).write_text(json.dumps(prompts))
    n = sum(len(p["completion_ids"]) for p in prompts)
    print(f"wrote {out}: {n} sampled tokens", flush=True)
    # vLLM's engine-core subprocess can keep the interpreter alive after the work is done
    # (check 4: hung 8 min on an idle H100), and a bare os._exit orphans it with ~70 GB of GPU
    # memory still held (the trainer phase then OOMed). Kill the children, then exit hard.
    import os
    import psutil
    for child in psutil.Process().children(recursive=True):
        child.kill()
    psutil.wait_procs(psutil.Process().children(recursive=True), timeout=30)
    os._exit(0)


# ---------------------------------------------------------------- phase 2: trainer
def phase_trainer(data: str, gdn_lora: bool) -> None:
    import torch
    from prime_rl.configs.trainer import LoRAConfig, ModelConfig
    from prime_rl.trainer.lora import get_lora_state
    from prime_rl.trainer.model import forward, setup_model
    from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
    from prime_rl.trainer.rl.loss import _mismatch_kl_from_log_ratio
    from prime_rl.trainer.utils import setup_torch_distributed

    lora = LoRAConfig()
    if gdn_lora:   # also adapt the GatedDeltaNet (linear attention) projections
        lora = LoRAConfig(target_modules=lora.target_modules + ["in_proj_qkv", "in_proj_z", "out_proj"])
    cfg = ModelConfig(name=MODEL, seq_len=MAX_LEN, lora=lora)
    setup_torch_distributed()
    resolve_ep(cfg)
    t0 = time.time()
    model = setup_model(cfg, get_parallel_dims(cfg))
    get_lora_state().reset_adapter_parameters()   # what train() does: fresh adapter, B = 0
    print(f"model ready in {time.time() - t0:.0f}s | {type(model).__name__}", flush=True)

    trainable = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    total = sum(p.numel() for p in model.parameters())
    kinds = {}
    for n, k in trainable.items():
        key = next((t for t in lora.target_modules + ["in_proj_qkv", "in_proj_z", "out_proj"]
                    if f".{t}." in n or n.endswith(f".{t}")), "other")
        kinds[key] = kinds.get(key, 0) + k
    print(f"LoRA targets {lora.target_modules}\n  trainable {sum(trainable.values()) / 1e6:.1f}M "
          f"of {total / 1e9:.2f}B params; by module: "
          + ", ".join(f"{k} {v / 1e6:.1f}M" for k, v in sorted(kinds.items())), flush=True)

    def score(ids, grad=False):
        x = torch.tensor([ids], device="cuda")
        labels = torch.roll(x, -1, dims=1)   # position t predicts token t+1
        with torch.set_grad_enabled(grad):
            out = forward(model, x, torch.arange(len(ids), device="cuda")[None],
                          seq_lens=torch.tensor([len(ids)], device="cuda"), labels=labels,
                          temperature=torch.ones_like(x, dtype=torch.float32))
        return out["logprobs"][0]

    if data:
        samples = json.loads(Path(data).read_text())
        d_all, kl_all, mask_all, pos_all = [], [], [], []
        t0 = time.time()
        for s in samples:
            ids = s["prompt_ids"] + s["completion_ids"]
            lp = score(ids)
            n_p = len(s["prompt_ids"])
            train = lp[n_p - 1: n_p - 1 + len(s["completion_ids"])].float()
            infer = torch.tensor(s["vllm_logprobs"], device="cuda", dtype=torch.float32)
            r = train - infer
            d_all.append(r)
            kl_all.append(_mismatch_kl_from_log_ratio(r))
            mask_all.append((train.exp() - infer.exp()).abs() > 0.3)
            pos_all.append(torch.arange(n_p, n_p + len(infer), device="cuda"))
        d, kl, masked, pos = (torch.cat(v) for v in (d_all, kl_all, mask_all, pos_all))
        print(f"\nscored {len(samples)} sequences ({d.numel()} sampled tokens) in {time.time() - t0:.0f}s")
        print(f"  log p_train - log p_vllm: mean {d.mean():+.5f} | mean |.| {d.abs().mean():.5f} | "
              f"p99 |.| {d.abs().quantile(0.99):.4f} | max |.| {d.abs().max():.3f}")
        print(f"  mismatch_kl mean {kl.mean():.6f} | IPO-masked (|dp| > 0.3) {masked.float().mean():.4%} "
              f"({int(masked.sum())} tokens)")
        for lo, hi in [(0, 8192), (8192, 16384), (16384, 24576), (24576, MAX_LEN)]:
            sel = (pos >= lo) & (pos < hi)
            if sel.any():
                print(f"  context {lo // 1024:2d}K-{hi // 1024:2d}K: {int(sel.sum()):6d} tok | "
                      f"mismatch_kl {kl[sel].mean():.6f} | masked {masked[sel].float().mean():.4%}")
        longest = max(samples, key=lambda s: len(s["prompt_ids"]) + len(s["completion_ids"]))
        ids = longest["prompt_ids"] + longest["completion_ids"]
    else:
        ids = list(range(1000, 1000 + MAX_LEN))

    # one backward pass at the longest real length: do gradients reach the adapters?
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    lp = score(ids, grad=True)
    (-lp[-MAX_NEW:].mean()).backward()
    torch.cuda.synchronize()
    grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    def nz(tag):   # fresh LoRA: B = 0, so grad(B) != 0 but grad(A) = B^T(...) = 0 at step 0
        sel = [g for n, g in grads.items() if tag in n]
        return f"{sum(1 for g in sel if g is not None and g.abs().sum() > 0)}/{len(sel)}"
    print(f"\nbackward on {len(ids)} tokens: {time.time() - t0:.1f}s | nonzero grads: "
          f"lora_B {nz('lora_B')}, lora_A {nz('lora_A')} (A is 0 by design at step 0) | "
          f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["vllm", "trainer"])
    ap.add_argument("--traces")
    ap.add_argument("--data", default="")
    ap.add_argument("--out")
    ap.add_argument("--gdn-lora", action="store_true")
    a = ap.parse_args()
    if a.phase == "vllm":
        phase_vllm(a.traces, a.out)
    else:
        phase_trainer(a.data, a.gdn_lora)
