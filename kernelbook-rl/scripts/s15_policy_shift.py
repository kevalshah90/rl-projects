"""Step 15: did rl-main-1 move the policy, and in the direction GRPO pushes? (offline check)

Runs inside the s11 image on one H100 (no sandboxes, no sampling):
    uv run modal run --detach scripts/s11_prime_rl_image.py --which policy_shift
    uv run modal run --detach scripts/s11_prime_rl_image.py --which policy_shift --run-name rl-main-2 --args 6
(env RUN / STEP; defaults rl-main-1 / 20. "v20" below means the checkpoint at STEP.)

Takes episodes the run already sampled (trace stream on the Volume) and scores every token the
model sampled, in its exact context, under two policies built with prime-rl's own model stack:
    v0   base Qwen3.5-9B (fresh LoRA, B = 0)
    v20  the step-20 checkpoint (rl/rl-main-1/checkpoints/step_20/trainer)
Per token: delta = log p_v20 - log p_v0.

Episode sets (advantage A = reward - mean of its group of 4):
    trained_v0     mixed groups sampled entirely by v0 (step 1); trained on at step 1
    trained_late   mixed groups sampled by v(STEP-5)+; trained on in the last ~5 steps
    untrained_pass all-pass groups (A = 0, dropped before training)
    untrained_fail all-fail groups (A = 0, dropped before training)

Expected if the policy learned: delta(A > 0) > delta(A < 0) within the trained sets; and
delta(untrained_pass) > delta(untrained_fail) if it generalizes to passing behaviour it never
trained on. KL(v0 || v20) on v0-sampled tokens measures how far the policy moved overall.

Version matching (2026-11-20): tokens sampled by a later policy are already more likely under the
checkpoint, whatever their reward. So compare only episodes sampled by the same policy version:
untrained pass and fail groups are drawn in equal numbers per sampling version (a group's version
= the newest version among its tokens), and the pass - fail difference is averaged over versions.
Within a trained group all 4 episodes share versions, so A > 0 vs A < 0 is already matched.
Sanity: on v0-sampled tokens, v0's logprobs must match vLLM's recorded ones (mismatch KL ~3e-4,
as in check 4), which proves the token sequences were rebuilt correctly.
"""

import json
import os
import random
import statistics as st
import time
from collections import defaultdict
from pathlib import Path

MODEL = "Qwen/Qwen3.5-9B"
MAX_LEN = 32768
RUN_NAME, STEP = os.environ.get("RUN", "rl-main-1"), int(os.environ.get("STEP", "20"))
RUN = Path("/runs/rl") / RUN_NAME
STREAM = RUN / "monitors/file/traces/stream/00000.jsonl"
CKPT = RUN / f"checkpoints/step_{STEP}/trainer"
OUT = Path("/runs/checks/policy_shift") if RUN_NAME == "rl-main-1" else Path(f"/runs/checks/policy_shift_{RUN_NAME}_step{STEP}")
CAPS = {"trained_v0": 48, "trained_late": 32, "untrained_pass": 24, "untrained_fail": 24}
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
           "in_proj_qkv", "in_proj_z", "out_proj"]   # as configs/rl_main.toml


# ---------------------------------------------------------------- data
def reward_of(rec):
    t = rec["traces"][0]
    r = t.get("reward")
    return float(r if r is not None else t["rewards"]["reward"]["score"])


def versions(rec):
    vs = set()
    for tr in rec["traces"]:
        for n in tr["nodes"]:
            for w in n.get("weight_versions") or []:
                vs.add(int(w["version"]))
    return vs


def branches(rec):
    """Each leaf path -> (token ids, [(position, vllm logprob, version)] for sampled tokens).
    A sampled node shared by several branches is scored once (in the first branch)."""
    out, seen = [], set()
    for ti, tr in enumerate(rec["traces"]):
        nodes = tr["nodes"]
        parents = {n.get("parent") for n in nodes}
        for leaf in (i for i in range(len(nodes)) if i not in parents):
            path, i = [], leaf
            while i is not None:
                path.append(i)
                i = nodes[i].get("parent")
            ids, picks = [], []
            for i in reversed(path):
                n = nodes[i]
                if n.get("sampled") and (ti, i) not in seen:
                    seen.add((ti, i))
                    lps = iter(n["logprobs"])
                    ver = int((n.get("weight_versions") or [{"version": -1}])[0]["version"])
                    for j, m in enumerate(n["mask"]):
                        if m:
                            picks.append((len(ids) + j, next(lps), ver))
                ids.extend(n["token_ids"])
            if picks and len(ids) <= MAX_LEN:
                out.append((ids, picks))
    return out


def select():
    recs = []
    with open(STREAM) as f:
        for line in f:
            r = json.loads(line)
            if r.get("ok") and r.get("traces"):
                recs.append({"id": r["id"], "task": r["task"]["data"]["name"], "group": r["group"]["id"],
                             "reward": reward_of(r), "versions": versions(r), "rec": r})
    groups = defaultdict(list)
    for r in recs:
        groups[r["group"]].append(r)
    sets = defaultdict(list)
    untrained = {"untrained_pass": defaultdict(list), "untrained_fail": defaultdict(list)}
    for g in groups.values():
        rs = [r["reward"] for r in g]
        if len(g) < 4:
            continue
        mean = st.mean(rs)
        for r in g:
            r["A"] = r["reward"] - mean
        vs = set().union(*(r["versions"] for r in g))
        if 0 < mean < 1:
            if vs == {0}:
                sets["trained_v0"].append(g)
            elif min(vs) >= max(1, STEP - 5) and max(vs) < STEP:
                sets["trained_late"].append(g)
        else:
            untrained["untrained_pass" if mean == 1 else "untrained_fail"][max(vs)].append(g)
    random.seed(0)
    pairs = []   # (pass group, fail group) sampled by the same policy version
    for v in sorted(set(untrained["untrained_pass"]) & set(untrained["untrained_fail"])):
        gp, gf = untrained["untrained_pass"][v][:], untrained["untrained_fail"][v][:]
        random.shuffle(gp)
        random.shuffle(gf)
        pairs += [(v, a, b) for a, b in zip(gp, gf)]
    random.shuffle(pairs)
    for v, a, b in pairs[:CAPS["untrained_pass"] // 4]:
        for r in a + b:
            r["gver"] = v
        sets["untrained_pass"].append(a)
        sets["untrained_fail"].append(b)
    chosen = []
    for name, cap in CAPS.items():
        gs = sets[name][:]
        random.shuffle(gs)
        eps = [r for g in gs for r in g][:cap]   # whole groups first, so A>0 and A<0 both appear
        for r in eps:
            r["set"] = name
        chosen += eps
        print(f"{name}: {len(sets[name])} groups available (version-matched for untrained), "
              f"{len(eps)} episodes chosen", flush=True)
    for r in chosen:
        r["branches"] = branches(r.pop("rec"))
    return chosen


# ---------------------------------------------------------------- scoring
def main():
    import torch
    from prime_rl.configs.trainer import LoRAConfig, ModelConfig
    from prime_rl.trainer.ckpt import _load_app_state
    from prime_rl.trainer.lora import get_lora_state
    from prime_rl.trainer.model import forward, setup_model
    from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
    from prime_rl.trainer.utils import setup_torch_distributed

    t0 = time.time()
    eps = select()
    n_tok = sum(len(p) for e in eps for _, p in e["branches"])
    print(f"{len(eps)} episodes, {sum(len(e['branches']) for e in eps)} branches, {n_tok} sampled tokens "
          f"(selected in {time.time() - t0:.0f}s)", flush=True)

    cfg = ModelConfig(name=MODEL, seq_len=MAX_LEN, lora=LoRAConfig(rank=16, alpha=32.0, target_modules=TARGETS))
    setup_torch_distributed()
    resolve_ep(cfg)
    model = setup_model(cfg, get_parallel_dims(cfg))
    get_lora_state().reset_adapter_parameters()   # B = 0: the base policy v0

    def score_all(key):
        t0 = time.time()
        for e in eps:
            for b, (ids, picks) in enumerate(e["branches"]):
                x = torch.tensor([ids], device="cuda")
                with torch.no_grad():
                    lp = forward(model, x, torch.arange(len(ids), device="cuda")[None],
                                 seq_lens=torch.tensor([len(ids)], device="cuda"),
                                 labels=torch.roll(x, -1, dims=1),
                                 temperature=torch.ones_like(x, dtype=torch.float32))["logprobs"][0].float()
                pos = torch.tensor([p for p, _, _ in picks], device="cuda") - 1   # lp[t] scores token t+1
                e.setdefault(key, {})[b] = lp[pos].tolist()
        print(f"scored {key} in {time.time() - t0:.0f}s", flush=True)

    score_all("v0")
    _load_app_state(CKPT, model, [], scheduler=None, progress=None, model_only=True)
    print(f"loaded {CKPT}", flush=True)
    score_all("v20")

    # ---------------------------------------------------------------- report
    def k3(d):   # KL(p || q) estimator on p-sampled tokens, d = log q - log p
        return [pow(2.718281828459045, x) - 1 - x for x in d]

    def boot(xs, n=5000):
        random.seed(1)
        m = sorted(st.mean(random.choices(xs, k=len(xs))) for _ in range(n))
        return m[int(.025 * n)], m[int(.975 * n)]

    sanity, kl_v0, per_ep = [], [], []
    for e in eps:
        deltas = []
        for b, (_, picks) in enumerate(e["branches"]):
            for (pos, vlp, ver), a, z in zip(picks, e["v0"][b], e["v20"][b]):
                deltas.append(z - a)
                if ver == 0:
                    sanity.append(vlp - a)    # vLLM (v0) vs trainer v0
                    kl_v0.append(z - a)
        per_ep.append({"id": e["id"], "task": e["task"], "set": e["set"], "reward": e["reward"], "A": e["A"],
                       "gver": e.get("gver"), "versions": sorted(e["versions"]),
                       "n": len(deltas), "mean_delta": st.mean(deltas), "sum_delta": sum(deltas)})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "episodes.json").write_text(json.dumps(per_ep, indent=1))

    print("\n=== sanity: trainer v0 vs vLLM logprobs on v0-sampled tokens ===")
    print(f"  {len(sanity)} tokens | mismatch KL {st.mean(k3(sanity)):.6f} | mean |d| {st.mean(abs(x) for x in sanity):.5f}")
    print("\n=== how far the policy moved: v0 -> v20 on v0-sampled tokens ===")
    print(f"  KL(v0 || v20) {st.mean(k3(kl_v0)):.6f} | mean delta {st.mean(kl_v0):+.5f} | "
          f"mean |delta| {st.mean(abs(x) for x in kl_v0):.5f} | tokens {len(kl_v0)}")
    print("\n=== direction: mean per-token delta (log p_v20 - log p_v0) per episode ===")
    for name in CAPS:
        sel = [p for p in per_ep if p["set"] == name]
        for lab, f in [("A>0", lambda p: p["A"] > 0), ("A<0", lambda p: p["A"] < 0), ("A=0", lambda p: p["A"] == 0)]:
            xs = [p["mean_delta"] for p in sel if f(p)]
            if len(xs) >= 2:
                lo, hi = boot(xs)
                print(f"  {name:15s} {lab}: n={len(xs):3d} | mean delta {st.mean(xs):+.5f} (95% CI {lo:+.5f} .. {hi:+.5f}) | "
                      f"seq delta {st.mean(p['sum_delta'] for p in sel if f(p)):+.2f}")
    tr = [p for p in per_ep if p["set"].startswith("trained")]
    if len(tr) > 2:
        xa, xd = [p["A"] for p in tr], [p["mean_delta"] for p in tr]
        print(f"\n  corr(A, mean delta) over trained episodes: {st.correlation(xa, xd):+.3f} (n={len(tr)})")
    for name in [n for n in CAPS if n.startswith("trained")]:
        pos = [p["mean_delta"] for p in per_ep if p["set"] == name and p["A"] > 0]
        neg = [p["mean_delta"] for p in per_ep if p["set"] == name and p["A"] < 0]
        if pos and neg:
            print(f"  {name}: A>0 - A<0 gap {st.mean(pos) - st.mean(neg):+.5f}")
    # generalization, version-matched: per sampling version, mean(pass) - mean(fail); then the
    # average over versions weighted by the smaller count, with a bootstrap over episodes
    by_v = defaultdict(lambda: {"untrained_pass": [], "untrained_fail": []})
    for p in per_ep:
        if p["set"].startswith("untrained"):
            by_v[p["gver"]][p["set"]].append(p["mean_delta"])
    strata = [(v, d["untrained_pass"], d["untrained_fail"]) for v, d in sorted(by_v.items())
              if d["untrained_pass"] and d["untrained_fail"]]
    def matched(st_):
        w = [min(len(a), len(b)) for _, a, b in st_]
        return sum(wi * (st.mean(a) - st.mean(b)) for wi, (_, a, b) in zip(w, st_)) / sum(w)
    if strata:
        for v, a, b in strata:
            print(f"  untrained v{v}: pass n={len(a)} {st.mean(a):+.5f} | fail n={len(b)} {st.mean(b):+.5f}")
        random.seed(2)
        bs = sorted(matched([(v, random.choices(a, k=len(a)), random.choices(b, k=len(b)))
                             for v, a, b in strata]) for _ in range(2000))
        print(f"  untrained pass - fail (version-matched): {matched(strata):+.5f} "
              f"(95% CI {bs[50]:+.5f} .. {bs[1950]:+.5f}) <- generalization to unseen passing behaviour")
    print(f"\nper-episode results -> {OUT / 'episodes.json'}", flush=True)


if __name__ == "__main__":
    main()
