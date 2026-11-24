"""Step 17: merge a run's LoRA checkpoint into the base weights -> a plain HF model dir for vLLM (E1).

Runs inside the s11 image on one H100:
    uv run modal run --detach scripts/s11_prime_rl_image.py --which merge --run-name rl-main-2 --args 20
(env RUN / STEP.) Writes /runs/models/<RUN>-step<STEP>/ on the kernelbook-runs Volume.

Why merge instead of serving base + adapter: E0 served the plain base model with s09_serve.py;
a merged model is served the exact same way (same vLLM, same flags), so E1 vs E0 differs only in
the weights, not in a LoRA code path at inference.

How: prime-rl's own model stack (as s15) loads the base model with the run's LoRA config, then the
step-N checkpoint. Each adapted layer gets W <- W + (alpha / r) * B @ A (in fp32, then bf16). The
LoRA tensors are dropped and prime-rl's prime -> HF conversion (the one its full-weight broadcast to
vLLM uses) writes safetensors. The non-weight files (config, tokenizer, chat template) are copied
from the base snapshot.
Checks: same tensor names and shapes as the base snapshot; tensors outside the LoRA targets are
bit-identical to the base; every targeted tensor changed.
"""

import json
import os
import shutil
import time
from pathlib import Path

MODEL = "Qwen/Qwen3.5-9B"
RUN_NAME, STEP = os.environ.get("RUN", "rl-main-2"), int(os.environ.get("STEP", "20"))
CKPT = Path("/runs/rl") / RUN_NAME / f"checkpoints/step_{STEP}/trainer"
OUT = Path("/runs/models") / f"{RUN_NAME}-step{STEP}"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
           "in_proj_qkv", "in_proj_z", "out_proj"]   # as configs/rl_main*.toml


def local(t):
    from torch.distributed.tensor import DTensor
    return t.to_local() if isinstance(t, DTensor) else t   # one GPU: the local shard is the full tensor


def main():
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from prime_rl.configs.trainer import LoRAConfig, ModelConfig
    from prime_rl.trainer.ckpt import _load_app_state
    from prime_rl.trainer.lora import get_lora_state, strip_lora_from_state_dict
    from prime_rl.trainer.model import setup_model
    from prime_rl.trainer.parallel_dims import get_parallel_dims, resolve_ep
    from prime_rl.trainer.utils import setup_torch_distributed
    from prime_rl.utils.weights import convert_state_dict_to_hf, resolve_fqn, save_state_dict

    t0 = time.time()
    assert (CKPT / ".metadata").exists(), f"no complete checkpoint at {CKPT}"
    cfg = ModelConfig(name=MODEL, seq_len=4096, lora=LoRAConfig(rank=16, alpha=32.0, target_modules=TARGETS))
    setup_torch_distributed()
    resolve_ep(cfg)
    model = setup_model(cfg, get_parallel_dims(cfg))
    _load_app_state(CKPT, model, [], scheduler=None, progress=None, model_only=True)
    print(f"loaded {CKPT} ({time.time() - t0:.0f}s)", flush=True)

    # merge: W += scaling * B @ A
    mods = get_lora_state()._modules
    norms = []
    with torch.no_grad():
        for prefix, m in mods:
            w = local(m.base_layer.weight)
            delta = m.scaling * (local(m.lora_B).float() @ local(m.lora_A).float())
            norms.append(delta.norm().item() / max(w.float().norm().item(), 1e-12))
            w.copy_((w.float() + delta).to(w.dtype))
    print(f"merged {len(mods)} LoRA modules; relative |dW|/|W| mean {sum(norms) / len(norms):.2e}, "
          f"max {max(norms):.2e}", flush=True)

    sd = {resolve_fqn(model, k): local(v).detach().to(torch.bfloat16).cpu()
          for k, v in model.state_dict().items()}
    sd = strip_lora_from_state_dict(sd)
    assert not any(".base_layer." in k for k in sd), "LoRA base_layer names left in the state dict"
    sd = convert_state_dict_to_hf(model, sd)

    # compare with the base snapshot: names, shapes, unchanged / changed tensors
    snap = Path(snapshot_download(MODEL))
    index = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    missing, extra = set(index) - set(sd), set(sd) - set(index)
    if extra == {"lm_head.weight"}:          # tied head: the snapshot ships no lm_head
        sd.pop("lm_head.weight")
        extra = set()
    assert not extra, f"tensors not in the base snapshot: {sorted(extra)[:5]}"
    print(f"{len(missing)} base tensors the trainer doesn't load (copied unchanged): {sorted(missing)[:4]}", flush=True)
    changed, same, bad = 0, 0, []
    for shard in sorted(set(index.values())):
        with safe_open(snap / shard, "pt") as f:
            for k in f.keys():
                base = f.get_tensor(k)
                if k in missing:
                    sd[k] = base.to(torch.bfloat16)
                assert base.shape == sd[k].shape, f"shape mismatch {k}: {base.shape} vs {sd[k].shape}"
                is_target = any(f".{t}." in k for t in TARGETS) and "visual" not in k
                eq = torch.equal(base.to(torch.bfloat16), sd[k])
                if is_target:
                    changed += not eq
                else:
                    same += eq
                    if not eq:
                        bad.append(("changed non-target", k))
    print(f"vs base: {len(index)} tensors, {changed} LoRA-target tensors changed ({len(mods)} LoRA modules), "
          f"{same} others identical, {len(bad)} problems {bad[:5]}", flush=True)
    assert not bad and changed > 0, "merge check failed"

    tmp = OUT.with_name(OUT.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    save_state_dict(sd, tmp, save_sharded=True)
    for f in snap.iterdir():   # config, tokenizer, chat template, generation config, ...
        if not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json" and f.is_file():
            shutil.copy(f, tmp / f.name)
    (tmp / "MERGED_FROM.json").write_text(json.dumps(
        {"base": MODEL, "checkpoint": str(CKPT), "lora_rank": 16, "lora_alpha": 32.0, "targets": TARGETS}, indent=1))
    shutil.rmtree(OUT, ignore_errors=True)
    tmp.rename(OUT)
    print(f"wrote {OUT} ({sorted(p.name for p in OUT.iterdir())[:6]}...) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
