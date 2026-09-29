"""Convert the verified working copy to the B0 training representation and gate it.

Steps (all reads come from the *working copy*; the canonical baseline is never opened
here except through the frozen manifest's hashes):

  1. verify the working copy against the canonical manifest (content hashes);
  2. load GPT2LMHeadModel (FP32) and re-save as safetensors into checkpoints/B0-...;
  3. copy tokenizer/config files byte-for-byte;
  4. run the pre-registered equivalence gate (configs/eval/conversion_gate.yaml);
  5. write reports/baseline/conversion-equivalence.json. Exit non-zero on failure.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file
from transformers import AutoTokenizer, GPT2LMHeadModel

from juniper_lm1.baseline.manifest import sha256_file, verify
from juniper_lm1.baseline.reference_gpt2 import ReferenceGPT2
from juniper_lm1.paths import REPO_ROOT, load_paths

B0_NAME = "B0-gpt2-124m-converted"
COPY_FILES = ["config.json", "generation_config.json", "vocab.json", "merges.txt",
              "tokenizer.json", "tokenizer_config.json"]

_PARAGRAPH = (
    "The history of the printing press shows how a single technical change can reshape "
    "the way ideas travel. Before movable type, books were copied by hand, slowly and "
    "expensively. Afterwards, pamphlets, maps, and scientific papers spread across Europe "
    "within a few decades. "
)
PROBE_TEXTS = [
    "Hello, my name is",
    "The quick brown fox jumps over the lazy dog.",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "import torch\nimport torch.nn as nn\n\nclass MLP(nn.Module):\n    def __init__(self, d):\n"
    "        super().__init__()\n        self.fc = nn.Linear(d, d)\n\n    def forward(self, x):\n",
    "x = torch.randn(4, 3, device='cuda')\ny = x.view(-1)\nprint(y.shape)  # torch.Size([",
    "User:\nWhat is a tensor?\n\nAssistant:\n",
    "Café naïve résumé — “smart quotes” and emoji 🙂 plus\ttabs\n\n\nand   spaces.",
    "<|endoftext|>In 1848, the city council voted to",
    _PARAGRAPH * 12,  # long input: exercises positions up to ~1024
]


def _tensor_equal(a: dict, b: dict) -> dict:
    a = {k.removeprefix("transformer."): v for k, v in a.items() if not k.endswith(".attn.bias")
         and not k.endswith("masked_bias") and k != "lm_head.weight"}
    b = {k.removeprefix("transformer."): v for k, v in b.items() if not k.endswith(".attn.bias")
         and not k.endswith("masked_bias") and k != "lm_head.weight"}
    mismatched = [k for k in a.keys() & b.keys() if not torch.equal(a[k].float(), b[k].float())]
    return {"only_in_converted": sorted(a.keys() - b.keys()),
            "only_in_reference": sorted(b.keys() - a.keys()),
            "bitwise_mismatched": sorted(mismatched), "tensors_compared": len(a.keys() & b.keys())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", type=Path, default=REPO_ROOT / "configs/eval/conversion_gate.yaml")
    ap.add_argument("--manifest", type=Path,
                    default=REPO_ROOT / "reports/baseline/gpt2-124m-original-flowbox.manifest.json")
    ap.add_argument("--force", action="store_true", help="overwrite an existing B0 directory")
    args = ap.parse_args(argv)

    gate = yaml.safe_load(args.gate.read_text())
    paths = load_paths()
    src = Path(paths["workspace"]["baseline_copy_dir"])
    dst = Path(paths["workspace"]["checkpoints_dir"]) / B0_NAME
    manifest = json.loads(args.manifest.read_text())
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    checks: dict[str, bool] = {}
    report: dict = {"started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "gate": gate, "source": str(src), "converted": str(dst),
                    "torch": torch.__version__}

    # 1. Working copy must match the canonical manifest byte-for-byte.
    problems = verify(src, manifest, check_mtime=False)
    report["working_copy_problems"] = problems
    checks["working_copy_matches_canonical"] = not problems
    if problems:
        return _finish(report, checks)

    # 2-3. Convert.
    if dst.exists():
        if not args.force:
            print(f"{dst} exists; pass --force to rebuild")
            return 2
        shutil.rmtree(dst)
    model = GPT2LMHeadModel.from_pretrained(src, torch_dtype=torch.float32,
                                            use_safetensors=True).eval()
    model.save_pretrained(dst, safe_serialization=True)
    for name in COPY_FILES:
        shutil.copyfile(src / name, dst / name)
    report["converted_files"] = {p.name: sha256_file(p) for p in sorted(dst.iterdir()) if p.is_file()}

    # 4. Gate.
    b0 = GPT2LMHeadModel.from_pretrained(dst, torch_dtype=torch.float32).eval()
    cfg = b0.config
    exp = gate["expected"]
    n_params = sum(p.numel() for p in b0.parameters())
    report["architecture"] = {"model_type": cfg.model_type, "n_layer": cfg.n_layer,
                              "n_head": cfg.n_head, "n_embd": cfg.n_embd,
                              "n_positions": cfg.n_positions, "vocab_size": cfg.vocab_size,
                              "tie_word_embeddings": cfg.tie_word_embeddings,
                              "parameter_count": n_params}
    checks["architecture_config"] = all(report["architecture"][k] == v for k, v in exp.items())

    base_hashes = {f["path"]: f["sha256"] for f in manifest["files"]}
    checks["tokenizer_hash"] = all(sha256_file(dst / f) == base_hashes[f]
                                   for f in gate["tokenizer_files"])
    tok = AutoTokenizer.from_pretrained(dst)
    report["tokenizer_len"] = len(tok)
    checks["vocab_size_50257"] = len(tok) == 50257 == cfg.vocab_size
    checks["max_positions_1024"] = cfg.n_positions == 1024

    ref_state = torch.load(src / "pytorch_model.bin", map_location="cpu", weights_only=True)
    report["weights_vs_pytorch_model_bin"] = _tensor_equal(load_file(dst / "model.safetensors"),
                                                           ref_state)
    ref = ReferenceGPT2(ref_state)
    ref64 = ReferenceGPT2(ref_state, dtype=torch.float64)
    b0_64 = GPT2LMHeadModel.from_pretrained(dst, torch_dtype=torch.float64).eval()
    source = GPT2LMHeadModel.from_pretrained(src, torch_dtype=torch.float32,
                                             use_safetensors=True).eval()

    k, margin = gate["topk_ranking_k"], gate["topk_near_tie_margin"]
    per_text = []
    worst = {"source_vs_converted_fp32": 0.0, "reference_vs_converted_fp64": 0.0,
             "reference_vs_converted_fp32_informational": 0.0}
    ranking_ok = greedy_ok = True
    for text in PROBE_TEXTS:
        ids = torch.tensor(tok(text)["input_ids"])[:1024]
        with torch.no_grad():
            got = b0(ids[None]).logits[0]
            src_logits = source(ids[None]).logits[0]
            got64 = b0_64(ids[None]).logits[0]
        want = ref.logits(ids)
        errs = {"source_vs_converted_fp32": (got - src_logits).abs().max().item(),
                "reference_vs_converted_fp64": (got64 - ref64.logits(ids)).abs().max().item(),
                "reference_vs_converted_fp32_informational": (got - want).abs().max().item()}
        for name, e in errs.items():
            worst[name] = max(worst[name], e)
        # Top-k sets must agree wherever the reference ranking is not a near-tie.
        top = want.topk(k + 1, dim=-1)
        decisive = (top.values[:, k - 1] - top.values[:, k]) > margin
        same = torch.tensor([set(a.tolist()) == set(b.tolist()) for a, b in
                             zip(got.topk(k, -1).indices, top.indices[:, :k])])
        argmax_same = bool((got.argmax(-1) == want.argmax(-1)).all())
        rank_pass = argmax_same and bool(same[decisive].all())
        n_new = min(gate["greedy_new_tokens"], 1024 - len(ids))
        g_ref = ref.greedy(ids, n_new) if n_new > 0 else []
        with torch.no_grad():
            g_b0 = b0.generate(ids[None], attention_mask=torch.ones_like(ids)[None],
                               max_new_tokens=n_new, do_sample=False,
                               pad_token_id=50256)[0, len(ids):].tolist() if n_new > 0 else []
        per_text.append({"prefix": text[:60], "tokens": len(ids), "max_abs_err": errs,
                         "argmax_all_positions_equal": argmax_same, "topk_pass": rank_pass,
                         "greedy_equal": g_ref == g_b0, "greedy_b0": tok.decode(g_b0)})
        ranking_ok &= rank_pass
        greedy_ok &= g_ref == g_b0
    report["probes"] = per_text
    report["max_abs_logit_error"] = worst
    w = report["weights_vs_pytorch_model_bin"]
    checks["weights_bitwise_equal_to_pytorch_model_bin"] = not (
        w["only_in_converted"] or w["only_in_reference"] or w["bitwise_mismatched"])
    checks["parameter_count"] = n_params == exp["parameter_count"]
    checks["logit_equivalence_source_vs_converted_fp32_exact"] = (
        worst["source_vs_converted_fp32"] <= gate["source_vs_converted_max_abs_logit_error"])
    checks["logit_equivalence_reference_fp64"] = (
        worst["reference_vs_converted_fp64"] <= gate["reference_fp64_max_abs_logit_error"])
    checks["topk_ranking_equivalence"] = ranking_ok
    checks["greedy_equivalence"] = greedy_ok
    return _finish(report, checks)


def _finish(report: dict, checks: dict) -> int:
    report["checks"] = checks
    report["passed"] = all(checks.values())
    report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    out = REPO_ROOT / "reports/baseline/conversion-equivalence.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print("GATE", "PASSED" if report["passed"] else "FAILED", "->", out)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
