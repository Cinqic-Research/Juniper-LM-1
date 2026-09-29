# Juniper LM 1: research log

This file records decisions in the order they were made, and becomes the technical report.
It keeps **Observed** results separate from **Hypothesis** explanations.

## 1. Environment (2026-09-29)

FLOWBOX: AMD Ryzen 7 5700G (8C/16T), 14 GiB RAM, NVIDIA GeForce RTX 2060 (Turing, compute
7.5, 6 GiB VRAM), driver 595.91.07 / CUDA 13.2, Ubuntu 24.04 (kernel 7.0.0-34), Python
3.12.3. Root NVMe is 98% full, so all project state lives on the 458 GB ext4 HDD
("Cinqic Storage"). Full inventory: `reports/environment/`.

Consequences recorded as design constraints:

- **Precision.** Turing has no native BF16, so training uses FP16 autocast with dynamic loss
  scaling. Baseline and equivalence checks stay FP32.
- **Memory.** 6 GiB VRAM for full-weight training of 124M params (FP32 master weights +
  AdamW moments ≈ 2 GB before activations) means micro-batches of a few 1,024-token
  sequences with gradient checkpointing and gradient accumulation.
- **Throughput.** A single RTX 2060 makes the 250M-token CPT budget a multi-day run. Pilots
  (25–50M tokens) must be sized accordingly. Measured throughput will be recorded here.
- **Sandbox.** Docker/Podman are absent; `bubblewrap` (`bwrap`) is available and will host
  the generated-code evaluator (no network, read-only binds, no secrets, rlimits).

## 2. Baseline preservation (2026-09-29)

- Canonical baseline: the pre-existing FLOWBOX snapshot of `openai-community/gpt2` at
  revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, ID `gpt2-124m-original-flowbox`.
  It holds 81 files (all upstream serializations: safetensors, PyTorch bin, TF h5, Flax,
  ONNX, TFLite, rust-bert, plus tokenizer and HF cache metadata).
- The original OpenAI TensorFlow `model.ckpt` release is **not** on FLOWBOX. The HF snapshot
  is the canonical artifact. Consequence: the conversion gate's reference is an independent
  PyTorch reimplementation over the snapshot's `pytorch_model.bin`, not OpenAI's TF code.
- External manifest (per-file size, SHA-256, mtime, format, and directory digest
  `756e80788d3a016d1de60bc329c6a5688acaac6e8883ca3224fb5e9d9f530a87`) is frozen at
  `reports/baseline/gpt2-124m-original-flowbox.manifest.json`. It agrees with an earlier,
  independent hash list made by the 2026-09-28 setup session.
- The canonical directory had already been made read-only by that earlier session. This
  project did not change its permissions or metadata; protection relies on the external
  manifest.
- The working copy (`cp -a`) in the work area verifies against the manifest, including
  mtimes.
- Open recommendation: a second physical backup of the canonical baseline on another device.

## 3. Conversion equivalence gate (2026-09-29)

- **Attempt 1 failed** under the pre-registered v1 gate: FP32 cross-implementation logit
  error was 3.97e-3, above the 1e-4 limit. Every structural, weight, ranking, and greedy
  check passed. Diagnosis showed the conversion is exact and the implementations agree to
  1e-11 in FP64; 1e-4 was below FP32's noise floor. See
  `reports/failures/conversion-gate-attempt1.md`.
- **Gate amended to v2** with the reason documented in the config before acceptance.
  Source→B0 is now exact in FP32 (tolerance 0), and the independent reference is checked
  in FP64 at 1e-4.
- **Attempt 2 passed** all 11 checks: source→B0 max error 0.0; reference FP64 1.5e-11;
  FP32 argmax, top-10, and greedy identical on 9 probes (up to 649 tokens).
- **B0 accepted:** `B0-gpt2-124m-converted`, `model.safetensors` SHA-256
  `c7d00560d8910fbed77ffad4065dee5011c41ba401b1064e749c498ba9e20373`. Config and
  tokenizer files are byte-identical to upstream. Report:
  `reports/baseline/conversion-equivalence.json`.

## 4. Benchmark authorship (decided 2026-09-29)

JuniperBench-Code-v1 (160 tasks with hidden tests) will be written and frozen by a
**different model from the one implementing and training Juniper**, so the implementer
does not grade its own homework. The implementing agent (Claude Opus 5.5) writes no
benchmark tasks, reference solutions, or hidden tests. It may consume the frozen suite
only mechanically: running the harness and running decontamination scans against the
training corpus. The benchmark author's identity and exact model revision, and the
freeze hash, will be recorded here when the suite lands.
