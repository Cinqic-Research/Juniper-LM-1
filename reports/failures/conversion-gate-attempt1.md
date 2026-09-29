# Failure: B0 conversion gate, attempt 1 (2026-09-29)

**Status:** failed (gate v1). Superseded by gate v2, which passed.
Raw report: `conversion-gate-attempt1.json`.

## Hypothesis

Re-serializing the verified working copy via `transformers` GPT2LMHeadModel → safetensors
preserves GPT-2 behavior to within 1e-4 max-abs logit error of an independent pure-PyTorch
GPT-2 reference, both in FP32 on CPU.

## What happened

| Check | Result |
|---|---|
| Working copy == canonical manifest | pass |
| Architecture / tokenizer hashes / vocab 50,257 / positions 1,024 / 124,439,808 params | pass |
| 148 weight tensors bitwise equal to upstream `pytorch_model.bin` | pass |
| FP32 argmax, top-10 set, 32-token greedy identical on all 9 probes | pass |
| **FP32 max-abs logit error ≤ 1e-4** | **fail: 3.97e-3** (worst probe: 81-token PyTorch snippet) |

## Diagnosis

On the worst probe:

| Comparison | Max abs diff |
|---|---|
| HF(original safetensors) vs HF(B0), FP32 | 0.0 |
| Reference vs HF, both FP64 | 1.0e-11 |
| HF SDPA vs HF eager attention (same weights, FP32) | 1.4e-3 |
| HF FP32 vs FP64 ground truth | 5.0e-3 |
| Reference FP32 vs FP64 ground truth | 9.9e-4 |

Maximum logit magnitude: 338. Error per position grows along the sequence (≈1e-5 at the
first tokens), consistent with FP32 rounding building up through 12 layers of GPT-2's
large-magnitude residual stream.

**Observed:** the conversion is exact, and the two implementations are mathematically
identical. The v1 tolerance sat below the FP32 noise floor of *any* correct GPT-2
implementation; `transformers` disagrees with itself by 1.4e-3 across attention kernels.

## Why it is considered a failure

The pre-registered criterion was not met. The v1 rationale ("logits O(10²) ⇒ 1e-4 is ≈1e-6
relative, far above FP32 noise") was wrong about the noise floor.

## Informative? Retry justified?

Yes. Gate v2 (`configs/eval/conversion_gate.yaml`, reason documented in-file before B0
was accepted) tests each property where it can be measured: exact FP32 equality between
source and converted models under the same implementation (tolerance 0), FP64
cross-implementation equality at the spec's 1e-4, and unchanged FP32 argmax/top-k/greedy
identity. Attempt 2 passed all 11 checks.

Consequence for later work: FP32 logit comparisons *between implementations or kernels*
carry ~1e-3 of noise and must never be read as a behavioral difference.
