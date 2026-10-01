# Juniper LM 1

**Research question:** how far can the original GPT-2 124M checkpoint be advanced using
modern data, training methodology, instruction tuning, and alignment experiments while
preserving its fundamental architecture and scale?

Juniper LM 1 is conversational-first. The aim is a small model that holds a clear,
calibrated conversation in contemporary American English; coding, including Python and
PyTorch, is a later specialization. Every stage is measured against the same untouched
stock GPT-2 baseline. It is a controlled research study, not a frontier model: a
124M-parameter GPT-2 derivative has severe capability limits, and we report them.

**No Juniper LM 1 checkpoint has been trained or released.**

## Invariants

| Property | Value (never changed) |
|---|---|
| Architecture | GPT-2 small: 12 layers, 12 heads, 768 width, learned positions, tied embeddings |
| Parameters | 124,439,808 |
| Tokenizer | Original GPT-2 byte-level BPE, 50,257 tokens, no added tokens |
| Context | 1,024 tokens |
| Baseline | `gpt2-124m-original-flowbox` (`openai-community/gpt2@607a30d`), never modified |

## Status

| Phase | State |
|---|---|
| Environment inventory | done: `reports/environment/` |
| Canonical baseline manifest | frozen: `reports/baseline/gpt2-124m-original-flowbox.manifest.json` |
| Verified working copy | done: content hashes match the manifest |
| B0 conversion + equivalence gate | passed on attempt 2 (gate v2); attempt 1 failure in `reports/failures/` |
| JuniperBench-Code | v1.2 is primary (successor suite, tag `eval-v1.2`); see `RESEARCH.md` §6 and the versioned errata |
| B0 benchmark, corpus, training | not started; benchmark inference/evaluation runner is not implemented |

## Direction

The project changed from coding-first to conversational-first on 2026-09-30 (`RESEARCH.md`
§7). The intended order is:

1. finish baseline inference and evaluation infrastructure;
2. establish a conversational evaluation baseline for stock GPT-2;
3. improve contemporary American-English language competence;
4. conversational instruction tuning;
5. preference and alignment experiments;
6. coding specialization, including Python and PyTorch;
7. integration with Juniper and [AAA](https://github.com/Cinqic-Research/AAA) as its
   language component, only if the evidence supports it.

This is a plan, not a record of progress; only the rows in the status table above have
happened. Alignment methods are treated as experiments to be measured, not as proven
techniques. JuniperBench-Code v1.2 stays frozen and is used as a secondary measurement:
a retention check during the conversational stages and the primary measurement for the
later coding stage.

## Setup

```bash
cp configs/local.paths.example.toml configs/local.paths.toml   # then edit paths
python3 -m venv venv && venv/bin/pip install -r requirements.lock.txt && venv/bin/pip install -e .
juniper baseline verify      # canonical + working copy vs the frozen manifest
juniper baseline convert     # build B0 and run the pre-registered equivalence gate
```

Raw corpora, checkpoints, caches, and the baseline itself live outside Git in the work
area configured in `configs/local.paths.toml`.

## Documents

- `RESEARCH.md`: protocol, decisions, and results log (eventually the technical report)
- `MODEL_CARD.md`, `DATA_CARD.md`: direction and constraints now; results are added only as artifacts are produced
- `THIRD_PARTY_NOTICES.md`: upstream terms (GPT-2 weights are *not* Apache-2.0)
- `reports/failures/`: every failed or rejected experiment
- `evals/juniperbench_code_v1/`: frozen v1 tasks, datasheet, and errata; `evals/juniperbench_code_v1_1/`: protocol-only v1.1; `evals/juniperbench_code_v1_2/`: corrected primary suite
- `SECURITY.md`: how generated code is sandboxed

## License

New Cinqic-authored code and documentation: Apache-2.0 (`LICENSE`). Upstream artifacts keep
their own terms; see `NOTICE` and `THIRD_PARTY_NOTICES.md`.
