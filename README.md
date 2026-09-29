# Juniper LM 1

**Research question:** how far can the original GPT-2 124M checkpoint be advanced using
modern data, training methodology, instruction tuning, and constitutional alignment while
preserving its fundamental architecture and scale?

Juniper LM 1 specializes GPT-2 small toward Python and PyTorch programming, clear American
English, and calibrated instruction following, and measures every stage against the same
untouched stock GPT-2 baseline. It is a controlled research study, not a frontier model: a
124M-parameter GPT-2 derivative has severe capability limits, and we report them.

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
| JuniperBench-Code | v1.1 is primary (tag `eval-v1.1`; same 160 tasks as v1, protocol fix for erratum E1); authorship caveat in `RESEARCH.md` §4 |
| B0 benchmark, corpus, training | not started |

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
- `MODEL_CARD.md`, `DATA_CARD.md`: filled in as artifacts are produced
- `THIRD_PARTY_NOTICES.md`: upstream terms (GPT-2 weights are *not* Apache-2.0)
- `reports/failures/`: every failed or rejected experiment
- `evals/juniperbench_code_v1/`: the frozen tasks, datasheet, and errata; `evals/juniperbench_code_v1_1/`: the primary suite
- `SECURITY.md`: how generated code is sandboxed

## License

New Cinqic-authored code and documentation: Apache-2.0 (`LICENSE`). Upstream artifacts keep
their own terms; see `NOTICE` and `THIRD_PARTY_NOTICES.md`.
