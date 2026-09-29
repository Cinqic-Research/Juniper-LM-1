# JuniperBench-Code-v1.2 errata

The v1 and v1.1 frozen JSONL files and tags are unchanged. v1.2 is a new task-file
version; its checksum and per-task hashes are recorded in `FREEZE.json`.

## E2: JBC1/py/003 did not normalize the exception set

The task required case-insensitive matching against `minor_words`, but the canonical
solution normalized only the input word. v1.2 case-folds both sides and adds an
uppercase/mixed-case regression case.

## E3: JBC1/test/007 could exceed its length limit

The reference appended the requested suffix even when the suffix itself was longer
than `limit`. v1.2 defines the edge case: when text must be cut and the suffix is
longer than the non-negative limit, the result is `suffix[:limit]`. The tests now
cover default and custom long suffixes.

## E4: prompt-level implementation constraints were not fully scored

Several prompts prescribed an algorithm or API that the hidden behavioral examples
did not distinguish. v1.2 makes these requirements observable:

- `JBC1/py/021`: a sequence probe rejects linear indexed traversal and enforces the
  stated binary-search behavior.
- `JBC1/py/022`: calls to `sorted()` and `.sort()` are rejected.
- `JBC1/pt/013`: loops and comprehension forms are rejected for the broadcasting
  implementation.
- `JBC1/pt/032`: explicit Hessian/Jacobian APIs are rejected and a large vector
  probe exercises a Hessian-vector product without a dense matrix.
- `JBC1/pt/039`: `nn.LayerNorm` and `F.layer_norm` are rejected.
- `JBC1/pt/043` and `JBC1/pt/045`: `torch.nn.functional` imports and the named
  functional losses are rejected.
- `JBC1/cin/008`: calls to `eval()` and `exec()` are rejected.

The checks are limited AST scoring rules. They cover ordinary direct/import/assignment
alias spellings, but they are not a proof against every obfuscated equivalent.

## E5: reference score repeatability was mislabeled as determinism

The historical v1/v1.1 freeze field `reference_deterministic` compares only two
boolean pass/fail scores. v1.2 calls this `reference_score_repeatable` and documents
that it does not measure output determinism or model-generation determinism. The
historical freeze records are left intact.

## E6: portable report paths

The tracked baseline and environment reports contained local absolute mount paths.
The current-tree copies replace those strings with portable artifact labels while
preserving the measurements, hashes, outcomes, and machine inventory. The original
Git history and frozen tags are unchanged; this is a current-report redaction only.
The pre-redaction file checksums at the review start commit are retained here:

| File | SHA-256 before redaction |
|---|---|
| `reports/baseline/conversion-equivalence.json` | `ceec645006f898eaf2645dda52ac1f1b5fddc24e4638c090fd7dfa91c1d92d43` |
| `reports/baseline/gpt2-124m-original-flowbox.manifest.json` | `68a56f3af94712b842febdf974eb6cc8321707f50d3e6c8a2e1e61e63fd6d15f` |
| `reports/environment/FLOWBOX-20260929.json` | `b78ba4a49d98261eaab4ca35183f4a6bb28c9929841765fdd04776773598cda5` |
| `reports/environment/FLOWBOX-20260929.md` | `a4c4f6daa4f1b478789d6bc8c3e1ca166795afba95e3e9bdd14718dc5f45e21a` |
| `reports/failures/conversion-gate-attempt1.json` | `1087c72d61744d7c0d99e027928073bb7a112c7a2a448cd6d6fa4df027fa76b6` |

## E7: freeze metadata was not independently checked

The prior verifier checked file and protocol digests but accepted internally
inconsistent metadata, including wrong task counts, category counts, canary, file
paths, validation records, and baseline manifest summary fields. The v1.2 verifier
checks these fields against the task rows, authoring source, protocol, live files,
directory digest, detected baseline identity, and (for v1.2) evaluator and sandbox
source hashes. Historical freeze files are read using their original flat validation
schema.
