# JuniperBench-Code-v1.1 (historical protocol successor)

**v1.1 = the 160 v1 tasks, byte-identical, plus protocol v1.1.** There is no separate
task file. `FREEZE.json` points to `../juniperbench_code_v1/juniperbench_code_v1.jsonl`
and records the same SHA-256 (`76ae86e7…`) and per-task hashes as v1.

The only change is in `configs/eval/juniperbench_code_v1_1.yaml`: `max_new_tokens`
goes from a fixed 512 to *remaining context* (1,024 − prompt tokens). This resolves
v1 erratum E1, where `JBC1/cin/008` could not be solved in normally written code under
512 tokens. All 160 tasks pass every validation check under v1.1, including the new
`solution_fits_generation_cap`.

v1.1 was the primary protocol after E1 and before the v1.2 successor. No model has been
evaluated on v1.1. v1 stays frozen as the historical record; v1.2 is now primary.
Task documentation, authorship, and threats to validity: `../juniperbench_code_v1/README.md`.

    juniper bench verify --suite v1.1
