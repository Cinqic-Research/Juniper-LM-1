# Failure: JuniperBench-Code-v1 froze a task that exceeds the generation cap

**Status:** defect in a frozen artifact; documented in
`evals/juniperbench_code_v1/ERRATA.md` (E1). v1 is not edited.

## What happened

The post-freeze self-review measured every canonical solution against the protocol's
`max_new_tokens: 512`. `JBC1/cin/008` needs 665 tokens as written. Compact rewrites
reached 552 and 536 tokens and still passed every test, but none fit under 512 without
unnatural golfing.

## Why it slipped through

The pre-freeze validator checked the context window (prompt + solution ≤ 992) but not
the generation cap. The two limits bind differently: cin/008's prompt is short (113
tokens), so the context check had plenty of room while the generation cap did not.

## Fix

- `solution_fits_generation_cap` is now part of `validate_task`. Only cin/008 fails it.
- A v1.1 correction is proposed in the errata. Scoring rules for v1 are fixed there
  before any evaluation.

## Lesson

Validate against *every* limit the protocol imposes, not only the model's context
window, and review the protocol and suite together before freezing either.
