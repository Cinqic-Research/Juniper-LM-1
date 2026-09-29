# JuniperBench-Code-v1 errata

The frozen suite (`juniperbench_code_v1.jsonl`, SHA-256 `76ae86e7…`) and its protocol are
**not modified** by anything listed here. Corrections take effect only in a new version.

## E1: JBC1/cin/008 cannot be solved in normally written Python under the v1 generation cap

*Found 2026-09-29 in the post-freeze self-review, before any model was evaluated.*

- **Defect.** The protocol caps generation at `max_new_tokens: 512`. The canonical
  solution for `JBC1/cin/008` (`evaluate_expression`, a safe infix evaluator) is 665 GPT-2
  tokens. Two deliberately compact rewrites that pass all tests are still 552 and
  536 tokens, because GPT-2's BPE spends about one token per indentation space. The task
  is solvable only with unnatural code golf, so it effectively measures the cap, not
  the capability.
- **Cause.** The validator checked prompt + solution ≤ 992 tokens (the context window)
  but did not check solution ≤ `max_new_tokens`. That check now exists
  (`solution_fits_generation_cap`); every other task passes it.
- **Handling in v1 results.** Report v1 scores on all 160 tasks *and* on the 159 tasks
  excluding cin/008, labeled as such. The exclusion is fixed here, before any result
  exists, and applies identically to every model, B0 included.
- **Correction for v1.1 (not yet published).** Either remove the fixed 512-token cap
  (`max_new_tokens = context_length − prompt_tokens`), or replace cin/008 with a task
  whose natural solution fits. The project owner decides.
