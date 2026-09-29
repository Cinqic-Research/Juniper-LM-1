# JuniperBench-Code-v1.2

v1.2 is the corrected primary task suite. It contains 160 tasks based on v1/v1.1,
with forward-only corrections recorded in `../juniperbench_code_v1/ERRATA.md` and
`ERRATA.md`. The v1 and v1.1 task files and tags remain immutable.

Run the independent task checks and verify the freeze with:

```bash
juniper bench validate --suite v1.2
juniper bench verify --suite v1.2
```

The frozen task file and `FREEZE.json` bind every task, the protocol, the authoring
source, evaluator and sandbox source hashes, and validation checks. The freeze command
was run once to create these records and refuses to overwrite an existing freeze.
`authoring/overrides.py` is the small correction layer over the original authoring
files; it never edits the v1/v1.1 sources.

## Scoring rules

Behavioral tests still run inside the bubblewrap sandbox. The v1.2 records add
explicit implementation constraints for tasks whose prompts already require a
particular method or API. The evaluator parses the prompt plus completion to resolve
names, but applies the configured AST rules only to the candidate completion. A
source-rule violation fails the task before behavioral execution. These rules are
scoring constraints, not a security boundary; bubblewrap remains the execution
boundary, and static source analysis is not a proof against every obfuscated form.

The reference validation field is `reference_score_repeatable`. It records whether
two executions produce the same pass/fail score. It does not establish equality of
outputs or determinism of model generation. Historical v1/v1.1 freeze files retain
their original `reference_deterministic` field name.

## Evaluation state

The protocol defines deterministic and sampling tracks, but Juniper LM 1 does not yet
have an inference/evaluation runner. No B0 or Juniper checkpoint has been evaluated
on this suite. Validation results describe the reference implementation, stubs,
buggy implementations, and test mutants only.
