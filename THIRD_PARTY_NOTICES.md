# Third-party notices

The Apache-2.0 license in `LICENSE` covers new Cinqic-authored code and documentation in
this repository only. It does **not** relicense any upstream model, dataset, benchmark, or
code listed below; each keeps its own terms.

## GPT-2 124M (baseline checkpoint and all Juniper weights derived from it)

| Field | Value |
|---|---|
| Artifact | `openai-community/gpt2` (Hugging Face Hub), OpenAI GPT-2 small |
| Exact revision | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| Declared license (Hub model card metadata) | `mit` |
| Original release | `openai/gpt-2` on GitHub (OpenAI), "Modified MIT License" |
| Local canonical copy | `gpt2-124m-original-flowbox`, see `reports/baseline/` |

**Release blocker (open):** before any weight release, fetch and pin the exact license
texts of both `openai-community/gpt2@607a30d` and `openai/gpt-2` (with commit SHA) into
`datasets/licenses/gpt2/` and reproduce the required notices here verbatim.

## Software dependencies

Exact versions are pinned in `requirements.lock.txt`. Each package retains its own license.

## Training / evaluation data

None yet. Every source is added here, and to `DATA_CARD.md`, when it passes the audit gate.
