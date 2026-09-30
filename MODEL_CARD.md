# Model card: Juniper LM 1

**No Juniper LM 1 checkpoint has been trained or released yet.** This card will be written
from measured results only.

Fixed facts: GPT-2 small lineage (`openai-community/gpt2@607a30d`), 124,439,808 parameters,
12 layers / 12 heads / 768 width, original 50,257-token byte-level BPE vocabulary,
1,024-token context. Upstream GPT-2 limitations (factual unreliability, inherited training-data
bias) apply to every derivative.

Intended direction (not a capability claim): a conversational model in contemporary American
English first, with coding (Python and PyTorch) as a later specialization. Alignment work will
be reported as experiments with measured effects. Juniper LM 1 is not a frontier model, and
stock GPT-2 remains its permanent baseline. See `RESEARCH.md` §7.

License: Cinqic's code is Apache-2.0; derived weights would inherit GPT-2's upstream terms,
not Apache-2.0 (`THIRD_PARTY_NOTICES.md`).
