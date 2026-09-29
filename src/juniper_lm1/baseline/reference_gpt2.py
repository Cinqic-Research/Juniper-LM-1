"""Independent, minimal GPT-2 forward pass used only as an equivalence reference.

Deliberately shares no code with transformers' GPT2 implementation: it follows the
original OpenAI GPT-2 definition (pre-LN blocks, tanh-GELU, Conv1D-style x @ W + b
projections, LayerNorm eps=1e-5, tied output embedding).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class ReferenceGPT2:
    def __init__(self, state: dict[str, torch.Tensor], n_layer=12, n_head=12, eps=1e-5,
                 dtype: torch.dtype = torch.float32):
        # Accept keys with or without a leading "transformer." prefix.
        self.w = {k.removeprefix("transformer."): v.to(dtype) for k, v in state.items()}
        self.dtype = dtype
        self.n_layer, self.n_head, self.eps = n_layer, n_head, eps

    def _ln(self, x, name):
        return F.layer_norm(x, x.shape[-1:], self.w[f"{name}.weight"], self.w[f"{name}.bias"],
                            self.eps)

    def _proj(self, x, name):
        return x @ self.w[f"{name}.weight"] + self.w[f"{name}.bias"]

    @torch.no_grad()
    def logits(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (T,) int64 -> (T, vocab) float32 logits."""
        T = ids.shape[0]
        x = self.w["wte.weight"][ids] + self.w["wpe.weight"][:T]
        C, H = x.shape[-1], self.n_head
        mask = torch.full((T, T), float("-inf"), dtype=self.dtype).triu(1)
        for i in range(self.n_layer):
            p = f"h.{i}"
            q, k, v = self._proj(self._ln(x, f"{p}.ln_1"), f"{p}.attn.c_attn").split(C, dim=-1)
            q, k, v = (t.view(T, H, C // H).transpose(0, 1) for t in (q, k, v))
            att = (q @ k.transpose(-1, -2)) / math.sqrt(C // H) + mask
            y = (att.softmax(-1) @ v).transpose(0, 1).reshape(T, C)
            x = x + self._proj(y, f"{p}.attn.c_proj")
            h = _gelu(self._proj(self._ln(x, f"{p}.ln_2"), f"{p}.mlp.c_fc"))
            x = x + self._proj(h, f"{p}.mlp.c_proj")
        return self._ln(x, "ln_f") @ self.w["wte.weight"].T

    @torch.no_grad()
    def greedy(self, ids: torch.Tensor, n_new: int) -> list[int]:
        out = ids.tolist()
        for _ in range(n_new):
            out.append(int(self.logits(torch.tensor(out[-1024:]))[-1].argmax()))
        return out[len(ids):]
