from _schema import task

T = ("torch",)

TASKS = [
# --- autograd ------------------------------------------------------------------
task("JBC1/pt/031", "autograd", "gradient_at", r'''
import torch


def gradient_at(f, x: torch.Tensor) -> torch.Tensor:
    """Return the gradient of the scalar-valued function f at x, as a tensor
    with x's shape. x itself must not be modified (including requires_grad).
    """
''', r'''
    x_var = x.detach().clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(f(x_var), x_var)
    return grad
''', r'''
def check(candidate):
    x = torch.tensor([1.0, -2.0, 3.0])
    g = candidate(lambda t: (t ** 2).sum(), x)
    assert torch.allclose(g, torch.tensor([2.0, -4.0, 6.0]))
    assert x.requires_grad is False and x.grad is None
    assert x.tolist() == [1.0, -2.0, 3.0]
    m = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    g2 = candidate(lambda t: (t.sin() * 3).sum(), m)
    assert g2.shape == (2, 2) and torch.allclose(g2, 3 * m.cos())
'''),

task("JBC1/pt/032", "autograd", "hessian_vector_product", r'''
import torch


def hessian_vector_product(f, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Return H @ v, where H is the Hessian of the scalar function f at the 1D
    tensor x, without forming H explicitly (use double backward).
    """
''', r'''
    x = x.detach().clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(f(x), x, create_graph=True)
    (hvp,) = torch.autograd.grad(grad @ v, x)
    return hvp
''', r'''
def check(candidate):
    A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])
    f = lambda x: 0.5 * x @ A @ x
    v = torch.tensor([1.0, -1.0])
    out = candidate(f, torch.tensor([0.3, 0.7]), v)
    assert torch.allclose(out, A @ v)
    g = lambda x: (x ** 3).sum()
    x = torch.tensor([1.0, 2.0, -1.0])
    assert torch.allclose(candidate(g, x, torch.ones(3)), 6 * x)
'''),

task("JBC1/pt/033", "autograd", "freeze", r'''
import torch.nn as nn


def freeze(module: nn.Module) -> int:
    """Disable gradients for every parameter of module and return the number of
    scalar parameter values frozen.
    """
''', r'''
    total = 0
    for p in module.parameters():
        p.requires_grad_(False)
        total += p.numel()
    return total
''', r'''
def check(candidate):
    import torch
    import torch.nn as nn
    m = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    assert candidate(m) == 3 * 4 + 4 + 4 * 2 + 2
    assert all(not p.requires_grad for p in m.parameters())
    out = m(torch.randn(5, 3))
    assert not out.requires_grad
    assert candidate(nn.ReLU()) == 0
'''),

task("JBC1/pt/034", "autograd", "count_trainable", r'''
import torch.nn as nn


def count_trainable(module: nn.Module) -> int:
    """Return the number of scalar parameter values in module that require
    gradients. Parameters shared between submodules are counted once.
    """
''', r'''
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
''', r'''
def check(candidate):
    import torch.nn as nn
    m = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 1))
    assert candidate(m) == 55 + 6
    m[0].weight.requires_grad_(False)
    assert candidate(m) == 5 + 6
    shared = nn.Linear(4, 4)
    tied = nn.Sequential(shared, nn.ReLU(), shared)
    assert candidate(tied) == 20
    emb = nn.Embedding(100, 8)
    head = nn.Linear(8, 100, bias=False)
    head.weight = emb.weight
    assert candidate(nn.ModuleList([emb, head])) == 800
'''),

task("JBC1/pt/035", "autograd", "detach_state", r'''
import torch


def detach_state(state):
    """Detach a recurrent hidden state from the autograd graph. state is a
    tensor or an arbitrarily nested tuple/list of tensors; keep the structure.
    """
''', r'''
    if isinstance(state, torch.Tensor):
        return state.detach()
    return type(state)(detach_state(s) for s in state)
''', r'''
def check(candidate):
    w = torch.ones(2, requires_grad=True)
    h = w * 2
    c = w * 3
    out = candidate((h, [c, (h,)]))
    assert isinstance(out, tuple) and isinstance(out[1], list) and isinstance(out[1][1], tuple)
    assert not out[0].requires_grad and not out[1][0].requires_grad
    assert not out[1][1][0].requires_grad
    assert torch.equal(out[1][0], c.detach())
    assert not candidate(h).requires_grad
'''),

task("JBC1/pt/036", "autograd", "clip_grad_norm", r'''
import torch


def clip_grad_norm(params, max_norm: float) -> float:
    """Scale the gradients of params in place so that their combined L2 norm is
    at most max_norm, and return the norm before clipping as a Python float.
    Parameters whose .grad is None are skipped.
    """
''', r'''
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    total = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads)).item()
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for g in grads:
            g.mul_(scale)
    return total
''', r'''
def check(candidate):
    a = torch.zeros(2, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    c = torch.zeros(3, requires_grad=True)
    a.grad = torch.tensor([3.0, 0.0])
    b.grad = torch.tensor([4.0])
    total = candidate([a, b, c], 1.0)
    assert isinstance(total, float) and abs(total - 5.0) < 1e-5
    new_norm = (a.grad.norm() ** 2 + b.grad.norm() ** 2).sqrt().item()
    assert abs(new_norm - 1.0) < 1e-4
    assert torch.allclose(a.grad, torch.tensor([0.6, 0.0]), atol=1e-4)
    d = torch.zeros(1, requires_grad=True)
    d.grad = torch.tensor([0.5])
    assert abs(candidate([d], 1.0) - 0.5) < 1e-6 and d.grad.item() == 0.5
'''),

# --- nn.Module -----------------------------------------------------------------
task("JBC1/pt/037", "modules", "MLP", r'''
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron: n_layers nn.Linear layers stored in an
    nn.ModuleList called `layers`, mapping in_dim -> hidden -> ... -> out_dim,
    with ReLU between layers and no activation after the last one.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int, n_layers: int = 2):
''', r'''
        super().__init__()
        dims = [in_dim] + [hidden] * (n_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x
''', r'''
def check(candidate):
    torch.manual_seed(0)
    m = candidate(5, 8, 3, n_layers=3)
    assert isinstance(m.layers, nn.ModuleList) and len(m.layers) == 3
    assert [l.in_features for l in m.layers] == [5, 8, 8]
    assert [l.out_features for l in m.layers] == [8, 8, 3]
    x = torch.randn(4, 5)
    out = m(x)
    assert out.shape == (4, 3)
    h = torch.relu(m.layers[1](torch.relu(m.layers[0](x))))
    assert torch.allclose(out, m.layers[2](h), atol=1e-6)
    assert (out < 0).any()
    one = candidate(2, 7, 2, n_layers=1)
    assert len(one.layers) == 1 and one.layers[0].in_features == 2
    assert sum(p.numel() for p in candidate(4, 6, 2).parameters()) == 4 * 6 + 6 + 6 * 2 + 2
'''),

task("JBC1/pt/038", "modules", "Residual", r'''
import torch.nn as nn


class Residual(nn.Module):
    """Wrap a module fn so that forward(x) returns x + fn(x). fn must be
    registered as a submodule named `fn`.
    """
''', r'''
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)
''', r'''
def check(candidate):
    import torch
    torch.manual_seed(0)
    lin = nn.Linear(4, 4)
    r = candidate(lin)
    x = torch.randn(3, 4)
    assert torch.allclose(r(x), x + lin(x))
    assert len(list(r.parameters())) == 2
    assert "fn.weight" in r.state_dict()
'''),

task("JBC1/pt/039", "modules", "LayerNorm", r'''
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer normalization over the last dimension, implemented without
    nn.LayerNorm or F.layer_norm. Learnable parameters `weight` (init ones) and
    `bias` (init zeros) have shape (dim,). Uses population variance and eps.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
''', r'''
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias
''', r'''
def check(candidate):
    import torch.nn.functional as F
    torch.manual_seed(0)
    ln = candidate(6)
    assert isinstance(ln.weight, nn.Parameter) and ln.weight.shape == (6,)
    assert torch.equal(ln.bias.data, torch.zeros(6))
    x = torch.randn(2, 3, 6) * 4 + 1
    ref = F.layer_norm(x, (6,), ln.weight, ln.bias, 1e-5)
    assert torch.allclose(ln(x), ref, atol=1e-5)
    with torch.no_grad():
        ln.weight.fill_(2.0)
        ln.bias.fill_(0.5)
    assert torch.allclose(ln(x), F.layer_norm(x, (6,), ln.weight, ln.bias, 1e-5), atol=1e-5)
    ln2 = candidate(6, eps=0.1)
    assert torch.allclose(ln2(x), F.layer_norm(x, (6,), eps=0.1), atol=1e-5)
'''),

task("JBC1/pt/040", "modules", "CausalSelfAttention", r'''
import math

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention for inputs of shape (B, T, d_model).

    Use one nn.Linear(d_model, 3 * d_model) named `qkv` whose output splits
    into q, k, v in that order along the last dim, and an output projection
    nn.Linear(d_model, d_model) named `proj`. Each head uses d_model // n_heads
    channels, scores are scaled by 1/sqrt(head_dim), and position i attends
    only to positions <= i.
    """

    def __init__(self, d_model: int, n_heads: int):
''', r'''
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        b, t, c = x.shape
        hd = c // self.n_heads
        q, k, v = self.qkv(x).split(c, dim=-1)
        q, k, v = (z.view(b, t, self.n_heads, hd).transpose(1, 2) for z in (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(hd)
        mask = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
        att = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(b, t, c)
        return self.proj(y)
''', r'''
def check(candidate):
    import torch.nn.functional as F
    torch.manual_seed(0)
    m = candidate(16, 4)
    assert isinstance(m.qkv, nn.Linear) and m.qkv.out_features == 48
    assert isinstance(m.proj, nn.Linear) and m.proj.in_features == 16
    x = torch.randn(2, 5, 16)
    out = m(x)
    assert out.shape == (2, 5, 16)
    q, k, v = m.qkv(x).split(16, dim=-1)
    q, k, v = (z.view(2, 5, 4, 4).transpose(1, 2) for z in (q, k, v))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = m.proj(ref.transpose(1, 2).reshape(2, 5, 16))
    assert torch.allclose(out, ref, atol=1e-5)
    x2 = x.clone()
    x2[:, 3:] = torch.randn(2, 2, 16)
    assert torch.allclose(m(x2)[:, :3], out[:, :3], atol=1e-6)
'''),

task("JBC1/pt/041", "modules", "LearnedPositions", r'''
import torch
import torch.nn as nn


class LearnedPositions(nn.Module):
    """Add learned position embeddings to x of shape (B, T, dim).

    Store an nn.Embedding(max_len, dim) named `emb`. forward(x) returns
    x + emb(positions 0..T-1). Raise ValueError if T > max_len.
    """

    def __init__(self, max_len: int, dim: int):
''', r'''
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(max_len, dim)

    def forward(self, x):
        t = x.shape[1]
        if t > self.max_len:
            raise ValueError(f"sequence length {t} exceeds max_len {self.max_len}")
        pos = torch.arange(t, device=x.device)
        return x + self.emb(pos)
''', r'''
def check(candidate):
    torch.manual_seed(0)
    m = candidate(8, 4)
    assert isinstance(m.emb, nn.Embedding) and m.emb.weight.shape == (8, 4)
    x = torch.randn(2, 5, 4)
    out = m(x)
    assert torch.allclose(out, x + m.emb.weight[:5])
    assert m(torch.zeros(1, 8, 4)).shape == (1, 8, 4)
    try:
        m(torch.zeros(1, 9, 4))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
'''),

task("JBC1/pt/042", "modules", "TinyConvNet", r'''
import torch
import torch.nn as nn


class TinyConvNet(nn.Module):
    """A small classifier for single-channel images of any size >= 2 x 2.

    self.net is an nn.Sequential of exactly: Conv2d(1, 8, 3, padding=1),
    ReLU, MaxPool2d(2), Conv2d(8, 16, 3, padding=1), ReLU,
    AdaptiveAvgPool2d(1), Flatten, Linear(16, num_classes).
    forward returns logits of shape (B, num_classes).
    """

    def __init__(self, num_classes: int):
''', r'''
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, num_classes),
        )

    def forward(self, x):
        return self.net(x)
''', r'''
def check(candidate):
    torch.manual_seed(0)
    m = candidate(10)
    assert m(torch.randn(4, 1, 28, 28)).shape == (4, 10)
    assert m(torch.randn(2, 1, 33, 17)).shape == (2, 10)
    expected = (1 * 8 * 9 + 8) + (8 * 16 * 9 + 16) + (16 * 10 + 10)
    assert sum(p.numel() for p in m.parameters()) == expected
    kinds = [type(layer).__name__ for layer in m.net]
    assert kinds == ["Conv2d", "ReLU", "MaxPool2d", "Conv2d", "ReLU",
                     "AdaptiveAvgPool2d", "Flatten", "Linear"]
'''),

# --- losses --------------------------------------------------------------------
task("JBC1/pt/043", "losses", "cross_entropy", r'''
import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy between logits (N, C) and int64 class targets (N,),
    implemented without torch.nn.functional. Must be numerically stable for
    very large logits.
    """
''', r'''
    shifted = logits - logits.max(dim=1, keepdim=True).values
    log_probs = shifted - shifted.exp().sum(dim=1, keepdim=True).log()
    return -log_probs.gather(1, targets.unsqueeze(1)).mean()
''', r'''
def check(candidate):
    import torch.nn.functional as F
    torch.manual_seed(0)
    logits, y = torch.randn(8, 5), torch.randint(0, 5, (8,))
    assert torch.allclose(candidate(logits, y), F.cross_entropy(logits, y), atol=1e-6)
    big = torch.tensor([[1000.0, 0.0], [0.0, 1000.0]])
    out = candidate(big, torch.tensor([1, 1]))
    assert torch.isfinite(out) and torch.allclose(out, torch.tensor(500.0))
    w = torch.randn(3, 4, requires_grad=True)
    candidate(w, torch.tensor([0, 1, 2])).backward()
    assert w.grad is not None and w.grad.shape == (3, 4)
'''),

task("JBC1/pt/044", "losses", "masked_token_loss", r'''
import torch
import torch.nn.functional as F


def masked_token_loss(logits, targets, ignore_index: int = -100):
    """Language-model loss: logits (B, T, V), targets (B, T). Return the mean
    cross-entropy over positions whose target != ignore_index, or a zero
    tensor if every position is ignored.
    """
''', r'''
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    keep = flat_targets != ignore_index
    if not keep.any():
        return logits.sum() * 0.0
    return F.cross_entropy(flat_logits[keep], flat_targets[keep])
''', r'''
def check(candidate):
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 7)
    t = torch.tensor([[1, -100, 3], [-100, -100, 6]])
    keep = t != -100
    ref = F.cross_entropy(logits[keep], t[keep])
    assert torch.allclose(candidate(logits, t), ref, atol=1e-6)
    t2 = torch.tensor([[1, 0, 3], [0, 0, 6]])
    assert torch.allclose(candidate(logits, t2, ignore_index=0), F.cross_entropy(
        logits[t2 != 0], t2[t2 != 0]), atol=1e-6)
    all_ignored = torch.full((2, 3), -100)
    z = candidate(logits, all_ignored)
    assert torch.isfinite(z) and z.item() == 0.0
'''),

task("JBC1/pt/045", "losses", "bce_with_logits", r'''
import torch


def bce_with_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean binary cross-entropy on raw logits, implemented without
    torch.nn.functional, stable for large positive and negative logits.
    """
''', r'''
    return (logits.clamp(min=0) - logits * targets + torch.log1p(torch.exp(-logits.abs()))).mean()
''', r'''
def check(candidate):
    import torch.nn.functional as F
    torch.manual_seed(0)
    x, y = torch.randn(20), torch.randint(0, 2, (20,)).float()
    assert torch.allclose(candidate(x, y), F.binary_cross_entropy_with_logits(x, y), atol=1e-6)
    ext = torch.tensor([200.0, -200.0, 200.0])
    tgt = torch.tensor([1.0, 0.0, 0.0])
    out = candidate(ext, tgt)
    assert torch.isfinite(out)
    assert torch.allclose(out, F.binary_cross_entropy_with_logits(ext, tgt), atol=1e-4)
    soft = torch.tensor([0.3, 0.9])
    z = torch.tensor([0.5, -1.0])
    assert torch.allclose(candidate(z, soft), F.binary_cross_entropy_with_logits(z, soft), atol=1e-6)
'''),

task("JBC1/pt/046", "losses", "label_smoothing_ce", r'''
import torch


def label_smoothing_ce(logits: torch.Tensor, targets: torch.Tensor, eps: float) -> torch.Tensor:
    """Mean cross-entropy with label smoothing: the target distribution puts
    1 - eps on the true class plus eps / C spread uniformly over all C classes.
    logits: (N, C); targets: int64 (N,).
    """
''', r'''
    log_probs = logits.log_softmax(dim=-1)
    nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    uniform = -log_probs.mean(dim=-1)
    return ((1 - eps) * nll + eps * uniform).mean()
''', r'''
def check(candidate):
    import torch.nn.functional as F
    torch.manual_seed(0)
    x, y = torch.randn(6, 4), torch.randint(0, 4, (6,))
    for eps in (0.0, 0.1, 0.5):
        ref = F.cross_entropy(x, y, label_smoothing=eps)
        assert torch.allclose(candidate(x, y, eps), ref, atol=1e-6)
'''),

task("JBC1/pt/047", "losses", "mse_per_sample", r'''
import torch


def mse_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error for each sample: inputs have shape (B, ...) and the
    result has shape (B,), averaging over all non-batch dimensions.
    """
''', r'''
    return ((pred - target) ** 2).reshape(pred.shape[0], -1).mean(dim=1)
''', r'''
def check(candidate):
    p = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    t = torch.tensor([[1.0, 4.0], [3.0, 1.0]])
    assert candidate(p, t).tolist() == [2.0, 5.0]
    torch.manual_seed(0)
    a, b = torch.randn(3, 2, 4, 5), torch.randn(3, 2, 4, 5)
    out = candidate(a, b)
    assert out.shape == (3,)
    assert torch.allclose(out[1], ((a[1] - b[1]) ** 2).mean())
    assert candidate(torch.ones(4), torch.zeros(4)).tolist() == [1.0] * 4
'''),

# --- optimizers ----------------------------------------------------------------
task("JBC1/pt/048", "optimizers", "sgd_step", r'''
import torch


def sgd_step(params, lr: float) -> None:
    """Perform one plain SGD update in place: p <- p - lr * p.grad for every
    parameter with a gradient. Must not be tracked by autograd.
    """
''', r'''
    with torch.no_grad():
        for p in params:
            if p.grad is not None:
                p.sub_(lr * p.grad)
''', r'''
def check(candidate):
    w = torch.tensor([1.0, 2.0], requires_grad=True)
    b = torch.tensor([0.5], requires_grad=True)
    (w ** 2).sum().backward()
    candidate([w, b], 0.1)
    assert torch.allclose(w, torch.tensor([0.8, 1.6]))
    assert b.item() == 0.5
    assert w.requires_grad and w.is_leaf and w.grad_fn is None
'''),

task("JBC1/pt/049", "optimizers", "make_adamw", r'''
import torch
import torch.nn as nn


def make_adamw(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """Create AdamW with two parameter groups: first, parameters with 2 or more
    dimensions (weight_decay applied); second, all other parameters (weight
    decay 0.0). Skip parameters that do not require gradients.
    """
''', r'''
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )
''', r'''
def check(candidate):
    m = nn.Sequential(nn.Embedding(10, 4), nn.LayerNorm(4), nn.Linear(4, 3))
    m[1].bias.requires_grad_(False)
    opt = candidate(m, 3e-4, 0.1)
    assert isinstance(opt, torch.optim.AdamW) and len(opt.param_groups) == 2
    g0, g1 = opt.param_groups
    assert g0["weight_decay"] == 0.1 and g1["weight_decay"] == 0.0
    assert {id(p) for p in g0["params"]} == {id(m[0].weight), id(m[2].weight)}
    assert {id(p) for p in g1["params"]} == {id(m[1].weight), id(m[2].bias)}
    assert g0["lr"] == 3e-4 and g1["lr"] == 3e-4
'''),

task("JBC1/pt/050", "optimizers", "warmup_cosine_lr", r'''
import math


def warmup_cosine_lr(step: int, warmup: int, total: int, base_lr: float,
                     min_lr: float = 0.0) -> float:
    """Learning rate at `step` (0-based): linear warmup from base_lr/warmup at
    step 0 to base_lr at step warmup-1, then cosine decay from base_lr at step
    warmup to min_lr at step total. Steps >= total return min_lr.
    """
''', r'''
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
''', r'''
def check(candidate):
    close = lambda a, b: abs(a - b) < 1e-9
    assert close(candidate(0, 10, 110, 1.0), 0.1)
    assert close(candidate(9, 10, 110, 1.0), 1.0)
    assert close(candidate(10, 10, 110, 1.0), 1.0)
    assert close(candidate(60, 10, 110, 1.0), 0.5)
    assert close(candidate(60, 10, 110, 1.0, min_lr=0.2), 0.6)
    assert close(candidate(110, 10, 110, 1.0, 0.1), 0.1)
    assert close(candidate(500, 10, 110, 1.0, 0.1), 0.1)
    assert close(candidate(0, 0, 100, 2.0), 2.0)
'''),

# --- data ----------------------------------------------------------------------
task("JBC1/pt/051", "data", "NextTokenDataset", r'''
import torch
from torch.utils.data import Dataset


class NextTokenDataset(Dataset):
    """Language-modelling windows over a 1D int64 tensor of token ids.

    Item i is (x, y) where x = tokens[i : i + block_size] and y is x shifted
    left by one: tokens[i + 1 : i + block_size + 1]. There are
    len(tokens) - block_size items (never negative).
    """

    def __init__(self, tokens: torch.Tensor, block_size: int):
''', r'''
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.tokens) - self.block_size)

    def __getitem__(self, i):
        if not 0 <= i < len(self):
            raise IndexError(i)
        x = self.tokens[i:i + self.block_size]
        y = self.tokens[i + 1:i + self.block_size + 1]
        return x, y
''', r'''
def check(candidate):
    from torch.utils.data import DataLoader
    ds = candidate(torch.arange(10), 4)
    assert len(ds) == 6
    x, y = ds[0]
    assert x.tolist() == [0, 1, 2, 3] and y.tolist() == [1, 2, 3, 4]
    x, y = ds[5]
    assert x.tolist() == [5, 6, 7, 8] and y.tolist() == [6, 7, 8, 9]
    assert len(candidate(torch.arange(3), 4)) == 0
    xb, yb = next(iter(DataLoader(ds, batch_size=3)))
    assert xb.shape == (3, 4) and torch.equal(yb[:, :-1], xb[:, 1:])
'''),

task("JBC1/pt/052", "data", "make_loader", r'''
import torch
from torch.utils.data import DataLoader, TensorDataset


def make_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool,
                seed: int) -> DataLoader:
    """Wrap (x, y) in a TensorDataset and DataLoader. When shuffle is True, the
    order must be reproducible from seed alone (use a seeded torch.Generator).
    Keep the final partial batch.
    """
''', r'''
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle,
                      generator=generator, drop_last=False)
''', r'''
def check(candidate):
    x = torch.arange(10).float().unsqueeze(1)
    y = torch.arange(10)
    batches = list(candidate(x, y, 4, False, 0))
    assert [len(b[1]) for b in batches] == [4, 4, 2]
    assert batches[0][1].tolist() == [0, 1, 2, 3]
    torch.manual_seed(123)
    a = [b[1].tolist() for b in candidate(x, y, 3, True, 7)]
    torch.manual_seed(999)
    b = [b[1].tolist() for b in candidate(x, y, 3, True, 7)]
    assert a == b
    c = [b[1].tolist() for b in candidate(x, y, 3, True, 8)]
    assert a != c
    assert sorted(sum(a, [])) == list(range(10))
'''),

task("JBC1/pt/053", "data", "collate_padded", r'''
import torch


def collate_padded(batch):
    """Collate a list of (token_ids, label) pairs, where token_ids is a 1D
    int64 tensor and label an int, into (tokens, lengths, labels): tokens is
    (B, max_len) right-padded with 0, lengths and labels are int64 (B,).
    """
''', r'''
    lengths = torch.tensor([len(ids) for ids, _ in batch], dtype=torch.int64)
    tokens = torch.zeros(len(batch), int(lengths.max()), dtype=torch.int64)
    for i, (ids, _) in enumerate(batch):
        tokens[i, :len(ids)] = ids
    labels = torch.tensor([label for _, label in batch], dtype=torch.int64)
    return tokens, lengths, labels
''', r'''
def check(candidate):
    from torch.utils.data import DataLoader
    data = [(torch.tensor([5, 6, 7]), 1), (torch.tensor([8]), 0)]
    tokens, lengths, labels = candidate(data)
    assert tokens.tolist() == [[5, 6, 7], [8, 0, 0]]
    assert lengths.tolist() == [3, 1] and labels.tolist() == [1, 0]
    assert tokens.dtype == lengths.dtype == labels.dtype == torch.int64
    loader = DataLoader(data * 3, batch_size=4, collate_fn=candidate)
    shapes = [b[0].shape for b in loader]
    assert shapes == [(4, 3), (2, 3)]
'''),

# --- training / evaluation loops -----------------------------------------------
task("JBC1/pt/054", "training", "train_one_epoch", r'''
import torch


def train_one_epoch(model, loader, optimizer, loss_fn) -> float:
    """Train model for one pass over loader, which yields (inputs, targets).
    Put the model in training mode, and for each batch zero the gradients,
    compute loss_fn(model(inputs), targets), backpropagate and step.
    Return the average loss per sample (weight each batch by its size).
    """
''', r'''
    model.train()
    total, count = 0.0, 0
    for inputs, targets in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(inputs)
        count += len(inputs)
    return total / count
''', r'''
def check(candidate):
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(0)
    x = torch.randn(10, 3)
    y = x @ torch.tensor([[1.0], [-2.0], [0.5]])
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    model = nn.Linear(3, 1)
    frozen = torch.optim.SGD(model.parameters(), lr=0.0)
    model.eval()
    expected = sum(nn.functional.mse_loss(model(a), b).item() * len(a) for a, b in loader) / 10
    got = candidate(model, loader, frozen, nn.functional.mse_loss)
    assert isinstance(got, float) and abs(got - expected) < 1e-5
    assert model.training
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    first = candidate(model, loader, opt, nn.functional.mse_loss)
    for _ in range(20):
        last = candidate(model, loader, opt, nn.functional.mse_loss)
    assert last < first * 0.5
    torch.manual_seed(1)
    a, b = nn.Linear(3, 1), nn.Linear(3, 1)
    b.load_state_dict(a.state_dict())
    opt_b = torch.optim.SGD(b.parameters(), lr=0.05)
    for xb, yb in loader:
        opt_b.zero_grad()
        nn.functional.mse_loss(b(xb), yb).backward()
        opt_b.step()
    candidate(a, loader, torch.optim.SGD(a.parameters(), lr=0.05), nn.functional.mse_loss)
    assert torch.allclose(a.weight, b.weight, atol=1e-6)
'''),

task("JBC1/pt/055", "training", "evaluate", r'''
import torch


def evaluate(model, loader, loss_fn):
    """Evaluate a classifier on loader, which yields (inputs, int64 targets).
    Run in eval mode without building autograd graphs, and restore the
    model's previous train/eval mode afterwards.
    Return (mean loss per sample, accuracy) as Python floats.
    """
''', r'''
    was_training = model.training
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            logits = model(inputs)
            total_loss += loss_fn(logits, targets).item() * len(targets)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            count += len(targets)
    model.train(was_training)
    return total_loss / count, correct / count
''', r'''
def check(candidate):
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(0)
    x, y = torch.randn(9, 4), torch.randint(0, 3, (9,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    model = nn.Sequential(nn.Linear(4, 16), nn.Dropout(0.5), nn.Linear(16, 3))
    model.train()
    loss, acc = candidate(model, loader, F.cross_entropy)
    assert model.training
    model.eval()
    with torch.no_grad():
        logits = model(x)
    assert abs(loss - F.cross_entropy(logits, y).item()) < 1e-5
    assert abs(acc - (logits.argmax(1) == y).float().mean().item()) < 1e-6
    assert all(p.grad is None for p in model.parameters())
    model.eval()
    candidate(model, loader, F.cross_entropy)
    assert not model.training
    assert isinstance(loss, float) and isinstance(acc, float)
'''),

task("JBC1/pt/056", "training", "fit_linear", r'''
import torch
import torch.nn as nn


def fit_linear(x: torch.Tensor, y: torch.Tensor, steps: int, lr: float) -> nn.Linear:
    """Fit nn.Linear(x.shape[1], y.shape[1]) to (x, y) by full-batch gradient
    descent on mean squared error with torch.optim.SGD. Initialize the weight
    and bias to zero so the result is deterministic. Return the model.
    """
''', r'''
    model = nn.Linear(x.shape[1], y.shape[1])
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.mse_loss(model(x), y).backward()
        opt.step()
    return model
''', r'''
def check(candidate):
    torch.manual_seed(0)
    x = torch.randn(64, 2)
    w_true = torch.tensor([[2.0, -1.0]])
    y = x @ w_true.T + 0.5
    m = candidate(x, y, 500, 0.1)
    assert isinstance(m, nn.Linear) and m.weight.shape == (1, 2)
    assert torch.allclose(m.weight, w_true, atol=1e-3)
    assert torch.allclose(m.bias, torch.tensor([0.5]), atol=1e-3)
    z = candidate(x, y, 0, 0.1)
    assert torch.equal(z.weight, torch.zeros(1, 2)) and torch.equal(z.bias, torch.zeros(1))
    a = candidate(x, y, 3, 0.1)
    b = candidate(x, y, 3, 0.1)
    assert torch.equal(a.weight, b.weight)
'''),

task("JBC1/pt/057", "training", "ema_update", r'''
import torch
import torch.nn as nn


def ema_update(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """Update ema_model's parameters in place as an exponential moving average
    of model's: ema = decay * ema + (1 - decay) * current. Buffers are copied
    directly. No autograd tracking. Both models share the same architecture.
    """
''', r'''
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(p, alpha=1 - decay)
        for ema_b, b in zip(ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)
''', r'''
def check(candidate):
    torch.manual_seed(0)
    make = lambda: nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
    model, ema = make(), make()
    with torch.no_grad():
        for p in ema.parameters():
            p.zero_()
    model.train()
    model(torch.randn(8, 3))
    before = [p.clone() for p in model.parameters()]
    candidate(ema, model, 0.9)
    for e, p in zip(ema.parameters(), before):
        assert torch.allclose(e, 0.1 * p, atol=1e-6)
        assert e.grad_fn is None
    assert torch.equal(ema[1].running_mean, model[1].running_mean)
    assert ema[1].num_batches_tracked.item() == 1
    candidate(ema, model, 0.0)
    for e, p in zip(ema.parameters(), model.parameters()):
        assert torch.allclose(e, p)
'''),

# --- saving / loading ----------------------------------------------------------
task("JBC1/pt/058", "checkpoints", "load_checkpoint", r'''
import torch


def save_checkpoint(path, model, optimizer, step: int) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step}, path)


def load_checkpoint(path, model, optimizer) -> int:
    """Restore model and optimizer state from a file written by
    save_checkpoint (load it on the CPU), and return the saved step.
    """
''', r'''
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return state["step"]
''', r'''
def check(candidate):
    import os
    import tempfile
    import torch.nn as nn
    torch.manual_seed(0)
    m = nn.Linear(4, 2)
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    m(torch.randn(3, 4)).sum().backward()
    opt.step()
    path = os.path.join(tempfile.mkdtemp(), "ckpt.pt")
    save_checkpoint(path, m, opt, 42)
    m2 = nn.Linear(4, 2)
    opt2 = torch.optim.Adam(m2.parameters(), lr=0.5)
    step = candidate(path, m2, opt2)
    assert step == 42
    assert torch.equal(m2.weight, m.weight) and torch.equal(m2.bias, m.bias)
    assert opt2.param_groups[0]["lr"] == 0.01
    assert opt2.state_dict()["state"][0]["step"] == opt.state_dict()["state"][0]["step"]
'''),

task("JBC1/pt/059", "checkpoints", "average_state_dicts", r'''
import torch


def average_state_dicts(state_dicts: list[dict]) -> dict:
    """Average a non-empty list of state dicts with identical keys. Floating
    point tensors are averaged elementwise; any other tensor (for example
    integer counters) is taken from the first state dict. Inputs are not
    modified.
    """
''', r'''
    out = {}
    for key, first in state_dicts[0].items():
        if torch.is_floating_point(first):
            out[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)
        else:
            out[key] = first.clone()
    return out
''', r'''
def check(candidate):
    a = {"w": torch.tensor([1.0, 2.0]), "n": torch.tensor(5)}
    b = {"w": torch.tensor([3.0, 6.0]), "n": torch.tensor(9)}
    c = {"w": torch.tensor([2.0, 1.0]), "n": torch.tensor(1)}
    out = candidate([a, b, c])
    assert torch.allclose(out["w"], torch.tensor([2.0, 3.0]))
    assert out["n"].item() == 5 and out["n"].dtype == torch.int64
    assert a["w"].tolist() == [1.0, 2.0]
    single = candidate([b])
    assert torch.equal(single["w"], b["w"])
    import torch.nn as nn
    m = nn.BatchNorm1d(2)
    avg = candidate([m.state_dict(), m.state_dict()])
    nn.BatchNorm1d(2).load_state_dict(avg)
'''),

task("JBC1/pt/060", "checkpoints", "strip_prefix", r'''
def strip_prefix(state_dict: dict, prefix: str = "module.") -> dict:
    """Return a new state dict in which keys starting with prefix (for example
    the "module." added by DataParallel) have it removed once. Other keys are
    kept unchanged. The input dict is not modified.
    """
''', r'''
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items()}
''', r'''
def check(candidate):
    import torch
    import torch.nn as nn
    sd = {"module.fc.weight": 1, "module.module.x": 2, "head.bias": 3}
    out = candidate(sd)
    assert out == {"fc.weight": 1, "module.x": 2, "head.bias": 3}
    assert "module.fc.weight" in sd
    assert candidate({"_orig_mod.w": 1}, "_orig_mod.") == {"w": 1}
    m = nn.Linear(2, 2)
    wrapped = {"module." + k: v for k, v in m.state_dict().items()}
    nn.Linear(2, 2).load_state_dict(candidate(wrapped))
    assert candidate({}) == {}
'''),
]

for _t in TASKS:
    _t["requires"] = list(T)
