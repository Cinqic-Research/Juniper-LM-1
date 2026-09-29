from _schema import task

T = ("torch",)

TASKS = [
# --- tensor construction -------------------------------------------------------
task("JBC1/pt/001", "construction", "make_grid", r'''
import torch


def make_grid(height: int, width: int) -> torch.Tensor:
    """Return an int64 tensor of shape (height, width, 2) whose entry [i, j] is
    the coordinate pair (i, j).
    """
''', r'''
    rows = torch.arange(height).view(height, 1).expand(height, width)
    cols = torch.arange(width).view(1, width).expand(height, width)
    return torch.stack([rows, cols], dim=-1)
''', r'''
def check(candidate):
    g = candidate(3, 4)
    assert g.shape == (3, 4, 2) and g.dtype == torch.int64
    assert g[2, 1].tolist() == [2, 1]
    assert g[0, 3].tolist() == [0, 3]
    assert candidate(1, 1).tolist() == [[[0, 0]]]
    assert candidate(0, 5).shape == (0, 5, 2)
'''),

task("JBC1/pt/002", "construction", "one_hot", r'''
import torch


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert an int64 tensor of class indices with any shape into a float32
    one-hot tensor with an extra trailing dimension of size num_classes.
    """
''', r'''
    out = torch.zeros(*labels.shape, num_classes, dtype=torch.float32, device=labels.device)
    return out.scatter_(-1, labels.unsqueeze(-1), 1.0)
''', r'''
def check(candidate):
    y = candidate(torch.tensor([0, 2, 1]), 3)
    assert y.dtype == torch.float32 and y.shape == (3, 3)
    assert torch.equal(y, torch.tensor([[1., 0, 0], [0, 0, 1], [0, 1, 0]]))
    z = candidate(torch.tensor([[1, 0], [3, 3]]), 4)
    assert z.shape == (2, 2, 4) and z.sum().item() == 4 and z[1, 0, 3] == 1
    assert candidate(torch.tensor([], dtype=torch.int64), 5).shape == (0, 5)
'''),

task("JBC1/pt/003", "construction", "identity_batch", r'''
import torch


def identity_batch(n: int, d: int) -> torch.Tensor:
    """Return a float32 tensor of shape (n, d, d) holding n independent copies
    of the d x d identity matrix (modifying one copy must not change others).
    """
''', r'''
    return torch.eye(d, dtype=torch.float32).repeat(n, 1, 1)
''', r'''
def check(candidate):
    x = candidate(3, 2)
    assert x.shape == (3, 2, 2) and x.dtype == torch.float32
    assert torch.equal(x[1], torch.eye(2))
    x[0, 0, 0] = 5
    assert x[1, 0, 0] == 1 and x[2, 0, 0] == 1
    assert candidate(0, 4).shape == (0, 4, 4)
'''),

task("JBC1/pt/004", "construction", "causal_mask", r'''
import torch


def causal_mask(t: int) -> torch.Tensor:
    """Return a bool tensor of shape (t, t) that is True where position j is in
    the future of position i (j > i), i.e. where attention must be blocked.
    """
''', r'''
    return torch.ones(t, t, dtype=torch.bool).triu(diagonal=1)
''', r'''
def check(candidate):
    m = candidate(3)
    assert m.dtype == torch.bool
    assert m.tolist() == [[False, True, True], [False, False, True], [False, False, False]]
    assert candidate(1).tolist() == [[False]]
    assert candidate(6).sum().item() == 15
'''),

# --- indexing ------------------------------------------------------------------
task("JBC1/pt/005", "indexing", "gather_rows", r'''
import torch


def gather_rows(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Given x of shape (N, D) and int64 indices idx of shape (B,), return the
    selected rows, shape (B, D). Indices may repeat.
    """
''', r'''
    return x[idx]
''', r'''
def check(candidate):
    x = torch.arange(12.).view(4, 3)
    out = candidate(x, torch.tensor([3, 0, 3]))
    assert out.shape == (3, 3)
    assert out.tolist() == [[9., 10., 11.], [0., 1., 2.], [9., 10., 11.]]
    assert candidate(x, torch.tensor([], dtype=torch.int64)).shape == (0, 3)
'''),

task("JBC1/pt/006", "indexing", "select_per_row", r'''
import torch


def select_per_row(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Given x of shape (N, C) and int64 idx of shape (N,), return the tensor of
    shape (N,) whose i-th entry is x[i, idx[i]].
    """
''', r'''
    return x.gather(1, idx.unsqueeze(1)).squeeze(1)
''', r'''
def check(candidate):
    x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    assert candidate(x, torch.tensor([2, 0])).tolist() == [3., 4.]
    torch.manual_seed(0)
    y = torch.randn(50, 7)
    i = torch.randint(0, 7, (50,))
    assert torch.equal(candidate(y, i), y[torch.arange(50), i])
    assert candidate(y, i).shape == (50,)
'''),

task("JBC1/pt/007", "indexing", "masked_mean", r'''
import torch


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average x of shape (B, T) over dim 1, counting only positions where the
    bool mask (B, T) is True. Rows with no True entries give 0.0. Returns (B,).
    """
''', r'''
    m = mask.to(x.dtype)
    total = (x * m).sum(dim=1)
    count = m.sum(dim=1)
    return torch.where(count > 0, total / count.clamp(min=1), torch.zeros_like(total))
''', r'''
def check(candidate):
    x = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    m = torch.tensor([[True, False, True], [False, False, False], [True, True, True]])
    out = candidate(x, m)
    assert out.shape == (3,)
    assert torch.allclose(out, torch.tensor([2., 0., 8.]))
    assert not torch.isnan(out).any()
    x2 = torch.tensor([[2.0, 1.0, 5.0]])
    assert torch.allclose(candidate(x2, torch.tensor([[False, True, False]])), torch.tensor([1.0]))
'''),

task("JBC1/pt/008", "indexing", "topk_positions", r'''
import torch


def topk_positions(x: torch.Tensor, k: int) -> torch.Tensor:
    """Return the (row, col) positions of the k largest values of a 2D tensor as
    an int64 tensor of shape (k, 2), ordered from largest value to smallest.
    Values are distinct.
    """
''', r'''
    flat = x.flatten().topk(k).indices
    return torch.stack([flat // x.shape[1], flat % x.shape[1]], dim=1)
''', r'''
def check(candidate):
    x = torch.tensor([[1., 9., 3.], [7., 2., 8.]])
    out = candidate(x, 3)
    assert out.dtype == torch.int64 and out.shape == (3, 2)
    assert out.tolist() == [[0, 1], [1, 2], [1, 0]]
    assert candidate(x, 1).tolist() == [[0, 1]]
    torch.manual_seed(1)
    y = torch.randperm(20).float().view(4, 5)
    pos = candidate(y, 20)
    vals = y[pos[:, 0], pos[:, 1]]
    assert torch.equal(vals, torch.arange(19, -1, -1).float())
'''),

task("JBC1/pt/009", "indexing", "replace_nan", r'''
import torch


def replace_nan(x: torch.Tensor, value: float) -> torch.Tensor:
    """Return a copy of x with every NaN replaced by value. Infinities are left
    unchanged and the input tensor must not be modified.
    """
''', r'''
    return torch.where(torch.isnan(x), torch.full_like(x, value), x)
''', r'''
def check(candidate):
    nan, inf = float("nan"), float("inf")
    x = torch.tensor([1.0, nan, inf, -inf, nan])
    out = candidate(x, 0.5)
    assert out.tolist() == [1.0, 0.5, inf, -inf, 0.5]
    assert torch.isnan(x[1]) and torch.isnan(x[4])
    assert candidate(torch.zeros(2, 2), 3.0).tolist() == [[0.0, 0.0], [0.0, 0.0]]
'''),

# --- broadcasting --------------------------------------------------------------
task("JBC1/pt/010", "broadcasting", "pairwise_sq_dists", r'''
import torch


def pairwise_sq_dists(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Given a of shape (N, D) and b of shape (M, D), return the (N, M) tensor
    of squared Euclidean distances between every row of a and every row of b.
    """
''', r'''
    diff = a.unsqueeze(1) - b.unsqueeze(0)
    return (diff ** 2).sum(dim=-1)
''', r'''
def check(candidate):
    a = torch.tensor([[0., 0.], [1., 1.]])
    b = torch.tensor([[1., 0.], [3., 4.], [0., 0.]])
    out = candidate(a, b)
    assert out.shape == (2, 3)
    assert torch.allclose(out, torch.tensor([[1., 25., 0.], [1., 13., 2.]]))
    torch.manual_seed(0)
    x, y = torch.randn(6, 5), torch.randn(4, 5)
    assert torch.allclose(candidate(x, y), torch.cdist(x, y) ** 2, atol=1e-4)
'''),

task("JBC1/pt/011", "broadcasting", "normalize_rows", r'''
import torch


def normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Scale each row of the 2D tensor x to unit L2 norm, dividing by
    max(norm, eps) so that all-zero rows stay zero.
    """
''', r'''
    norms = x.norm(dim=1, keepdim=True).clamp(min=eps)
    return x / norms
''', r'''
def check(candidate):
    x = torch.tensor([[3., 4.], [0., 0.], [0., -2.]])
    out = candidate(x)
    assert torch.allclose(out, torch.tensor([[0.6, 0.8], [0., 0.], [0., -1.]]))
    torch.manual_seed(0)
    y = torch.randn(10, 6)
    assert torch.allclose(candidate(y).norm(dim=1), torch.ones(10), atol=1e-6)
'''),

task("JBC1/pt/012", "broadcasting", "standardize", r'''
import torch


def standardize(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Standardize x along dim: subtract the mean and divide by the population
    standard deviation (unbiased=False) plus 1e-5.
    """
''', r'''
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, unbiased=False, keepdim=True)
    return (x - mean) / (std + 1e-5)
''', r'''
def check(candidate):
    torch.manual_seed(0)
    x = torch.randn(4, 8) * 3 + 2
    y = candidate(x, 1)
    assert y.shape == x.shape
    assert torch.allclose(y.mean(dim=1), torch.zeros(4), atol=1e-5)
    assert torch.allclose(y.std(dim=1, unbiased=False), torch.ones(4), atol=1e-4)
    z = candidate(x, 0)
    assert torch.allclose(z.mean(dim=0), torch.zeros(8), atol=1e-5)
    c = candidate(torch.full((2, 3), 7.0), 1)
    assert torch.allclose(c, torch.zeros(2, 3))
'''),

task("JBC1/pt/013", "broadcasting", "outer_add", r'''
import torch


def outer_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Given 1D tensors a (N,) and b (M,), return the (N, M) tensor with entry
    [i, j] = a[i] + b[j], without Python loops.
    """
''', r'''
    return a[:, None] + b[None, :]
''', r'''
def check(candidate):
    out = candidate(torch.tensor([1., 2.]), torch.tensor([10., 20., 30.]))
    assert out.shape == (2, 3)
    assert out.tolist() == [[11., 21., 31.], [12., 22., 32.]]
    assert candidate(torch.tensor([5]), torch.tensor([1])).tolist() == [[6]]
'''),

task("JBC1/pt/014", "broadcasting", "scale_channels", r'''
import torch


def scale_channels(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Multiply each channel of an image batch x (B, C, H, W) by the matching
    entry of scale (C,).
    """
''', r'''
    return x * scale.view(1, -1, 1, 1)
''', r'''
def check(candidate):
    x = torch.ones(2, 3, 4, 5)
    s = torch.tensor([1., 2., 3.])
    out = candidate(x, s)
    assert out.shape == (2, 3, 4, 5)
    assert out[1, 2].unique().tolist() == [3.0] and out[0, 1].unique().tolist() == [2.0]
    y = torch.ones(1, 4, 4, 4)
    assert candidate(y, torch.arange(4.))[0, :, 0, 0].tolist() == [0., 1., 2., 3.]
'''),

# --- matrix operations ---------------------------------------------------------
task("JBC1/pt/015", "matrix_ops", "batched_matvec", r'''
import torch


def batched_matvec(A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Multiply each matrix A[b] (shape (B, N, M)) by its vector v[b]
    (shape (B, M)), returning shape (B, N).
    """
''', r'''
    return torch.bmm(A, v.unsqueeze(-1)).squeeze(-1)
''', r'''
def check(candidate):
    torch.manual_seed(0)
    A, v = torch.randn(5, 3, 4), torch.randn(5, 4)
    out = candidate(A, v)
    assert out.shape == (5, 3)
    expected = torch.stack([A[i] @ v[i] for i in range(5)])
    assert torch.allclose(out, expected, atol=1e-6)
    assert candidate(torch.randn(1, 1, 1), torch.randn(1, 1)).shape == (1, 1)
'''),

task("JBC1/pt/016", "matrix_ops", "gram_matrix", r'''
import torch


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    """For features x of shape (B, C, H, W), return the (B, C, C) Gram matrices
    F @ F^T / (H * W), where F is x reshaped to (B, C, H * W).
    """
''', r'''
    b, c, h, w = x.shape
    f = x.reshape(b, c, h * w)
    return f @ f.transpose(1, 2) / (h * w)
''', r'''
def check(candidate):
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4, 5)
    g = candidate(x)
    assert g.shape == (2, 3, 3)
    f = x[1].reshape(3, 20)
    assert torch.allclose(g[1], f @ f.T / 20, atol=1e-6)
    assert torch.allclose(g, g.transpose(1, 2), atol=1e-6)
    assert torch.allclose(candidate(torch.ones(1, 2, 2, 2)), torch.ones(1, 2, 2))
'''),

task("JBC1/pt/017", "matrix_ops", "cosine_similarity_matrix", r'''
import torch


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return the (N, M) matrix of cosine similarities between the rows of
    a (N, D) and b (M, D). Use an epsilon of 1e-8 on the norms.
    """
''', r'''
    a_n = a / a.norm(dim=1, keepdim=True).clamp(min=1e-8)
    b_n = b / b.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return a_n @ b_n.T
''', r'''
def check(candidate):
    a = torch.tensor([[1., 0.], [1., 1.]])
    b = torch.tensor([[0., 2.], [3., 0.], [-1., -1.]])
    out = candidate(a, b)
    s = 2 ** -0.5
    assert torch.allclose(out, torch.tensor([[0., 1., -s], [s, s, -1.]]), atol=1e-6)
    torch.manual_seed(0)
    x, y = torch.randn(5, 7), torch.randn(3, 7)
    ref = torch.nn.functional.cosine_similarity(x[:, None], y[None], dim=-1)
    assert torch.allclose(candidate(x, y), ref, atol=1e-5)
'''),

task("JBC1/pt/018", "matrix_ops", "causal_attention_weights", r'''
import math

import torch


def causal_attention_weights(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Given queries and keys of shape (B, T, D), return attention weights of
    shape (B, T, T): softmax over the last dim of q @ k^T / sqrt(D), where
    position i may only attend to positions j <= i.
    """
''', r'''
    t, d = q.shape[1], q.shape[2]
    scores = q @ k.transpose(1, 2) / math.sqrt(d)
    future = torch.ones(t, t, dtype=torch.bool, device=q.device).triu(1)
    scores = scores.masked_fill(future, float("-inf"))
    return scores.softmax(dim=-1)
''', r'''
def check(candidate):
    torch.manual_seed(0)
    q, k = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    w = candidate(q, k)
    assert w.shape == (2, 5, 5)
    assert torch.allclose(w.sum(-1), torch.ones(2, 5), atol=1e-6)
    assert torch.all(w.triu(1) == 0)
    assert torch.allclose(w[:, 0, 0], torch.ones(2))
    s = (q[0, 3] @ k[0, :4].T) / 8 ** 0.5
    assert torch.allclose(w[0, 3, :4], s.softmax(-1), atol=1e-6)
'''),

task("JBC1/pt/019", "matrix_ops", "attention", r'''
import math

import torch


def attention(q, k, v, mask=None):
    """Scaled dot-product attention.

    q: (B, Tq, D), k: (B, Tk, D), v: (B, Tk, Dv). mask, if given, is a bool
    tensor broadcastable to (B, Tq, Tk) that is True where attention is NOT
    allowed. Returns (B, Tq, Dv).
    """
''', r'''
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    return scores.softmax(dim=-1) @ v
''', r'''
def check(candidate):
    import torch.nn.functional as F
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 3, 4), torch.randn(2, 5, 4), torch.randn(2, 5, 6)
    out = candidate(q, k, v)
    assert out.shape == (2, 3, 6)
    assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v), atol=1e-5)
    mask = torch.zeros(3, 5, dtype=torch.bool)
    mask[:, 3:] = True
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=~mask)
    assert torch.allclose(candidate(q, k, v, mask), ref, atol=1e-5)
    only_first = torch.ones(1, 1, 5, dtype=torch.bool)
    only_first[..., 0] = False
    assert torch.allclose(candidate(q, k, v, only_first), v[:, :1].expand(2, 3, 6), atol=1e-6)
'''),

# --- shape manipulation --------------------------------------------------------
task("JBC1/pt/020", "shapes", "split_heads", r'''
import torch


def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Reshape x of shape (B, T, C) into (B, n_heads, T, C // n_heads), where
    head h holds channels h*(C//n_heads) to (h+1)*(C//n_heads) - 1.
    Raise ValueError if C is not divisible by n_heads.
    """
''', r'''
    b, t, c = x.shape
    if c % n_heads != 0:
        raise ValueError("channels not divisible by n_heads")
    return x.view(b, t, n_heads, c // n_heads).transpose(1, 2)
''', r'''
def check(candidate):
    x = torch.arange(2 * 3 * 8.).view(2, 3, 8)
    y = candidate(x, 4)
    assert y.shape == (2, 4, 3, 2)
    assert torch.equal(y[1, 2, 0], x[1, 0, 4:6])
    assert torch.equal(y[0, 3, 2], x[0, 2, 6:8])
    try:
        candidate(x, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
'''),

task("JBC1/pt/021", "shapes", "merge_heads", r'''
import torch


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """Inverse of splitting attention heads: turn x of shape (B, H, T, D) into
    (B, T, H * D), placing head h in channels h*D to (h+1)*D - 1.
    """
''', r'''
    b, h, t, d = x.shape
    return x.transpose(1, 2).reshape(b, t, h * d)
''', r'''
def check(candidate):
    torch.manual_seed(0)
    x = torch.randn(2, 4, 3, 5)
    y = candidate(x)
    assert y.shape == (2, 3, 20)
    assert torch.equal(y[1, 2, 10:15], x[1, 2, 2])
    z = torch.randn(2, 3, 20)
    split = z.view(2, 3, 4, 5).transpose(1, 2)
    assert torch.equal(candidate(split), z)
'''),

task("JBC1/pt/022", "shapes", "patchify", r'''
import torch


def patchify(images: torch.Tensor, p: int) -> torch.Tensor:
    """Split images (B, C, H, W) into non-overlapping p x p patches.

    Return (B, N, C * p * p) where N = (H // p) * (W // p). Patches are ordered
    row by row, and each patch is flattened in (C, p, p) order.
    """
''', r'''
    b, c, h, w = images.shape
    x = images.reshape(b, c, h // p, p, w // p, p)
    x = x.permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, (h // p) * (w // p), c * p * p)
''', r'''
def check(candidate):
    torch.manual_seed(0)
    img = torch.randn(2, 3, 4, 6)
    out = candidate(img, 2)
    assert out.shape == (2, 6, 12)
    assert torch.equal(out[0, 0], img[0, :, 0:2, 0:2].flatten())
    assert torch.equal(out[1, 4], img[1, :, 2:4, 2:4].flatten())
    assert torch.equal(out[1, 5], img[1, :, 2:4, 4:6].flatten())
    assert torch.equal(candidate(img, 1)[0, 7], img[0, :, 1, 1])
'''),

task("JBC1/pt/023", "shapes", "pad_sequences", r'''
import torch


def pad_sequences(seqs: list[torch.Tensor], pad_value: int = 0):
    """Right-pad a non-empty list of 1D int64 tensors to the longest length.

    Return (padded, lengths): padded has shape (len(seqs), max_len) and
    lengths is an int64 tensor of the original lengths.
    """
''', r'''
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.int64)
    max_len = int(lengths.max()) if len(seqs) else 0
    padded = torch.full((len(seqs), max_len), pad_value, dtype=torch.int64)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = s
    return padded, lengths
''', r'''
def check(candidate):
    seqs = [torch.tensor([1, 2, 3]), torch.tensor([4]), torch.tensor([5, 6])]
    padded, lengths = candidate(seqs, -1)
    assert padded.tolist() == [[1, 2, 3], [4, -1, -1], [5, 6, -1]]
    assert lengths.tolist() == [3, 1, 2] and lengths.dtype == torch.int64
    p2, l2 = candidate([torch.tensor([7, 8])])
    assert p2.tolist() == [[7, 8]] and l2.tolist() == [2]
    p3, l3 = candidate([torch.tensor([], dtype=torch.int64), torch.tensor([9])])
    assert p3.tolist() == [[0], [9]] and l3.tolist() == [0, 1]
'''),

task("JBC1/pt/024", "shapes", "flatten_except_batch", r'''
import torch


def flatten_except_batch(x: torch.Tensor) -> torch.Tensor:
    """Flatten every dimension except the first. Must also work when x is not
    contiguous (for example after a transpose).
    """
''', r'''
    return x.reshape(x.shape[0], -1)
''', r'''
def check(candidate):
    x = torch.arange(24.).view(2, 3, 4)
    assert candidate(x).shape == (2, 12)
    t = x.transpose(1, 2)
    assert not t.is_contiguous()
    out = candidate(t)
    assert out.shape == (2, 12)
    assert torch.equal(out[0], t[0].flatten())
    assert candidate(torch.zeros(5)).shape == (5, 1)
'''),

# --- dtypes --------------------------------------------------------------------
task("JBC1/pt/025", "dtypes", "to_uint8_image", r'''
import torch


def to_uint8_image(x: torch.Tensor) -> torch.Tensor:
    """Convert a float image with values nominally in [0, 1] to uint8 in
    [0, 255]: clamp to [0, 1], multiply by 255, round to nearest.
    """
''', r'''
    return (x.clamp(0, 1) * 255).round().to(torch.uint8)
''', r'''
def check(candidate):
    x = torch.tensor([0.0, 1.0, 0.5, -0.2, 1.7, 0.999, 0.002])
    out = candidate(x)
    assert out.dtype == torch.uint8
    assert out.tolist() == [0, 255, 128, 0, 255, 255, 1]
    assert candidate(torch.rand(2, 3, 4)).shape == (2, 3, 4)
'''),

task("JBC1/pt/026", "dtypes", "mean_of_ints", r'''
import torch


def mean_of_ints(x: torch.Tensor) -> torch.Tensor:
    """Return the mean of an integer tensor as a float32 scalar tensor.
    (torch.mean does not accept integer tensors directly.)
    """
''', r'''
    return x.to(torch.float32).mean()
''', r'''
def check(candidate):
    out = candidate(torch.tensor([1, 2, 4], dtype=torch.int64))
    assert out.dtype == torch.float32 and out.dim() == 0
    assert abs(out.item() - 7 / 3) < 1e-6
    assert candidate(torch.tensor([[250, 250]], dtype=torch.uint8)).item() == 250.0
    assert candidate(torch.tensor([-3, 3], dtype=torch.int32)).item() == 0.0
'''),

task("JBC1/pt/027", "dtypes", "stable_half_sum", r'''
import torch


def stable_half_sum(x: torch.Tensor) -> torch.Tensor:
    """Sum a float16 tensor without float16 overflow or precision loss:
    accumulate in float32 and return a float32 scalar tensor.
    """
''', r'''
    return x.to(torch.float32).sum()
''', r'''
def check(candidate):
    x = torch.full((10,), 60000.0, dtype=torch.float16)
    out = candidate(x)
    assert out.dtype == torch.float32 and torch.isfinite(out)
    assert out.item() == 600000.0
    y = torch.full((4096,), 1.0, dtype=torch.float16)
    y[0] = 2048.0
    assert candidate(y).item() == 2048.0 + 4095.0
'''),

# --- devices -------------------------------------------------------------------
task("JBC1/pt/028", "devices", "zeros_like_on", r'''
import torch


def zeros_like_on(x: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Return a zero tensor with the given shape on the same device and with the
    same dtype as x.
    """
''', r'''
    return torch.zeros(shape, dtype=x.dtype, device=x.device)
''', r'''
def check(candidate):
    x = torch.ones(3, dtype=torch.float64)
    z = candidate(x, (2, 2))
    assert z.shape == (2, 2) and z.dtype == torch.float64 and z.sum().item() == 0
    m = torch.empty(4, dtype=torch.int16, device="meta")
    zm = candidate(m, (5,))
    assert zm.device.type == "meta" and zm.dtype == torch.int16 and zm.shape == (5,)
'''),

task("JBC1/pt/029", "devices", "position_ids", r'''
import torch


def position_ids(tokens: torch.Tensor) -> torch.Tensor:
    """Given token ids of shape (B, T), return int64 positions of shape (B, T)
    where every row is 0, 1, ..., T-1, on the same device as tokens.
    """
''', r'''
    b, t = tokens.shape
    return torch.arange(t, device=tokens.device).unsqueeze(0).expand(b, t)
''', r'''
def check(candidate):
    out = candidate(torch.zeros(2, 4, dtype=torch.int64))
    assert out.tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]] and out.dtype == torch.int64
    m = torch.empty(3, 7, dtype=torch.int64, device="meta")
    pm = candidate(m)
    assert pm.device.type == "meta" and pm.shape == (3, 7)
    emb = torch.empty(7, 5, device="meta")
    assert emb[pm].shape == (3, 7, 5)
'''),

task("JBC1/pt/030", "devices", "move_batch", r'''
import torch


def move_batch(batch, device):
    """Recursively move every tensor in batch (nested dicts, lists and tuples)
    to device. Containers keep their type; non-tensor values are unchanged.
    """
''', r'''
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: move_batch(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [move_batch(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch(v, device) for v in batch)
    return batch
''', r'''
def check(candidate):
    batch = {"x": torch.ones(2), "meta": {"ids": [torch.zeros(1), "keep"]},
             "pair": (torch.ones(1), 3), "name": "b0"}
    out = candidate(batch, "meta")
    assert out["x"].device.type == "meta"
    assert out["meta"]["ids"][0].device.type == "meta"
    assert out["meta"]["ids"][1] == "keep"
    assert isinstance(out["pair"], tuple) and out["pair"][0].device.type == "meta"
    assert out["pair"][1] == 3 and out["name"] == "b0"
    assert isinstance(out["meta"]["ids"], list)
    assert batch["x"].device.type == "cpu"
    assert candidate(torch.ones(1), "cpu").device.type == "cpu"
'''),
]

for _t in TASKS:
    _t["requires"] = list(T)
