from _schema import task

TASKS = [
task("JBC1/dbg/001", "python_exception", "mean_or_zero", r'''
def mean_or_zero_buggy(values):
    return sum(values) / len(values)


def mean_or_zero(values):
    """Fixed version of mean_or_zero_buggy.

    Bug report: mean_or_zero_buggy([]) raises ZeroDivisionError. The mean of an
    empty list should be 0.0. Behavior for non-empty lists is unchanged.
    """
''', r'''
    if not values:
        return 0.0
    return sum(values) / len(values)
''', r'''
def check(candidate):
    assert candidate([]) == 0.0
    assert candidate([2, 4]) == 3.0
    assert candidate([5]) == 5.0
    assert candidate([-1, 1, 3]) == 1.0
'''),

task("JBC1/dbg/002", "python_logic", "last_n", r'''
def last_n_buggy(items, n):
    return items[-n:]


def last_n(items, n):
    """Fixed version of last_n_buggy: return the last n items (n >= 0).

    Bug report: last_n_buggy([1, 2, 3], 0) returns [1, 2, 3] instead of [].
    """
''', r'''
    if n == 0:
        return []
    return items[-n:]
''', r'''
def check(candidate):
    assert candidate([1, 2, 3], 0) == []
    assert candidate([1, 2, 3], 2) == [2, 3]
    assert candidate([1, 2, 3], 3) == [1, 2, 3]
    assert candidate([1, 2, 3], 5) == [1, 2, 3]
    assert candidate([], 0) == []
'''),

task("JBC1/dbg/003", "python_mutable_default", "add_tag", r'''
def add_tag_buggy(tag, tags=[]):
    tags.append(tag)
    return tags


def add_tag(tag, tags=None):
    """Fixed version of add_tag_buggy: append tag to tags and return the list;
    when tags is not given, start from a new empty list.

    Bug report: add_tag_buggy("a") then add_tag_buggy("b") returns ['a', 'b'];
    each call without tags should return a fresh list.
    """
''', r'''
    if tags is None:
        tags = []
    tags.append(tag)
    return tags
''', r'''
def check(candidate):
    assert candidate("a") == ["a"]
    assert candidate("b") == ["b"]
    existing = ["x"]
    out = candidate("y", existing)
    assert out == ["x", "y"] and out is existing
    assert candidate("c") == ["c"]
'''),

task("JBC1/dbg/004", "python_iteration", "remove_evens", r'''
def remove_evens_buggy(numbers):
    for n in numbers:
        if n % 2 == 0:
            numbers.remove(n)
    return numbers


def remove_evens(numbers):
    """Fixed version of remove_evens_buggy: return a new list containing only
    the odd numbers, in order. The input list must not be modified.

    Bug report: remove_evens_buggy([2, 4, 5]) returns [4, 5] because the list
    is modified while it is being iterated.
    """
''', r'''
    return [n for n in numbers if n % 2 != 0]
''', r'''
def check(candidate):
    data = [2, 4, 5]
    assert candidate(data) == [5]
    assert data == [2, 4, 5]
    assert candidate([1, 2, 2, 3, 4, 4, 5]) == [1, 3, 5]
    assert candidate([]) == []
    assert candidate([-3, -2, 0]) == [-3]
'''),

task("JBC1/dbg/005", "python_arithmetic", "median", r'''
def median_buggy(values):
    s = sorted(values)
    mid = len(s) // 2
    if len(s) % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) // 2


def median(values):
    """Fixed version of median_buggy for a non-empty list of numbers.

    Bug report: median_buggy([1, 2]) returns 1 instead of 1.5.
    """
''', r'''
    s = sorted(values)
    mid = len(s) // 2
    if len(s) % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2
''', r'''
def check(candidate):
    assert candidate([1, 2]) == 1.5
    assert candidate([3, 1, 2]) == 2
    assert candidate([4, 1, 3, 2]) == 2.5
    assert candidate([7]) == 7
    assert candidate([-1, -2]) == -1.5
'''),

task("JBC1/dbg/006", "python_sorting", "sort_versions", r'''
def sort_versions_buggy(versions):
    return sorted(versions)


def sort_versions(versions):
    """Fixed version of sort_versions_buggy: sort dotted version strings
    such as "1.10.2" in ascending numeric order.

    Bug report: sort_versions_buggy(["1.10", "1.9"]) returns ['1.10', '1.9'],
    because the strings are compared as text.
    """
''', r'''
    return sorted(versions, key=lambda v: [int(part) for part in v.split(".")])
''', r'''
def check(candidate):
    assert candidate(["1.10", "1.9"]) == ["1.9", "1.10"]
    assert candidate(["2.0.1", "2.0", "10.0", "2.0.10"]) == ["2.0", "2.0.1", "2.0.10", "10.0"]
    assert candidate([]) == []
    assert candidate(["0.1"]) == ["0.1"]
'''),

task("JBC1/dbg/007", "python_closures", "make_multipliers", r'''
def make_multipliers_buggy(n):
    return [lambda x: x * i for i in range(n)]


def make_multipliers(n):
    """Fixed version of make_multipliers_buggy: return n functions where the
    i-th function multiplies its argument by i.

    Bug report: with the buggy version, every function multiplies by n - 1.
    """
''', r'''
    return [lambda x, i=i: x * i for i in range(n)]
''', r'''
def check(candidate):
    fs = candidate(4)
    assert len(fs) == 4
    assert [f(10) for f in fs] == [0, 10, 20, 30]
    assert candidate(0) == []
    assert candidate(1)[0](5) == 0
'''),

task("JBC1/dbg/008", "python_aliasing", "make_grid", r'''
def make_grid_buggy(rows, cols, fill=0):
    return [[fill] * cols] * rows


def make_grid(rows, cols, fill=0):
    """Fixed version of make_grid_buggy: a rows x cols list of lists.

    Bug report: after g = make_grid_buggy(2, 2); g[0][0] = 1, the grid is
    [[1, 0], [1, 0]] because every row is the same list object.
    """
''', r'''
    return [[fill] * cols for _ in range(rows)]
''', r'''
def check(candidate):
    g = candidate(2, 2)
    g[0][0] = 1
    assert g == [[1, 0], [0, 0]]
    assert candidate(3, 1, "x") == [["x"], ["x"], ["x"]]
    assert candidate(0, 5) == []
    h = candidate(3, 3)
    assert len({id(r) for r in h}) == 3
'''),

task("JBC1/dbg/009", "python_identity", "count_equal", r'''
def count_equal_buggy(values, target):
    return sum(1 for v in values if v is target)


def count_equal(values, target):
    """Fixed version of count_equal_buggy: count the items equal to target.

    Bug report: count_equal_buggy([int("1000")], 1000) returns 0, because it
    compares identity instead of equality.
    """
''', r'''
    return sum(1 for v in values if v == target)
''', r'''
def check(candidate):
    assert candidate([int("1000"), 1000, 5], int("1000")) == 2
    assert candidate(["ab", "a" + "b".upper().lower()], "ab") == 2
    assert candidate([], 1) == 0
    assert candidate([1.0, 1, True], 1) == 3
'''),

task("JBC1/dbg/010", "python_exception", "parse_ints", r'''
def parse_ints_buggy(texts):
    out = []
    try:
        for t in texts:
            out.append(int(t))
    except:
        pass
    return out


def parse_ints(texts):
    """Fixed version of parse_ints_buggy: parse every string as an int and
    return (values, bad_indices), where invalid entries are skipped and their
    indices listed.

    Bug report: parse_ints_buggy(["1", "x", "3"]) silently stops at "x" and
    returns [1], losing "3" and hiding the error.
    """
''', r'''
    values, bad = [], []
    for i, t in enumerate(texts):
        try:
            values.append(int(t))
        except (ValueError, TypeError):
            bad.append(i)
    return values, bad
''', r'''
def check(candidate):
    assert candidate(["1", "x", "3"]) == ([1, 3], [1])
    assert candidate([]) == ([], [])
    assert candidate(["a", "b"]) == ([], [0, 1])
    assert candidate([" 7 ", "-2", "4.5"]) == ([7, -2], [2])
'''),

task("JBC1/dbg/011", "python_recursion", "tree_depth", r'''
def tree_depth_buggy(node):
    return 1 + max(tree_depth_buggy(node.get("left")), tree_depth_buggy(node.get("right")))


def tree_depth(node):
    """Fixed version of tree_depth_buggy. A tree node is a dict with optional
    "left" and "right" children; a missing child or node is None. The depth of
    None is 0 and of a single node is 1.

    Bug report: tree_depth_buggy({}) raises AttributeError on None.
    """
''', r'''
    if node is None:
        return 0
    return 1 + max(tree_depth(node.get("left")), tree_depth(node.get("right")))
''', r'''
def check(candidate):
    assert candidate(None) == 0
    assert candidate({}) == 1
    assert candidate({"left": {}, "right": None}) == 2
    deep = {"left": {"right": {"left": {}}}, "right": {}}
    assert candidate(deep) == 4
'''),

task("JBC1/dbg/012", "python_floats", "sums_to", r'''
def sums_to_buggy(values, total):
    return sum(values) == total


def sums_to(values, total):
    """Fixed version of sums_to_buggy: return True if the floats in values add
    up to total within a relative tolerance of 1e-9 (absolute 1e-12).

    Bug report: sums_to_buggy([0.1, 0.2], 0.3) returns False.
    """
''', r'''
    import math
    return math.isclose(sum(values), total, rel_tol=1e-9, abs_tol=1e-12)
''', r'''
def check(candidate):
    assert candidate([0.1, 0.2], 0.3) is True
    assert candidate([0.1] * 10, 1.0) is True
    assert candidate([1.0, 2.0], 3.1) is False
    assert candidate([], 0.0) is True
    assert candidate([1e-3], 1.1e-3) is False
'''),

task("JBC1/dbg/013", "python_iteration", "drop_empty", r'''
def drop_empty_buggy(d):
    for key in d:
        if not d[key]:
            del d[key]
    return d


def drop_empty(d):
    """Fixed version of drop_empty_buggy: delete, in place, every key whose
    value is falsy, and return the same dict object.

    Bug report: drop_empty_buggy({"a": 0, "b": 1}) raises RuntimeError:
    dictionary changed size during iteration.
    """
''', r'''
    for key in [k for k, v in d.items() if not v]:
        del d[key]
    return d
''', r'''
def check(candidate):
    d = {"a": 0, "b": 1, "c": "", "d": [1], "e": None}
    out = candidate(d)
    assert out is d and d == {"b": 1, "d": [1]}
    assert candidate({}) == {}
    assert candidate({"x": []}) == {}
'''),

task("JBC1/dbg/014", "python_arithmetic", "celsius_to_fahrenheit", r'''
def celsius_to_fahrenheit_buggy(c):
    return c * 9 / (5 + 32)


def celsius_to_fahrenheit(c):
    """Fixed version of celsius_to_fahrenheit_buggy.

    Bug report: celsius_to_fahrenheit_buggy(100) returns about 24.3 instead
    of 212.0.
    """
''', r'''
    return c * 9 / 5 + 32
''', r'''
def check(candidate):
    assert candidate(100) == 212.0
    assert candidate(0) == 32.0
    assert candidate(-40) == -40.0
    assert abs(candidate(37) - 98.6) < 1e-9
'''),

task("JBC1/dbg/015", "python_generators", "min_max", r'''
def min_max_buggy(values):
    return min(values), max(values)


def min_max(values):
    """Fixed version of min_max_buggy. values may be any non-empty iterable,
    including a generator that can only be consumed once.

    Bug report: min_max_buggy(x for x in [3, 1, 2]) raises ValueError because
    the generator is exhausted by min() before max() runs.
    """
''', r'''
    iterator = iter(values)
    lo = hi = next(iterator)
    for v in iterator:
        if v < lo:
            lo = v
        if v > hi:
            hi = v
    return lo, hi
''', r'''
def check(candidate):
    assert candidate(x for x in [3, 1, 2]) == (1, 3)
    assert candidate([5]) == (5, 5)
    assert candidate(iter([-1, -5, 4, 4])) == (-5, 4)
    assert candidate(range(10)) == (0, 9)
'''),

# --- PyTorch bugs ---------------------------------------------------------------
task("JBC1/dbg/016", "torch_contiguity", "rows_to_flat", r'''
import torch


def rows_to_flat_buggy(x):
    return x.t().view(-1)


def rows_to_flat(x):
    """Fixed version of rows_to_flat_buggy: return the columns of the 2D
    tensor x concatenated into one 1D tensor (column-major flattening).

    Bug report: rows_to_flat_buggy raises "RuntimeError: view size is not
    compatible with input tensor's size and stride".
    """
''', r'''
    return x.t().reshape(-1)
''', r'''
def check(candidate):
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    assert candidate(x).tolist() == [1, 4, 2, 5, 3, 6]
    assert candidate(torch.tensor([[7]])).tolist() == [7]
    assert candidate(torch.zeros(0, 3)).shape == (0,)
'''),

task("JBC1/dbg/017", "torch_dtype", "classification_loss", r'''
import torch
import torch.nn.functional as F


def classification_loss_buggy(logits, labels):
    return F.cross_entropy(logits, torch.tensor(labels, dtype=torch.float32))


def classification_loss(logits, labels):
    """Fixed version of classification_loss_buggy. logits is (N, C); labels is
    a Python list of N integer class indices. Return the mean cross-entropy.

    Bug report: for float class-index targets, F.cross_entropy either raises
    a shape error or treats them as class probabilities.
    """
''', r'''
    return F.cross_entropy(logits, torch.tensor(labels, dtype=torch.int64))
''', r'''
def check(candidate):
    torch.manual_seed(0)
    logits = torch.randn(4, 3)
    labels = [0, 2, 1, 2]
    ref = F.cross_entropy(logits, torch.tensor(labels))
    assert torch.allclose(candidate(logits, labels), ref)
    sq = torch.randn(3, 3)
    assert torch.allclose(candidate(sq, [2, 0, 1]), F.cross_entropy(sq, torch.tensor([2, 0, 1])))
'''),

task("JBC1/dbg/018", "torch_autograd", "normalized", r'''
import torch


def normalized_buggy(w):
    w /= w.norm()
    return w


def normalized(w):
    """Fixed version of normalized_buggy: return w divided by its L2 norm.

    Bug report: when w is a leaf tensor with requires_grad=True,
    normalized_buggy raises "a leaf Variable that requires grad is being used
    in an in-place operation". The result must stay differentiable and w
    must not change.
    """
''', r'''
    return w / w.norm()
''', r'''
def check(candidate):
    w = torch.tensor([3.0, 4.0], requires_grad=True)
    out = candidate(w)
    assert torch.allclose(out, torch.tensor([0.6, 0.8]))
    assert w.tolist() == [3.0, 4.0]
    out[0].backward()
    assert w.grad is not None and torch.isfinite(w.grad).all()
    assert torch.allclose(candidate(torch.tensor([0.0, 2.0])), torch.tensor([0.0, 1.0]))
'''),

task("JBC1/dbg/019", "torch_dims", "row_probabilities", r'''
import torch


def row_probabilities_buggy(logits):
    return torch.softmax(logits, dim=0)


def row_probabilities(logits):
    """Fixed version of row_probabilities_buggy: convert logits of shape
    (batch, classes) to class probabilities for each row.

    Bug report: the rows of row_probabilities_buggy's output do not sum to 1.
    """
''', r'''
    return torch.softmax(logits, dim=-1)
''', r'''
def check(candidate):
    torch.manual_seed(0)
    x = torch.randn(5, 3)
    p = candidate(x)
    assert torch.allclose(p.sum(dim=1), torch.ones(5), atol=1e-6)
    assert torch.allclose(p, x.softmax(-1))
    assert torch.allclose(candidate(torch.zeros(2, 4)), torch.full((2, 4), 0.25))
'''),

task("JBC1/dbg/020", "torch_eval_mode", "predict", r'''
import torch


def predict_buggy(model, x):
    return model(x).argmax(dim=-1)


def predict(model, x):
    """Fixed version of predict_buggy: return predicted classes for x.

    Bug report: predictions change from call to call because dropout is still
    active, and an autograd graph is built. Predict in eval mode without
    gradients, then restore the model's previous mode.
    """
''', r'''
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(x).argmax(dim=-1)
    model.train(was_training)
    return out
''', r'''
def check(candidate):
    import torch.nn as nn
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 32), nn.Dropout(0.9), nn.Linear(32, 5))
    model.train()
    x = torch.randn(40, 6)
    a, b = candidate(model, x), candidate(model, x)
    assert torch.equal(a, b)
    assert model.training
    model.eval()
    with torch.no_grad():
        assert torch.equal(a, model(x).argmax(-1))
    assert not candidate(model, x).requires_grad
    assert not model.training
'''),

task("JBC1/dbg/021", "torch_training", "train_steps", r'''
import torch


def train_steps_buggy(model, x, y, optimizer, loss_fn, steps):
    for _ in range(steps):
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    return loss.item()


def train_steps(model, x, y, optimizer, loss_fn, steps):
    """Fixed version of train_steps_buggy: take `steps` optimization steps on
    the batch (x, y) and return the final loss value as a float.

    Bug report: gradients accumulate across steps, so the effective step size
    keeps growing and training diverges.
    """
''', r'''
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    return loss.item()
''', r'''
def check(candidate):
    import torch.nn as nn
    torch.manual_seed(0)
    x = torch.randn(32, 4)
    y = x.sum(dim=1, keepdim=True)
    a, b = nn.Linear(4, 1), nn.Linear(4, 1)
    b.load_state_dict(a.state_dict())
    opt_b = torch.optim.SGD(b.parameters(), lr=0.05)
    for _ in range(30):
        opt_b.zero_grad()
        nn.functional.mse_loss(b(x), y).backward()
        opt_b.step()
    final = candidate(a, x, y, torch.optim.SGD(a.parameters(), lr=0.05), nn.functional.mse_loss, 30)
    assert isinstance(final, float)
    assert torch.allclose(a.weight, b.weight, atol=1e-5)
'''),

task("JBC1/dbg/022", "torch_graph_retention", "average_loss", r'''
import torch


def average_loss_buggy(model, batches, loss_fn):
    total = 0
    for x, y in batches:
        total += loss_fn(model(x), y)
    return total / len(batches)


def average_loss(model, batches, loss_fn):
    """Fixed version of average_loss_buggy: return the mean of the per-batch
    losses as a Python float.

    Bug report: the buggy version returns a tensor that keeps every batch's
    autograd graph alive, so memory grows with the number of batches.
    """
''', r'''
    total = 0.0
    with torch.no_grad():
        for x, y in batches:
            total += loss_fn(model(x), y).item()
    return total / len(batches)
''', r'''
def check(candidate):
    import torch.nn as nn
    torch.manual_seed(0)
    model = nn.Linear(3, 1)
    batches = [(torch.randn(4, 3), torch.randn(4, 1)) for _ in range(3)]
    out = candidate(model, batches, nn.functional.mse_loss)
    assert isinstance(out, float)
    with torch.no_grad():
        ref = sum(nn.functional.mse_loss(model(x), y).item() for x, y in batches) / 3
    assert abs(out - ref) < 1e-6
    assert all(p.grad is None for p in model.parameters())
'''),

task("JBC1/dbg/023", "torch_device", "add_positions", r'''
import torch


def add_positions_buggy(x, pos_emb):
    positions = torch.arange(x.shape[1])
    return x + pos_emb(positions)


def add_positions(x, pos_emb):
    """Fixed version of add_positions_buggy: x is (B, T, D) and pos_emb an
    nn.Embedding(max_len, D). Add the embedding of positions 0..T-1.

    Bug report: with the model and x on a GPU, add_positions_buggy raises
    "Expected all tensors to be on the same device".
    """
''', r'''
    positions = torch.arange(x.shape[1], device=x.device)
    return x + pos_emb(positions)
''', r'''
def check(candidate):
    import torch.nn as nn
    torch.manual_seed(0)
    emb = nn.Embedding(10, 4)
    x = torch.randn(2, 3, 4)
    assert torch.allclose(candidate(x, emb), x + emb.weight[:3])
    class DeviceFollowingEmbedding(nn.Module):
        # Stands in for an embedding on an accelerator: output lives wherever
        # the position indices were created.
        def forward(self, idx):
            self.seen = idx.device
            return torch.zeros(idx.shape[0], 4, device=idx.device)
    fake = DeviceFollowingEmbedding()
    out = candidate(torch.empty(2, 3, 4, device="meta"), fake)
    assert fake.seen.type == "meta"
    assert out.device.type == "meta" and out.shape == (2, 3, 4)
'''),

task("JBC1/dbg/024", "torch_broadcasting", "abs_errors", r'''
import torch


def abs_errors_buggy(pred, target):
    return (pred - target).abs()


def abs_errors(pred, target):
    """Fixed version of abs_errors_buggy. pred has shape (B, 1) (a regression
    head) and target has shape (B,). Return per-sample absolute errors (B,).

    Bug report: abs_errors_buggy returns a (B, B) tensor due to broadcasting.
    """
''', r'''
    return (pred.squeeze(-1) - target).abs()
''', r'''
def check(candidate):
    pred = torch.tensor([[1.0], [2.0], [5.0]])
    target = torch.tensor([1.5, 2.0, 3.0])
    out = candidate(pred, target)
    assert out.shape == (3,)
    assert out.tolist() == [0.5, 0.0, 2.0]
    assert candidate(torch.tensor([[4.0]]), torch.tensor([1.0])).tolist() == [3.0]
'''),

task("JBC1/dbg/025", "torch_loss", "loss_from_logits", r'''
import torch
import torch.nn.functional as F


def loss_from_logits_buggy(logits, targets):
    probs = torch.softmax(logits, dim=-1)
    return F.cross_entropy(probs, targets)


def loss_from_logits(logits, targets):
    """Fixed version of loss_from_logits_buggy: mean cross-entropy between
    logits (N, C) and int64 targets (N,).

    Bug report: the loss barely decreases during training, because
    F.cross_entropy already applies log-softmax and the buggy version applies
    softmax twice.
    """
''', r'''
    return F.cross_entropy(logits, targets)
''', r'''
def check(candidate):
    torch.manual_seed(0)
    logits = torch.randn(6, 4) * 5
    t = torch.randint(0, 4, (6,))
    assert torch.allclose(candidate(logits, t), F.cross_entropy(logits, t))
    perfect = torch.tensor([[20.0, 0.0], [0.0, 20.0]])
    assert candidate(perfect, torch.tensor([0, 1])).item() < 1e-6
'''),

task("JBC1/dbg/026", "torch_api_misuse", "accuracy", r'''
import torch


def accuracy_buggy(logits, targets):
    preds = logits.max(dim=1)
    return (preds == targets).float().mean().item()


def accuracy(logits, targets):
    """Fixed version of accuracy_buggy: fraction of rows of logits (N, C) whose
    highest-scoring class equals targets (N,), as a Python float.

    Bug report: accuracy_buggy raises an error, because max(dim=...) returns
    a (values, indices) pair rather than a tensor of class indices.
    """
''', r'''
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()
''', r'''
def check(candidate):
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    targets = torch.tensor([1, 0, 0, 0])
    out = candidate(logits, targets)
    assert isinstance(out, float) and abs(out - 0.75) < 1e-9
    assert candidate(torch.eye(3), torch.tensor([0, 1, 2])) == 1.0
'''),

task("JBC1/dbg/027", "torch_masking", "mask_padding", r'''
import torch


def mask_padding_buggy(scores, lengths):
    t = scores.shape[1]
    valid = torch.arange(t)[None, :] < lengths[:, None]
    return scores.masked_fill(valid, float("-inf"))


def mask_padding(scores, lengths):
    """Fixed version of mask_padding_buggy. scores is (B, T); lengths (B,)
    gives the number of real tokens in each row. Set the padding positions
    (index >= length) to -inf and leave real positions unchanged.

    Bug report: mask_padding_buggy hides the real tokens and keeps the padding.
    """
''', r'''
    t = scores.shape[1]
    valid = torch.arange(t, device=scores.device)[None, :] < lengths[:, None]
    return scores.masked_fill(~valid, float("-inf"))
''', r'''
def check(candidate):
    inf = float("-inf")
    scores = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = candidate(scores, torch.tensor([2, 3]))
    assert out.tolist() == [[1.0, 2.0, inf], [4.0, 5.0, 6.0]]
    assert candidate(scores, torch.tensor([0, 1])).tolist() == [[inf, inf, inf], [4.0, inf, inf]]
    assert scores[0, 2].item() == 3.0
'''),

task("JBC1/dbg/028", "torch_module_registration", "Stack", r'''
import torch.nn as nn


class StackBuggy(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = [nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Stack(nn.Module):
    """Fixed version of StackBuggy: a chain of nn.Linear layers mapping
    dims[0] -> dims[1] -> ... -> dims[-1], applied in order, stored in
    self.layers.

    Bug report: StackBuggy().parameters() is empty, so the optimizer trains
    nothing and .to(device) does not move the layers.
    """
''', r'''
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
''', r'''
def check(candidate):
    import torch
    torch.manual_seed(0)
    m = candidate([4, 8, 2])
    assert sum(p.numel() for p in m.parameters()) == 4 * 8 + 8 + 8 * 2 + 2
    x = torch.randn(3, 4)
    assert torch.allclose(m(x), m.layers[1](m.layers[0](x)))
    assert "layers.1.weight" in m.state_dict()
    assert m.to("meta").layers[0].weight.device.type == "meta"
'''),

task("JBC1/dbg/029", "torch_autograd", "weighted_total", r'''
import torch


def weighted_total_buggy(values, weights):
    v = torch.tensor(values.detach().numpy())
    return (v * weights).sum()


def weighted_total(values, weights):
    """Fixed version of weighted_total_buggy: return sum(values * weights) as a
    scalar tensor through which gradients flow to both values and weights.

    Bug report: after weighted_total_buggy(values, weights).backward(),
    values.grad is None.
    """
''', r'''
    return (values * weights).sum()
''', r'''
def check(candidate):
    v = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    w = torch.tensor([0.5, -1.0, 2.0], requires_grad=True)
    out = candidate(v, w)
    assert out.dim() == 0 and abs(out.item() - 4.5) < 1e-6
    out.backward()
    assert v.grad is not None and torch.allclose(v.grad, w.detach())
    assert torch.allclose(w.grad, v.detach())
'''),

task("JBC1/dbg/030", "torch_overflow", "brighten", r'''
import torch


def brighten_buggy(image, amount):
    return image + amount


def brighten(image, amount):
    """Fixed version of brighten_buggy: add amount (an int, possibly negative)
    to a uint8 image, saturating at 0 and 255, and return a uint8 tensor.

    Bug report: brighten_buggy(torch.tensor([250], dtype=torch.uint8), 10)
    returns tensor([4]) because uint8 arithmetic wraps around.
    """
''', r'''
    return (image.to(torch.int16) + amount).clamp(0, 255).to(torch.uint8)
''', r'''
def check(candidate):
    img = torch.tensor([0, 100, 250, 255], dtype=torch.uint8)
    out = candidate(img, 10)
    assert out.dtype == torch.uint8 and out.tolist() == [10, 110, 255, 255]
    assert candidate(img, -120).tolist() == [0, 0, 130, 135]
    assert candidate(img, 0).tolist() == [0, 100, 250, 255]
'''),
]

for _t in TASKS:
    if _t["subcategory"].startswith("torch_"):
        _t["requires"] = ["torch"]
