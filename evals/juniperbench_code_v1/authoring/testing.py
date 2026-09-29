from _schema import task, test_task

_HEADER = '''def test_{name}():
    """Assert-based unit tests for {name}. They must pass for the correct
    implementation above and fail for implementations with common mistakes,
    so cover {focus}.
    """
'''


def header(name: str, focus: str) -> str:
    return _HEADER.format(name=name, focus=focus)


TASKS = [
# --- write tests (mutation-scored) ---------------------------------------------
test_task("JBC1/test/001", "write_tests", "test_clamp", r'''
def clamp(x, lo, hi):
    """Limit x to the closed range [lo, hi]. Raise ValueError if lo > hi."""
    if lo > hi:
        raise ValueError("lo must be <= hi")
    return max(lo, min(x, hi))
''', header("clamp", "values below, inside, above and on the bounds, and invalid ranges"), r'''
    assert clamp(5, 0, 10) == 5
    assert clamp(-3, 0, 10) == 0
    assert clamp(12, 0, 10) == 10
    assert clamp(0, 0, 10) == 0 and clamp(10, 0, 10) == 10
    assert clamp(4, 4, 4) == 4
    try:
        clamp(1, 5, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
''', [r'''
def clamp(x, lo, hi):
    if lo > hi:
        raise ValueError("lo must be <= hi")
    return min(lo, max(x, hi))
''', r'''
def clamp(x, lo, hi):
    return max(lo, min(x, hi))
''', r'''
def clamp(x, lo, hi):
    if lo > hi:
        raise ValueError("lo must be <= hi")
    if x < lo:
        return lo
    if x >= hi:
        return hi - 1
    return x
''']),

test_task("JBC1/test/002", "write_tests", "test_is_leap_year", r'''
def is_leap_year(year):
    """Gregorian leap year rule."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
''', header("is_leap_year", "ordinary years, years divisible by 4, by 100 and by 400"), r'''
    assert is_leap_year(2024) is True
    assert is_leap_year(2023) is False
    assert is_leap_year(1900) is False
    assert is_leap_year(2000) is True
    assert is_leap_year(2100) is False
''', [r'''
def is_leap_year(year):
    return year % 4 == 0
''', r'''
def is_leap_year(year):
    return year % 4 == 0 and year % 100 != 0
''', r'''
def is_leap_year(year):
    return year % 400 == 0 or year % 100 == 0
''']),

test_task("JBC1/test/003", "write_tests", "test_split_evenly", r'''
def split_evenly(total, parts):
    """Split the integer total into `parts` integers that differ by at most 1,
    larger ones first, summing to total. Raise ValueError if parts < 1."""
    if parts < 1:
        raise ValueError("parts must be >= 1")
    base, extra = divmod(total, parts)
    return [base + 1] * extra + [base] * (parts - extra)
''', header("split_evenly", "exact division, remainders, ordering, sums and invalid parts"), r'''
    assert split_evenly(10, 3) == [4, 3, 3]
    assert split_evenly(9, 3) == [3, 3, 3]
    assert split_evenly(2, 4) == [1, 1, 0, 0]
    assert sum(split_evenly(17, 5)) == 17
    assert split_evenly(5, 1) == [5]
    try:
        split_evenly(5, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
''', [r'''
def split_evenly(total, parts):
    if parts < 1:
        raise ValueError("parts must be >= 1")
    base = total // parts
    return [base] * parts
''', r'''
def split_evenly(total, parts):
    if parts < 1:
        raise ValueError("parts must be >= 1")
    base, extra = divmod(total, parts)
    return [base] * (parts - extra) + [base + 1] * extra
''', r'''
def split_evenly(total, parts):
    base, extra = divmod(total, max(parts, 1))
    return [base + 1] * extra + [base] * (parts - extra)
''']),

test_task("JBC1/test/004", "write_tests", "test_count_words", r'''
def count_words(text):
    """Number of whitespace-separated words in text."""
    return len(text.split())
''', header("count_words", "empty strings, repeated spaces, tabs and newlines"), r'''
    assert count_words("hello world") == 2
    assert count_words("") == 0
    assert count_words("   ") == 0
    assert count_words("  a   b  ") == 2
    assert count_words("one\ttwo\nthree") == 3
''', [r'''
def count_words(text):
    return len(text.split(" "))
''', r'''
def count_words(text):
    return text.count(" ") + 1
''', r'''
def count_words(text):
    return len([w for w in text.split(" ") if w])
''']),

test_task("JBC1/test/005", "write_tests", "test_median", r'''
def median(values):
    """Median of a non-empty list of numbers. Raise ValueError if empty."""
    if not values:
        raise ValueError("empty")
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
''', header("median", "odd and even lengths, unsorted input and the empty case"), r'''
    assert median([3, 1, 2]) == 2
    assert median([4, 1, 3, 2]) == 2.5
    assert median([7]) == 7
    assert median([5, 5, 1, 9]) == 5
    try:
        median([])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
''', [r'''
def median(values):
    if not values:
        raise ValueError("empty")
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
''', r'''
def median(values):
    if not values:
        raise ValueError("empty")
    s = sorted(values)
    return s[len(s) // 2]
''', r'''
def median(values):
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2
''']),

test_task("JBC1/test/006", "write_tests", "test_safe_divide", r'''
def safe_divide(a, b, default=0.0):
    """Return a / b, or default when b == 0."""
    if b == 0:
        return default
    return a / b
''', header("safe_divide", "normal division, zero divisors and custom defaults"), r'''
    assert safe_divide(6, 3) == 2.0
    assert safe_divide(1, 4) == 0.25
    assert safe_divide(1, 0) == 0.0
    assert safe_divide(1, 0, default=None) is None
    assert safe_divide(-3, 2) == -1.5
''', [r'''
def safe_divide(a, b, default=0.0):
    if b == 0:
        return 0.0
    return a / b
''', r'''
def safe_divide(a, b, default=0.0):
    if b == 0:
        return default
    return a // b
''', r'''
def safe_divide(a, b, default=0.0):
    if a == 0:
        return default
    return a / b
''']),

test_task("JBC1/test/007", "write_tests", "test_truncate", r'''
def truncate(text, limit, suffix="..."):
    """Shorten text to at most `limit` characters, ending with suffix when
    anything was cut. Text that already fits is returned unchanged."""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(suffix))] + suffix
''', header("truncate", "short text, exact-length text, long text and custom suffixes"), r'''
    assert truncate("hello", 10) == "hello"
    assert truncate("hello", 5) == "hello"
    assert truncate("hello world", 8) == "hello..."
    assert len(truncate("abcdefghij", 6)) == 6
    assert truncate("abcdef", 4, suffix="~") == "abc~"
''', [r'''
def truncate(text, limit, suffix="..."):
    if len(text) < limit:
        return text
    return text[: max(0, limit - len(suffix))] + suffix
''', r'''
def truncate(text, limit, suffix="..."):
    if len(text) <= limit:
        return text
    return text[:limit] + suffix
''', r'''
def truncate(text, limit, suffix="..."):
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."
''']),

test_task("JBC1/test/008", "write_tests", "test_stable_softmax", r'''
import torch


def stable_softmax(x):
    """Softmax over the last dimension, safe for very large inputs."""
    shifted = x - x.max(dim=-1, keepdim=True).values
    e = shifted.exp()
    return e / e.sum(dim=-1, keepdim=True)
''', header("stable_softmax", "row sums, agreement with torch.softmax, 2D inputs and huge values"), r'''
    x = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    out = stable_softmax(x)
    assert torch.allclose(out, torch.softmax(x, dim=-1))
    assert torch.allclose(out.sum(dim=-1), torch.ones(2))
    big = stable_softmax(torch.tensor([1000.0, 1000.0]))
    assert torch.isfinite(big).all() and torch.allclose(big, torch.tensor([0.5, 0.5]))
''', [r'''
import torch


def stable_softmax(x):
    e = x.exp()
    return e / e.sum(dim=-1, keepdim=True)
''', r'''
import torch


def stable_softmax(x):
    shifted = x - x.max(dim=0, keepdim=True).values
    e = shifted.exp()
    return e / e.sum(dim=0, keepdim=True)
''', r'''
import torch


def stable_softmax(x):
    shifted = x - x.max(dim=-1, keepdim=True).values
    return shifted.exp()
'''], requires=("torch",)),

test_task("JBC1/test/009", "write_tests", "test_accuracy", r'''
import torch


def accuracy(logits, targets):
    """Fraction of rows of logits (N, C) whose argmax equals targets (N,)."""
    return (logits.argmax(dim=1) == targets).float().mean().item()
''', header("accuracy", "all-correct, all-wrong and partial batches with more than two classes"), r'''
    logits = torch.tensor([[0.1, 0.7, 0.2], [0.9, 0.05, 0.05], [0.2, 0.3, 0.5], [0.4, 0.5, 0.1]])
    assert accuracy(logits, torch.tensor([1, 0, 2, 1])) == 1.0
    assert accuracy(logits, torch.tensor([0, 1, 0, 2])) == 0.0
    assert abs(accuracy(logits, torch.tensor([1, 0, 0, 0])) - 0.5) < 1e-9
''', [r'''
import torch


def accuracy(logits, targets):
    return (logits.argmax(dim=0)[: len(targets)] == targets).float().mean().item()
''', r'''
import torch


def accuracy(logits, targets):
    return (logits.argmax(dim=1) == targets).float().sum().item()
''', r'''
import torch


def accuracy(logits, targets):
    return (logits.argmin(dim=1) == targets).float().mean().item()
'''], requires=("torch",)),

test_task("JBC1/test/010", "write_tests", "test_flatten_dict", r'''
def flatten_dict(d, sep="."):
    """Flatten nested dicts: {"a": {"b": 1}} -> {"a.b": 1}. Non-dict values
    (including lists and empty dicts) are leaves."""
    out = {}
    for key, value in d.items():
        if isinstance(value, dict) and value:
            for sub_key, sub_value in flatten_dict(value, sep).items():
                out[f"{key}{sep}{sub_key}"] = sub_value
        else:
            out[key] = value
    return out
''', header("flatten_dict", "several nesting levels, custom separators, lists and empty dicts"), r'''
    assert flatten_dict({"a": {"b": {"c": 1}}, "d": 2}) == {"a.b.c": 1, "d": 2}
    assert flatten_dict({"a": {"b": 1}}, sep="/") == {"a/b": 1}
    assert flatten_dict({"x": [1, {"y": 2}]}) == {"x": [1, {"y": 2}]}
    assert flatten_dict({"e": {}}) == {"e": {}}
    assert flatten_dict({}) == {}
''', [r'''
def flatten_dict(d, sep="."):
    out = {}
    for key, value in d.items():
        if isinstance(value, dict) and value:
            for sub_key, sub_value in value.items():
                out[f"{key}{sep}{sub_key}"] = sub_value
        else:
            out[key] = value
    return out
''', r'''
def flatten_dict(d, sep="."):
    out = {}
    for key, value in d.items():
        if isinstance(value, dict) and value:
            for sub_key, sub_value in flatten_dict(value, sep).items():
                out[f"{key}.{sub_key}"] = sub_value
        else:
            out[key] = value
    return out
''', r'''
def flatten_dict(d, sep="."):
    out = {}
    for key, value in d.items():
        if isinstance(value, dict):
            for sub_key, sub_value in flatten_dict(value, sep).items():
                out[f"{key}{sep}{sub_key}"] = sub_value
        else:
            out[key] = value
    return out
''']),

# --- edge-case-heavy implementations -------------------------------------------
task("JBC1/test/011", "edge_cases", "percent", r'''
def percent(part: float, whole: float) -> float:
    """Return part as a percentage of whole, rounded to one decimal place.
    A whole of 0 gives 0.0. Negative values are allowed.
    """
''', r'''
    if whole == 0:
        return 0.0
    return round(part / whole * 100, 1)
''', r'''
def check(candidate):
    assert candidate(1, 4) == 25.0
    assert candidate(1, 3) == 33.3
    assert candidate(2, 3) == 66.7
    assert candidate(5, 0) == 0.0
    assert candidate(0, 7) == 0.0
    assert candidate(-1, 4) == -25.0
    assert candidate(3, 2) == 150.0
'''),

task("JBC1/test/012", "edge_cases", "parse_bool", r'''
def parse_bool(text: str) -> bool:
    """Parse a configuration boolean. Accept (case-insensitively, ignoring
    surrounding whitespace) "true", "yes", "on", "1" as True and "false",
    "no", "off", "0" as False. Anything else raises ValueError.
    """
''', r'''
    value = text.strip().lower()
    if value in ("true", "yes", "on", "1"):
        return True
    if value in ("false", "no", "off", "0"):
        return False
    raise ValueError(f"not a boolean: {text!r}")
''', r'''
def check(candidate):
    for t in ("true", "YES", " On ", "1"):
        assert candidate(t) is True
    for f in ("False", "no", "OFF", " 0\n"):
        assert candidate(f) is False
    for bad in ("", "2", "tru", "yes please", "none"):
        try:
            candidate(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
'''),

task("JBC1/test/013", "edge_cases", "ordinal", r'''
def ordinal(n: int) -> str:
    """Return n with its English ordinal suffix: 1st, 2nd, 3rd, 4th, 11th,
    12th, 13th, 21st, 111th. n must be >= 0, otherwise raise ValueError.
    """
''', r'''
    if n < 0:
        raise ValueError("n must be non-negative")
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"
''', r'''
def check(candidate):
    expected = {0: "0th", 1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 11: "11th", 12: "12th",
                13: "13th", 21: "21st", 22: "22nd", 101: "101st", 111: "111th",
                112: "112th", 1003: "1003rd"}
    for n, s in expected.items():
        assert candidate(n) == s, n
    try:
        candidate(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
'''),

task("JBC1/test/014", "edge_cases", "float_range", r'''
def float_range(start: float, stop: float, step: float) -> list[float]:
    """Like range() but for floats: values start + i * step for i = 0, 1, ...
    that are strictly before stop (in the direction of step). Computing each
    value as start + i * step avoids accumulated error. Raise ValueError if
    step is 0.
    """
''', r'''
    if step == 0:
        raise ValueError("step must not be zero")
    out = []
    i = 0
    while True:
        value = start + i * step
        if (step > 0 and value >= stop) or (step < 0 and value <= stop):
            return out
        out.append(value)
        i += 1
''', r'''
def check(candidate):
    assert candidate(0, 1, 0.25) == [0.0, 0.25, 0.5, 0.75]
    assert candidate(1, 0, -0.5) == [1.0, 0.5]
    assert candidate(0, 0, 1) == []
    assert candidate(0, -1, 1) == []
    vals = candidate(0, 1, 0.1)
    assert len(vals) == 10 and vals[3] == 0 + 3 * 0.1
    try:
        candidate(0, 1, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
'''),

task("JBC1/test/015", "edge_cases", "unique_filename", r'''
def unique_filename(name: str, existing: set[str]) -> str:
    """Return name if it is not in existing; otherwise insert " (k)" before
    the extension using the smallest k >= 1 that is free: "a.txt" ->
    "a (1).txt". The extension is the part after the last dot, except that a
    leading dot (as in ".env") does not start an extension.
    """
''', r'''
    if name not in existing:
        return name
    dot = name.rfind(".")
    if dot <= 0:
        stem, ext = name, ""
    else:
        stem, ext = name[:dot], name[dot:]
    k = 1
    while f"{stem} ({k}){ext}" in existing:
        k += 1
    return f"{stem} ({k}){ext}"
''', r'''
def check(candidate):
    assert candidate("a.txt", set()) == "a.txt"
    assert candidate("a.txt", {"a.txt"}) == "a (1).txt"
    assert candidate("a.txt", {"a.txt", "a (1).txt", "a (2).txt"}) == "a (3).txt"
    assert candidate("a.txt", {"a.txt", "a (2).txt"}) == "a (1).txt"
    assert candidate("README", {"README"}) == "README (1)"
    assert candidate(".env", {".env"}) == ".env (1)"
    assert candidate("x.tar.gz", {"x.tar.gz"}) == "x.tar (1).gz"
'''),

task("JBC1/test/016", "edge_cases", "window_max", r'''
def window_max(values: list[int], k: int) -> list[int]:
    """Return the maximum of every window of k consecutive values. If k is
    larger than len(values), return []. Raise ValueError if k < 1.
    """
''', r'''
    if k < 1:
        raise ValueError("k must be >= 1")
    return [max(values[i:i + k]) for i in range(len(values) - k + 1)]
''', r'''
def check(candidate):
    assert candidate([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
    assert candidate([4, 2], 1) == [4, 2]
    assert candidate([4, 2], 2) == [4]
    assert candidate([4, 2], 3) == []
    assert candidate([], 1) == []
    for bad in (0, -1):
        try:
            candidate([1], bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
'''),

task("JBC1/test/017", "edge_cases", "masked_softmax", r'''
import torch


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dim of scores, using only positions where the
    bool mask is True. Masked positions get probability 0, and a row with no
    True positions is all zeros (never NaN).
    """
''', r'''
    filled = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(filled, dim=-1)
    return torch.nan_to_num(probs, nan=0.0) * mask
''', r'''
def check(candidate):
    scores = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [5.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False], [False, False, False], [True, True, True]])
    out = candidate(scores, mask)
    assert not torch.isnan(out).any()
    first = torch.cat([torch.tensor([1.0, 2.0]).softmax(0), torch.zeros(1)])
    assert torch.allclose(out[0], first)
    assert torch.equal(out[1], torch.zeros(3))
    assert torch.allclose(out[2], scores[2].softmax(0))
'''),

task("JBC1/test/018", "edge_cases", "mean_std", r'''
def mean_std(values: list[float]) -> tuple[float, float]:
    """Return (mean, population standard deviation) of values. A single value
    has standard deviation 0.0. Raise ValueError for an empty list.
    """
''', r'''
    if not values:
        raise ValueError("empty")
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, var ** 0.5
''', r'''
def check(candidate):
    assert candidate([2, 4, 4, 4, 5, 5, 7, 9]) == (5.0, 2.0)
    assert candidate([3]) == (3.0, 0.0)
    m, s = candidate([1.5, 2.5])
    assert m == 2.0 and abs(s - 0.5) < 1e-12
    try:
        candidate([])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
'''),

task("JBC1/test/019", "edge_cases", "int_to_le_bytes", r'''
def int_to_le_bytes(n: int, length: int) -> bytes:
    """Encode the non-negative integer n as exactly `length` little-endian
    bytes. Raise ValueError if n is negative and OverflowError if n does not
    fit in length bytes.
    """
''', r'''
    if n < 0:
        raise ValueError("n must be non-negative")
    if n >= 256 ** length:
        raise OverflowError("n does not fit")
    return bytes((n >> (8 * i)) & 0xFF for i in range(length))
''', r'''
def check(candidate):
    assert candidate(1, 2) == b"\x01\x00"
    assert candidate(0x1234, 2) == b"\x34\x12"
    assert candidate(0, 3) == b"\x00\x00\x00"
    assert candidate(255, 1) == b"\xff"
    assert candidate(0, 0) == b""
    try:
        candidate(256, 1)
    except OverflowError:
        pass
    else:
        raise AssertionError("expected OverflowError")
    try:
        candidate(-1, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
'''),

task("JBC1/test/020", "edge_cases", "pad_or_truncate", r'''
import torch


def pad_or_truncate(seq: torch.Tensor, length: int, pad_value: int = 0) -> torch.Tensor:
    """Return a 1D tensor of exactly `length` items: seq truncated if longer,
    or right-padded with pad_value if shorter. Keep seq's dtype.
    """
''', r'''
    if len(seq) >= length:
        return seq[:length].clone()
    pad = torch.full((length - len(seq),), pad_value, dtype=seq.dtype, device=seq.device)
    return torch.cat([seq, pad])
''', r'''
def check(candidate):
    s = torch.tensor([1, 2, 3])
    assert candidate(s, 5).tolist() == [1, 2, 3, 0, 0]
    assert candidate(s, 2).tolist() == [1, 2]
    assert candidate(s, 3).tolist() == [1, 2, 3]
    assert candidate(s, 0).tolist() == []
    assert candidate(s, 4, -1).tolist() == [1, 2, 3, -1]
    assert candidate(torch.tensor([], dtype=torch.int64), 2).tolist() == [0, 0]
    assert candidate(s.to(torch.int16), 4).dtype == torch.int16
    out = candidate(s, 2)
    out[0] = 99
    assert s[0].item() == 1
'''),
]

for _t in TASKS:
    if "torch" in _t["prompt"] and "torch" not in _t["requires"]:
        _t["requires"] = ["torch"]
