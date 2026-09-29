from _schema import task

TASKS = [
task("JBC1/py/021", "algorithms", "binary_search_leftmost", r'''
def binary_search_leftmost(items: list[int], target: int) -> int:
    """Return the index of the first occurrence of target in the sorted list
    items, or -1 if it is absent. Use binary search.

    >>> binary_search_leftmost([1, 2, 2, 2, 5], 2)
    1
    """
''', r'''
    lo, hi = 0, len(items)
    while lo < hi:
        mid = (lo + hi) // 2
        if items[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo if lo < len(items) and items[lo] == target else -1
''', r'''
def check(candidate):
    assert candidate([1, 2, 2, 2, 5], 2) == 1
    assert candidate([], 3) == -1
    assert candidate([3], 3) == 0
    assert candidate([1, 3, 5], 4) == -1
    assert candidate([1, 3, 5], 6) == -1
    assert candidate([1, 3, 5], 0) == -1
    assert candidate([7] * 100, 7) == 0
    big = list(range(0, 200000, 2))
    assert candidate(big, 123456) == 61728
'''),

task("JBC1/py/022", "algorithms", "merge_sorted", r'''
def merge_sorted(a: list[int], b: list[int]) -> list[int]:
    """Merge two ascending lists into one ascending list in linear time
    without calling sort() or sorted().

    >>> merge_sorted([1, 4, 9], [2, 4, 10, 11])
    [1, 2, 4, 4, 9, 10, 11]
    """
''', r'''
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out
''', r'''
def check(candidate):
    assert candidate([1, 4, 9], [2, 4, 10, 11]) == [1, 2, 4, 4, 9, 10, 11]
    assert candidate([], []) == []
    assert candidate([], [1, 2]) == [1, 2]
    assert candidate([3], []) == [3]
    assert candidate([-5, 0], [-6, 100]) == [-6, -5, 0, 100]
    import random
    rng = random.Random(0)
    a = sorted(rng.randint(-50, 50) for _ in range(40))
    b = sorted(rng.randint(-50, 50) for _ in range(25))
    assert candidate(a, b) == sorted(a + b)
'''),

task("JBC1/py/023", "lists", "moving_average", r'''
def moving_average(values: list[float], window: int) -> list[float]:
    """Return the averages of each full window of consecutive values.

    The result has len(values) - window + 1 items, or is empty if window is
    longer than values. Raise ValueError if window < 1.

    >>> moving_average([1, 2, 3, 4], 2)
    [1.5, 2.5, 3.5]
    """
''', r'''
    if window < 1:
        raise ValueError("window must be >= 1")
    return [sum(values[i:i + window]) / window for i in range(len(values) - window + 1)]
''', r'''
def check(candidate):
    assert candidate([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]
    assert candidate([1, 2, 3], 3) == [2.0]
    assert candidate([1, 2], 3) == []
    assert candidate([], 1) == []
    assert candidate([5, 7], 1) == [5.0, 7.0]
    try:
        candidate([1, 2], 0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
'''),

task("JBC1/py/024", "lists", "transpose", r'''
def transpose(matrix: list[list]) -> list[list]:
    """Transpose a rectangular matrix given as a list of rows.

    An empty matrix transposes to []. Raise ValueError if rows differ in length.

    >>> transpose([[1, 2, 3], [4, 5, 6]])
    [[1, 4], [2, 5], [3, 6]]
    """
''', r'''
    if not matrix:
        return []
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("ragged matrix")
    return [[row[c] for row in matrix] for c in range(width)]
''', r'''
def check(candidate):
    assert candidate([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
    assert candidate([]) == []
    assert candidate([[7]]) == [[7]]
    assert candidate([[1], [2], [3]]) == [[1, 2, 3]]
    try:
        candidate([[1, 2], [3]])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
'''),

task("JBC1/py/025", "algorithms", "spiral_order", r'''
def spiral_order(matrix: list[list[int]]) -> list[int]:
    """Return the elements of a rectangular matrix in clockwise spiral order,
    starting at the top-left corner.

    >>> spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    [1, 2, 3, 6, 9, 8, 7, 4, 5]
    """
''', r'''
    out = []
    if not matrix:
        return out
    top, bottom, left, right = 0, len(matrix) - 1, 0, len(matrix[0]) - 1
    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            out.append(matrix[top][c])
        for r in range(top + 1, bottom + 1):
            out.append(matrix[r][right])
        if top < bottom:
            for c in range(right - 1, left - 1, -1):
                out.append(matrix[bottom][c])
        if left < right:
            for r in range(bottom - 1, top, -1):
                out.append(matrix[r][left])
        top, bottom, left, right = top + 1, bottom - 1, left + 1, right - 1
    return out
''', r'''
def check(candidate):
    assert candidate([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [1, 2, 3, 6, 9, 8, 7, 4, 5]
    assert candidate([]) == []
    assert candidate([[1]]) == [1]
    assert candidate([[1, 2, 3, 4]]) == [1, 2, 3, 4]
    assert candidate([[1], [2], [3]]) == [1, 2, 3]
    assert candidate([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]) == [
        1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
    assert candidate([[1, 2], [3, 4], [5, 6]]) == [1, 2, 4, 6, 5, 3]
'''),

task("JBC1/py/026", "generators", "take_while_increasing", r'''
def take_while_increasing(iterable):
    """Yield items from iterable as long as each is strictly greater than the
    previous one. Stop at the first item that is not. Must work lazily on
    infinite iterators.

    >>> list(take_while_increasing([1, 3, 4, 4, 5]))
    [1, 3, 4]
    """
''', r'''
    previous = None
    first = True
    for item in iterable:
        if not first and item <= previous:
            return
        yield item
        previous, first = item, False
''', r'''
def check(candidate):
    import itertools
    assert list(candidate([1, 3, 4, 4, 5])) == [1, 3, 4]
    assert list(candidate([])) == []
    assert list(candidate([5])) == [5]
    assert list(candidate([5, 1, 9])) == [5]
    assert list(candidate([-3, -2, 0, 10])) == [-3, -2, 0, 10]
    gen = candidate(itertools.count())
    assert list(itertools.islice(gen, 5)) == [0, 1, 2, 3, 4]
    assert list(candidate(iter([2, 4, 3, 8]))) == [2, 4]
'''),

task("JBC1/py/027", "generators", "sliding_pairs", r'''
def sliding_pairs(iterable):
    """Yield (previous, current) tuples for each pair of consecutive items,
    lazily.

    >>> list(sliding_pairs("abcd"))
    [('a', 'b'), ('b', 'c'), ('c', 'd')]
    """
''', r'''
    it = iter(iterable)
    try:
        previous = next(it)
    except StopIteration:
        return
    for current in it:
        yield previous, current
        previous = current
''', r'''
def check(candidate):
    import itertools
    assert list(candidate("abcd")) == [("a", "b"), ("b", "c"), ("c", "d")]
    assert list(candidate([])) == []
    assert list(candidate([1])) == []
    assert list(candidate(iter([1, 2]))) == [(1, 2)]
    assert list(itertools.islice(candidate(itertools.count()), 3)) == [(0, 1), (1, 2), (2, 3)]
'''),

task("JBC1/py/028", "generators", "batched", r'''
def batched(iterable, n: int):
    """Yield tuples of up to n consecutive items; the last tuple may be shorter.
    Raise ValueError if n < 1. Must work lazily on infinite iterators.

    >>> list(batched(range(5), 2))
    [(0, 1), (2, 3), (4,)]
    """
''', r'''
    if n < 1:
        raise ValueError("n must be >= 1")
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield tuple(batch)
            batch = []
    if batch:
        yield tuple(batch)
''', r'''
def check(candidate):
    import itertools
    assert list(candidate(range(5), 2)) == [(0, 1), (2, 3), (4,)]
    assert list(candidate([], 3)) == []
    assert list(candidate("abc", 3)) == [("a", "b", "c")]
    assert list(itertools.islice(candidate(itertools.count(), 2), 2)) == [(0, 1), (2, 3)]
    try:
        list(candidate([1], 0))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
'''),

task("JBC1/py/029", "classes", "Stack", r'''
class Stack:
    """A LIFO stack.

    Methods: push(item); pop() removes and returns the top item; peek()
    returns it without removing; is_empty(); len(stack) gives the size.
    pop() and peek() on an empty stack raise IndexError.
    """
''', r'''
    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self):
        if not self._items:
            raise IndexError("peek at empty stack")
        return self._items[-1]

    def is_empty(self):
        return not self._items

    def __len__(self):
        return len(self._items)
''', r'''
def check(candidate):
    s = candidate()
    assert s.is_empty() and len(s) == 0
    s.push(1)
    s.push("two")
    assert len(s) == 2 and not s.is_empty()
    assert s.peek() == "two" and len(s) == 2
    assert s.pop() == "two"
    assert s.pop() == 1
    for method in (s.pop, s.peek):
        try:
            method()
        except IndexError:
            pass
        else:
            raise AssertionError("expected IndexError")
    t = candidate()
    t.push(None)
    assert len(t) == 1 and t.pop() is None
    assert len(candidate()) == 0
'''),

task("JBC1/py/030", "classes", "LRUCache", r'''
class LRUCache:
    """A least-recently-used cache holding at most `capacity` entries.

    get(key, default=None) returns the value and marks the key as recently
    used. put(key, value) inserts or updates the key, marks it recently used,
    and evicts the least recently used key if the capacity is exceeded.
    len(cache) gives the number of entries.
    """
''', r'''
    def __init__(self, capacity: int):
        from collections import OrderedDict
        self.capacity = capacity
        self._data = OrderedDict()

    def get(self, key, default=None):
        if key not in self._data:
            return default
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key, value):
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.capacity:
            self._data.popitem(last=False)

    def __len__(self):
        return len(self._data)
''', r'''
def check(candidate):
    c = candidate(2)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1
    c.put("c", 3)
    assert c.get("b") is None
    assert c.get("a") == 1 and c.get("c") == 3
    assert len(c) == 2
    c.put("a", 10)
    c.put("d", 4)
    assert c.get("c", "gone") == "gone"
    assert c.get("a") == 10
    one = candidate(1)
    one.put(1, 1)
    one.put(2, 2)
    assert one.get(1) is None and one.get(2) == 2 and len(one) == 1
'''),

task("JBC1/py/031", "classes", "Vector2", r'''
class Vector2:
    """A 2D vector with attributes x and y.

    Supports v + w, v - w, v * scalar, scalar * v, equality, abs(v) for the
    Euclidean length, and repr(v) == "Vector2(x, y)" using repr of each part.
    """
''', r'''
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vector2(self.x * scalar, self.y * scalar)

    __rmul__ = __mul__

    def __eq__(self, other):
        return isinstance(other, Vector2) and (self.x, self.y) == (other.x, other.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __abs__(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def __repr__(self):
        return f"Vector2({self.x!r}, {self.y!r})"
''', r'''
def check(candidate):
    V = candidate
    a, b = V(1, 2), V(3, -1)
    assert a + b == V(4, 1)
    assert a - b == V(-2, 3)
    assert a * 3 == V(3, 6)
    assert 2 * a == V(2, 4)
    assert abs(V(3, 4)) == 5
    assert repr(V(1, 2.5)) == "Vector2(1, 2.5)"
    assert V(0, 0) != V(0, 1)
    assert a == V(1, 2)
    assert (a.x, a.y) == (1, 2)
'''),

task("JBC1/py/032", "algorithms", "merge_intervals", r'''
def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or touching closed intervals (start <= end) and return
    them sorted by start. The input may be unsorted.

    >>> merge_intervals([(5, 7), (1, 3), (2, 4), (8, 8)])
    [(1, 4), (5, 7), (8, 8)]
    """
''', r'''
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
''', r'''
def check(candidate):
    assert candidate([(5, 7), (1, 3), (2, 4), (8, 8)]) == [(1, 4), (5, 7), (8, 8)]
    assert candidate([]) == []
    assert candidate([(1, 5)]) == [(1, 5)]
    assert candidate([(1, 2), (2, 3)]) == [(1, 3)]
    assert candidate([(1, 10), (2, 3), (4, 5)]) == [(1, 10)]
    assert candidate([(-3, -1), (0, 0)]) == [(-3, -1), (0, 0)]
'''),

task("JBC1/py/033", "exceptions", "safe_int", r'''
def safe_int(text: str, default=None):
    """Parse text as a base-10 integer, allowing surrounding whitespace and an
    optional leading + or - sign. Return default if parsing fails.

    >>> safe_int("  -42 ")
    -42
    """
''', r'''
    try:
        return int(text.strip(), 10)
    except (ValueError, TypeError, AttributeError):
        return default
''', r'''
def check(candidate):
    assert candidate("  -42 ") == -42
    assert candidate("+7") == 7
    assert candidate("0") == 0
    assert candidate("abc") is None
    assert candidate("", 0) == 0
    assert candidate("3.5", "bad") == "bad"
'''),

task("JBC1/py/034", "exceptions", "retry", r'''
def retry(func, attempts: int, exceptions: tuple = (Exception,)):
    """Call func() up to `attempts` times and return its first successful result.

    Only exceptions of the given types trigger another attempt; any other
    exception propagates immediately. If every attempt fails, re-raise the
    last exception.
    """
''', r'''
    last_error = None
    for _ in range(attempts):
        try:
            return func()
        except exceptions as error:
            last_error = error
    raise last_error
''', r'''
def check(candidate):
    calls = []
    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("try again")
        return "ok"
    assert candidate(flaky, 5, (ConnectionError,)) == "ok" and len(calls) == 3

    calls.clear()
    try:
        candidate(flaky, 2, (ConnectionError,))
    except ConnectionError as e:
        assert str(e) == "try again" and len(calls) == 2
    else:
        raise AssertionError("expected ConnectionError")

    count = []
    def wrong_kind():
        count.append(1)
        raise KeyError("x")
    try:
        candidate(wrong_kind, 4, (ValueError,))
    except KeyError:
        assert len(count) == 1
    else:
        raise AssertionError("expected KeyError")
    assert candidate(lambda: 5, 1) == 5
'''),

task("JBC1/py/035", "parsing", "parse_key_value_lines", r'''
def parse_key_value_lines(text: str) -> dict[str, str]:
    """Parse lines of the form "key = value".

    Blank lines and lines starting with "#" (after leading whitespace) are
    ignored. Keys and values are stripped; later keys override earlier ones.
    Split on the first "=" only. A non-empty, non-comment line without "="
    raises ValueError whose message contains the 1-based line number.
    """
''', r'''
    result = {}
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise ValueError(f"line {number}: missing '='")
        key, value = stripped.split("=", 1)
        result[key.strip()] = value.strip()
    return result
''', r'''
def check(candidate):
    text = "# settings\nname = Juniper\n\n  theme=dark \nurl = a=b\nname = LM\n"
    assert candidate(text) == {"name": "LM", "theme": "dark", "url": "a=b"}
    assert candidate("") == {}
    assert candidate("   # only a comment") == {}
    assert candidate("empty =") == {"empty": ""}
    try:
        candidate("a = 1\n\nbroken line\n")
    except ValueError as e:
        assert "3" in str(e)
    else:
        raise AssertionError("expected ValueError")
'''),

task("JBC1/py/036", "strings", "to_snake_case", r'''
def to_snake_case(name: str) -> str:
    """Convert a CamelCase or camelCase identifier to snake_case. A run of
    capitals is treated as one word (an acronym); the final capital of the
    run starts a new word when followed by a lowercase letter.

    >>> to_snake_case("HTTPServerError")
    'http_server_error'
    """
''', r'''
    import re
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()
''', r'''
def check(candidate):
    assert candidate("HTTPServerError") == "http_server_error"
    assert candidate("camelCase") == "camel_case"
    assert candidate("Simple") == "simple"
    assert candidate("already_snake") == "already_snake"
    assert candidate("parseJSON") == "parse_json"
    assert candidate("") == ""
'''),

task("JBC1/py/037", "regex", "extract_emails", r'''
import re


def extract_emails(text: str) -> list[str]:
    """Return the unique email addresses in text, lowercased, in order of first
    appearance.

    An address is a local part of letters, digits, ".", "_", "+" or "-",
    then "@", then one or more domain labels of letters, digits or "-"
    separated by dots, ending with a dot and a top-level domain of at least
    two letters.
    """
''', r'''
    pattern = r"[A-Za-z0-9._+-]+@(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}"
    seen = []
    for match in re.findall(pattern, text):
        address = match.lower()
        if address not in seen:
            seen.append(address)
    return seen
''', r'''
def check(candidate):
    text = "Mail Ann@Example.com or bob.smith+x@mail.co.uk; again ann@example.com."
    assert candidate(text) == ["ann@example.com", "bob.smith+x@mail.co.uk"]
    assert candidate("no addresses here") == []
    assert candidate("bad@domain and x@y.z") == []
    assert candidate("<dev-team@cinqic.org>") == ["dev-team@cinqic.org"]
'''),

task("JBC1/py/038", "datetime", "days_between", r'''
from datetime import date


def days_between(start: str, end: str) -> int:
    """Return the absolute number of days between two ISO dates (YYYY-MM-DD).

    >>> days_between("2024-02-27", "2024-03-01")
    3
    """
''', r'''
    return abs((date.fromisoformat(end) - date.fromisoformat(start)).days)
''', r'''
def check(candidate):
    assert candidate("2024-02-27", "2024-03-01") == 3
    assert candidate("2023-02-27", "2023-03-01") == 2
    assert candidate("2020-01-01", "2020-01-01") == 0
    assert candidate("2021-01-01", "2020-01-01") == 366
    assert candidate("1999-12-31", "2000-01-01") == 1
'''),

task("JBC1/py/039", "datetime", "add_business_days", r'''
from datetime import date, timedelta


def add_business_days(start: str, days: int) -> str:
    """Return the ISO date that is `days` business days (Monday to Friday)
    after the ISO date start. days >= 0; with days == 0 return start unchanged.

    >>> add_business_days("2024-05-03", 1)  # Friday -> Monday
    '2024-05-06'
    """
''', r'''
    current = date.fromisoformat(start)
    remaining = days
    while remaining > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            remaining -= 1
    return current.isoformat()
''', r'''
def check(candidate):
    assert candidate("2024-05-03", 1) == "2024-05-06"
    assert candidate("2024-05-06", 0) == "2024-05-06"
    assert candidate("2024-05-04", 0) == "2024-05-04"
    assert candidate("2024-05-04", 1) == "2024-05-06"
    assert candidate("2024-05-06", 5) == "2024-05-13"
    assert candidate("2024-12-30", 3) == "2025-01-02"
'''),

task("JBC1/py/040", "lists", "histogram", r'''
def histogram(values: list[float], bins: int, lo: float, hi: float) -> list[int]:
    """Count values into `bins` equal-width bins covering [lo, hi].

    Bin i covers [lo + i*w, lo + (i+1)*w) where w = (hi - lo) / bins, except
    that values equal to hi go in the last bin. Values outside [lo, hi] are
    ignored.

    >>> histogram([0, 1, 2, 3, 4], 2, 0, 4)
    [2, 3]
    """
''', r'''
    counts = [0] * bins
    width = (hi - lo) / bins
    for v in values:
        if v < lo or v > hi:
            continue
        index = min(int((v - lo) / width), bins - 1)
        counts[index] += 1
    return counts
''', r'''
def check(candidate):
    assert candidate([0, 1, 2, 3, 4], 2, 0, 4) == [2, 3]
    assert candidate([], 3, 0, 1) == [0, 0, 0]
    assert candidate([-1, 5, 11], 2, 0, 10) == [0, 1]
    assert candidate([0.5, 0.25, 0.75, 1.0], 4, 0.0, 1.0) == [0, 1, 1, 2]
    assert candidate([10, 10, 10], 5, 0, 10) == [0, 0, 0, 0, 3]
'''),
]
