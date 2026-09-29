from _schema import task

TASKS = [
task("JBC1/py/001", "strings", "interleave_strings", r'''
def interleave_strings(a: str, b: str) -> str:
    """Alternate characters from a and b, starting with a.

    When one string runs out, append the rest of the other unchanged.

    >>> interleave_strings("abc", "12")
    'a1b2c'
    """
''', r'''
    out = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            out.append(a[i])
        if i < len(b):
            out.append(b[i])
    return "".join(out)
''', r'''
def check(candidate):
    assert candidate("abc", "12") == "a1b2c"
    assert candidate("", "") == ""
    assert candidate("", "xyz") == "xyz"
    assert candidate("xy", "") == "xy"
    assert candidate("ab", "1234") == "a1b234"
    assert candidate("héllo", "WÖ") == "hWéÖllo"
'''),

task("JBC1/py/002", "strings", "collapse_whitespace", r'''
def collapse_whitespace(text: str) -> str:
    """Replace every run of whitespace (spaces, tabs, newlines) with a single space
    and remove leading and trailing whitespace.

    >>> collapse_whitespace("  hello \t\n world  ")
    'hello world'
    """
''', r'''
    return " ".join(text.split())
''', r'''
def check(candidate):
    assert candidate("  hello \t\n world  ") == "hello world"
    assert candidate("") == ""
    assert candidate("   \n\t ") == ""
    assert candidate("one") == "one"
    assert candidate("a  b   c") == "a b c"
    assert candidate("\nline1\n\nline2\n") == "line1 line2"
'''),

task("JBC1/py/003", "strings", "title_except", r'''
def title_except(text: str, minor_words: set[str]) -> str:
    """Title-case each word of text, except words in minor_words (compared
    case-insensitively), which are lowercased. The first word is always
    title-cased. Words are separated by single spaces.

    >>> title_except("the lord of the rings", {"of", "the"})
    'The Lord of the Rings'
    """
''', r'''
    words = text.split(" ")
    out = []
    for i, w in enumerate(words):
        if i > 0 and w.lower() in minor_words:
            out.append(w.lower())
        else:
            out.append(w.capitalize())
    return " ".join(out)
''', r'''
def check(candidate):
    assert candidate("the lord of the rings", {"of", "the"}) == "The Lord of the Rings"
    assert candidate("A TALE OF TWO CITIES", {"of"}) == "A Tale of Two Cities"
    assert candidate("of mice and men", {"of", "and"}) == "Of Mice and Men"
    assert candidate("python", set()) == "Python"
    assert candidate("war AND peace", {"and"}) == "War and Peace"
'''),

task("JBC1/py/004", "strings", "run_length_encode", r'''
def run_length_encode(text: str) -> str:
    """Encode text as character/count pairs for each run of repeated characters.

    >>> run_length_encode("aaabcc")
    'a3b1c2'
    """
''', r'''
    if not text:
        return ""
    out = []
    current, count = text[0], 1
    for ch in text[1:]:
        if ch == current:
            count += 1
        else:
            out.append(f"{current}{count}")
            current, count = ch, 1
    out.append(f"{current}{count}")
    return "".join(out)
''', r'''
def check(candidate):
    assert candidate("aaabcc") == "a3b1c2"
    assert candidate("") == ""
    assert candidate("z") == "z1"
    assert candidate("aaaaaaaaaaaa") == "a12"
    assert candidate("abab") == "a1b1a1b1"
    assert candidate("  !!") == " 2!2"
'''),

task("JBC1/py/005", "strings", "run_length_decode", r'''
def run_length_decode(encoded: str) -> str:
    """Invert run-length encoding where each character is followed by a
    decimal count that may have several digits. Characters are never digits.

    >>> run_length_decode("a3b1c2")
    'aaabcc'
    """
''', r'''
    out = []
    i = 0
    while i < len(encoded):
        ch = encoded[i]
        i += 1
        start = i
        while i < len(encoded) and encoded[i].isdigit():
            i += 1
        out.append(ch * int(encoded[start:i]))
    return "".join(out)
''', r'''
def check(candidate):
    assert candidate("a3b1c12") == "aaab" + "c" * 12
    assert candidate("") == ""
    assert candidate("x1") == "x"
    assert candidate("z100") == "z" * 100
    assert candidate(" 2!3") == "  !!!"
    assert candidate("a0b2") == "bb"
'''),

task("JBC1/py/006", "lists", "chunk", r'''
def chunk(items: list, size: int) -> list[list]:
    """Split items into consecutive lists of length size; the last list may be
    shorter. Raise ValueError if size is not positive.

    >>> chunk([1, 2, 3, 4, 5], 2)
    [[1, 2], [3, 4], [5]]
    """
''', r'''
    if size <= 0:
        raise ValueError("size must be positive")
    return [items[i:i + size] for i in range(0, len(items), size)]
''', r'''
def check(candidate):
    assert candidate([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert candidate([], 3) == []
    assert candidate([1, 2, 3], 3) == [[1, 2, 3]]
    assert candidate([1, 2, 3], 10) == [[1, 2, 3]]
    assert candidate(list("abcd"), 1) == [["a"], ["b"], ["c"], ["d"]]
    for bad in (0, -2):
        try:
            candidate([1], bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
'''),

task("JBC1/py/007", "lists", "flatten", r'''
def flatten(nested) -> list:
    """Flatten arbitrarily nested lists and tuples into a flat list, preserving
    order. Strings and all other values are treated as single items.

    >>> flatten([1, [2, (3, [4])], "ab"])
    [1, 2, 3, 4, 'ab']
    """
''', r'''
    out = []
    for item in nested:
        if isinstance(item, (list, tuple)):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out
''', r'''
def check(candidate):
    assert candidate([1, [2, (3, [4])], "ab"]) == [1, 2, 3, 4, "ab"]
    assert candidate([]) == []
    assert candidate([[], [[]], ()]) == []
    assert candidate([[[[[5]]]]]) == [5]
    assert candidate(["x", ["yz"]]) == ["x", "yz"]
    assert candidate([None, {"a": 1}, [0.5]]) == [None, {"a": 1}, 0.5]
'''),

task("JBC1/py/008", "lists", "rotate_left", r'''
def rotate_left(items: list, k: int) -> list:
    """Return a new list rotated left by k positions. k may be larger than the
    list length, and a negative k rotates right. An empty list stays empty.

    >>> rotate_left([1, 2, 3, 4, 5], 2)
    [3, 4, 5, 1, 2]
    """
''', r'''
    if not items:
        return []
    k %= len(items)
    return items[k:] + items[:k]
''', r'''
def check(candidate):
    assert candidate([1, 2, 3, 4, 5], 2) == [3, 4, 5, 1, 2]
    assert candidate([1, 2, 3, 4, 5], 7) == [3, 4, 5, 1, 2]
    assert candidate([1, 2, 3, 4, 5], -1) == [5, 1, 2, 3, 4]
    assert candidate([1, 2, 3], 0) == [1, 2, 3]
    assert candidate([], 3) == []
    data = [1, 2, 3]
    candidate(data, 1)
    assert data == [1, 2, 3]
'''),

task("JBC1/py/009", "lists", "dedupe_keep_order", r'''
def dedupe_keep_order(items: list) -> list:
    """Remove duplicates, keeping the first occurrence of each item in order.
    Items are hashable.

    >>> dedupe_keep_order([3, 1, 3, 2, 1])
    [3, 1, 2]
    """
''', r'''
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
''', r'''
def check(candidate):
    assert candidate([3, 1, 3, 2, 1]) == [3, 1, 2]
    assert candidate([]) == []
    assert candidate(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]
    assert candidate([1, 1, 1]) == [1]
    assert candidate([(1, 2), (1, 2), (2, 1)]) == [(1, 2), (2, 1)]
    assert candidate(list(range(50)) * 2) == list(range(50))
'''),

task("JBC1/py/010", "lists", "second_largest_distinct", r'''
def second_largest_distinct(numbers: list[int]):
    """Return the second largest distinct value, or None if there are fewer
    than two distinct values.

    >>> second_largest_distinct([4, 9, 9, 2])
    4
    """
''', r'''
    distinct = sorted(set(numbers), reverse=True)
    return distinct[1] if len(distinct) >= 2 else None
''', r'''
def check(candidate):
    assert candidate([4, 9, 9, 2]) == 4
    assert candidate([]) is None
    assert candidate([7]) is None
    assert candidate([5, 5, 5]) is None
    assert candidate([-1, -5, -3]) == -3
    assert candidate([1, 2]) == 1
'''),

task("JBC1/py/011", "dicts", "invert_dict", r'''
def invert_dict(mapping: dict) -> dict:
    """Map each value to a sorted list of the keys that had that value.

    >>> invert_dict({"a": 1, "b": 2, "c": 1})
    {1: ['a', 'c'], 2: ['b']}
    """
''', r'''
    out = {}
    for key, value in mapping.items():
        out.setdefault(value, []).append(key)
    return {value: sorted(keys) for value, keys in out.items()}
''', r'''
def check(candidate):
    assert candidate({"a": 1, "b": 2, "c": 1}) == {1: ["a", "c"], 2: ["b"]}
    assert candidate({}) == {}
    assert candidate({"z": 0, "y": 0, "x": 0}) == {0: ["x", "y", "z"]}
    assert candidate({3: "q", 1: "q", 2: "r"}) == {"q": [1, 3], "r": [2]}
'''),

task("JBC1/py/012", "dicts", "merge_counts", r'''
def merge_counts(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    """Add two count dictionaries key by key. Keys whose total is zero are
    dropped. The inputs are not modified.

    >>> merge_counts({"x": 2, "y": 1}, {"y": -1, "z": 4})
    {'x': 2, 'z': 4}
    """
''', r'''
    out = dict(a)
    for key, value in b.items():
        out[key] = out.get(key, 0) + value
    return {k: v for k, v in out.items() if v != 0}
''', r'''
def check(candidate):
    assert candidate({"x": 2, "y": 1}, {"y": -1, "z": 4}) == {"x": 2, "z": 4}
    assert candidate({}, {}) == {}
    assert candidate({"a": 1}, {}) == {"a": 1}
    assert candidate({}, {"a": -3}) == {"a": -3}
    a, b = {"k": 1}, {"k": 2}
    assert candidate(a, b) == {"k": 3}
    assert a == {"k": 1} and b == {"k": 2}
'''),

task("JBC1/py/013", "dicts", "group_by_first_letter", r'''
def group_by_first_letter(words: list[str]) -> dict[str, list[str]]:
    """Group words by their lowercased first character, keeping input order
    within each group. Empty strings are skipped.

    >>> group_by_first_letter(["apple", "Bob", "avocado", "", "banana"])
    {'a': ['apple', 'avocado'], 'b': ['Bob', 'banana']}
    """
''', r'''
    groups = {}
    for word in words:
        if word:
            groups.setdefault(word[0].lower(), []).append(word)
    return groups
''', r'''
def check(candidate):
    assert candidate(["apple", "Bob", "avocado", "", "banana"]) == {
        "a": ["apple", "avocado"], "b": ["Bob", "banana"]}
    assert candidate([]) == {}
    assert candidate(["", ""]) == {}
    assert candidate(["Zed", "zoo", "Zulu"]) == {"z": ["Zed", "zoo", "Zulu"]}
    assert candidate(["1st", "2nd"]) == {"1": ["1st"], "2": ["2nd"]}
'''),

task("JBC1/py/014", "dicts", "deep_get", r'''
def deep_get(data, path: str, default=None):
    """Follow a dot-separated path through nested dicts and lists. Path parts
    that index into a list are decimal integers. Return default if any step
    is missing or invalid.

    >>> deep_get({"a": {"b": [10, {"c": 3}]}}, "a.b.1.c")
    3
    """
''', r'''
    current = data
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
        elif isinstance(current, list):
            if not part.isdigit() or int(part) >= len(current):
                return default
            current = current[int(part)]
        else:
            return default
    return current
''', r'''
def check(candidate):
    data = {"a": {"b": [10, {"c": 3}]}, "x": None, "n": 0}
    assert candidate(data, "a.b.1.c") == 3
    assert candidate(data, "a.b.0") == 10
    assert candidate(data, "a.b.2") is None
    assert candidate(data, "a.b.x", "d") == "d"
    assert candidate(data, "a.z", 5) == 5
    assert candidate(data, "n") == 0
    assert candidate(data, "x", "d") is None
    assert candidate(data, "n.q", "d") == "d"
    assert candidate(data, "a.b.-1", "d") == "d"
'''),

task("JBC1/py/015", "dicts", "word_frequencies", r'''
import re


def word_frequencies(text: str, top_n: int) -> list[tuple[str, int]]:
    """Count words in text and return the top_n (word, count) pairs.

    Words are maximal runs of the characters a-z and apostrophes after
    lowercasing. Sort by descending count, then alphabetically.

    >>> word_frequencies("The cat and the hat. THE end!", 2)
    [('the', 3), ('and', 1)]
    """
''', r'''
    counts = {}
    for word in re.findall(r"[a-z']+", text.lower()):
        counts[word] = counts.get(word, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
''', r'''
def check(candidate):
    assert candidate("The cat and the hat. THE end!", 2) == [("the", 3), ("and", 1)]
    assert candidate("", 3) == []
    assert candidate("b a c", 5) == [("a", 1), ("b", 1), ("c", 1)]
    assert candidate("don't DON'T stop", 1) == [("don't", 2)]
    assert candidate("x1x 22 y", 10) == [("x", 2), ("y", 1)]
    assert candidate("a a b", 0) == []
'''),

task("JBC1/py/016", "control_flow", "is_balanced", r'''
def is_balanced(text: str) -> bool:
    """Return True if the brackets (), [] and {} in text are balanced and
    properly nested. All other characters are ignored.

    >>> is_balanced("f(a[1], {b: 2})")
    True
    """
''', r'''
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in text:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack
''', r'''
def check(candidate):
    assert candidate("f(a[1], {b: 2})") is True
    assert candidate("") is True
    assert candidate("no brackets") is True
    assert candidate("(]") is False
    assert candidate("([)]") is False
    assert candidate("((") is False
    assert candidate("))((") is False
    assert candidate("{[()()]}") is True
'''),

task("JBC1/py/017", "parsing", "parse_duration", r'''
def parse_duration(text: str) -> int:
    """Parse a duration such as "1h30m15s" into seconds.

    Units are h, m and s; each may appear at most once and in that order,
    and at least one must appear. Anything else raises ValueError.

    >>> parse_duration("1h30m15s")
    5415
    """
''', r'''
    import re
    match = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", text)
    if not text or match is None:
        raise ValueError(f"invalid duration: {text!r}")
    hours, minutes, seconds = (int(g) if g else 0 for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds
''', r'''
def check(candidate):
    assert candidate("1h30m15s") == 5415
    assert candidate("45s") == 45
    assert candidate("2h") == 7200
    assert candidate("90m") == 5400
    assert candidate("0s") == 0
    assert candidate("1h5s") == 3605
    for bad in ("", "5", "1s1h", "1x", "h", "1h 2m", "-5s"):
        try:
            candidate(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
'''),

task("JBC1/py/018", "strings", "format_duration", r'''
def format_duration(seconds: int) -> str:
    """Format a non-negative number of seconds.

    Under a minute: "45s". Under an hour: "2m 03s". Otherwise: "1h 02m 03s".
    Minutes and seconds after the leading unit are two digits.

    >>> format_duration(3723)
    '1h 02m 03s'
    """
''', r'''
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"
''', r'''
def check(candidate):
    assert candidate(3723) == "1h 02m 03s"
    assert candidate(0) == "0s"
    assert candidate(59) == "59s"
    assert candidate(60) == "1m 00s"
    assert candidate(123) == "2m 03s"
    assert candidate(3600) == "1h 00m 00s"
    assert candidate(90061) == "25h 01m 01s"
'''),

task("JBC1/py/019", "math", "next_palindrome", r'''
def next_palindrome(n: int) -> int:
    """Return the smallest palindromic integer strictly greater than n (n >= 0).

    >>> next_palindrome(123)
    131
    """
''', r'''
    candidate = n + 1
    while str(candidate) != str(candidate)[::-1]:
        candidate += 1
    return candidate
''', r'''
def check(candidate):
    assert candidate(123) == 131
    assert candidate(0) == 1
    assert candidate(9) == 11
    assert candidate(99) == 101
    assert candidate(131) == 141
    assert candidate(808) == 818
    assert candidate(1991) == 2002
'''),

task("JBC1/py/020", "math", "lcm_range", r'''
def lcm_range(a: int, b: int) -> int:
    """Return the least common multiple of all integers from a to b inclusive,
    where 1 <= a <= b.

    >>> lcm_range(1, 10)
    2520
    """
''', r'''
    from math import gcd
    result = 1
    for k in range(a, b + 1):
        result = result * k // gcd(result, k)
    return result
''', r'''
def check(candidate):
    assert candidate(1, 10) == 2520
    assert candidate(5, 5) == 5
    assert candidate(4, 6) == 60
    assert candidate(1, 1) == 1
    assert candidate(7, 9) == 504
    assert candidate(1, 20) == 232792560
'''),
]
