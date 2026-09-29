# Cinqic-style integration tasks: calculator, conversion, finance, history,
# settings and forecasting utilities. Written independently of the Cinqic
# repositories (which are candidate training data) to avoid contamination.
from _schema import task

TASKS = [
task("JBC1/cin/001", "calculator", "evaluate_rpn", r'''
def evaluate_rpn(tokens: list[str]) -> float:
    """Evaluate a reverse Polish notation expression for a calculator.

    Tokens are numbers (such as "3", "-2.5") or the operators + - * / ^
    (^ is exponentiation). Return the result as a float. Division by zero
    raises ZeroDivisionError. An unknown token, too few operands, or leftover
    operands raises ValueError.
    """
''', r'''
    stack = []
    ops = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
           "*": lambda a, b: a * b, "/": lambda a, b: a / b, "^": lambda a, b: a ** b}
    for token in tokens:
        if token in ops:
            if len(stack) < 2:
                raise ValueError("not enough operands")
            b, a = stack.pop(), stack.pop()
            if token == "/" and b == 0:
                raise ZeroDivisionError("division by zero")
            stack.append(float(ops[token](a, b)))
        else:
            try:
                stack.append(float(token))
            except ValueError:
                raise ValueError(f"unknown token {token!r}") from None
    if len(stack) != 1:
        raise ValueError("malformed expression")
    return stack[0]
''', r'''
def check(candidate):
    assert candidate(["3", "4", "+", "2", "*"]) == 14.0
    assert candidate(["2", "3", "^"]) == 8.0
    assert candidate(["10", "4", "-"]) == 6.0
    assert candidate(["-2.5", "2", "*"]) == -5.0
    assert candidate(["7"]) == 7.0
    assert candidate(["1", "4", "/"]) == 0.25
    try:
        candidate(["1", "0", "/"])
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError("expected ZeroDivisionError")
    for bad in (["1", "+"], ["1", "2"], ["1", "x", "+"], []):
        try:
            candidate(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
'''),

task("JBC1/cin/002", "conversion", "convert_length", r'''
FACTORS_TO_METERS = {"mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0,
                     "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344}


def convert_length(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a length between the units in FACTORS_TO_METERS. Unit names
    are case-insensitive. Unknown units raise ValueError naming the unit.
    """
''', r'''
    f, t = from_unit.lower(), to_unit.lower()
    for unit in (f, t):
        if unit not in FACTORS_TO_METERS:
            raise ValueError(f"unknown unit: {unit}")
    return value * FACTORS_TO_METERS[f] / FACTORS_TO_METERS[t]
''', r'''
def check(candidate):
    close = lambda a, b: abs(a - b) < 1e-9 * max(1.0, abs(b))
    assert close(candidate(1, "km", "m"), 1000.0)
    assert close(candidate(12, "in", "ft"), 1.0)
    assert close(candidate(1, "mi", "km"), 1.609344)
    assert close(candidate(5, "M", "CM"), 500.0)
    assert close(candidate(0, "yd", "mm"), 0.0)
    assert close(candidate(3, "ft", "ft"), 3.0)
    try:
        candidate(1, "furlong", "m")
    except ValueError as e:
        assert "furlong" in str(e)
    else:
        raise AssertionError("expected ValueError")
'''),

task("JBC1/cin/003", "finance", "compound_interest", r'''
def compound_interest(principal: float, annual_rate: float, years: float,
                      compounds_per_year: int = 12) -> float:
    """Return the final balance of principal compounded compounds_per_year
    times a year at annual_rate (0.05 means 5%) for `years` years, rounded
    to cents.
    """
''', r'''
    n = compounds_per_year
    return round(principal * (1 + annual_rate / n) ** (n * years), 2)
''', r'''
def check(candidate):
    assert candidate(1000, 0.05, 10, 1) == 1628.89
    assert candidate(1000, 0.05, 10) == 1647.01
    assert candidate(1000, 0.0, 5) == 1000.0
    assert candidate(2500, 0.04, 0) == 2500.0
    assert candidate(100, 0.12, 1, 4) == 112.55
'''),

task("JBC1/cin/004", "finance", "monthly_payment", r'''
def monthly_payment(principal: float, annual_rate: float, months: int) -> float:
    """Fixed monthly payment for an amortized loan, rounded to cents.

    With monthly rate r = annual_rate / 12, payment = principal * r /
    (1 - (1 + r) ** -months). A zero rate means principal / months.
    Raise ValueError if months < 1.
    """
''', r'''
    if months < 1:
        raise ValueError("months must be >= 1")
    r = annual_rate / 12
    if r == 0:
        return round(principal / months, 2)
    return round(principal * r / (1 - (1 + r) ** -months), 2)
''', r'''
def check(candidate):
    assert candidate(200000, 0.06, 360) == 1199.1
    assert candidate(12000, 0.0, 12) == 1000.0
    assert candidate(5000, 0.12, 1) == 5050.0
    assert candidate(10000, 0.05, 60) == 188.71
    try:
        candidate(1000, 0.05, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
'''),

task("JBC1/cin/005", "history", "CalculationHistory", r'''
import json


class CalculationHistory:
    """Bounded calculator history.

    add(expression, result) appends an entry; when more than max_items are
    stored the oldest is dropped. recent(n) returns up to n entries, newest
    first, as (expression, result) tuples. clear() removes everything.
    to_json() and CalculationHistory.from_json(text, max_items) round-trip
    the entries (oldest first) and the order.
    """

    def __init__(self, max_items: int = 100):
''', r'''
        self.max_items = max_items
        self._entries = []

    def add(self, expression, result):
        self._entries.append((expression, result))
        if len(self._entries) > self.max_items:
            del self._entries[: len(self._entries) - self.max_items]

    def recent(self, n):
        return list(reversed(self._entries))[:n]

    def clear(self):
        self._entries.clear()

    def __len__(self):
        return len(self._entries)

    def to_json(self):
        return json.dumps([[e, r] for e, r in self._entries])

    @classmethod
    def from_json(cls, text, max_items=100):
        history = cls(max_items)
        for expression, result in json.loads(text):
            history.add(expression, result)
        return history
''', r'''
def check(candidate):
    h = candidate(max_items=3)
    for i in range(5):
        h.add(f"{i}+{i}", 2 * i)
    assert h.recent(10) == [("4+4", 8), ("3+3", 6), ("2+2", 4)]
    assert h.recent(1) == [("4+4", 8)]
    assert h.recent(0) == []
    restored = candidate.from_json(h.to_json(), 3)
    assert restored.recent(3) == h.recent(3)
    small = candidate.from_json(h.to_json(), 2)
    assert small.recent(5) == [("4+4", 8), ("3+3", 6)]
    h.clear()
    assert h.recent(5) == []
    assert candidate().recent(3) == []
'''),

task("JBC1/cin/006", "settings", "merge_settings", r'''
def merge_settings(defaults: dict, user: dict) -> dict:
    """Merge user settings over defaults and return a new dict.

    Nested dicts are merged recursively. User keys that do not exist in
    defaults are ignored. A user value whose type differs from the default's
    type is ignored (keep the default), except that an int may replace a
    float. Inputs are not modified.
    """
''', r'''
    merged = {}
    for key, default in defaults.items():
        if key not in user:
            merged[key] = merge_settings(default, {}) if isinstance(default, dict) else default
            continue
        value = user[key]
        if isinstance(default, dict):
            merged[key] = merge_settings(default, value) if isinstance(value, dict) \
                else merge_settings(default, {})
        elif type(value) is type(default) or (type(default) is float and type(value) is int):
            merged[key] = value
        else:
            merged[key] = default
    return merged
''', r'''
def check(candidate):
    defaults = {"theme": "light", "precision": 2.0, "sound": True,
                "display": {"font": "mono", "size": 14}}
    user = {"theme": "dark", "precision": 4, "sound": "yes", "unknown": 1,
            "display": {"size": 18, "color": "red"}}
    out = candidate(defaults, user)
    assert out == {"theme": "dark", "precision": 4, "sound": True,
                   "display": {"font": "mono", "size": 18}}
    assert defaults["display"]["size"] == 14 and "color" not in defaults["display"]
    assert candidate(defaults, {}) == defaults
    out["display"]["font"] = "serif"
    assert defaults["display"]["font"] == "mono"
    assert candidate({"n": 1}, {"n": 2.5}) == {"n": 1}
    assert candidate({"d": {"a": 1}}, {"d": 5}) == {"d": {"a": 1}}
'''),

task("JBC1/cin/007", "calculator", "tokenize_expression", r'''
def tokenize_expression(text: str) -> list[str]:
    """Split a calculator expression into tokens: numbers (digits with an
    optional single decimal point, like "3", "0.5", ".5"), the operators
    + - * / ^ and parentheses. Whitespace is skipped. Any other character
    raises ValueError.

    >>> tokenize_expression("3.5*(2 - .5)")
    ['3.5', '*', '(', '2', '-', '.5', ')']
    """
''', r'''
    tokens = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
        elif ch in "+-*/^()":
            tokens.append(ch)
            i += 1
        elif ch.isdigit() or ch == ".":
            start, seen_dot = i, False
            while i < len(text) and (text[i].isdigit() or (text[i] == "." and not seen_dot)):
                seen_dot |= text[i] == "."
                i += 1
            number = text[start:i]
            if number == ".":
                raise ValueError("lone decimal point")
            tokens.append(number)
        else:
            raise ValueError(f"unexpected character {ch!r}")
    return tokens
''', r'''
def check(candidate):
    assert candidate("3.5*(2 - .5)") == ["3.5", "*", "(", "2", "-", ".5", ")"]
    assert candidate("") == []
    assert candidate("  42  ") == ["42"]
    assert candidate("2^10/4") == ["2", "^", "10", "/", "4"]
    for bad in ("2 $ 3", "a+1", "3 % 2"):
        try:
            candidate(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
'''),

task("JBC1/cin/008", "calculator", "evaluate_expression", r'''
def evaluate_expression(text: str) -> float:
    """Safely evaluate an arithmetic expression with + - * /, parentheses,
    unary minus and decimal numbers, returning a float. * and / bind tighter
    than + and -, and operators of equal precedence are left-associative.
    Never use eval() or exec(). Malformed input or any other character raises
    ValueError; division by zero raises ZeroDivisionError.
    """
''', r'''
    import re
    tokens = re.findall(r"\d+\.?\d*|\.\d+|[-+*/()]|\S", text)
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def take():
        nonlocal pos
        pos += 1
        return tokens[pos - 1]

    def expr():
        value = term()
        while peek() in ("+", "-"):
            op, rhs = take(), term()
            value = value + rhs if op == "+" else value - rhs
        return value

    def term():
        value = factor()
        while peek() in ("*", "/"):
            op, rhs = take(), factor()
            if op == "/" and rhs == 0:
                raise ZeroDivisionError("division by zero")
            value = value * rhs if op == "*" else value / rhs
        return value

    def factor():
        token = peek()
        if token == "-":
            take()
            return -factor()
        if token == "(":
            take()
            value = expr()
            if peek() != ")":
                raise ValueError("missing ')'")
            take()
            return value
        if token is not None and re.fullmatch(r"\d+\.?\d*|\.\d+", token):
            return float(take())
        raise ValueError(f"unexpected token {token!r}")

    result = expr()
    if pos != len(tokens):
        raise ValueError("unexpected trailing input")
    return result
''', r'''
def check(candidate):
    assert candidate("1 + 2 * 3") == 7.0
    assert candidate("(1 + 2) * 3") == 9.0
    assert candidate("10 - 4 - 3") == 3.0
    assert candidate("8 / 4 / 2") == 1.0
    assert candidate("-3 + 5") == 2.0
    assert candidate("2 * -(1.5 + .5)") == -4.0
    assert candidate("  7 ") == 7.0
    try:
        candidate("1 / (2 - 2)")
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError("expected ZeroDivisionError")
    for bad in ("", "1 +", "(1 + 2", "1 2", "__import__('os')", "2 ** 3", "abs(-1)"):
        try:
            candidate(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
'''),

task("JBC1/cin/009", "forecast_metrics", "error_report", r'''
def error_report(predictions: list[float], actuals: list[float]) -> dict:
    """Summarize forecast errors. Return a dict with "mae" (mean absolute
    error), "rmse" (root mean squared error), "max_abs" (largest absolute
    error) and "bias" (mean of prediction - actual), each rounded to 4
    decimals. Raise ValueError if the lists are empty or differ in length.
    """
''', r'''
    if not predictions or len(predictions) != len(actuals):
        raise ValueError("need two non-empty lists of equal length")
    errors = [p - a for p, a in zip(predictions, actuals)]
    n = len(errors)
    return {
        "mae": round(sum(abs(e) for e in errors) / n, 4),
        "rmse": round((sum(e * e for e in errors) / n) ** 0.5, 4),
        "max_abs": round(max(abs(e) for e in errors), 4),
        "bias": round(sum(errors) / n, 4),
    }
''', r'''
def check(candidate):
    out = candidate([2.0, 4.0, 6.0], [1.0, 5.0, 6.0])
    assert out == {"mae": 0.6667, "rmse": 0.8165, "max_abs": 1.0, "bias": 0.0}
    assert candidate([1.0], [1.0]) == {"mae": 0.0, "rmse": 0.0, "max_abs": 0.0, "bias": 0.0}
    assert candidate([3, 3], [1, 1])["bias"] == 2.0
    for p, a in (([], []), ([1.0], [1.0, 2.0])):
        try:
            candidate(p, a)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
'''),

task("JBC1/cin/010", "forecast_torch", "rolling_forecast", r'''
import torch


def rolling_forecast(model, history: torch.Tensor, steps: int, window: int) -> torch.Tensor:
    """Forecast `steps` future values of a 1D series autoregressively.

    At each step, feed the last `window` values (shape (1, window)) to model,
    which returns shape (1, 1); append the prediction to the series and
    repeat. Run in eval mode without gradients and restore the model's
    previous mode. Return the predictions as a 1D tensor of length steps.
    """
''', r'''
    was_training = model.training
    model.eval()
    series = history.clone().float()
    preds = []
    with torch.no_grad():
        for _ in range(steps):
            nxt = model(series[-window:].unsqueeze(0)).reshape(())
            preds.append(nxt)
            series = torch.cat([series, nxt.reshape(1)])
    model.train(was_training)
    return torch.stack(preds) if preds else torch.empty(0)
''', r'''
def check(candidate):
    import torch.nn as nn
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 1.0]]))
    model.train()
    out = candidate(model, torch.tensor([1.0, 1.0]), 5, 2)
    assert out.tolist() == [2.0, 3.0, 5.0, 8.0, 13.0]
    assert model.training and not out.requires_grad
    avg = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        avg.weight.fill_(1 / 3)
    avg.eval()
    res = candidate(avg, torch.tensor([9.0, 3.0, 6.0, 3.0]), 1, 3)
    assert torch.allclose(res, torch.tensor([4.0]))
    assert not avg.training
    assert candidate(avg, torch.tensor([1.0, 2.0, 3.0]), 0, 3).shape == (0,)
'''),
]

TASKS[-1]["requires"] = ["torch"]
