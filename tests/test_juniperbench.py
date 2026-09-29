import json

import pytest

from juniper_lm1.eval.juniperbench import (
    SUITES,
    implementation_constraint_failures,
    load_authoring,
    score,
    truncate_completion,
    verify_suite,
)
from juniper_lm1.eval.sandbox import Limits

FAST = Limits(wall_s=10, cpu_s=5)

COMPLETE = {
    "kind": "complete", "entry_point": "double",
    "prompt": 'def double(x):\n    """Return 2 * x."""\n',
    "test": "def check(candidate):\n    assert candidate(3) == 6\n",
}
WRITE_TESTS = {
    "kind": "write_tests", "entry_point": "test_double",
    "reference_impl": "def double(x):\n    return 2 * x\n",
    "test_header": "def test_double():\n",
    "mutants": ["def double(x):\n    return x + 2\n"],
}


def test_truncate_completion_uses_earliest_stop():
    text = "    return 1\n\ndef other():\n    pass\n# trailing"
    assert truncate_completion(text, ["\n#", "\ndef "]) == "    return 1\n"
    assert truncate_completion("    return 1\n", ["\ndef "]) == "    return 1\n"


def test_complete_task_scoring():
    assert score(COMPLETE, "    return 2 * x\n", FAST)["passed"]
    assert not score(COMPLETE, "    return x\n", FAST)["passed"]
    assert not score(COMPLETE, "    while True:\n        pass\n", Limits(wall_s=2, cpu_s=1))["passed"]


def test_write_tests_requires_killing_mutants():
    assert score(WRITE_TESTS, "    assert double(3) == 6\n", FAST)["passed"]
    # passes on the reference but does not distinguish the mutant (2 + 2 == 2 * 2)
    assert not score(WRITE_TESTS, "    assert double(2) == 4\n", FAST)["passed"]
    # rejects the reference itself
    assert not score(WRITE_TESTS, "    assert double(3) == 5\n", FAST)["passed"]
    assert not score(WRITE_TESTS, "    pass\n", FAST)["passed"]


def test_generation_budget():
    from juniper_lm1.eval.juniperbench import generation_budget
    assert generation_budget(113, {"context_length": 1024, "max_new_tokens": 512}) == 512
    assert generation_budget(113, {"context_length": 1024, "max_new_tokens": None}) == 911
    assert generation_budget(900, {"context_length": 1024, "max_new_tokens": 512}) == 124


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("suite", "wrong-suite"),
        ("task_count", 999),
        ("category_counts", {}),
        ("canary", "wrong-canary"),
        ("tasks_file", "wrong.jsonl"),
        ("protocol_file", "wrong.yaml"),
        ("validation", {}),
    ],
)
def test_verify_suite_rejects_freeze_metadata_tampering(tmp_path, monkeypatch, field, value):
    original = SUITES["v1.1"]["freeze"]
    freeze = json.loads(original.read_text())
    freeze[field] = value
    altered = tmp_path / "FREEZE.json"
    altered.write_text(json.dumps(freeze))
    monkeypatch.setitem(SUITES["v1.1"], "freeze", altered)

    checks = verify_suite("v1.1")
    assert checks["jsonl_sha256"] and checks["task_sha256"] and checks["protocol_sha256"]
    assert checks["authoring_reproduces_jsonl"] and checks["structural_integrity"]
    assert not checks["freeze_metadata"]


def test_historical_v1_freeze_schema_is_still_verified():
    assert verify_suite("v1")["freeze_metadata"]


@pytest.mark.parametrize("field", ["evaluator_sha256", "sandbox_sha256"])
def test_verify_suite_binds_v1_2_execution_source_hashes(tmp_path, monkeypatch, field):
    original = SUITES["v1.2"]["freeze"]
    freeze = json.loads(original.read_text())
    freeze[field] = "0" * 64
    altered = tmp_path / "FREEZE.json"
    altered.write_text(json.dumps(freeze))
    monkeypatch.setitem(SUITES["v1.2"], "freeze", altered)

    checks = verify_suite("v1.2")
    assert all(value for key, value in checks.items() if key != field)
    assert not checks[field]


def test_verify_suite_rejects_boolean_token_counts(tmp_path, monkeypatch):
    original = SUITES["v1.2"]["freeze"]
    freeze = json.loads(original.read_text())
    first_task = next(iter(freeze["validation"]))
    freeze["validation"][first_task]["prompt_tokens"] = True
    altered = tmp_path / "FREEZE.json"
    altered.write_text(json.dumps(freeze))
    monkeypatch.setitem(SUITES["v1.2"], "freeze", altered)

    checks = verify_suite("v1.2")
    assert all(value for key, value in checks.items() if key != "freeze_metadata")
    assert not checks["freeze_metadata"]


def test_method_constraint_aliases_and_attribute_calls_are_rejected():
    task = {
        "kind": "complete",
        "prompt": "def merge(a, b):\n",
        "entry_point": "merge",
        "test": "def check(candidate):\n    assert candidate([1], [2]) == [1, 2]\n",
        "implementation_constraints": {
            "forbidden_calls": ["builtins.sorted"],
            "forbidden_attribute_names": ["sort"],
        },
    }
    assert implementation_constraint_failures(task, "    import builtins as b\n    s = b.sorted\n    return s(a + b)\n")
    assert implementation_constraint_failures(task, "    a.sort()\n    return a + b\n")
    assert implementation_constraint_failures(
        task,
        "    import builtins\n"
        "    return getattr(builtins, 'sorted')(a + b)\n",
    )
    assert implementation_constraint_failures(
        {
            **task,
            "implementation_constraints": {
                "forbidden_imports": ["torch.nn.functional"],
                "forbidden_calls": ["torch.nn.functional.cross_entropy"],
            },
        },
        "    from torch.nn import functional as F\n"
        "    return F.cross_entropy(a, b)\n",
    )
    assert not implementation_constraint_failures(task, "    return a + b\n")
    assert not score(task, "    import builtins as b\n    return b.sorted(a + b)\n")["passed"]


def test_constraints_apply_to_completion_not_problem_statement_or_tests():
    task = {
        "kind": "complete", "entry_point": "loss",
        "prompt": (
            "import torch.nn.functional as F\n\n"
            "def loss(x):\n"
            "    # F.cross_entropy appears here only in the task instructions.\n"
        ),
        "test": "def check(candidate):\n    assert candidate(1) == 1\n",
        "implementation_constraints": {
            "forbidden_imports": ["torch.nn.functional"],
            "forbidden_calls": ["torch.nn.functional.cross_entropy"],
        },
    }
    assert not implementation_constraint_failures(task, "    return x\n")
    assert implementation_constraint_failures(task, "    return F.cross_entropy(x, x)\n")


def test_v1_2_rejects_linear_search_that_passes_v1_tests():
    task = next(t for t in load_authoring("v1.2") if t["task_id"] == "JBC1/py/021")
    linear = "    return next((i for i, x in enumerate(items) if x == target), -1)\n"
    assert not score(task, linear)["passed"]
