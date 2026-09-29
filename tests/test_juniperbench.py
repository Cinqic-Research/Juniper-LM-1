import shutil

import pytest

from juniper_lm1.eval.juniperbench import score, truncate_completion
from juniper_lm1.eval.sandbox import Limits

sandboxed = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap not installed")
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


@sandboxed
def test_complete_task_scoring():
    assert score(COMPLETE, "    return 2 * x\n", FAST)["passed"]
    assert not score(COMPLETE, "    return x\n", FAST)["passed"]
    assert not score(COMPLETE, "    while True:\n        pass\n", Limits(wall_s=2, cpu_s=1))["passed"]


@sandboxed
def test_write_tests_requires_killing_mutants():
    assert score(WRITE_TESTS, "    assert double(3) == 6\n", FAST)["passed"]
    # passes on the reference but does not distinguish the mutant (2 + 2 == 2 * 2)
    assert not score(WRITE_TESTS, "    assert double(2) == 4\n", FAST)["passed"]
    # rejects the reference itself
    assert not score(WRITE_TESTS, "    assert double(3) == 5\n", FAST)["passed"]
    assert not score(WRITE_TESTS, "    pass\n", FAST)["passed"]
