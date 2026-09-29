"""Authoring helpers for JuniperBench-Code-v1. The frozen artifact is the built JSONL;
these sources exist so the construction is reviewable."""

from __future__ import annotations

CATEGORIES = {
    "py": "python_fundamentals",
    "pt": "pytorch",
    "dbg": "debugging_repair",
    "test": "testing_edge_cases",
    "cin": "cinqic_integration",
}


def task(tid: str, subcategory: str, entry_point: str, prompt: str, solution: str, test: str,
         requires: tuple[str, ...] = ()) -> dict:
    """A completion task: the model continues `prompt`; `check(candidate)` must pass."""
    prefix = tid.split("/")[1]
    return {"task_id": tid, "category": CATEGORIES[prefix], "subcategory": subcategory,
            "kind": "complete", "entry_point": entry_point, "prompt": prompt.lstrip("\n"),
            "canonical_solution": solution.lstrip("\n"), "test": test.strip() + "\n",
            "requires": list(requires)}


def test_task(tid: str, subcategory: str, entry_point: str, impl: str, test_header: str,
              solution: str, mutants: list[str], requires: tuple[str, ...] = ()) -> dict:
    """A test-writing task: the model completes `test_header` (a test function). It passes
    iff its test passes on `impl` and fails on every mutant implementation."""
    impl = impl.strip("\n") + "\n"
    test_header = test_header.lstrip("\n")
    return {"task_id": tid, "category": CATEGORIES["test"], "subcategory": subcategory,
            "kind": "write_tests", "entry_point": entry_point,
            "prompt": impl + "\n\n" + test_header, "canonical_solution": solution.lstrip("\n"),
            "reference_impl": impl, "test_header": test_header,
            "mutants": [m.strip("\n") + "\n" for m in mutants], "requires": list(requires)}
