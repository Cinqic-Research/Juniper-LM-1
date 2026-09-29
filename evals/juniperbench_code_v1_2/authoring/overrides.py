"""Small, explicit corrections applied to the immutable v1 authoring source."""

from __future__ import annotations


def apply_overrides(tasks: list[dict]) -> list[dict]:
    """Return the v1.2 task set without mutating the v1/v1.1 records."""
    result = [dict(task) for task in tasks]
    by_id = {task["task_id"]: task for task in result}

    # E2: the comparison must normalize both the input words and the exception set.
    title = by_id["JBC1/py/003"]
    title["canonical_solution"] = (
        "    minor = {word.casefold() for word in minor_words}\n"
        "    words = text.split(\" \")\n"
        "    out = []\n"
        "    for i, w in enumerate(words):\n"
        "        if i > 0 and w.casefold() in minor:\n"
        "            out.append(w.lower())\n"
        "        else:\n"
        "            out.append(w.capitalize())\n"
        "    return \" \".join(out)\n"
    )
    title["test"] += (
        "    assert candidate(\"THE LORD OF THE RINGS\", {\"oF\", \"the\"}) == "
        "\"The Lord of the Rings\"\n"
    )

    # E4: make the advertised O(log n) requirement observable through indexed reads.
    binary = by_id["JBC1/py/021"]
    binary["prompt"] = (
        "from collections.abc import Sequence\n\n"
        + binary["prompt"].replace("items: list[int]", "items: Sequence[int]", 1)
    )
    binary["test"] += (
        "\n"
        "    class Probe(Sequence):\n"
        "        def __init__(self, size):\n"
        "            self.values = list(range(0, size * 2, 2))\n"
        "            self.reads = 0\n"
        "        def __len__(self):\n"
        "            return len(self.values)\n"
        "        def __getitem__(self, index):\n"
        "            self.reads += 1\n"
        "            return self.values[index]\n"
        "\n"
        "    probe = Probe(200000)\n"
        "    assert candidate(probe, 123456) == 61728\n"
        "    assert probe.reads <= 19\n"
    )

    # E4: turn the source-level parts of explicit algorithm/API constraints into scored rules.
    by_id["JBC1/py/022"]["implementation_constraints"] = {
        "forbidden_calls": ["builtins.sorted"],
        "forbidden_attribute_names": ["sort"],
    }
    by_id["JBC1/pt/013"]["implementation_constraints"] = {
        "forbidden_ast_nodes": ["For", "While", "AsyncFor", "ListComp", "SetComp",
                                "DictComp", "GeneratorExp"],
    }
    hessian = by_id["JBC1/pt/032"]
    hessian["implementation_constraints"] = {
        "forbidden_calls": ["torch.autograd.functional.hessian",
                             "torch.autograd.functional.jacobian", "torch.func.hessian"],
    }
    hessian["test"] += (
        "\n    # A dense 50,000 x 50,000 Hessian cannot fit the evaluator's 4 GiB limit.\n"
        "    large_x = torch.ones(50000)\n"
        "    large_v = torch.ones_like(large_x)\n"
        "    large = candidate(lambda z: (z ** 3).sum(), large_x, large_v)\n"
        "    assert torch.allclose(large, 6 * large_x * large_v)\n"
    )
    by_id["JBC1/pt/039"]["implementation_constraints"] = {
        "forbidden_calls": ["torch.nn.LayerNorm", "torch.nn.functional.layer_norm"],
    }
    for task_id, function_name in (
        ("JBC1/pt/043", "cross_entropy"),
        ("JBC1/pt/045", "binary_cross_entropy_with_logits"),
    ):
        by_id[task_id]["implementation_constraints"] = {
            "forbidden_imports": ["torch.nn.functional"],
            "forbidden_calls": [f"torch.nn.functional.{function_name}"],
        }
    by_id["JBC1/cin/008"]["implementation_constraints"] = {
        "forbidden_calls": ["builtins.eval", "builtins.exec"],
    }

    # E3: keep output within limit when the requested suffix is itself too long.
    truncate = by_id["JBC1/test/007"]
    doc = (
        "Shorten text to at most `limit` characters, ending with the full suffix when\n"
        "anything was cut. If suffix is longer than limit, use suffix[:limit] as the\n"
        "result. `limit` is non-negative. Text that already fits is returned unchanged."
    )
    impl = (
        "def truncate(text, limit, suffix=\"...\"):\n"
        f"    \"\"\"{doc}\"\"\"\n"
        "    if len(text) <= limit:\n"
        "        return text\n"
        "    if len(suffix) > limit:\n"
        "        return suffix[:limit]\n"
        "    return text[: limit - len(suffix)] + suffix\n"
    )
    header = (
        "def test_truncate():\n"
        "    \"\"\"Assert-based tests for the length, suffix, exact-fit, and custom-suffix rules.\"\"\"\n"
    )
    truncate["reference_impl"] = impl
    truncate["test_header"] = header
    truncate["prompt"] = impl + "\n\n" + header
    truncate["canonical_solution"] = (
        "    assert truncate(\"hello\", 10) == \"hello\"\n"
        "    assert truncate(\"hello\", 5) == \"hello\"\n"
        "    assert truncate(\"hello world\", 8) == \"hello...\"\n"
        "    assert len(truncate(\"abcdefghij\", 6)) == 6\n"
        "    assert truncate(\"abcdef\", 4, suffix=\"~\") == \"abc~\"\n"
        "    assert truncate(\"abcdef\", 2) == \"..\"\n"
        "    assert truncate(\"abcdef\", 2, suffix=\"uvwxyz\") == \"uv\"\n"
    )
    return sorted(result, key=lambda task: task["task_id"])
