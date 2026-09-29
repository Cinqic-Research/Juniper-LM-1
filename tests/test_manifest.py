import json
import os
import copy

import pytest

from juniper_lm1.baseline.manifest import build_manifest, main, verify


@pytest.fixture
def ckpt(tmp_path):
    root = tmp_path / "ckpt"
    (root / "sub").mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"model_type": "gpt2", "n_layer": 12}))
    (root / "sub" / "weights.bin").write_bytes(b"\x00\x01\x02")
    return root


def test_roundtrip_verifies(ckpt):
    m = build_manifest(ckpt, "t", "")
    assert m["file_count"] == 2
    assert verify(ckpt, m) == []


def test_detects_content_change(ckpt):
    m = build_manifest(ckpt, "t", "")
    (ckpt / "sub" / "weights.bin").write_bytes(b"\x00\x01\x03")
    assert any("content changed" in p for p in verify(ckpt, m))


def test_detects_mtime_only_change(ckpt):
    m = build_manifest(ckpt, "t", "")
    os.utime(ckpt / "config.json", ns=(0, 0))
    assert verify(ckpt, m) == ["mtime changed: config.json"]
    assert verify(ckpt, m, check_mtime=False) == []


def test_detects_added_and_missing(ckpt):
    m = build_manifest(ckpt, "t", "")
    (ckpt / "extra.txt").write_text("x")
    (ckpt / "config.json").unlink()
    problems = verify(ckpt, m)
    assert "missing: config.json" in problems and "unexpected: extra.txt" in problems


def test_refuses_to_write_inside_baseline(ckpt):
    with pytest.raises(SystemExit):
        main(["build", str(ckpt), "--baseline-id", "t", "--out", str(ckpt / "m.json")])


def test_detects_added_symlink(ckpt):
    m = build_manifest(ckpt, "t", "")
    (ckpt / "link").symlink_to(ckpt / "config.json")
    assert verify(ckpt, m) == ["unexpected symlink: link"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("file_count", 99),
        ("total_bytes", 99),
        ("directory_digest_sha256", "0" * 64),
        ("identity", {"config": {"model_type": "wrong"}}),
    ],
)
def test_detects_inconsistent_manifest_summary(ckpt, field, value):
    manifest = build_manifest(ckpt, "t", "")
    manifest[field] = value
    assert any(field in problem for problem in verify(ckpt, manifest))


def test_detects_baseline_id_mismatch_when_expected_id_is_supplied(ckpt):
    manifest = build_manifest(ckpt, "original", "")
    assert verify(ckpt, manifest, expected_baseline_id="working-copy") == [
        "baseline_id does not match configured baseline"
    ]


def test_detects_duplicate_manifest_path(ckpt):
    manifest = build_manifest(ckpt, "t", "")
    manifest["files"].append(copy.deepcopy(manifest["files"][0]))
    assert any("duplicate manifest path" in problem for problem in verify(ckpt, manifest))


def test_manifest_is_portable_across_absolute_locations(ckpt, tmp_path):
    manifest = build_manifest(ckpt, "t", "")
    other_root = tmp_path / "other-parent" / ckpt.name
    other_root.parent.mkdir()
    import shutil
    shutil.copytree(ckpt, other_root)
    assert verify(other_root, manifest, check_mtime=False) == []
