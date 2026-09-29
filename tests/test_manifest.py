import json
import os

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
