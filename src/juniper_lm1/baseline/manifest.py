"""External, read-only fingerprinting of the canonical GPT-2 baseline.

This module only *reads* the baseline directory. It never writes, renames, chmods,
or touches anything inside it; manifests are always written elsewhere.

Standard library only, so the manifest does not depend on the ML stack.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

MANIFEST_SCHEMA = "juniper-baseline-manifest/1"

FORMATS = {
    ".safetensors": "safetensors (PyTorch/HF weights)",
    ".bin": "PyTorch pickle state_dict (HF legacy)",
    ".h5": "Keras/TensorFlow HDF5 weights",
    ".msgpack": "Flax msgpack weights",
    ".ot": "rust-bert / libtorch tensor archive",
    ".onnx": "ONNX graph + weights",
    ".tflite": "TensorFlow Lite flatbuffer",
    ".json": "JSON (config/tokenizer)",
    ".txt": "text (BPE merges)",
    ".md": "Markdown (upstream model card)",
    ".metadata": "huggingface_hub download metadata",
    ".lock": "huggingface_hub download lock (empty)",
    ".TAG": "cache directory tag",
}


def sha256_file(path: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def _utc(ns: int) -> str:
    return dt.datetime.fromtimestamp(ns / 1e9, dt.timezone.utc).isoformat()


def file_records(root: Path) -> list[dict]:
    records = []
    for p in sorted(root.rglob("*"), key=lambda q: q.relative_to(root).as_posix()):
        if not p.is_file() or p.is_symlink():
            continue
        st = p.stat()
        rel = p.relative_to(root).as_posix()
        records.append(
            {
                "path": rel,
                "size": st.st_size,
                "sha256": sha256_file(p),
                "mtime_ns": st.st_mtime_ns,
                "mtime_utc": _utc(st.st_mtime_ns),
                "format": FORMATS.get(p.suffix, FORMATS.get(p.name, "other")),
            }
        )
    return records


def directory_digest(records: list[dict]) -> str:
    """SHA-256 over the ordered '<path>\\t<size>\\t<sha256>\\n' lines."""
    h = hashlib.sha256()
    for r in sorted(records, key=lambda r: r["path"]):
        h.update(f"{r['path']}\t{r['size']}\t{r['sha256']}\n".encode())
    return h.hexdigest()


def detect_identity(root: Path) -> dict:
    ident: dict = {}
    cfg = root / "config.json"
    if cfg.exists():
        c = json.loads(cfg.read_text())
        ident["config"] = {
            k: c.get(k)
            for k in ("model_type", "architectures", "n_layer", "n_head", "n_embd",
                      "n_positions", "n_ctx", "vocab_size", "bos_token_id", "eos_token_id")
        }
    revisions = set()
    dl = root / ".cache" / "huggingface" / "download"
    if dl.is_dir():
        for m in dl.glob("*.metadata"):
            first = m.read_text().splitlines()[:1]
            if first:
                revisions.add(first[0].strip())
    ident["hf_revisions_in_download_metadata"] = sorted(revisions)
    return ident


def build_manifest(root: Path, baseline_id: str, source_notes: str) -> dict:
    records = file_records(root)
    return {
        "schema": MANIFEST_SCHEMA,
        "baseline_id": baseline_id,
        "canonical_dir": str(root),
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "file_count": len(records),
        "total_bytes": sum(r["size"] for r in records),
        "directory_digest_sha256": directory_digest(records),
        "identity": detect_identity(root),
        "source_notes": source_notes,
        "files": records,
    }


def verify(root: Path, manifest: dict, check_mtime: bool = True) -> list[str]:
    """Return a list of discrepancies (empty list == verified)."""
    problems = []
    current = {r["path"]: r for r in file_records(root)}
    expected = {r["path"]: r for r in manifest["files"]}
    for path in sorted(expected.keys() - current.keys()):
        problems.append(f"missing: {path}")
    for path in sorted(current.keys() - expected.keys()):
        problems.append(f"unexpected: {path}")
    for link in sorted(p for p in root.rglob("*") if p.is_symlink()):
        problems.append(f"unexpected symlink: {link.relative_to(root).as_posix()}")
    for path in sorted(expected.keys() & current.keys()):
        e, c = expected[path], current[path]
        if e["sha256"] != c["sha256"] or e["size"] != c["size"]:
            problems.append(f"content changed: {path}")
        elif check_mtime and e["mtime_ns"] != c["mtime_ns"]:
            problems.append(f"mtime changed: {path}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("root", type=Path)
    b.add_argument("--baseline-id", required=True)
    b.add_argument("--source-notes", default="")
    b.add_argument("--out", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("root", type=Path)
    v.add_argument("--manifest", type=Path, required=True)
    v.add_argument("--ignore-mtime", action="store_true",
                   help="for copies, where content (not metadata) is what must match")
    args = ap.parse_args(argv)

    if args.cmd == "build":
        root = args.root.resolve()
        if args.out.resolve().is_relative_to(root):
            ap.error("manifest must be written outside the baseline directory")
        m = build_manifest(root, args.baseline_id, args.source_notes)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(m, indent=2) + "\n")
        print(f"{m['file_count']} files, digest {m['directory_digest_sha256']}")
        return 0

    manifest = json.loads(args.manifest.read_text())
    problems = verify(args.root.resolve(), manifest, check_mtime=not args.ignore_mtime)
    for p in problems:
        print(p)
    print("VERIFIED" if not problems else f"FAILED ({len(problems)} discrepancies)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
