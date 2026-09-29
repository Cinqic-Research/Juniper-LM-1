"""`juniper` command-line entry point. Subcommands are added as phases land."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from juniper_lm1.paths import REPO_ROOT, load_paths

MANIFEST = REPO_ROOT / "reports/baseline/gpt2-124m-original-flowbox.manifest.json"


def _baseline(args: argparse.Namespace) -> int:
    from juniper_lm1.baseline.manifest import verify

    paths = load_paths()
    manifest = json.loads(MANIFEST.read_text())
    status = 0
    targets = [("canonical", paths["baseline"]["canonical_dir"], True),
               ("working copy", paths["workspace"]["baseline_copy_dir"], False)]
    for label, root, check_mtime in targets:
        problems = verify(Path(root), manifest, check_mtime=check_mtime)
        print(f"{label}: {'VERIFIED' if not problems else 'FAILED'}")
        for p in problems:
            print(f"  {p}")
        status |= bool(problems)
    return status


def _convert(args: argparse.Namespace) -> int:
    from juniper_lm1.baseline.convert import main as convert_main

    return convert_main(["--force"] if args.force else [])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="juniper")
    sub = ap.add_subparsers(dest="group", required=True)
    b = sub.add_parser("baseline").add_subparsers(dest="cmd", required=True)
    b.add_parser("verify", help="verify canonical baseline and working copy against the manifest"
                 ).set_defaults(fn=_baseline)
    c = b.add_parser("convert", help="build B0 from the working copy and run the equivalence gate")
    c.add_argument("--force", action="store_true")
    c.set_defaults(fn=_convert)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
