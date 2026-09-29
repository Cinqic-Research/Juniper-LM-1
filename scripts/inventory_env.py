"""Record the FLOWBOX hardware/software inventory to reports/environment/.

Usage: python scripts/inventory_env.py [--python /path/to/venv/python]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from juniper_lm1.paths import REPO_ROOT, load_paths  # noqa: E402

COMMANDS = {
    "hostname": ["hostname"],
    "uname": ["uname", "-a"],
    "os_release": ["cat", "/etc/os-release"],
    "lscpu": ["lscpu"],
    "memory": ["free", "-h"],
    # No UUIDs/serials, no unrelated mounts: this report is published.
    "block_devices": ["lsblk", "-o", "NAME,FSTYPE,SIZE,ROTA,MODEL"],
    "gpu_pci": ["sh", "-c", "lspci | grep -iE 'vga|3d|display'"],
    "nvidia_smi": ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,driver_version",
                   "--format=csv"],
    "cuda_driver_version": ["sh", "-c", "nvidia-smi | grep -o 'CUDA Version: [0-9.]*'"],
    "system_python": ["python3", "--version"],
    "git": ["git", "--version"],
    "sandbox_bwrap": ["sh", "-c", "command -v bwrap && bwrap --version"],
    "sandbox_docker": ["sh", "-c", "command -v docker || echo 'not installed'"],
}

TORCH_PROBE = r"""
import json, platform, torch, transformers, safetensors, tokenizers, accelerate
d = {"python": platform.python_version(), "torch": torch.__version__,
     "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
     "transformers": transformers.__version__, "safetensors": safetensors.__version__,
     "tokenizers": tokenizers.__version__, "accelerate": accelerate.__version__,
     "cuda_available": torch.cuda.is_available()}
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    d.update(gpu=p.name, capability=f"{p.major}.{p.minor}",
             vram_gib=round(p.total_memory / 2**30, 2),
             bf16_supported=torch.cuda.is_bf16_supported(including_emulation=False))
print(json.dumps(d))
"""


def run(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (r.stdout + r.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"<unavailable: {e}>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable, help="interpreter of the training venv")
    args = ap.parse_args()

    paths = load_paths()
    canonical = Path(paths["baseline"]["canonical_dir"])
    out = {k: run(v) for k, v in COMMANDS.items()}
    out["training_env"] = run([args.python, "-c", TORCH_PROBE])
    out["canonical_checkpoint_dir"] = str(canonical)
    out["canonical_checkpoint_listing"] = run(
        ["find", str(canonical), "-type", "f", "-not", "-path", "*/.cache/*",
         "-printf", "%P\t%s\t%TY-%Tm-%Td %TH:%TM:%TS\n"])
    for label, p in (("root_fs", Path("/")), ("workspace", Path(paths["workspace"]["root"]))):
        out[f"{label}_filesystem"] = run(["df", "-hT", str(p)])
    stamp = dt.datetime.now(dt.timezone.utc)
    out["recorded_utc"] = stamp.isoformat()

    dest = REPO_ROOT / "reports" / "environment"
    dest.mkdir(parents=True, exist_ok=True)
    name = f"{out['hostname']}-{stamp:%Y%m%d}"
    (dest / f"{name}.json").write_text(json.dumps(out, indent=2) + "\n")
    md = [f"# Environment inventory: {out['hostname']} ({stamp:%Y-%m-%d})", ""]
    for k, v in out.items():
        md += [f"## {k}", "", "```text", str(v), "```", ""]
    (dest / f"{name}.md").write_text("\n".join(md))
    print(dest / f"{name}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
