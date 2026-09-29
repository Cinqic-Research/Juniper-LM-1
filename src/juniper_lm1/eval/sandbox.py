"""Run untrusted Python (model generations, benchmark references) in a bubblewrap sandbox.

Isolation (see SECURITY.md):
  * all namespaces unshared: no network, private PID/IPC/UTS, unprivileged user
  * filesystem: read-only /usr and the evaluation venv; tmpfs /tmp; nothing else
    from the host (no $HOME, no repositories, no credentials, no SSH/GPG agents)
  * environment cleared; CUDA hidden
  * rlimits via prlimit: address space, CPU seconds, process count, file size
  * wall-clock timeout; the whole process group is killed on expiry
  * the program is passed on stdin, so no host file is ever shared

Generated code is never exec()'d in the evaluating process.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

BWRAP = shutil.which("bwrap") or "/usr/bin/bwrap"
PRLIMIT = shutil.which("prlimit") or "/usr/bin/prlimit"
OUTPUT_LIMIT = 4096


@dataclasses.dataclass(frozen=True)
class Limits:
    wall_s: float = 30.0
    cpu_s: int = 20
    address_space_gib: int = 4   # torch's CPU libraries fail to map below ~4 GiB
    nproc: int = 64
    file_size_mib: int = 64


@dataclasses.dataclass
class Result:
    status: str          # "passed" | "failed" | "timeout"
    returncode: int | None
    duration_s: float
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        return self.status == "passed"


def _command(venv: Path, limits: Limits) -> list[str]:
    return [
        BWRAP, "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
        "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
        "--symlink", "usr/bin", "/bin",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--ro-bind", str(venv), "/venv",
        "--chdir", "/tmp",
        "--setenv", "PATH", "/venv/bin:/usr/bin", "--setenv", "HOME", "/tmp",
        "--setenv", "TMPDIR", "/tmp", "--setenv", "CUDA_VISIBLE_DEVICES", "",
        "--setenv", "OMP_NUM_THREADS", "1", "--setenv", "PYTHONHASHSEED", "0",
        "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        "--", PRLIMIT,
        f"--as={limits.address_space_gib * 1024**3}", f"--cpu={limits.cpu_s}",
        f"--nproc={limits.nproc}", f"--fsize={limits.file_size_mib * 1024**2}",
        "/venv/bin/python", "-I", "-",
    ]


def run_program(source: str, limits: Limits = Limits(), venv: Path | None = None) -> Result:
    """Execute `source` in the sandbox. Exit code 0 == passed."""
    venv = Path(venv or sys.prefix)
    start = time.monotonic()
    proc = subprocess.Popen(_command(venv, limits), stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=True, env={"PATH": "/usr/bin:/bin"})
    try:
        out, err = proc.communicate(source.encode(), timeout=limits.wall_s)
        status = "passed" if proc.returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
        status = "timeout"
    return Result(status, proc.returncode, round(time.monotonic() - start, 3),
                  out.decode(errors="replace")[-OUTPUT_LIMIT:],
                  err.decode(errors="replace")[-OUTPUT_LIMIT:])
