import os
import shutil

import pytest

from juniper_lm1.eval.sandbox import Limits, run_program

pytestmark = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap not installed")


def test_pass_and_fail():
    assert run_program("assert 1 + 1 == 2").passed
    r = run_program("assert 1 + 1 == 3")
    assert r.status == "failed" and "AssertionError" in r.stderr


def test_no_network():
    src = ("import socket\n"
           "try:\n    socket.create_connection(('1.1.1.1', 80), timeout=2)\n"
           "except OSError:\n    raise SystemExit(0)\nraise SystemExit(1)\n")
    assert run_program(src).passed


def test_network_namespace_is_private():
    host_net_namespace = os.stat("/proc/self/ns/net").st_ino
    r = run_program("import os\nprint(os.stat('/proc/self/ns/net').st_ino)")
    assert r.passed
    assert r.stdout.strip() != str(host_net_namespace)


def test_host_filesystem_hidden_and_readonly():
    src = ("import os\n"
           "assert not os.path.exists('/home') and not os.path.exists('/media')\n"
           "assert not os.path.exists('/etc/passwd')\n"
           "try:\n    open('/usr/x', 'w')\nexcept OSError:\n    pass\nelse:\n    raise SystemExit(1)\n"
           "open('/tmp/ok', 'w').write('fine')\n")
    assert run_program(src).passed


def test_environment_is_clean():
    src = ("import os\n"
           "bad = [k for k in os.environ if any(s in k for s in ('TOKEN', 'KEY', 'SSH', 'GH_', 'AWS'))]\n"
           "assert not bad, bad\nassert os.environ.get('CUDA_VISIBLE_DEVICES') == ''\n")
    assert run_program(src).passed


def test_wall_timeout():
    r = run_program("while True:\n    pass", Limits(wall_s=2, cpu_s=10))
    assert r.status in ("timeout", "failed") and not r.passed


def test_memory_limit():
    assert not run_program("x = bytearray(6 * 1024**3)").passed


def test_tmpfs_is_capped():
    src = ("try:\n    open('/tmp/big', 'wb').write(b'x' * (300 * 1024**2))\n"
           "except OSError:\n    raise SystemExit(0)\nraise SystemExit(1)\n")
    assert run_program(src, Limits(file_size_mib=1024)).passed


def test_hash_seed_is_fixed():
    outs = {run_program('print(hash("juniper"), list({"a", "b", "c", "d"}))').stdout
            for _ in range(3)}
    assert len(outs) == 1


def test_output_volume_is_capped():
    r = run_program("import sys\nwhile True:\n    sys.stdout.write('x' * 65536)\n",
                    Limits(wall_s=20, cpu_s=10, file_size_mib=8))
    assert not r.passed and len(r.stdout) <= 4096
