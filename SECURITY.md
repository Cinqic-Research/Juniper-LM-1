# Security

## Executing model-generated code

Benchmark scoring runs code written by language models. That code is untrusted. It is
never `exec()`'d in the evaluating process; every program runs through
`src/juniper_lm1/eval/sandbox.py`:

| Control | Setting |
|---|---|
| Isolation | `bubblewrap` with all namespaces unshared (no network; private PID/IPC/UTS; unprivileged user) |
| Filesystem | read-only `/usr` and evaluation venv; private tmpfs `/tmp`; no `$HOME`, repositories, `/etc`, credentials, SSH/GPG agents, or host mounts |
| Environment | cleared, apart from `PATH`, `HOME=/tmp`, and thread and seed settings; CUDA hidden |
| Limits | wall clock 30 s (process group killed), CPU 20 s, address space 4 GiB, 64 processes, 64 MiB file size |
| Input | program passed on stdin; no host file is shared |

`tests/test_sandbox.py` checks the network block, the hidden filesystem, the clean
environment, the timeout, and the memory limit. Do not run the evaluator on a machine
where `bwrap` is unavailable; there is deliberately no unsandboxed fallback.

## Reporting a vulnerability

Please do not post exploit details publicly. Open a GitHub issue that says only that
you have a security report, and a maintainer will arrange a private channel.
