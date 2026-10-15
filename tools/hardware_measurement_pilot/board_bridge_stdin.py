"""Large-code stdin adapter around the user's existing local Jupyter bridge.

CLI is compatible with the existing bridge: CODE_FILE LOG_FILE [--keep-kernel].
The existing bridge alone obtains the credential. This adapter forwards its
stdin write unchanged, appends the base64 code on stdin, and executes the same
hash-bound remote executor in-process. No credential/configuration is copied,
printed, persisted, or passed as a command-line argument. The local helper and
remote executor files are not modified.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import runpy
import shlex
import subprocess
import sys


UPSTREAM = Path("E:/Codex/.codex/local-tools/zcu104-ssh/jupyter_reuse_bridge.py")
UPSTREAM_SHA256 = "af76dd1eec75d17144f39be3892b5f4c760c2d098daf5798c194779342ca704f"
EXECUTOR = "/home/jiajia/experiments/detector_execution_20260922_dma_attempt01/jupyter_execute.py"
EXECUTOR_SHA256 = "2d8a071af0c84a516bd6b3609214d275f995bbec8597a1368b7c60d7a27ae1cc"


class _AppendPayload:
    """Forward the upstream authentication write; append code when it closes."""
    def __init__(self, stream, payload):
        self._stream, self._payload, self._sent = stream, payload, False

    def write(self, value):
        # Deliberately do not inspect, retain, parse, or log this value.
        return self._stream.write(value)

    def close(self):
        if self._stream.closed:
            return
        try:
            if not self._sent:
                self._sent = True
                self._stream.write(self._payload)
                self._stream.flush()
                self._payload = None
        finally:
            self._stream.close()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _remote_wrapper(keep_kernel):
    # argv and the shell command contain only a fixed program and known paths.
    # Read the original auth line intact: the upstream executor retains its own
    # CRLF handling. Large code becomes Python sys.argv only inside this process,
    # so neither Windows command limits nor Linux per-argument limits apply.
    return (
        "import hashlib,io,runpy,sys;from pathlib import Path;"
        f"p={EXECUTOR!r};"
        f"assert hashlib.sha256(Path(p).read_bytes()).hexdigest()=={EXECUTOR_SHA256!r},'existing executor changed';"
        "auth=io.StringIO(sys.stdin.readline());payload=sys.stdin.read();"
        f"sys.argv=[p,payload]+{['--keep-kernel'] if keep_kernel else []!r};"
        "sys.stdin=auth;runpy.run_path(p,run_name='__main__');auth.close()"
    )


def run(code_file, log_file, keep_kernel=False):
    if not Path(code_file).is_file():
        raise FileNotFoundError("code file is missing")
    if hashlib.sha256(UPSTREAM.read_bytes()).hexdigest() != UPSTREAM_SHA256:
        raise RuntimeError("Existing local bridge changed; review adapter contract before execution")
    original_popen, original_argv = subprocess.Popen, sys.argv
    adapted = False

    def popen(command, *args, **kwargs):
        nonlocal adapted
        if adapted or not isinstance(command, (list, tuple)):
            raise RuntimeError("Unexpected additional subprocess in existing bridge")
        command = list(command)
        try:
            python_index = command.index("python3")
            executor_index = command.index(EXECUTOR)
        except ValueError as exc:
            raise RuntimeError("Existing bridge subprocess contract changed") from exc
        if command[python_index:executor_index] != ["python3", "-u"]:
            raise RuntimeError("Unexpected remote Python invocation")
        tail = command[executor_index + 1:]
        if len(tail) not in (1, 2) or (len(tail) == 2 and tail[1] != "--keep-kernel"):
            raise RuntimeError("Unexpected remote executor arguments")
        if bool(len(tail) == 2) != bool(keep_kernel):
            raise RuntimeError("Kernel retention argument differs")
        if kwargs.get("stdin") != subprocess.PIPE or kwargs.get("text") is not True:
            raise RuntimeError("Existing bridge stdin framing changed")
        payload = tail[0]
        short_command = command[:python_index] + ["python3", "-u", "-c", shlex.quote(_remote_wrapper(keep_kernel))]
        process = original_popen(short_command, *args, **kwargs)
        process.stdin = _AppendPayload(process.stdin, payload)
        adapted = True
        return process

    try:
        subprocess.Popen = popen
        sys.argv = [str(UPSTREAM), str(Path(code_file).resolve()), str(log_file)]
        if keep_kernel:
            sys.argv.append("--keep-kernel")
        runpy.run_path(str(UPSTREAM), run_name="__main__")
    finally:
        subprocess.Popen, sys.argv = original_popen, original_argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("code_file", type=Path)
    parser.add_argument("log_file")
    parser.add_argument("--keep-kernel", action="store_true")
    args = parser.parse_args()
    run(args.code_file, args.log_file, args.keep_kernel)


if __name__ == "__main__":
    main()
