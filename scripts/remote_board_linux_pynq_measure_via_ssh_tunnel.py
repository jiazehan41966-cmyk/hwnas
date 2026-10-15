#!/usr/bin/env python3
"""Run a ZCU104 PYNQ/XRT/INA226 operator measurement through an OpenSSH tunnel.

This launcher avoids re-asking for the experiment-server password. OpenSSH owns
the server hop, using the local SSH/askpass setup. Only the board Linux password
is read, from ``ZCU104_BOARD_PASSWORD`` or a local prompt.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import importlib.util
import json
import os
import posixpath
import re
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import paramiko


ROOT = Path(__file__).resolve().parents[1]
LEGACY_SCRIPT = ROOT / "scripts" / "remote_board_linux_pynq_skip_power_measure.py"

SERVER_HOST = "10.70.23.121"
SERVER_PORT = 7889
SERVER_USER = "jiajia"
BOARD_HOSTS = ("192.168.1.100", "192.168.31.217")
BOARD_USER = "xilinx"
DEFAULT_SERVER_OUT_ROOT = "/home/jiajia/experiments/zcu104_full_campaign_v1/board_evidence"
DEFAULT_LOCAL_OUT_ROOT = (
    ROOT
    / "artifacts"
    / "zcu104_full_campaign_v1"
    / "operator_cost_lut"
    / "fourstage_cfem20_v1"
    / "new_board_evidence"
)


def load_board_script() -> str:
    spec = importlib.util.spec_from_file_location("zcu104_legacy_pynq_measure", LEGACY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {LEGACY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return str(module.BOARD_SCRIPT)


def read_secret(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    if sys.stdin.isatty():
        value = getpass.getpass(f"{name}: ")
    else:
        print(f"{name}: ", end="", flush=True)
        value = sys.stdin.readline().rstrip("\n")
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "operator"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_port(port: int, process: subprocess.Popen[bytes], timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
            raise RuntimeError(f"SSH tunnel exited early rc={process.returncode}: {stderr}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"SSH tunnel did not open localhost:{port}")


def start_tunnel(board_host: str, local_port: int) -> subprocess.Popen[bytes]:
    command = [
        "ssh",
        "-p",
        str(SERVER_PORT),
        "-o",
        "BatchMode=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-N",
        "-L",
        f"127.0.0.1:{local_port}:{board_host}:22",
        f"{SERVER_USER}@{SERVER_HOST}",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wait_for_port(local_port, process)
    return process


def connect_board(local_port: int, password: str) -> paramiko.SSHClient:
    board = paramiko.SSHClient()
    board.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    board.connect(
        "127.0.0.1",
        port=local_port,
        username=BOARD_USER,
        password=password,
        timeout=10,
        banner_timeout=10,
        auth_timeout=10,
    )
    return board


def run_server_command(*remote_args: str, input_bytes: bytes | None = None) -> bytes:
    command = [
        "ssh",
        "-p",
        str(SERVER_PORT),
        "-o",
        "BatchMode=yes",
        f"{SERVER_USER}@{SERVER_HOST}",
        *remote_args,
    ]
    result = subprocess.run(command, input=input_bytes, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "server command failed: "
            + " ".join(shlex.quote(arg) for arg in remote_args)
            + "\n"
            + result.stderr.decode(errors="replace")
        )
    return result.stdout


def remote_sha256(path: str) -> str:
    out = run_server_command("sha256sum", path).decode(errors="replace").strip()
    return out.split()[0]


def remote_size(path: str) -> int:
    out = run_server_command("stat", "-c", "%s", path).decode(errors="replace").strip()
    return int(out)


def ensure_board_dir(board: paramiko.SSHClient, path: str) -> None:
    command = f"mkdir -p {shlex.quote(path)}"
    _, stdout, stderr = board.exec_command(command, timeout=20)
    rc = stdout.channel.recv_exit_status()
    if rc != 0:
        raise RuntimeError(stderr.read().decode(errors="replace"))


def board_sha256(board: paramiko.SSHClient, path: str) -> str | None:
    command = f"sha256sum {shlex.quote(path)} 2>/dev/null || true"
    _, stdout, _ = board.exec_command(command, timeout=20)
    text = stdout.read().decode(errors="replace").strip()
    return text.split()[0] if text else None


def copy_server_to_board(board: paramiko.SSHClient, src: str, dst: str) -> dict[str, Any]:
    src_sha = remote_sha256(src)
    size = remote_size(src)
    observed = board_sha256(board, dst)
    if observed == src_sha:
        return {"src": src, "dst": dst, "bytes": size, "sha256": src_sha, "copied": False}
    sftp = board.open_sftp()
    tmp = dst + ".tmp"
    try:
        process = subprocess.Popen(
            [
                "ssh",
                "-p",
                str(SERVER_PORT),
                "-o",
                "BatchMode=yes",
                f"{SERVER_USER}@{SERVER_HOST}",
                "cat",
                src,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None
        digest = hashlib.sha256()
        with sftp.open(tmp, "wb") as handle:
            while True:
                block = process.stdout.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
                handle.write(block)
        stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
        rc = process.wait()
        if rc != 0:
            raise RuntimeError(f"server cat failed rc={rc}: {stderr}")
        if digest.hexdigest() != src_sha:
            raise RuntimeError(f"copied SHA mismatch for {src}")
        _, stdout, stderr_b = board.exec_command(
            f"mv -f {shlex.quote(tmp)} {shlex.quote(dst)}", timeout=20
        )
        rc = stdout.channel.recv_exit_status()
        if rc != 0:
            raise RuntimeError(stderr_b.read().decode(errors="replace"))
        return {"src": src, "dst": dst, "bytes": size, "sha256": src_sha, "copied": True}
    finally:
        sftp.close()


def write_board_file(board: paramiko.SSHClient, path: str, data: bytes) -> None:
    sftp = board.open_sftp()
    try:
        with sftp.open(path, "wb") as handle:
            handle.write(data)
    finally:
        sftp.close()


def read_board_file(board: paramiko.SSHClient, path: str) -> bytes:
    sftp = board.open_sftp()
    try:
        with sftp.open(path, "rb") as handle:
            return handle.read()
    finally:
        sftp.close()


def write_server_file(path: str, data: bytes) -> None:
    directory = posixpath.dirname(path)
    remote_command = f"mkdir -p {shlex.quote(directory)} && cat > {shlex.quote(path)}"
    run_server_command("sh", "-lc", remote_command, input_bytes=data)


def run_board_measurement(
    board: paramiko.SSHClient,
    *,
    board_script: str,
    board_script_path: str,
    board_out: str,
    board_bit: str,
    board_vectors: str | None,
    args: argparse.Namespace,
) -> tuple[int, str, str]:
    write_board_file(board, board_script_path, board_script.encode("utf-8"))
    command = [
        "python3",
        board_script_path,
        "--out-dir",
        board_out,
        "--bit",
        board_bit,
        "--iterations",
        str(args.iterations),
        "--warmup",
        str(args.warmup),
        "--sample-every",
        str(args.sample_every),
        "--idle-samples",
        str(args.idle_samples),
        "--operator",
        args.operator,
        "--control-mode",
        args.control_mode,
        "--param-write-method",
        args.param_write_method,
    ]
    if board_vectors:
        command.extend(["--vectors", board_vectors])
    quoted = " ".join(shlex.quote(part) for part in command)
    _, stdout, stderr = board.exec_command(quoted, timeout=args.command_timeout_s)
    stdout_text = stdout.read().decode(errors="replace")
    stderr_text = stderr.read().decode(errors="replace")
    rc = stdout.channel.recv_exit_status()
    return rc, stdout_text, stderr_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator", required=True)
    parser.add_argument("--server-bit", required=True)
    parser.add_argument("--server-hwh", required=True)
    parser.add_argument("--server-vectors")
    parser.add_argument("--control-mode", choices=("wrapper", "hls_ap_ctrl"), default="hls_ap_ctrl")
    parser.add_argument("--param-write-method", choices=("array", "word"), default="word")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--idle-samples", type=int, default=25)
    parser.add_argument("--command-timeout-s", type=int, default=900)
    parser.add_argument("--server-out-root", default=DEFAULT_SERVER_OUT_ROOT)
    parser.add_argument("--local-out-root", type=Path, default=DEFAULT_LOCAL_OUT_ROOT)
    parser.add_argument("--board-host", choices=BOARD_HOSTS)
    args = parser.parse_args()

    board_password = read_secret("ZCU104_BOARD_PASSWORD")
    board_script = load_board_script()
    driver_sha = sha256_file(Path(__file__).resolve())
    board_script_sha = sha256_bytes(board_script.encode("utf-8"))
    name = safe_name(args.operator)
    run_id = time.strftime(f"pynq_{name}_full_sampling_%Y%m%d_%H%M%S")
    server_out = posixpath.join(args.server_out_root, run_id)
    board_out = posixpath.join("/home/xilinx/zcu104_measurements", run_id)
    board_script_path = posixpath.join("/home/xilinx", run_id + ".py")
    board_bit = posixpath.join("/home/xilinx", name + ".bit")
    board_hwh = posixpath.join("/home/xilinx", name + ".hwh")
    board_vectors = posixpath.join("/home/xilinx", name + "_vectors.npz") if args.server_vectors else None

    last_error: Exception | None = None
    for host in ([args.board_host] if args.board_host else BOARD_HOSTS):
        if host is None:
            continue
        local_port = find_free_port()
        tunnel: subprocess.Popen[bytes] | None = None
        try:
            tunnel = start_tunnel(host, local_port)
            board = connect_board(local_port, board_password)
            try:
                ensure_board_dir(board, board_out)
                copies = [
                    copy_server_to_board(board, args.server_bit, board_bit),
                    copy_server_to_board(board, args.server_hwh, board_hwh),
                ]
                if args.server_vectors and board_vectors:
                    copies.append(copy_server_to_board(board, args.server_vectors, board_vectors))
                rc, stdout_text, stderr_text = run_board_measurement(
                    board,
                    board_script=board_script,
                    board_script_path=board_script_path,
                    board_out=board_out,
                    board_bit=board_bit,
                    board_vectors=board_vectors,
                    args=args,
                )
                summary_bytes = read_board_file(board, posixpath.join(board_out, "summary.json"))
                metadata = {
                    "board_bit": board_bit,
                    "board_host": host,
                    "board_hwh": board_hwh,
                    "board_out": board_out,
                    "board_returncode": rc,
                    "board_script": board_script_path,
                    "board_script_sha256": board_script_sha,
                    "copies": copies,
                    "driver_script": str(Path(__file__).resolve()),
                    "driver_script_sha256": driver_sha,
                    "local_script_sha256": driver_sha,
                    "server_out": server_out,
                    "stdout_tail": stdout_text[-8000:],
                    "stderr_tail": stderr_text[-8000:],
                }
                write_server_file(posixpath.join(server_out, "summary.json"), summary_bytes)
                write_server_file(
                    posixpath.join(server_out, "orchestrator_metadata.json"),
                    json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8"),
                )
                local_out = args.local_out_root / run_id
                local_out.mkdir(parents=True, exist_ok=True)
                (local_out / "summary.json").write_bytes(summary_bytes)
                (local_out / "orchestrator_metadata.json").write_text(
                    json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
                )
                result = {
                    "board_host": host,
                    "board_returncode": rc,
                    "local_out": str(local_out),
                    "server_out": server_out,
                    "summary_sha256": sha256_bytes(summary_bytes),
                }
                print(json.dumps(result, indent=2, sort_keys=True))
                if rc != 0:
                    raise RuntimeError(f"board measurement failed rc={rc}")
                return
            finally:
                board.close()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        finally:
            if tunnel is not None:
                tunnel.terminate()
                try:
                    tunnel.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    tunnel.kill()
    raise RuntimeError(f"could not measure on any board host: {last_error!r}")


if __name__ == "__main__":
    main()
