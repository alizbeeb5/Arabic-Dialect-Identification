#!/usr/bin/env python3
"""Gracefully stop only local prepare.py workers using the specified data root."""

import argparse
import os
import signal
import time
from pathlib import Path


def find_workers(data_root, proc_root=Path("/proc")):
    wanted = Path(data_root).expanduser().resolve()
    own_namespace = os.readlink(proc_root / "self/ns/pid")
    own_status = (proc_root / "self/status").read_text()
    own_nspid = next((line.split()[1:] for line in own_status.splitlines()
                      if line.startswith("NSpid:")), [])
    matches = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            if os.readlink(entry / "ns/pid") != own_namespace:
                continue
            argv = (entry / "cmdline").read_bytes().decode("utf-8", errors="replace").split("\0")
            if not any(Path(arg).name == "prepare.py" for arg in argv if arg):
                continue
            roots = [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == "--data-root"]
            roots += [arg.split("=", 1)[1] for arg in argv if arg.startswith("--data-root=")]
            if not any(Path(root).expanduser().resolve() == wanted for root in roots):
                continue
            # A zombie has exited and cannot write dataset files.
            status = (entry / "stat").read_text().rsplit(")", 1)[1].strip().split()[0]
            if status != "Z":
                nspid = next((line.split()[1:] for line in (entry / "status").read_text().splitlines()
                              if line.startswith("NSpid:")), [])
                if own_nspid:
                    if len(nspid) < len(own_nspid):
                        continue
                    matches.append(int(nspid[len(own_nspid) - 1]))
                elif (proc_root / "self/stat").read_text().split(" ", 1)[0] == str(os.getpid()):
                    matches.append(int(entry.name))
                else:
                    raise RuntimeError("Cannot safely identify worker PID in this process namespace")
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError):
            continue
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--clear-lock", type=Path,
                        help="After workers exit, remove only this empty preparation lock")
    args = parser.parse_args()
    if not 0 < args.timeout <= 60:
        parser.error("--timeout must be in (0, 60]")
    workers = find_workers(args.data_root)
    for pid in workers:
        print(f"Requesting graceful stop of preparation worker {pid}", flush=True)
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    started = time.monotonic()
    deadline = started + args.timeout
    terminated = False
    while (remaining := find_workers(args.data_root)):
        # Some notebook launchers inherit ignored SIGINT. SIGTERM is then needed.
        # V2 handles SIGTERM with cleanup; legacy code may leave an empty lock.
        if not terminated and time.monotonic() - started >= min(3, args.timeout / 3):
            for pid in remaining:
                print(f"Worker {pid} did not stop on SIGINT; sending SIGTERM", flush=True)
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            terminated = True
        if time.monotonic() >= deadline:
            raise RuntimeError("Worker has not exited yet. No lock was removed; wait and rerun this stop command.")
        time.sleep(0.2)
    print("No matching preparation workers are running.", flush=True)
    if args.clear_lock is not None:
        lock = args.clear_lock.expanduser().resolve()
        allowed = args.data_root.expanduser().resolve() / "reports" / "adi20_micro"
        if not lock.is_relative_to(allowed) or lock.name != ".prepare.lock":
            raise ValueError("--clear-lock must identify a .prepare.lock under this dataset's reports")
        if lock.exists():
            lock.rmdir()  # Never recursively deletes data or nonempty directories.
            print(f"Cleared empty stale lock: {lock}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
