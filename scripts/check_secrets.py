#!/usr/bin/env python3
"""Fail if staged files appear to contain secrets."""
from __future__ import annotations

import re
import subprocess
import sys
from typing import Iterable

PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z\-_]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}"),
]


def run_git(args: Iterable[str]) -> str:
    return subprocess.check_output(["git", *args], text=True)


def staged_files() -> list[str]:
    output = run_git(["diff", "--cached", "--name-only", "--diff-filter=ACM"]).strip()
    return [line for line in output.splitlines() if line]


def read_staged(path: str) -> str | None:
    try:
        blob = subprocess.check_output(["git", "show", f":{path}"], stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None
    try:
        return blob.decode("utf-8")
    except UnicodeDecodeError:
        return None


def scan_text(path: str, text: str) -> list[str]:
    hits: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pattern in PATTERNS:
            match = pattern.search(line)
            if match:
                hits.append(f"{path}:{lineno}: {match.group(0)}")
    return hits


def tracked_files() -> list[str]:
    output = run_git(["ls-files"]).strip()
    return [line for line in output.splitlines() if line]


def read_tracked(path: str) -> str | None:
    try:
        blob = subprocess.check_output(["git", "show", f"HEAD:{path}"], stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None
    try:
        return blob.decode("utf-8")
    except UnicodeDecodeError:
        return None


def main() -> int:
    # --all scans every tracked file (CI); the default scans the staged diff
    # (pre-commit hook).
    scan_all = "--all" in sys.argv[1:]
    if scan_all:
        targets = [(path, read_tracked(path)) for path in tracked_files()]
        scope = "tracked"
    else:
        targets = [(path, read_staged(path)) for path in staged_files()]
        scope = "staged"

    hits: list[str] = []
    for path, text in targets:
        if text is None:
            continue
        hits.extend(scan_text(path, text))

    if not hits:
        return 0

    print(f"[secrets-check] Potential secrets found in {scope} files:")
    for hit in hits:
        print(f"  - {hit}")
    print("\nRemove secrets or move them to config.local.yaml before committing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
