#!/usr/bin/env bash
# Run frontend vitest on WINDOWS-native node from a WSL shell.
#
# Why: on a /mnt/c checkout, WSL node pays the 9P filesystem tax on every
# module read — a bare require('jsdom') measures 90-115s, blowing vitest's
# hardcoded 60s worker-start limit (workers then get silently DROPPED while
# vitest exits 0; frontend/scripts/run-tests.mjs exists to catch that).
# Windows node on the same checkout reads NTFS natively: measured 2026-08-01,
# the same test file went 133.5s (WSL) -> 1.8s (Windows), 29/29 passing.
#
# Usage (from anywhere in the repo, WSL side):
#   scripts/vitest-win.sh                    # full suite
#   scripts/vitest-win.sh src/vrm/petRoam.test.ts   # one file, extra args ok
#
# The WSL-side `npm run test:ci` drop-guard path still works and stays the
# fallback for environments without Windows interop (CI, native Linux).

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
win_frontend="$(wslpath -w "$repo_root/frontend")"

exec powershell.exe -NoProfile -ExecutionPolicy Bypass -Command \
    "cd '$win_frontend'; npx.cmd vitest run $*"
