"""Amadeus Kurisu assistant package."""

from __future__ import annotations

__all__ = ["__version__"]

# Source of truth for the running app's version (/healthz and /api/status
# report this constant, and the frontend's update nudge compares it against
# the latest GitHub release). Bump it together with pyproject.toml and
# frontend/src-tauri/tauri.conf.json — the release workflow's version-sync
# check fails the build on drift.
__version__ = "0.17.0"
