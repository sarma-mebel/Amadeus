"""Tripwire: named god-file candidates must stay under a line budget.

Plain line counts via pathlib — no frontend tooling, no AST parsing. This
is not a style gate; it's an early-warning system for regrowth. Budgets
are set ~10% above the size measured right after a deliberate extraction
pass (see amadeus/presence.py, split out of amadeus/engine.py, and
amadeus/memory/introspection.py, split out of amadeus/memory/manager.py
in an earlier pass) — so a file can absorb small, honest growth before
tripping, but sustained feature-adding will hit the ceiling and force a
conscious decision rather than quietly regrowing to 1,500+ lines again.

If this test fails: extract a coherent module (see amadeus/presence.py
and memory/introspection.py for the pattern) instead of raising the
budget; raise the budget only with a reviewed reason.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# path -> max line count. One line per file, keep it that way.
BUDGETS = {
    "amadeus/engine.py": 1200,  # post-presence-extraction: 1060
    "amadeus/presence.py": 700,  # extracted from engine.py: 609; post-co_activity-extraction: 356
    "amadeus/co_activity.py": 600,  # read-along + watch-together, extracted from presence.py: 455
    "amadeus/memory/manager.py": 1550,  # post-introspection: 1383; post-recall-extraction: 1394
    "amadeus/memory/recall.py": 245,  # per-turn recall ranking, extracted from manager.py: 221
    "frontend/src/components/ChatInterface.tsx": 1250,  # 1102
    "frontend/src/hooks/usePetChat.ts": 735,  # post-petTurn/useReadTogether-extraction: 666
    "frontend/src/hooks/usePetAmbient.ts": 275,  # extracted from usePetChat.ts: 249
    "frontend/src/components/VRMViewer.tsx": 1630,  # post-vrm-orchestration-extraction: 1479
    "frontend/src/components/PetView.tsx": 1005,  # 913
    "frontend/src/vrm/clipLoader.ts": 900,  # 814
}

BUDGET_MESSAGE = (
    "this file hit its size budget — extract a coherent module (see "
    "amadeus/presence.py and memory/introspection.py for the pattern) "
    "instead of raising the budget; raise the budget only with a "
    "reviewed reason."
)


@pytest.mark.parametrize("relative_path,budget", sorted(BUDGETS.items()))
def test_file_stays_under_line_budget(relative_path: str, budget: int) -> None:
    path = REPO_ROOT / relative_path
    assert path.is_file(), f"budgeted file not found: {relative_path}"
    line_count = sum(1 for _ in path.open("r", encoding="utf-8", errors="replace"))
    assert line_count <= budget, (
        f"{relative_path} is {line_count} lines (budget {budget}) — " + BUDGET_MESSAGE
    )
