"""Amadeus tool-calling and analyzer helpers.

The multi-agent council that used to live here was removed 2026-07-17
(unused in practice; see git history for the last version). What remains
is the single-model chat path's supporting cast: the tool registry
(``tools``/``factory``), the hardened ReAct loop (``tool_loop``), and the
VADER fallback emotion analyzer.
"""
