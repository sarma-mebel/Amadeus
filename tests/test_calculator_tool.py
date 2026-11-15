"""CalculatorTool hardening: ``eval`` with an empty ``__builtins__`` still
allowed dunder traversal (``().__class__.__mro__...`` reaches every loaded
class), so expressions are now AST-whitelisted before eval sees them. These
tests pin both sides: the supported math surface keeps working, and
everything outside it fails as a ToolResult error — never an exception, and
never an evaluation.
"""

import pytest

from amadeus.agents.tools import CalculatorTool


def _run(expression: str):
    return CalculatorTool().execute(expression=expression)


# ---------------------------------------------------------------------------
# Supported surface stays green
# ---------------------------------------------------------------------------


def test_basic_arithmetic():
    result = _run("2 + 2")
    assert result.success
    assert result.result == 4


def test_operator_mix_precedence_and_unary():
    assert _run("-2 ** 2").result == -4
    assert _run("(7 // 2) % 3").result == 0
    assert _run("10 / 4").result == 2.5


def test_math_functions_and_constants():
    result = _run("sqrt(16) * 3.14")
    assert result.success
    assert result.result == pytest.approx(12.56)
    assert _run("sin(pi / 2)").result == pytest.approx(1.0)
    assert _run("log10(100)").result == pytest.approx(2.0)
    assert _run("round(2.567, 2)").result == pytest.approx(2.57)
    assert _run("factorial(5)").result == 120


def test_list_and_tuple_arguments():
    assert _run("sum([1, 2, 3])").result == 6
    assert _run("mean([1, 2, 3])").result == pytest.approx(2.0)
    assert _run("min(3, 1, 2)").result == 1
    assert _run("max((4, 9), (5, 1))").result == (5, 1)


def test_comparisons_and_boolean_operators():
    assert _run("2 < 3").result is True
    assert _run("1 <= 1 and 2 != 3").result is True
    assert _run("not (2 > 3)").result is True


# ---------------------------------------------------------------------------
# Everything else is rejected before eval
# ---------------------------------------------------------------------------


def _assert_rejected(expression: str):
    result = _run(expression)
    assert not result.success
    assert result.error and "Calculation error" in result.error
    return result


def test_dunder_traversal_is_rejected():
    _assert_rejected("().__class__.__mro__")
    _assert_rejected("(1).__class__")


def test_import_smuggling_is_rejected():
    _assert_rejected("__import__('os').system('echo pwned')")


def test_unknown_names_are_rejected():
    _assert_rejected("open('/etc/passwd')")
    _assert_rejected("getattr(abs, 'x')")
    _assert_rejected("x + 1")


def test_attribute_access_is_rejected():
    # Even without dunders: no attribute traversal at all.
    _assert_rejected("pi.real")


def test_string_constants_are_rejected():
    _assert_rejected("'a' * 3")


def test_comprehensions_lambdas_and_subscripts_are_rejected():
    _assert_rejected("[x for x in (1, 2)]")
    _assert_rejected("(lambda: 1)()")
    _assert_rejected("[1, 2, 3][0]")


def test_keyword_arguments_are_rejected():
    _assert_rejected("round(2.5, ndigits=1)")


def test_indirect_call_targets_are_rejected():
    # Calls only to bare names in the safe namespace — never to the result
    # of another expression.
    _assert_rejected("(min or max)(1, 2)")


def test_syntax_errors_are_tool_errors_not_exceptions():
    result = _run("2 +")
    assert not result.success
    assert "Calculation error" in result.error
