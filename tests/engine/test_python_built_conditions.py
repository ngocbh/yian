"""A CASE `when` or LOOP predicate set in Python that doesn't parse is a setup error."""

import traceback

import pytest

from yian.expr import ExpressionError
from yian.nodes.case import Case, CaseNode
from yian.nodes.loop import LoopNode


def test_unparseable_case_when_raises_its_parse_error_from_setup():
    """`Node.run` names the node, case, and field instead of failing inside `forward`."""
    node = CaseNode("gate", [Case("ok", "x > 0"), Case("pos", "${x} > 0")])

    with pytest.raises(ExpressionError) as excinfo:
        node.run({"x": 1})

    message = str(excinfo.value)
    assert message.startswith(
        "node 'gate' case 'pos' `when:` '${x} > 0' is not a valid boolean expression: "
    )
    assert "write `x > 0`" in message


def test_case_when_evaluation_error_still_fails_in_forward():
    """A `when` that parses but can't evaluate is unchanged: it raises from `forward`."""
    node = CaseNode("gate", [Case("pos", "x > 0")])

    with pytest.raises(ExpressionError) as excinfo:
        node.run({"x": {"a": 1}})

    assert "is not a valid boolean expression" not in str(excinfo.value)
    assert any(frame.name == "forward" for frame in traceback.extract_tb(excinfo.tb))


@pytest.mark.parametrize("kind", ["while", "until"])
def test_unparseable_loop_predicate_raises_before_the_first_body(kind):
    """Both predicate kinds are checked at setup, including `until` at iteration 0."""
    node = LoopNode("loop", flow_id="f", child=object(), predicate_kind=kind,
                    predicate="not ${exited}", max_iters=10)

    with pytest.raises(ExpressionError) as excinfo:
        node.run({"exited": False})

    message = str(excinfo.value)
    assert message.startswith(
        f"node 'loop' `{kind}:` 'not ${{exited}}' is not a valid boolean expression: "
    )
    assert "write `not exited`" in message


def test_times_loop_has_no_predicate_to_check():
    """A `times` loop carries no predicate; setup accepts it."""
    node = LoopNode("loop", flow_id="f", child=object(), predicate_kind="times",
                    predicate=None, times=2, max_iters=2)

    node.validate_conditions()


def test_validate_conditions_names_the_given_path():
    """The pre-run check names a node by its static path."""
    node = CaseNode("gate", [Case("pos", "${x} > 0")])

    with pytest.raises(ExpressionError, match=r"^node 'loop/gate' case 'pos' `when:`"):
        node.validate_conditions(path="loop/gate")
