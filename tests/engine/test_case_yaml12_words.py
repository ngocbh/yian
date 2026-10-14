"""CASE `when:` words that YAML 1.1 read as booleans are text under YAML 1.2 rules."""

from yian.compose import load_flow
from yian.events import NodeStarted
from yian.run import run_flow


def _started(result) -> set[str]:
    """Return the ids of every node that started in a run."""
    return {e.node_id for e in result.events if isinstance(e, NodeStarted)}


def _flow(cmd_type: str, case_tail: str) -> str:
    """Return a flow whose `route` case matches `input.cmd` and routes to echo nodes."""
    return (
        "id: f\n"
        "name: f\n"
        "input:\n"
        f"  cmd: {cmd_type}\n"
        "nodes:\n"
        "  route:\n"
        "    kind: case\n"
        "    on: ${input.cmd}\n"
        "    cases:\n"
        "      - when: yes\n"
        "        then: agree\n"
        f"{case_tail}"
        "  agree:\n"
        "    kind: code\n"
        "    code: tests.engine._compose_codefns:echo\n"
        "    input:\n"
        "      topic: agreed\n"
        "    output: str\n"
        "  refuse:\n"
        "    kind: code\n"
        "    code: tests.engine._compose_codefns:echo\n"
        "    input:\n"
        "      topic: refused\n"
        "    output: str\n"
        "output: ${agree.output | refuse.output}\n"
    )


def test_yes_and_no_cover_a_literal_input_without_else():
    loaded = load_flow(
        _flow(
            "Literal[yes, no]",
            "      - when: no\n"
            "        then: refuse\n",
        )
    )

    agreed = run_flow(loaded, {"cmd": "yes"})
    refused = run_flow(loaded, {"cmd": "no"})

    assert agreed.status == "succeeded", agreed.failure
    assert "agree" in _started(agreed)
    assert "refuse" not in _started(agreed)
    assert refused.status == "succeeded", refused.failure
    assert "refuse" in _started(refused)
    assert "agree" not in _started(refused)


def test_when_yes_does_not_match_a_true_bool():
    # Pins current behavior: `when: yes` is the text "yes", which never equals True.
    # A typed `when:` check is planned separately.
    loaded = load_flow(_flow("bool", "    else: refuse\n"))

    result = run_flow(loaded, {"cmd": True})

    assert result.status == "succeeded", result.failure
    assert "refuse" in _started(result)
    assert "agree" not in _started(result)
