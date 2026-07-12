"""Agent pause = continuation — full loader->engine round-trips.

A stub model drives the agent loop; each `ask_user` control call lowers to a
continuation PAIR (a `human_input` leaf + a resume continuation — an `AgentNode` with a
`Resume` entry) via `Grow`, the engine's generic `_apply_grow` clones it
namespaced at the spawner, the human_input leaf parks, and the deliver-as-Output resume
feeds the answer into the resume continuation. These tests pin: the pause->resume->finish round trip, no
replay of turn 1 (the carried memo is replayed as messages, not re-invoked — asserted
via the stub call count), a 2-pause agent does not deadlock, and a many-pause agent
(> MAX_REF_DEPTH pauses) does NOT trip MAX_REF_DEPTH (the agent arm carries the parent
depth UNCHANGED — it is bounded by MAX_TOOL_ITERATIONS / MAX_TOTAL_NODES).
"""

import agent_composer.llm_clients as llm
import pytest
from langchain_core.messages import AIMessage

from agent_composer import load_flow
from agent_composer.run import run_flow
from agent_composer.run import resume_command
from agent_composer.events import RunPaused, RunSucceeded
from agent_composer.suspension.pause import HumanInputRequired

ASK = """
id: ag
name: ag
nodes:
  agent: {kind: agent, prompt: go, controls: [ask_user], output: str}
output: ${agent.output}
"""


def _chat(replies):
    class C:
        calls = 0

        def bind_tools(self, t):
            return self

        def invoke(self, m):
            type(self).calls += 1
            return replies.pop(0)

    return C()


def _ask(args, cid="q1"):
    return AIMessage(content="", tool_calls=[{"name": "ask_user", "args": args, "id": cid, "type": "tool_call"}])


def _record_from(engine, evs):
    """Re-summarize a run record (status + pause_reasons/output) from a resume drain on
    the same live engine — mirrors resume_flow's summary so the loop can re-resume."""
    from types import SimpleNamespace

    status, output, reasons = "incomplete", None, []
    for e in evs:
        if isinstance(e, RunSucceeded):
            status, output = "succeeded", e.output
        elif isinstance(e, RunPaused):
            status, reasons = "paused", list(e.reasons)
    return SimpleNamespace(engine=engine, status=status, output=output, pause_reasons=reasons)


@pytest.mark.parametrize(
    "cid",
    [
        "q1",  # dashless (Anthropic/OpenAI-style) — must keep passing
        "adebc542-e4a3-4b7c-8f12-deadbeef0000",  # dashed uuid (Ollama) — slug-sanitized
    ],
)
def test_ask_user_pauses_resumes_finishes(monkeypatch, cid):
    chat = _chat([_ask({"question": "ok?"}, cid), AIMessage(content="FINAL")])
    monkeypatch.setattr(llm, "model_from_config", lambda cfg: chat)
    loaded = load_flow(ASK)
    rec = run_flow(loaded, {})
    assert rec.status == "paused"
    reason = rec.pause_reasons[0]
    assert isinstance(reason, HumanInputRequired) and reason.prompt == "ok?"
    # the parked leaf is namespaced under the spawner; resume resolves it on the LIVE graph
    assert reason.node_id.startswith("agent/") and reason.node_id in rec.engine.flow.nodes
    cmd = resume_command(loaded, reason, "yes")
    evs = list(rec.engine.resume(commands=[cmd]))
    assert isinstance(evs[-1], RunSucceeded) and evs[-1].output == "FINAL"
    assert type(chat).calls == 2  # turn 1 NOT re-invoked (replayed from carried memo)


def test_slugged_hi_id_is_path_safe():
    # The minted human-input node id must parse as a reference path so the resume answer
    # ref `${__ask#<slug>.output}` parses — a raw dashed uuid would not.
    from agent_composer.expr.grammar import parse_expr
    from agent_composer.nodes.agent.modes.tool_calling import _slug_call_id

    slug = _slug_call_id("adebc542-e4a3-4b7c-8f12-deadbeef0000")
    parse_expr("__ask#" + slug + ".output")  # no ExpressionError == path-safe


def test_two_pause_agent_does_not_deadlock(monkeypatch):
    chat = _chat([_ask({"question": "a?"}, "q1"), _ask({"question": "b?"}, "q2"),
                  AIMessage(content="FINAL")])
    monkeypatch.setattr(llm, "model_from_config", lambda cfg: chat)
    loaded = load_flow(ASK)
    rec = run_flow(loaded, {})
    cmd1 = resume_command(loaded, rec.pause_reasons[0], "x")
    evs1 = list(rec.engine.resume(commands=[cmd1]))
    paused2 = [e for e in evs1 if isinstance(e, RunPaused)]
    assert paused2, "second pause must surface, not deadlock"
    cmd2 = resume_command(loaded, paused2[-1].reasons[0], "y")
    evs2 = list(rec.engine.resume(commands=[cmd2]))
    assert isinstance(evs2[-1], RunSucceeded) and evs2[-1].output == "FINAL"
    assert type(chat).calls == 3  # one model turn per segment, none re-invoked


def test_many_pause_agent_does_not_trip_max_ref_depth(monkeypatch):
    # An agent pausing K > MAX_REF_DEPTH times is NOT recursion — each
    # resume_agent carries the PARENT depth UNCHANGED (no +1), so the agent arm NEVER trips
    # MAX_REF_DEPTH. It is bounded by MAX_TOOL_ITERATIONS (the agent_step cap) and by
    # MAX_TOTAL_NODES (the +2 nodes/pause budget check on the agent arm), NOT by the
    # REF depth bound. This is the regression test for the carry-depth-unchanged invariant.
    from agent_composer.runtime.engine import MAX_REF_DEPTH

    # n must EXCEED MAX_REF_DEPTH (to prove the agent arm never trips the depth bound). The default
    # tool-iteration cap is -1 (NO cap), so the K-pause chain is bounded only by MAX_TOTAL_NODES,
    # never the tool cap — MAX_REF_DEPTH+1=6 pauses is comfortably within that node budget.
    n = MAX_REF_DEPTH + 1
    assert n > MAX_REF_DEPTH  # exceed the REF-depth bound; the default tool cap (-1) never fires
    chat = _chat([_ask({"question": f"q{i}?"}, f"q{i}") for i in range(n)] + [AIMessage(content="FINAL")])
    monkeypatch.setattr(llm, "model_from_config", lambda cfg: chat)
    loaded = load_flow(ASK)
    rec = run_flow(loaded, {})
    for i in range(n):  # resume past MAX_REF_DEPTH pauses
        assert rec.status == "paused", f"pause {i} must surface (NOT a MAX_REF_DEPTH RunFailed)"
        evs = list(rec.engine.resume(commands=[resume_command(loaded, rec.pause_reasons[0], f"a{i}")]))
        # the depth bound must never fire on the agent arm, no matter how many pauses
        assert not any("MAX_REF_DEPTH" in getattr(e, "error", "") for e in evs)
        rec = _record_from(rec.engine, evs)  # paused-again or succeeded; never failed-on-depth
    assert rec.status == "succeeded" and rec.output == "FINAL"
    assert type(chat).calls == n + 1  # one model turn per segment, none re-invoked
