"""An AGENT node's node-level POST `asserts:` fire from the RESUME TERMINAL after a pause.

An AGENT is a spawner: a mid-loop control pause (`ask_user`) lowers to a continuation
`Grow`, so the agent's `${output}` is only known when its RESUME terminal produces the
final answer under the origin's `commit_as`. The node's node-level `asserts:` reading
`${output}` must therefore ride that resume terminal and fire through the ordinary
`eval_node` post-assert seam (the node's `run` wrapper) — exactly as CALL lowers onto the
cloned child END.

Regression guard: before the fix, the resume terminal carried no `post_asserts`, so a
`${output}` post-assert on a *pausing* agent was silently dropped on the resumed final
answer (the run succeeded even when the assert was violated).

Live and durable paths both use the compile-owned builder with the spawner's
`post_asserts`/`assert_texts`, so a resumed run enforces the same assert.
"""

from types import SimpleNamespace

from langchain_core.messages import AIMessage

import agent_composer.llm_clients as llm_clients_mod
from agent_composer.compile.expand import build_grow
from agent_composer.compile.model import END_ID, START_ID, CompiledFlow, Edge, FlowOutput
from agent_composer.run import resume_command
from agent_composer.events import RunFailed, RunPaused, RunResumed, RunSucceeded
from agent_composer.llm_clients import LLMConfig
from agent_composer.nodes.agent import AgentNode
from agent_composer.nodes.end import EndNode
from agent_composer.nodes.start import StartNode
from agent_composer.runtime.engine import FlowEngine
from agent_composer.suspension.pause import HumanInputRequired


def _ai_tool_call(name, args, call_id="q1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


class _AskChat:
    """A fake chat that asks once (ask_user), then returns a fixed final text answer."""

    def __init__(self, final_answer: str):
        self._replies = [
            _ai_tool_call("ask_user", {"question": "Approve?"}),
            AIMessage(content=final_answer),
        ]

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return self._replies.pop(0)


def _agent_flow_with_post_assert():
    node = AgentNode(
        "agent", prompt="go", controls=["ask_user"], llm_config=LLMConfig(), mode="tool_calling"
    )
    # A node-level POST assert reading `${output}` — the agent's committed final answer.
    node.post_asserts = ["output == 'GOOD'"]
    node.assert_texts = {"output == 'GOOD'": "output == 'GOOD'"}
    return CompiledFlow.from_parts(
        {
            "agent": node,
            START_ID: StartNode(START_ID, input_decls=[]),
            END_ID: EndNode.record(END_ID, output_names=["output"]),
        },
        [
            Edge("e0", START_ID, "agent"),
            Edge("agent->__end__#0", "agent", END_ID, input_group="output"),
        ],
        outputs=[FlowOutput(name="output", from_="${agent.output}")],
        wiring={END_ID: {"output": "${agent.output}"}},
    )


def _run_to_pause(graph):
    eng = FlowEngine(graph, grow_builder=build_grow)
    evs = list(eng.run())
    assert isinstance(evs[-1], RunPaused), f"expected pause, got {evs[-1]}"
    reason = evs[-1].reasons[0]
    assert isinstance(reason, HumanInputRequired)
    return eng, reason


def test_agent_post_assert_violated_on_resume_fails(monkeypatch):
    # Final answer "FINAL" violates `output == 'GOOD'` -> the resume terminal's run wrapper
    # fires the lowered post-assert and the run FAILS. Guards the drop regression.
    chat = _AskChat("FINAL")
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda cfg: chat)

    graph = _agent_flow_with_post_assert()
    eng, reason = _run_to_pause(graph)
    cmd = resume_command(SimpleNamespace(compiled=graph), reason, "yes")
    evs2 = list(eng.resume(commands=[cmd]))
    assert isinstance(evs2[0], RunResumed)
    last = evs2[-1]
    assert isinstance(last, RunFailed), f"expected failure, got {last}"
    assert last.error_type == "NodeAssertFailed"
    assert "post-assert" in (last.error or "").lower()


def test_agent_post_assert_satisfied_on_resume_succeeds(monkeypatch):
    # Final answer "GOOD" satisfies `output == 'GOOD'` -> the run succeeds with that output.
    chat = _AskChat("GOOD")
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda cfg: chat)

    graph = _agent_flow_with_post_assert()
    eng, reason = _run_to_pause(graph)
    cmd = resume_command(SimpleNamespace(compiled=graph), reason, "yes")
    evs2 = list(eng.resume(commands=[cmd]))
    assert isinstance(evs2[-1], RunSucceeded), f"expected success, got {evs2[-1]}"
    assert evs2[-1].output == "GOOD"


# =========================================================================== #
# durable resume-safety: the injected builder re-lowers the node-level ${output} assert
# =========================================================================== #

# A YAML-authored agent carrying a node-level `${output}` post-assert (the realistic form —
# the assert keeps its `${output}` head, resolved record-only by `_assert_holds`).
_ASK_WITH_ASSERT = """
id: ag
name: ag
nodes:
  agent:
    kind: agent
    prompt: go
    controls: [ask_user]
    output: str
    asserts:
      - ${output} == 'GOOD'
output: ${agent.output}
"""


def test_agent_post_assert_fires_on_durable_resume(monkeypatch):
    # Run -> pause -> cross-process snapshot/dumps -> RESTORE on a freshly recompiled flow ->
    # resume. Restoring on a fresh flow forces the injected builder to rebuild the resume terminal,
    # proving durable reconstruction carries the node-level `${output}` post-assert. The answer
    # "FINAL" violates `${output} == 'GOOD'` -> the rebuilt resume terminal fires it and FAILS.
    from agent_composer import load_flow
    from agent_composer.run import run_flow
    from agent_composer.suspension.checkpoint import RunCheckpoint

    chat = _AskChat("FINAL")
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda cfg: chat)

    loaded = load_flow(_ASK_WITH_ASSERT)
    rec = run_flow(loaded, {})
    assert rec.status == "paused"
    ckpt = RunCheckpoint.loads(rec.checkpoint.dumps())

    fresh = FlowEngine.restore(
        load_flow(_ASK_WITH_ASSERT).compiled,
        ckpt,
        grow_builder=build_grow,
    )
    cmd = resume_command(loaded, ckpt.pause_reasons[0], "yes")
    evs = list(fresh.resume(commands=[cmd]))
    last = evs[-1]
    assert isinstance(last, RunFailed), f"expected failure on durable resume, got {last}"
    assert last.error_type == "NodeAssertFailed"
    assert "post-assert" in (last.error or "").lower()
