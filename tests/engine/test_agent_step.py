"""`agent_step` — the scratch-free agent loop body.

`agent_step(messages, pending, iterations, ctx) -> Output | Grow` carries the
re-entry frame as args/return, never scratch. On ENTRY it always invokes the model on
the passed-in `messages` (no scratch resume-replay branch). On a final answer ->
`Output(text)`; on a control call -> a seed-first `Grow` carrying the descriptor PAIR
(a `human_input` descriptor + a `resume_agent` descriptor reading `answer` via the BARE
forward-ref `${<hi>.output}`).
"""

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from agent_composer.nodes.agent.modes.common import AgentRunContext
from agent_composer.nodes.agent.modes.tool_calling import agent_step
from agent_composer.nodes.base import Grow, Output


class _Chat:
    def __init__(self, replies):
        self._r = list(replies)
        self.calls = 0

    def bind_tools(self, t):
        return self

    def invoke(self, m):
        self.calls += 1
        return self._r.pop(0)


def _ai_call(name, args, cid="q1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid, "type": "tool_call"}])


def _ctx(chat, controls=()):
    return AgentRunContext(node_id="agent", prompt="go", tools=[], controls=list(controls), model=chat)


def test_agent_step_entry_always_invokes_model_once():
    # ENTRY always calls the model on the passed-in messages (no scratch resume-replay branch).
    chat = _Chat([AIMessage(content="DONE")])
    msgs = [SystemMessage(content="s"), HumanMessage(content="go")]
    out = agent_step(msgs, None, 0, _ctx(chat))
    assert isinstance(out, Output) and out.value == "DONE"
    assert chat.calls == 1                  # exactly one model turn


def test_agent_step_control_call_returns_grow_pair_no_scratch():
    chat = _Chat([_ai_call("ask_user", {"question": "ok?"})])
    ctx = _ctx(chat, controls=["ask_user"])
    msgs = [SystemMessage(content="s"), HumanMessage(content="go")]
    res = agent_step(msgs, None, 0, ctx)
    assert isinstance(res, Grow)
    hi = res.seed["hi_desc"]           # the continuation PAIR rides on Grow.seed
    resume = res.seed["resume_desc"]
    assert hi["kind"] == "human_input" and hi["prompt"] == "ok?"
    assert hi["slot"] == "q1"
    assert resume["kind"] == "resume_agent"
    assert resume["pending"]["call_id"] == "q1"
    assert "answer_key" not in resume["pending"]   # the forward-ref edge replaces it
    assert resume["iterations"] == 1   # the control turn counted
    # node-first ref ${<hi>.output}, the new shape
    assert resume["answer"] == "${%s.output}" % hi["node_id"]
    assert "memo" in resume and isinstance(resume["memo"], list)  # messages_to_dict, not scratch
