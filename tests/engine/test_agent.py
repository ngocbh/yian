"""The AGENT node with a fake langchain model.

No API keys: `model_from_config` is monkeypatched to return a scripted fake chat
model that yields canned `AIMessage`s; tool execution goes through `TOOL_REGISTRY`
(monkeypatched). Covers one model call without tools and the tool loop with them
(a tool round-trip, strict grants, exception propagation, and the iteration cap).

(The AGENT-through-the-loader run path is covered once the Compose-loader agent
run lands; AGENT compile-time validation moved with the v0 compiler.)
"""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

import yian.llm_clients as llm_clients_mod
import yian.tools as tools_mod
from yian.diagnostics import OriginRef
from yian.nodes.agent import AgentNode
from yian.nodes.contracts import ContractOrigins, Parameter, ValueContract
from yian.runtime.eval_node import EvaluationFailure, eval_node
from yian.typesys.pool import VariablePool
from yian.typesys.values import Type, ValueKind
from yian.llm_clients import LLMConfig


class _FakeChat:
    """Returns queued AIMessages in order; records calls and bound tools."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.calls += 1
        return self._replies.pop(0)


class _InvalidStructuredChat:
    """Return one invalid generated value or raise one provider error."""

    def __init__(self, *, payload=None, error=None):
        self.payload = payload
        self.error = error

    def with_structured_output(self, schema):
        payload = self.payload
        error = self.error

        class _Bound:
            """One schema-bound invalid provider response."""

            def invoke(self, _messages):
                if error is not None:
                    raise error
                return schema.model_validate(payload)

        return _Bound()


def _ai_tool_call(name, args, call_id="1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _patch_model(monkeypatch, chat):
    monkeypatch.setattr(llm_clients_mod, "model_from_config", lambda cfg: chat)


class _NoToolArgs(BaseModel):
    """Schema for fake tools that accept no public arguments."""


class _TopicToolArgs(BaseModel):
    """Schema for the fake value tool's one public argument."""

    topic: str


class _FakeTool:
    def __init__(self, fn, args_schema=_NoToolArgs):
        self._fn = fn
        self.seen = []
        self.args_schema = args_schema
        self.tool_call_schema = args_schema

    def invoke(self, args):
        self.seen.append(args)
        return self._fn(args)


def _node(tools=None, prompt="hi"):
    return AgentNode("n", prompt=prompt, tools=tools or [], llm_config=LLMConfig())


def _run_node(node, pool=None):
    """Evaluate one agent into its raw output or failure carrier."""
    return eval_node(node, None, pool or VariablePool()).outcome


@pytest.mark.parametrize("variables", [
    (), {}, [object()], [Parameter("choice")],
    [Parameter("choice", has_default=True, default=1)],
    [Parameter("", ValueContract(Type.scalar(ValueKind.INTEGER)), has_default=True, default=1)],
    [Parameter(1, ValueContract(Type.scalar(ValueKind.INTEGER)), has_default=True, default=1)],
    [Parameter("choice", ValueContract(Type.scalar(ValueKind.INTEGER)), has_default=True, default="1")],
    [Parameter("choice", ValueContract(Type(ValueKind.INTEGER, bounds=(1, 2))), has_default=True, default=3)],
    [Parameter("choice", ValueContract(Type(ValueKind.INTEGER, bounds=(2, 1))), has_default=True, default=1)],
])
def test_direct_agent_rejects_invalid_variable_declarations(variables):
    with pytest.raises((TypeError, ValueError)):
        AgentNode("writer", variables=variables)


def test_direct_agent_rejects_duplicate_variables_and_preserves_valid_default_payloads():
    variable = Parameter(
        "count", ValueContract(Type(ValueKind.NUMBER, bounds=(0, 4))),
        has_default=True, default=4,
    )
    with pytest.raises(ValueError, match="duplicate"):
        AgentNode("writer", variables=[variable, variable])
    node = AgentNode("writer", variables=[variable])
    assert node.variables == [variable]
    assert node.variables[0].default is variable.default
    assert type(node.variables[0].default) is int
    assert node.params is None


@pytest.mark.parametrize("field,value", [("has_default", 1), ("required", 0), ("required", True)])
def test_direct_agent_rejects_forged_variable_default_presence(field, value):
    variable = Parameter("count", ValueContract(Type.scalar(ValueKind.INTEGER)),
                         has_default=True, default=4)
    object.__setattr__(variable, field, value)
    with pytest.raises(ValueError, match="requires a default and required=False"):
        AgentNode("writer", variables=[variable])


def test_direct_agent_requires_an_exact_variable_contract():
    class CustomContract(ValueContract):
        pass

    variable = Parameter("count", CustomContract(Type.scalar(ValueKind.INTEGER)),
                         has_default=True, default=4)
    with pytest.raises(TypeError, match="exact ValueContract"):
        AgentNode("writer", variables=[variable])


def test_no_tools_returns_plain_text(monkeypatch):
    _patch_model(monkeypatch, _FakeChat([AIMessage(content="answer")]))
    assert _run_node(_node()).value == "answer"


def test_tool_loop_round_trip(monkeypatch):
    tool = _FakeTool(lambda a: f"px:{a['topic']}", _TopicToolArgs)
    monkeypatch.setitem(tools_mod.TOOL_REGISTRY, "value", tool)
    chat = _FakeChat([
        _ai_tool_call("value", {"topic": "ACME"}),
        AIMessage(content="done"),
    ])
    _patch_model(monkeypatch, chat)
    assert _run_node(_node(tools=["value"])).value == "done"
    assert tool.seen == [{"topic": "ACME"}]   # the tool actually ran
    assert chat.calls == 2                      # ask -> tool -> ask


def test_tool_loop_executes_model_calls_in_returned_order(monkeypatch):
    seen = []

    def first(value: int) -> str:
        """Record the first call."""
        seen.append(("first", value))
        return "one"

    def second(value: int) -> str:
        """Record the second call."""
        seen.append(("second", value))
        return "two"

    monkeypatch.setitem(
        tools_mod.TOOL_REGISTRY,
        "first",
        StructuredTool.from_function(first, name="first"),
    )
    monkeypatch.setitem(
        tools_mod.TOOL_REGISTRY,
        "second",
        StructuredTool.from_function(second, name="second"),
    )
    chat = _FakeChat(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "first", "args": {"value": 1}, "id": "1", "type": "tool_call"},
                    {"name": "second", "args": {"value": 2}, "id": "2", "type": "tool_call"},
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    _patch_model(monkeypatch, chat)

    assert _run_node(_node(tools=["first", "second"])).value == "done"
    assert seen == [("first", 1), ("second", 2)]


def test_agent_rejects_globally_registered_but_ungranted_tool(monkeypatch):
    seen = []

    def secret() -> str:
        """A process tool this agent was not granted."""
        seen.append("called")
        return "secret"

    monkeypatch.setitem(
        tools_mod.TOOL_REGISTRY,
        "secret",
        StructuredTool.from_function(secret, name="secret"),
    )
    chat = _FakeChat([_ai_tool_call("secret", {})])
    _patch_model(monkeypatch, chat)

    event = _run_node(_node(tools=[]))
    assert isinstance(event, EvaluationFailure)
    assert "not granted" in event.failure.message
    assert seen == []


def test_tool_loop_rejects_model_injected_arguments(monkeypatch):
    seen = []

    def lookup(question: str) -> str:
        """Answer one public question."""
        seen.append(question)
        return question

    monkeypatch.setitem(
        tools_mod.TOOL_REGISTRY,
        "lookup",
        StructuredTool.from_function(lookup, name="lookup"),
    )
    chat = _FakeChat(
        [_ai_tool_call("lookup", {"question": "status?", "node_id": "injected"})]
    )
    _patch_model(monkeypatch, chat)

    event = _run_node(_node(tools=["lookup"]))
    assert isinstance(event, EvaluationFailure)
    assert "unsupported argument" in event.failure.message
    assert seen == []


def test_tool_loop_propagates_tool_exception_as_node_failure(monkeypatch):
    def explode(topic: str) -> str:
        """Raise the failure the node boundary must report."""
        raise RuntimeError(f"cannot load {topic}")

    monkeypatch.setitem(
        tools_mod.TOOL_REGISTRY,
        "explode",
        StructuredTool.from_function(explode, name="explode"),
    )
    chat = _FakeChat([_ai_tool_call("explode", {"topic": "ACME"})])
    _patch_model(monkeypatch, chat)

    event = _run_node(_node(tools=["explode"]))
    assert isinstance(event, EvaluationFailure)
    assert event.failure.cause_type == "RuntimeError"
    assert "cannot load ACME" in event.failure.message


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        (
            (("profile", "age"), OriginRef("output/profile/age")),
            OriginRef("output/profile/age"),
        ),
        (
            (("profile",), OriginRef("output/profile")),
            OriginRef("output/profile"),
        ),
    ],
)
def test_structured_failure_selects_exact_or_ancestor_output_origin(
    monkeypatch,
    entries,
    expected,
):
    """Generated record errors reuse exact-to-ancestor contract lookup."""
    typ = Type(
        kind=ValueKind.OBJECT,
        fields={
            "profile": Type(
                kind=ValueKind.OBJECT,
                fields={"age": Type.scalar(ValueKind.INTEGER)},
                required=frozenset({"age"}),
            )
        },
        required=frozenset({"profile"}),
    )
    node = AgentNode(
        "n",
        retries=0,
        llm_config={"provider": "openai", "model": "gpt-4o"},
    )
    node.origin = OriginRef("node")
    node.output_contract = ValueContract(
        type=typ,
        origins=ContractOrigins(
            default=OriginRef("output"),
            entries=(entries,),
        ),
    )
    _patch_model(
        monkeypatch,
        _InvalidStructuredChat(payload={"profile": {"age": "bad"}}),
    )

    event = _run_node(node)

    assert isinstance(event, EvaluationFailure)
    assert event.failure.cause_type == "ValidationError"
    assert event.failure.trace == (expected,)


@pytest.mark.parametrize(
    ("typ", "payload"),
    [
        (Type.scalar(ValueKind.INTEGER), {"value": "bad"}),
        (
            Type(kind=ValueKind.STRING, tags=frozenset({"positive", "negative"})),
            {"value": "not-a-tag"},
        ),
    ],
)
def test_structured_scalar_failure_selects_contract_root(monkeypatch, typ, payload):
    """The generated scalar `value` wrapper maps to the output root."""
    node = AgentNode(
        "n",
        retries=0,
        llm_config={"provider": "openai", "model": "gpt-4o"},
    )
    node.origin = OriginRef("node")
    node.output_contract = ValueContract(
        type=typ,
        origins=ContractOrigins(default=OriginRef("output")),
    )
    _patch_model(monkeypatch, _InvalidStructuredChat(payload=payload))

    event = _run_node(node)

    assert isinstance(event, EvaluationFailure)
    assert event.failure.trace == (OriginRef("output"),)


def test_non_validation_structured_failure_keeps_node_origin(monkeypatch):
    """Provider errors outside generated-schema validation keep common fallback."""
    node = AgentNode(
        "n",
        retries=0,
        llm_config={"provider": "openai", "model": "gpt-4o"},
    )
    node.origin = OriginRef("node")
    node.output_contract = ValueContract(
        type=Type.scalar(ValueKind.INTEGER),
        origins=ContractOrigins(default=OriginRef("output")),
    )
    _patch_model(
        monkeypatch,
        _InvalidStructuredChat(error=RuntimeError("provider failed")),
    )

    event = _run_node(node)

    assert isinstance(event, EvaluationFailure)
    assert event.failure.cause_type == "RuntimeError"
    assert event.failure.trace == (OriginRef("node"),)


def test_tool_validation_error_does_not_claim_scalar_output_origin(monkeypatch):
    """A tool's wrapper-shaped Pydantic location remains a node-level failure."""

    class ScalarWrapper(BaseModel):
        """Deliberately collide with the generated scalar output schema title."""

        value: int

    monkeypatch.setitem(
        tools_mod.TOOL_REGISTRY,
        "scalar_wrapper",
        _FakeTool(lambda args: args["value"], ScalarWrapper),
    )
    node = _node(tools=["scalar_wrapper"])
    node.origin = OriginRef("node")
    node.output_contract = ValueContract(
        type=Type.scalar(ValueKind.INTEGER),
        origins=ContractOrigins(default=OriginRef("output")),
    )
    _patch_model(
        monkeypatch,
        _FakeChat([_ai_tool_call("scalar_wrapper", {"value": "bad"})]),
    )

    event = _run_node(node)

    assert isinstance(event, EvaluationFailure)
    assert event.failure.cause_type == "ValidationError"
    assert event.failure.trace == (OriginRef("node"),)


def test_source_free_structured_failure_keeps_empty_trace(monkeypatch):
    """Generated validation does not invent provenance for Python-authored agents."""
    node = AgentNode(
        "n",
        retries=0,
        llm_config={"provider": "openai", "model": "gpt-4o"},
    )
    node.output_contract = ValueContract(type=Type.scalar(ValueKind.INTEGER))
    _patch_model(monkeypatch, _InvalidStructuredChat(payload={"value": "bad"}))

    event = _run_node(node)

    assert isinstance(event, EvaluationFailure)
    assert event.failure.trace == ()


def test_tool_loop_cap_trips_when_bounded(monkeypatch):
    # A POSITIVE env cap bounds the loop: always-asks-for-a-tool -> never answers -> trips
    # the cap after `cap` turns. The raised ToolIterationLimitError becomes EvaluationFailure at
    # eval_node. (The DEFAULT is -1 / no cap — see the no-cap test below.)
    cap = 5
    monkeypatch.setitem(tools_mod.TOOL_REGISTRY, "noop", _FakeTool(lambda a: "ok"))
    looping = _FakeChat([_ai_tool_call("noop", {}, str(i)) for i in range(cap)])
    _patch_model(monkeypatch, looping)
    node = _node(tools=["noop"])
    node.env = {"max_tool_iterations": cap}
    ev = _run_node(node)
    assert isinstance(ev, EvaluationFailure)
    assert ev.failure.code == "agent.max_tool_iterations"
    assert ev.failure.cause_type == "ToolIterationLimitError"
    assert ev.failure.message == (
        "agent node 'n' hit the tool-iteration cap (5) without a final answer; "
        "the last reply's 1 tool call(s) were not run"
    )


def test_default_no_cap_runs_unbounded(monkeypatch):
    from yian.defaults import current

    # The default cap is the -1 no-cap sentinel: an agent making many tool turns (well past the
    # old default of 100) then answering still finishes, no AgentLoopError.
    assert current().agent_max_tool_iterations == -1
    monkeypatch.setitem(tools_mod.TOOL_REGISTRY, "noop", _FakeTool(lambda a: "ok"))
    replies = [_ai_tool_call("noop", {}, str(i)) for i in range(150)]
    replies.append(AIMessage(content="done"))
    chat = _FakeChat(replies)
    _patch_model(monkeypatch, chat)
    assert _run_node(_node(tools=["noop"])).value == "done"
    assert chat.calls == 151          # 150 tool turns + the final answer, all uncapped


def test_env_max_tool_iterations_overrides_cap(monkeypatch):
    # A per-node `env: {max_tool_iterations: 2}` bounds the loop: the agent trips the cap after
    # exactly 2 model turns (vs the -1 no-cap default). Proves the env value threads
    # node.env -> _max_tool_iterations -> AgentRunContext -> the tool loop.
    monkeypatch.setitem(tools_mod.TOOL_REGISTRY, "noop", _FakeTool(lambda a: "ok"))
    looping = _FakeChat([_ai_tool_call("noop", {}, str(i)) for i in range(2)])
    _patch_model(monkeypatch, looping)
    node = _node(tools=["noop"])
    node.env = {"max_tool_iterations": 2}
    ev = _run_node(node)
    assert isinstance(ev, EvaluationFailure)
    assert ev.failure.code == "agent.max_tool_iterations"
    assert "(2)" in ev.failure.message  # the message uses the env override, not the default
    assert looping.calls == 2         # stopped at exactly the env bound


def test_tool_loop_cap_does_not_run_the_last_replys_tools(monkeypatch):
    # The reply that uses the last allowed model turn asks for a tool. No later model call
    # could read its result, so the node fails without running it.
    tool = _FakeTool(lambda a: "ok")
    monkeypatch.setitem(tools_mod.TOOL_REGISTRY, "noop", tool)
    looping = _FakeChat([_ai_tool_call("noop", {}, str(i)) for i in range(2)])
    _patch_model(monkeypatch, looping)
    node = _node(tools=["noop"])
    node.env = {"max_tool_iterations": 2}
    ev = _run_node(node)
    assert isinstance(ev, EvaluationFailure)
    assert ev.failure.code == "agent.max_tool_iterations"
    assert "not run" in ev.failure.message
    assert looping.calls == 2
    assert len(tool.seen) == 1        # only the first reply's call ran


def test_ungranted_tool_stays_node_execute(monkeypatch):
    # Only the iteration cap has its own code; asking for a tool the node was not
    # granted is an ordinary node-body failure.
    monkeypatch.setitem(tools_mod.TOOL_REGISTRY, "noop", _FakeTool(lambda a: "ok"))
    _patch_model(monkeypatch, _FakeChat([_ai_tool_call("other", {})]))
    ev = _run_node(_node(tools=["noop"]))
    assert isinstance(ev, EvaluationFailure)
    assert ev.failure.code == "node.execute"
    assert ev.failure.cause_type == "AgentLoopError"
    assert "not granted" in ev.failure.message


def test_tool_loop_cap_of_one_still_answers_without_tools(monkeypatch):
    # A cap of 1 allows one model turn: a reply with no tool calls finishes the node.
    monkeypatch.setitem(tools_mod.TOOL_REGISTRY, "noop", _FakeTool(lambda a: "ok"))
    _patch_model(monkeypatch, _FakeChat([AIMessage(content="done")]))
    node = _node(tools=["noop"])
    node.env = {"max_tool_iterations": 1}
    assert _run_node(node).value == "done"


def test_env_max_tool_iterations_minus_one_accepted():
    # -1 is the explicit no-cap sentinel and passes validation (unlike 0 / other negatives).
    node = _node()
    node.env = {"max_tool_iterations": -1}
    assert node._max_tool_iterations() == -1


def test_env_max_tool_iterations_rejects_bad_value(monkeypatch):
    # A non-(positive-int-or-(-1)) env value is rejected as EvaluationFailure (ValueError at the
    # boundary), not silently coerced. bool is explicitly rejected (isinstance(True, int) is True).
    _patch_model(monkeypatch, _FakeChat([AIMessage(content="unused")]))
    node = _node()
    node.env = {"max_tool_iterations": 0}
    ev = _run_node(node)
    assert isinstance(ev, EvaluationFailure)
    assert "max_tool_iterations" in ev.failure.message


def test_agent_run_has_no_scratch_kwarg():
    # AgentNode.run is a pure `run(self, inputs)` with no engine scratch capability.
    import inspect

    assert "scratch" not in inspect.signature(AgentNode.run).parameters
