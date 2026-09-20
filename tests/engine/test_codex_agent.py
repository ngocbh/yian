"""Offline native AGENT execution and explicitly granted Composer tools."""

from copy import deepcopy
from concurrent.futures import CancelledError as FutureCancelledError
from threading import Event

from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from pydantic import ValidationError
import pytest

import agent_composer.llm_clients as llm_clients
from agent_composer.memory.session import SessionMemory
from agent_composer.diagnostics import OriginRef
from agent_composer.memory.contracts import MemorySection
from agent_composer.nodes.agent import AgentNode
from agent_composer.nodes.base import _CapabilityFault, _HostStop
from agent_composer.nodes.contracts import ContractOrigins, ValueContract
from agent_composer.tools.registry import TOOL_REGISTRY, install_ask_user
from agent_composer.typesys.types import Type, ValueKind
from tests.engine._session_fakes import harness, request, snapshot


@pytest.fixture
def native(monkeypatch):
    from agent_composer.nodes.agent import codex

    calls, workers = [], []
    behavior = {"texts": [("final_answer", "answer")], "status": "completed"}

    class Worker:
        def __init__(self, **options):
            self.options = options
            self.closed = False
            workers.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.closed = True

        def initialize(self):
            return {}

        def _result(self, operation, result):
            self.options["on_result"](operation, deepcopy(result))
            if behavior.get("after_result") is not None:
                behavior["after_result"](operation)
            self.options["check_stopped"]()
            return result

        def start_thread(self, params=None):
            calls.append(("start", deepcopy(params)))
            return self._result(
                "start_thread", {"thread": {"id": "thread", "sessionId": "wrong"}}
            )

        def resume_thread(self, thread_id, params=None):
            calls.append(("resume", thread_id, deepcopy(params)))
            if behavior.get("resume_error") is not None:
                raise behavior["resume_error"]
            return self._result("resume_thread", {"thread": {"id": thread_id}})

        def run_turn(self, thread_id, input_items, params=None):
            calls.append(("turn", thread_id, input_items, deepcopy(params)))
            if behavior.get("run") is not None:
                behavior["run"](self)
            texts = behavior["texts"]
            if behavior.get("replies"):
                texts = [("final_answer", behavior["replies"].pop(0))]
            for index, (phase, text) in enumerate(texts):
                self.options["on_event"](
                    "item/completed",
                    {
                        "threadId": thread_id,
                        "turnId": "turn",
                        "item": {
                            "id": str(index),
                            "type": "agentMessage",
                            "text": text,
                            "phase": phase,
                        },
                    },
                )
            result = {
                "threadId": thread_id,
                "turn": {
                    "id": "turn",
                    "status": behavior["status"],
                    "items": [],
                    "error": behavior.get("error"),
                },
            }
            self.options["on_event"]("turn/completed", deepcopy(result))
            if behavior.get("after_terminal") is not None:
                behavior["after_terminal"]()
            self.options["check_stopped"]()
            return self._result("run_turn", result)

    monkeypatch.setattr(codex, "CodexWorkerClient", Worker)
    monkeypatch.setattr(
        llm_clients, "model_from_config", lambda *_a, **_k: pytest.fail("API fallback")
    )
    return behavior, calls, workers


def node(*, tools=None, mode="tool_calling", **kwargs):
    return AgentNode(
        "native",
        prompt="question",
        mode=mode,
        tools=["builtin_tools"] if tools is None else tools,
        llm_config={
            "provider": "codex",
            "model": "native-model",
            "cwd": "/workspace",
            "sandbox": "read-only",
        },
        **kwargs,
    )


def _tool(monkeypatch, name="lookup", function=None):
    function = (lambda query: "found " + query) if function is None else function
    tool = StructuredTool.from_function(
        function, name=name, description="Find a value."
    )
    monkeypatch.setitem(TOOL_REGISTRY, name, tool)
    return tool


def _call(worker, name="lookup", arguments=None, **extra):
    return worker.options["callback"](
        "item/tool/call",
        {
            "threadId": "thread",
            "turnId": "turn",
            "callId": "call",
            "tool": name,
            "arguments": {"query": "needle"} if arguments is None else arguments,
            **extra,
        },
    )


def test_native_start_passes_current_options_and_uses_event_text(native):
    _, calls, workers = native
    assert node().forward({}).value == "answer"
    assert [call[0] for call in calls] == ["start", "turn"]
    assert calls[0][1]["model"] == "native-model"
    assert calls[0][1]["cwd"] == "/workspace"
    assert calls[0][1]["sandbox"] == "read-only"
    assert calls[0][1]["approvalPolicy"] == "never"
    assert calls[0][1]["developerInstructions"]
    assert workers[0].closed


@pytest.mark.parametrize(
    "mode,tools,message",
    [
        ("plain", ["builtin_tools"], "plain"),
        ("tool_calling", [], "builtin_tools"),
        ("tool_calling", ["lookup"], "builtin_tools"),
    ],
)
def test_unsupported_native_modes_and_missing_opt_in_fail_before_worker(
    native, mode, tools, message
):
    with pytest.raises(ValueError, match=message):
        node(mode=mode, tools=tools).forward({})
    assert native[2] == []


@pytest.mark.parametrize("value", [1, -1, 0, True, "bad", None, {}])
def test_explicit_native_iteration_cap_warns_and_is_ignored(native, value):
    agent = node()
    agent.env["max_tool_iterations"] = value
    with pytest.warns(UserWarning, match="max_tool_iterations.*ignored"):
        assert agent.forward({}).value == "answer"


def test_omitted_native_iteration_cap_does_not_warn(native):
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        assert node().forward({}).value == "answer"
    assert caught == []


@pytest.mark.parametrize("name", ["", "has.dot", "é", "a" * 129, "mcp", "mcp__tool"])
def test_invalid_dynamic_names_fail_before_worker(native, name):
    with pytest.raises(ValueError, match="tool"):
        node(tools=["builtin_tools", name]).forward({})
    assert native[2] == []


@pytest.mark.parametrize(
    "tools", [["builtin_tools", "builtin_tools"], ["builtin_tools", "lookup", "lookup"]]
)
def test_duplicate_tool_grants_fail_before_worker(native, tools):
    with pytest.raises(ValueError, match="duplicate"):
        node(tools=tools).forward({})
    assert native[2] == []


def test_dynamic_tools_use_authored_names_and_only_public_schema(native, monkeypatch):
    with install_ask_user(lambda **kwargs: "yes"):
        assert node(tools=["builtin_tools", "ask_user"]).forward({}).value == "answer"
    declaration = native[1][0][1]["dynamicTools"]
    assert len(declaration) == 1
    assert declaration[0]["name"] == "ask_user"
    assert declaration[0]["type"] == "function"
    assert set(declaration[0]["inputSchema"]["properties"]) == {"question"}


def test_native_tool_callback_uses_existing_public_boundary(native, monkeypatch):
    _tool(monkeypatch)
    outputs = []
    native[0]["run"] = lambda worker: outputs.append(_call(worker))
    assert node(tools=["builtin_tools", "lookup"]).forward({}).value == "answer"
    assert outputs == [
        {
            "contentItems": [{"type": "inputText", "text": "found needle"}],
            "success": True,
        }
    ]


@pytest.mark.parametrize(
    "name,extra",
    [("not-granted", {}), ("ask_user", {}), ("lookup", {"namespace": "forged"})],
)
def test_ungranted_or_namespaced_callback_never_invokes_host(
    native, monkeypatch, name, extra
):
    _tool(monkeypatch, function=lambda query: pytest.fail("tool reached"))
    native[0]["run"] = lambda worker: _call(worker, name=name, **extra)
    with pytest.raises(ValueError, match="granted|namespace"):
        node(tools=["builtin_tools", "lookup"]).forward({})
    assert native[2][0].closed


@pytest.mark.parametrize(
    "method,expected",
    [
        ("item/commandExecution/requestApproval", {"decision": "decline"}),
        ("item/fileChange/requestApproval", {"decision": "decline"}),
        ("item/permissions/requestApproval", {"permissions": {}, "scope": "turn"}),
        ("mcpServer/elicitation/request", {"action": "decline"}),
        ("item/tool/requestUserInput", {"answers": {}}),
    ],
)
def test_native_approval_callbacks_are_noninteractive_denials(native, method, expected):
    returned = []
    native[0]["run"] = lambda worker: returned.append(
        worker.options["callback"](method, {})
    )
    node().forward({})
    assert returned == [expected]


@pytest.mark.parametrize(
    "method", ["unknown/request", "account/chatgptAuthTokens/refresh"]
)
def test_unknown_and_token_refresh_callbacks_fail_explicitly(native, method):
    native[0]["run"] = lambda worker: worker.options["callback"](
        method, {"secret": "do-not-echo"}
    )
    with pytest.raises(ValueError, match="unsupported") as caught:
        node().forward({})
    assert "do-not-echo" not in str(caught.value)


@pytest.mark.parametrize(
    "texts,expected",
    [
        ([(None, "legacy"), ("commentary", "not-final")], "legacy"),
        (
            [("final_answer", "first"), (None, "fallback"), ("final_answer", "last")],
            "last",
        ),
    ],
)
def test_output_prefers_last_explicit_final_else_last_absent_phase(
    native, texts, expected
):
    native[0]["texts"] = texts
    assert node().forward({}).value == expected


def test_commentary_and_terminal_items_never_supply_final_output(native):
    native[0]["texts"] = [("commentary", "not-final")]
    with pytest.raises(RuntimeError, match="final"):
        node().forward({})


@pytest.mark.parametrize("status", ["failed", "interrupted"])
def test_failed_native_terminal_never_returns_visible_text(native, status):
    native[0]["status"] = status
    native[0]["error"] = {"message": "secret-native-error"}
    with pytest.raises(RuntimeError, match=status) as caught:
        node().forward({})
    assert "secret-native-error" not in str(caught.value)


def test_native_session_uses_memory_reference_and_current_settings(native):
    from agent_composer.nodes.agent import codex

    memory = SessionMemory()
    for index in range(2):
        with harness(memory) as h:
            with h.frame.open_turn(
                request(str(index), compatibility=codex.compatibility(())),
                check_stopped=lambda: None,
            ) as turn:
                assert (
                    node().forward({}, begin_conversation=lambda _: turn).value
                    == "answer"
                )
    assert [call[0] for call in native[1]] == ["start", "turn", "resume", "turn"]
    view = snapshot(memory)["conversations"][0]
    assert view["reference"] == {"provider": "codex", "session_id": "thread"}
    assert all(effect["status"] == "completed" for effect in view["effects"])


def test_stop_after_native_terminal_retains_completed_truth_once(native):
    memory, stop = SessionMemory(), Event()
    native[0]["after_terminal"] = stop.set
    with harness(memory) as h:
        with pytest.raises(_HostStop):
            with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
                node().forward({}, stop_event=stop, begin_conversation=lambda _: turn)
    effect = snapshot(memory)["conversations"][0]["effects"][-1]
    assert effect["kind"] == "model" and effect["status"] == "completed"
    assert len(effect["observation_ordinals"]) == 2
    assert native[2][0].closed


@pytest.mark.parametrize(
    "typ,text,expected",
    [
        (Type.scalar(ValueKind.INTEGER), '{"value": 7}', 7),
        (
            Type(kind=ValueKind.STRING, tags=frozenset({"yes", "no"})),
            '{"value": "yes"}',
            "yes",
        ),
        (Type(kind=ValueKind.LIST_INTEGER), '{"items": [1, 2]}', [1, 2]),
        (
            Type(
                kind=ValueKind.OBJECT,
                fields={"n": Type.scalar(ValueKind.INTEGER)},
                required=frozenset({"n"}),
            ),
            '{"n": 3}',
            {"n": 3},
        ),
    ],
)
def test_native_typed_output_uses_existing_schema_and_unwrap(
    native, typ, text, expected
):
    native[0]["texts"] = [("final_answer", text)]
    agent = node()
    agent.output_contract = ValueContract(typ)
    assert agent.forward({}).value == expected
    assert native[1][-1][-1]["outputSchema"]["type"] == "object"


def test_native_bare_string_has_no_schema(native):
    agent = node()
    agent.output_contract = ValueContract(Type.scalar(ValueKind.STRING))
    assert agent.forward({}).value == "answer"
    assert "outputSchema" not in native[1][-1][-1]


def test_native_structured_retry_stays_on_thread_and_retains_actual_replies(native):
    memory = SessionMemory()
    native[0]["replies"] = ["not json", '{"value": "bad"}', '{"value": 4}']
    agent = node(retries=2)
    agent.output_contract = ValueContract(Type.scalar(ValueKind.INTEGER))
    with harness(memory) as h:
        with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
            assert agent.forward({}, begin_conversation=lambda _: turn).value == 4
    assert [call[0] for call in native[1]] == ["start", "turn", "turn", "turn"]
    assert all(call[1] == "thread" for call in native[1][1:])
    view = snapshot(memory)["conversations"][0]
    assert [effect["status"] for effect in view["effects"]] == ["completed"] * 4
    replies = [
        item["payload"]["params"]["item"]["text"]
        for item in view["items"]
        if item["payload"].get("event") == "item/completed"
    ]
    assert replies == ["not json", '{"value": "bad"}', '{"value": 4}']


def test_exhausted_native_validation_retains_exact_generated_origin(native):
    native[0]["texts"] = [("final_answer", '{"n": "bad"}')]
    agent = node(retries=1)
    agent.origin = OriginRef("node")
    typ = Type(
        kind=ValueKind.OBJECT,
        fields={"n": Type.scalar(ValueKind.INTEGER)},
        required=frozenset({"n"}),
    )
    agent.output_contract = ValueContract(
        typ,
        ContractOrigins(
            default=OriginRef("output"), entries=((("n",), OriginRef("field")),)
        ),
    )
    with pytest.raises(ValidationError) as caught:
        agent.forward({})
    assert len([call for call in native[1] if call[0] == "turn"]) == 2
    assert agent.exception_origin(caught.value) == OriginRef("field")


def test_tool_validation_error_does_not_retry_or_claim_output_origin(
    native, monkeypatch
):
    def lookup(value: int):
        return value

    _tool(monkeypatch, function=lookup)
    native[0]["run"] = lambda worker: _call(worker, arguments={"value": "bad"})
    agent = node(tools=["builtin_tools", "lookup"], retries=4)
    agent.origin = OriginRef("node")
    agent.output_contract = ValueContract(
        Type.scalar(ValueKind.INTEGER), ContractOrigins(default=OriginRef("output"))
    )
    with pytest.raises(ValidationError) as caught:
        agent.forward({})
    assert agent.exception_origin(caught.value) == OriginRef("node")
    assert len([call for call in native[1] if call[0] == "turn"]) == 1


@pytest.mark.parametrize("operation", ["model", "tool"])
def test_external_exceptions_preserve_identity_and_sanitized_memory(
    native, monkeypatch, operation
):
    primary = RuntimeError("sensitive-external-failure")

    def explode(*_args, **_kwargs):
        raise primary

    agent = node()
    if operation == "tool":
        _tool(monkeypatch, function=explode)
        agent.tools.append("lookup")
        native[0]["run"] = lambda worker: _call(worker, arguments={})
    else:
        native[0]["run"] = explode
    memory = SessionMemory()
    with harness(memory) as h:
        with pytest.raises(RuntimeError) as caught:
            with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
                agent.forward({}, begin_conversation=lambda _: turn)
    assert caught.value is primary
    view = snapshot(memory)["conversations"][0]
    assert view["effects"][-1]["status"] == "failed"
    assert "sensitive-external-failure" not in str(view)
    assert native[2][0].closed


def test_stop_after_real_tool_result_retains_success_before_cancel(native, monkeypatch):
    stop, memory = Event(), SessionMemory()

    def lookup(query: str):
        stop.set()
        return "already executed"

    _tool(monkeypatch, function=lookup)
    native[0]["run"] = lambda worker: _call(worker)
    with harness(memory) as h:
        with pytest.raises(_HostStop):
            with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
                node(tools=["builtin_tools", "lookup"]).forward(
                    {}, stop_event=stop, begin_conversation=lambda _: turn
                )
    effects = snapshot(memory)["conversations"][0]["effects"]
    assert [(effect["kind"], effect["status"]) for effect in effects] == [
        ("native_session", "completed"),
        ("model", "interrupted"),
        ("tool", "completed"),
    ]


def test_completed_tool_with_unrepresentable_result_does_not_become_failed(
    native, monkeypatch
):
    primary = ValueError("cannot represent")

    class Result:
        def __str__(self):
            raise primary

    _tool(monkeypatch, function=lambda query: Result())
    native[0]["run"] = lambda worker: _call(worker)
    memory = SessionMemory()
    with harness(memory) as h:
        with pytest.raises(ValueError) as caught:
            with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
                node(tools=["builtin_tools", "lookup"]).forward(
                    {}, begin_conversation=lambda _: turn
                )
    assert caught.value is primary
    effects = snapshot(memory)["conversations"][0]["effects"]
    assert effects[-1]["kind"] == "tool" and effects[-1]["status"] == "completed"


def test_memory_faults_are_capability_faults_without_native_retry(native, monkeypatch):
    memory, primary = SessionMemory(), RuntimeError("storage unavailable")

    def fail(*_args, **_kwargs):
        raise primary

    with harness(memory) as h:
        with pytest.raises(_CapabilityFault) as caught:
            with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
                monkeypatch.setattr(turn, "request", fail)
                node().forward({}, begin_conversation=lambda _: turn)
    assert caught.value.original is primary
    assert native[1] == []


def test_missing_native_capability_fails_without_fallback(native):
    class Turn:
        pass

    with pytest.raises(_CapabilityFault, match="native_session"):
        node().forward({}, begin_conversation=lambda _: Turn())
    assert native[1] == []


def test_native_resume_failure_preserves_identity_without_start_fallback(native):
    memory = SessionMemory()
    for index in range(2):
        with harness(memory) as h:
            if index:
                primary = RuntimeError("thread missing")
                native[0]["resume_error"] = primary
                with pytest.raises(RuntimeError) as caught:
                    with h.frame.open_turn(
                        request(str(index)), check_stopped=lambda: None
                    ) as turn:
                        node().forward({}, begin_conversation=lambda _: turn)
                assert caught.value is primary
            else:
                with h.frame.open_turn(
                    request(str(index)), check_stopped=lambda: None
                ) as turn:
                    node().forward({}, begin_conversation=lambda _: turn)
    assert [call[0] for call in native[1]] == ["start", "turn", "resume"]


def test_observed_thread_reference_survives_stop_before_method_returns(native):
    stop, memory = Event(), SessionMemory()
    native[0]["after_result"] = lambda operation: (
        stop.set() if operation == "start_thread" else None
    )
    with harness(memory) as h:
        with pytest.raises(_HostStop):
            with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
                node().forward({}, stop_event=stop, begin_conversation=lambda _: turn)
    view = snapshot(memory)["conversations"][0]
    assert view["reference"]["session_id"] == "thread"
    assert view["effects"][0]["status"] == "completed"
    assert len(view["effects"][0]["observation_ordinals"]) == 1


def test_only_verified_unauthorized_terminal_claims_login_failure(native):
    native[0]["status"] = "failed"
    native[0]["error"] = {"message": "ignored", "codexErrorInfo": "unauthorized"}
    with pytest.raises(RuntimeError, match="authentication.*sign in"):
        node().forward({})


def test_current_recall_and_instruction_changes_apply_on_resume(native):
    memory = SessionMemory()
    for index in range(2):
        with harness(memory) as h:
            with h.frame.open_turn(
                request(str(index)), check_stopped=lambda: None
            ) as turn:
                agent = node()
                agent.prompt = f"question {index}"
                agent.llm_config.update(
                    model=f"model-{index}",
                    cwd=f"/work-{index}",
                    sandbox="workspace-write",
                )
                agent.forward(
                    {},
                    begin_conversation=lambda _: turn,
                    recall_memory=lambda *_: (
                        MemorySection(h.frame._spec.ref, f"note-{index}"),
                    ),
                )
    connections = [call for call in native[1] if call[0] != "turn"]
    assert "note-0" in connections[0][-1]["developerInstructions"]
    assert "note-1" in connections[1][-1]["developerInstructions"]
    assert connections[1][-1]["cwd"] == "/work-1"
    assert connections[1][-1]["model"] == "model-1"
    assert "dynamicTools" not in connections[1][-1]


def test_descriptor_is_detached_and_order_independent(native, monkeypatch):
    from agent_composer.nodes.agent import codex

    _tool(monkeypatch, "left")
    _tool(monkeypatch, "right")
    bridge = codex.prepare_tools(["builtin_tools", "left", "right"])
    descriptor = codex.compatibility(bridge)
    assert descriptor == codex.compatibility(tuple(reversed(bridge)))
    bridge[0]["inputSchema"]["properties"].clear()
    assert descriptor["tools"][0]["inputSchema"]["properties"]


@pytest.mark.parametrize("change", ["name", "description", "schema"])
def test_changed_dynamic_tool_contract_requires_reset_before_native_effect(
    native, monkeypatch, change
):
    from agent_composer.nodes.agent import codex
    from agent_composer.memory.contracts import MemoryOperationError

    _tool(monkeypatch)
    memory = SessionMemory()
    for index in range(2):
        if index:
            if change == "name":
                _tool(monkeypatch, "other")
            elif change == "description":
                TOOL_REGISTRY["lookup"].description = "Changed description."
            else:

                def lookup(other: int):
                    return other

                _tool(monkeypatch, function=lookup)
        agent = node(
            tools=["builtin_tools", "other" if index and change == "name" else "lookup"]
        )
        bridge = codex.prepare_tools(agent.tools)
        with harness(memory) as h:
            if index:
                before = deepcopy(native[1])
                with pytest.raises(MemoryOperationError, match="compatib"):
                    with h.frame.open_turn(
                        request(str(index), compatibility=codex.compatibility(bridge)),
                        check_stopped=lambda: None,
                    ) as turn:
                        agent.forward({}, begin_conversation=lambda _: turn)
                assert native[1] == before
            else:
                with h.frame.open_turn(
                    request(str(index), compatibility=codex.compatibility(bridge)),
                    check_stopped=lambda: None,
                ) as turn:
                    agent.forward({}, begin_conversation=lambda _: turn)


def test_tool_callback_rechecks_current_grants(native, monkeypatch):
    from agent_composer.nodes.agent import codex
    from agent_composer.nodes.agent.modes.common import AgentRunContext

    _tool(monkeypatch, function=lambda query: pytest.fail("revoked tool invoked"))
    ctx = AgentRunContext(
        node_id="native",
        prompt="question",
        tools=["builtin_tools", "lookup"],
        llm_config=node().llm_config,
    )
    bridge = codex.prepare_tools(ctx.tools)

    def revoke(worker):
        ctx.tools.remove("lookup")
        _call(worker)

    native[0]["run"] = revoke
    with pytest.raises(ValueError, match="not granted"):
        codex.execute(ctx, bridge=bridge)


def test_injected_tool_metadata_is_rejected_by_existing_public_boundary(native):
    calls = []
    native[0]["run"] = lambda worker: _call(
        worker, name="ask_user", arguments={"question": "hello", "node_id": "forged"}
    )
    with install_ask_user(lambda **kwargs: calls.append(kwargs)):
        with pytest.raises(ValueError, match="unsupported argument"):
            node(tools=["builtin_tools", "ask_user"]).forward({})
    assert calls == []


@pytest.mark.parametrize(
    "error", [ValueError("transport-like"), KeyboardInterrupt(), FutureCancelledError()]
)
def test_structured_native_never_retries_external_faults(native, error):
    def fail(_worker):
        raise error

    native[0]["run"] = fail
    agent = node(retries=3)
    agent.output_contract = ValueContract(Type.scalar(ValueKind.INTEGER))
    with pytest.raises(type(error)) as caught:
        agent.forward({})
    assert caught.value is error
    assert len([call for call in native[1] if call[0] == "turn"]) == 1


@pytest.mark.parametrize("operation", ["start_thread", "run_turn"])
def test_stop_raised_by_acceptance_observer_keeps_completed_status(
    native, monkeypatch, operation
):
    stop, memory = Event(), SessionMemory()
    with harness(memory) as h:
        with pytest.raises(_HostStop):
            with h.frame.open_turn(
                request(),
                check_stopped=lambda: node()
                ._ctx("unused", stop_event=stop)
                .check_stopped(),
            ) as turn:
                original = turn.observe

                def observe(ordinal, status, **kwargs):
                    if status == "completed" and (
                        (operation == "start_thread") == ("reference" in kwargs)
                    ):
                        stop.set()
                    return original(ordinal, status, **kwargs)

                monkeypatch.setattr(turn, "observe", observe)
                node().forward({}, stop_event=stop, begin_conversation=lambda _: turn)
    view = snapshot(memory)["conversations"][0]
    assert all(effect["status"] == "completed" for effect in view["effects"])
    assert view["reference"]["session_id"] == "thread"


def test_observation_failure_preserves_initiating_tool_exception(native, monkeypatch):
    primary = RuntimeError("primary tool failure")

    def explode(query: str):
        raise primary

    _tool(monkeypatch, function=explode)
    native[0]["run"] = lambda worker: _call(worker)
    memory = SessionMemory()
    with harness(memory) as h:
        with pytest.raises(_CapabilityFault) as caught:
            with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
                original = turn.observe

                def observe(ordinal, status, **kwargs):
                    if status == "failed":
                        raise OSError("storage failed")
                    return original(ordinal, status, **kwargs)

                monkeypatch.setattr(turn, "observe", observe)
                node(tools=["builtin_tools", "lookup"]).forward(
                    {}, begin_conversation=lambda _: turn
                )
    assert caught.value.original is primary


def test_invalid_native_effective_config_fails_before_worker(native):
    agent = node()
    agent.llm_config["cwd"] = "relative"
    with pytest.raises(ValueError, match="resolved absolute cwd"):
        agent.forward({})
    assert native[2] == []


def test_observational_trace_does_not_replace_native_output_contract_check(native):
    import math

    class Trace:
        def record_prompt(self, *_args):
            pass

        def record_interaction(self, _interaction):
            pass

    native[0]["texts"] = [("final_answer", '{"value": NaN}')]
    agent = node()
    agent.output_contract = ValueContract(Type.scalar(ValueKind.NUMBER))
    # The existing typed-output boundary, not optional observational evidence,
    # owns rejection of non-finite generated values.
    assert math.isnan(agent.forward({}, evaluation_trace=Trace()).value)


@pytest.mark.parametrize(
    "status,error_code",
    [
        ("completed", None),
        ("failed", "rateLimitExceeded"),
        ("interrupted", {"httpConnectionFailed": {"httpStatusCode": 503}}),
    ],
)
def test_terminal_identity_and_classification_are_retained_once_before_stop(
    native, status, error_code
):
    memory, stop = SessionMemory(), Event()
    native[0].update(
        status=status,
        error={
            "codexErrorInfo": error_code,
            "message": "private-error-message",
            "additionalDetails": "private-error-details",
        },
        after_terminal=stop.set,
    )
    with harness(memory) as h:
        with pytest.raises(_HostStop):
            with h.frame.open_turn(request(), check_stopped=lambda: None) as turn:
                node().forward({}, stop_event=stop, begin_conversation=lambda _: turn)
    view = snapshot(memory)["conversations"][0]
    terminals = [
        item["payload"]["terminal"]
        for item in view["items"]
        if "terminal" in item["payload"]
    ]
    assert terminals == [
        {
            "threadId": "thread",
            "turn": {
                "id": "turn",
                "status": status,
                "codexErrorInfo": (
                    "httpConnectionFailed"
                    if isinstance(error_code, dict)
                    else error_code
                ),
            },
        }
    ]
    assert "private-error-message" not in str(view)
    assert "private-error-details" not in str(view)
    assert view["effects"][-1]["status"] == status


@pytest.mark.parametrize("legacy_exception", [False, True])
@pytest.mark.parametrize("through_flow", [False, True])
def test_generic_sdk_failure_has_actionable_normal_diagnostics_without_losing_identity(
    native, legacy_exception, through_flow
):
    from agent_composer import load_flow, run_flow
    from agent_composer.llm_clients.codex.protocol import CodexWorkerError

    class WithoutNotes(CodexWorkerError):
        def __getattribute__(self, name):
            if name == "add_note":
                raise AttributeError(name)
            return super().__getattribute__(name)

    primary = (WithoutNotes if legacy_exception else CodexWorkerError)("sdk")
    primary.args = ("private SDK request detail",)

    def fail(_worker):
        raise primary

    native[0]["run"] = fail
    if through_flow:
        flow = load_flow("""id: native-error
name: native_error
nodes:
  answer:
    kind: agent
    mode: tool_calling
    tools: [builtin_tools]
    prompt: question
    llm_config: {provider: codex, model: offline-model}
    output: str
output: ${answer.output}
""")
        result = run_flow(flow, {})
        assert result.status == "failed"
        message = flow.diagnose(result.failure).message
        assert result.failure.message == message == str(primary)
    else:
        with pytest.raises(CodexWorkerError) as caught:
            node().forward({})
        assert caught.value is primary
        message = str(caught.value)
    assert primary.code == "sdk"
    assert all(word in message for word in ("runtime", "configuration", "login"))
    assert "private SDK request detail" not in message
    assert "authentication was rejected" not in message
    assert "no API fallback" in message
