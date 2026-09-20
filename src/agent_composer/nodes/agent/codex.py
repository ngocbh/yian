"""Native AGENT execution over the isolated SDK and memory-owned thread references.

Composer grants only the named dynamic tools. The required ``builtin_tools``
marker separately opts into Codex's configured native, MCP, and app environment.
Host tool execution is not constrained by the native filesystem sandbox.
"""

from __future__ import annotations

from asyncio import CancelledError as AsyncCancelledError
from concurrent.futures import CancelledError as FutureCancelledError
import json
import os.path
import re
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from agent_composer.llm_clients.codex.client import CodexWorkerClient
from agent_composer.llm_clients.codex.protocol import CodexWorkerError, detached
from agent_composer.llm_clients.config import LLMConfig
from agent_composer.memory.contracts import (
    ConversationItem,
    MemoryOperationFailure,
    ModelValueInteraction,
    ToolCallInteraction,
)
from agent_composer.nodes.agent.conversation import (
    _memory_boundary,
    _observe_preserving_error,
)
from agent_composer.nodes.agent.modes.common import (
    AgentLoopError,
    AgentRunContext,
    DEFAULT_SYSTEM,
)
from agent_composer.nodes.agent.modes.utils import (
    record_model_text,
    record_tool_result,
    system_prompt_with_memory,
)
from agent_composer.nodes.agent.structured import (
    _strip_code_fence,
    _unwrap,
    type_to_schema,
)
from agent_composer.nodes.base import Output, _CapabilityFault, _HostStop
from agent_composer.tools import invoke_tool, resolve_tools

MESSAGE_FORMAT = "agent-composer.codex-events.v1"
_NAME = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


def prepare_tools(tool_ids: list[str]) -> tuple[dict, ...]:
    """Validate explicit native opt-in and detach public Composer declarations."""
    if "builtin_tools" not in tool_ids:
        raise ValueError(
            "Codex requires tools: [builtin_tools] to opt into its configured tool environment"
        )
    seen = set()
    for name in tool_ids:
        if (
            type(name) is not str
            or not _NAME.fullmatch(name)
            or name == "mcp"
            or name.startswith("mcp__")
        ):
            raise ValueError(
                "Codex tool names must be flat ASCII names of 1–128 characters, without reserved mcp names"
            )
        if name in seen:
            raise ValueError(f"duplicate Codex tool grant {name!r}")
        seen.add(name)
    names = [name for name in tool_ids if name != "builtin_tools"]
    declarations = []
    for name, tool in zip(names, resolve_tools(names)):
        schema = tool.tool_call_schema
        if not isinstance(schema, type):
            raise TypeError(
                f"tool {name!r} requires a validated Pydantic public argument schema"
            )
        if hasattr(schema, "model_fields") and callable(
            getattr(schema, "model_json_schema", None)
        ):
            wire = schema.model_json_schema()
        elif hasattr(schema, "__fields__") and callable(
            getattr(schema, "schema", None)
        ):
            wire = schema.schema()
        else:
            raise TypeError(
                f"tool {name!r} requires a validated Pydantic public argument schema"
            )
        if (
            type(wire) is not dict
            or wire.get("type") != "object"
            or type(tool.description) is not str
        ):
            raise ValueError(f"tool {name!r} has an invalid public declaration")
        declarations.append(
            detached(
                {
                    "type": "function",
                    "name": name,
                    "description": tool.description,
                    "inputSchema": wire,
                }
            )
        )
    return tuple(declarations)


def compatibility(bridge: tuple[dict, ...]) -> dict:
    """Pin native message format and the dynamic tool environment across resume."""
    return detached(
        {
            "provider": "codex",
            "message_format": MESSAGE_FORMAT,
            "tools": sorted(bridge, key=lambda tool: tool["name"]),
        }
    )


def _item(payload: dict) -> ConversationItem:
    return ConversationItem(uuid4().hex, MESSAGE_FORMAT, payload)


def _failure(turn, ordinal, operation, primary):
    """Publish sanitized external failure evidence without replacing its identity."""
    if turn is None:
        return
    interrupted = isinstance(
        primary, (_HostStop, AsyncCancelledError, FutureCancelledError)
    ) or not isinstance(primary, Exception)
    status = "interrupted" if interrupted else "failed"
    failure = MemoryOperationFailure(
        operation,
        f"codex.{operation}.{status}",
        f"Codex {operation} {status} ({type(primary).__name__}).",
        False,
    )
    _observe_preserving_error(turn, ordinal, status, primary, failure=failure)


def _terminal_failure(status: str, error: Any) -> AgentLoopError:
    if (
        status == "failed"
        and type(error) is dict
        and error.get("codexErrorInfo") == "unauthorized"
    ):
        return AgentLoopError(
            "Codex turn failed: authentication was rejected; sign in with Codex and retry."
        )
    return AgentLoopError(
        f"Codex turn {status}; check the native Codex configuration and runtime."
    )


class _Invocation:
    """Keep one worker's transient callbacks, never the stored conversation."""

    def __init__(self, ctx: AgentRunContext, bridge: tuple[dict, ...]):
        self.ctx, self.bridge = ctx, bridge
        self.turn = ctx.conversation_turn
        self.thread_id = None
        self.accept = None
        self.ordinal = None
        self.terminal = None
        self.messages = []

    def result(self, operation: str, payload: dict) -> None:
        if operation in {"start_thread", "resume_thread"}:
            thread_id = payload["thread"]["id"]
            if type(thread_id) is not str or not thread_id:
                raise ValueError("Codex returned an invalid native thread ID")
            self.thread_id = thread_id
            if self.accept is not None:
                with _memory_boundary():
                    self.accept(thread_id)
        elif operation == "run_turn":
            self.finish(payload)

    def event(self, method: str, payload: dict) -> None:
        if self.ordinal is None:
            return
        if method == "turn/completed":
            self.finish(payload)
        elif method == "item/completed":
            item = payload.get("item")
            if type(item) is not dict:
                raise ValueError("Codex completed item is malformed")
            if self.turn is not None:
                with _memory_boundary():
                    self.turn.observe(
                        self.ordinal,
                        "pending",
                        items=(_item({"event": method, "params": payload}),),
                    )
            if item.get("type") == "agentMessage":
                if type(item.get("text")) is not str:
                    raise ValueError("Codex agent message has no text")
                self.messages.append((item.get("phase"), item["text"]))

    def finish(self, payload: dict) -> None:
        if self.ordinal is None or self.terminal is not None:
            return
        native_turn = payload.get("turn")
        if type(native_turn) is not dict or native_turn.get("status") not in {
            "completed",
            "failed",
            "interrupted",
        }:
            raise ValueError("Codex returned an invalid terminal turn")
        status = native_turn["status"]
        error = (
            None
            if status == "completed"
            else _terminal_failure(status, native_turn.get("error"))
        )
        # A terminal notification is already actual truth. Set this before
        # observe, which may persist successfully and then raise a stop signal.
        self.terminal = (status, error)
        if self.turn is not None:
            native_error = native_turn.get("error")
            classification = (
                native_error.get("codexErrorInfo")
                if type(native_error) is dict
                else None
            )
            if type(classification) is dict and len(classification) == 1:
                classification = next(iter(classification))
            if type(classification) is not str:
                classification = None
            failure = (
                None
                if error is None
                else MemoryOperationFailure(
                    "model", f"codex.model.{status}", str(error), False
                )
            )
            with _memory_boundary():
                terminal = _item(
                    {
                        "terminal": {
                            "threadId": payload["threadId"],
                            "turn": {
                                "id": native_turn["id"],
                                "status": status,
                                "codexErrorInfo": classification,
                            },
                        }
                    }
                )
                self.turn.observe(
                    self.ordinal, status, items=(terminal,), failure=failure
                )

    def callback(self, method: str, payload: dict | None) -> dict:
        self.ctx.check_stopped()
        if method in {
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
        }:
            return {"decision": "decline"}
        if method == "item/permissions/requestApproval":
            return {"permissions": {}, "scope": "turn"}
        if method == "mcpServer/elicitation/request":
            return {"action": "decline"}
        if method == "item/tool/requestUserInput":
            return {"answers": {}}
        if method != "item/tool/call":
            raise ValueError("Codex requested an unsupported host callback")
        if type(payload) is not dict or payload.get("namespace") not in (None, ""):
            raise ValueError("Codex Composer tool callback requires an empty namespace")
        name, arguments = payload.get("tool"), payload.get("arguments")
        if (
            type(name) is not str
            or name == "builtin_tools"
            or name not in self.ctx.tools
            or name not in {tool["name"] for tool in self.bridge}
        ):
            raise ValueError(
                f"Codex requested a Composer tool that is not granted: {name!r}"
            )
        if type(arguments) is not dict:
            raise ValueError("Codex Composer tool arguments must be an object")
        arguments = detached(arguments)
        if self.ctx.evaluation_trace is not None:
            self.ctx.evaluation_trace.record_interaction(
                ToolCallInteraction(name, arguments)
            )
        ordinal = None
        if self.turn is not None:
            with _memory_boundary():
                ordinal = self.turn.request(
                    "tool",
                    {
                        "name": name,
                        "arguments": arguments,
                        "call_id": payload.get("callId"),
                        "thread_id": self.thread_id,
                    },
                )
        try:
            result = invoke_tool(name, arguments)
        except BaseException as primary:
            _failure(self.turn, ordinal, "tool", primary)
            raise
        try:
            text = str(result)
            if self.turn is not None:
                with _memory_boundary():
                    item = _item(
                        {"tool": name, "call_id": payload.get("callId"), "result": text}
                    )
        except BaseException as primary:
            if self.turn is not None:
                _observe_preserving_error(self.turn, ordinal, "completed", primary)
            raise
        if self.turn is not None:
            with _memory_boundary():
                self.turn.observe(ordinal, "completed", items=(item,))
        record_tool_result(self.ctx.evaluation_trace, name, text)
        self.ctx.check_stopped()
        return {"contentItems": [{"type": "inputText", "text": text}], "success": True}

    def connect(self, worker: CodexWorkerClient, params: dict) -> None:
        callback_error = None

        def connect(accept, thread_id=None):
            nonlocal callback_error
            self.accept = accept
            try:
                if thread_id is None:
                    worker.start_thread({**params, "dynamicTools": list(self.bridge)})
                else:
                    worker.resume_thread(thread_id, params)
            except BaseException as exc:
                callback_error = exc
                raise
            finally:
                self.accept = None

        if self.turn is None:
            connect(None)
        else:
            try:
                with _memory_boundary():
                    native_session = getattr(self.turn, "native_session", None)
                    if not callable(native_session):
                        raise TypeError(
                            "Codex SessionMemory requires the native_session capability"
                        )
                    native_session(
                        provider="codex",
                        start=connect,
                        resume=lambda identity, accept: connect(accept, identity),
                    )
            except _CapabilityFault as fault:
                if callback_error is not None and fault.original is callback_error:
                    raise callback_error
                raise
        if self.thread_id is None:
            raise ValueError("Codex did not publish its native thread response")

    def run(self, worker: CodexWorkerClient, prompt: str, params: dict) -> str:
        self.ctx.check_stopped()
        self.messages, self.terminal = [], None
        self.ordinal = 0
        if self.turn is not None:
            with _memory_boundary():
                self.ordinal = self.turn.request(
                    "model",
                    {"thread_id": self.thread_id, "params": params},
                    items=(_item({"input": prompt}),),
                )
        try:
            worker.run_turn(self.thread_id, prompt, params)
        except BaseException as primary:
            if self.terminal is None:
                _failure(self.turn, self.ordinal, "model", primary)
            raise
        finally:
            self.ordinal = None
        self.ctx.check_stopped()
        if self.terminal is None:
            raise AgentLoopError("Codex returned without a terminal turn")
        status, error = self.terminal
        if status != "completed":
            raise error
        final = [text for phase, text in self.messages if phase == "final_answer"]
        fallback = [text for phase, text in self.messages if phase is None]
        if not final and not fallback:
            raise AgentLoopError("Codex completed without a final answer message")
        text = (final or fallback)[-1]
        record_model_text(self.ctx.evaluation_trace, text)
        return text


def _generate_codex(
    invoke: Callable[[str, dict], str],
    prompt: str,
    typ,
    schema: type[BaseModel],
    max_retries: int,
    check_stopped: Callable[[], None],
    params: dict,
):
    """Correct only surfaced JSON/schema deviations, on the same native thread."""
    last_err = None
    for _ in range(max_retries + 1):
        last_err = None
        text = invoke(prompt, params)
        try:
            obj = schema.model_validate(json.loads(_strip_code_fence(text)))
        except (json.JSONDecodeError, ValidationError) as err:
            last_err = err
            check_stopped()
            prompt = f"Your previous output was invalid: {err}. Respond with ONLY a JSON object matching the output schema."
        else:
            return _unwrap(obj, typ)
    raise last_err


def execute(ctx: AgentRunContext, *, bridge: tuple[dict, ...]) -> Output:
    """Execute one native node invocation without an API-model fallback."""
    config = dict(ctx.llm_config or {})
    config["provider"] = "codex"
    normalized = LLMConfig(**config)
    if (
        not normalized.cwd
        or not os.path.isabs(normalized.cwd)
        or normalized.sandbox is None
    ):
        raise ValueError(
            "Codex requires a resolved absolute cwd and explicit sandbox; use run_flow to resolve configuration"
        )
    ctx.check_stopped()
    schema = type_to_schema(ctx.output_type) if ctx.output_type is not None else None
    turn_params = {} if schema is None else {"outputSchema": schema.model_json_schema()}
    params = {
        "model": normalized.model,
        "cwd": normalized.cwd,
        "sandbox": normalized.sandbox,
        "approvalPolicy": "never",
        "developerInstructions": system_prompt_with_memory(
            DEFAULT_SYSTEM, ctx.memory_sections
        ),
    }
    if ctx.evaluation_trace is not None:
        ctx.evaluation_trace.record_prompt(
            DEFAULT_SYSTEM, ctx.prompt, ctx.memory_sections
        )
    invocation = _Invocation(ctx, bridge)
    try:
        with CodexWorkerClient(
            config={"cwd": normalized.cwd},
            callback=invocation.callback,
            on_event=invocation.event,
            on_result=invocation.result,
            check_stopped=ctx.check_stopped,
        ) as worker:
            invocation.connect(worker, params)
            if schema is None:
                value = invocation.run(worker, ctx.prompt, turn_params)
            else:
                value = _generate_codex(
                    lambda prompt, options: invocation.run(worker, prompt, options),
                    ctx.prompt,
                    ctx.output_type,
                    schema,
                    ctx.retries,
                    ctx.check_stopped,
                    turn_params,
                )
                if ctx.evaluation_trace is not None:
                    try:
                        interaction = ModelValueInteraction(value)
                    except (TypeError, ValueError, RecursionError):
                        # The ordinary output contract still owns rejection of
                        # values that observational evidence cannot represent.
                        pass
                    else:
                        ctx.evaluation_trace.record_interaction(interaction)
    except CodexWorkerError as error:
        if error.code == "sdk":
            # Normal diagnostics use str(error), including on Python 3.10.
            # Keep the original exception identity and discard opaque SDK text.
            error.args = (
                "The Codex SDK operation failed. Check native runtime availability, "
                "Codex configuration, and login state; no API fallback was attempted.",
            )
        raise
    ctx.check_stopped()
    return Output(value)
