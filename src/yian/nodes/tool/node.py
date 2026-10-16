"""TOOL — a direct, deterministic invocation of one tool from the run's `ToolSet`.

No LLM in the loop (distinct from an AGENT node's tool loop). The node's `input:` entries are
bindings like any other kind's, resolved by the engine's `eval_node` bind
seam — a whole-string `${ref}` resolves against the pool, a literal passes through,
and every value is deep-copied (the tool sees only its own record `inputs`, never
the live pool). The tool is looked up in the run's `ToolSet` (the global registry when
the run passed no `tools=`, otherwise only the run's mapping) and invoked through its
public schema. An optional `output:` declaration sets the node's output contract, which the
engine checks against the tool's return value on every call.

With `retries:` the node calls the tool again after a failed attempt, waiting an
exponential backoff with jitter, and records each retried attempt's failure and start
as node steps in the run log, so a resumed run continues the attempt count.
"""

import random
import threading
from types import MappingProxyType
from typing import Any

from yian.defaults import current
from yian.nodes.base import Node, NodeKind, Output, _CapabilityFault, _HostStop
from yian.nodes.contracts import HostToolCall, NodeSteps, output_contract_type
from yian.tools import ToolSet
from yian.tools.errors import PermanentToolError, ToolCheckError, TransientToolError
from yian.typesys import Type
from yian.typesys.values import shown_text


class ToolAttemptsFailed(RuntimeError):
    """A TOOL call failed after at least one retry; the message lists every attempt.

    It is raised from the last attempt's error. An attempt that started but has no
    recorded failure (the run stopped or crashed inside it) shows as `interrupted`.
    """


def _jitter() -> float:
    """Return a uniform draw in [0, 1) that spreads a backoff delay."""
    return random.random()


def _sleep(stop_event: threading.Event | None, seconds: float) -> None:
    """Wait `seconds` before the next attempt, or until the run is told to stop.

    Raises:
        `_HostStop`:
            Raised when `stop_event` is set before or during the wait.
    """
    event = stop_event if stop_event is not None else threading.Event()
    if event.wait(seconds):
        raise _HostStop(primary=False)


def _backoff(failure: int, error: Exception) -> float:
    """Return the wait before retrying after failure number `failure` (1-based).

    The delay doubles from `current().tool_retry_delay_seconds` up to
    `current().tool_retry_max_delay_seconds` and is drawn uniformly from its upper
    half. A `TransientToolError` with `retry_after` waits that long instead, up to
    the cap, with no jitter.
    """
    defaults = current()
    base, cap = defaults.tool_retry_delay_seconds, defaults.tool_retry_max_delay_seconds
    if isinstance(error, TransientToolError) and error.retry_after is not None:
        return min(cap, error.retry_after)
    # Doubling past the cap's bit length always reaches the cap (for a base of at
    # least 1), so the shift stays small however large `failure` is.
    delay = min(cap, base << min(failure - 1, cap.bit_length()))
    return delay / 2 + _jitter() * delay / 2


def _clean(text: str) -> str:
    """Return `text` with lone surrogates replaced, so it can be written as JSON."""
    return text.encode("utf-8", errors="replace").decode("utf-8")


def _error_step(error: Exception) -> dict[str, str]:
    """Return the recorded form of a failed attempt's error: its type name and message."""
    name = _clean(type(error).__name__)
    try:
        text = str(error)
    except Exception:
        text = f"<{name} object>"
    return {"type": name, "message": shown_text(_clean(text))}


def _attempts_message(
    tool_id: str, history: list[dict[str, Any]], attempt: int, error: Exception
) -> str:
    """Return the final failure's message, naming every attempt's error up to `attempt`."""
    failed = {
        step.get("attempt"): step.get("error") for step in history if step.get("event") == "failed"
    }
    parts = []
    for number in range(1, attempt):
        recorded = failed.get(number)
        shown = "interrupted" if recorded is None else f"{recorded['type']}: {recorded['message']}"
        parts.append(f"attempt {number}: {shown}")
    last = _error_step(error)
    parts.append(f"attempt {attempt}: {last['type']}: {last['message']}")
    return f"tool {tool_id!r} failed after {attempt} attempts: " + "; ".join(parts)


class ToolNode(Node):
    """
    A direct, deterministic invocation of one tool from the run's `ToolSet` (no LLM in the loop).

    The node's bound input record is passed to `ToolSet.invoke`; the tool sees only its
    own record, never the live pool. A tool that declares a `ToolContext` first
    parameter also receives one naming this node, with the call id
    `"<run_id>/<node id>"`, which a resumed re-run of the node sees again. When
    `output_contract` is set (from an optional authored `output:`), the engine
    validates every returned value against it.

    The arguments are checked once, before the first attempt. A failed call is retried
    up to `retries` times unless the tool raised `PermanentToolError` or a check yian
    runs failed (`ToolCheckError`); argument and output checks are never retried. Before
    a retry the node waits an exponential backoff (`current().tool_retry_delay_seconds`
    doubling up to `current().tool_retry_max_delay_seconds`, drawn from its upper half),
    or a `TransientToolError`'s `retry_after` up to that cap. Each attempt sees
    `ToolContext.attempt`; the call id stays the same.

    Through the engine's node steps the node records `{"event": "failed", ...}` after
    each failure it will retry and `{"event": "started", ...}` before the retry, so a
    resumed run continues the attempt count; an attempt interrupted by a crash uses up
    no retry. A failure without a retry propagates unchanged; after a retry the node
    raises `ToolAttemptsFailed`, whose message lists every attempt's error.

    `input_types` maps each input name to the type the loader proved its binding has
    (a literal's exact type, a typed binding's declared type, or a reference's inferred
    type); an input whose type is not proven is absent.

    Args:
        node_id (`str`):
            The node's unique id.
        tool_id (`str`):
            The tool id, looked up in the run's `ToolSet`; an unknown id raises at run time.
        title (`str`, *optional*, defaults to `None`):
            Display title.
        retries (`int`, *optional*, defaults to `None`):
            How many times a failed tool call may be retried, at least 0; `None`
            means `current().tool_retries`.

    Raises:
        `TypeError`:
            Raised when `retries` is not an int or `None` (a bool is refused).
        `ValueError`:
            Raised when `retries` is negative.
    """

    kind = NodeKind.TOOL
    needs_tools = True
    needs_stop_event = True
    needs_steps = True

    def __init__(
        self, node_id: str, *, tool_id: str, title=None, retries: int | None = None
    ) -> None:
        super().__init__(node_id, title=title)
        if retries is not None:
            # bool is an int subclass, so `retries=True` would otherwise mean 1.
            if type(retries) is not int:
                raise TypeError(
                    f"node {node_id!r}: retries must be a non-negative int, got {retries!r}"
                )
            if retries < 0:
                raise ValueError(
                    f"node {node_id!r}: retries must be a non-negative int, got {retries!r}"
                )
        self.tool_id = tool_id
        self.retries = current().tool_retries if retries is None else retries
        self.input_types: dict[str, Type] = {}

    def host_tool_call(self) -> HostToolCall:
        """Return this node's tool id, input names, proven input types, and output type."""
        return HostToolCall(
            tool_id=self.tool_id,
            inputs=tuple(parameter.name for parameter in self.params or ()),
            input_types=MappingProxyType(self.input_types),
            output_type=output_contract_type(self.output_contract),
        )

    def forward(
        self,
        inputs: dict,
        *,
        tools: ToolSet | None = None,
        stop_event: threading.Event | None = None,
        steps: NodeSteps | None = None,
    ) -> Output:
        """Call the tool, retrying failed attempts, and return its result.

        Args:
            inputs (`dict`):
                The bound arguments.
            tools (`ToolSet | None`, *optional*, defaults to `None`):
                The run's tools; `None` means the global registry.
            stop_event (`threading.Event | None`, *optional*, defaults to `None`):
                The run's stop event, which ends a backoff wait early.
            steps (`NodeSteps | None`, *optional*, defaults to `None`):
                This start's recorded steps. `None` retries in memory, recording
                nothing, as the first start.

        Returns:
            `Output`:
                The tool's result.

        Raises:
            `ToolAttemptsFailed`:
                Raised when the call fails for good after at least one retry.
        """
        tools = tools if tools is not None else ToolSet()
        history = list(steps.prior) if steps is not None else []
        starts = steps.starts if steps is not None else 1
        # The first attempt of each start is the node's start itself; only
        # in-evaluation retries record a `started` step.
        attempt = starts + sum(step.get("event") == "started" for step in history)
        failures = sum(step.get("event") == "failed" for step in history)

        def record(step: dict[str, Any]) -> None:
            history.append(step)
            if steps is not None:
                steps.record(step)

        tools.validate(self.tool_id, inputs)
        if history and history[-1].get("event") == "failed":
            # The previous start recorded a failure and stopped before the retry
            # finished, so this start waits that backoff again.
            _sleep(stop_event, history[-1]["retry_in"])
        while True:
            try:
                value = tools.invoke(self.tool_id, inputs, node=self.id, attempt=attempt)
            except (_HostStop, _CapabilityFault):
                raise
            except Exception as exc:
                retryable = not isinstance(exc, (PermanentToolError, ToolCheckError))
                if retryable:
                    failures += 1
                if not retryable or failures > self.retries:
                    if not any(step.get("event") == "failed" for step in history):
                        raise
                    raise ToolAttemptsFailed(
                        _attempts_message(self.tool_id, history, attempt, exc)
                    ) from exc
                delay = _backoff(failures, exc)
                error = _error_step(exc)
            else:
                return Output(value=value)
            record({"event": "failed", "attempt": attempt, "error": error, "retry_in": delay})
            _sleep(stop_event, delay)
            attempt += 1
            record({"event": "started", "attempt": attempt})
