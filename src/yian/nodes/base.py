"""The Node base contract — the single most portable idea from graphon.

A node is a **pure function of its bound input record**: it implements
`run(inputs, **caps) -> NodeResult` and returns ONE of the closed sum
`Output | Route | Grow` (the node-result sum type) — **or**, for a streaming
kind, a generator that yields `StreamChunk` and then *returns* a `NodeResult`.
The node never receives the pool: the engine's `runtime.eval_node` seam binds its
inputs (the read boundary) and hands it a record. Narrow effects enter through trait-gated
capabilities: MAP receives `bind_item`, a cooperatively blocking node may receive
`stop_event`, and a scoped AGENT may receive narrow memory context plus a write-only
`evaluation_trace`. Failure is **not** a variant — a failing node `raise`s; expected
evaluator rejection becomes a canonical `EvaluationFailure(Failure)` retained by the
accepted failure record. A streaming
kind is a generator that yields `StreamChunk` and *returns* its `NodeResult` (drained
by `_drain_node_generator`).

Invariant: a node never writes the pool. It *describes* its one output value as
`Output(value)`; the engine performs the write under `node_id`. Keeps nodes pure.
"""

import copy
from abc import ABC
from collections.abc import Generator
from dataclasses import dataclass, field, replace
from enum import Enum
from math import isfinite
from typing import Any, Callable, ClassVar, Iterator, Optional, Protocol, Union, runtime_checkable

from yian.diagnostics import Failure, OriginRef
from yian.expr import ExpressionError
from yian.expr.expressions import evaluate_when_record
from yian.expr.lineage import FeedbackValue, has_feedback, validate_feedback
from yian.memory.contracts import MemoryScope
from yian.nodes.binding import BindingError, ReservedBinder, _validate_parameters, apply_defaults
from yian.nodes.contracts import (
    Assertion,
    ExecutionScope,
    HostToolCall,
    OutputContract,
    Parameter,
    PartialRecordContract,
    ValueContract,
    _iter_origin_carrier,
    _map_origin,
    _rewrite_origin_carrier,
    _validate_optional_origin,
)
from yian.typesys.values import Type

class NodeKind(str, Enum):
    """Closed vocabulary. Dispatch is an explicit `match`, never a registry."""

    AGENT = "agent"
    CODE = "code"
    TOOL = "tool"
    CASE = "case"
    HUMAN_INPUT = "human_input"  # block through the host-owned ask_user tool
    WAIT = "wait"  # authorable: block until an absolute UTC date/datetime
    LOOP = "loop"  # authorable: iterate a callable to a fixpoint (`('a -> 'a) -> 'a -> 'a`)
    START = "start"  # internal-only: loader-synthesized input boundary (parameter binding)
    END = "end"      # internal-only: loader-synthesized return boundary (record + list modes)
    CALL = "call"    # authorable: consult another flow once (`kind: call`)
    MAP = "map"      # authorable: map a callable over a list (`kind: map` + `over:`)


# --- the node's return type is a closed sum ----------------------------------------
# A pure node returns ONE of these (or a generator that yields StreamChunk and returns one).


@dataclass(frozen=True)
class Output:
    """A produced value that the dispatcher records before the reducer binds it.

    `commit_as` is the node-chosen commit redirect: when set, the reducer writes `value`
    under `commit_as` instead of the node's own id and fires that target's out-edges.
    A `str` target id, or `None` (the common case) to commit under the node's own id.
    LOOP uses it when its final driver commits under the compiled origin; the baked
    `Node.commit_as` field covers child-flow terminal redirects.
    """

    value: Any = None
    commit_as: Optional[str] = None


@dataclass(frozen=True)
class Route:
    """A routing-only outcome (a terminal): carries only the chosen case handle — no value,
    no post-asserts. Returned by a router (CASE); the dispatcher records the handle and the
    reducer takes its out-edge while skip-flooding siblings."""

    handle: str


@dataclass(frozen=True)
class Grow:
    """A spawner describes one deterministic graph expansion for the runtime to record.

    Fields:
      - `seed`: the pure input to the compile-owned growth builder. Kind-shaped but opaque to the
        engine (a call record dict, map records list, or loop `(record, index)`).
        The same canonical value is persisted so durable replay rebuilds identical topology without
        re-running the node that produced it.
      - `prune`: ids to retire in the same step (∅ for call/map; a self-respawn loop's finished
        iteration ids). Defaults to ∅. The reducer retires a pruned id with its whole grown
        subtree once every node in it is settled; until then it stays live.
    """

    seed: Any
    prune: "frozenset[str]" = field(default_factory=frozenset)


# The closed sum a pure `run(inputs)` returns.
NodeResult = Union[Output, Route, Grow]


def _validate_text_feedback(messages: tuple[str, ...], name: str) -> None:
    """Require an immutable batch of independent textual messages."""
    if type(messages) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    if any(type(message) is not str for message in messages):
        raise TypeError(f"{name} messages must be strings")


@dataclass(frozen=True, slots=True, init=False)
class BackwardResult:
    """Independent feedback for an owner's inputs, behavior, and local variables.

    Input messages retain ordinary dictionaries and positional lists. Public
    container properties return detached copies so callers cannot alter the
    retained result.

    Attributes:
        input_feedback (`dict[str, tuple[FeedbackValue, ...]]`): Authored input
            names mapped to ordered batches of independent feedback values.
        self_feedback (`tuple[str, ...]`): Observations about this owner's behavior.
        variable_feedback (`dict[str, tuple[str, ...]]`): Textual messages for
            declared local variables; these do not update variable values.
    """

    _input_feedback: dict[str, tuple[FeedbackValue, ...]] = field(repr=False)
    self_feedback: tuple[str, ...]
    _variable_feedback: dict[str, tuple[str, ...]] = field(repr=False)

    def __init__(
        self,
        input_feedback: dict[str, tuple[FeedbackValue, ...]] | None = None,
        self_feedback: tuple[str, ...] = (),
        variable_feedback: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        """Validate channel values and retain detached feedback snapshots.

        Args:
            input_feedback (`dict[str, tuple[FeedbackValue, ...]]`, *optional*,
                defaults to `None`): Input message batches; `None` creates an
                empty channel. Lists represent a single positional message.
            self_feedback (`tuple[str, ...]`, *optional*, defaults to `()`):
                Independent messages about the owner's behavior.
            variable_feedback (`dict[str, tuple[str, ...]]`, *optional*,
                defaults to `None`): Local variable message batches; `None`
                creates an empty channel.

        Raises:
            `TypeError`: Raised for invalid channel, key, batch, or message types.
            `ValueError`: Raised for an invalid nested feedback selection.
        """
        inputs = {} if input_feedback is None else input_feedback
        variables = {} if variable_feedback is None else variable_feedback
        if type(inputs) is not dict:
            raise TypeError("input_feedback must be a dictionary")
        for name, messages in inputs.items():
            if type(name) is not str:
                raise TypeError("input_feedback keys must be strings")
            if type(messages) is not tuple:
                raise TypeError(f"input_feedback[{name!r}] must be a tuple")
            for message in messages:
                validate_feedback(message, message, path=(name,))
        _validate_text_feedback(self_feedback, "self_feedback")
        if type(variables) is not dict:
            raise TypeError("variable_feedback must be a dictionary")
        for name, messages in variables.items():
            if type(name) is not str:
                raise TypeError("variable_feedback keys must be strings")
            _validate_text_feedback(messages, f"variable_feedback[{name!r}]")
        object.__setattr__(self, "_input_feedback", copy.deepcopy(inputs))
        object.__setattr__(self, "self_feedback", self_feedback)
        object.__setattr__(self, "_variable_feedback", variables.copy())

    @property
    def input_feedback(self) -> dict[str, tuple[FeedbackValue, ...]]:
        """Return fresh ordinary values with independent message order preserved."""
        return copy.deepcopy(self._input_feedback)

    @property
    def variable_feedback(self) -> dict[str, tuple[str, ...]]:
        """Return local variable messages without exposing the retained mapping."""
        return self._variable_feedback.copy()


def _copy_backward_value(value: Any, name: str) -> Any:
    """Copy one retained JSON value without coercing keys or container types."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError(f"{name} must contain only finite JSON numbers")
        return value
    if type(value) is list:
        return [
            _copy_backward_value(item, f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError(f"{name} object keys must be strings")
        return {
            key: _copy_backward_value(item, f"{name}[{key!r}]")
            for key, item in value.items()
        }
    raise TypeError(f"{name} must contain only JSON values")


def _validate_declared_names(names: tuple[str, ...], field_name: str) -> None:
    """Require one immutable inventory of distinct authored names."""
    if type(names) is not tuple:
        raise TypeError(f"{field_name} must be a tuple")
    if any(type(name) is not str for name in names):
        raise TypeError(f"{field_name} must contain only strings")
    if len(set(names)) != len(names):
        raise ValueError(f"{field_name} must contain unique names")


def _validate_input_output_paths(data: dict[str, Any], input_names: tuple[str, ...]) -> None:
    """Validate optional source-neutral paths from declared inputs into the output."""
    if "input_output_paths" not in data:
        return
    paths = data["input_output_paths"]
    if type(paths) is not dict:
        raise TypeError("input_output_paths must be a dictionary")
    for name, path in paths.items():
        if type(name) is not str or name not in input_names:
            raise ValueError("input_output_paths must name declared inputs")
        if type(path) is not list:
            raise TypeError("input_output_paths values must be lists")
        if any(type(part) is not str and (type(part) is not int or part < 0) for part in path):
            raise ValueError("output path components must be strings or non-negative integers")


@dataclass(frozen=True, slots=True, init=False)
class BackwardContext:
    """Detached forward material and attribution supplied to one backward hook.

    Input declarations stay separate from recorded values: an early binding
    failure has ``inputs=None`` even when the owner declares input names.
    Mutable JSON values are copied on construction and on every public read.
    Failure traces retain canonical origin keys without front-end subclass state.
    """

    _inputs: dict[str, Any] | None = field(repr=False)
    input_names: tuple[str, ...]
    variable_names: tuple[str, ...]
    _output: Any = field(repr=False)
    failure: Failure | None
    scope: ExecutionScope | None
    run_id: str | None
    node_id: str | None
    node_started_seq: int | None
    _node_data: dict[str, Any] = field(repr=False)
    _downstream_context: tuple[dict[str, Any], ...] = field(repr=False)

    def __init__(
        self,
        inputs: dict[str, Any] | None,
        *,
        input_names: tuple[str, ...] = (),
        variable_names: tuple[str, ...] = (),
        output: Any = None,
        failure: Failure | None = None,
        scope: ExecutionScope | None = None,
        run_id: str | None = None,
        node_id: str | None = None,
        node_started_seq: int | None = None,
        node_data: dict[str, Any] | None = None,
        downstream_context: tuple[dict[str, Any], ...] = (),
    ) -> None:
        """Validate the available invocation metadata and snapshot its values.

        Args:
            inputs (`dict[str, Any] | None`): Canonical bound JSON input record;
                `None` means no usable input record was retained.
            input_names (`tuple[str, ...]`, *optional*, defaults to `()`):
                Authored input names this owner may address, including defaults.
            variable_names (`tuple[str, ...]`, *optional*, defaults to `()`):
                Declared local variables. An empty inventory rejects all targets.
            output (`Any`, *optional*, defaults to `None`): Recorded JSON output,
                when the forward invocation produced one.
            failure (`Failure | None`, *optional*, defaults to `None`):
                The recorded forward failure, independent of backward's outcome.
            scope (`ExecutionScope | None`, *optional*, defaults to `None`):
                Authored path and structured child position, when available.
            run_id (`str | None`, *optional*, defaults to `None`): Recorded run ID.
            node_id (`str | None`, *optional*, defaults to `None`): Display ID of
                the executing node; it is never parsed to infer scope.
            node_started_seq (`int | None`, *optional*, defaults to `None`):
                Positive accepted node-start sequence, paired with the run ID.
            node_data (`dict[str, Any]`, *optional*, defaults to `None`):
                Node-owned JSON material retained for backward; `None` uses `{}`.
            downstream_context (`tuple[dict[str, Any], ...]`, *optional*, defaults
                to `()`): Ordered JSON attribution accompanying feedback messages.

        Raises:
            `TypeError`: Raised for malformed metadata or non-JSON values.
            `ValueError`: Raised for duplicate declarations, invalid positions,
                or non-finite JSON numbers.
        """
        if inputs is not None and type(inputs) is not dict:
            raise TypeError("inputs must be a dictionary or None")
        _validate_declared_names(input_names, "input_names")
        _validate_declared_names(variable_names, "variable_names")
        if failure is not None:
            if type(failure) is not Failure:
                raise TypeError("failure must be a Failure or None")
            Failure.__post_init__(failure)
            failure = replace(
                failure, trace=tuple(OriginRef(origin.key) for origin in failure.trace)
            )
        if scope is not None:
            if type(scope) is not ExecutionScope:
                raise TypeError("scope must be an ExecutionScope or None")
            ExecutionScope.__post_init__(scope)
        if run_id is not None and type(run_id) is not str:
            raise TypeError("run_id must be a string or None")
        if node_id is not None and type(node_id) is not str:
            raise TypeError("node_id must be a string or None")
        if node_started_seq is not None:
            if type(node_started_seq) is not int:
                raise TypeError("node_started_seq must be an integer or None")
            if node_started_seq <= 0:
                raise ValueError("node_started_seq must be positive")
        data = {} if node_data is None else node_data
        if type(data) is not dict:
            raise TypeError("node_data must be a dictionary")
        _validate_input_output_paths(data, input_names)
        if type(downstream_context) is not tuple:
            raise TypeError("downstream_context must be a tuple")
        if any(type(item) is not dict for item in downstream_context):
            raise TypeError("downstream_context must contain dictionaries")

        object.__setattr__(self, "_inputs", _copy_backward_value(inputs, "inputs"))
        object.__setattr__(self, "input_names", input_names)
        object.__setattr__(self, "variable_names", variable_names)
        object.__setattr__(self, "_output", _copy_backward_value(output, "output"))
        object.__setattr__(self, "failure", failure)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "node_id", node_id)
        object.__setattr__(self, "node_started_seq", node_started_seq)
        object.__setattr__(self, "_node_data", _copy_backward_value(data, "node_data"))
        object.__setattr__(
            self,
            "_downstream_context",
            tuple(
                _copy_backward_value(item, f"downstream_context[{index}]")
                for index, item in enumerate(downstream_context)
            ),
        )

    @property
    def inputs(self) -> dict[str, Any] | None:
        """Return a detached bound input record, or no usable record."""
        return copy.deepcopy(self._inputs)

    @property
    def output(self) -> Any:
        """Return a detached copy of the recorded forward output."""
        return copy.deepcopy(self._output)

    @property
    def node_data(self) -> dict[str, Any]:
        """Return a detached copy of node-owned backward material."""
        return copy.deepcopy(self._node_data)

    @property
    def downstream_context(self) -> tuple[dict[str, Any], ...]:
        """Return detached attribution in the order accompanying feedback."""
        return copy.deepcopy(self._downstream_context)


def validate_backward_result(result: BackwardResult, context: BackwardContext) -> None:
    """Validate every feedback channel before routing any input message.

    Args:
        result (`BackwardResult`): The owner's proposed feedback channels.
        context (`BackwardContext`): Recorded bound values and the owner's
            declared input and local variable inventories.

    Raises:
        `TypeError`: Raised for an invalid result, context, or message type.
        `ValueError`: Raised for an undeclared target, unavailable input value,
            or nested feedback shape mismatch.
    """
    if type(result) is not BackwardResult:
        raise TypeError("result must be a BackwardResult")
    if type(context) is not BackwardContext:
        raise TypeError("context must be a BackwardContext")
    checked = BackwardResult(
        result.input_feedback, result.self_feedback, result.variable_feedback
    )
    inputs = context.inputs
    input_feedback = checked.input_feedback
    if input_feedback and inputs is None:
        raise ValueError("input_feedback requires usable recorded inputs")
    for name, messages in input_feedback.items():
        if name not in context.input_names:
            raise ValueError(f"input_feedback selects undeclared input {name!r}")
        if inputs is None or name not in inputs:
            raise ValueError(f"input_feedback has no recorded input {name!r}")
        for message in messages:
            validate_feedback(message, inputs[name], path=(name,))
    for name in checked.variable_feedback:
        if name not in context.variable_names:
            raise ValueError(f"variable_feedback selects undeclared variable {name!r}")


def _project_structural_feedback(
    context: BackwardContext, feedback: tuple[FeedbackValue, ...],
) -> BackwardResult:
    """Select input messages through captured output paths without rewriting text."""
    if type(feedback) is not tuple:
        raise TypeError("feedback must be a tuple of independent messages")
    inputs = context.inputs
    if inputs is None:
        return BackwardResult()
    data = context.node_data
    _validate_input_output_paths(data, context.input_names)
    if "input_output_paths" not in data:
        raise ValueError("structural backward requires captured input_output_paths")
    output = None if context.failure is not None else context.output
    for message in feedback:
        validate_feedback(message, output)
    projected: dict[str, tuple[FeedbackValue, ...]] = {}
    paths = data["input_output_paths"]
    for name in context.input_names:
        if name not in inputs or name not in paths:
            continue
        path = paths[name]
        if context.failure is None:
            value = output
            for part in path:
                if type(part) is str and type(value) is dict and part in value:
                    value = value[part]
                elif type(part) is int and type(value) is list and part < len(value):
                    value = value[part]
                else:
                    raise ValueError("captured output path does not exist in the accepted output")
        messages = []
        for message in feedback:
            selected = message
            for part in path:
                if selected is None or type(selected) is str:
                    break
                selected = (
                    selected.get(part) if type(selected) is dict
                    else selected[part]
                )
            if has_feedback(selected):
                messages.append(selected)
        if messages:
            projected[name] = tuple(messages)
    result = BackwardResult(input_feedback=projected)
    validate_backward_result(result, context)
    return result


class _HostStop(Exception):
    """Private control signal: the host requested execution to stop cooperatively.

    This is deliberately outside `NodeResult`: it unwinds node evaluation without becoming
    a node failure or a flow failure.
    """

    def __init__(self, message: str = "", *, primary: bool = True) -> None:
        super().__init__(message)
        # False means this exception merely observed an already-set event. A pooled drive
        # prefers an exact primary exception after every worker has joined, so an initiating
        # terminal EOF is not replaced by a sibling's derived cancellation wakeup.
        self.primary = primary


class NodeAssertFailed(Exception):
    """A node-local `asserts:` clause evaluated false (or errored) — raised by a node's own
    `run()` template around its kind body (including StartNode after binding).

    `assertion` is the exact wrapper-owned semantic carrier, or `None` when the
    exception was raised outside that wrapper.

    Only the base wrapper's structurally marked instance becomes ``assertion.failed``.
    The same exception class raised by node code remains ``node.execute``.
    """

    def __init__(
        self,
        message: str,
        *,
        assertion: Assertion | None = None,
    ) -> None:
        """Retain the assertion that classified this wrapper-owned failure."""
        super().__init__(message)
        self.assertion = assertion


class _NodeBodyError(Exception):
    """Private stage marker retaining one exception raised by node implementation."""

    def __init__(self, original: Exception) -> None:
        """Keep the exact body exception and its original traceback identity."""
        super().__init__(str(original))
        self.original = original


class _BindingStageError(Exception):
    """Private marker for a failure produced by an evaluator-owned binding stage."""

    def __init__(self, original: BindingError) -> None:
        """Keep the binder-owned error distinct from arbitrary node-body exceptions."""
        super().__init__(str(original))
        self.original = original


class _DeferredBindingFault(Exception):
    """Private carrier for an unexpected deferred-binder exception that must escape."""

    def __init__(self, original: Exception) -> None:
        """Retain the exact host/setup fault raised by the injected binder."""
        super().__init__(str(original))
        self.original = original


class _CapabilityFault(Exception):
    """Private carrier for an unexpected injected-capability host fault."""

    def __init__(self, original: Exception) -> None:
        """Retain the exact host fault so the evaluator can propagate its identity."""
        super().__init__(str(original))
        self.original = original


@runtime_checkable
class Steering(Protocol):
    """A host inbox read by a running steerable node, keyed by its node id.

    Calls come from worker threads, so an implementation must be thread-safe.
    An expanded LOOP or MAP node has its own id per occurrence.
    """

    def drain(self, node_id: str) -> list[str]:
        """Remove and return the steer messages queued for `node_id`, oldest first."""
        ...

    def interrupted(self, node_id: str) -> bool:
        """Whether the host asked `node_id` to stop its loop and return now."""
        ...


class _AssertionStageError(Exception):
    """Private marker for an assertion failure produced by the base run wrapper."""

    def __init__(self, original: NodeAssertFailed) -> None:
        """Keep the wrapper-owned assertion distinct from node-body exceptions."""
        super().__init__(str(original))
        self.original = original


class Node(ABC):
    """
    The base contract every node kind implements: a pure function of its input record.

    A node implements its `forward` body; the inherited
    [`run`][yian.nodes.base.Node.run] template returns one of the closed sum
    `Output | Route | Grow` (or, for a streaming kind, a generator that yields `StreamChunk`
    and returns a `NodeResult`). The node never receives the pool — the engine's
    `eval_node` seam binds its inputs and hands it a record — and never writes the pool; it
    *describes* its one output as `Output(value)` and the engine performs the write under the
    node id. Failure is a `raise`, not a variant.

    Attributes:
        kind (`NodeKind`):
            The closed-vocabulary tag used by compose/compile and node-owned diagnostics.
            Runtime never dispatches on it. Set per subclass.
        id (`str`):
            The node's unique id within its (possibly namespaced) flow.
        title (`str`, *optional*):
            A human-friendly display title, or `None`.
        output_contract (`OutputContract`, *optional*):
            The canonical declared output contract: one exact value or a closed record
            with field-local type evidence. `None` leaves the write unenforced.
        params (`list[Parameter]`, *optional*):
            The node-side declared params (no source — the flow owns the wiring); `None`
            for a node that declares no inputs.
        pre_asserts (`list[Assertion]`):
            Ordered node-local contracts checked against the bound record before `forward`.
        post_asserts (`list[Assertion]`):
            Ordered node-local contracts (reading `${output}`) checked after `forward`.
        commit_as (`str`, *optional*):
            Engine-baked commit redirect: write this node's Output under `commit_as`
            instead of its own id (and fire that target's out-edges). Set by the
            compile-owned growth builders on a child-flow terminal (a CALL/MAP child END
            filler or a MAP END-list filler) to point back at the
            spawner id; `None` for an ordinary node (commit under its own id).
        origin_id (`str`, *optional*):
            "Attribute my grows and self-commit to this origin node." `None` on every
            ordinary node. Set only on a self-respawning loop driver: the compiled loop
            `L` sets `origin_id = L` (== self) and each fresh per-iteration clone `L~k`
            sets `origin_id = L` too, so the engine can attribute every iteration's
            `Grow`/`Output` to the single durable origin `L`. A driver is the origin iff
            `self.id == self.origin_id`, which is what gates "do not self-prune".
        origin (`OriginRef`, *optional*):
            Opaque authored-source identity for this semantic node, or `None` for a
            source-free or generated node. This diagnostic value is unrelated to
            `origin_id`, which controls LOOP execution attribution.
        origin_context (`tuple[OriginRef, ...]`):
            Generated outer-to-inner authored ancestry for failures inside this
            node. Directly constructed nodes always start with an empty tuple;
            compile-owned expansion may derive a non-empty context from enclosing
            spawners. This diagnostic value is not a local authored-origin carrier.
        memory_scope (`MemoryScope | None`):
            Compile-derived static memory context. Authors cannot supply it;
            memory layout resolution stamps it before execution when memory is
            enabled, and ordinary nodes otherwise retain ``None``.
        execution_scope (`ExecutionScope | None`):
            Compile-derived authored location, independent of memory declarations.
            Static layout resolution stamps root-relative paths on private node
            templates; direct construction starts with ``None``.
        env (`dict[str, Any]`):
            Author-supplied static per-node config (`{str -> literal}`), baked at build
            time from the flow-level `env:` default merged with the node's own `env:`
            (node override wins). Opaque to the engine core — kind-blind: only a node's
            own `forward()` path reads the keys it understands (AGENT:
            `max_tool_iterations`).
            Unknown keys are carried but ignored. `{}` when none is authored.
        types (`dict[str, Type]`):
            The type table `isinstance(x, T)` looks `T` up in when this node evaluates its
            bindings, asserts, and predicates, keyed by type name. Filled at load from the
            names the node's expressions use; `{}` when none are used.
    """

    kind: ClassVar[NodeKind]
    # Declares "I may grow the graph" (a spawner returns a `Grow`). Overridden True by
    # the spawner kinds (CALL/MAP/LOOP); the `eval_node` seam gates the grow path on it.
    is_spawner: ClassVar[bool] = False
    # Declares "I bind my inputs PER ELEMENT via a `bind_item` cap rather than once up front"
    # (MAP). When True, the read seam (`eval_node`) starts the record empty and supplies
    # `caps['bind_item']`; when False (the common case) it binds `params` once from the pool.
    # Overridden True by MapNode; every other kind binds up front.
    binds_per_item: ClassVar[bool] = False
    # A per-item binder may project its local scope from this reserved input key.
    # None means the node has no recorded source collection for the local scope.
    item_source_key: ClassVar[str | None] = None
    # Nested-call depth policy for the engine's growth core. The depth stamped on a grow's subgraph
    # spawners + its terminal is `parent_depth + grow_depth_delta`; a positive delta is also
    # bounded by MAX_REF_DEPTH. Possible values:
    #   None  — a non-spawner; the core does no depth work.
    #   0     — LOOP inherits its parent's call depth without consuming another level.
    #   1     — a nested call, +1 and bounded (CALL/MAP: each child is one deeper level).
    # Overridden by LOOP (0) and CALL/MAP (1); the default None fits non-spawners.
    grow_depth_delta: ClassVar[Optional[int]] = None
    # Declares "I am the self-respawning fixpoint-iteration driver" (LOOP). The growth reducer
    # gates the origin-keyed single-live-record ledger invariant under this trait:
    # each iteration is a fresh driver clone whose grow is attributed to `origin_id` (the compiled
    # loop) and supersedes the prior live iteration. A plain trait check suffices — the core
    # never dispatches on the kind.
    # Overridden True by LoopNode; False for every other kind.
    is_loop: ClassVar[bool] = False
    # Declares "I can cooperate with a host-owned execution stop". When True, the read seam
    # (`eval_node`) supplies the exact `caps['stop_event']` object owned by the run. Nodes that
    # do not opt in receive no stop capability, keeping the effect boundary explicit.
    needs_stop_event: ClassVar[bool] = False
    # Declares access to notes and an optional structured transcript through one
    # lazy ``open_memory_context(system, prompt, compatibility)`` capability.
    # AgentNode is the sole built-in opt-in; the runtime remains kind-blind.
    needs_memory_context: ClassVar[bool] = False
    # Declares "I can use host-supplied skills". The evaluator supplies the run's
    # ``skills`` mapping (name -> Skill) only when the run boundary injected one and this
    # trait is True. AgentNode is the sole built-in opt-in; the runtime remains kind-blind.
    needs_skills: ClassVar[bool] = False
    # Declares "I can emit visible prompt/model/tool evidence". The evaluator supplies a
    # write-only invocation-local trace only when evidence capture is enabled and this exact
    # trait is True. AgentNode is the sole built-in opt-in; the runtime remains kind-blind.
    needs_evaluation_trace: ClassVar[bool] = False
    # Declares "I can publish actual render locals for retained backward context".
    # The evaluator supplies a single-write capture callable only during retention.
    needs_render_capture: ClassVar[bool] = False
    # Declares "I look up or invoke tools". The evaluator supplies the run's exact
    # ``caps['tools']`` ToolSet, so per-run tools and the per-run ``ask_user`` handler
    # never leak between runs. TOOL, HUMAN_INPUT, and AGENT opt in.
    needs_tools: ClassVar[bool] = False
    # Declares "I can publish transient in-loop observers". The evaluator supplies
    # ``caps['emit_stream']`` / ``caps['emit_event']`` only for the host sinks that exist.
    # AgentNode is the sole built-in opt-in; the runtime remains kind-blind.
    needs_live_events: ClassVar[bool] = False
    # Declares "I read the host's steering inbox". The evaluator supplies
    # ``caps['steering']`` only when the run boundary injected one and this is True.
    # AgentNode reports its authored ``steerable:`` flag; the runtime remains kind-blind.
    needs_steering: bool = False

    def __init__(
        self,
        node_id: str,
        *,
        title: Optional[str] = None,
        output_contract: OutputContract | None = None,
        origin: OriginRef | None = None,
    ) -> None:
        self.id = node_id
        self.title = title
        if output_contract is not None and type(output_contract) not in (
            ValueContract,
            PartialRecordContract,
        ):
            raise TypeError(
                "output_contract must be an exact ValueContract, "
                "PartialRecordContract, or None, "
                f"got {type(output_contract).__name__}"
            )
        # One canonical output declaration shared by compiled and Python-authored nodes.
        self.output_contract: OutputContract | None = output_contract
        # The node-side signature (the node/flow split): declared params with NO source — the flow owns
        # the wiring in `CompiledFlow.wiring[node_id][param]`. Stamped by the compiler
        # (`build_*`/case desugar); the engine's `eval_node` binds via `params` + `flow.wiring`.
        # `None` for a fake / directly-constructed node that declares no inputs (== no params).
        self.params: Optional[list[Parameter]] = None
        # Node-local `asserts:` (a per-node contract), classified + stamped by the loader and
        # enforced by the inherited `run` template: PRE checked against the bound input record
        # before `forward`; POST (reads `${output}`) afterward against `{**inputs, output}`.
        # Empty for most nodes.
        self.pre_asserts: list[Assertion] = []
        self.post_asserts: list[Assertion] = []
        # Engine-baked commit redirect (see the class docstring): a child-flow terminal
        # commits its Output under this spawner id instead of its own id. Stamped by the
        # compile-owned growth builders; `None` for an ordinary node.
        self.commit_as: Optional[str] = None
        # "Attribute my grows/self-commit to this origin" (see the class docstring). `None`
        # on ordinary nodes; set to the compiled loop id on a loop driver (origin == self on
        # the compiled `L`, == `L` on each fresh iteration clone `L~k`). A driver is the
        # origin iff `self.id == self.origin_id` (which gates "do not self-prune").
        self.origin_id: Optional[str] = None
        if origin is not None and not isinstance(origin, OriginRef):
            raise TypeError("node origin must be an OriginRef or None")
        # Opaque source identity for diagnostics. Unlike `origin_id`, this has no
        # execution meaning and is absent on directly constructed/generated nodes.
        self.origin: OriginRef | None = origin
        # Generated outer-to-inner failure ancestry. Direct construction never accepts
        # or invents ancestry; compile-owned expansion derives it after authored origins
        # have been rewritten. It therefore stays outside local origin traversal.
        self.origin_context: tuple[OriginRef, ...] = ()
        # Derived beside generated ancestry because neither value is authored node
        # configuration. The compile layer owns all non-None assignments.
        self.memory_scope: MemoryScope | None = None
        self.execution_scope: ExecutionScope | None = None
        # Author-supplied static per-node config, opaque to the engine core (kind-blind): a
        # plain `{str -> literal}` map baked at build time from the flow-level `env:` default
        # merged with this node's own `env:` (node override wins; see the compose builder). The
        # engine NEVER reads it — only a node's own `forward()` path interprets understood keys
        # (e.g. AGENT reads `max_tool_iterations`); unknown keys are carried but ignored. `{}`
        # for a node that declares no `env:` and inherits no flow default.
        self.env: dict[str, Any] = {}
        # The type table for `isinstance(x, T)` (see the class docstring). Runtime reads it
        # generically to evaluate this node's bindings; the node passes it to its own
        # asserts and predicates.
        self.types: dict[str, Type] = {}
        # `self.wiring_labels`: wiring param -> author-facing phrase for a reference error, for a
        #   SYNTHESIZED param the author never wrote (a CASE `__rN` from a `when:` ref, `__on` from
        #   `on:`) — e.g. `"cases[1].when"` / `"on:"`. Render-only and absent from run records;
        #   populated by the desugar that mints the param. Empty for every authored param (the
        #   default label below reproduces the plain `input '<param>'` text).
        self.wiring_labels: dict[str, str] = {}
    def failure_trace(
        self,
        operation_origin: OriginRef | None = None,
    ) -> tuple[OriginRef, ...]:
        """Return generated ancestry plus the best local failure origin.

        ``operation_origin`` wins when present; otherwise the authored node
        ``origin`` is used. When neither exists, only the generated ancestry is
        returned. Order and duplicate origins are preserved.

        Args:
            operation_origin (`OriginRef | None`, *optional*, defaults to `None`):
                More precise authored identity for the failing operation.

        Raises:
            TypeError:
                Raised when ancestry is not an exact tuple or a selected value is
                not an `OriginRef`.
            ValueError:
                Raised when an `OriginRef` subclass bypassed the canonical key
                invariant.
        """
        if type(self.origin_context) is not tuple:
            raise TypeError("node origin_context must be a tuple")
        for origin in self.origin_context:
            if not isinstance(origin, OriginRef):
                raise TypeError("node origin_context item must be an OriginRef")
            OriginRef(origin.key)

        selected = operation_origin if operation_origin is not None else self.origin
        _validate_optional_origin(selected, "failure operation origin")
        if selected is None:
            return self.origin_context
        OriginRef(selected.key)
        return self.origin_context + (selected,)

    def iter_local_origins(
        self,
        *,
        _seen_carriers: Optional[set[int]] = None,
    ) -> Iterator[OriginRef]:
        """Yield this node's source-neutral origins without following child flows.

        Args:
            _seen_carriers (`set[int]`, *optional*, defaults to `None`):
                Shared identities already visited by a graph traversal. `None`
                starts an independent local traversal.

        Yields:
            `OriginRef`:
                Present node, output-contract, parameter, and assertion origins
                in deterministic structural order. Shared frozen carriers yield
                once per traversal.
        """
        seen = _seen_carriers if _seen_carriers is not None else set()
        _validate_optional_origin(self.origin, "node origin")
        if self.origin is not None:
            yield self.origin
        if self.output_contract is not None:
            yield from _iter_origin_carrier(self.output_contract, seen)
        if self.params is not None:
            for parameter in self.params:
                yield from _iter_origin_carrier(parameter, seen)
        for assertion in self.pre_asserts:
            yield from _iter_origin_carrier(assertion, seen)
        for assertion in self.post_asserts:
            yield from _iter_origin_carrier(assertion, seen)

    def rewrite_local_origins(
        self,
        mapper: Callable[[OriginRef], OriginRef],
        *,
        _seen_carriers: Optional[set[int]] = None,
    ) -> None:
        """Rewrite this node's local origins while retaining carrier aliases.

        This mutating hook is for a compile-owned private graph copy. Public graph
        rewriting copies the complete root before invoking it, so an authored graph
        is never changed in place.

        Args:
            mapper (`Callable[[OriginRef], OriginRef]`):
                Pure transform for each present origin. It must return an
                `OriginRef`; exceptions propagate.
            _seen_carriers (`set[int]`, *optional*, defaults to `None`):
                Graph-wide carrier identities already rewritten. `None` starts
                an independent local rewrite.

        Raises:
            TypeError:
                Raised when `mapper` returns a value other than `OriginRef`.
        """
        seen = _seen_carriers if _seen_carriers is not None else set()
        _validate_optional_origin(self.origin, "node origin")
        self.origin = _map_origin(self.origin, mapper)
        if self.output_contract is not None:
            _rewrite_origin_carrier(self.output_contract, mapper, seen)
        if self.params is not None:
            for parameter in self.params:
                _rewrite_origin_carrier(parameter, mapper, seen)
        for assertion in self.pre_asserts:
            _rewrite_origin_carrier(assertion, mapper, seen)
        for assertion in self.post_asserts:
            _rewrite_origin_carrier(assertion, mapper, seen)

    def origin_children(self) -> tuple[Any, ...]:
        """Return static child flows whose origins belong to this node.

        The base node has no child. Higher-order node packages override this
        without importing the compile layer that owns traversal.
        """
        return ()

    def exception_origin(self, exception: Exception) -> OriginRef | None:
        """Return the authored operation origin for one node-body exception.

        Kinds with finer source-neutral exception mapping may override this hook.
        The default is the authored node context; source-free nodes return ``None``.
        """
        return self.origin

    def skipped_output_target(self) -> str:
        """Return the node whose successors observe this node's absent output.

        Successful output commits under the baked ``commit_as`` redirect when
        present and otherwise under this node's own ID.  Route skip propagation
        follows the same logical boundary so generated completion nodes do not
        strand their parent's outgoing edges.  Kinds whose output target is
        selected by other node-owned state override this hook.
        """
        return self.id if self.commit_as is None else self.commit_as

    def validate_execution_setup(self) -> None:
        """Reject malformed node-owned execution setup before entering the body stage.

        The base implementation requires a concrete ``forward``. Composite kinds
        extend this hook for invariants such as requiring their compiled child flow.
        """
        if type(self).forward is Node.forward:
            raise NotImplementedError(
                f"node {self.id!r} ({type(self).__name__}) defines no forward"
            )

    def run(self, inputs: dict[str, Any], **caps: Any) -> "NodeResult":
        """Run the node's complete pure-execution template.

        Declared defaults are filled first, node-owned setup is validated, pre-asserts
        run against the bound record, ``forward`` executes, and post-asserts inspect
        an ``Output`` against the pristine inputs. The method returns exactly
        ``Output | Route | Grow`` or a streaming generator with one of those terminal
        values. Narrow effects are passed only through node-owned trait gates.

        Direct calls preserve the original exception and traceback. The evaluator uses
        the private companion path to tag wrapper-owned binding, assertion, and body
        stages before converting expected rejection into ``EvaluationFailure``.
        """
        return self._run_template(inputs, classify_body_errors=False, **caps)

    def _run_for_evaluator(
        self,
        inputs: dict[str, Any],
        *,
        _capture_inputs: Callable[[dict[str, Any]], None] | None = None,
        **caps: Any,
    ) -> "NodeResult":
        """Run the base template with private failure markers and input capture.

        The optional callback receives the completed input record after defaults
        and before assertions or the body; it is consumed here, never forwarded
        as a node capability.
        """
        return self._run_template(
            inputs, classify_body_errors=True, _capture_inputs=_capture_inputs, **caps,
        )

    def _run_template(
        self,
        inputs: dict[str, Any],
        *,
        classify_body_errors: bool,
        _capture_inputs: Callable[[dict[str, Any]], None] | None = None,
        **caps: Any,
    ) -> "NodeResult":
        """Template method: fill declared defaults for omitted inputs, fire pre-asserts
        (record-only), run the kind's `forward`, then fire post-asserts (record-only)
        against `{**inputs, output}`.

        Default-fill runs FIRST — before the pre-asserts — so a pre-assert may read an
        input the caller OMITTED but the node declares a default for. `apply_defaults`
        fills only omitted keys (a caller's explicit bound-null SHADOWS the default,
        `f(x=None)`), so it never overrides a present key; running it here is a no-op for
        body nodes (their Parameters carry no default). START uses this same wrapper,
        so the generic parameter seam is its sole default authority.

        Firing reads ONLY the bound `inputs` record — never the pool — so node purity is
        preserved. `eval_node` drives this wrapper (not `forward`), so firing happens
        EXACTLY ONCE, owned by the node. Record-only is sufficient because every assert
        reaching a node is node-local after the ref lowering (build-time flow-asserts + the
        grow-path spawner-asserts): a cross-node/namespaced ref is lowered to a synth param
        that `bind_params` binds INTO the record, so there is no pool ref left to fall back to.

        Returns a `NodeResult` (`Output | Route | Grow`), or — for a streaming kind —
        a generator that yields `StreamChunk` and *returns* a `NodeResult`. Narrow effect caps
        are passed only to nodes whose traits request them (``bind_item``,
        ``stop_event``, ``open_memory_context``, ``evaluation_trace``, ``tools``,
        ``skills``, ``emit_stream``, and ``emit_event``).
        Expected rejection is a `raise`, not a result variant; the evaluator classifies
        its structural stage before the dispatcher records the canonical `Failure`."""
        params = self.params if self.params is not None else []
        if type(params) is not list:
            raise TypeError(
                f"node {self.id!r} params must be an exact list, got {type(params).__name__}"
            )
        for parameter in params:
            if type(parameter) is not Parameter:
                raise TypeError(
                    f"node {self.id!r} params must contain Parameter values, "
                    f"got {type(parameter).__name__}"
                )
        try:
            inputs = apply_defaults(params, inputs)
        except BindingError as exc:
            if not classify_body_errors:
                raise
            raise _BindingStageError(exc) from exc
        if _capture_inputs is not None:
            _capture_inputs(inputs)
        self.validate_execution_setup()
        # Fire pre-asserts IN ORDER, first-failure-wins. The assertion carrier supplies the
        # canonical operation origin used by the evaluator's failure trace.
        bad = next(
            (
                assertion
                for assertion in self.pre_asserts
                if not self._assert_holds(assertion.expression, inputs)
            ),
            None,
        )
        if bad is not None:
            failure = NodeAssertFailed(
                f"node {self.id!r} pre-assert failed: {bad.display}",
                assertion=bad,
            )
            if classify_body_errors:
                try:
                    raise failure
                except NodeAssertFailed as original:
                    raise _AssertionStageError(original) from original
            raise failure
        # Pristine snapshot for the post-asserts, taken BEFORE `forward` runs: a leaf may
        # mutate the dict it receives (a CODE body transforming its input in place), which must
        # not corrupt the post-assert scope — the contract check has to see the inputs as bound,
        # not as the body left them. Only paid when this node declares post-asserts.
        post_input = copy.deepcopy(inputs) if self.post_asserts else None
        try:
            result = self.forward(inputs, **caps)
        except (
            _HostStop,
            _BindingStageError,
            _DeferredBindingFault,
            _CapabilityFault,
            _NodeBodyError,
        ):
            raise
        except Exception as exc:
            if not classify_body_errors:
                raise
            raise _NodeBodyError(exc) from exc
        # Post-asserts fire ONLY for an `Output` result — never Route/Grow. The
        # growth-specific expansion path owns spawner post-assert handling. A generator result
        # also falls through the guard untouched — the DORMANT streaming path: the wrapper must
        # not fire post on it; it is returned unchanged for `eval_node` to drain. No node kind
        # returns a generator today.
        if isinstance(result, Output) and self.post_asserts:
            post_record = {**post_input, "output": result.value}
            bad = next(
                (
                    assertion
                    for assertion in self.post_asserts
                    if not self._assert_holds(assertion.expression, post_record)
                ),
                None,
            )
            if bad is not None:
                failure = NodeAssertFailed(
                    f"node {self.id!r} post-assert failed: {bad.display}",
                    assertion=bad,
                )
                if classify_body_errors:
                    try:
                        raise failure
                    except NodeAssertFailed as original:
                        raise _AssertionStageError(original) from original
                raise failure
        return result

    def forward(self, inputs: dict[str, Any], **caps: Any) -> "NodeResult":
        """Execute the node as a pure function of its bound input record (the per-kind body
        the `run` template method wraps).

        Every concrete kind defines this method; `eval_node` drives the `run` wrapper, which fires
        asserts around this body exactly once. Not `@abstractmethod` only because the base `Node`
        is instantiated directly in a few tests/fakes; a kind that omits it raises this loud error.

        Returns a `NodeResult` (`Output | Route | Grow`), or — for a streaming kind —
        a generator that yields `StreamChunk` and *returns* a `NodeResult`. Narrow effect caps
        are passed only to nodes whose traits request them (``bind_item``,
        ``stop_event``, ``open_memory_context``, ``evaluation_trace``, ``tools``,
        and ``skills``).
        A body exception is classified as ``node.execute`` by the evaluator."""
        raise NotImplementedError(
            f"node {self.id!r} ({type(self).__name__}) defines no forward"
        )

    def declared_variables(self) -> tuple[Parameter, ...]:
        """Return this node's canonical local declarations as an immutable inventory."""
        return ()

    def granted_tools(self) -> tuple[str, ...]:
        """Return the tool and tool-group ids a model inside this node may call.

        The run boundary checks these against the run's tools before any record is
        written. The base node grants none.
        """
        return ()

    def host_tool_call(self) -> HostToolCall | None:
        """Return the tool call this node makes directly, without a model.

        The run boundary checks it against the run's tools before any record is
        written. The base node makes none.
        """
        return None

    def backward_input_names(self) -> tuple[str, ...]:
        """Return declared inputs eligible for feedback in this node's contract.

        Returns:
            `tuple[str, ...]`: Canonical parameter names in declaration order.
                Kinds with control-only parameters override this inventory.

        Raises:
            `TypeError`: If params is not a canonical list of `Parameter` values.
        """
        parameters = [] if self.params is None else self.params
        _validate_parameters(parameters)
        return tuple(parameter.name for parameter in parameters)

    def capture_backward_context(self) -> dict[str, Any]:
        """Return static backward material before this invocation runs.

        This pure hook reads authored layout only. It does not resolve defaults,
        render prompts, or execute the body. Common input and variable names are
        captured separately. With retention enabled, the base run template calls
        this hook after defaults and before assertions or body execution. A custom
        run override retains the evaluator's completed-binding, pre-run boundary.

        Returns:
            `dict[str, Any]`: Finite JSON material, empty by default. The reserved
                `input_output_paths` mapping addresses declared inputs by paths
                of string fields or non-negative list indices into the output.
        """
        return {}

    def backward(
        self, context: BackwardContext, feedback: tuple[FeedbackValue, ...],
    ) -> BackwardResult:
        """Stop feedback at the default behavioral boundary with empty channels.

        Args:
            context (`BackwardContext`): Detached recorded inputs, optional
                accepted output or failure, declaration names, and node-owned
                static material for this invocation. Read this snapshot rather
                than mutable node fields when interpreting feedback.
            feedback (`tuple[FeedbackValue, ...]`): Ordered independent messages.
                Each is ordinary text, a partial record, an exact-length nested
                item list, or `None`; a list is one positional message.

        Returns:
            `BackwardResult`: Input, self, and variable feedback channels. This
                default returns all three empty and performs no model call.

        Examples:
            `node.backward(BackwardContext(None), ("Review.",))` returns an empty
            result for a node that inherits this behavioral stopping rule.
        """
        return BackwardResult()

    def on_failure(self, exc: Exception, inputs: dict[str, Any], **caps: Any) -> "NodeResult":
        """Reserved error-strategy seam, not currently driven by ``eval_node``. Default: re-raise."""
        raise exc

    def bind_reserved(
        self, node_wiring: dict, pool, *, bind: ReservedBinder | None = None,
    ) -> dict[str, Any]:
        """Reserved input keys the read seam (`eval_node`) must pre-resolve from
        `node_wiring` + `pool` before `run`, merged into the bound record.

        Possible keys (per kind): WAIT returns `{"until": <ISO deadline>}`; a MAP returns
        `{"over": <list>}`. Default: `{}` (an ordinary node reserves no keys). `node_wiring`
        is the flow-owned wiring for this node (`flow.wiring[id]`); `pool` is the live
        `VariablePool` the seam reads sources from. The optional ``bind`` callable
        resolves a named source and applies the node-owned validation/normalization
        before retaining its delivered value; its arguments are name, source, and
        normalization callable. Omitting it preserves direct read-hook calls."""
        return {}

    def reserved_wiring_keys(self) -> set[str]:
        """The reserved AUTHOR-wiring key names this node owns beyond its declared `params`
        — the keys `check_wiring_parity` must allow, and validation treats specially.

        These are the static NAMES of the same keys `bind_reserved` resolves at run
        (WAIT: `until`; MAP: `over`), but returned with NO pool — load-time only. Default:
        the empty set (an ordinary node reserves no keys)."""
        return set()

    def wiring_ref_label(self, param: str) -> str:
        """The author-facing phrase for a wiring `param` in a reference error — e.g. `input 'x'`
        for an authored input, or a synthetic param's own site (`cases[1].when`, `on:`) when the
        desugar recorded one in `wiring_labels`. The default reproduces the plain `input '<param>'`
        text, so a message about an authored param is byte-identical to before."""
        return self.wiring_labels.get(param, f"input {param!r}")

    def _assert_holds(self, expr: str, record: dict) -> bool:
        """Evaluate a node assert against `record` with this node's type table; a raising
        assert (ordered/arith over a non-scalar / None `${output}`) is treated as not holding,
        which the wrapper marks as ``assertion.failed`` at the evaluator boundary."""
        try:
            return bool(evaluate_when_record(expr, record, self.types))
        except ExpressionError:
            return False

    def _drain_node_generator(self, gen: Generator) -> Generator[Any, Any, Any]:
        """Forward transient chunks and return the generator's raw terminal value.

        Generator execution belongs to the node-body stage. The evaluator validates
        the returned value afterward so body exceptions and result-contract rejection
        cannot collapse into one category.
        """
        try:
            while True:
                try:
                    event = next(gen)
                except StopIteration as stop:
                    return stop.value
                except (
                    _HostStop,
                    _BindingStageError,
                    _DeferredBindingFault,
                    _CapabilityFault,
                    _NodeBodyError,
                ):
                    raise
                except Exception as exc:
                    raise _NodeBodyError(exc) from exc
                yield event
        finally:
            gen.close()
