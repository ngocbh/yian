"""AGENT node — one authored node running one local model/tool loop.

The selected mode owns the conversation for the duration of ``run``. Tools,
including the host-installed ``ask_user``, are ordinary explicitly authored
tool grants. A crash or host interruption unwinds the node; no conversation or
continuation state is stored in the graph or variable pool.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Iterator, Optional

from agent_composer.diagnostics import OriginRef
from agent_composer.expr.template import render_template_record
from agent_composer.memory.contracts import (
    EvaluationTrace,
    MemoryPolicySpec,
    MemoryRecallError,
    MemorySection,
    _iter_memory_policy_origin,
    _rewrite_memory_policy_origin,
)
from agent_composer.nodes.agent.modes import DEFAULT_SYSTEM, MODES, AgentRunContext
from agent_composer.nodes.agent.modes.tool_calling import MAX_TOOL_ITERATIONS
from agent_composer.nodes.agent.structured import _structured_validation_path
from agent_composer.nodes.base import Node, NodeKind, Output, _CapabilityFault, _HostStop
from agent_composer.nodes.binding import apply_defaults
from agent_composer.nodes.contracts import (
    Parameter,
    ValueContract,
    _iter_origin_carrier,
    _rewrite_origin_carrier,
)
from agent_composer.typesys.types import canonical_type_label
from agent_composer.typesys.values import build_value_as

DEFAULT_MODE = "plain"


class AgentNode(Node):
    """An LLM-backed leaf configured by a prompt, mode, and authored tools.

    ``llm_config`` is normalized by ``model_from_config`` when the node runs.
    The package attribute is resolved at call time so tests and hosts may
    monkeypatch the model factory without an engine-owned capability. An optional
    ``memory_policy`` is authored identity only until compile derives runtime scope.
    ``variables`` retains typed local declarations with mandatory literal defaults,
    resolved into a fresh prompt record for each invocation.
    """

    kind = NodeKind.AGENT
    needs_stop_event = True
    needs_evaluation_trace = True
    needs_memory_recall = True
    needs_render_capture = True

    def __init__(
        self,
        node_id: str,
        *,
        prompt: str = "",
        tools: Optional[list[str]] = None,
        llm_config: Optional[dict[str, Any]] = None,
        llm_inherit: bool = True,
        mode: str = DEFAULT_MODE,
        retries: int = 2,
        title: Optional[str] = None,
        memory_policy: MemoryPolicySpec | None = None,
        variables: list[Parameter] | None = None,
    ) -> None:
        super().__init__(node_id, title=title)
        self.prompt = prompt or ""
        self.tools = tools or []
        # own_llm_config is the authored source re-resolved by each cascade pass;
        # llm_config is the effective config baked for this run.
        self.own_llm_config: dict[str, Any] = dict(llm_config) if llm_config else {}
        self.llm_config = llm_config
        self.llm_inherit = llm_inherit
        self.mode = mode or DEFAULT_MODE
        self.retries = retries
        if memory_policy is not None and type(memory_policy) is not MemoryPolicySpec:
            raise TypeError(
                "memory_policy must be an exact MemoryPolicySpec or None"
            )
        self.memory_policy = memory_policy
        self.variables = [] if variables is None else variables
        if self.mode not in MODES:
            raise ValueError(
                f"AGENT node {node_id!r}: unknown mode {self.mode!r}; known: {sorted(MODES)}"
            )
        self._validate_variables()

    def _validate_variables(self) -> None:
        """Require canonical, uniquely named declarations with valid typed defaults."""
        if type(self.variables) is not list:
            raise TypeError("agent variables must be a list of Parameter values")
        names: set[str] = set()
        for variable in self.variables:
            if type(variable) is not Parameter:
                raise TypeError("agent variables must contain exact Parameter values")
            if type(variable.name) is not str or not variable.name:
                raise TypeError("variable names must be non-empty strings")
            if variable.name in names:
                raise ValueError(f"duplicate variable name {variable.name!r}")
            names.add(variable.name)
            if type(variable.contract) is not ValueContract:
                raise TypeError(f"variable {variable.name!r} requires an exact ValueContract")
            if variable.has_default is not True or variable.required is not False:
                raise ValueError(f"variable {variable.name!r} requires a default and required=False")
            variable.contract.type.validate_shape()
            build_value_as(variable.contract.type, variable.default)

    def iter_local_origins(
        self,
        *,
        _seen_carriers: Optional[set[int]] = None,
    ) -> Iterator[OriginRef]:
        """Yield common origins, variable declarations, and the optional policy site."""
        seen = _seen_carriers if _seen_carriers is not None else set()
        yield from super().iter_local_origins(_seen_carriers=seen)
        for variable in self.variables:
            yield from _iter_origin_carrier(variable, seen)
        if self.memory_policy is not None:
            yield from _iter_memory_policy_origin(self.memory_policy, seen)

    def rewrite_local_origins(
        self,
        mapper: Callable[[OriginRef], OriginRef],
        *,
        _seen_carriers: Optional[set[int]] = None,
    ) -> None:
        """Rewrite common, variable, and policy origins while retaining aliases."""
        seen = _seen_carriers if _seen_carriers is not None else set()
        super().rewrite_local_origins(mapper, _seen_carriers=seen)
        for variable in self.variables:
            _rewrite_origin_carrier(variable, mapper, seen)
        if self.memory_policy is not None:
            _rewrite_memory_policy_origin(self.memory_policy, mapper, seen)

    def _max_tool_iterations(self) -> int:
        """Return the positive authored loop bound or ``-1`` for no bound."""
        raw = self.env.get("max_tool_iterations", MAX_TOOL_ITERATIONS)
        if isinstance(raw, bool) or not isinstance(raw, int) or (raw < 1 and raw != -1):
            raise ValueError(
                f"AGENT node {self.id!r}: env max_tool_iterations must be a positive int "
                f"or -1 (no cap), got {raw!r}"
            )
        return raw

    def _ctx(
        self,
        prompt: str,
        *,
        stop_event=None,
        evaluation_trace: EvaluationTrace | None = None,
        memory_sections: tuple[MemorySection, ...] = (),
    ) -> AgentRunContext:
        """Build the complete process-local context for one mode invocation."""
        output_type = (
            self.output_contract.type
            if type(self.output_contract) is ValueContract
            else None
        )
        return AgentRunContext(
            node_id=self.id,
            prompt=prompt,
            tools=list(self.tools),
            llm_config=self.llm_config,
            output_type=output_type,
            retries=self.retries,
            max_tool_iterations=self._max_tool_iterations(),
            stop_event=stop_event,
            evaluation_trace=evaluation_trace,
            memory_sections=memory_sections,
        )

    def exception_origin(self, exception: Exception) -> OriginRef | None:
        """Select a generated structured-output contract origin when proven.

        Unrelated Pydantic errors, non-validation failures, and contracts without
        authored provenance retain the common node-origin fallback.
        """
        if isinstance(exception, MemoryRecallError):
            return (
                exception.origin
                if exception.origin is not None
                else super().exception_origin(exception)
            )
        contract = self.output_contract
        if type(contract) is not ValueContract:
            return super().exception_origin(exception)
        path = _structured_validation_path(exception, contract.type)
        if path is None:
            return super().exception_origin(exception)
        origin = contract.origins.at(path)
        return origin if origin is not None else super().exception_origin(exception)

    def forward(
        self,
        inputs: dict,
        *,
        stop_event=None,
        evaluation_trace: EvaluationTrace | None = None,
        recall_memory: Callable[[str, str], tuple[MemorySection, ...]] | None = None,
        capture_render: Callable[[dict[str, Any]], None] | None = None,
    ) -> Output:
        """Render fresh locals once, publish established capture, and run the mode.

        ``capture_render`` is an optional evaluator-owned sink. It receives the
        actual template, resolved variable values/domains, and rendered prompt
        once, including established partial locals if preparation fails. Unknown
        variables or an unfinished prompt remain None; capture faults propagate.
        No invocation state is stored on the node.

        Args:
            inputs (`dict`): Delivered inputs, kept separate from local variables.
            stop_event (*optional*, defaults to `None`): Cooperative host stop
                signal checked before recall and mode execution.
            evaluation_trace (`EvaluationTrace | None`, *optional*, defaults to
                `None`): Write-only visible prompt and interaction collector.
            recall_memory (`Callable | None`, *optional*, defaults to `None`):
                Recall ordered sections from the base and actual rendered prompt.
            capture_render (`Callable | None`, *optional*, defaults to `None`):
                Publish one dictionary of actual or explicitly unavailable locals.
                Omission leaves render capture dormant.

        Returns:
            `Output`: The selected mode's result.

        Raises:
            `ValueError`: A variable collides with a declared or delivered input.
            `_CapabilityFault`: Capture-only preparation fails; the evaluator
                re-raises its original host exception.
            `_HostStop`: The host has requested cooperative cancellation.
        """
        captured = None if capture_render is None else {
            "prompt_template": self.prompt, "variables": None,
            "variable_domains": None, "rendered_prompt": None,
        }
        try:
            input_names = set(inputs) | {parameter.name for parameter in (self.params or [])}
            for variable in self.variables:
                if variable.name in input_names:
                    raise ValueError(f"variable {variable.name!r} collides with an input name")
            if captured is not None:
                try:
                    captured["variable_domains"] = {
                        variable.name: canonical_type_label(variable.contract.type) for variable in self.variables
                    }
                except (_CapabilityFault, _HostStop):
                    raise
                except Exception as exc:
                    raise _CapabilityFault(exc) from exc
            variables = self._resolve_variables()
            template = self.prompt
            if captured is not None:
                try:
                    captured["variables"] = deepcopy(variables)
                except (_CapabilityFault, _HostStop):
                    raise
                except Exception as exc:
                    raise _CapabilityFault(exc) from exc
                captured["prompt_template"] = template
            prompt = render_template_record(template, {**inputs, **variables})
            if captured is not None:
                captured["rendered_prompt"] = prompt
        finally:
            if capture_render is not None:
                capture_render(captured)
        if stop_event is not None and stop_event.is_set():
            raise _HostStop(primary=False)
        memory_sections = (
            ()
            if recall_memory is None
            else recall_memory(DEFAULT_SYSTEM, prompt)
        )
        if stop_event is not None and stop_event.is_set():
            raise _HostStop(primary=False)
        return MODES[self.mode](
            self._ctx(
                prompt,
                stop_event=stop_event,
                evaluation_trace=evaluation_trace,
                memory_sections=memory_sections,
            )
        )

    def _resolve_variables(self) -> dict[str, Any]:
        """Resolve validated literal defaults into one detached invocation record."""
        return deepcopy(apply_defaults(self.variables, {}))

    def declared_variables(self) -> tuple[Parameter, ...]:
        """Return the canonical local declarations in their authored order."""
        return tuple(self.variables)
