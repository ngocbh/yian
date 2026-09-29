"""AGENT node — one authored leaf invoking a model through its selected mode.

Modes drive their local model/tool loop through the shared provider interface.
An optional memory backend owns
conversation admission and retained history; the evaluator owns the turn's
lifetime. Tools, including the host-installed ``ask_user``, are ordinary
explicitly authored tool grants. A crash or host interruption unwinds the node;
no conversation or continuation state is stored in the graph or variable pool.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Iterator, Literal, Mapping, Optional

from agent_composer.diagnostics import OriginRef
from agent_composer.expr.template import render_template_record
from agent_composer.expr.lineage import FeedbackValue
from agent_composer.memory.contracts import (
    MemoryContext,
    EvaluationTrace,
    MemoryPolicySpec,
    MemoryRecallError,
    _iter_memory_policy_origin,
    _rewrite_memory_policy_origin,
)
from agent_composer.nodes.agent.modes import DEFAULT_SYSTEM, MODES, AgentRunContext
from agent_composer.nodes.agent.context import compatibility
from agent_composer.nodes.agent.backward import backward_agent
from agent_composer.nodes.agent.modes.tool_calling import MAX_TOOL_ITERATIONS
from agent_composer.nodes.agent.structured import _structured_validation_path
from agent_composer.nodes.base import BackwardContext, BackwardResult, Node, NodeKind, Output, _CapabilityFault, _HostStop
from agent_composer.nodes.binding import apply_defaults
from agent_composer.nodes.contracts import (
    Parameter,
    ValueContract,
    _iter_origin_carrier,
    _rewrite_origin_carrier,
)
from agent_composer.skills.catalog import Skill, SkillError, check_skill_name
from agent_composer.skills.tool import LOAD_SKILL, system_prompt_with_skills
from agent_composer.typesys.types import canonical_type_label
from agent_composer.typesys.values import build_value_as

DEFAULT_MODE = "plain"


class AgentNode(Node):
    """An LLM-backed leaf configured by a prompt, mode, and authored tools.

    Modes normalize ``llm_config`` through the call-time ``model_from_config``
    seam. An optional
    ``memory_policy`` is authored identity only until compile derives runtime scope.
    ``variables`` retains typed local declarations with mandatory literal defaults,
    resolved into a fresh prompt record for each invocation.
    """

    kind = NodeKind.AGENT
    needs_stop_event = True
    needs_evaluation_trace = True
    needs_memory_context = True
    needs_render_capture = True
    needs_skills = True

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
        skills: list[str] | Literal["all"] | None = None,
    ) -> None:
        super().__init__(node_id, title=title)
        self.prompt = prompt or ""
        self.tools = tools or []
        if "builtin_tools" in self.tools:
            raise ValueError(
                "builtin_tools belonged to the removed native Codex integration. "
                "Remove it and grant registered Composer tools explicitly."
            )
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
        self.skills: list[str] | Literal["all"] = [] if skills is None else skills
        self._validate_skills()
        self._validate_variables()

    def _validate_skills(self) -> None:
        """Require `all` or unique valid names, in tool-calling mode."""
        where = f"AGENT node {self.id!r}"
        if LOAD_SKILL in self.tools:
            raise SkillError(f"{where}: {LOAD_SKILL!r} is a reserved tool name; remove it from tools")
        if self.skills != "all":
            if type(self.skills) is not list:
                raise SkillError(f"{where}: skills must be a list of names or 'all'")
            for name in self.skills:
                check_skill_name(name)
            if len(set(self.skills)) != len(self.skills):
                raise SkillError(f"{where}: duplicate skill grant in {self.skills}")
        if self.skills and self.mode != "tool_calling":
            raise SkillError(f"{where}: skills need mode 'tool_calling', not {self.mode!r}")

    def _granted_skills(self, supplied: Mapping[str, Skill] | None) -> tuple[Skill, ...]:
        """Select this node's grants from the run's host-supplied skills, by name."""
        if not self.skills:
            return ()
        if supplied is None:
            raise SkillError(
                f"AGENT node {self.id!r} grants skills, but the host supplied no skills= mapping"
            )
        if self.skills == "all":
            names = sorted(supplied)
        else:
            missing = [name for name in self.skills if name not in supplied]
            if missing:
                raise SkillError(
                    f"AGENT node {self.id!r}: granted skills not supplied by the host: {missing}"
                )
            names = sorted(self.skills)
        return tuple(supplied[name] for name in names)

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
        memory_context: MemoryContext | None = None,
        base_system_prompt: str = DEFAULT_SYSTEM,
        skills: tuple[Skill, ...] = (),
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
            memory_context=memory_context if memory_context is not None else MemoryContext(),
            base_system_prompt=base_system_prompt,
            skills=skills,
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
        path = _structured_validation_path(
            exception, contract.type
        )
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
        open_memory_context: Callable[[str, str, dict[str, Any]], MemoryContext] | None = None,
        capture_render: Callable[[dict[str, Any]], None] | None = None,
        skills: Mapping[str, Skill] | None = None,
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
            evaluation_trace (`EvaluationTrace | None`, *optional*, defaults to `None`):
                Write-only visible prompt and interaction collector.
            open_memory_context (`Callable | None`, *optional*, defaults to `None`):
                Recall notes and admit an optional transcript after rendering.
                The evaluator binds identity and owns the context lifetime.
            capture_render (`Callable | None`, *optional*, defaults to `None`):
                Publish one dictionary of actual or explicitly unavailable locals.
                Omission leaves render capture dormant.
            skills (`Mapping[str, Skill] | None`, *optional*, defaults to `None`):
                The run's host-supplied skills. Granted ones are listed in the
                system message and served by the built-in ``load_skill`` tool.

        Returns:
            `Output`: The selected mode's result.

        Raises:
            `ValueError`: A variable collides with a declared or delivered input.
            `SkillError`: The node grants skills the host did not supply.
            `_CapabilityFault`: Capture or conversation access fails; the
                evaluator re-raises its original host exception.
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
        granted = self._granted_skills(skills)
        base_system_prompt = system_prompt_with_skills(DEFAULT_SYSTEM, granted)
        memory_context = (
            MemoryContext()
            if open_memory_context is None
            else open_memory_context(DEFAULT_SYSTEM, prompt, compatibility(self.llm_config))
        )
        if stop_event is not None and stop_event.is_set():
            raise _HostStop(primary=False)
        return MODES[self.mode](
            self._ctx(
                prompt,
                stop_event=stop_event,
                evaluation_trace=evaluation_trace,
                memory_context=memory_context,
                base_system_prompt=base_system_prompt,
                skills=granted,
            )
        )

    def _resolve_variables(self) -> dict[str, Any]:
        """Resolve validated literal defaults into one detached invocation record."""
        return deepcopy(apply_defaults(self.variables, {}))

    def backward(self, context: BackwardContext, feedback: tuple[FeedbackValue, ...]) -> BackwardResult:
        """Use only recorded invocation material for one tool-free backward call.

        Args:
            context (`BackwardContext`): Accepted context, including recorded model
                configuration and render/evidence components; live node fields are unused.
            feedback (`tuple[FeedbackValue, ...]`): Ordered ordinary output messages.
                Empty effective feedback skips model construction.

        Returns:
            `BackwardResult`: Independent input, self, and variable channels for
                common result validation by the caller before upstream routing.

        Raises:
            `TypeError`: Context, feedback, captured fields, or response channels are malformed.
            `ValueError`: Feedback shape, required capture, or response JSON is invalid.
                Model and host exceptions propagate to the existing driver boundary.

        Examples:
            `node.backward(recorded_context, ())` returns empty channels without
            constructing a model. Effective messages use only that recorded context.
        """
        return backward_agent(context, feedback)

    def declared_variables(self) -> tuple[Parameter, ...]:
        """Return the canonical local declarations in their authored order."""
        return tuple(self.variables)
