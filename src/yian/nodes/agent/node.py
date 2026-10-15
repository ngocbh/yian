"""AGENT node — one authored leaf invoking a model.

It makes one call without tools or skills, and runs the tool loop with them.

The loop drives local model and tool calls through the shared provider interface.
An optional memory backend owns
conversation admission and retained history; the evaluator owns the turn's
lifetime. Tools, including the run's ``ask_user``, are ordinary
explicitly authored tool grants. A crash or host interruption unwinds the node;
no conversation or continuation state is stored in the graph or variable pool.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Iterator, Literal, Mapping, Optional

from yian.defaults import current
from yian.diagnostics import OriginRef
from yian.expr.template import render_template_record
from yian.expr.lineage import FeedbackValue
from yian.llm_clients.connection import scrub_call_credentials
from yian.memory.contracts import (
    MemoryContext,
    EvaluationTrace,
    MemoryDecl,
    MemoryRecallError,
    _iter_memory_decl_origins,
    _rewrite_memory_decl_origins,
)
from yian.nodes.agent.context import DEFAULT_SYSTEM, AgentRunContext
from yian.nodes.agent.memory import compatibility
from yian.nodes.agent.backward import backward_agent
from yian.nodes.agent.loop import run_agent
from yian.nodes.agent.structured import _structured_validation_path
from yian.nodes.base import (
    BackwardContext,
    BackwardResult,
    Node,
    NodeKind,
    Output,
    Steering,
    _CapabilityFault,
    _HostStop,
    _check_named_prompt,
)
from yian.nodes.binding import BindingError, apply_defaults
from yian.nodes.contracts import (
    Parameter,
    ValueContract,
    _iter_origin_carrier,
    _rewrite_origin_carrier,
)
from yian.skills.catalog import Skill, SkillError, check_skill_name
from yian.skills.tool import LOAD_SKILL, system_prompt_with_skills
from yian.tools import ToolSet
from yian.typesys.values import ValueKind, build_value_as, canonical_type_label


def is_text_output(contract: Any) -> bool:
    """Whether an AGENT output contract is plain text: none, or an untagged `str`."""
    return contract is None or (
        type(contract) is ValueContract
        and contract.type.kind is ValueKind.STRING
        and not contract.type.tags
    )


def is_max_tool_iterations(value: Any) -> bool:
    """Whether `value` is a valid `max_tool_iterations`: a positive int, or `-1` for no cap."""
    return type(value) is int and (value >= 1 or value == -1)


class AgentNode(Node):
    """An LLM-backed leaf configured by a prompt and authored tools.

    ``system_prompt`` is an optional template that replaces ``DEFAULT_SYSTEM``;
    it renders against the same local record as ``prompt``.

    The loop normalizes ``llm_config`` through the call-time ``model_from_config``
    seam. ``memories`` holds the AGENT's authored memory declarations; compile
    derives runtime scope. ``variables`` retains typed local declarations with
    mandatory literal defaults, resolved into a fresh prompt record for each
    invocation.
    """

    kind = NodeKind.AGENT
    needs_stop_event = True
    needs_evaluation_trace = True
    needs_memory_context = True
    needs_render_capture = True
    needs_tools = True
    needs_live_events = True
    needs_skills = True

    def __init__(
        self,
        node_id: str,
        *,
        prompt: str = "",
        system_prompt: str | None = None,
        tools: Optional[list[str]] = None,
        llm_config: Optional[dict[str, Any]] = None,
        llm_inherit: bool = True,
        retries: int | None = None,
        title: Optional[str] = None,
        memories: tuple[MemoryDecl, ...] = (),
        variables: list[Parameter] | None = None,
        skills: list[str] | Literal["all"] | None = None,
        steerable: bool = False,
    ) -> None:
        super().__init__(node_id, title=title)
        self.prompt = prompt or ""
        if system_prompt is not None and (type(system_prompt) is not str or not system_prompt):
            raise TypeError(
                f"AGENT node {node_id!r}: system_prompt must be a non-empty string or None"
            )
        self.system_prompt = system_prompt
        self.tools = tools or []
        if "builtin_tools" in self.tools:
            raise ValueError(
                "builtin_tools belonged to the removed native Codex integration. "
                "Remove it and grant registered Yian tools explicitly."
            )
        # own_llm_config is the authored source re-resolved by each cascade pass;
        # llm_config is the effective config baked for this run.
        self.own_llm_config: dict[str, Any] = dict(llm_config) if llm_config else {}
        self.llm_config = llm_config
        self.llm_inherit = llm_inherit
        self.retries = current().agent_retries if retries is None else retries
        if type(memories) is not tuple or any(
            type(decl) is not MemoryDecl for decl in memories
        ):
            raise TypeError("memories must be a tuple of exact MemoryDecl values")
        if any(decl.site.node is None for decl in memories):
            raise ValueError("AGENT memories must carry AGENT DeclSites")
        self.memories = memories
        self.variables = [] if variables is None else variables
        self.skills: list[str] | Literal["all"] = [] if skills is None else skills
        self._validate_skills()
        self._validate_variables()
        if type(steerable) is not bool:
            raise TypeError(f"AGENT node {node_id!r}: steerable must be a bool")
        self.steerable = steerable
        # `self.llm_config_label`: the author-facing phrase naming this agent's `llm_config` when
        #   a desugar synthesized the agent from another node's text (e.g.
        #   `node 'ask' adaptive_questions.llm_config`). Render-only and absent from run records
        #   and the manifest. `None` for an authored agent, whose messages name its path.
        self.llm_config_label: str | None = None

    @property
    def needs_steering(self) -> bool:
        """Receive the host's steering inbox only when authored ``steerable: true``."""
        return self.steerable

    def validate_execution_setup(self) -> None:
        """Reject a steerable agent whose output contract is not plain text.

        The contract is assigned after construction, so this is checked here as
        well as at YAML load.
        """
        super().validate_execution_setup()
        if self.steerable and not is_text_output(self.output_contract):
            # An interrupted node returns its text so far; a structured value can't be partial.
            raise ValueError(f"AGENT node {self.id!r}: steerable needs `str` output")

    def validate_prompts(self, *, path: str | None = None) -> None:
        """Check the prompt and system prompt templates.

        Args:
            path (`str`, *optional*, defaults to `None`):
                The static node path to name in the error instead of the node's id,
                used by a pre-run check over a whole flow.

        Raises:
            `ExpressionError`:
                Raised when a template's `${...}` span doesn't parse, calls `isinstance` or
                a function that isn't a builtin, or calls a builtin with arguments it doesn't
                take; the message names the node and the field.
        """
        for field, text in (("prompt", self.prompt), ("system_prompt", self.system_prompt)):
            if isinstance(text, str):
                _check_named_prompt(text, f"node {(path or self.id)!r} {field}")

    def _validate_skills(self) -> None:
        """Require `all` or unique valid names."""
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

    def granted_tools(self) -> tuple[str, ...]:
        """Return the authored `tools:` grants, which may name host-filled groups."""
        return tuple(self.tools)

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
        """Yield common origins, variable declarations, and memory declaration sites."""
        seen = _seen_carriers if _seen_carriers is not None else set()
        yield from super().iter_local_origins(_seen_carriers=seen)
        for variable in self.variables:
            yield from _iter_origin_carrier(variable, seen)
        yield from _iter_memory_decl_origins(self.memories, seen)

    def rewrite_local_origins(
        self,
        mapper: Callable[[OriginRef], OriginRef],
        *,
        _seen_carriers: Optional[set[int]] = None,
    ) -> None:
        """Rewrite common, variable, and memory declaration origins while retaining aliases."""
        seen = _seen_carriers if _seen_carriers is not None else set()
        super().rewrite_local_origins(mapper, _seen_carriers=seen)
        for variable in self.variables:
            _rewrite_origin_carrier(variable, mapper, seen)
        _rewrite_memory_decl_origins(self.memories, mapper, seen)

    def _max_tool_iterations(self) -> int:
        """Return the authored or default positive loop bound, or ``-1`` for no bound."""
        raw = self.env.get("max_tool_iterations", current().agent_max_tool_iterations)
        if not is_max_tool_iterations(raw):
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
        toolset: ToolSet | None = None,
        emit_stream: Callable[[Any], None] | None = None,
        emit_event: Callable[[Any], None] | None = None,
        base_system_prompt: str = DEFAULT_SYSTEM,
        skills: tuple[Skill, ...] = (),
        steering: Steering | None = None,
    ) -> AgentRunContext:
        """Build the complete process-local context for one invocation."""
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
            toolset=toolset if toolset is not None else ToolSet(),
            emit_stream=emit_stream,
            emit_event=emit_event,
            base_system_prompt=base_system_prompt,
            skills=skills,
            steering=steering,
        )

    def exception_origin(self, exception: Exception) -> OriginRef | None:
        """Select a generated structured-output contract origin when proven.

        A memory recall or variable-default binding failure keeps its own origin.
        Unrelated Pydantic errors, non-validation failures, and contracts without
        authored provenance retain the common node-origin fallback.
        """
        if isinstance(exception, (MemoryRecallError, BindingError)):
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
        tools: ToolSet | None = None,
        emit_stream: Callable[[Any], None] | None = None,
        emit_event: Callable[[Any], None] | None = None,
        skills: Mapping[str, Skill] | None = None,
        steering: Steering | None = None,
    ) -> Output:
        """Render fresh locals once, publish established capture, and run the loop.

        The authored ``system_prompt`` renders against the same inputs and
        variables as ``prompt``; without one the loop receives ``DEFAULT_SYSTEM``.
        ``capture_render`` is an optional evaluator-owned sink. It receives the
        actual template, resolved variable values/domains, and rendered prompt
        once, including established partial locals if preparation fails. Unknown
        variables or an unfinished prompt remain None; capture faults propagate.
        No invocation state is stored on the node.

        Args:
            inputs (`dict`): Delivered inputs, kept separate from local variables.
            stop_event (*optional*, defaults to `None`): Cooperative host stop
                signal checked before recall and the model call.
            evaluation_trace (`EvaluationTrace | None`, *optional*, defaults to `None`):
                Write-only visible prompt and interaction collector.
            open_memory_context (`Callable | None`, *optional*, defaults to `None`):
                Recall notes and admit an optional transcript after rendering.
                The evaluator binds identity and owns the context lifetime.
            capture_render (`Callable | None`, *optional*, defaults to `None`):
                Publish one dictionary of actual or explicitly unavailable locals.
                Omission leaves render capture dormant.
            tools (`ToolSet | None`, *optional*, defaults to `None`):
                The run's tools. The node's `tools` list grants ids from it.
                `None` uses the global registry.
            emit_stream (`Callable | None`, *optional*, defaults to `None`):
                Transient `StreamChunk` sink; when present, chat replies stream.
            emit_event (`Callable | None`, *optional*, defaults to `None`):
                Transient sink for model/tool call observers with token usage.
            skills (`Mapping[str, Skill] | None`, *optional*, defaults to `None`):
                The run's host-supplied skills. Granted ones are listed in the
                system message and served by the built-in ``load_skill`` tool.
            steering (`Steering | None`, *optional*, defaults to `None`):
                The host inbox, supplied only when this node is steerable. The
                tool loop drains steer messages and honors interrupts.

        Returns:
            `Output`: The model's answer.

        Raises:
            `ValueError`: A variable collides with a declared or delivered input.
            `ExpressionError`: The prompt or system prompt names a missing value or
                renders a null.
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
            system_prompt = (
                DEFAULT_SYSTEM
                if self.system_prompt is None
                else render_template_record(self.system_prompt, {**inputs, **variables})
            )
        finally:
            if capture_render is not None:
                capture_render(captured)
        if stop_event is not None and stop_event.is_set():
            raise _HostStop(primary=False)
        granted = self._granted_skills(skills)
        base_system_prompt = system_prompt_with_skills(system_prompt, granted)
        # Recall sees the system prompt without the skill listing, so grants never change recall.
        memory_context = (
            MemoryContext()
            if open_memory_context is None
            else open_memory_context(system_prompt, prompt, compatibility(self.llm_config))
        )
        if stop_event is not None and stop_event.is_set():
            raise _HostStop(primary=False)
        try:
            return run_agent(self._ctx(
                prompt,
                stop_event=stop_event,
                evaluation_trace=evaluation_trace,
                memory_context=memory_context,
                toolset=tools,
                emit_stream=emit_stream,
                emit_event=emit_event,
                base_system_prompt=base_system_prompt,
                skills=granted,
                steering=steering,
            ))
        except BaseException as exc:
            # Providers may echo any credential the call sent; the failure record must not carry it.
            scrub_call_credentials(exc, self.llm_config)
            raise

    def _host_api_key(self) -> Any:
        """Return the host key bound to this node for the run, if any."""
        return self.llm_config.get("api_key") if isinstance(self.llm_config, dict) else None

    def _resolve_variables(self) -> dict[str, Any]:
        """Resolve validated literal defaults into one detached invocation record."""
        return deepcopy(apply_defaults(self.variables, {}))

    def backward(self, context: BackwardContext, feedback: tuple[FeedbackValue, ...]) -> BackwardResult:
        """Use only recorded invocation material for one tool-free backward call.

        Args:
            context (`BackwardContext`): Accepted context, including recorded model
                configuration and render/evidence components. Live node fields are
                unused, except a host ``api_key`` bound by `resume_flow` or
                `RunResult.backward`.
            feedback (`tuple[FeedbackValue, ...]`): Ordered ordinary output messages.
                Empty effective feedback skips model construction.

        Returns:
            `BackwardResult`: Independent input, self, and variable channels for
                common result validation by the caller before upstream routing.

        Raises:
            `TypeError`: Context, feedback, captured fields, or response channels are malformed.
            `ValueError`: Feedback shape, required capture, or response JSON is invalid.
                Model and host exceptions propagate to the existing driver boundary,
                model errors with every credential the call sent scrubbed.

        Examples:
            `node.backward(recorded_context, ())` returns empty channels without
            constructing a model. Effective messages use only that recorded context.
        """
        # The one live field: a host key, which is never recorded, bound for this call.
        return backward_agent(context, feedback, api_key=self._host_api_key())

    def declared_variables(self) -> tuple[Parameter, ...]:
        """Return the canonical local declarations in their authored order."""
        return tuple(self.variables)
