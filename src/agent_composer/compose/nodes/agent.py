"""kind=agent — the LLM-reasoner node's descriptor + parse/build handle.

One of the two-axis `compose/nodes/<kind>.py` modules: it owns the `agent` node's frozen
`AgentDescriptor`, inheriting the template `parse` from `NodeDescriptor` (which records each
field's `SourceRange`), and overriding `build` (which constructs the runtime `AgentNode` +
wiring). `_finish` stamps only the descriptor's opaque origin onto that node and retains
concrete node/field ranges on the Compose-owned `BuildResult`. Registered into
`KINDS`/`BINDING_FIELDS` at import so the shared dispatch routes here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Optional

from agent_composer.compose.helpers.errors import LoadDiagnostics, LoadError
from agent_composer.compose.helpers.memory import parse_memory_declarations
from agent_composer.compose.helpers.types import read_variables
from agent_composer.compose.nodes.base import BuildContext, BuildResult, NodeDescriptor
from agent_composer.compose.source import SourceCursor
from agent_composer.diagnostics import SourceRange
from agent_composer.llm_clients import LLMConfig
from agent_composer.llm_clients.config import host_only_field_error
from agent_composer.memory.contracts import DeclSite, MemoryDecl
from agent_composer.nodes.agent import AgentNode
from agent_composer.nodes.agent.node import is_text_output
from agent_composer.nodes.contracts import Parameter, ValueContract
from agent_composer.skills import SkillError

if TYPE_CHECKING:
    from agent_composer.compose.source.catalog import _SourceCatalogBuilder


def _validate_llm_config(
    cfg: Mapping[str, Any] | None,
    node_id: str,
    source_range: SourceRange | None = None,
    diagnostics: LoadDiagnostics | None = None,
) -> tuple[dict[str, Any], bool]:
    """Validate a raw llm_config mapping at LOAD time and split off `inherit`.

    `inherit` (whole-node cascade opt-out, default True) is NOT an `LLMConfig` field, so it
    must be popped before the `extra="forbid"` round-trip that fires on typo'd keys (e.g.
    `temparature`). Returns `(config_without_inherit, inherit)`. Raises `LoadError` (not
    pydantic ValidationError) so the surface speaks one error type.

    Module-local to `agent.py`: agent is the only kind with an `llm_config:` field (model has
    none), so the validator lives next to its sole caller rather than in shared machinery.

    Args:
        cfg (`Mapping[str, Any] | None`): Authored config, or `None` for defaults.
        node_id (`str`): Node id included in load-error messages.
        source_range (`SourceRange | None`): Authored config location, when retained.

    Returns:
        `tuple[dict[str, Any], bool]`: Config without `inherit`, then its boolean value.

    Raises:
        `LoadError`: The config is not a mapping, sets `api_key` or `base_url`, or
            violates `LLMConfig`.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    if cfg is None:
        return {}, True
    if not isinstance(cfg, Mapping):
        raise diagnostics.error(
            f"node {node_id!r}: llm_config must be a mapping",
            range=source_range,
        )
    cfg = dict(cfg)
    if not cfg:
        return {}, True
    inherit = bool(cfg.pop("inherit", True))
    host_only = host_only_field_error(cfg)
    if host_only is not None:
        raise diagnostics.error(f"node {node_id!r}: {host_only}", range=source_range)
    try:
        LLMConfig(**cfg)
    except Exception as exc:
        raise diagnostics.error(
            f"node {node_id!r}: llm_config: {exc}",
            range=source_range,
        ) from exc
    return cfg, inherit


@dataclass(frozen=True, kw_only=True)
class AgentDescriptor(NodeDescriptor):
    """kind=agent — an LLM reasoner (prompt + optional system prompt + mode + optional tools/llm).

    Inherits the common descriptor fields (`id`/`node_name`/`depends_on`/`runs_after`/`env` +
    the Compose-only `range`/`field_ranges` parse metadata) from `NodeDescriptor`, along with the `input:`/
    `output:` node-signature surface and the template `parse`. Its cursor hook rejects the
    retired memory key before the default allowed/required key check; only the agent-specific
    fields are declared here, while `inputs`/`outputs` come from the base.

    Class attrs: `kind="agent"`; `allowed` adds normalized `inputs`/`outputs` plus `prompt`/
    `system_prompt`/`tools`/`variables`/`mode`/`llm_config`/`long_term_memory`/`retries`/`asserts`/
    `steerable` to the commons; `required` is empty (no mandatory flat field);
    `binding_fields=("inputs",)` — the `${...}` refs of `prompt:` and `system_prompt:` are handled
    by template scanning, not the inline-call desugar, so only `inputs` is walked.
    """

    kind = "agent"
    allowed = NodeDescriptor.allow(
        "inputs",
        "outputs",
        "prompt",
        "system_prompt",
        "variables",
        "tools",
        "skills",
        "mode",
        "llm_config",
        "long_term_memory",
        "retries",
        "asserts",
        "steerable",
    )
    required = NodeDescriptor.require()
    binding_fields = ("inputs",)
    keyword_remap = {"long_term_memory": "memories"}

    prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    variables: dict[str, Any] = field(default_factory=dict)
    tools: list[str] = field(default_factory=list)
    skills: list[str] | str = field(default_factory=list)
    mode: str = "tool_calling"
    llm_config: dict[str, Any] = field(default_factory=dict)
    retries: int = 2
    asserts: list[str] = field(default_factory=list)
    steerable: bool = False
    memories: tuple[MemoryDecl, ...] = ()
    # INTERNAL — never parsed from YAML; set only by the adaptive_questions desugar pass.
    # When present it overrides the surface-derived contract for a synthesized structured
    # agent whose codomain is code-built and has no surface type-string.
    output_contract_override: Optional[ValueContract] = None

    def __post_init__(self) -> None:
        """Validate the source-neutral declaration tuple."""
        super().__post_init__()
        if type(self.memories) is not tuple or any(
            type(decl) is not MemoryDecl for decl in self.memories
        ):
            raise TypeError("agent descriptor memories must be a tuple of exact MemoryDecl values")

    @classmethod
    def _normalize_cursor(cls, cursor: SourceCursor) -> SourceCursor:
        """Reject the retired authored memory key before generic key checking."""
        if cursor.has("memory_policy"):
            diagnostics = LoadDiagnostics.from_cursor(cursor)
            raise diagnostics.error(
                "node field `memory_policy:` was renamed to `long_term_memory:`",
                range=cursor.key_range_of("memory_policy"),
            )
        return cursor

    @classmethod
    def _validate_fields(
        cls,
        node_id: str,
        kwargs: dict[str, Any],
        ranges: dict[tuple[str, str], SourceRange],
        own_range: SourceRange | None,
        diagnostics: LoadDiagnostics,
    ) -> None:
        """Check the `system_prompt:`, `skills:`, and `steerable:` shapes and parse a present named policy."""
        if "system_prompt" in kwargs:
            system_prompt = kwargs["system_prompt"]
            if not isinstance(system_prompt, str) or not system_prompt:
                raise diagnostics.error(
                    f"node {node_id!r} (kind={cls.kind}): `system_prompt:` must be a "
                    "non-empty string",
                    range=ranges.get(("field", "system_prompt")) or own_range,
                )
        skills = kwargs.get("skills")
        if skills is not None and skills != "all" and not (
            isinstance(skills, list) and all(isinstance(name, str) for name in skills)
        ):
            raise diagnostics.error(
                f"node {node_id!r} (kind={cls.kind}): `skills:` must be a list of "
                "skill names or `all`",
                range=ranges.get(("field", "skills")) or own_range,
            )
        if type(kwargs.get("steerable", False)) is not bool:
            raise diagnostics.error(
                f"node {node_id!r} (kind={cls.kind}): `steerable:` must be true or false",
                range=ranges.get(("field", "steerable")) or own_range,
            )
        if "memories" in kwargs:
            value = kwargs["memories"]
            kwargs["memories"] = (
                ()
                if value is None
                else parse_memory_declarations(
                    value,
                    diagnostics=diagnostics,
                    source_range=ranges.get(("field", "long_term_memory")),
                    site=DeclSite((), ("root",), node_id),
                )
            )
        super()._validate_fields(
            node_id,
            kwargs,
            ranges,
            own_range,
            diagnostics,
        )

    def with_source_origins(
        self,
        cursor: SourceCursor,
        *,
        site_path: tuple[str | int, ...],
        catalog_builder: "_SourceCatalogBuilder",
    ) -> "AgentDescriptor":
        """Attach one origin per authored ``long_term_memory:`` entry."""
        if not self.memories or not cursor.has("long_term_memory"):
            return self
        value_cursor = cursor.child("long_term_memory")
        stamped = []
        for decl in self.memories:
            path = (*site_path, "field", "memories", decl.name)
            if value_cursor.has(decl.name):
                origin = catalog_builder.register(
                    path, value_cursor.child(decl.name), anchor="key"
                )
            else:  # sugar: the whole scalar value is the one entry
                origin = catalog_builder.register(path, value_cursor)
            stamped.append(
                MemoryDecl(decl.name, decl.type, decl.config, decl.site, origin)
            )
        return replace(self, memories=tuple(stamped))

    def build(self, ctx: BuildContext) -> BuildResult:
        """Construct the runtime `AgentNode` (+ wiring) for this `AgentDescriptor`.

        Builds the agent leaf (LLM-config and local-variable validation, node-side
        `params` + flow-owned `inputs` wiring, `output_contract` from authored `output:`/override).
        `_finish` stamps the descriptor origin on the node and keeps its ranges on the
        Compose-owned `BuildResult`. `ctx.resolver` is unused (leaf kind), and the inline-CODE
        anchors on `ctx` are read only by the CODE build.
        """
        llm_config_range = self.field_ranges.get(("field", "llm_config")) or self.range
        own_cfg, inherit = _validate_llm_config(
            self.llm_config,
            self.id,
            llm_config_range,
            ctx.diagnostics,
        )
        mode_range = self.field_ranges.get(("field", "mode")) or self.range
        if not isinstance(self.mode, str):
            raise ctx.diagnostics.error(
                f"node {self.id!r}: mode must be a string",
                range=mode_range,
            )
        steerable_range = self.field_ranges.get(("field", "steerable")) or self.range
        if self.steerable and self.mode != "tool_calling":
            raise ctx.diagnostics.error(
                f"node {self.id!r}: steerable needs mode 'tool_calling', not {self.mode!r}",
                range=steerable_range,
            )
        variables = self._read_variables(ctx)
        memories = tuple(
            MemoryDecl(decl.name, decl.type, decl.config, DeclSite((), ctx.owner, self.id), decl.origin)
            for decl in self.memories
        )
        try:
            node = AgentNode(
                self.id,
                prompt=self.prompt or "",
                system_prompt=self.system_prompt,
                tools=list(self.tools),
                llm_config=own_cfg,
                llm_inherit=inherit,
                mode=self.mode,
                retries=self.retries,
                title=self.node_name,
                memories=memories,
                variables=variables,
                skills=self.skills,
                steerable=self.steerable,
            )
        except ValueError as exc:
            # AgentNode validates the mode registry and skill grants; surface a
            # rejection as the loader's one error type (loud), not a bare ValueError.
            skills_range = self.field_ranges.get(("field", "skills")) or self.range
            raise ctx.diagnostics.error(
                f"node {self.id!r}: {exc}",
                range=skills_range if isinstance(exc, SkillError) else mode_range,
            ) from exc

        override = self.output_contract_override
        node.output_contract = (
            override
            if override is not None
            else self.output_contract(ctx)
        )
        if self.steerable and not is_text_output(node.output_contract):
            # Checked here too so the error points at the `steerable:` line.
            raise ctx.diagnostics.error(
                f"node {self.id!r}: steerable needs `str` output",
                range=steerable_range,
            )
        node.params = self.sink_params()
        input_names = {parameter.name for parameter in node.params}
        for variable in variables:
            if variable.name in input_names:
                frame = (
                    ctx.resolve_origin(variable.origin)
                    if ctx.resolve_origin is not None and variable.origin is not None
                    else None
                )
                message = f"node {self.id!r}: variable {variable.name!r} collides with an input name"
                if frame is not None:
                    raise ctx.diagnostics.error(message, frames=(frame,))
                raise ctx.diagnostics.error(
                    message,
                    range=self.field_ranges.get(("field", "variables")) or self.range,
                )
        wiring = self.sink_wiring()
        return self._finish(node, wiring)

    def _read_variables(self, ctx: BuildContext) -> list[Parameter]:
        """Compile this node's declarations at their authored or source-free site."""
        try:
            if ctx.variable_declaration_reader is not None:
                variables = ctx.variable_declaration_reader(self.id, self.variables)
                if variables is not None:
                    if type(variables) is not list or any(
                        type(variable) is not Parameter for variable in variables
                    ):
                        raise TypeError(
                            "variable declaration reader must return a list of exact Parameters or None"
                        )
                    return variables
            return read_variables(self.variables, ctx.registry, diagnostics=ctx.diagnostics)
        except LoadError as exc:
            if exc.diagnostic.frames:
                raise
            raise ctx.diagnostics.prepend(
                exc,
                range=self.field_ranges.get(("field", "variables")) or self.range,
            ) from exc
