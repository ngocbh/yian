"""kind=agent — the LLM-reasoner node's descriptor + parse/build handle.

One of the two-axis `compose/nodes/<kind>.py` modules: it owns the `agent` node's frozen
`AgentDescriptor`, inheriting the template `parse` from `NodeDescriptor` (which records each
field's `SourceRange`), and overriding `build` (which constructs the runtime `AgentNode` +
wiring and stamps span metadata onto it). Registered into `KINDS`/`BINDING_FIELDS` at import
so the shared dispatch routes here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

from agent_composer.compose.helpers.errors import LoadDiagnostics
from agent_composer.compose.nodes.base import BuildContext, BuildResult, NodeDescriptor
from agent_composer.diagnostics import SourceRange
from agent_composer.llm_clients import LLMConfig
from agent_composer.nodes.agent import AgentNode
from agent_composer.nodes.contracts import ValueContract


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
        `LoadError`: The config is not a mapping or violates `LLMConfig`.
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
    """kind=agent — an LLM reasoner (prompt + mode + optional tools/llm).

    Inherits the common node fields (`id`/`node_name`/`depends_on`/`runs_after`/`env` + the
    `range`/`field_ranges` parse metadata) from `NodeDescriptor`, along with the `inputs:`/
    `outputs:` node-signature surface and the template `parse` (no custom parse hook — agent's
    field legality is the default allowed/required key check + the asserts-list check). Only the
    agent-specific fields are declared here; `inputs`/`outputs` come from the base.

    Class attrs: `kind="agent"`; `allowed` adds `inputs`/`outputs`/`prompt`/`tools`/
    `mode`/`llm_config`/`retries`/`asserts` to the commons; `required` is empty (no mandatory
    flat field); `binding_fields=("inputs",)` — a `prompt:`'s `${...}` refs are handled by
    template scanning, not the inline-call desugar, so only `inputs` is walked.
    """

    kind = "agent"
    allowed = NodeDescriptor.allow(
        "inputs",
        "outputs",
        "prompt",
        "tools",
        "mode",
        "llm_config",
        "retries",
        "asserts",
    )
    required = NodeDescriptor.require()
    binding_fields = ("inputs",)

    prompt: Optional[str] = None
    tools: list[str] = field(default_factory=list)
    mode: str = "tool_calling"
    llm_config: dict[str, Any] = field(default_factory=dict)
    retries: int = 2
    asserts: list[str] = field(default_factory=list)
    # INTERNAL — never parsed from YAML; set only by the adaptive_questions desugar pass.
    # When present it overrides the surface-derived contract for a synthesized structured
    # agent whose codomain is code-built and has no surface type-string.
    output_contract_override: Optional[ValueContract] = None

    def build(self, ctx: BuildContext) -> BuildResult:
        """Construct the runtime `AgentNode` (+ wiring) for this `AgentDescriptor`.

        Builds the agent leaf (LLM-config validation, node-side
        `params` + flow-owned `inputs` wiring, `output_contract` from `outputs:`/override) and
        additionally stamps `.range`/`.field_ranges` from the descriptor via `_finish` (which
        wraps them in a `BuildResult`). `ctx.resolver` is unused (leaf kind) and the inline-CODE
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
        try:
            node = AgentNode(
                self.id,
                prompt=self.prompt or "",
                tools=list(self.tools),
                llm_config=own_cfg,
                llm_inherit=inherit,
                mode=self.mode,
                retries=self.retries,
                title=self.node_name,
            )
        except ValueError as exc:
            # AgentNode validates the mode registry; surface a rejection as
            # the loader's one error type (loud), not a bare ValueError.
            raise ctx.diagnostics.error(
                f"node {self.id!r}: {exc}",
                range=mode_range,
            ) from exc

        override = self.output_contract_override
        node.output_contract = (
            override
            if override is not None
            else self.output_contract(ctx)
        )
        node.params = self.sink_params()
        wiring = self.sink_wiring()
        return self._finish(node, wiring)
