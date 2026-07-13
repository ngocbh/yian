"""AGENT node — one node configured by `mode` + `tools` + `controls`, entered two ways.

Three knobs:
- **mode** (`AgentBody.mode`) — *how* it runs: the loop / prompting method, one of
  the registered `MODES` (`plain` = a single call, `tool_calling` = a loop, later
  `react`). Pick one.
- **tools** (`AgentBody.tools`) — ordinary tools (from `agent_composer.tools`, e.g.
  `web_search`, data fetchers) a mode may call. (`plain` ignores them;
  `tool_calling` binds and loops over them.)
- **controls** (`AgentBody.controls`) — *control tools* (from `nodes/agent/controls/`,
  e.g. `ask_user`) that drive an engine effect when called, like suspending the run.

**Entry — the algebraic-effect split (`entry: Fresh | Resume`).** An AGENT node is
entered one of two ways, a closed two-variant sum:

- `Fresh(prompt)` — the author's agent: render the prompt against the node's declared
  inputs and start a new conversation; the selected `mode` runs the loop.
- `Resume(memo, iterations, pending)` — the **delimited continuation** of an `ask_user`
  control pause: rebuild the conversation from `memo`, append the human `answer` (its
  sole declared input) as the `ToolMessage` matching `pending["call_id"]`, then continue
  the loop. Minted only by the engine's continuation cloner (`clone_continuation_pair`);
  never authored. The re-entry frame rides as node DATA, not the pool — that is what
  keeps the resume entry self-contained and lets the closed-kind law hold with no exception.

A fresh agent and a resumed agent share the loop body (`agent_step`) but are DIFFERENT
typed arrows — `declared_inputs -> NodeResult` vs `answer -> NodeResult` — so the
distinction is an internal variant matched in `run`, NOT a second `NodeKind` (one closed
`kind = AGENT`).

The node builds its chat model from the engine-injected `caps['llm']` provider (a
`model_from_config`-shaped factory the engine owns; gated by the `needs_llm` trait) and hands
the mode the prompt rendered against its **declared inputs** (strict AGENT — the prompt sees
only `${name}` bound inputs, not the pool), the tools/controls, and the model. The engine
imports an LLM SDK by design — the old "engine imports no LLM SDK" rule was dropped;
a mode talks to langchain directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Optional, Union

from agent_composer.expr.template import render_template_record
from agent_composer.nodes.agent.controls import CONTROL_TOOLS
from agent_composer.nodes.agent.modes import MODES, AgentRunContext
from agent_composer.nodes.agent.modes.tool_calling import MAX_TOOL_ITERATIONS
from agent_composer.nodes.base import Node, NodeKind, NodeResult
from agent_composer.nodes.binding import ParamDecl

DEFAULT_MODE = "plain"


@dataclass(frozen=True)
class Fresh:
    """A fresh agent entry: render `prompt` from the node's declared inputs and start a
    new conversation. The author path (the loader builds this from the YAML `prompt:`)."""

    prompt: str = ""


@dataclass(frozen=True)
class Resume:
    """A continuation entry — the `ask_user` pause resumed. Rebuild the conversation from
    `memo` (a `messages_to_dict` blob), append the human answer as the `pending`
    `ToolMessage`, then run one more loop segment. Internal-only graph data minted by
    `clone_continuation_pair`; never authored. Keeping the frame here (not in the pool) is
    what lets the resume entry stay self-contained."""

    memo: list
    iterations: int
    pending: dict


# The closed entry sum. `Fresh` reads the author-declared inputs (rendered into the prompt);
# `Resume` reads a single `answer` param (the human_input forward-ref edge).
AgentEntry = Union[Fresh, Resume]


class AgentNode(Node):
    """
    An LLM-backed node, configured by `mode` + `tools` + `controls` and entered two ways.

    A `Fresh` entry renders `prompt` against the node's declared inputs and runs the selected
    `mode`'s loop; a `Resume` entry is the delimited continuation of an `ask_user` control pause
    (see the module docstring for the entry split). The node builds its chat model from the
    engine-injected `caps['llm']` provider (using `llm_config` as the carrier) and gives the mode
    the rendered prompt, tools, and controls.

    Args:
        node_id (`str`):
            The node's unique id.
        entry (`AgentEntry`, *optional*, defaults to `None`):
            The `Fresh | Resume` entry. `None` builds a `Fresh` from `prompt` (the author path).
        prompt (`str`, *optional*, defaults to `""`):
            The author prompt, used only when `entry` is `None`.
        tools (`list[str]`, *optional*, defaults to `None`):
            Ordinary tool ids the mode may call (resolved from `agent_composer.tools`).
        controls (`list[str]`, *optional*, defaults to `None`):
            Control-tool ids (e.g. `ask_user`) that drive an engine effect.
        llm_config (`dict`, *optional*, defaults to `None`):
            Plain-dict model selection (normalized to `LLMConfig` by `model_from_config`).
            On construction this is the node's OWN authored config; `resolve_llm_cascade`
            later overwrites `self.llm_config` with the EFFECTIVE config (own gap-filled by
            the flow/parent/CLI layers). The authored source is kept on `own_llm_config`.
        llm_inherit (`bool`, *optional*, defaults to `True`):
            Whole-node cascade opt-out. `False` (authored as `llm_config: {inherit: false}`)
            makes the effective config the node's own dict only — no flow/parent/CLI layers.
        mode (`str`, *optional*, defaults to `"plain"`):
            The loop method; must be a registered mode.
        retries (`int`, *optional*, defaults to `2`):
            Structured-generation self-correction cap — extra attempts after the first when
            the node declares a non-text `output:` and the provider deviates from the schema.
            Authorable as `retries:` on the node.
        title (`str`, *optional*, defaults to `None`):
            Display title.

    Raises:
        ValueError: If `mode` is unknown or a control id is not registered.
    """

    kind = NodeKind.AGENT
    is_spawner: ClassVar[bool] = True  # grows the graph: a control pause splices a Grow(Flow)
    grow_depth_delta: ClassVar[int] = 0  # a K-pause chain is ONE call: carry parent depth, no bound
    grow_restamps_self: ClassVar[bool] = True  # re-pause nests under this same spawner id (self-stamp)
    needs_llm: ClassVar[bool] = True  # LLM-backed: engine hands the model factory via caps["llm"]

    def __init__(
        self,
        node_id: str,
        *,
        entry: Optional[AgentEntry] = None,
        prompt: str = "",
        tools: Optional[list[str]] = None,
        controls: Optional[list[str]] = None,
        llm_config: Optional[dict[str, Any]] = None,  # plain dict, not LLMConfig
        llm_inherit: bool = True,
        mode: str = DEFAULT_MODE,
        retries: int = 2,
        title: Optional[str] = None,
    ) -> None:
        super().__init__(node_id, title=title)
        # The entry sum (Fresh | Resume). Back-compat: a bare `prompt=` (the loader/author
        # path) is a Fresh entry; an explicit `entry=Resume(...)` is the continuation path.
        self.entry: AgentEntry = entry if entry is not None else Fresh(prompt=prompt or "")
        self.tools = tools or []
        self.controls = controls or []
        # own_llm_config: the authored dict, the immutable source re-resolved each cascade
        # pass. llm_config: the EFFECTIVE config — defaults to own until resolve_llm_cascade
        # bakes the gap-filled result. llm_inherit: the whole-node opt-out flag.
        self.own_llm_config: dict[str, Any] = dict(llm_config) if llm_config else {}
        self.llm_config = llm_config
        self.llm_inherit = llm_inherit
        self.mode = mode or DEFAULT_MODE
        # retries: the structured-generation self-correction cap (extra attempts after the
        # first) used when the node declares a non-text `output:`; authorable as `retries:`.
        self.retries = retries
        if self.mode not in MODES:
            raise ValueError(
                f"AGENT node {node_id!r}: unknown mode {self.mode!r}; known: {sorted(MODES)}"
            )
        unknown = [c for c in self.controls if c not in CONTROL_TOOLS]
        if unknown:
            raise ValueError(
                f"AGENT node {node_id!r}: unknown control tool(s) {unknown}; "
                f"known: {sorted(CONTROL_TOOLS)}"
            )
        # A Resume entry binds a single `answer` param — the human_input forward-ref edge
        # `${<hi>.output}` the engine wires onto it. A Fresh entry's params are the
        # author-declared inputs, stamped on the node by the compiler.
        if isinstance(self.entry, Resume):
            self.params = [ParamDecl(name="answer")]

    @property
    def prompt(self) -> str:
        """The authored prompt (read accessor over the entry sum): a `Fresh` agent's prompt,
        or `""` for a `Resume` continuation (which has no authored prompt — its conversation is
        the replayed `memo`). The validator + parser read this; `entry` stays the source of truth."""
        return self.entry.prompt if isinstance(self.entry, Fresh) else ""

    def _build_model(self, llm) -> Any:
        """Build this node's chat model via the injected `llm` provider (caps['llm']),
        a `model_from_config`-shaped callable. `llm_config` (a plain dict|LLMConfig) is the
        carrier the provider normalizes."""
        return llm(self.llm_config if self.llm_config is not None else {})

    def _max_tool_iterations(self) -> int:
        """The per-node tool-iteration cap: `env: {max_tool_iterations: N}` when authored,
        else the module default (-1, no cap). A valid value is a positive int (bound the loop
        at N turns) or -1 (no cap — loop until a final answer); anything else is a loud
        NodeFailed. `bool` is rejected explicitly (it is an `int` subclass but never valid);
        0 and other negatives are rejected too."""
        raw = self.env.get("max_tool_iterations", MAX_TOOL_ITERATIONS)
        if isinstance(raw, bool) or not isinstance(raw, int) or (raw < 1 and raw != -1):
            raise ValueError(
                f"AGENT node {self.id!r}: env max_tool_iterations must be a positive int "
                f"or -1 (no cap), got {raw!r}"
            )
        return raw

    def _ctx(self, prompt: str, llm) -> AgentRunContext:
        """Build the per-run mode context shared by both entry arms. `llm_config` carries
        forward (config-as-data), so a resumed continuation rebuilds the same model. `llm` is
        the injected caps['llm'] provider used to construct the chat model."""
        return AgentRunContext(
            node_id=self.id,
            prompt=prompt,
            tools=list(self.tools),
            controls=list(self.controls),
            model=self._build_model(llm),
            llm_config=self.llm_config,
            output_type=self.output_type,
            retries=self.retries,
            max_tool_iterations=self._max_tool_iterations(),
            post_asserts=list(self.post_asserts),
            assert_texts=dict(self.assert_texts),
        )

    def _run_impl(self, inputs: dict, **caps: Any) -> NodeResult:
        # caps['llm'] is the engine-owned model factory (the needs_llm gate). Fall back to the
        # lazy package thunk for a direct node._run_impl(...) with no caps (direct-driver tests). The
        # import is lazy to avoid an import cycle (runtime.eval_node imports nodes.base).
        llm = caps.get("llm")
        if llm is None:
            from agent_composer.runtime.eval_node import _default_llm

            llm = _default_llm
        # Dispatch on the closed `entry` sum. A mid-loop control pause lowers to a
        # continuation `Grow`; the engine mints the Resume half and the re-entry
        # frame rides as graph data. `llm_config` carries forward.
        if isinstance(self.entry, Resume):
            # Delimited continuation: replay `memo`, append the human answer as the pending
            # ToolMessage, then run ONE more loop segment. `agent_step` enters with
            # pending=None (the answer is already appended), so the model is invoked once on
            # the resumed messages — the memo is replayed as messages, not re-invoked.
            from langchain_core.messages import ToolMessage, messages_from_dict

            from agent_composer.nodes.agent.modes.tool_calling import agent_step

            messages = messages_from_dict(self.entry.memo)
            messages.append(
                ToolMessage(
                    content=str(inputs["answer"]), tool_call_id=self.entry.pending["call_id"]
                )
            )
            return agent_step(messages, None, self.entry.iterations, self._ctx(prompt="", llm=llm))
        # Fresh: strict AGENT — the prompt interpolates only this node's
        # declared inputs, rendered against the bound record. The selected mode builds the
        # conversation and runs the loop.
        ctx = self._ctx(prompt=render_template_record(self.entry.prompt, inputs), llm=llm)
        return MODES[self.mode](ctx)

    def replay_grow(self, seed: Any):
        """Durable-replay inverse of the agent's pause `Grow`: rebuild ONE continuation segment's
        subgraph from the persisted `{hi_desc, resume_desc}` seed via the SAME
        `agent_segment_subgraph` builder — no LLM turn re-run. The seed is re-validated through the
        `AgentSegment` schema so the re-entry frame keeps its round-trip validation. For a K-pause
        chain each segment's resume terminal becomes the next segment's spawner, so `callsite`/
        `output_type`/`post_asserts` come off THIS node (the previous segment's terminal on segment 2+),
        so a resumed run re-lowers the same node-level `${output}` post-asserts onto the resume terminal."""
        from agent_composer.compile.expand import agent_segment_subgraph
        from agent_composer.suspension.expansions import AgentSegment

        seg = AgentSegment.model_validate(seed)
        return agent_segment_subgraph(
            [seg.hi_desc, seg.resume_desc],
            callsite=self.id,
            output_type=self.output_type,
            retries=self.retries,
            max_tool_iterations=self._max_tool_iterations(),
            post_asserts=list(self.post_asserts),
            assert_texts=dict(self.assert_texts),
        )
