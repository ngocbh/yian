"""GrowRecord — the uniform durable record of one runtime `Grow`.

The kill-recovery half of durable suspension. WRITE half: a spawner node returning `Grow`
grows the live graph at runtime; `FlowEngine._apply_grow` captures each grow as ONE
`GrowRecord` and attaches it to an ordered ledger on `FlowEngine.expansions` (a nested grow
rides under its enclosing record's flat `children`), which `snapshot()` persists. READ half
(`FlowEngine._replay_expansions`): on restore the engine replays the record tree top-down,
passing each spawner and canonical seed to the injected growth builder and re-splicing the result
with effects suppressed, so a run paused mid-expansion resumes in a fresh process.

ONE uniform record per grow — no per-kind variants:

- CALL   — `seed` is the call-arg record dict; one record per call.
- MAP    — `seed` is the per-element records list; one record per map.
- AGENT  — `seed` is one pause's `{hi_desc, resume_desc}` (validated as an `AgentSegment` on
           read); a K-pause agent is a NESTING CHAIN of K records — segment i+1 rides under
           segment i's `children`, keyed by the resume-terminal id (`{spawner}/__resume#…`).
- LOOP   — `seed` is the LIVE iteration's `(record, index)`; exactly ONE record per loop
           spawner (superseded iterations' records are dropped as their overlay is pruned).

Element/iteration PLACEMENT is not stored — it is re-derived on replay from each child's
namespaced spawner_id (`{spawner}#{i}/…`), so the flat `children` list round-trips faithfully
regardless of order.
"""

from typing import Any

from pydantic import BaseModel, Field

from agent_composer.nodes.agent_segment import AgentSegment


class GrowRecord(BaseModel):
    """Uniform durable record of ONE `Grow` (see the module docstring for the per-kind `seed`
    shapes). `seed` is the pure builder input for that single grow — kind-shaped but opaque to
    the engine. `children` are grows spliced UNDER this one (nested spawners), flat + ordered;
    element/iteration placement is re-derived on replay from each child's `{spawner}#{i}/…`
    spawner_id prefix (never from the list order)."""

    spawner_id: str
    seed: Any = None
    children: list["GrowRecord"] = Field(default_factory=list)


# Rebuild the forward ref (`children: list["GrowRecord"]`).
GrowRecord.model_rebuild()
