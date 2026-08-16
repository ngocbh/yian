"""Graph-stage: `case` then:/else: inline-call branch targets -> synth call nodes.

A `case` branch may route to a FRESH owned call instead of a placed node id:
`then: call(take, stance="pro")` (and `else:` symmetric). It desugars to a synth
`__call_<n>` node (the same machinery as an inline-binding call); the then:/else:
target is rewritten to that synth id (which is then both the CaseNode handle and
the control-edge target, exactly like a plain `then: id`). Accepted grammar: a
whole-value `call(...)` directive — a route target resolves to exactly one node id, so
a coalesce / embedded text / dotted call (or any `${...}` span) is a located LoadError.
Sound because the case-route veto skip-floods the non-chosen synth branch even when its
call args carry data edges.

Runs in `_assemble` AFTER `desugar_inline_calls` (sharing its `next_id` minter — else
the ids collide) and BEFORE `expand_case_outputs`, so the synth call's args (which may
read `${<case>.output}`) are expanded by that pass and `${<case>.output}` over THIS
case sees the rewritten (synth-id) targets.

Located parse metadata: a synth branch-target call node locates at the `then:`/`else:` it was
lifted from — the case descriptor's positional `then_ranges[N]` for a `then:`, the `("field",
"else")` range for the `else:` — with an empty `field_ranges`, falling back to the case node's
own range when a span is unavailable.

Imports flow down/peer only: `compose.parse.file` (CaseDescriptor), `compose.desugar.inline_calls`
(the shared `call(...)` recognizer + synth-node factory), `compose.helpers.errors`. Only the loader
imports this back.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Optional

from agent_composer.events import SourceRange
from agent_composer.compose.helpers.errors import LoadDiagnostics
from agent_composer.compose.parse.file import CaseDescriptor
from agent_composer.compose.desugar.inline_calls import (
    _to_call_descriptor,
    desugar_call_directives,
)


def desugar_case_call_targets(
    descriptors: dict,
    mint: Callable[[], str],
    *,
    diagnostics: LoadDiagnostics | None = None,
) -> dict:
    """Rewrite each `case` then:/else: that is an inline call — a whole-value
    `call(<flow-id>, kw=...)` directive — to a synth call node id (minting its
    `CallDescriptor`, `over=None`); return the new descriptor map (augmented with the synth
    nodes). A plain node id (no `${`, not a `call(...)` directive) passes through unchanged.

    Each synth branch-target node locates at the `then:`/`else:` it was lifted from (its own
    source line), falling back to the case node's range when that span is unavailable.
    """
    diagnostics = diagnostics or LoadDiagnostics()
    synth: dict = {}
    new_descriptors: dict = {}
    for nid, desc in descriptors.items():
        if not isinstance(desc, CaseDescriptor):
            new_descriptors[nid] = desc
            continue
        rng = desc.range
        new_cases = [
            {
                **c,
                "then": _lift_case_target(
                    c.get("then"),
                    mint,
                    synth,
                    host=f"case node {nid!r} then:",
                    range=_nth(desc.then_ranges, ci) or rng,
                    diagnostics=diagnostics,
                ),
            }
            for ci, c in enumerate(desc.cases)
        ]
        else_rng = desc.field_ranges.get(("field", "else")) or rng
        new_else = (
            _lift_case_target(
                desc.else_,
                mint,
                synth,
                host=f"case node {nid!r} else:",
                range=else_rng,
                diagnostics=diagnostics,
            )
            if desc.else_ is not None
            else None
        )
        new_descriptors[nid] = replace(desc, cases=new_cases, else_=new_else)
    new_descriptors.update(synth)
    return new_descriptors


def _nth(seq: list, i: int) -> Optional[SourceRange]:
    """The i-th positional range, or None when absent (a spanless flow, or a short list)."""
    return seq[i] if 0 <= i < len(seq) else None


def _lift_case_target(
    target: Any,
    mint: Callable[[], str],
    synth: dict,
    *,
    host: str,
    range: Optional[SourceRange] = None,
    diagnostics: LoadDiagnostics | None = None,
) -> Any:
    """A then:/else: target rewritten to a single node id. Accepts TWO forms and
    rejects everything else with a located `LoadError`:

    - a whole-value `call(<flow-id>, kw=...)` directive (the compile-time form) -> lift
      its synth call node(s) into `synth`, return the OUTER call's synth id;
    - a plain node id (no `${`, not a `call(...)` directive) -> passed through unchanged.

    Anything else — a coalesce / embedded text / dotted call, or any `${...}` span — is a
    `LoadError`: a route target must resolve to exactly one node id."""
    diagnostics = diagnostics or LoadDiagnostics()
    if not isinstance(target, str):
        return target  # None (handled by the caller) or a native literal
    stripped = target.strip()
    # A whole-value `call(...)` directive: recognized BEFORE the plain-id passthrough,
    # since a directive carries no `${` and would otherwise look like a bare node id.
    new_value, calls = desugar_call_directives(target, mint)
    if calls:  # inner-first: the OUTER call is calls[-1] and names the branch target
        outer = calls[-1]
        if new_value.strip() == f"${{{outer.id}.output}}":
            for c in calls:
                synth[c.id] = _to_call_descriptor(
                    c,
                    host=host,
                    range=range,
                    diagnostics=diagnostics,
                )
            return outer.id
    if "${" not in target:
        # Not a directive and no span: a plain node id, UNLESS it is a rejected `call(...)`
        # attempt (e.g. a coalesce of directives) that the recognizer declined — those must
        # not slip through as a node id.
        if stripped.startswith("call("):
            raise diagnostics.error(
                f"{host}: a branch target must be a node id or a single call "
                f"(got {target!r}) — not a coalesce / embedded text / dotted call",
                range=range,
            )
        return target
    # A `${...}` branch target is the retired inline `${flow(args)}` form: a route target
    # is a node id or a whole-value call(...) directive (handled above) — nothing else.
    raise diagnostics.error(
        f"{host}: a branch target must be a node id or a call(...) directive "
        f"(got {target!r}) — an inline ${{flow(args)}} target is no longer supported",
        range=range,
    )
