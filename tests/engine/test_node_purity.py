"""Structural purity: NO Node attribute holds a `${...}` SOURCE.

The headline payoff of the wiring relocation: `CompiledFlow.wiring` is the
SOLE owner of input *bindable* sources. A node carries only its signature (`params`) + its logic
— an AGENT prompt / CASE `when` reference bare `${param}` names, never a pool ref. This walks
every representative seed's nodes (recursing into baked `call`/`map` child flows) and asserts no
node-held string is a pool-ref source (`${outputs|inputs|system|item ...}`); a stray leaf source
or a `WaitNode.until` left on a node would trip it. Sources live only on `flow.wiring` (and the
flow-level `outputs:`), never on a `Node`.

No exemptions for *bindable sources*: the MAP re-split retired the last one (the old
`CallNode.over` discriminator). `MapNode` discriminates by KIND, not an `over` source string.
"""

import re
from pathlib import Path

import pytest

from agent_composer.compile.model import CompiledFlow
from agent_composer.compose import load_flow
from agent_composer.nodes.contracts import Binding

_SEEDS = Path(__file__).resolve().parents[2] / "tests" / "seeds"

# A POOL ref (a source): a `${...}` span containing a pool head ANYWHERE — `outputs.`/`inputs.`/
# `system.` (always dotted) or a bare/dotted `item`. Matches a head buried in a coalesce operand
# (`${default | y.output}`) too, not only the first. A bare `${param}` (prompt / when) has no head.
# Pool refs also include the singular `input.` and node-first `<id>.output` shapes.
_POOL_REF = re.compile(r"\$\{[^}]*?(?:\b(?:outputs|inputs|input|system)\.|\bitem\b|\b[A-Za-z_][A-Za-z0-9_#/]*\.output\b)")

_SEEDS_TO_SCAN = [
    "00-hello-agent.yaml",
    "01-structured-agent.yaml",
    "02-case.yaml",
    "04-call.yaml",
    "05-call-map.yaml",
    "06-case-on.yaml",
    "07-model-rating.yaml",
    "08-tool-news.yaml",
    "14-agent-tools.yaml",
    "17-effects-human-wait.yaml",  # the ONLY seed with an absolute WAIT + HUMAN_INPUT
    "18-research-pipeline.yaml",
]


def _strings(obj, seen):
    """Every string reachable from `obj` (recursing dicts / lists / dataclass-ish __dict__ /
    baked child `CompiledFlow.nodes`), id-cycle-guarded. A `CompiledFlow`'s `wiring`/`outputs`
    are NOT walked — those are the legitimate homes for sources."""
    if isinstance(obj, str):
        yield obj
        return
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, CompiledFlow):
        for n in obj.nodes.values():
            yield from _strings(n, seen)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            yield from _strings(v, seen)
    elif hasattr(obj, "__dict__"):
        for v in vars(obj).values():
            yield from _strings(v, seen)


@pytest.mark.parametrize("seed", _SEEDS_TO_SCAN)
def test_no_node_attribute_holds_a_pool_ref_source(seed):
    loaded = load_flow((_SEEDS / seed).read_text(), search_paths=[_SEEDS])
    for nid, node in loaded.compiled.nodes.items():
        for s in _strings(node, set()):
            assert not _POOL_REF.search(s), (
                f"{seed} node {nid!r} holds a pool-ref source {s!r} on a Node attribute — "
                f"input sources must live only on flow.wiring"
            )


def test_flow_wiring_is_where_the_sources_are():
    # Positive counterpart: the sources DID land somewhere — flow.wiring carries the pool refs.
    loaded = load_flow((_SEEDS / "01-structured-agent.yaml").read_text(), search_paths=[_SEEDS])
    all_wiring_sources = [
        src for node_w in loaded.compiled.wiring.values() for src in node_w.values()
    ]
    assert all(isinstance(source, Binding) for source in all_wiring_sources)
    assert any(
        isinstance(source.source, str) and _POOL_REF.search(source.source)
        for source in all_wiring_sources
    )


def test_purity_guard_actually_bites():
    # The lock-in is only as good as its walk: prove a source RE-STAMPED onto a node attribute
    # (a simulated regression — exactly the WaitNode.until / leaf-source class that was removed) is
    # caught by _strings + _POOL_REF, so the green scan above is meaningful.
    loaded = load_flow((_SEEDS / "17-effects-human-wait.yaml").read_text(), search_paths=[_SEEDS])
    settle = loaded.compiled.nodes["settle"]  # the absolute-deadline WAIT
    assert not any(_POOL_REF.search(s) for s in _strings(settle, set()))  # clean before
    settle.until = "${input.settle_at}"  # regression: a source put back on the WAIT node
    assert any(_POOL_REF.search(s) for s in _strings(settle, set()))  # ...is detected
