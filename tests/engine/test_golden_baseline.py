"""Golden baselines (a *characterization* test).

Pins the current edge-id sets of representative flows so the wiring relocation (wiring moves
off the node onto `CompiledFlow.wiring`, edges become its derived projection) can prove
**byte-identical edge ids / no drift** against a recorded baseline rather than a remembered
one.

This is GREEN on first run by construction. If a later change legitimately changes the
topology, update the literals here in the same commit and explain why.

Coverage across the three seeds: data edges, CASE/case control edges + skip-flood targets,
synthesized root (`__start__->`) + terminal (`->__end__#0`) edges, the ordering separator (`~>`),
and a mapped `call` (MAP) node. CALL/MAP child-message byte identity is guarded by the existing
lock-in tests (test_call_run / test_map / test_call_map); the format constants are recorded here for
the lock-in that re-homes the MAP `over`-not-a-list message into the engine bind seam.
"""

from pathlib import Path

import pytest

from yian.compose import load_flow

_SEEDS = Path(__file__).resolve().parents[2] / "tests" / "seeds"


def _text(name: str) -> str:
    return (_SEEDS / name).read_text()


def _edge_ids(name: str, **kw) -> list[str]:
    return sorted(e.id for e in load_flow(_text(name), **kw).compiled.edges)


# --- recorded edge-id baselines (sorted) -------------------------------------------------- #
# An `${input.X}` reader gets a `START_ID->reader#i` DATA edge (input_group = the param/over/until
# name) instead of a bare `__start__->reader` root edge — the input-producer edges the loader
# mints. A case branch target that ALSO reads `${input.X}` therefore carries BOTH its control edge
# and the START_ID data edge (the control veto still skip-floods it). The END_ID producer edges
# carry the output name.
GOLDEN_EDGE_IDS: dict[str, list[str]] = {
    # case: data (score->gate), control (gate->positive/cautious), START_ID input-producer edges, terminals
    "02-case.yaml": [
        "__start__->cautious#0",
        "__start__->positive#0",
        "__start__->score#0",
        "cautious->__end__#0",
        "gate->cautious#0",
        "gate->positive#0",
        "positive->__end__#0",
        "score->gate#0",
    ],
    # mapped call (MAP) + fan-in; research_each reads ${input.topics} (over) + ${input.as_of} (opt)
    "05-call-map.yaml": [
        "__start__->research_each#0",
        "__start__->research_each#1",
        "compare->__end__#0",
        "research_each->__end__#0",
        "research_each->compare#0",
    ],
    # human_input + absolute WAIT: CASE gate control edges + the `~>` ordering separator
    "17-effects-human-wait.yaml": [
        "__start__->propose#0",
        "__start__->settle#0",
        "abort->__end__#0",
        "approve->gate#0",
        "confirm->__end__#0",
        "gate->abort#0",
        "gate->settle#0",
        "propose->abort#0",
        "propose->approve#0",
        "propose->confirm#0",
        "settle~>confirm#0",
    ],
}

# seed 05 resolves its external `call` via the seeds dir
GOLDEN_KW: dict[str, dict] = {"05-call-map.yaml": {"search_paths": [_SEEDS]}}


@pytest.mark.parametrize("seed", sorted(GOLDEN_EDGE_IDS))
def test_edge_ids_match_baseline(seed: str):
    assert _edge_ids(seed, **GOLDEN_KW.get(seed, {})) == GOLDEN_EDGE_IDS[seed]


def _edge_tuples(name: str, **kw) -> list[tuple]:
    """Ordered `(id, input_group, optional)` per edge in `CompiledFlow.edges` EMISSION order —
    the full projection, not just the id set (over-then-inputs / data-then-control order +
    input_group per sink + the co-skip `optional` stance)."""
    return [(e.id, e.input_group, e.optional) for e in load_flow(_text(name), **kw).compiled.edges]


# --- recorded ORDERED edge projection ----------------------------------------------------- #
# Stronger than GOLDEN_EDGE_IDS (sorted ids): catches an emission-ORDER / input_group / optional
# drift that preserves the id SET. The wiring relocation must keep this byte-identical.
# The synthesized END_ID producer edges carry `input_group=<output_name>` (the recoverable name)
# and are emitted BEFORE the START_ID root edges (synthesize_boundary_graph returns END_ID-edges
# then START_ID-edges, in output-declaration order). The id SET is unchanged
# (test_edge_ids_match_baseline still passes); only the input_group + emission order moved.
GOLDEN_EDGE_TUPLES: dict[str, list[tuple]] = {
    "02-case.yaml": [
        ("__start__->score#0", "topic", False),
        ("__start__->positive#0", "topic", False),
        ("__start__->cautious#0", "topic", False),
        ("score->gate#0", "__r0", False),
        ("gate->positive#0", None, False),
        ("gate->cautious#0", None, False),
        ("positive->__end__#0", "result", False),
        ("cautious->__end__#0", "result", False),
    ],
    "05-call-map.yaml": [
        ("__start__->research_each#0", "over", False),
        ("__start__->research_each#1", "as_of", True),
        ("research_each->compare#0", "briefs", False),
        ("compare->__end__#0", "report", False),
        ("research_each->__end__#0", "briefs", False),
    ],
    "17-effects-human-wait.yaml": [
        ("__start__->propose#0", "topic", False),
        ("propose->approve#0", "action", False),
        ("__start__->settle#0", "until", False),
        ("propose->confirm#0", "action", False),
        ("propose->abort#0", "action", False),
        ("approve->gate#0", "__on", False),
        ("gate->settle#0", None, False),
        ("gate->abort#0", None, False),
        ("settle~>confirm#0", None, False),
        ("confirm->__end__#0", "result", False),
        ("abort->__end__#0", "result", False),
    ],
}


MAP_OVER_NOT_LIST_FMT = "MAP node {id!r}: `over` ({src}) did not resolve to a list"


@pytest.mark.parametrize("seed", sorted(GOLDEN_EDGE_TUPLES))
def test_edge_tuples_match_baseline(seed: str):
    assert _edge_tuples(seed, **GOLDEN_KW.get(seed, {})) == GOLDEN_EDGE_TUPLES[seed]
