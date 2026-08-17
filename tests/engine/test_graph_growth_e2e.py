"""End-to-end lock-in — main engine complete.

Companion to the node-purity test. Where node purity locks the node->pure `run(inputs)`
contract, this locks the engine results end-to-end:

- **scratch is eliminated** structurally: no `pool.scratch`/`NodeScratch`, no
  per-kind scratch cap module, no `Node` attribute or string referencing scratch.
- **the live graph grows** under a CALL/MAP run: a `Grow` deep-namespaces the child
  into the live `CompiledFlow` and `len(flow.nodes)` increases — driven via `FlowEngine.run()`
  directly so the test can inspect the engine-owned projection.
- the **START_ID..END_ID splice** is exercised: each call splices the namespaced child
  START_ID..END_ID, MAP fans into one END_ID(list-mode), and every synthesized ref is a **bare**
  `${<id>.output}` — no spurious `.output` suffix.
- **nested blocking input reaches terminal**: a HUMAN_INPUT inside a called child passes
  its namespaced leaf id to the trusted host and returns the answer as Output.
- the goldens still hold (graph-growth must not perturb the static edge ids).
- authoritative expansion is driven through the prepared-state engine surface.
- the engine suite carries no expected failures.

LLM-terminating seeds (00/04/05/14/18) are asserted LOAD+structure only; run assertions use
the CODE-child pattern (mirroring test_golden_run_results.py), plus seed 25 for an LLM-free
nested blocking interaction.
"""

import importlib
import re
from pathlib import Path

import pytest

from agent_composer.compile.expand import build_grow
from agent_composer.compose import LoadError, load_flow
from agent_composer.compose.helpers.errors import LoadDiagnostics
from agent_composer.run import run_flow
from agent_composer.nodes.base import NodeKind
from agent_composer.typesys.pool import VariablePool
from agent_composer.events import RunSucceeded
from agent_composer.tools import install_ask_user
from tests.engine._graph_builder import _engine

_SEEDS = Path(__file__).resolve().parents[2] / "tests" / "seeds"


# --- CODE-child CALL/MAP fixtures (Ollama-free; mirror test_golden_run_results.py) ------------ #

_CHILD = """
id: child-one
name: child_one
input:
  topic: str
nodes:
  emit:
    kind: code
    input:
      topic: ${input.topic}
    output:
      report: str
      n: int
    code: tests.engine._compose_codefns:make_report
output:
  report: ${emit.output.report}
  n: ${emit.output.n}
"""

_MAP = """
id: map-parent
name: map_parent
input:
  topics: list[str]
uses:
  child-one: child-one
nodes:
  research_each:
    kind: map
    call: child-one
    over: ${input.topics}
    input:
      topic: ${item}
output: ${research_each.output}
"""


def _resolver(**children):
    loaded = {fid: load_flow(text) for fid, text in children.items()}

    def resolve(flow_id, version=None):
        try:
            return loaded[flow_id]
        except KeyError as exc:
            raise LoadDiagnostics().error(f"unknown child {flow_id!r}") from exc

    return resolve


def _run_map(topics):
    """Drive MAP directly and return its engine/events for live-graph inspection."""
    parent = load_flow(_MAP, child_resolver=_resolver(**{"child-one": _CHILD}))
    eng = _engine(
        parent.compiled,
        run_inputs={"topics": topics},
        grow_builder=build_grow,
    )
    events = list(eng.run())
    return eng, events


# --- scratch eliminated structurally ------------------------ #


def test_scratch_is_eliminated_structurally():
    pool = VariablePool()
    assert not hasattr(pool, "scratch")
    assert not hasattr(pool, "scratch_set") and not hasattr(pool, "scratch_get")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("agent_composer.nodes.scratch_cap")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("agent_composer.nodes.agent.scratch")


def test_no_node_attr_or_method_reads_scratch():
    loaded = load_flow((_SEEDS / "04-call.yaml").read_text(), search_paths=[_SEEDS])
    for node in loaded.compiled.nodes.values():
        for name, val in vars(node).items():
            assert "scratch" not in name.lower()
            assert not (isinstance(val, str) and "scratch" in val.lower())


# --- the live graph GROWS under a MAP run -------------------------------- #


def test_live_graph_grows_under_a_map_run():
    parent = load_flow(_MAP, child_resolver=_resolver(**{"child-one": _CHILD}))
    eng = _engine(
        parent.compiled,
        run_inputs={"topics": ["ACME", "BETA"]},
        grow_builder=build_grow,
    )
    before = len(eng.state.flow.nodes)
    events = list(eng.run())
    assert isinstance(events[-1], RunSucceeded)
    # the Grow expansion + add_subgraph grew the LIVE graph (child cloned per element)
    assert len(eng.state.flow.nodes) > before


def test_start_end_splice_is_exercised():
    eng, events = _run_map(["ACME", "BETA"])
    assert isinstance(events[-1], RunSucceeded)
    ids = list(eng.state.flow.nodes)
    # each element splices the namespaced child START_ID..END_ID; MAP also synthesizes ONE
    # `research_each/__start__` fan-out start + ONE END_ID(list-mode) aggregator under the spawner
    # namespace.
    starts = sorted(i for i in ids if i.endswith("/__start__"))
    assert starts == ["research_each#0/__start__", "research_each#1/__start__",
                      "research_each/__start__"]
    map_start_id = "research_each/__start__"                     # ns(spawner, START_ID)
    # the synthetic start fans out to each element START via an ordering edge.
    fanout = {
        e.to for e in eng.state.flow.edges if e.from_ == map_start_id and e.ordering
    }
    assert fanout == {"research_each#0/__start__", "research_each#1/__start__"}
    map_end_id = "research_each/__end__"                         # ns(spawner, END_ID)
    assert eng.state.flow.nodes[map_end_id].kind == NodeKind.END
    into_end = {
        (e.from_, e.input_group)
        for e in eng.state.flow.edges
        if e.to == map_end_id
    }
    assert into_end == {("research_each#0/__end__", "e0"),
                        ("research_each#1/__end__", "e1")}
    # the END_ID(list-mode) aggregator joins in OVER ORDER (element index)
    assert events[-1].output == [
        {"report": "report for ACME", "n": 4},
        {"report": "report for BETA", "n": 4},
    ]


_SPURIOUS_OUTPUT = re.compile(r"\}\.output\b|outputs\.[^}]*\.output\b")


def test_synthesized_refs_are_bare_outputs():
    # every synthesized ref is a BARE ${<id>.output} — no trailing `.output` suffix.
    # (Legitimate dotted FIELD access like `${<id>/emit.output.report}` is NOT a `.output`
    # suffix and is allowed; the regex only catches a spurious `.output`.)
    eng, _ = _run_map(["ACME", "BETA"])
    for nid, node_w in eng.state.flow.wiring.items():
        if "/" not in nid and "#" not in nid:
            continue  # static node — not minted by expansion
        for src in node_w.values():
            if isinstance(src, str):
                assert not _SPURIOUS_OUTPUT.search(src), (
                    f"synthesized wiring {nid!r} carries a `.output` suffix: {src!r}"
                )


# --- nested blocking interaction reaches terminal ----------------------- #


def test_nested_human_input_reaches_terminal():
    # Seed 25: a HUMAN_INPUT inside a called child invokes the trusted host with
    # its live namespaced id and completes in one run.
    loaded = load_flow((_SEEDS / "25-nested-suspension.yaml").read_text(), search_paths=[_SEEDS])
    requests = []
    with install_ask_user(lambda **request: requests.append(request) or "approve"):
        result = run_flow(loaded, {"action": "order 10 units"})

    assert requests[0]["node_id"] == "gate/approve"
    assert result.status == "succeeded"
    assert result.output == "approve"  # confirm_action(rec={answer:"approve"})


# --- goldens hold (graph-growth must not perturb static edge ids) ----- #


def test_f0_goldens_hold():
    from tests.engine.test_golden_run_results import (
        test_golden_map_run_result,
        test_golden_call_run_result,
        test_golden_static_edge_ids_case_and_call_and_map,
    )

    # Reassert the CALL/MAP run results and static edge-id sets (graph-growth
    # adds namespaced nodes; the STATIC topology + run results are unperturbed).
    test_golden_call_run_result()
    test_golden_map_run_result()
    test_golden_static_edge_ids_case_and_call_and_map()


# --- no engine xfails remain ---------------------------------------------- #


def test_no_remaining_engine_xfails():
    # Keep the dedicated engine collection free of tests explicitly selected by an xfail marker.
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/engine", "--collect-only", "-q", "-m", "xfail"],
        cwd=str(Path(__file__).resolve().parents[2]),
        capture_output=True,
        text=True,
    )
    lines = [ln.strip() for ln in out.stdout.splitlines() if "::" in ln]
    assert lines == [], out.stdout


# --- representative seeds: LOAD (+ structure) end-to-end ------------------------------- #

_REPRESENTATIVE = [
    "00-hello-agent.yaml",
    "04-call.yaml",
    "05-call-map.yaml",
    "14-agent-tools.yaml",
    "17-effects-human-wait.yaml",
    "18-research-pipeline.yaml",
    "25-nested-suspension.yaml",
]


@pytest.mark.parametrize("seed", _REPRESENTATIVE)
def test_representative_seed_loads_with_structure(seed):
    # LLM-terminating seeds (00/04/05/14/18) are asserted LOAD + structure only; the
    # run assertions are covered by the CODE-child MAP run + seed 25's nested
    # blocking interaction above.
    loaded = load_flow((_SEEDS / seed).read_text(), search_paths=[_SEEDS])
    assert loaded.compiled.nodes  # loaded + has a body
    assert loaded.compiled.start_id in loaded.compiled.nodes  # a runnable entry
