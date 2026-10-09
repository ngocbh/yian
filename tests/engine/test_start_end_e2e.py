"""End-to-end lock-in: every flow is `START_ID -> body -> END_ID`; the retired constructs
(OUTPUT_RESOLVER/COLLECTOR/terminal_output/__start__-as-sentinel/RefNode/NodeKind.REF) are gone
(MAP re-split into its own `kind: map`/`MapNode`); `${input.X}` resolves via START_ID;
the representative seeds run/load author-identically.

Companion to the node-purity test and `test_graph_growth_e2e.py` (graph growth/splice). This is a
characterization/structural test, GREEN on first run. LLM-terminating seeds (00/03/04/05/14/18)
are asserted LOAD + structure only; run-result behavior reuses the CODE-child goldens
(test_golden_run_results.py).
"""

import ast
from pathlib import Path

import pytest

from yian.compile.model import END_ID, START_ID
from yian.compose import load_flow
from yian.run import run_flow
from yian.nodes.base import NodeKind

_REPO = Path(__file__).resolve().parents[2]
_SEEDS = _REPO / "tests" / "seeds"

_REPRESENTATIVE = [
    "00-hello-agent.yaml",
    "03-research-one.yaml",
    "04-call.yaml",
    "05-call-map.yaml",
    "14-agent-tools.yaml",
    "17-effects-human-wait.yaml",
    "18-research-pipeline.yaml",
]

# Retired runtime/IR symbols that no module may reference as a LIVE identifier (history in
# comments/docstrings is fine — those are excluded by the AST walk, which never sees them).
# `MapNode` is NOT retired — it is the re-split MAP driver (`kind: map`).
_RETIRED_NAMES = {"RefNode", "OutputResolverNode", "CollectorNode"}
_RETIRED_KIND_ATTRS = {"REF", "OUTPUT_RESOLVER", "COLLECTOR"}
_RETIRED_FUNCS = {"terminal_output", "_emit_terminal", "_terminal_coskip"}


# --- (1) exactly one START_ID + one END_ID per representative flow --------------------------- #


@pytest.mark.parametrize("seed", _REPRESENTATIVE)
def test_exactly_one_start_and_one_end_per_flow(seed):
    compiled = load_flow((_SEEDS / seed).read_text(), search_paths=[_SEEDS]).compiled
    starts = [nid for nid, n in compiled.nodes.items() if n.kind is NodeKind.START]
    ends = [nid for nid, n in compiled.nodes.items() if n.kind is NodeKind.END]
    assert starts == [START_ID], f"{seed}: expected one START_ID ({START_ID}), got {starts}"
    assert ends == [END_ID], f"{seed}: expected one END_ID ({END_ID}), got {ends}"
    assert compiled.start_id == START_ID and compiled.end_id == END_ID


# --- (2) no retired constructs anywhere (structural, over the compiled IR + the API) ---- #


def test_nodekind_has_no_retired_members():
    for attr in _RETIRED_KIND_ATTRS:
        assert not hasattr(NodeKind, attr), f"NodeKind still carries {attr}"
    # CALL and MAP are the two composition kinds.
    assert NodeKind.CALL == "call" and NodeKind.MAP == "map"


@pytest.mark.parametrize("seed", _REPRESENTATIVE)
def test_no_retired_node_kinds_in_compiled_ir(seed):
    compiled = load_flow((_SEEDS / seed).read_text(), search_paths=[_SEEDS]).compiled
    kinds = {n.kind for n in compiled.nodes.values()}
    # Every compiled kind belongs to the closed current enum.
    assert all(k in NodeKind for k in kinds)
    # __start__/__end__ appear ONLY as the real boundary NODE ids — never a stray sentinel
    # string on a non-boundary node, and every edge touching them is to/from a real node.
    for nid, n in compiled.nodes.items():
        if nid in (START_ID, END_ID):
            assert n.kind in (NodeKind.START, NodeKind.END)
    boundary = {START_ID, END_ID}
    for e in compiled.edges:
        if e.from_ in boundary:
            assert compiled.nodes[e.from_].kind is NodeKind.START
        if e.to in boundary:
            assert compiled.nodes[e.to].kind is NodeKind.END


def test_retired_node_modules_are_gone():
    import importlib

    for mod in (
        "yian.nodes.ref",
        "yian.nodes.output_resolver",
        "yian.nodes.collector",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(mod)
    # MAP is re-split back into its own package (no longer retired).
    importlib.import_module("yian.nodes.map")


# --- (3) ${input.X} resolves via START_ID (a defaulted input fills end-to-end) - #

_DEFAULTED = """
id: defaulted
name: defaulted
input:
  topic: str
  window: int = 30
nodes:
  emit:
    kind: code
    input:
      topic: ${input.topic}
      window: ${input.window}
    output:
      report: str
      n: int
    code: tests.engine._compose_codefns:make_report
output:
  report: ${emit.output.report}
  n: ${emit.output.n}
"""


def test_inputs_default_resolves_via_start_d_b_closed():
    # The default binds at the synthesized START_ID's wrapper (not the driver, not run.py).
    # Omitting `window` fills 30 INSIDE START, and `${input.window}` reads it from START_ID's
    # output — proven by the terminal output. `RunResult.input` is the caller-passed args coerced
    # (NOT defaulted): defaulting now lives solely in START's wrapper, so the boundary surface
    # reflects what the caller supplied, while the flow behavior reflects the filled default.
    loaded = load_flow(_DEFAULTED)
    res = run_flow(loaded, {"topic": "ACME"})  # window omitted -> START fills default 30
    assert res.status == "succeeded"
    assert res.input == {"topic": "ACME"}  # caller-passed, coerced (default NOT re-surfaced here)
    assert res.output == {"report": "report for ACME", "n": 4}  # window=30 default used in-flow


# --- (4) author-identical seeds: the recorded run-result goldens still hold -------------- #


def test_author_identical_run_results_hold():
    # LLM-terminating seeds run via the CODE-child goldens (test_golden_run_results.py); re-assert the
    # CALL/MAP run results, blocking interaction, and output arity remain covered.
    from tests.engine.test_golden_run_results import (
        test_golden_human_input_then_past_wait_completes_in_one_run,
        test_golden_map_run_result,
        test_golden_output_arity_0,
        test_golden_output_arity_1,
        test_golden_output_arity_2,
        test_golden_call_run_result,
    )

    test_golden_call_run_result()
    test_golden_map_run_result()
    test_golden_human_input_then_past_wait_completes_in_one_run()
    test_golden_output_arity_0()
    test_golden_output_arity_1()
    test_golden_output_arity_2()


@pytest.mark.parametrize("seed", _REPRESENTATIVE)
def test_representative_seed_loads_with_structure(seed):
    loaded = load_flow((_SEEDS / seed).read_text(), search_paths=[_SEEDS])
    assert loaded.compiled.nodes  # loaded + has a body
    assert loaded.compiled.start_id in loaded.compiled.nodes  # a runnable entry


# --- (5) repo-wide grep-clean: no LIVE reference to a retired symbol -------------------- #


def _live_identifier_refs(path: Path) -> set[str]:
    """Every retired symbol referenced as a LIVE identifier in `path` — `RefNode`,
    `NodeKind.REF`, `terminal_output(...)` — via an AST walk (comments/docstrings are
    invisible to the parser, so intentional history never false-trips)."""
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in (_RETIRED_NAMES | _RETIRED_FUNCS):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            # NodeKind.REF / NodeKind.MAP / x.terminal_output / x._emit_terminal
            if node.attr in _RETIRED_KIND_ATTRS and isinstance(node.value, ast.Name) \
                    and node.value.id == "NodeKind":
                found.add(f"NodeKind.{node.attr}")
            elif node.attr in _RETIRED_FUNCS:
                found.add(node.attr)
    return found


def test_engine_source_has_no_live_reference_to_retired_constructs():
    offenders: dict[str, set[str]] = {}
    for py in (_REPO / "src" / "yian").rglob("*.py"):
        refs = _live_identifier_refs(py)
        if refs:
            offenders[str(py.relative_to(_REPO))] = refs
    assert not offenders, f"live references to retired constructs remain: {offenders}"


def test_grep_clean_guard_actually_bites(tmp_path):
    # The AST walk is only as good as its detection: prove a re-introduced LIVE reference
    # (RefNode / NodeKind.REF / terminal_output) is caught, so the green sweep is meaningful.
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from x import RefNode\n"
        "RefNode()\n"
        "NodeKind.REF\n"
        "obj.terminal_output()\n"
        "# RefNode in a comment is fine\n"
        "'''NodeKind.MAP in a docstring is fine'''\n"
    )
    refs = _live_identifier_refs(probe)
    assert refs == {"RefNode", "NodeKind.REF", "terminal_output"}
