"""End-to-end run tests for the `run_flow` entrypoint.

Ollama-FREE: every flow here uses a TEST-LOCAL CODE function (`tests.engine._compose_codefns:fn`),
so no agent runtime / Ollama is needed. `run_flow` mirrors `run_flow`'s shape — it never
raises on a flow failure; a failed run (incl. a false assert) comes back as a `RunResult`
with `status == "failed"`. The four behaviors pinned:

- a CODE-only flow runs end-to-end -> the declared output object;
- a coalesce-output flow (seed-02 shape, test-local CODE branches via a `case`) runs with
  one branch SKIPPED and END resolves the taken branch;
- a BOUNDARY assert false -> `status == "failed"` BEFORE any node runs (no node output / no
  NodeSucceeded event);
- a POST-TERMINAL assert false -> `status == "failed"` after the run (a succeeded run flips
  to failed); all asserts true -> `"succeeded"`.
"""

from agent_composer.events import NodeSucceeded
from agent_composer.compose import load_flow
from agent_composer.nodes.contracts import Binding
from agent_composer.run import run_flow


# --------------------------------------------------------------------------- #
# fixtures (inline Compose YAML referencing test-local CODE fns)
# --------------------------------------------------------------------------- #


_PLAN_FLOW = """
id: run-plan
name: plan_flow
input:
  topic: str
nodes:
  plan:
    kind: code
    code: tests.engine._compose_codefns:make_plan
    input:
      topic: ${input.topic}
    output:
      rating: str
      score: float
output:
  rating: ${plan.output.rating}
  score: ${plan.output.score}
"""


# A seed-02-shape coalesce flow with test-local CODE branches. `score` produces a
# number from the `seed` input; the `gate` case routes to one branch; the flow returns
# whichever ran via a coalesce.
_CASE_FLOW = """
id: run-case
name: case_flow
input:
  topic: str
  seed: float
nodes:
  score:
    kind: code
    code: tests.engine._compose_codefns:score
    input:
      seed: ${input.seed}
    output: float
  gate:
    kind: case
    cases:
      - when: "${score.output} >= 0.5"
        then: positive
    else: cautious
  positive:
    kind: code
    code: tests.engine._compose_codefns:positive
    input:
      topic: ${input.topic}
    output: str
  cautious:
    kind: code
    code: tests.engine._compose_codefns:cautious
    input:
      topic: ${input.topic}
    output: str
output: ${positive.output | cautious.output}
"""


# A flow with both a boundary assert (`${input.*}` only) and a post-terminal assert
# (`${node.output}`). The booleans are driven by the run inputs so a single fixture pins
# all-true / boundary-false / post-false.
_ASSERT_FLOW = """
id: run-asserts
name: assert_flow
input:
  topic: str
  seed: float
nodes:
  plan:
    kind: code
    code: tests.engine._compose_codefns:make_plan
    input:
      topic: ${input.topic}
    output:
      rating: str
      score: float
output:
  rating: ${plan.output.rating}
  score: ${plan.output.score}
asserts:
  - "${input.seed} >= 0.0"
  - "${plan.output.score} >= 0.5"
"""


def _has_node_output(result) -> bool:
    return any(isinstance(e, NodeSucceeded) for e in result.events)


# --------------------------------------------------------------------------- #
# a CODE-only flow runs end-to-end -> the declared output object
# --------------------------------------------------------------------------- #


def test_code_flow_runs_to_declared_output():
    loaded = load_flow(_PLAN_FLOW)
    result = run_flow(loaded, {"topic": "ACME"})
    assert result.status == "succeeded", result.failure
    assert result.output == {"rating": "plan for ACME", "score": 0.9}


def test_leaf_binds_from_flow_wiring():
    # The engine binds a leaf from flow.wiring + node.params (the wiring split): the flow owns the
    # source, the node carries only the param name, and the loaded flow runs end-to-end from it.
    loaded = load_flow(_PLAN_FLOW)
    wiring = loaded.compiled.wiring["plan"]
    parameters = loaded.compiled.nodes["plan"].params
    assert {name: binding.source for name, binding in wiring.items()} == {
        "topic": "${input.topic}",
    }
    assert wiring["topic"].origin is not None
    assert [p.name for p in parameters] == ["topic"]
    assert parameters[0].origin is not None
    result = run_flow(loaded, {"topic": "ACME"})
    assert result.status == "succeeded", result.failure
    assert result.output == {"rating": "plan for ACME", "score": 0.9}


# --------------------------------------------------------------------------- #
# A coalesce-output flow runs with one branch skipped; END resolves
# the taken branch
# --------------------------------------------------------------------------- #


def test_coalesce_output_resolves_taken_branch_positive():
    loaded = load_flow(_CASE_FLOW)
    result = run_flow(loaded, {"topic": "BETA", "seed": 0.7})
    assert result.status == "succeeded", result.failure
    assert result.output == "pro case for BETA"


def test_case_gate_binds_from_flow_wiring():
    # a case's desugared CaseNode binds its __rN inputs from
    # flow.wiring (the sources relocated off the node); the gate carries only param names.
    loaded = load_flow(_CASE_FLOW)
    gate = loaded.compiled.nodes["gate"]
    binding = loaded.compiled.wiring["gate"]["__r0"]
    assert isinstance(binding, Binding)
    assert binding.source == "${score.output}"
    assert binding.origin == gate.cases[0].origin
    assert [p.name for p in gate.params] == ["__r0"]
    result = run_flow(loaded, {"topic": "BETA", "seed": 0.7})
    assert result.status == "succeeded", result.failure
    assert result.output == "pro case for BETA"


def test_coalesce_output_resolves_taken_branch_cautious():
    loaded = load_flow(_CASE_FLOW)
    result = run_flow(loaded, {"topic": "BETA", "seed": 0.2})
    assert result.status == "succeeded", result.failure
    assert result.output == "cautious note for BETA"


def test_coalesce_skips_untaken_branch():
    # only the taken branch ever produces a value (the other is skip-flooded).
    loaded = load_flow(_CASE_FLOW)
    result = run_flow(loaded, {"topic": "BETA", "seed": 0.7})
    succeeded = {e.node_id for e in result.events if isinstance(e, NodeSucceeded)}
    assert "positive" in succeeded
    assert "cautious" not in succeeded


# --------------------------------------------------------------------------- #
# e08 — input-type enforcement at the flow boundary (read-boundary mirror of the
# node write-boundary check). A non-coercible value fails BEFORE any node runs; a
# coercible string is still coerced and accepted.
# --------------------------------------------------------------------------- #


def test_input_type_mismatch_fails_at_boundary():
    # `seed` is declared float; "high" can't coerce -> a located boundary failure.
    loaded = load_flow(_CASE_FLOW)
    result = run_flow(loaded, {"topic": "BETA", "seed": "high"})
    assert result.status == "failed"
    assert result.failure is not None
    msg = result.failure.message
    assert "input 'seed'" in msg          # names the offending input
    assert "does not match declared type float" in msg  # the precise reason + declared scalar
    assert "high" in msg                  # the offending value
    # the engine never ran: no node produced a value.
    assert not _has_node_output(result)


def test_coercible_string_input_passes_boundary():
    # "0.7" coerces to float 0.7 -> passes enforcement and runs to success.
    loaded = load_flow(_CASE_FLOW)
    result = run_flow(loaded, {"topic": "BETA", "seed": "0.7"})
    assert result.status == "succeeded", result.failure
    assert result.output == "pro case for BETA"


# --------------------------------------------------------------------------- #
# boundary assert false -> failed BEFORE any node runs
# --------------------------------------------------------------------------- #


def test_boundary_assert_false_fails_before_any_node_runs():
    loaded = load_flow(_ASSERT_FLOW)
    result = run_flow(loaded, {"topic": "ACME", "seed": -1.0})
    assert result.status == "failed"
    assert result.failure is not None
    assert "assert failed" in result.failure.message
    # the engine never ran: no node produced a value (no NodeSucceeded event).
    assert not _has_node_output(result)


# --------------------------------------------------------------------------- #
# post-terminal assert false -> failed AFTER the run (succeeded flips to failed)
# --------------------------------------------------------------------------- #


def test_post_terminal_assert_false_flips_succeeded_to_failed():
    # make_plan always returns score 0.9, so to fail the post assert we raise the
    # threshold via... no — instead use a flow whose post assert is unsatisfiable.
    flow = _ASSERT_FLOW.replace(
        "${plan.output.score} >= 0.5", "${plan.output.score} >= 2.0"
    )
    loaded = load_flow(flow)
    result = run_flow(loaded, {"topic": "ACME", "seed": 1.0})
    assert result.status == "failed"
    assert result.failure is not None
    assert "assert failed" in result.failure.message
    # the run DID reach the node (the assert fired post-terminal, not at the boundary).
    assert _has_node_output(result)


# --------------------------------------------------------------------------- #
# all asserts true -> succeeded
# --------------------------------------------------------------------------- #


def test_all_asserts_true_succeeds():
    loaded = load_flow(_ASSERT_FLOW)
    result = run_flow(loaded, {"topic": "ACME", "seed": 1.0})
    assert result.status == "succeeded", result.failure
    assert result.output == {"rating": "plan for ACME", "score": 0.9}


# --------------------------------------------------------------------------- #
# ${system.run_id} — host-injected ambient (override) / minted default
# --------------------------------------------------------------------------- #


_RUN_ID_FLOW = """
id: run-runid
name: runid_flow
input:
  topic: str
nodes:
  echo:
    kind: code
    code: tests.engine._compose_codefns:echo_rid
    input:
      rid: ${system.run_id}
    output: str
output: ${echo.output}
"""


def test_run_id_host_override_resolves():
    loaded = load_flow(_RUN_ID_FLOW)
    result = run_flow(loaded, {"topic": "ACME"}, run_id="rid-123")
    assert result.status == "succeeded", result.failure
    assert result.output == "rid-123"


def test_run_id_minted_when_omitted():
    loaded = load_flow(_RUN_ID_FLOW)
    result = run_flow(loaded, {"topic": "ACME"})
    assert result.status == "succeeded", result.failure
    assert isinstance(result.output, str) and result.output  # a fresh non-empty id
