/**
 * Forward captures: a port of `tests/engine/test_forward_trace.py`. Ported:
 * `test_forward_capture_rejects_duplicate_record_names`, and the constructor halves of
 * `test_forward_capture_round_trips_invocation_names_and_structural_layout` (the caller's later
 * edit of `node_data` doesn't show),
 * `test_forward_capture_rejects_malformed_invocation_inventory_and_paths` (10 rows) and
 * `test_forward_capture_rejects_noncanonical_embedded_json` (5 rows), and
 * `test_scope_free_evidence_rejects_recalled_sections`. Python's `ValueError` is a `RangeError`
 * here.
 *
 * Ported, the record halves on the run log: of
 * `test_forward_capture_round_trips_invocation_names_and_structural_layout`,
 * `test_forward_capture_rejects_malformed_invocation_inventory_and_paths` (10 rows) and
 * `test_forward_capture_rejects_noncanonical_embedded_json` (5 rows, the reader's
 * `JSONDecodeError`); `test_schema_six_requires_explicit_invocation_inventories`,
 * `test_forward_capture_round_trips_without_memory_scope`,
 * `test_forward_capture_rejects_malformed_nested_route_wire` (15 rows),
 * `test_forward_capture_keeps_the_existing_canonical_depth_range`,
 * `test_codec_preserves_evaluated_route_variants` (9 rows),
 * `test_codec_preserves_opaque_argument_labels` and
 * `test_codec_rejects_misaligned_or_nonstring_opaque_argument_labels`. Each rejected row names its
 * class and message (a `JSONDecodeError` its reason).
 *
 * TS-only (run log): repeated or unsorted `evaluated_keys` on the wire, and an evaluated key added
 * to a capture's mutable `Set` after construction, refused by the encoder.
 *
 * TS-only: one row per `ForwardCapture` message with its class; a capture with items, scope and
 * node data keeps frozen copies; `ForwardCapture`'s defaults; an unfinished capture may keep an
 * unfinished item; `BackwardCapture`'s defaults, one row per check its constructor wires in, and
 * its frozen copy of `nodeData`; a `Set` subclass whose iterator changes between passes stores the
 * keys that were checked.
 *
 * Changed:
 * - Captures hold frozen copies, with no copy on read: a reader's edit throws a `TypeError` where
 *   Python changes a private copy, and `evaluatedKeys` is a private `Set` copy, read-only by
 *   contract, where Python keeps the caller's `frozenset`.
 * - Inventories and items are arrays, not tuples; `evaluatedKeys` is a `Set`, not a `frozenset`.
 * - Each rejected row names its one error class, where Python accepts either `TypeError` or
 *   `ValueError`. A malformed `value_json` throws the reader's `JSONDecodeError`.
 * - Schema 10 routes have no `count`, so the two `count` rows of the nested-route test are extra
 *   fields; schema 6 reads `unsupported run-record schema: 6`.
 * - A decoded capture is frozen, so editing its inputs or node data throws a `TypeError` where
 *   Python edits a private copy.
 * - `partial()` returns a record holding a file (the TS leaf JSON can't hold) for Python's
 *   `object()`; `collect_args` takes the named parameters `first`, `second` and `value`, each
 *   defaulting to 0, for Python's `*args, **kwargs`, so its call log holds the bound values.
 *
 * Not ported: the forged-field rows of `test_codec_revalidates_forged_immutable_capture_fields`
 * (`children`, `path`, `scope`): a TS route, source and scope are frozen by their constructors,
 * so a capture holds them without rebuilding them; the repetition rows (`${a * 2}`,
 * `${-2 * a}`) of `test_codec_preserves_evaluated_route_variants`, since repetition is a type
 * error in TS.
 *
 * TS-only (trace), with entries installed by setting the trace's internal maps: `captureBindings`
 * over a hand-built `BindingCapture` with one item and its two type errors; `SourceEntry.value`
 * restoring files through `files` where `storedValue` doesn't, each read fresh; `sameSourceEntry`
 * over each field; one row per `TraceTerminal` message and its `output`; `checkMemoryScope`; a
 * `TraceOccurrence` keeping a decoded copy of its envelope; `selectSeedRoute`; `RetainedTrace`'s
 * defaults, `lookupSource`, `resolve`, `occurrence` and `backwardContext`.
 *
 * TS-only (trace deltas), driving `prepareTrace` and `TraceDelta.apply` over `learnableRun()`: the
 * trace after each apply (started, occurrence, writer, sources, handler), a learnable RunStarted's
 * pool entries, root writer and system values, and the terminals (root writer, failed occurrence,
 * first causal route) and the root START source; `null` for a RunStarted that doesn't learn and
 * for records before one; one row per `prepareTrace` message with its class, the lookup error as
 * the cause of an absent source path, and the canonical terminal compare; one row per
 * `validateOrigin` message (another trace, a stale sequence, a forged occurrence, a consumed
 * connection gone); seed connections sorted, then consumed by the child start with evaluated keys
 * winning; boundaries sorted and filled by the child's endpoints; an expansion pruning current
 * names only; a resume clearing unfinished starts; an applied delta leaving earlier occurrences
 * alone; a `TraceDelta`'s defaults and frozen arrays.
 *
 * TS-only (frontier trace), on `learnableStarted` with START's `NodeStarted` committed: a learnable
 * `RunStopped` carries a sequence-only trace delta and commits; a forged stop plan whose trace
 * delta starts a node is refused; a frontier plan holding an outcome record is refused by the
 * frontier trace check.
 *
 * Ported, the run-state trace checks, on the shared `startTraceNode` (Python's
 * `_start_trace_node`) and a local `preparedScopedOutcome` (Python's `_prepared_scoped_outcome`,
 * its `success` kind): `test_outcome_trace_writer_matches_the_same_accepted_physical_output`
 * (`target`, `source`, `sequence`, `value`, `missing`),
 * `test_terminal_trace_root_writer_must_be_the_physical_accepted_output` (`none`, `source`,
 * `sequence`, `value`), and the `success` rows of
 * `test_terminal_trace_output_must_match_the_prepared_outcome_before_append`,
 * `test_outcome_trace_occurrence_identity_matches_the_canonical_record` (`replace`: `source`,
 * `node`, `sequence`, `record`), `test_outcome_trace_cannot_override_dispatch_authority`
 * (`initialized`, `started`, `resumed`), `test_terminal_trace_rejects_forged_fields_before_append`
 * (`replace`: every change but `source_float`),
 * `test_nonterminal_outcome_rejects_fabricated_terminal_trace_authority` (4 rows),
 * `test_outcome_rejects_valid_scope_substitution_before_append` (`replace`: 3 rows) and
 * `test_outcome_scope_authority_cannot_be_selected_or_omitted_by_trace_delta` (3 rows). Each
 * rejected row names its one class and full message (all `RangeError`s, where Python matches a
 * part of a `ValueError` or accepts either class); a refused commit leaves the log, the state
 * (the shared `snapshot`) and the trace (the shared `traceSnapshot`) unchanged.
 *
 * Ported, after rewrite (the start is built with prepareRunStarted, not the engine), on
 * `learnableStarted`: `test_trace_is_applied_only_after_successful_append` (the log's `append`
 * patched on the instance, its error an `Error`, Python's `OSError`),
 * `test_terminal_trace_root_start_latch_must_match_the_accepted_start` (`none`, `sequence`, `run`),
 * `test_paired_resolver_preserves_pool_namespaces` (10 rows; the system value is the fixture's run
 * id) and `test_reducer_requires_capture_exactly_when_learning` (its run that doesn't learn on
 * `initializedAndStarted`); the frontier trace checks, on a local `preparedTraceFrontier` (Python's
 * `_prepared_trace_frontier`): `test_frontier_trace_cannot_inject_a_causal_terminal_source_before_append`,
 * `test_frontier_trace_rejects_inappropriate_fields_before_append` (19 of its 34 rows; the rest
 * below) and `test_frontier_trace_started_identity_is_exact` (`node`, `sequence`).
 *
 * Changed:
 * - A forged trace delta, occurrence, source entry or terminal is rebuilt with its constructor,
 *   for Python's `dataclasses.replace`; a value its constructor refuses (a terminal whose
 *   `succeeded` is `1`, a success terminal with a failure, an occurrence with a scope that isn't a
 *   `MemoryScope`) is forged through its class's prototype, for Python's `object.__setattr__`.
 * - The `carrier` row of `test_trace_rejects_forged_memory_scope_before_append_and_when_copied`:
 *   the occurrence constructor refuses the scope, and the commit refuses a forged occurrence that
 *   holds it, each with a `TypeError` naming the historical memory scope.
 * - The trace snapshot keeps the backward handlers by reference, for Python's bound-method
 *   identities.
 * - `preparedTraceFrontier` dispatches `work` of the shared `flow()`, for Python's `router` of
 *   `_terminal_skip_route_flow()`: the frontier checks read only the dispatched node, and routes
 *   come in PR 2.
 *
 * Not ported (TS types): every `mutate` row (a TS delta, occurrence and terminal are frozen); the
 * `float_sequence` and `source_float` rows, and the `sequence` row of
 * `test_outcome_trace_cannot_override_dispatch_authority` (`float(n)` is the number `n`); the
 * `tuple` row of `test_outcome_trace_writer_matches_the_same_accepted_physical_output` (a list
 * and a tuple are both an array); the `nested` row of
 * `test_trace_rejects_forged_memory_scope_before_append_and_when_copied` (a `MemoryScope` is
 * frozen by its constructor); of `test_frontier_trace_rejects_inappropriate_fields_before_append`,
 * the `sequence` rows (`float(n)`), the `resumed` rows (`int(bool)`), the `prune_ids`,
 * `connections` and `boundaries` rows (`EmptyChanges`; a `TraceDelta` holds frozen array copies)
 * and the `consumed_connection` and `boundary_update` rows (`UnavailableChange`; it holds frozen
 * pair copies), and the `started` row of a dispatch (a list and a tuple are both an array); of
 * `test_frontier_trace_started_identity_is_exact`, the `float_sequence` row and the
 * `tuple_subclass` row (the delta copies the pair by index).
 *
 * TS-only (frontier trace): the same history-field rows, with `started` and `resumed`, on a
 * prepared `NodeStep` and `RunStopped` of the in-flight `work`.
 *
 * Not ported (thread race): `test_a_lookup_retains_its_entry_when_a_new_writer_commits`.
 *
 * Later (T4-4 PR 2), routes and failures: the `route` and `failure` rows of the outcome and
 * terminal trace tests above, and
 * `test_outcome_trace_source_must_match_the_physical_accepted_start_before_append` (a failure).
 *
 * Later (T4-5), growth and replay: the `growth` rows of the outcome trace tests above,
 * `test_terminal_trace_accepts_detached_authority_without_repreparing`,
 * `test_terminal_trace_uses_the_accepted_route_latch_until_an_explicit_failure` and
 * `test_outcome_accepts_detached_physical_scope_and_replays_it`.
 *
 * Later (T4-6, with the engine): `test_accepted_backward_inventories_and_layout_replay_without_recapturing`,
 * `test_static_capture_host_fault_does_not_append_a_node_outcome`,
 * `test_nested_agent_render_capture_uses_authored_model_map_and_replays_without_execution`,
 * `test_agent_render_capture_host_and_append_failures_do_not_install_worker_data`,
 * `test_discarded_agent_worker_keeps_render_capture_out_of_accepted_trace`,
 * `test_trace_source_is_the_actual_commit_as_producer`,
 * `test_failed_binding_and_rejected_outcomes_retain_canonical_failures`,
 * `test_pooled_lookup_keeps_value_and_producer_identity_together`,
 * `test_trace_memory_scope_is_detached_and_replayed_without_evidence`,
 * `test_learn_false_omits_backward_capture`,
 * `test_nested_growth_preserves_authored_paths_and_local_driver_zero`,
 * `test_missing_seed_selector_is_an_accepted_growth_failure`,
 * `test_growth_recipe_tampering_fails_replay_before_apply`,
 * `test_reference_looking_seed_strings_reach_the_child_as_text`,
 * `test_failed_child_binding_joins_captured_fields`,
 * `test_seed_connections_wait_for_append_and_survive_unfinished_start_retry`,
 * `test_pruned_seed_connections_do_not_leak_into_reused_node_ids`,
 * `test_call_seed_routes_survive_materialized_values`,
 * `test_null_materialized_seed_keeps_its_selected_source`,
 * `test_map_seed_routes_preserve_equal_item_positions`,
 * `test_loop_carried_routes_survive_live_pruning`, `test_trace_replay_matches_live_accepted_prefixes`,
 * `test_retry_clears_unfinished_start_identity_and_primed_input_is_replaced`,
 * `test_capture_canonicalizes_supported_binding_objects`,
 * `test_learning_retains_opaque_calls_with_arguments`,
 * `test_learning_retains_valid_results_of_noncanonical_unicode_temporaries` and
 * `test_learning_retains_noncanonical_partial_temporary_comparisons`.
 *
 * Later (T4-7, with `computeBackward`): `test_discarded_pooled_outcomes_are_not_trace_occurrences`
 * and `test_frontier_trace_and_terminal_source_match_live_replay_without_repreparing`.
 */

import { afterEach, expect, test } from "bun:test";

import { type CompiledFlow, END_ID, START_ID } from "../../src/compile/model.ts";
import { registerTemplateFn, TEMPLATE_FNS, type TemplateFn, type TemplateParam } from "../../src/expr/builtins.ts";
import { Failure } from "../../src/diagnostics.ts";
import { ExpressionError } from "../../src/expr/expressions.ts";
import {
  type FeedbackValue,
  ResolvedValue,
  routeFeedback,
  SourceRef,
  sourceKey,
  ValueRoute,
} from "../../src/expr/lineage.ts";
import { evalBinding } from "../../src/expr/template.ts";
import { EvaluationEvidence, MemoryScope, MemorySection } from "../../src/memory/contracts.ts";
import { BindingCapture } from "../../src/nodes/binding.ts";
import { BackwardContext, BackwardResult, Node, NodeKind } from "../../src/nodes/base.ts";
import { ChildFlowBoundary, ExecutionPosition, ExecutionScope, SeedConnection } from "../../src/nodes/contracts.ts";
import { BackwardCapture, ForwardCapture } from "../../src/runtime/contracts.ts";
import {
  canonicalSeed,
  canonicalTyped,
  decodeRecord,
  encodePool,
  encodeRecord,
  evidencePayload,
  type MemoryRunLog,
  NodeExpandedRecord,
  NodeFailedRecord,
  NodeRoutedRecord,
  NodeStartedRecord,
  NodeSucceededRecord,
  RecordEnvelope,
  type RunRecord,
  RunResumedRecord,
  RunStartedRecord,
  RunStoppedRecord,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";
import {
  captureBindings,
  checkMemoryScope,
  type OutcomeRecord,
  type PrepareTraceOptions,
  prepareTrace,
  RetainedTrace,
  SeedRoutes,
  SourceEntry,
  sameSourceEntry,
  selectSeedRoute,
  TraceDelta,
  type TraceDeltaFields,
  TraceOccurrence,
  TraceTerminal,
} from "../../src/runtime/trace.ts";
import {
  commitPrepared,
  PreparedTransition,
  prepareNodeStarted,
  prepareNodeStep,
  prepareNodeSucceeded,
  prepareRunResumed,
  prepareRunStopped,
  type RunState,
} from "../../src/runtime/state.ts";
import { FileRef, Type, type Value, ValueKind, VariablePool } from "../../src/typesys/index.ts";
import { JSONDecodeError, logJson, parseJson } from "../../src/typesys/json.ts";
import {
  commitStartSuccess,
  expectUnchanged,
  flow,
  forwardCapture,
  FuncNode,
  graph,
  initializedAndStarted,
  learnableStarted,
  RUN_ID,
  snapshot,
  startTraceNode,
  traceSnapshot,
} from "./run_state_fixtures.ts";

type ErrorClass = typeof TypeError | typeof RangeError;

/** A record route with one literal field `x`. */
const ROUTE_X = ValueRoute.record([["x", ValueRoute.literal(1)]]);

/** A finished capture of `x`. */
const FINISHED = new ForwardCapture({ x: 1 }, ROUTE_X);

/** An unfinished capture of `x`. */
const UNFINISHED = new ForwardCapture(null, ROUTE_X);

test("a forward capture keeps invocation names and a detached copy of node data", () => {
  const data = { input_output_paths: { result: [] as string[] } };
  const route = ValueRoute.record([["result", ValueRoute.literal({ items: [1] })]]);
  const capture = new ForwardCapture({ result: { items: [1] } }, route, {
    nodeData: data,
    inputNames: ["result"],
    variableNames: ["instruction"],
  });
  data.input_output_paths.result.push("changed");
  expect(capture.inputNames).toEqual(["result"]);
  expect(capture.variableNames).toEqual(["instruction"]);
  expect(capture.nodeData).toEqual({ input_output_paths: { result: [] } });
});

test.each<[string, string, unknown, ErrorClass, string]>([
  ["input names null", "inputNames", null, TypeError, "input_names must be an array"],
  ["duplicate input names", "inputNames", ["result", "result"], RangeError, "input_names must contain unique names"],
  ["a non-string input name", "inputNames", [1], TypeError, "input_names must contain only strings"],
  [
    "duplicate variable names",
    "variableNames",
    ["instruction", "instruction"],
    RangeError,
    "variable_names must contain unique names",
  ],
  ["a non-string variable name", "variableNames", [false], TypeError, "variable_names must contain only strings"],
  ["paths as an array", "nodeData", { input_output_paths: [] }, TypeError, "input_output_paths must be a record"],
  [
    "a path for an unknown input",
    "nodeData",
    { input_output_paths: { unknown: [] } },
    RangeError,
    "input_output_paths must name declared inputs",
  ],
  [
    "a boolean path part",
    "nodeData",
    { input_output_paths: { result: [true] } },
    RangeError,
    "output path components must be strings or non-negative integers",
  ],
  [
    "a negative path part",
    "nodeData",
    { input_output_paths: { result: [-1] } },
    RangeError,
    "output path components must be strings or non-negative integers",
  ],
  [
    "a path as a string",
    "nodeData",
    { input_output_paths: { result: "result" } },
    TypeError,
    "input_output_paths values must be arrays",
  ],
])("a forward capture rejects malformed invocation inventory and paths: %s", (_label, field, value, cls, message) => {
  const options = { inputNames: ["result"], variableNames: ["instruction"], [field]: value };
  const route = ValueRoute.record([["result", ValueRoute.literal(1)]]);
  const build = () => new ForwardCapture({ result: 1 }, route, options as never);
  expect(build).toThrow(cls);
  expect(build).toThrow(message);
});

/** Embedded `value_json` texts the reader refuses, each with its `JSONDecodeError` reason. */
const NONCANONICAL_JSON: [string, string, string][] = [
  ["duplicate key", '{"a":1,"a":2}', 'duplicate JSON object key: "a"'],
  ["nested duplicate key", '{"nested":{"a":1,"a":2}}', 'duplicate JSON object key: "a"'],
  ["surrogate string", '"\\ud800"', "string has an unpaired surrogate"],
  ["surrogate value", '{"nested":"\\ud800"}', "string has an unpaired surrogate"],
  ["surrogate key", '{"\\ud800":1}', "object key has an unpaired surrogate"],
];

/** Assert that `fn` throws exactly a `JSONDecodeError` with `reason`. */
function expectJsonDecodeError(fn: () => unknown, reason: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect((caught as Error)?.constructor).toBe(JSONDecodeError);
  expect((caught as JSONDecodeError).reason).toBe(reason);
}

test.each(NONCANONICAL_JSON)(
  "a forward capture rejects noncanonical embedded JSON: %s",
  (_label, valueJson, reason) => {
    const route = new ValueRoute("literal", { valueJson });
    expectJsonDecodeError(() => new ForwardCapture({ x: null }, ValueRoute.record([["x", route]])), reason);
  },
);

test("a forward capture rejects duplicate record names", () => {
  const route = ValueRoute.record([
    ["x", ValueRoute.literal(1)],
    ["x", ValueRoute.literal(2)],
  ]);
  const build = () => new ForwardCapture({ x: 2 }, route);
  expect(build).toThrow(RangeError);
  expect(build).toThrow("unique");
});

test("a forward capture keeps frozen copies of its values", () => {
  const bound = { x: [1, 2] };
  const keys = new Set(["x"]);
  const items: [number, ForwardCapture][] = [[0, FINISHED], [2, FINISHED]];
  const data = { static: { label: "a" }, input_output_paths: { x: [0] } };
  const names = ["x"];
  const scope = new ExecutionScope(["work"]);
  const capture = new ForwardCapture(bound, ROUTE_X, {
    evaluatedKeys: keys,
    items,
    scope,
    nodeData: data,
    inputNames: names,
    variableNames: ["tone"],
  });
  bound.x.push(3);
  keys.add("y");
  items.push([3, FINISHED]);
  items[0]![0] = 9;
  data.static.label = "b";
  names.push("y");

  expect(capture.boundInputs).toEqual({ x: [1, 2] });
  expect(capture.route).toBe(ROUTE_X);
  expect([...capture.evaluatedKeys]).toEqual(["x"]);
  expect(capture.evaluatedKeys).not.toBe(keys);
  expect(capture.items).toEqual([[0, FINISHED], [2, FINISHED]]);
  expect(capture.items[0]![1]).toBe(FINISHED);
  expect(capture.scope).toBe(scope);
  expect(capture.nodeData).toEqual({ static: { label: "a" }, input_output_paths: { x: [0] } });
  expect(capture.inputNames).toEqual(["x"]);
  expect(capture.variableNames).toEqual(["tone"]);

  expect(() => (capture.boundInputs!.x as number[]).push(4)).toThrow(TypeError);
  expect(() => (capture.items as unknown[]).push([5, FINISHED])).toThrow(TypeError);
  expect(() => {
    (capture.items[0] as unknown as number[])[0] = 7;
  }).toThrow(TypeError);
  expect(() => {
    (capture.nodeData!.static as { label: string }).label = "c";
  }).toThrow(TypeError);
  expect(() => (capture.inputNames as string[]).push("z")).toThrow(TypeError);
  expect(() => (capture.variableNames as string[]).push("z")).toThrow(TypeError);
  expect(() => {
    (capture as { scope: ExecutionScope | null }).scope = null;
  }).toThrow(TypeError);
});

test("a forward capture stores the evaluated keys it checked", () => {
  // Yields `x` on the first pass and an uncaptured key on any later pass.
  class ShiftingSet extends Set<string> {
    passes = 0;
    override *[Symbol.iterator](): SetIterator<string> {
      yield this.passes++ === 0 ? "x" : "nope";
    }
  }
  const capture = new ForwardCapture({ x: 1 }, ROUTE_X, { evaluatedKeys: new ShiftingSet(["x"]) });
  expect([...capture.evaluatedKeys]).toEqual(["x"]);
});

test("a forward capture defaults to empty keys, items and inventories", () => {
  const capture = new ForwardCapture({}, ValueRoute.record([]));
  expect(capture.evaluatedKeys.size).toBe(0);
  expect(capture.items).toEqual([]);
  expect(capture.scope).toBeNull();
  expect(capture.nodeData).toBeNull();
  expect(capture.inputNames).toEqual([]);
  expect(capture.variableNames).toEqual([]);
});

test("an unfinished capture may keep unfinished items", () => {
  const capture = new ForwardCapture(null, ROUTE_X, { items: [[0, UNFINISHED]] });
  expect(capture.boundInputs).toBeNull();
  expect(capture.items[0]![1]).toBe(UNFINISHED);
});

test.each<[string, () => unknown, ErrorClass, string]>([
  [
    "a route that isn't a record",
    () => new ForwardCapture(null, ValueRoute.literal(1)),
    RangeError,
    "input route must be a record",
  ],
  [
    "evaluated keys that aren't a set",
    () => new ForwardCapture({ x: 1 }, ROUTE_X, { evaluatedKeys: ["x"] as never }),
    TypeError,
    "evaluated_keys must be a set of strings",
  ],
  [
    "a non-string evaluated key",
    () => new ForwardCapture({ x: 1 }, ROUTE_X, { evaluatedKeys: new Set([1]) as never }),
    TypeError,
    "evaluated_keys must be a set of strings",
  ],
  [
    "an evaluated key that isn't captured",
    () => new ForwardCapture({ x: 1 }, ROUTE_X, { evaluatedKeys: new Set(["y"]) }),
    RangeError,
    "evaluated_keys must name captured routes",
  ],
  [
    "bound inputs as an array",
    () => new ForwardCapture([1] as never, ROUTE_X),
    TypeError,
    "bound_inputs must be a JSON object or null",
  ],
  [
    "bound input keys that differ from the routes",
    () => new ForwardCapture({ y: 1 }, ROUTE_X),
    RangeError,
    "bound_inputs keys must match input routes",
  ],
  [
    "items that aren't an array",
    () => new ForwardCapture({ x: 1 }, ROUTE_X, { items: new Set() as never }),
    TypeError,
    "items must be an array",
  ],
  [
    "an item that isn't a pair",
    () => new ForwardCapture({ x: 1 }, ROUTE_X, { items: [[0]] as never }),
    TypeError,
    "item captures must be index/capture pairs",
  ],
  [
    "a negative item index",
    () => new ForwardCapture({ x: 1 }, ROUTE_X, { items: [[-1, FINISHED]] }),
    RangeError,
    "item indices must be non-negative and increasing",
  ],
  [
    "a repeated item index",
    () => new ForwardCapture({ x: 1 }, ROUTE_X, { items: [[1, FINISHED], [1, FINISHED]] }),
    RangeError,
    "item indices must be non-negative and increasing",
  ],
  [
    "an item that isn't a ForwardCapture",
    () => new ForwardCapture({ x: 1 }, ROUTE_X, { items: [[0, { ...FINISHED }]] as never }),
    TypeError,
    "item capture must be a ForwardCapture",
  ],
  [
    "an unfinished capture with a finished item",
    () => new ForwardCapture(null, ROUTE_X, { items: [[0, FINISHED]] }),
    RangeError,
    "unfinished binding cannot retain usable item inputs",
  ],
  [
    "a scope that isn't an ExecutionScope",
    () => new ForwardCapture({ x: 1 }, ROUTE_X, { scope: new ExecutionPosition("call", ["work"]) as never }),
    TypeError,
    "scope must be an ExecutionScope or null",
  ],
  [
    "node data as an array",
    () => new ForwardCapture({ x: 1 }, ROUTE_X, { nodeData: [] as never }),
    TypeError,
    "node_data must be a record or null",
  ],
])("a forward capture rejects %s", (_label, build, cls, message) => {
  expect(build).toThrow(cls);
  expect(build).toThrow(message);
});

test("a backward capture defaults to empty inventories, no scope and empty node data", () => {
  for (const capture of [new BackwardCapture(), new BackwardCapture({ nodeData: null })]) {
    expect(capture.inputNames).toEqual([]);
    expect(capture.variableNames).toEqual([]);
    expect(capture.scope).toBeNull();
    expect(capture.nodeData).toEqual({});
  }
});

test("a backward capture keeps a frozen copy of its node data", () => {
  const data = { static: { label: "a" }, input_output_paths: { result: [] as string[] } };
  const names = ["result"];
  const scope = new ExecutionScope(["work"]);
  const capture = new BackwardCapture({ inputNames: names, variableNames: ["tone"], scope, nodeData: data });
  data.static.label = "b";
  data.input_output_paths.result.push("changed");
  names.push("other");
  expect(capture.nodeData).toEqual({ static: { label: "a" }, input_output_paths: { result: [] } });
  expect(capture.inputNames).toEqual(["result"]);
  expect(capture.variableNames).toEqual(["tone"]);
  expect(capture.scope).toBe(scope);
  expect(() => {
    (capture.nodeData.static as { label: string }).label = "c";
  }).toThrow(TypeError);
  expect(() => (capture.inputNames as string[]).push("z")).toThrow(TypeError);
  expect(() => (capture.variableNames as string[]).push("z")).toThrow(TypeError);
  expect(() => {
    (capture as { scope: ExecutionScope | null }).scope = null;
  }).toThrow(TypeError);
});

test.each<[string, string, unknown, ErrorClass, string]>([
  ["duplicate input names", "inputNames", ["result", "result"], RangeError, "input_names must contain unique names"],
  [
    "duplicate variable names",
    "variableNames",
    ["instruction", "instruction"],
    RangeError,
    "variable_names must contain unique names",
  ],
  [
    "a path for an unknown input",
    "nodeData",
    { input_output_paths: { unknown: [] } },
    RangeError,
    "input_output_paths must name declared inputs",
  ],
  [
    "a scope that isn't an ExecutionScope",
    "scope",
    { staticPath: ["work"] },
    TypeError,
    "scope must be an ExecutionScope or null",
  ],
  ["node data as an array", "nodeData", [], TypeError, "node_data must be a record"],
])("a backward capture rejects %s", (_label, field, value, cls, message) => {
  const options = { inputNames: ["result"], variableNames: ["instruction"], [field]: value };
  const build = () => new BackwardCapture(options as never);
  expect(build).toThrow(cls);
  expect(build).toThrow(message);
});

test("scope-free evidence rejects recalled sections", () => {
  const build = () => new EvaluationEvidence(null, {}, null, null, [new MemorySection("outer", "recalled")]);
  expect(build).toThrow(RangeError);
  expect(build).toThrow("memory_sections memories must appear in scope.memories");
});

// --------------------------------------------------------------------------------------- //
// Captures on the run log
// --------------------------------------------------------------------------------------- //

/** A node-success envelope at seq 3 carrying `capture`, with a typed value of `value`. */
function capturedEnvelope(capture: ForwardCapture, value: unknown = null, evidence: EvaluationEvidence | null = null) {
  const record = new NodeSucceededRecord("work", "work", canonicalTyped(value, null)[0], {
    evidence,
    forwardCapture: capture,
  });
  return new RecordEnvelope(SCHEMA_VERSION, "run", 3, record);
}

/** The decoded JSON of `envelope`'s line, ints as `bigint`s, as a mutable object. */
function wireOf(envelope: RecordEnvelope): Record<string, any> {
  return parseJson(encodeRecord(envelope), "codec") as Record<string, any>;
}

/** Decode `wire` written back as a log line. */
function decodeWire(wire: unknown): RecordEnvelope {
  return decodeRecord(logJson(wire));
}

/** Assert that `fn` throws exactly `cls` with exactly `message`. */
function expectError(fn: () => unknown, cls: new (...args: never[]) => Error, message: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect((caught as Error)?.constructor).toBe(cls);
  expect((caught as Error).message).toBe(message);
}

test("a forward capture round trips invocation names and structural layout", () => {
  const data = { input_output_paths: { result: [] as string[] } };
  const capture = new ForwardCapture(
    { result: { items: [1n] } },
    ValueRoute.record([["result", ValueRoute.literal({ items: [1n] })]]),
    { nodeData: data, inputNames: ["result"], variableNames: ["instruction"] },
  );
  data.input_output_paths.result.push("changed");
  const envelope = capturedEnvelope(capture, { items: [1n] });

  const decoded = decodeRecord(encodeRecord(envelope));

  expect(decoded).toEqual(envelope);
  const decodedCapture = (decoded.record as NodeSucceededRecord).forwardCapture!;
  expect(decodedCapture.inputNames).toEqual(["result"]);
  expect(decodedCapture.variableNames).toEqual(["instruction"]);
  expect(decodedCapture.nodeData).toEqual({ input_output_paths: { result: [] } });
  expect(decoded.schema).toBe(SCHEMA_VERSION);
  const old = wireOf(envelope);
  old.schema = 6n;
  expectError(() => decodeWire(old), RangeError, "unsupported run-record schema: 6");
});

const CAPTURE_LABEL = "node_succeeded payload forward_capture";

test.each<[string, string, unknown, ErrorClass, string]>([
  ["input names null", "input_names", null, TypeError, `${CAPTURE_LABEL} input_names must be an array of strings`],
  ["duplicate input names", "input_names", ["result", "result"], RangeError, "input_names must contain unique names"],
  [
    "a non-string input name",
    "input_names",
    [1n],
    TypeError,
    `${CAPTURE_LABEL} input_names must be an array of strings`,
  ],
  [
    "duplicate variable names",
    "variable_names",
    ["instruction", "instruction"],
    RangeError,
    "variable_names must contain unique names",
  ],
  [
    "a non-string variable name",
    "variable_names",
    [false],
    TypeError,
    `${CAPTURE_LABEL} variable_names must be an array of strings`,
  ],
  ["paths as an array", "node_data", { input_output_paths: [] }, TypeError, "input_output_paths must be a record"],
  [
    "a path for an unknown input",
    "node_data",
    { input_output_paths: { unknown: [] } },
    RangeError,
    "input_output_paths must name declared inputs",
  ],
  [
    "a boolean path part",
    "node_data",
    { input_output_paths: { result: [true] } },
    RangeError,
    "output path components must be strings or non-negative integers",
  ],
  [
    "a negative path part",
    "node_data",
    { input_output_paths: { result: [-1n] } },
    RangeError,
    "output path components must be strings or non-negative integers",
  ],
  [
    "a path as a string",
    "node_data",
    { input_output_paths: { result: "result" } },
    TypeError,
    "input_output_paths values must be arrays",
  ],
])("a capture's record rejects malformed invocation inventory and paths: %s", (_label, field, value, cls, message) => {
  const capture = new ForwardCapture({ result: 1n }, ValueRoute.record([["result", ValueRoute.literal(1n)]]), {
    inputNames: ["result"],
    variableNames: ["instruction"],
  });
  const wire = wireOf(capturedEnvelope(capture, 1n));
  wire.payload.forward_capture[field] = value;
  expectError(() => decodeWire(wire), cls, message);
});

test.each(NONCANONICAL_JSON)(
  "a capture's record rejects noncanonical embedded JSON: %s",
  (_label, valueJson, reason) => {
    const capture = new ForwardCapture({ x: null }, ValueRoute.record([["x", ValueRoute.literal(null)]]));
    const wire = wireOf(capturedEnvelope(capture));
    wire.payload.forward_capture.route.children[0].value_json = valueJson;
    expectJsonDecodeError(() => decodeWire(wire), reason);
  },
);

// TS-only: the evaluated keys on the wire, and a key added to the capture's mutable `Set` after
// construction, which the encoder checks again.
test.each<[string, string[]]>([
  ["repeated", ["x", "x"]],
  ["unsorted", ["y", "x"]],
])("a capture's record rejects %s evaluated keys", (_label, keys) => {
  const route = ValueRoute.record([
    ["x", ValueRoute.literal(1n)],
    ["y", ValueRoute.literal(2n)],
  ]);
  const wire = wireOf(capturedEnvelope(new ForwardCapture({ x: 1n, y: 2n }, route)));
  wire.payload.forward_capture.evaluated_keys = keys;
  expectError(() => decodeWire(wire), RangeError, "evaluated_keys must be sorted and unique");
});

test("the encoder refuses an evaluated key added after construction", () => {
  const capture = new ForwardCapture({ x: 1n }, ValueRoute.record([["x", ValueRoute.literal(1n)]]));
  (capture.evaluatedKeys as Set<string>).add("y");
  const envelope = capturedEnvelope(capture);
  expectError(() => encodeRecord(envelope), RangeError, "evaluated_keys must name captured routes");
});

test.each(["input_names", "variable_names"])("a capture's record requires its explicit inventory %s", (field) => {
  const wire = wireOf(capturedEnvelope(new ForwardCapture({}, ValueRoute.record([]))));
  delete wire.payload.forward_capture[field];
  expectError(() => decodeWire(wire), RangeError, `invalid ${CAPTURE_LABEL} fields; missing=["${field}"], extra=[]`);
});

test("a forward capture round trips without a memory scope", () => {
  const inputs = { value: [1n, null] };
  const nested = new ValueRoute("source", { source: new SourceRef("run", 1, ["nested"]) });
  const route = ValueRoute.record([["value", nested]]);
  const item = new ForwardCapture({ x: null }, ValueRoute.record([["x", ValueRoute.literal(null)]]), {
    evaluatedKeys: new Set(["x"]),
  });
  const capture = new ForwardCapture(inputs, route, {
    evaluatedKeys: new Set(["value"]),
    items: [[0, item]],
    scope: new ExecutionScope(["work"]),
    nodeData: { saved: [2n] },
  });
  inputs.value.push(99n);
  const evidence = new EvaluationEvidence(null, { value: [1n, null] }, null, null);
  const envelope = capturedEnvelope(capture, null, evidence);

  const decoded = decodeRecord(encodeRecord(envelope));

  expect(decoded).toEqual(envelope);
  const decodedCapture = (decoded.record as NodeSucceededRecord).forwardCapture!;
  expect(decodedCapture.boundInputs).toEqual({ value: [1n, null] });
  expect(() => (decodedCapture.boundInputs!.value as unknown[]).push(8n)).toThrow(TypeError);
  expect(() => (decodedCapture.nodeData!.saved as unknown[]).push(9n)).toThrow(TypeError);
  expect(capture.boundInputs).toEqual({ value: [1n, null] });
  expect(capture.nodeData).toEqual({ saved: [2n] });
  expect(() => {
    (capture as { scope: ExecutionScope | null }).scope = null;
  }).toThrow(TypeError);
  expect(decoded.schema).toBe(SCHEMA_VERSION);
  const old = wireOf(envelope);
  old.schema = 6n;
  expectError(() => decodeWire(old), RangeError, "unsupported run-record schema: 6");
});

test.each<[string, unknown, ErrorClass, string]>([
  [
    "source",
    { run_id: "run", execution_seq: true, path: [] },
    TypeError,
    "route source execution_seq must be an integer",
  ],
  [
    "source",
    { run_id: "run", execution_seq: 1n, path: [-1n] },
    RangeError,
    "route path components must be strings or non-negative integers",
  ],
  [
    "source",
    { run_id: "run", execution_seq: 1n, path: [false] },
    RangeError,
    "route path components must be strings or non-negative integers",
  ],
  ["source", { run_id: "", execution_seq: 1n, path: [] }, RangeError, "source run_id must be non-empty"],
  [
    "source",
    { run_id: "run", execution_seq: 1n, path: [], extra: 1n },
    RangeError,
    'invalid route source fields; missing=[], extra=["extra"]',
  ],
  [
    "children",
    [{}],
    RangeError,
    "invalid route fields; " +
      'missing=["children","expression","kind","names","path","result_type","sizes","snapshot","source","spans",' +
      '"value_json"], extra=[]',
  ],
  // Schema 10 routes have no `count`, so Python's two `count` rows are extra fields here.
  ["count", true, RangeError, 'invalid route fields; missing=[], extra=["count"]'],
  ["count", 2n, RangeError, 'invalid route fields; missing=[], extra=["count"]'],
  ["kind", "unknown", RangeError, "unknown route kind"],
  ["names", ["unused"], RangeError, "route source cannot carry names"],
  ["sizes", [2n], RangeError, "route source cannot carry sizes"],
  ["value_json", "Infinity", RangeError, "route source cannot carry value_json"],
  ["expression", 2n, TypeError, "route expression must be a string"],
  ["path", [0n], RangeError, "route source cannot carry path"],
  ["spans", [[0n, 1n]], RangeError, "route source cannot carry spans"],
])("a capture's record rejects a malformed nested route wire: %s %p", (field, value, cls, message) => {
  const capture = new ForwardCapture(
    { x: 1n },
    ValueRoute.record([["x", new ValueRoute("source", { source: new SourceRef("run", 1) })]]),
  );
  const wire = wireOf(capturedEnvelope(capture));
  wire.payload.forward_capture.route.children[0][field] = value;
  expectError(() => decodeWire(wire), cls, message);
});

test("a forward capture keeps the existing canonical depth range", () => {
  let value: unknown = 1n;
  for (let i = 0; i < 140; i++) value = [value];
  const [, inputs] = canonicalSeed({ x: value });
  const source = new ValueRoute("source", { source: new SourceRef("run", 0, ["x"]) });
  for (const route of [ValueRoute.literal(value), source]) {
    const capture = new ForwardCapture(inputs as Record<string, unknown>, ValueRoute.record([["x", route]]));
    const envelope = capturedEnvelope(capture);
    expect(decodeRecord(encodeRecord(envelope))).toEqual(envelope);
    expect(capture.boundInputs).toEqual(inputs as Record<string, unknown>);
  }
});

// Each registration's previous entry (or its absence), put back after the test, as Python's
// `monkeypatch.setitem` does.
const saved: [string, TemplateFn | undefined][] = [];

/** Register a stand-in builtin `name` running `call` for this test only. */
function standIn(name: string, call: (...args: Value[]) => unknown, params: TemplateParam[] = []): void {
  saved.push([name, TEMPLATE_FNS.get(name)]);
  registerTemplateFn(name, { params, call });
}

afterEach(() => {
  for (const [name, fn] of saved.reverse()) {
    if (fn === undefined) TEMPLATE_FNS.delete(name);
    else TEMPLATE_FNS.set(name, fn);
  }
  saved.length = 0;
});

/** A traced resolver over `a`, `b` and `n`, each its own run input. */
function resolveVariants(path: string): ResolvedValue {
  const values: Record<string, Value> = { a: [1n], b: [2n], n: 3n };
  if (!Object.hasOwn(values, path)) return new ResolvedValue(null, false);
  return new ResolvedValue(values[path]!, true, new SourceRef("run", 0, [path]));
}

test.each([
  "${a}",
  "${a | b}",
  "${[a, b]}",
  "${a + b}",
  "${n + 1}",
  "n=${n}",
  "${absent | absent}",
  "partial: ${partial()}",
  "score: ${score()}",
])("the codec preserves evaluated route variants: %s", (expression) => {
  // A file is the TS leaf JSON can't hold, so `partial()` gives a partial snapshot.
  standIn("partial", () => ({ kept: [1n], lost: new FileRef({ uri: "file:///tmp/x" }) }));
  standIn("score", () => Infinity);
  const traced = evalBinding(expression, resolveVariants, { trace: true });
  const capture = new ForwardCapture({ x: traced.value }, ValueRoute.record([["x", traced.route]]));
  const envelope = capturedEnvelope(capture);
  expect(decodeRecord(encodeRecord(envelope))).toEqual(envelope);
});

// TS builtins take named parameters, so `collect_args` takes `first`, `second` and `value`, each
// defaulting to 0, where Python's takes `*args, **kwargs`; its call log holds the bound values.
const OPAQUE_ARGUMENTS: [string, string[], Value[], bigint][] = [
  ["1", [""], [1n, 0n, 0n], 1n],
  ["1, 2", ["", ""], [1n, 2n, 0n], 3n],
  ["value=1", ["value"], [0n, 0n, 1n], 1n],
  ["1, value=2", ["", "value"], [1n, 0n, 2n], 3n],
];

test.each(OPAQUE_ARGUMENTS)("the codec preserves opaque argument labels: %s", (args, names, bound, result) => {
  const calls: Value[][] = [];
  standIn(
    "collect_args",
    (...values) => {
      calls.push(values);
      return (values as bigint[]).reduce((sum, item) => sum + item, 0n);
    },
    [
      { name: "first", default: 0 },
      { name: "second", default: 0 },
      { name: "value", default: 0 },
    ],
  );
  const traced = evalBinding("${collect_args(" + args + ")}", () => new ResolvedValue(null, false), {
    trace: true,
  });
  const capture = new ForwardCapture({ x: traced.value }, ValueRoute.record([["x", traced.route]]));
  const envelope = capturedEnvelope(capture, result);

  const decoded = decodeRecord(encodeRecord(envelope));

  expect(decoded).toEqual(envelope);
  const route = (decoded.record as NodeSucceededRecord).forwardCapture!.route.children[0]!;
  expect(route.kind).toBe("opaque");
  expect(route.names).toEqual(names);
  const stopped = routeFeedback(route, "Explain this result.", () => null);
  expect(stopped.contributions).toEqual([]);
  expect(stopped.stopped).toHaveLength(1);
  expect(stopped.stopped[0]!.context.at(-1)!.result).toBe(result);
  expect(stopped.stopped[0]!.context.at(-1)!.expression).toBe(traced.route.expression);
  expect(calls).toEqual([bound]);
});

test.each<[unknown[], string]>([
  [[], "opaque route names must match its children"],
  [["", "", ""], "opaque route names must match its children"],
  [[1n, "value"], "route names must be strings"],
])("the codec rejects misaligned or non-string opaque argument labels: %p", (names, message) => {
  const route = new ValueRoute("opaque", {
    valueJson: "3",
    names: ["", "value"],
    children: [ValueRoute.literal(1n), ValueRoute.literal(2n)],
  });
  const wire = wireOf(capturedEnvelope(new ForwardCapture({ x: 3n }, ValueRoute.record([["x", route]]))));
  wire.payload.forward_capture.route.children[0].names = names;
  expectError(() => decodeWire(wire), RangeError, message);
});

// TS-only (trace): the trace entries and `RetainedTrace`'s reads. Entries are installed by setting
// the trace's internal maps directly.

const RUN = "run";

/** A source route reading `path` of execution `seq`. */
function sourceRoute(seq: number, path: (string | number)[] = []): ValueRoute {
  return new ValueRoute("source", { source: new SourceRef(RUN, seq, path), expression: `\${source_${seq}}` });
}

/** The canonical typed wire text of `value`, as a source entry holds it. */
function typedJson(value: unknown): string {
  return logJson(canonicalTyped(value, null)[0]);
}

/**
 * The envelopes of a learnable run: RunStarted, whose pool holds START's `{x: 1n}` (source 0),
 * `NodeStarted("a")` (seq 1) and `NodeSucceeded("a")` with `2n` (seq 2), whose capture routes input
 * `v` to source 0 path `["x"]`.
 */
function learnableRun(): RecordEnvelope[] {
  const pool = new VariablePool("__start__");
  pool.set("__start__", { x: 1n });
  const started = new RunStartedRecord({
    flowFingerprint: "a".repeat(64),
    initialPool: encodePool(pool),
    startWiringOverride: {},
    effectiveLlmByStaticPath: {},
    numWorkers: 0,
    learnable: true,
  });
  const forwardCapture = new ForwardCapture({ v: 1n }, ValueRoute.record([["v", sourceRoute(0, ["x"])]]), {
    inputNames: ["v"],
  });
  return [
    new RecordEnvelope(SCHEMA_VERSION, RUN, 0, started),
    new RecordEnvelope(SCHEMA_VERSION, RUN, 1, new NodeStartedRecord("a")),
    new RecordEnvelope(
      SCHEMA_VERSION,
      RUN,
      2,
      new NodeSucceededRecord("a", "a", { kind: "int", value: 2n }, { forwardCapture }),
    ),
  ];
}

/** Install `entry` as an accepted source and as `target`'s current writer. */
function install(trace: RetainedTrace, target: string, entry: SourceEntry): void {
  trace._sources.set(sourceKey(entry.source!), entry);
  trace._currentWriters.set(target, entry);
}

/** Install the occurrence of `a` started at seq 1, accepted with `record` at seq 2. */
function installOccurrence(trace: RetainedTrace, record: OutcomeRecord): TraceOccurrence {
  const capture = record.forwardCapture!;
  const envelope = new RecordEnvelope(SCHEMA_VERSION, RUN, 2, record);
  const occurrence = new TraceOccurrence(new SourceRef(RUN, 1), "a", 2, envelope, capture.route);
  trace._occurrences.push(occurrence);
  trace.occurrenceBySource.set(sourceKey(occurrence.source), occurrence);
  return occurrence;
}

/**
 * A trace of `learnableRun()`: START's source 0, `a`'s output (source 1) and its occurrence, accepted
 * with `record` (by default the run's `NodeSucceeded`).
 */
function learnableTrace(record: OutcomeRecord = learnableRun()[2]!.record as NodeSucceededRecord): RetainedTrace {
  const trace = new RetainedTrace(RUN, "__start__", "end", ["x"]);
  install(trace, "__start__", new SourceEntry(new SourceRef(RUN, 0), 0, typedJson({ x: 1n })));
  install(trace, "a", new SourceEntry(new SourceRef(RUN, 1), 2, typedJson(2n)));
  installOccurrence(trace, record);
  return trace;
}

test("captureBindings joins a binding capture and its backward capture", () => {
  const file = new FileRef({ uri: "file:///tmp/x" });
  const fileForm = { uri: "file:///tmp/x", mime: null, name: null };
  const item = new BindingCapture({ f: file }, ValueRoute.record([["f", sourceRoute(0, ["f"])]]), new Set());
  const bindings = new BindingCapture(
    { x: 1n, f: file },
    ValueRoute.record([
      ["x", sourceRoute(0, ["x"])],
      ["f", sourceRoute(0, ["f"])],
    ]),
    new Set(["x"]),
    [[0, item]],
  );
  const scope = new ExecutionScope(["a"]);
  const backward = new BackwardCapture({
    inputNames: ["x", "f"],
    variableNames: ["v"],
    scope,
    nodeData: { k: 1n },
  });

  const capture = captureBindings(bindings, backward);

  expect(capture.boundInputs).toEqual({ f: fileForm, x: 1n });
  expect(capture.route).toBe(bindings.route);
  expect([...capture.evaluatedKeys]).toEqual(["x"]);
  expect(capture.scope).toBe(scope);
  expect(capture.nodeData).toEqual({ k: 1n });
  expect(capture.inputNames).toEqual(["x", "f"]);
  expect(capture.variableNames).toEqual(["v"]);
  expect(capture.items).toHaveLength(1);
  const [index, itemCapture] = capture.items[0]!;
  expect(index).toBe(0);
  expect(itemCapture.boundInputs).toEqual({ f: fileForm });
  expect(itemCapture.inputNames).toEqual([]);
  expect(itemCapture.variableNames).toEqual([]);
  expect(itemCapture.scope).toBeNull();
  expect(itemCapture.nodeData).toEqual({});
});

test("captureBindings requires exact carriers", () => {
  const bindings = new BindingCapture({ x: 1n }, ROUTE_X, new Set());
  expectError(
    () => captureBindings({ ...bindings } as BindingCapture, new BackwardCapture()),
    TypeError,
    "retained evaluation requires a BindingCapture",
  );
  expectError(
    () => captureBindings(bindings, { inputNames: [] } as unknown as BackwardCapture),
    TypeError,
    "retained evaluation requires a BackwardCapture",
  );
});

test("a source entry's value restores files through its type, its stored value doesn't", () => {
  const fileForm = { uri: "f", mime: null, name: null };
  const files = new Type({
    kind: ValueKind.OBJECT,
    fields: new Map([["f", Type.scalar(ValueKind.FILE)]]),
    required: new Set(),
  });
  const entry = new SourceEntry(null, 0, typedJson({ f: new FileRef({ uri: "f" }) }), files);

  const value = entry.value as { f: FileRef };
  expect(FileRef.isFileRef(value.f)).toBe(true);
  expect(value.f.uri).toBe("f");
  expect(entry.storedValue).toEqual({ f: fileForm });
  expect(new SourceEntry(null, 0, entry.valueJson).value).toEqual({ f: fileForm });
  expect(entry.value).not.toBe(entry.value);
  expect(entry.storedValue).not.toBe(entry.storedValue);
  expect(Object.isFrozen(entry)).toBe(true);
});

test("sameSourceEntry compares source keys, outcome sequences, value texts and files", () => {
  const files = Type.scalar(ValueKind.FILE);
  const valueJson = logJson({ kind: "int", value: 1n });
  const entry = (
    source: SourceRef | null = new SourceRef(RUN, 1),
    seq = 2,
    text = valueJson,
    type: Type | null = files,
  ) => new SourceEntry(source, seq, text, type);

  expect(sameSourceEntry(entry(), entry())).toBe(true);
  expect(sameSourceEntry(entry(), entry(new SourceRef(RUN, 2)))).toBe(false);
  expect(sameSourceEntry(entry(null), entry())).toBe(false);
  expect(sameSourceEntry(entry(), entry(null))).toBe(false);
  expect(sameSourceEntry(entry(), entry(undefined, 3))).toBe(false);
  expect(sameSourceEntry(entry(), entry(undefined, 2, logJson({ kind: "int", value: 2n })))).toBe(false);
  expect(sameSourceEntry(entry(undefined, 2, valueJson, null), entry())).toBe(false);
});

const TERMINAL_FAILURE = new Failure({ code: "node.execute", message: "Failed.", nodeId: "a" });

test.each<[string, () => unknown, ErrorClass, string]>([
  [
    "succeeded not a boolean",
    () => new TraceTerminal(1 as unknown as boolean, null, null, TERMINAL_FAILURE),
    TypeError,
    "terminal succeeded must be a bool",
  ],
  [
    "a source of another class",
    () => new TraceTerminal(false, {} as SourceRef, null, TERMINAL_FAILURE),
    TypeError,
    "source must be an exact SourceRef",
  ],
  [
    "a source with a path",
    () => new TraceTerminal(false, new SourceRef(RUN, 1, ["x"]), null, TERMINAL_FAILURE),
    RangeError,
    "terminal source must identify a root value",
  ],
  [
    "a success without output",
    () => new TraceTerminal(true, null, null, null),
    RangeError,
    "successful terminal requires output text and no failure",
  ],
  [
    "a success with a failure",
    () => new TraceTerminal(true, null, typedJson(1n), TERMINAL_FAILURE),
    RangeError,
    "successful terminal requires output text and no failure",
  ],
  [
    "a failure without a failure",
    () => new TraceTerminal(false, null, null, null),
    RangeError,
    "failed terminal requires a failure and no output",
  ],
  [
    "a failure with output",
    () => new TraceTerminal(false, null, typedJson(1n), TERMINAL_FAILURE),
    RangeError,
    "failed terminal requires a failure and no output",
  ],
  [
    "a failure of another class",
    () => new TraceTerminal(false, null, null, { ...TERMINAL_FAILURE } as Failure),
    RangeError,
    "failed terminal requires a failure and no output",
  ],
  [
    "output that isn't a typed wire",
    () => new TraceTerminal(true, null, "1", null),
    RangeError,
    "typed-value wire must be a JSON object",
  ],
])("a trace terminal refuses %s", (_name, build, cls, message) => {
  expectError(build, cls, message);
});

test("a trace terminal's output is fresh canonical JSON, or null when it failed", () => {
  const output = typedJson({ f: new FileRef({ uri: "f" }) });
  const succeeded = new TraceTerminal(true, new SourceRef(RUN, 3), output, null);
  expect(succeeded.output).toEqual({ f: { uri: "f", mime: null, name: null } });
  expect(succeeded.output).not.toBe(succeeded.output);
  const failed = new TraceTerminal(false, new SourceRef(RUN, 1), null, TERMINAL_FAILURE);
  expect(failed.output).toBeNull();
  expect(failed.failure).not.toBe(TERMINAL_FAILURE);
  expect(failed.failure).toEqual(TERMINAL_FAILURE);
  expect(Object.isFrozen(failed)).toBe(true);
});

test("checkMemoryScope returns null or an exact MemoryScope", () => {
  const scope = new MemoryScope(["a"], { memories: ["notes"] });
  expect(checkMemoryScope(null)).toBeNull();
  expect(checkMemoryScope(scope)).toBe(scope);
  expectError(
    () => checkMemoryScope({ ...scope } as MemoryScope),
    TypeError,
    "historical memory scope must be a MemoryScope or null",
  );
});

test("a trace occurrence keeps a decoded copy of its envelope", () => {
  const keys = new Set<string>();
  const capture = new ForwardCapture({ v: 1n }, ValueRoute.record([["v", sourceRoute(0, ["x"])]]), {
    evaluatedKeys: keys,
  });
  const record = new NodeSucceededRecord("a", "a", { kind: "int", value: 2n }, { forwardCapture: capture });
  const envelope = new RecordEnvelope(SCHEMA_VERSION, RUN, 2, record);
  const hidden = new Set(["v"]);
  const scope = new MemoryScope(["a"], { memories: ["notes"] });

  const occurrence = new TraceOccurrence(new SourceRef(RUN, 1), "a", 2, envelope, capture.route, scope, hidden);
  const recordJson = encodeRecord(envelope);
  (capture.evaluatedKeys as Set<string>).add("v");
  hidden.add("w");

  expect(occurrence.recordJson).toBe(recordJson);
  expect(occurrence.envelope).not.toBe(envelope);
  expect(occurrence.envelope.record).toBe(occurrence.record);
  expect([...(occurrence.record as NodeSucceededRecord).forwardCapture!.evaluatedKeys]).toEqual([]);
  expect(occurrence.memoryScope).toBe(scope);
  expect([...occurrence.hiddenInputs]).toEqual(["v"]);
  expect(Object.isFrozen(occurrence)).toBe(true);
  expectError(
    () => new TraceOccurrence(new SourceRef(RUN, 1), "a", 2, envelope, capture.route, {} as MemoryScope),
    TypeError,
    "historical memory scope must be a MemoryScope or null",
  );
});

test("selectSeedRoute selects a seed field by name, or by item index and name", () => {
  const field = sourceRoute(0, ["x"]);
  const joined = sourceRoute(0, ["y"]);
  const itemField = sourceRoute(0, ["z"]);
  const item = new ForwardCapture({ n: 1n }, ValueRoute.record([["n", itemField]]));
  const capture = new ForwardCapture({ v: 1n }, ValueRoute.record([["v", field]]), { items: [[2, item]] });

  expect(selectSeedRoute(capture, new SeedConnection("child", "v", ["v"]))).toBe(field);
  const inputRoute = ValueRoute.record([["v", joined]]);
  expect(selectSeedRoute(capture, new SeedConnection("child", "v", ["v"]), inputRoute)).toBe(joined);
  expect(selectSeedRoute(capture, new SeedConnection("child", "n", [2, "n"]), inputRoute)).toBe(itemField);
  expect(selectSeedRoute(capture, new SeedConnection("child", "n", [3, "n"]))).toBeNull();
  expect(selectSeedRoute(capture, new SeedConnection("child", "w", ["w"]))).toBeNull();
});

test("a retained trace starts empty and checks its root input names", () => {
  const trace = new RetainedTrace(RUN, "__start__");
  expect(trace.endId).toBeNull();
  expect(trace.rootInputNames).toEqual([]);
  expect(trace.rootStartSource).toBeNull();
  expect(trace.terminal).toBeNull();
  expect(trace.nextSequence).toBe(0);
  expect(trace.occurrences).toEqual([]);
  expect(trace.boundaries).toEqual([]);
  expect(trace.sources.size).toBe(0);
  expectError(
    () => new RetainedTrace(RUN, "__start__", null, ["x", "x"]),
    RangeError,
    "root_input_names must contain unique names",
  );
});

test("lookupSource reads a root value or a path inside it as fresh canonical JSON", () => {
  const trace = new RetainedTrace(RUN, "__start__");
  const file = new FileRef({ uri: "f" });
  install(trace, "b", new SourceEntry(new SourceRef(RUN, 3), 4, typedJson({ items: [10n, { f: file }] })));

  expect(trace.lookupSource(new SourceRef(RUN, 3))).toEqual({
    items: [10n, { f: { uri: "f", mime: null, name: null } }],
  });
  expect(trace.lookupSource(new SourceRef(RUN, 3))).not.toBe(trace.lookupSource(new SourceRef(RUN, 3)));
  expect(trace.lookupSource(new SourceRef(RUN, 3, ["items", 1, "f", "uri"]))).toBe("f");
  expect(trace.lookupSource(new SourceRef(RUN, 3, ["items", 0]))).toBe(10n);
  const absent = "source path is absent from its accepted value";
  expectError(() => trace.lookupSource(new SourceRef(RUN, 3, ["missing"])), RangeError, absent);
  expectError(() => trace.lookupSource(new SourceRef(RUN, 3, ["items", 2])), RangeError, absent);
  expectError(() => trace.lookupSource(new SourceRef(RUN, 3, ["items", -1])), RangeError, absent);
  expectError(() => trace.lookupSource(new SourceRef(RUN, 3, ["items", "0"])), RangeError, absent);
  expectError(() => trace.lookupSource(new SourceRef(RUN, 3, [0])), RangeError, absent);
  expectError(() => trace.lookupSource(new SourceRef(RUN, 3, ["items", 0, "x"])), RangeError, absent);
  expectError(
    () => trace.lookupSource(new SourceRef(RUN, 5, ["x"])),
    RangeError,
    `no accepted source for ${sourceKey(new SourceRef(RUN, 5))}`,
  );
});

test("resolve pairs each read's value, presence and producer", () => {
  const trace = learnableTrace();
  install(trace, "b", new SourceEntry(new SourceRef(RUN, 3), 4, typedJson({ f: "text" })));
  trace.system.set("k", typedJson("ambient"));

  expect(trace.resolve("input.x")).toEqual(new ResolvedValue(1n, true, new SourceRef(RUN, 0, ["x"])));
  expect(trace.resolve(" b.output.f ")).toEqual(new ResolvedValue("text", true, new SourceRef(RUN, 3, ["f"])));
  expect(trace.resolve("b.output")).toEqual(new ResolvedValue({ f: "text" }, true, new SourceRef(RUN, 3)));
  expect(trace.resolve("system.k")).toEqual(new ResolvedValue("ambient", true));
  expect(trace.resolve("system.absent")).toEqual(new ResolvedValue(null, false));
  expect(trace.resolve("unknown.output")).toEqual(new ResolvedValue(null, false));
  expect(trace.resolve("input.y")).toEqual(new ResolvedValue(null, false));
  expect(trace.resolve("b.output.g")).toEqual(new ResolvedValue(null, false));
  expectError(() => trace.resolve("input..x"), ExpressionError, "invalid variable path: '${input..x}'");
});

test("occurrence finds an accepted occurrence by its start", () => {
  const trace = learnableTrace();
  expect(trace.occurrence(new SourceRef(RUN, 1)).nodeId).toBe("a");
  expect(trace.occurrences.map((item) => item.nodeId)).toEqual(["a"]);
  expectError(
    () => trace.occurrence(new SourceRef(RUN, 0)),
    RangeError,
    `no accepted occurrence for ${sourceKey(new SourceRef(RUN, 0))}`,
  );
});

test("backwardContext adds the evaluation evidence and reads a success's output", () => {
  const evidence = new EvaluationEvidence(null, { v: 1n }, "System.", "Prompt.");
  const { value, forwardCapture } = learnableRun()[2]!.record as NodeSucceededRecord;
  const trace = learnableTrace(new NodeSucceededRecord("a", "a", value, { evidence, forwardCapture }));
  const context = trace.backwardContext(new SourceRef(RUN, 1), { downstreamContext: [{ index: 0n }] });

  expect(context.inputs).toEqual({ v: 1n });
  expect(context.inputNames).toEqual(["v"]);
  expect(context.output).toBe(2n);
  expect(context.failure).toBeNull();
  expect(context.runId).toBe(RUN);
  expect(context.nodeId).toBe("a");
  expect(context.nodeStartedSeq).toBe(1);
  expect(context.nodeData).toEqual({ _evaluation_evidence: evidencePayload(evidence) });
  expect((context.nodeData["_evaluation_evidence"] as { prompt: string }).prompt).toBe("Prompt.");
  expect(context.downstreamContext).toEqual([{ index: 0n }]);
});

test("backwardContext refuses captured node data using the reserved evidence key", () => {
  const capture = new ForwardCapture({ v: 1n }, ValueRoute.record([["v", ValueRoute.literal(1n)]]), {
    nodeData: { _evaluation_evidence: "forged" },
  });
  const trace = learnableTrace(new NodeRoutedRecord("a", "next", { forwardCapture: capture }));
  expectError(
    () => trace.backwardContext(new SourceRef(RUN, 1)),
    RangeError,
    "captured node data uses reserved _evaluation_evidence",
  );
});

test("backwardContext takes a routed terminal failure and reads output only for a success", () => {
  const capture = new ForwardCapture({ v: 1n }, ValueRoute.record([["v", ValueRoute.literal(1n)]]), {
    nodeData: { k: "kept" },
  });
  const failure = new Failure({ code: "graph.terminal_unreachable", message: "Unreachable.", nodeId: "a" });
  // Source 1 holds a value, so the null output comes from the record's class.
  const record = new NodeRoutedRecord("a", "next", { terminalFailure: failure, forwardCapture: capture });
  const trace = learnableTrace(record);

  const context = trace.backwardContext(new SourceRef(RUN, 1));

  expect(context.failure).toEqual(failure);
  expect(context.output).toBeNull();
  expect(context.nodeData).toEqual({ k: "kept", _evaluation_evidence: null });
});

// TS-only (trace): trace deltas and `prepareTrace`, driven over `learnableRun()`.

/** A CODE node with a settable scope, memory scope and hidden inputs, recording its backward calls. */
class TraceNode extends Node {
  readonly kind = NodeKind.CODE;
  /** The inputs `modelHiddenInputs` reports. */
  hidden: ReadonlySet<string> = new Set();
  /** Each backward call's context and feedback, in call order. */
  readonly calls: [BackwardContext, readonly FeedbackValue[]][] = [];

  override modelHiddenInputs(): ReadonlySet<string> {
    return this.hidden;
  }

  override backward(context: BackwardContext, feedback: readonly FeedbackValue[]): BackwardResult {
    this.calls.push([context, feedback]);
    return new BackwardResult();
  }
}

/** The root metadata a learnable RunStarted is prepared with. */
const ROOT: PrepareTraceOptions = { rootEndId: "end", rootInputNames: ["x"] };

/** `learnableRun()`'s capture: input `v` routed to source 0 path `["x"]`. */
function captureV(options: ConstructorParameters<typeof ForwardCapture>[2] = {}): ForwardCapture {
  return new ForwardCapture({ v: 1n }, ValueRoute.record([["v", sourceRoute(0, ["x"])]]), {
    inputNames: ["v"],
    ...options,
  });
}

/** An envelope of `record` at `seq`. */
function at(seq: number, record: RunRecord): RecordEnvelope {
  return new RecordEnvelope(SCHEMA_VERSION, RUN, seq, record);
}

/** A success of `nodeId` writing the int `value` under `target`, captured by `capture`. */
function success(nodeId: string, target = nodeId, value = 2n, capture: ForwardCapture | null = captureV()) {
  return new NodeSucceededRecord(nodeId, target, { kind: "int", value }, { forwardCapture: capture });
}

/** Prepare `envelope`, check the delta's origin and apply it; the trace unchanged when there is no delta. */
function step(
  trace: RetainedTrace | null,
  envelope: RecordEnvelope,
  node: Node | null = null,
  options: PrepareTraceOptions = {},
): RetainedTrace | null {
  const delta = prepareTrace(trace, envelope, node, options);
  if (delta === null) return trace;
  delta.validateOrigin(trace);
  return delta.apply();
}

/** The trace after `learnableRun()`'s RunStarted, and after `NodeStarted("a")` too when `started`. */
function startedTrace(started = false): RetainedTrace {
  const [run, start] = learnableRun();
  const trace = step(null, run!, null, ROOT)!;
  return started ? step(trace, start!)! : trace;
}

/** The trace after all of `learnableRun()`, with `node` as `a`. */
function drivenTrace(node: TraceNode = new TraceNode("a"), options: PrepareTraceOptions = {}): RetainedTrace {
  return step(startedTrace(true), learnableRun()[2]!, node, options)!;
}

/** Start `node` at `seq` and accept `record` for it at `seq + 1`. */
function runNode(
  trace: RetainedTrace,
  seq: number,
  node: TraceNode,
  record: RunRecord,
  options: PrepareTraceOptions = {},
): RetainedTrace {
  return step(step(trace, at(seq, new NodeStartedRecord(node.id)))!, at(seq + 1, record), node, options)!;
}

test("prepareTrace and apply build the trace record by record", () => {
  const [run, start, succeeded] = learnableRun();
  const node = new TraceNode("a");
  node.hidden = new Set(["v"]);
  node.memoryScope = new MemoryScope(["a"], { memories: ["notes"] });

  const trace = step(null, run!, null, ROOT)!;
  expect(trace.runId).toBe(RUN);
  expect(trace.startId).toBe("__start__");
  expect(trace.endId).toBe("end");
  expect(trace.rootInputNames).toEqual(["x"]);
  expect(trace.nextSequence).toBe(1);
  const startEntry = new SourceEntry(new SourceRef(RUN, 0), 0, typedJson({ x: 1n }));
  expect(sameSourceEntry(trace.currentWriters.get("__start__")!, startEntry)).toBe(true);
  expect(trace.sources.get(sourceKey(new SourceRef(RUN, 0)))).toBe(trace.currentWriters.get("__start__"));
  expect(trace.rootWriter).toBeNull();
  expect(trace.started.size).toBe(0);

  expect(step(trace, start!)).toBe(trace);
  expect([...trace.started]).toEqual([["a", 1]]);
  expect(trace.nextSequence).toBe(2);

  const files = Type.scalar(ValueKind.FILE);
  expect(step(trace, succeeded!, node, { writerFiles: files })).toBe(trace);
  const source = new SourceRef(RUN, 1);
  const occurrence = trace.occurrences[0]!;
  expect(occurrence.source).toEqual(source);
  expect(occurrence.nodeId).toBe("a");
  expect(occurrence.outcomeSeq).toBe(2);
  expect(occurrence.recordJson).toBe(encodeRecord(succeeded!));
  expect(occurrence.inputRoute).toBe((succeeded!.record as NodeSucceededRecord).forwardCapture!.route);
  expect(occurrence.memoryScope).toBe(node.memoryScope);
  expect([...occurrence.hiddenInputs]).toEqual(["v"]);
  expect(trace.occurrence(source)).toBe(occurrence);
  expect(trace.started.size).toBe(0);
  expect(trace.currentWriters.get("a")!.files).toBe(files);
  const writer = new SourceEntry(source, 2, logJson({ kind: "int", value: 2n }), files);
  expect(sameSourceEntry(trace.currentWriters.get("a")!, writer)).toBe(true);
  expect(trace.sources.get(sourceKey(source))).toBe(trace.currentWriters.get("a"));
  expect(trace.rootStartSource).toBeNull();
  expect(trace.rootWriter).toBeNull();
  expect(trace.terminal).toBeNull();
  expect(trace.nextSequence).toBe(3);

  const handler = trace.backwardHandlers.get(sourceKey(source))!;
  const context = trace.backwardContext(source);
  handler(context, ["Review."]);
  expect(node.calls).toEqual([[context, ["Review."]]]);
});

test("a learnable RunStarted installs every pool entry, the root writer and the system values", () => {
  const pool = new VariablePool("__start__");
  pool.set("__start__", { x: 1n });
  pool.set("end", "done");
  pool.addSystem("k", "ambient");
  const record = new RunStartedRecord({
    flowFingerprint: "a".repeat(64),
    initialPool: encodePool(pool),
    startWiringOverride: {},
    effectiveLlmByStaticPath: {},
    numWorkers: 0,
    learnable: true,
  });

  const trace = step(null, at(0, record), null, ROOT)!;

  const end = trace.currentWriters.get("end")!;
  expect(sameSourceEntry(end, new SourceEntry(null, 0, typedJson("done")))).toBe(true);
  expect(trace.rootWriter).toBe(end);
  expect([...trace.sources.keys()]).toEqual([sourceKey(new SourceRef(RUN, 0))]);
  expect(trace.resolve("system.k")).toEqual(new ResolvedValue("ambient", true));
});

test("prepareTrace returns null for a run that doesn't learn and for records before its RunStarted", () => {
  const [run, start, succeeded] = learnableRun();
  const { flowFingerprint, initialPool } = run!.record as RunStartedRecord;
  const plain = new RunStartedRecord({
    flowFingerprint,
    initialPool,
    startWiringOverride: {},
    effectiveLlmByStaticPath: {},
    numWorkers: 0,
  });
  expect(prepareTrace(null, at(0, plain), null, ROOT)).toBeNull();
  expect(prepareTrace(null, start!, null)).toBeNull();
  expect(prepareTrace(null, at(2, success("a", "a", 2n, null)), new TraceNode("a"))).toBeNull();
  expect(prepareTrace(startedTrace(true), succeeded!, new TraceNode("a"))).not.toBeNull();
});

test("terminals take the root writer, the failed occurrence or the first causal route", () => {
  const output = typedJson(2n);
  const succeeded = drivenTrace();
  const root = runNode(succeeded, 3, new TraceNode("b"), success("b", "end", 2n), {
    terminalSucceeded: true,
    terminalOutputJson: output,
  });
  const writer = root.currentWriters.get("end")!;
  expect(root.rootWriter).toBe(writer);
  expect(root.terminal!.succeeded).toBe(true);
  expect(root.terminal!.source).toEqual(new SourceRef(RUN, 3));
  expect(root.terminal!.outputJson).toBe(output);

  const failure = new Failure({ code: "node.execute", message: "Failed.", nodeId: "b" });
  const failedRecord = new NodeFailedRecord(failure, { forwardCapture: captureV() });
  const failed = runNode(drivenTrace(), 3, new TraceNode("b"), failedRecord, {
    terminalSucceeded: false,
    terminalFailure: failure,
  });
  expect(failed.terminal!.source).toEqual(new SourceRef(RUN, 3));
  expect(failed.terminal!.failure).toEqual(failure);

  const unreachable = new Failure({ code: "graph.terminal_unreachable", message: "Unreachable.", nodeId: "r" });
  const routedRecord = new NodeRoutedRecord("r", "next", { terminalFailure: unreachable, forwardCapture: captureV() });
  const routed = runNode(drivenTrace(), 3, new TraceNode("r"), routedRecord);
  expect(routed.causalRouteSource).toEqual(new SourceRef(RUN, 3));
  expect(routed.terminal).toBeNull();
  const second = new NodeRoutedRecord("q", "next", { terminalFailure: unreachable, forwardCapture: captureV() });
  const later = runNode(routed, 5, new TraceNode("q"), second, {
    terminalSucceeded: false,
    terminalFailure: unreachable,
  });
  expect(later.causalRouteSource).toEqual(new SourceRef(RUN, 3));
  expect(later.terminal!.source).toEqual(new SourceRef(RUN, 3));
});

test("the root START's first outcome sets the root start source", () => {
  const node = new TraceNode("__start__");
  const trace = runNode(startedTrace(), 1, node, success("__start__", "__start__", 2n));
  expect(trace.rootStartSource).toEqual(new SourceRef(RUN, 1));
  expect(trace.currentWriters.get("__start__")!.outcomeSeq).toBe(2);
});

/** A trace after `a` started, with `edit` applied to it, for one failing preparation. */
function editedTrace(edit: (trace: RetainedTrace) => void): RetainedTrace {
  const trace = startedTrace(true);
  edit(trace);
  return trace;
}

/** `a` with the execution scope `scope`. */
function scopedNode(scope: ExecutionScope | null): TraceNode {
  const node = new TraceNode("a");
  node.executionScope = scope;
  return node;
}

/** Prepare `a`'s success at seq 2 to `target` over `trace` with `capture`, `node` and `options`. */
function prepareA({
  capture = captureV(),
  options = {},
  node = new TraceNode("a"),
  trace = startedTrace(true),
  target = "a",
}: {
  capture?: ForwardCapture | null;
  options?: PrepareTraceOptions;
  node?: Node | null;
  trace?: RetainedTrace;
  target?: string;
} = {}): () => unknown {
  return () => prepareTrace(trace, at(2, success("a", target, 2n, capture)), node, options);
}

const SCOPE_A = new ExecutionScope(["a"]);
const TYPED_TWO = typedJson(2n);

test.each<[string, () => unknown, ErrorClass, string]>([
  [
    "a learnable RunStarted without a root end",
    () => prepareTrace(null, learnableRun()[0]!, null, { rootInputNames: ["x"] }),
    RangeError,
    "retained RunStarted requires verified root metadata",
  ],
  [
    "a learnable RunStarted without root input names",
    () => prepareTrace(null, learnableRun()[0]!, null, { rootEndId: "end" }),
    RangeError,
    "retained RunStarted requires verified root metadata",
  ],
  [
    "a capture with no trace",
    () => prepareTrace(null, learnableRun()[2]!, new TraceNode("a")),
    RangeError,
    "forward capture requires recorded learning",
  ],
  [
    "a record at another sequence",
    () => prepareTrace(startedTrace(), learnableRun()[2]!, new TraceNode("a")),
    RangeError,
    "trace disagrees with accepted record coordinates",
  ],
  [
    "a record of another run",
    () =>
      prepareTrace(
        startedTrace(),
        new RecordEnvelope(SCHEMA_VERSION, "other", 1, new NodeStartedRecord("a")),
        null,
      ),
    RangeError,
    "trace disagrees with accepted record coordinates",
  ],
  [
    "terminal output without a disposition",
    () => prepareTrace(startedTrace(), learnableRun()[1]!, null, { terminalOutputJson: TYPED_TWO }),
    RangeError,
    "terminal payload requires a terminal disposition",
  ],
  [
    "a terminal failure without a disposition",
    () => prepareTrace(startedTrace(), learnableRun()[1]!, null, { terminalFailure: TERMINAL_FAILURE }),
    RangeError,
    "terminal payload requires a terminal disposition",
  ],
  [
    "a resume naming other starts",
    () => prepareTrace(startedTrace(true), at(2, new RunResumedRecord(["b"], 0)), null),
    RangeError,
    "trace unfinished starts disagree with retry frontier",
  ],
  [
    "an outcome without a capture",
    prepareA({ capture: null }),
    RangeError,
    "accepted outcome requires forward capture while learning",
  ],
  [
    "an outcome without a node",
    prepareA({ node: null }),
    RangeError,
    "trace outcome requires its accepted start",
  ],
  [
    "an outcome of a node that didn't start",
    prepareA({ node: new TraceNode("b") }),
    RangeError,
    "trace outcome requires its accepted start",
  ],
  [
    "a scoped capture on a node without a scope",
    prepareA({ capture: captureV({ scope: SCOPE_A }), node: scopedNode(null) }),
    RangeError,
    "forward capture scope does not match the executing node",
  ],
  [
    "a capture without a scope on a scoped node",
    prepareA({ node: scopedNode(SCOPE_A) }),
    RangeError,
    "forward capture scope does not match the executing node",
  ],
  [
    "a capture and a node with different scopes",
    prepareA({ capture: captureV({ scope: SCOPE_A }), node: scopedNode(new ExecutionScope(["b"])) }),
    RangeError,
    "forward capture scope does not match the executing node",
  ],
  [
    "a route to an unaccepted source",
    prepareA({ capture: new ForwardCapture({ v: 1n }, ValueRoute.record([["v", sourceRoute(5)]])) }),
    RangeError,
    "route source must name an earlier accepted producer",
  ],
  [
    "a route to a source accepted at this record",
    prepareA({
      trace: editedTrace((trace) =>
        install(trace, "b", new SourceEntry(new SourceRef(RUN, 0), 2, typedJson({ x: 1n }))),
      ),
    }),
    RangeError,
    "route source must name an earlier accepted producer",
  ],
  [
    "an item route to an unaccepted source",
    prepareA({
      capture: captureV({ items: [[0, new ForwardCapture({ n: 1n }, ValueRoute.record([["n", sourceRoute(5)]]))]] }),
    }),
    RangeError,
    "route source must name an earlier accepted producer",
  ],
  [
    "a seed route to an unaccepted source",
    prepareA({
      trace: editedTrace((trace) =>
        trace._pendingConnections.set("a", new SeedRoutes(new SourceRef(RUN, 0), [["v", sourceRoute(5)]])),
      ),
    }),
    RangeError,
    "route source must name an earlier accepted producer",
  ],
  [
    "a seed field that wasn't captured",
    prepareA({ options: { inputConnections: [new SeedConnection("child", "w", ["w"])] } }),
    RangeError,
    "seed connection field or item was not captured",
  ],
  [
    "a seed item that wasn't captured",
    prepareA({ options: { inputConnections: [new SeedConnection("child", "n", [0, "n"])] } }),
    RangeError,
    "seed connection field or item was not captured",
  ],
  [
    "a seed target that already has an owner",
    prepareA({
      options: { inputConnections: [new SeedConnection("child", "p", ["v"])] },
      trace: editedTrace((trace) => trace._pendingConnections.set("child", new SeedRoutes(new SourceRef(RUN, 0), []))),
    }),
    RangeError,
    "seed connection target already has an accepted owner",
  ],
  [
    "a child boundary endpoint that already has an owner",
    prepareA({
      options: { childBoundaries: [new ChildFlowBoundary("cs", "ce")] },
      trace: editedTrace((trace) => trace.currentBoundaries.set("ce", [0, "endSource"])),
    }),
    RangeError,
    "child boundary target already has an accepted owner",
  ],
  [
    "a node whose backward isn't a function",
    prepareA({ node: Object.assign(new TraceNode("a"), { backward: "not callable" }) }),
    TypeError,
    "node backward handler must be callable",
  ],
  [
    "terminal output without a root writer",
    prepareA({ options: { terminalSucceeded: true, terminalOutputJson: TYPED_TWO } }),
    RangeError,
    "terminal output disagrees with the accepted root writer",
  ],
  [
    "terminal output other than the root writer's",
    prepareA({ options: { terminalSucceeded: true, terminalOutputJson: typedJson(3n) }, target: "end" }),
    RangeError,
    "terminal output disagrees with the accepted root writer",
  ],
  [
    "a successful terminal without output",
    prepareA({ options: { terminalSucceeded: true }, target: "end" }),
    RangeError,
    "terminal output disagrees with the accepted root writer",
  ],
  [
    "a terminal failure without a causal occurrence",
    prepareA({ options: { terminalSucceeded: false, terminalFailure: TERMINAL_FAILURE } }),
    RangeError,
    "accepted failure requires an accepted causal occurrence",
  ],
])("prepareTrace refuses %s", (_label, prepare, cls, message) => {
  expectError(prepare, cls, message);
});

test("prepareTrace keeps the lookup error as the cause of an absent source path", () => {
  const prepare = prepareA({
    capture: new ForwardCapture({ v: 1n }, ValueRoute.record([["v", sourceRoute(0, ["y"])]])),
  });
  let caught: Error | undefined;
  try {
    prepare();
  } catch (error) {
    caught = error as Error;
  }
  expect(caught!.constructor).toBe(RangeError);
  expect(caught!.message).toBe("route source path is absent from its accepted value");
  const cause = caught!.cause as Error;
  expect(cause.constructor).toBe(RangeError);
  expect(cause.message).toBe("source path is absent from its accepted value");
});

test("a trace delta's terminal compare reads the output text canonically", () => {
  // The same int written with other spacing is the same decoded typed value.
  const trace = step(startedTrace(true), at(2, success("a", "end")), new TraceNode("a"), {
    terminalSucceeded: true,
    terminalOutputJson: '{ "value": 2, "kind": "int" }',
  })!;
  expect(trace.terminal!.source).toEqual(new SourceRef(RUN, 1));
});

test.each<[string, (trace: RetainedTrace) => RetainedTrace | null, string]>([
  ["another trace", () => startedTrace(true), "trace transition requires its originating trace"],
  ["no trace", () => null, "trace transition requires its originating trace"],
  [
    "a stale sequence",
    (trace) => {
      trace.nextSequence += 1;
      return trace;
    },
    "trace transition has a stale sequence",
  ],
])("validateOrigin refuses %s", (_label, target, message) => {
  const trace = startedTrace(true);
  const delta = prepareTrace(trace, learnableRun()[2]!, new TraceNode("a"))!;
  expectError(() => delta.validateOrigin(target(trace)), RangeError, message);
});

test("validateOrigin refuses an occurrence that isn't exactly a TraceOccurrence", () => {
  const trace = startedTrace(true);
  const { occurrence } = prepareTrace(trace, learnableRun()[2]!, new TraceNode("a"))!;
  // Reads every field of the real occurrence through its prototype.
  const forged = new TraceDelta(trace, 2, { occurrence: Object.create(occurrence) as TraceOccurrence });
  expectError(() => forged.validateOrigin(trace), TypeError, "trace delta requires an exact occurrence");
});

test("validateOrigin refuses a consumed connection that is no longer pending", () => {
  const trace = drivenTrace(new TraceNode("a"), { inputConnections: [new SeedConnection("c", "p", ["v"])] });
  step(trace, at(3, new NodeStartedRecord("c")));
  const capture = new ForwardCapture({ p: 1n }, ValueRoute.record([["p", ValueRoute.literal(1n)]]));
  const delta = prepareTrace(trace, at(4, success("c", "c", 1n, capture)), new TraceNode("c"))!;
  const { owner, fields } = trace.pendingConnections.get("c")!;
  trace._pendingConnections.set("c", new SeedRoutes(owner, fields));
  expectError(() => delta.validateOrigin(trace), RangeError, "seed connection no longer has its prepared owner");
  trace._pendingConnections.delete("c");
  expectError(() => delta.validateOrigin(trace), RangeError, "seed connection no longer has its prepared owner");
});

test("seed connections install sorted by target and input, and the child start consumes them", () => {
  const connections = [
    new SeedConnection("c", "q", ["v"]),
    new SeedConnection("b", "p", ["v"]),
    new SeedConnection("c", "p", ["v"]),
  ];
  const trace = drivenTrace(new TraceNode("a"), { inputConnections: connections });
  const seeded = (learnableRun()[2]!.record as NodeSucceededRecord).forwardCapture!.route.children[0]!;
  expect([...trace.pendingConnections.keys()]).toEqual(["b", "c"]);
  const c = trace.pendingConnections.get("c")!;
  expect(c.owner).toEqual(new SourceRef(RUN, 1));
  expect(c.fields.map(([name]) => name)).toEqual(["p", "q"]);
  expect(c.fields[0]![1].source).toEqual(seeded.source);

  const literal = ValueRoute.literal(1n);
  const evaluated = ValueRoute.literal(1n);
  const capture = new ForwardCapture(
    { p: 1n, q: 1n },
    ValueRoute.record([
      ["p", literal],
      ["q", evaluated],
    ]),
    { evaluatedKeys: new Set(["q"]) },
  );
  runNode(trace, 3, new TraceNode("c"), success("c", "c", 1n, capture));

  const route = trace.occurrence(new SourceRef(RUN, 3)).inputRoute;
  expect(route.names).toEqual(["p", "q"]);
  expect(route.children[0]).toBe(c.fields[0]![1]);
  expect(route.children[1]).toBe(evaluated);
  expect([...trace.pendingConnections.keys()]).toEqual(["b"]);
});

test("child boundaries install sorted and take their endpoints' accepted starts", () => {
  const boundaries = [new ChildFlowBoundary("s2", "e2"), new ChildFlowBoundary("s1", "e1")];
  const trace = drivenTrace(new TraceNode("a"), { childBoundaries: boundaries });
  const installed = trace.boundaries;
  expect(installed.map((item) => [item.startId, item.endId])).toEqual([
    ["s1", "e1"],
    ["s2", "e2"],
  ]);
  expect(installed[0]!.owner).toEqual(new SourceRef(RUN, 1));
  expect(installed[0]!.startSource).toBeNull();

  runNode(trace, 3, new TraceNode("s1"), success("s1"));
  expect(trace.boundaries[0]!.startSource).toEqual(new SourceRef(RUN, 3));
  expect(trace.boundaries[0]!.endSource).toBeNull();
  expect(trace.currentBoundaries.has("s1")).toBe(false);
  runNode(trace, 5, new TraceNode("e1"), success("e1"));
  expect(trace.boundaries[0]!.startSource).toEqual(new SourceRef(RUN, 3));
  expect(trace.boundaries[0]!.endSource).toEqual(new SourceRef(RUN, 5));
  expect(trace.boundaries[0]!.owner).toEqual(new SourceRef(RUN, 1));
  expect(installed[0]!.startSource).toBeNull();
  expect(trace.boundaries[1]).toBe(installed[1]);
});

test("an expansion prunes current names only, and other outcomes ignore retired ids", () => {
  const trace = drivenTrace(new TraceNode("a"), {
    inputConnections: [new SeedConnection("b", "p", ["v"])],
    childBoundaries: [new ChildFlowBoundary("s1", "e1")],
    retiredIds: ["a"],
  });
  expect(trace.currentWriters.has("a")).toBe(true);

  const expanded = new NodeExpandedRecord("g", { v: 1n }, ["a", "b"], "b".repeat(64), { forwardCapture: captureV() });
  runNode(trace, 3, new TraceNode("g"), expanded, { retiredIds: ["a", "b", "s1"] });

  expect(trace.currentWriters.has("a")).toBe(false);
  expect(trace.pendingConnections.has("b")).toBe(false);
  expect(trace.currentBoundaries.has("s1")).toBe(false);
  expect(trace.currentBoundaries.has("e1")).toBe(true);
  expect(trace.sources.has(sourceKey(new SourceRef(RUN, 1)))).toBe(true);
  expect(trace.occurrences.map((item) => item.nodeId)).toEqual(["a", "g"]);
  expect(trace.boundaries).toHaveLength(1);
  expect(trace.resolve("a.output")).toEqual(new ResolvedValue(null, false));
});

test("a resumed record clears the unfinished starts", () => {
  const trace = startedTrace(true);
  step(trace, at(2, new RunResumedRecord(["a"], 0)));
  expect(trace.started.size).toBe(0);
  expect(trace.nextSequence).toBe(3);
});

test("an applied delta never changes an earlier occurrence", () => {
  const trace = drivenTrace();
  const occurrences = trace.occurrences;
  const [occurrence] = occurrences;
  const { recordJson, inputRoute } = occurrence!;

  runNode(trace, 3, new TraceNode("b"), success("b", "a", 5n), { childBoundaries: [new ChildFlowBoundary("s", "e")] });

  expect(trace.occurrences[0]).toBe(occurrence);
  expect(occurrence!.recordJson).toBe(recordJson);
  expect(occurrence!.inputRoute).toBe(inputRoute);
  expect(Object.isFrozen(occurrence)).toBe(true);
  expect(occurrences).toHaveLength(1);
  expect(trace.lookupSource(new SourceRef(RUN, 1))).toBe(2n);
  expect(trace.resolve("a.output")).toEqual(new ResolvedValue(5n, true, new SourceRef(RUN, 3)));
});

test("a trace delta holds frozen copies of its arrays and defaults to an empty change", () => {
  const trace = startedTrace();
  const empty = new TraceDelta(trace, 1, {});
  expect(empty.originatingTrace).toBe(trace);
  expect(empty.sequence).toBe(1);
  for (const field of [
    "initialized",
    "started",
    "occurrence",
    "handler",
    "writer",
    "consumedConnection",
    "boundaryUpdate",
    "rootStartSource",
    "rootWriter",
    "causalRouteSource",
    "terminal",
  ] as const) {
    expect(empty[field]).toBeNull();
  }
  expect(empty.resumed).toBe(false);
  expect([empty.pruneIds, empty.connections, empty.boundaries]).toEqual([[], [], []]);

  const pruneIds = ["a"];
  const started: [string, number] = ["a", 1];
  const delta = new TraceDelta(trace, 1, { started, pruneIds });
  pruneIds.push("b");
  started[1] = 9;
  expect(delta.pruneIds).toEqual(["a"]);
  expect(delta.started).toEqual(["a", 1]);
  expect(Object.isFrozen(delta)).toBe(true);
  expect(Object.isFrozen(delta.pruneIds)).toBe(true);
  expect(Object.isFrozen(delta.started)).toBe(true);
});

// TS-only (frontier trace): the learnable stop.

/** A learnable started state with START in flight, its log and a prepared `RunStopped`. */
function preparedLearnableStop(): { state: RunState; log: MemoryRunLog; prepared: PreparedTransition } {
  const { state, log } = learnableStarted(flow());
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  return { state, log, prepared: prepareRunStopped(state, "signal", "SIGTERM") };
}

/** Return `prepared` with its record or its plan's trace delta replaced. */
function forgedStop(
  prepared: PreparedTransition,
  changes: { record?: RunRecord; traceDelta?: TraceDelta },
): PreparedTransition {
  const plan = Object.freeze({ ...prepared._plan, traceDelta: changes.traceDelta ?? prepared._plan.traceDelta });
  return new PreparedTransition(
    changes.record ?? prepared.record,
    prepared.expectedRunId,
    prepared.expectedSequence,
    plan as never,
    [],
  );
}

test("a learnable RunStopped carries a sequence-only trace delta and commits", () => {
  const { state, log, prepared } = preparedLearnableStop();
  const trace = state.trace!;
  const delta = prepared._plan.traceDelta!;

  expect(delta).toBeInstanceOf(TraceDelta);
  expect(delta.originatingTrace).toBe(trace);
  expect(delta.sequence).toBe(2);
  expect(delta.started).toBeNull();
  expect(delta.resumed).toBe(false);

  expect(commitPrepared(log, state, prepared)).toEqual([]);

  expect(prepared.record).toStrictEqual(new RunStoppedRecord("signal", "SIGTERM", [START_ID]));
  // The trace applies in place: only its sequence moves.
  expect(state.trace).toBe(trace);
  expect(trace.nextSequence).toBe(3);
  expect([...trace._started]).toEqual([[START_ID, 1]]);
  expect(log.nextSequence).toBe(state.nextSequence);
});

test("a learnable stop whose trace delta starts a node is refused", () => {
  const { state, log, prepared } = preparedLearnableStop();
  const forged = forgedStop(prepared, {
    traceDelta: new TraceDelta(state.trace, state.nextSequence, { started: [START_ID, state.nextSequence] }),
  });

  expectError(
    () => commitPrepared(log, state, forged),
    RangeError,
    "frontier trace stop must only advance the accepted sequence",
  );

  expect(log.nextSequence).toBe(2);
  expect(state.trace!.nextSequence).toBe(2);
});

test("a learnable frontier plan holding an outcome record is refused", () => {
  const { state, log, prepared } = preparedLearnableStop();
  const failure = new Failure({ code: "node.execute", message: "done", nodeId: START_ID });
  const forged = forgedStop(prepared, { record: new NodeFailedRecord(failure) });

  expectError(
    () => commitPrepared(log, state, forged),
    RangeError,
    "frontier trace requires a dispatch, resume, step or stop record",
  );

  expect(log.nextSequence).toBe(2);
});

// The run-state reducer's trace authority: outcome and terminal trace checks before append.

/** Return `prepared` with its plan's trace delta replaced by `traceDelta`. */
function withTrace(prepared: PreparedTransition, traceDelta: TraceDelta | null): PreparedTransition {
  return new PreparedTransition(
    prepared.record,
    prepared.expectedRunId,
    prepared.expectedSequence,
    Object.freeze({ ...prepared._plan, traceDelta }) as never,
    prepared.observers,
  );
}

/** Return a copy of `delta` with `changes`, for Python's `dataclasses.replace`. */
function changedDelta(delta: TraceDelta, changes: TraceDeltaFields): TraceDelta {
  return new TraceDelta(delta.originatingTrace, delta.sequence, { ...delta, ...changes });
}

/** Return a copy of `occurrence` with `changes`, built through its constructor. */
function changedOccurrence(
  occurrence: TraceOccurrence,
  changes: {
    source?: SourceRef;
    nodeId?: string;
    outcomeSeq?: number;
    envelope?: RecordEnvelope;
    memoryScope?: MemoryScope | null;
  },
): TraceOccurrence {
  return new TraceOccurrence(
    changes.source ?? occurrence.source,
    changes.nodeId ?? occurrence.nodeId,
    changes.outcomeSeq ?? occurrence.outcomeSeq,
    changes.envelope ?? occurrence.envelope,
    occurrence.inputRoute,
    changes.memoryScope === undefined ? occurrence.memoryScope : changes.memoryScope,
    occurrence.hiddenInputs,
  );
}

/** Return a copy of `entry` with `changes`. */
function changedEntry(entry: SourceEntry, changes: { source?: SourceRef; outcomeSeq?: number; valueJson?: string }) {
  return new SourceEntry(
    changes.source ?? entry.source,
    changes.outcomeSeq ?? entry.outcomeSeq,
    changes.valueJson ?? entry.valueJson,
    entry.files,
  );
}

/** Return a frozen copy of `value`'s own fields under `cls`'s prototype, skipping its constructor's checks. */
function forged<T extends object>(cls: abstract new (...args: never[]) => T, value: T, changes: object): T {
  return Object.freeze(Object.assign(Object.create(cls.prototype) as T, value, changes));
}

/**
 * Assert that committing `prepared` throws exactly `cls` with `message` and leaves the log, the
 * state and its trace unchanged.
 */
function expectRefused(
  state: RunState,
  log: MemoryRunLog,
  prepared: PreparedTransition,
  cls: ErrorClass,
  message: string,
): void {
  const records = log.records();
  const before = snapshot(state);
  const beforeTrace = traceSnapshot(state.trace!);

  expectError(() => commitPrepared(log, state, prepared), cls, message);

  expect(log.records()).toEqual(records);
  expectUnchanged(state, before);
  expect(traceSnapshot(state.trace!)).toEqual(beforeTrace);
  expect(state.nextSequence).toBe(log.nextSequence);
  expect(state.trace!.nextSequence).toBe(log.nextSequence);
}

/**
 * Python's `_prepared_scoped_outcome("success")`: a scoped CODE node `physical`, the sole in-flight
 * node of a learnable run, and its prepared success committed to END, which ends the run.
 */
function preparedScopedOutcome(): { node: FuncNode; state: RunState; log: MemoryRunLog; prepared: PreparedTransition } {
  const node = new FuncNode("physical", () => "done");
  node.memoryScope = new MemoryScope([node.id], { memories: ["memory"] });
  const { state, log } = startTraceNode(node, { keys: ["memory"], learnable: true });
  const prepared = prepareNodeSucceeded(state, node.id, "done", {
    commitAs: END_ID,
    forwardCapture: forwardCapture(node, {}),
  });
  return { node, state, log, prepared };
}

/** A START -> END flow for the learnable trace tests, Python's `_graph([], [(START_ID, END_ID)])`. */
function startEndFlow(): CompiledFlow {
  return graph([], [[START_ID, END_ID]]);
}

test("the trace is applied only after a successful append", () => {
  const { state, log } = learnableStarted(startEndFlow());
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  const trace = state.trace!;
  const prior = traceSnapshot(trace);
  const prepared = prepareNodeSucceeded(state, START_ID, {}, {
    forwardCapture: forwardCapture(state.flow.nodes.get(START_ID)!, {}),
  });
  expect(trace.occurrences).toEqual([]);
  const append = log.append;

  log.append = () => {
    throw new Error("append failed");
  };
  expectError(() => commitPrepared(log, state, prepared), Error, "append failed");
  expect(traceSnapshot(trace)).toEqual(prior);

  log.append = append;
  commitPrepared(log, state, prepared);
  expect(trace.occurrences).toHaveLength(1);
  expect(trace.currentWriters.get(START_ID)!.source!.executionSeq).toBe(1);
});

test("the trace rejects a forged memory scope before append and when copied: carrier", () => {
  const node = new FuncNode("code", () => "done");
  node.memoryScope = new MemoryScope(["code"], { memories: ["memory"] });
  const { state, log } = startTraceNode(node, { keys: ["memory"], learnable: true });
  const prepared = prepareNodeSucceeded(state, node.id, "done", { forwardCapture: forwardCapture(node, {}) });
  const delta = prepared._plan.traceDelta!;
  const malformed = {} as MemoryScope;
  const message = "historical memory scope must be a MemoryScope or null";

  expectError(() => changedOccurrence(delta.occurrence!, { memoryScope: malformed }), TypeError, message);
  const occurrence = forged(TraceOccurrence, delta.occurrence!, { memoryScope: malformed });
  expectRefused(state, log, withTrace(prepared, changedDelta(delta, { occurrence })), TypeError, message);
});

test("a terminal trace output must match the prepared outcome before append", () => {
  const { state, log, prepared } = preparedScopedOutcome();
  const delta = prepared._plan.traceDelta!;
  const terminal = delta.terminal!;
  const changed = new TraceTerminal(terminal.succeeded, terminal.source, typedJson("forged"), terminal.failure);

  expectRefused(
    state,
    log,
    withTrace(prepared, changedDelta(delta, { terminal: changed })),
    RangeError,
    "terminal trace disagrees with the prepared terminal outcome",
  );
});

test.each(["source", "node", "sequence", "record"] as const)(
  "an outcome trace occurrence must match the canonical record: success, %s",
  (change) => {
    const { state, log, prepared } = preparedScopedOutcome();
    const delta = prepared._plan.traceDelta!;
    const occurrence = delta.occurrence!;
    const changes = {
      source: { source: new SourceRef(state.runId!, 0) },
      node: { nodeId: START_ID },
      sequence: { outcomeSeq: occurrence.outcomeSeq + 1 },
      record: { envelope: log.records().at(-1)! },
    }[change];

    expectRefused(
      state,
      log,
      withTrace(prepared, changedDelta(delta, { occurrence: changedOccurrence(occurrence, changes) })),
      RangeError,
      "physical trace occurrence disagrees with the accepted outcome",
    );
  },
);

test.each(["initialized", "started", "resumed"] as const)(
  "an outcome trace cannot override dispatch authority: success, %s",
  (field) => {
    const { state, log, prepared } = preparedScopedOutcome();
    const changes: TraceDeltaFields = {
      initialized: { initialized: state.trace },
      started: { started: ["physical", 0] as const },
      resumed: { resumed: true },
    }[field];

    expectRefused(
      state,
      log,
      withTrace(prepared, changedDelta(prepared._plan.traceDelta!, changes)),
      RangeError,
      "outcome trace cannot change initialization or dispatch authority",
    );
  },
);

test.each([
  ["target", "physical trace writer disagrees with the accepted output target"],
  ["source", "physical trace writer disagrees with the accepted outcome"],
  ["sequence", "physical trace writer disagrees with the accepted outcome"],
  ["value", "physical trace writer disagrees with the accepted outcome"],
  ["missing", "physical trace writer disagrees with the accepted output target"],
] as const)("an outcome trace writer must match the same accepted physical output: %s", (change, message) => {
  const { node, state, log } = preparedScopedOutcome();
  const prepared = prepareNodeSucceeded(state, node.id, "done", { forwardCapture: forwardCapture(node, {}) });
  const delta = prepared._plan.traceDelta!;
  const [target, writer] = delta.writer!;
  const writers: Record<typeof change, readonly [string, SourceEntry] | null> = {
    target: [state.flow.startId, writer],
    source: [target, changedEntry(writer, { source: new SourceRef(state.runId!, 0) })],
    sequence: [target, changedEntry(writer, { outcomeSeq: writer.outcomeSeq + 1 })],
    value: [target, changedEntry(writer, { valueJson: [...state.trace!.sources.values()][0]!.valueJson })],
    missing: null,
  };

  const changed = changedDelta(delta, { writer: writers[change] });
  expectRefused(state, log, withTrace(prepared, changed), RangeError, message);
});

test.each([
  "none",
  "carrier",
  "source_none",
  "source_sequence",
  "source_run",
  "succeeded",
  "disposition",
  "output",
  "failure",
] as const)("a terminal trace rejects forged fields before append: success, %s", (change) => {
  const { state, log, prepared } = preparedScopedOutcome();
  const delta = prepared._plan.traceDelta!;
  const terminal = delta.terminal!;
  const failure = new Failure({ code: "node.execute", message: "forged", nodeId: "router" });
  const source = terminal.source!;
  const terminals: Record<typeof change, () => TraceTerminal | null> = {
    none: () => null,
    carrier: () => ({}) as TraceTerminal,
    source_none: () => new TraceTerminal(true, null, terminal.outputJson, null),
    source_sequence: () => new TraceTerminal(true, new SourceRef(state.runId!, 0), terminal.outputJson, null),
    source_run: () =>
      new TraceTerminal(true, new SourceRef("foreign-run", source.executionSeq), terminal.outputJson, null),
    succeeded: () => forged(TraceTerminal, terminal, { succeeded: 1 }),
    disposition: () => new TraceTerminal(false, source, null, failure),
    output: () => new TraceTerminal(true, source, typedJson("forged"), null),
    failure: () => forged(TraceTerminal, terminal, { failure }),
  };
  const message =
    change === "none" || change === "carrier"
      ? "terminal outcome requires an exact terminal trace"
      : "terminal trace disagrees with the prepared terminal outcome";

  expectRefused(
    state,
    log,
    withTrace(prepared, changedDelta(delta, { terminal: terminals[change]() })),
    RangeError,
    message,
  );
});

test.each<[string, string]>([
  ["terminal", "nonterminal outcome cannot install a terminal trace"],
  ["rootWriter", "terminal trace root writer disagrees with the accepted outcome"],
  ["causalRouteSource", "terminal trace causal source disagrees with the accepted outcome"],
  ["rootStartSource", "terminal trace root START source disagrees with the accepted outcome"],
])("a nonterminal outcome rejects fabricated terminal trace authority: success, %s", (field, message) => {
  const { node, state, log } = preparedScopedOutcome();
  const prepared = prepareNodeSucceeded(state, node.id, "done", { forwardCapture: forwardCapture(node, {}) });
  const delta = prepared._plan.traceDelta!;
  expect(delta.terminal).toBeNull();
  const source = delta.occurrence!.source;
  const changes: Record<string, TraceDeltaFields> = {
    terminal: {
      terminal: new TraceTerminal(
        false,
        source,
        null,
        new Failure({ code: "node.execute", message: "forged", nodeId: node.id }),
      ),
    },
    rootWriter: { rootWriter: [...state.trace!.currentWriters.values()][0]! },
    causalRouteSource: { causalRouteSource: source },
    rootStartSource: { rootStartSource: source },
  };

  expectRefused(state, log, withTrace(prepared, changedDelta(delta, changes[field]!)), RangeError, message);
});

test.each(["none", "source", "sequence", "value"] as const)(
  "a terminal trace root writer must be the physical accepted output: %s",
  (change) => {
    const { state, log, prepared } = preparedScopedOutcome();
    const delta = prepared._plan.traceDelta!;
    const writer = delta.rootWriter!;
    const writers = {
      none: null,
      source: changedEntry(writer, { source: new SourceRef(state.runId!, 0) }),
      sequence: changedEntry(writer, { outcomeSeq: writer.outcomeSeq + 1 }),
      value: changedEntry(writer, { valueJson: [...state.trace!.currentWriters.values()][0]!.valueJson }),
    };

    expectRefused(
      state,
      log,
      withTrace(prepared, changedDelta(delta, { rootWriter: writers[change] })),
      RangeError,
      "terminal trace root writer disagrees with the accepted outcome",
    );
  },
);

test.each(["none", "sequence", "run"] as const)(
  "a terminal trace root START latch must match the accepted start: %s",
  (change) => {
    const { state, log } = learnableStarted(startEndFlow());
    commitPrepared(log, state, prepareNodeStarted(state, START_ID));
    const prepared = prepareNodeSucceeded(state, START_ID, {}, {
      forwardCapture: forwardCapture(state.flow.nodes.get(START_ID)!, {}),
    });
    const delta = prepared._plan.traceDelta!;
    const source = delta.rootStartSource!;
    const sources = {
      none: null,
      sequence: new SourceRef(state.runId!, 0),
      run: new SourceRef("foreign-run", source.executionSeq),
    };

    expectRefused(
      state,
      log,
      withTrace(prepared, changedDelta(delta, { rootStartSource: sources[change] })),
      RangeError,
      "terminal trace root START source disagrees with the accepted outcome",
    );
    commitPrepared(log, state, prepared);
    expect(state.trace!.rootStartSource).toEqual(source);
  },
);

test.each(["none", "foreign_path", "foreign_memory"] as const)(
  "an outcome rejects a valid scope substitution before append: success, %s",
  (change) => {
    const { node, state, log, prepared } = preparedScopedOutcome();
    const scope = node.memoryScope!;
    const scopes = {
      none: null,
      foreign_path: new MemoryScope(["other"], { memories: scope.memories }),
      foreign_memory: new MemoryScope(scope.staticPath, { memories: ["foreign"] }),
    };
    const delta = prepared._plan.traceDelta!;
    const occurrence = changedOccurrence(delta.occurrence!, { memoryScope: scopes[change] });

    expectRefused(
      state,
      log,
      withTrace(prepared, changedDelta(delta, { occurrence })),
      RangeError,
      "outcome memory scope disagrees with the physical executing node",
    );
  },
);

test.each([
  ["display_node", "outcome memory scope disagrees with the physical executing node"],
  ["delta", "a retained outcome requires its physical trace occurrence"],
  ["occurrence", "a retained outcome requires its physical trace occurrence"],
] as const)("outcome scope authority can't be selected or omitted by the trace delta: success, %s", (change, message) => {
  const { state, log, prepared } = preparedScopedOutcome();
  const delta = prepared._plan.traceDelta!;
  const deltas: Record<typeof change, TraceDelta | null> = {
    display_node: changedDelta(delta, {
      occurrence: changedOccurrence(delta.occurrence!, { nodeId: START_ID, memoryScope: null }),
    }),
    delta: null,
    occurrence: changedDelta(delta, { occurrence: null }),
  };

  expectRefused(state, log, withTrace(prepared, deltas[change]), RangeError, message);
});

test.each<[string, Value, boolean, (string | number)[] | null]>([
  ["input.x", null, true, ["x"]],
  ["input.absent", null, false, null],
  ["input.x.more", null, false, null],
  ["input.items.0", null, false, null],
  [`${START_ID}.output`, { x: null, items: [1n] }, true, []],
  ["system.run_id.ignored", RUN_ID, true, null],
  ["system", null, false, null],
  ["input", null, false, null],
  [" input.x ", null, true, ["x"]],
  ["input. x", null, false, null],
])("the paired resolver preserves pool namespaces: %s", (path, value, present, sourcePath) => {
  const { state } = learnableStarted(startEndFlow(), { inputs: { x: null, items: [1n] } });
  const resolved = state.trace!.resolve(path);

  expect(resolved.value).toEqual(value);
  expect(resolved.present).toBe(present);
  expect(resolved.source).toEqual(sourcePath === null ? null : new SourceRef(state.runId!, 0, sourcePath));
  const [head, ...rest] = path.trim().split(".");
  expect(resolved.value).toEqual(state.pool.resolve(head!, rest));
});

test("the reducer requires a capture exactly when learning", () => {
  const { state: plain } = initializedAndStarted(startEndFlow());
  const capture = forwardCapture(plain.flow.nodes.get(START_ID)!, {});
  expectError(
    () => prepareNodeSucceeded(plain, START_ID, {}, { forwardCapture: capture }),
    RangeError,
    "forward capture requires recorded learning",
  );

  const { state, log } = learnableStarted(startEndFlow());
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  expectError(
    () => prepareNodeSucceeded(state, START_ID, {}),
    RangeError,
    "accepted outcome requires forward capture while learning",
  );
});

// The frontier trace checks on a learnable run.

/** A frontier record a learnable run prepares after START's success. */
type FrontierKind = "started" | "resumed" | "step" | "stopped";

/**
 * Python's `_prepared_trace_frontier`: a learnable `flow()` after START's success, with `work`'s
 * prepared dispatch or, once `work` is in flight, a prepared resume, step or stop.
 */
function preparedTraceFrontier(kind: FrontierKind): {
  state: RunState;
  log: MemoryRunLog;
  prepared: PreparedTransition;
} {
  const { state, log } = learnableStarted(flow());
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  commitStartSuccess(state, log, {});
  let prepared = prepareNodeStarted(state, "work");
  if (kind !== "started") {
    commitPrepared(log, state, prepared);
    if (kind === "resumed") prepared = prepareRunResumed(state, { numWorkers: 0 });
    else if (kind === "step") prepared = prepareNodeStep(state, "work", { event: "progress" });
    else prepared = prepareRunStopped(state, "signal", "SIGTERM");
  }
  return { state, log, prepared };
}

test("a frontier trace can't inject a causal terminal source before append", () => {
  const { state, log, prepared } = preparedTraceFrontier("started");
  const delta = changedDelta(prepared._plan.traceDelta!, { causalRouteSource: new SourceRef(state.runId!, 0) });

  expectRefused(
    state,
    log,
    withTrace(prepared, delta),
    RangeError,
    "frontier trace cannot change outcome or terminal history",
  );
});

/** The message each frontier kind's record check gives a forged `started` or `resumed`. */
const FRONTIER_RECORD_MESSAGES: Record<FrontierKind, string> = {
  started: "frontier trace start disagrees with its canonical record",
  resumed: "frontier trace resume disagrees with its canonical record",
  step: "frontier trace step must only advance the accepted sequence",
  stopped: "frontier trace stop must only advance the accepted sequence",
};

/** The fields that would change outcome or terminal history, plus the missing delta. */
const HISTORY_FIELDS = [
  "missing",
  "initialized",
  "occurrence",
  "handler",
  "writer",
  "rootStartSource",
  "rootWriter",
  "causalRouteSource",
  "terminal",
] as const;

/** A forged field of a frontier trace delta. */
type FrontierField = (typeof HISTORY_FIELDS)[number] | "started" | "resumed";

/**
 * Each kind's forged fields: a dispatch's history fields (its `started` and `resumed` rows aren't
 * portable), a resume's `started` too, and a step's and a stop's `started` and `resumed` too.
 */
const FRONTIER_ROWS: (readonly [FrontierKind, FrontierField])[] = [
  ...HISTORY_FIELDS.map((field) => ["started", field] as const),
  ...(["started", ...HISTORY_FIELDS] as const).map((field) => ["resumed", field] as const),
  ...(["step", "stopped"] as const).flatMap((kind) =>
    (["started", "resumed", ...HISTORY_FIELDS] as const).map((field) => [kind, field] as const),
  ),
];

test.each(FRONTIER_ROWS)("a frontier trace rejects an inappropriate field before append: %s, %s", (kind, field) => {
  const { state, log, prepared } = preparedTraceFrontier(kind);
  const trace = state.trace!;
  const delta = prepared._plan.traceDelta!;
  const writer = [...trace.currentWriters][0]!;
  const changes: Record<Exclude<FrontierField, "missing">, TraceDeltaFields> = {
    initialized: { initialized: trace },
    started: { started: ["work", delta.sequence] },
    resumed: { resumed: true },
    occurrence: { occurrence: trace.occurrences.at(-1)! },
    handler: {
      handler: () => {
        throw new Error("invoked a forged frontier handler");
      },
    },
    writer: { writer },
    rootStartSource: { rootStartSource: new SourceRef(state.runId!, 0) },
    rootWriter: { rootWriter: writer[1] },
    causalRouteSource: { causalRouteSource: new SourceRef(state.runId!, 0) },
    terminal: { terminal: new TraceTerminal(true, writer[1].source, writer[1].valueJson, null) },
  };
  const message =
    field === "missing"
      ? "frontier trace delta requires its exact accepted sequence"
      : field === "started" || field === "resumed"
        ? FRONTIER_RECORD_MESSAGES[kind]
        : "frontier trace cannot change outcome or terminal history";

  expectRefused(
    state,
    log,
    withTrace(prepared, field === "missing" ? null : changedDelta(delta, changes[field])),
    RangeError,
    message,
  );
});

test.each(["node", "sequence"] as const)("a frontier trace start's identity is exact: %s", (change) => {
  const { state, log, prepared } = preparedTraceFrontier("started");
  const delta = prepared._plan.traceDelta!;
  const started = { node: [START_ID, delta.sequence] as const, sequence: ["work", 0] as const }[change];

  expectRefused(
    state,
    log,
    withTrace(prepared, changedDelta(delta, { started })),
    RangeError,
    "frontier trace start disagrees with its canonical record",
  );
});
