/**
 * Transactional flat graph expansion in the run-state reducer: a port of
 * `tests/engine/test_run_state_growth.py`. Python's `ValueError` is a `RangeError` here, and a
 * live rejection is a `GrowthRejected` whose `original` has the exact class and message.
 *
 * Ported: `test_growth_rejects_a_nonfinite_seed_before_builder`,
 * `test_growth_builder_non_flow_return_is_a_raw_host_fault`.
 *
 * Changed:
 * - `test_growth_rejects_a_cyclic_seed_before_builder`: the original is a `TypeCheckError`
 *   (`value contains itself`), not a `ValueError` about a cyclic container.
 * - `test_growth_rejects_an_excessively_nested_seed_before_builder`: the seed nests one level past
 *   `current().maxValueNesting` and `rejectDeepNesting` refuses it with a `TypeCheckError`, where
 *   Python's codec hits its recursion limit (`nesting exceeds`, a `ValueError`).
 * - `test_growth_builder_receives_a_detached_spawner_and_failure_is_pure`: no `env` half
 *   (`copyGraph` copies the spawner shallowly, and `env` is read-only by contract); the builder
 *   gets a copy, its title change doesn't reach the live spawner, its error keeps its identity and
 *   the state is unchanged.
 * - `test_growth_requires_exact_live_carriers`: the `{1, 2}` seed row is a JS `Set` in the seed (a
 *   `TypeCheckError`, `<Set object> is not a JSON value`); the empty prune id and the missing
 *   builder rows are as Python's.
 *
 * TS-only: a non-text prune id (`new Grow({}, [1])`) is `Grow.prune must contain only strings`.
 *
 * Not ported:
 * - `test_growth_seed_rejection_retains_the_original_exception` and
 *   `test_growth_host_helper_faults_propagate_with_identity`: they monkeypatch module functions;
 *   the nonfinite and cyclic seed tests pin the original of a real seed error.
 * - `test_growth_rejects_non_string_record_keys_before_builder`: JS object keys are always text.
 * - `test_growth_detachment_faults_propagate_with_identity`: `copyGraph` has no `__deepcopy__`
 *   hook to fail.
 * - the non-frozenset `prune` row of `test_growth_requires_exact_live_carriers`: `Grow` always
 *   copies `prune` into a new `Set`.
 *
 * Later (next commits of this PR; update this list as they land):
 * `test_growth_canonicalizes_seed_and_isolates_builder_record_and_observers`,
 * `test_growth_rejects_invalid_node_owned_topology_targets`,
 * `test_growth_rejects_invalid_skipped_output_hook_target`,
 * `test_growth_rejects_duplicate_redirect_owner_in_final_topology`,
 * `test_growth_rejects_executable_redirect_target_in_final_topology`,
 * `test_growth_accepts_disconnected_inert_redirect_placeholder`,
 * `test_growth_rejects_invalid_edge_field_types`,
 * `test_growth_append_failure_leaves_topology_and_every_state_coordinate_unchanged`,
 * `test_growth_replay_rejects_a_changed_but_valid_builder_digest`,
 * `test_growth_replay_unwraps_shared_growth_rejection_identity`,
 * `test_growth_live_and_record_replay_produce_equal_state`,
 * `test_growth_record_replay_requires_the_injected_builder`,
 * `test_growth_accepts_the_closed_system_ambients`,
 * `test_growth_accepts_typed_fields_and_nested_map_item_scope`,
 * `test_growth_rejects_invalid_fragments_atomically`,
 * `test_growth_rejects_a_field_read_on_a_scalar_contract`,
 * `test_growth_reference_validation_reads_a_binding_source`,
 * `test_growth_rejects_a_binding_wrapped_unknown_reference_before_digesting`,
 * `test_growth_rejects_an_unknown_reference_nested_in_a_template`,
 * `test_growth_rejects_malformed_static_contract_as_a_host_invariant`,
 * `test_growth_enforces_peak_node_budget_before_same_transition_pruning`,
 * `test_growth_accepts_peak_node_count_equal_to_budget`,
 * `test_growth_rejects_invalid_exact_depth_deltas`,
 * `test_growth_treats_corrupt_stored_depth_as_a_host_invariant_error`,
 * `test_growth_stamps_positive_reference_depth_on_the_boundary`,
 * `test_growth_prunes_only_named_settled_dynamic_overlay_coordinates`,
 * `test_growth_preserves_unrelated_in_flight_siblings_and_fifo_order`,
 * `test_growth_rejects_live_node_collision_even_when_the_id_is_pruned`,
 * `test_growth_rejects_unsafe_pruning_matrix_atomically`,
 * `test_growth_rejects_a_binding_wrapped_dependency_on_a_pruned_node`,
 * `test_growth_rejects_a_template_dependency_on_a_pruned_node`,
 * `test_read_ids_include_references_nested_in_a_template`,
 * `test_growth_prune_scan_rejects_malformed_surviving_wiring_as_raw_invariant`,
 * `test_growth_commit_performs_no_builder_codec_graph_or_event_work_after_append`,
 * `test_growth_records_the_spawner_as_owner_of_its_fragment`,
 * `test_pruning_keeps_a_grown_subtree_with_unsettled_work`,
 * `test_pruning_retires_a_grown_subtree_once_settled`,
 * `test_growth_keeps_an_unsettled_prune_id_live_and_deferred`,
 * `test_growth_keeps_a_settled_prune_id_that_kept_work_reads`,
 * `test_growth_retires_a_settled_prune_id_the_kept_work_does_not_read`.
 *
 * Later (T3-22, they need `build_grow`):
 * `test_actual_build_grow_splices_call_map_and_loop_with_fifo_and_depth`,
 * `test_actual_call_growth_allows_an_omitted_optional_defaulted_child_input`,
 * `test_zero_field_loop_growth_schedules_its_next_driver_after_the_body`,
 * `test_loop_clone_record_uses_actual_id_self_prunes_and_inherits_depth_zero`.
 */

import { expect, test } from "bun:test";

import { CompiledFlow, Edge, END_ID, Flow, GrowthRejected, START_ID } from "../../src/compile/model.ts";
import { current } from "../../src/defaults.ts";
import { Grow, Node, NodeKind } from "../../src/nodes/base.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { MemoryRunLog } from "../../src/runtime/log.ts";
import {
  commitPrepared,
  type GrowBuilder,
  prepareExpanded,
  prepareNodeStarted,
  prepareNodeSucceeded,
  prepareRunStarted,
  RunState,
} from "../../src/runtime/state.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { TypeCheckError } from "../../src/typesys/values.ts";
import { expectUnchanged, FuncNode, snapshot } from "./run_state_fixtures.ts";

const FINGERPRINT = "d".repeat(64);
const RUN_ID = "growth-run";

/** A minimal kind-blind spawner used to isolate reducer behavior. */
class Spawner extends Node {
  readonly kind = NodeKind.CALL;
  override readonly isSpawner = true;
  override readonly growDepthDelta = 1;
}

/** Return a complete static graph whose one spawner can be put in flight. */
function parentFlow(options: { spawner?: Node; sibling?: boolean } = {}): CompiledFlow {
  const spawner = options.spawner ?? new Spawner("spawn");
  const nodes = new Map<string, Node>([
    [START_ID, new StartNode(START_ID, { params: [] })],
    [spawner.id, spawner],
    [END_ID, EndNode.value(END_ID, { outputName: null })],
  ]);
  if (options.sibling) nodes.set("side", new FuncNode("side", (inputs) => inputs));
  const edges = [new Edge("start-spawn", START_ID, spawner.id), new Edge("spawn-end", spawner.id, END_ID)];
  if (options.sibling) edges.push(new Edge("start-side", START_ID, "side"));
  return CompiledFlow.fromParts(nodes, edges, { wiring: new Map([...nodes.keys()].map((id) => [id, new Map()])) });
}

/** Return a replay-valid state and log with the static spawner in flight. */
function initializedSpawner(flow: CompiledFlow = parentFlow()): { state: RunState; log: MemoryRunLog } {
  const initialPool = new VariablePool(flow.startId);
  initialPool.set(flow.startId, {});
  initialPool.addSystem("run_id", RUN_ID);
  const state = new RunState(flow);
  const log = new MemoryRunLog(RUN_ID);
  commitPrepared(
    log,
    state,
    prepareRunStarted(state, {
      runId: RUN_ID,
      flowFingerprint: FINGERPRINT,
      initialPool,
      startWiringOverride: {},
      effectiveLlmByStaticPath: {},
      numWorkers: 0,
    }),
  );
  commitPrepared(log, state, prepareNodeStarted(state, START_ID));
  commitPrepared(log, state, prepareNodeSucceeded(state, START_ID, {}));
  commitPrepared(log, state, prepareNodeStarted(state, "spawn"));
  return { state, log };
}

/** Return a manifestable two-boundary fragment committing under `spawnerId`. */
function fragment(spawnerId: string, suffix: string = "child"): Flow {
  const startId = `${spawnerId}/${suffix}/start`;
  const endId = `${spawnerId}/${suffix}/end`;
  const end = EndNode.value(endId, { outputName: null });
  end.commitAs = spawnerId;
  return new Flow(
    new Map<string, Node>([
      [startId, new StartNode(startId, { params: [] })],
      [endId, end],
    ]),
    [new Edge(`${startId}->${endId}`, startId, endId, { ordering: true })],
    {
      wiring: new Map([
        [startId, new Map()],
        [endId, new Map()],
      ]),
      startId,
      endId,
    },
  );
}

/** A builder returning `fragment(spawner.id)`. */
const fragmentBuilder: GrowBuilder = (spawner) => fragment(spawner.id);

/** Return what `fn` throws. */
function caught(fn: () => unknown): unknown {
  try {
    fn();
  } catch (error) {
    return error;
  }
  throw new Error("expected an error");
}

/** Assert `fn` throws an error of exactly class `cls` whose message matches `message`. */
function expectError(fn: () => unknown, cls: new (...args: never[]) => Error, message: RegExp): void {
  const error = caught(fn) as Error;
  expect(error.constructor).toBe(cls);
  expect(error.message).toMatch(message);
}

/**
 * Assert `fn` throws a `GrowthRejected` whose `original` has exactly class `cls` and a message
 * matching `message`; return that original.
 */
function expectRejected(fn: () => unknown, cls: new (...args: never[]) => Error, message: RegExp): Error {
  const error = caught(fn) as GrowthRejected;
  expect(error.constructor).toBe(GrowthRejected);
  expect(error.original.constructor).toBe(cls);
  expect(error.original.message).toMatch(message);
  expect(error.cause).toBe(error.original);
  return error.original;
}

test("growth rejects a cyclic seed before the builder", () => {
  const { state } = initializedSpawner();
  const seed: unknown[] = [];
  seed.push(seed);

  expectRejected(
    () => prepareExpanded(state, "spawn", new Grow(seed), fragmentBuilder),
    TypeCheckError,
    /^value contains itself$/,
  );
});

test("growth rejects a nonfinite seed before the builder", () => {
  const { state } = initializedSpawner();
  let builderCalled = false;
  const builder: GrowBuilder = () => {
    builderCalled = true;
    return fragment("spawn");
  };

  const original = expectRejected(
    () => prepareExpanded(state, "spawn", new Grow({ x: Number.NaN }), builder),
    TypeCheckError,
    /^NaN is not a yian value$/,
  );

  expect((original as TypeCheckError).path).toEqual(["x"]);
  expect(builderCalled).toBe(false);
});

test("growth rejects an excessively nested seed before the builder", () => {
  const { state } = initializedSpawner();
  const limit = current().maxValueNesting;
  let seed: unknown = "leaf";
  for (let level = 0; level <= limit; level += 1) seed = [seed];
  let builderCalled = false;
  const builder: GrowBuilder = () => {
    builderCalled = true;
    return fragment("spawn");
  };

  expectRejected(
    () => prepareExpanded(state, "spawn", new Grow(seed), builder),
    TypeCheckError,
    new RegExp(`^growth seed at (\\[0\\]){${limit}}: value nests deeper than ${limit} levels, `),
  );
  expect(builderCalled).toBe(false);
});

test("the growth builder receives a detached spawner and its failure is pure", () => {
  const { state } = initializedSpawner();
  const liveSpawner = state.flow.nodes.get("spawn")!;
  liveSpawner.title = "live";
  const before = snapshot(state);
  const original = new Error("builder failed");
  let received: Node | null = null;
  const mutatingFailingBuilder: GrowBuilder = (spawner) => {
    received = spawner;
    spawner.title = "builder-mutated";
    throw original;
  };

  const error = caught(() => prepareExpanded(state, "spawn", new Grow({ value: 1 }), mutatingFailingBuilder));

  expect(error).toBe(original);
  expect(received).not.toBeNull();
  expect(received).not.toBe(liveSpawner);
  expectUnchanged(state, before);
  expect(state.flow.nodes.get("spawn")).toBe(liveSpawner);
  expect(liveSpawner.title).toBe("live");
});

test("a growth builder's non-Flow return is a raw host fault", () => {
  const { state } = initializedSpawner();

  expectError(
    () => prepareExpanded(state, "spawn", new Grow({}), (() => ({})) as unknown as GrowBuilder),
    TypeError,
    /^grow_builder must return a Flow$/,
  );
});

test.each([
  {
    name: "empty prune id",
    grow: () => new Grow({}, [""]),
    original: RangeError as new (...args: never[]) => Error,
    message: /^Grow\.prune node IDs must be non-empty$/,
  },
  {
    name: "non-text prune id",
    grow: () => new Grow({}, [1] as unknown as string[]),
    original: TypeError,
    message: /^Grow\.prune must contain only strings$/,
  },
  {
    name: "Set in the seed",
    grow: () => new Grow({ bad: new Set([1, 2]) }),
    original: TypeCheckError,
    message: /^<Set object> is not a JSON value$/,
  },
])("growth wraps a node-produced carrier rejection: $name", ({ grow, original, message }) => {
  const { state } = initializedSpawner();

  expectRejected(() => prepareExpanded(state, "spawn", grow(), fragmentBuilder), original, message);
});

test("growth requires a callable builder, a raw host fault", () => {
  const { state } = initializedSpawner();

  expectError(
    () => prepareExpanded(state, "spawn", new Grow({}), null as unknown as GrowBuilder),
    TypeError,
    /^grow_builder must be callable$/,
  );
});
