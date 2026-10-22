/**
 * Inline composite bodies: a port of `tests/engine/test_inline_composite_bodies.py`: the rows the
 * CALL, MAP and LOOP parse decides, the inline body extraction and registration rows, and the load
 * halves of the rows that build inline children.
 *
 * Ported: `test_child_label_is_not_authorable`, through `parseNodes` with the registered kinds;
 * the TS message names the field and lists the allowed keys. Through `extractInlineBodies`:
 * `test_unchanged_inline_projection_preserves_alias_sharing`,
 * `test_inline_alias_projection_keeps_shared_payload_and_lightweight_sources`,
 * `test_extract_one_inline_body_masks_only_the_occurrence_and_retains_locations`,
 * `test_inline_projection_leaves_original_cursor_and_document_unchanged`,
 * `test_extract_alias_occurrences_get_distinct_ids_and_share_retained_children`,
 * `test_extract_merged_inline_body_retains_anchor_values_and_ranges`,
 * `test_extract_merged_local_body_override_uses_effective_values_and_ranges`,
 * `test_extract_rejects_authored_private_call_target_without_mutating_source`,
 * `test_extract_rejects_sibling_private_id_attack_before_masking_any_body`,
 * `test_extract_allows_private_prefix_in_graph_node_id`,
 * `test_inline_body_shape_errors_are_located_without_private_ids`. Through the exported
 * `registerInlineBodies`: `test_register_inline_bodies_associates_call_map_loop_and_stays_lazy`,
 * `test_register_inline_body_rejects_bad_association_located_and_redacted`,
 * `test_register_inline_bodies_preflights_all_before_first_registration`. Through
 * `assembleText`: `test_load_translates_recursive_inline_preflight_access_to_load_error`,
 * `test_loader_allows_private_callable_prefix_in_graph_node_id`,
 * `test_inline_body_shape_matrix_flows_through_loader`,
 * `test_inline_body_non_text_input_key_fails_at_the_key_through_loader`,
 * `test_loader_rejects_actual_sibling_private_digest_without_exposing_it`,
 * `test_inline_signature_self_cycle_uses_binding_path_and_range`,
 * `test_inline_signature_mutual_cycle_uses_closing_binding_range`,
 * `test_inline_signature_unknown_producer_is_located_and_strict`,
 * `test_inline_signature_producer_without_output_is_located_and_strict`,
 * `test_inline_map_direct_item_on_opaque_over_fails_at_requesting_binding`,
 * `test_inline_map_item_proof_is_lazy_and_cached`,
 * `test_inline_map_missing_list_element_fails_at_item_binding`,
 * `test_inline_outer_env_stays_on_driver_and_child_env_stays_inner`,
 * `test_inline_composites_reject_outer_llm_config_without_private_identity`,
 * `test_inline_loop_contract_diagnostics_are_friendly_and_redacted`,
 * `test_inline_wrapper_bad_input_uses_outer_binding_range`,
 * `test_inline_map_output_error_uses_friendly_child_label`, the node-assert check's
 * `test_inline_map_rejects_outer_asserts_without_private_identity`, the reference validator's
 * `test_inline_child_scope_is_closed_and_redacts_private_identity`,
 * `test_inline_map_child_keeps_item_namespace_closed`,
 * `test_inline_sibling_keeps_outer_bad_binding_range` and
 * `test_inline_child_bad_output_uses_retained_output_range`, and the load halves of
 * `test_typed_inline_call_preserves_nullable_parent_scalar`,
 * `test_typed_inline_call_canonicalizes_literal_record_and_retains_input_order`,
 * `test_forward_producer_type_builds_once_and_keeps_source_order`,
 * `test_producer_declared_none_type_is_not_treated_as_missing_output`,
 * `test_inline_alias_and_merge_occurrences_run_as_distinct_private_children`,
 * `test_inline_nested_containers_are_evaluated_at_edges_and_runtime` (the edge),
 * `test_inline_optional_record_paths_make_generated_inputs_nullable`,
 * `test_inline_input_inference_projects_known_partial_loop_field`,
 * `test_inline_map_literal_over_runs_with_inferred_item_type`,
 * `test_inline_map_opaque_over_without_direct_item_walk_loads_and_runs`,
 * `test_inline_map_typed_producer_maps_whole_item_and_field_in_parallel_order`,
 * `test_inline_map_proves_list_any_date_element_and_runs`,
 * `test_inline_map_nullable_root_and_intermediate_infer_optional_and_intermediate_runs_none`,
 * `test_basic_inline_loop_times_bakes_private_child_signature_and_runs`,
 * `test_inline_loop_predicates_run_to_the_same_carried_result`. Where
 * Python checks a message fragment, the row pins the whole message.
 *
 * TS-only: `inlineFlowId` pinned for two owners, one holding non-ASCII, control, quote and
 * backslash characters (the digest of Python's `json.dumps(..., ensure_ascii=False)` bytes); and
 * a CALL descriptor built with `childLabel: "inline body of node 'each'"` over a hand-built child
 * names the body by that label in an undeclared argument's message.
 *
 * Changed: the direct shape row drops its `non-string-input-key` param (the TS document gives
 * every key as text, so `extractInlineBodies` sees `"1"`); the non-text key row expects the
 * document's refusal of the key, as Python's loader row does. The sibling-attack row cannot patch
 * the private `maskInlineBody`, so it spies on `SourceCursor.prototype.projectMapping`, which
 * masking calls. A wrong-kind descriptor is a CODE descriptor, not an arbitrary object. Descriptor
 * equality leaves out `childLabel`, as Python's `compare=False` does. The monkeypatching rows spy
 * with `spyOn`: on `CodeDescriptor.prototype.build` (the producer's builds) and on the
 * `inferInlineBindingType` export (the MAP item proof); the row patching `read_source_flow_inputs`
 * gives `assemble` its `list` input with no element as a resolved parameter instead. The
 * partial-LOOP row's `defs:` body is a hand-built child whose `note` producer has no contract, so
 * the LOOP's output is a partial record, as the def's expression output makes it. The
 * nullable-scalar row's `value` param only feeds its run half, so the load half is one row. The basic LOOP row compares the output contract's type (the contract
 * carries the body's origins).
 *
 * Pinned at the end (`parsedNodes`):
 * `load-map-call/engine_inline_composite_bodies__llm_config_on_an_inline_{call,loop,map}`, whose
 * body-key check reaches `llm_config:`, the second key, before the inline `nodes:` key and before
 * the missing-`call:` check; and (`assembled`) the extraction cases
 * `load-map-call/engine_inline_composite_bodies__{call_node_naming_a_private_digest,
 * inline_body_missing_output,named_call_with_output,neither_call_nor_nodes,
 * non_mapping_inline_nodes}`, the child-building cases
 * `load-map-call/engine_inline_composite_bodies__{inline_map_body_without_output,
 * inline_loop_body_returns_an_uncarried_field,inline_signature_self_cycle}`, the reference cases
 * `load-map-call/engine_inline_composite_bodies__{inline_child_scope_is_closed,
 * inline_child_unknown_output_ref_is_located,inline_map_child_keeps_item_namespace_closed}` and
 * `load-types/compose_typed_bindings_downstream__inline_map_item_over_an_opaque_list`, and the ok
 * cases `load-map-call/engine_inline_composite_bodies__private_prefix_in_a_node_id_loads` and
 * `load-map-call/engine_templates_load__inline_composites_template_loads` (no mirror of
 * `tests/engine/test_templates_load.py` yet), and the node-assert check's
 * `load-map-call/engine_inline_composite_bodies__inline_map_rejects_outer_asserts`.
 *
 * Ported through `loadFlow` (the fixture CODE references resolve against the conformance dir):
 * `test_eager_def_inline_child_bad_output_preserves_absolute_range`,
 * `test_inline_call_resolver_failure_uses_friendly_child_label`, and the load halves of
 * `test_inline_multinode_body_matches_named_def_signature_and_runtime`,
 * `test_eager_def_inline_call_uses_origin_and_shared_scope`,
 * `test_eager_def_nested_inline_calls_use_exact_lexical_owners_once`,
 * `test_zero_input_inline_call_loads_and_runs_in_one_file_scope` and
 * `test_nested_inline_composites_register_four_lexical_children_once`, the rows counting
 * registrations and resolutions with spies on `CallableScope.prototype`'s `registerPrivate` and
 * `resolve`; on the scope itself, `test_def_assembly_cursor_fallback_restores_only_node_surfaces`
 * (through `defAssemblyCursor`, exported for it) and the `test_callable_scope_*` rows.
 * Changed there: the scope's private table is not readable, so the idempotent-registration row
 * checks the first builder is the one that builds, not the kept label; the eager-def row cannot
 * patch the module-private def build, so a subclass's `resolve` fakes it for `outer` (and the
 * row leaves out the def's cached identity). Where Python checks a message fragment, the row
 * pins the whole message.
 *
 * The runs, each loading the flow (`loadConformanceFlow`, with any `defs:` Python writes in it) and
 * running it through `runFlow`: the run halves of
 * `test_zero_input_inline_call_loads_and_runs_in_one_file_scope`,
 * `test_eager_def_inline_call_uses_origin_and_shared_scope`,
 * `test_nested_inline_composites_register_four_lexical_children_once` and
 * `test_eager_def_nested_inline_calls_use_exact_lexical_owners_once`, each spying on
 * `CallableScope.prototype` to check the run registers and resolves no private flow beyond the
 * load's (Python counts after the run); of `test_typed_inline_call_preserves_nullable_parent_scalar`
 * (2), `test_typed_inline_call_canonicalizes_literal_record_and_retains_input_order`,
 * `test_forward_producer_type_builds_once_and_keeps_source_order` (the producer built once across
 * the load and the run), `test_producer_declared_none_type_is_not_treated_as_missing_output`,
 * `test_inline_alias_and_merge_occurrences_run_as_distinct_private_children`,
 * `test_inline_nested_containers_are_evaluated_at_edges_and_runtime`,
 * `test_inline_optional_record_paths_make_generated_inputs_nullable` (both runs),
 * `test_inline_input_inference_projects_known_partial_loop_field` (with the real `body` def),
 * `test_inline_map_opaque_over_without_direct_item_walk_loads_and_runs`,
 * `test_inline_map_proves_list_any_date_element_and_runs`,
 * `test_basic_inline_loop_times_bakes_private_child_signature_and_runs` and
 * `test_inline_loop_predicates_run_to_the_same_carried_result`; and
 * `test_inline_call_and_loop_outer_post_asserts_still_fire` (2, run only). The runs' ints are
 * bigints (decision 4a). Changed: the date row's dates go in and come out as text (rule `dates`).
 *
 * Not ported: the run halves of `test_inline_multinode_body_matches_named_def_signature_and_runtime`,
 * covered by `run-map-call/engine_inline_composite_bodies__inline_body_runs_like_the_named_def`;
 * `test_inline_map_literal_over_runs_with_inferred_item_type`, covered by
 * `run-map-call/engine_inline_composite_bodies__inline_map_over_a_literal_list`;
 * `test_inline_map_typed_producer_maps_whole_item_and_field_in_parallel_order`, covered by
 * `run-map-call/engine_inline_composite_bodies__parallel_inline_map_keeps_item_order`; and
 * `test_inline_map_nullable_root_and_intermediate_infer_optional_and_intermediate_runs_none`,
 * covered by `run-map-call/engine_inline_composite_bodies__inline_map_item_paths_through_nulls`.
 */

import { expect, spyOn, test } from "bun:test";

import { CompiledFlow, END_ID, START_ID } from "../../src/compile/model.ts";
import * as inlineInputs from "../../src/compose/build/inline_inputs.ts";
import { type ComposedFlow, type PrivateRegistrar, registerInlineBodies } from "../../src/compose/build/pipeline.ts";
import {
  extractInlineBodies,
  INLINE_FLOW_PREFIX,
  type InlineBody,
  inlineFlowId,
} from "../../src/compose/desugar/inline_bodies.ts";
import { LoadDiagnostics, type LoadError } from "../../src/compose/helpers/errors.ts";
import { BuildContext, type NodeDescriptor } from "../../src/compose/nodes/base.ts";
import { CallDescriptor } from "../../src/compose/nodes/call.ts";
import { CodeDescriptor } from "../../src/compose/nodes/code.ts";
import { LoopDescriptor } from "../../src/compose/nodes/loop.ts";
import { CallableScope, defAssemblyCursor } from "../../src/compose/loader.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import { RunFailed } from "../../src/events.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import type { LoopNode } from "../../src/nodes/loop/node.ts";
import type { MapNode } from "../../src/nodes/map/node.ts";
import { runtime } from "../../src/platform.ts";
import { runFlow } from "../../src/run.ts";
import { Type, type Value, ValueKind, type ValueRecord } from "../../src/typesys/values.ts";
import {
  assembled,
  assembleText,
  type CaseRow,
  handChild,
  loadConformanceFlow,
  loadError,
  loadErrorAsync,
  parametersFromYaml,
  parsedNodes,
  runCase,
} from "../compose/declarations.ts";

/** Each mapping pair of `cursor`: its key's built value, key range, value range and value. */
function mappingSnapshot(cursor: SourceCursor): (Value | SourceRange | null)[][] {
  return cursor.mappingOccurrences().map((pair) => [pair.sourceKeyValue, pair.keyRange, pair.range, pair.value]);
}

/** The authored key of each mapping pair of `cursor`, merged pairs included. */
function mappingKeys(cursor: SourceCursor): Value[] {
  return cursor.mappingOccurrences().map((pair) => pair.sourceKeyValue);
}

/** The keys of `cursor`'s mapping value, in order. */
function keysOf(cursor: SourceCursor): string[] {
  return Object.keys(cursor._valueView() as ValueRecord);
}

/** The frozen value of entry `node` of `cursor`'s mapping, aliases kept. */
function viewOf(cursor: SourceCursor, node: string): ValueRecord {
  return (cursor._valueView() as ValueRecord)[node] as ValueRecord;
}

test("inlineFlowId hashes the owner's compact JSON text", () => {
  expect(inlineFlowId(["root", "node", "victim", "inline"])).toBe(
    "__inline_flow_920d718ce90797241b1db8556e28c4150ff1b7c89aebaf79197e2a0a4a16522b",
  );
});

test("inlineFlowId keeps non-ASCII text and escapes control, quote and backslash characters", () => {
  expect(inlineFlowId(["root", "node", 'é\u0001"\\ 😀 \u007f\u001f', "inline"])).toBe(
    "__inline_flow_8e086ba78a7a412352d888c1b343ffc8b6c2f5ee5d04a00dc9faecd6b493041b",
  );
});

test("an unchanged inline projection preserves alias sharing", () => {
  const aliases = 32;
  const lines = [
    "nodes:",
    "  original: &shared",
    "    kind: code",
    "    code: ./fixtures/compose_codefns.ts:echo_x",
    "    input:",
    "      x: &payload [one, two, three]",
    "    output: str",
  ];
  for (let index = 0; index < aliases; index++) lines.push(`  alias_${index}: *shared`);
  const document = SourceDocument.parse(`${lines.join("\n")}\n`);

  const [projected, bodies] = extractInlineBodies(document.root.child("nodes"), { owner: ["root"] });
  const original = viewOf(projected, "original");

  expect(bodies.size).toBe(0);
  for (let index = 0; index < aliases; index++) {
    const alias = viewOf(projected, `alias_${index}`);
    expect(alias).toBe(original);
    expect((alias["input"] as ValueRecord)["x"]).toBe((original["input"] as ValueRecord)["x"]!);
  }
});

test("an inline alias projection keeps the shared payload and lightweight sources", () => {
  const aliases = 16;
  const lines = [
    "nodes:",
    "  original: &inline",
    "    kind: call",
    "    input:",
    "      payload: &payload [one, two, three]",
    "    nodes:",
    "      echo:",
    "        kind: code",
    "        code: ./fixtures/compose_codefns.ts:echo_x",
    "        input:",
    "          x: ${input.payload}",
    "        output: str",
    "    output: ${echo.output}",
  ];
  for (let index = 0; index < aliases; index++) lines.push(`  alias_${index}: *inline`);
  const document = SourceDocument.parse(`${lines.join("\n")}\n`);

  const [projected, bodies] = extractInlineBodies(document.root.child("nodes"), { owner: ["root"] });
  const originalInput = viewOf(projected, "original")["input"];

  expect(bodies.size).toBe(aliases + 1);
  for (let index = 0; index < aliases; index++) {
    expect(viewOf(projected, `alias_${index}`)["input"]).toBe(originalInput!);
  }
  const sources = (projected._projectedEntries ?? []).map((entry) => entry.source);
  expect(sources.length).toBe(aliases + 1);
  for (const source of sources) {
    expect(source).not.toBeNull();
    expect(source!._value).toEqual({ kind: null, input: null, call: null });
  }
});

const ASSOCIATION_FLOW = `nodes:
  call_node:
    kind: call
    nodes:
      echo:
        kind: code
        run: ./fixtures/compose_codefns.ts:echo
        input:
          value: hello
        output: str
    output: \${echo.output}
  map_node:
    kind: map
    over: \${input.items}
    nodes:
      echo:
        kind: code
        run: ./fixtures/compose_codefns.ts:echo
        input:
          value: hello
        output: str
    output: \${echo.output}
  loop_node:
    kind: loop
    times: 1
    input:
      value: 0
    nodes:
      echo:
        kind: code
        run: ./fixtures/compose_codefns.ts:echo
        input:
          value: hello
        output: str
    output: \${echo.output}
`;

/** Located retained bodies and their final CALL/MAP/LOOP descriptors. */
function associationFixture(): [Map<string, NodeDescriptor>, Map<string, InlineBody>] {
  const root = SourceDocument.parse(ASSOCIATION_FLOW).root;
  const [masked, bodies] = extractInlineBodies(root.child("nodes"), { owner: ["root"] });
  return [parseNodes(masked), bodies];
}

/** A registrar that records each private registration without resolving it. */
class RecordingRegistrar implements PrivateRegistrar {
  readonly registrations: [string, readonly string[], string, () => ComposedFlow][] = [];

  registerPrivate(flowId: string, owner: readonly string[], label: string, builder: () => ComposedFlow): void {
    this.registrations.push([flowId, owner, label, builder]);
  }
}

test("extracting one inline body masks only the occurrence and retains locations", () => {
  const document = SourceDocument.parse(`nodes:
  wrapper:
    kind: call
    input:
      text: \${input.text}
    nodes:
      echo:
        kind: code
        run: ./fixtures/compose_codefns.ts:echo
        input:
          value: \${input.text}
        output: str
    output: \${echo.output}
`);
  const originalNodes = document.root.child("nodes");
  const originalBody = originalNodes.child("wrapper");
  const originalChildNodes = originalBody.child("nodes");
  const originalBodyValue = originalBody.value;
  const originalPairs = mappingSnapshot(originalBody);
  const originalDocumentValue = document.root.value;
  const childOwner = ["root", "node", "wrapper", "inline"];

  const [maskedNodes, bodies] = extractInlineBodies(originalNodes, { owner: ["root"] });

  expect(inlineFlowId(childOwner)).toBe(
    `${INLINE_FLOW_PREFIX}78882e0096fb763300120ec6c6428e700d2dd5094af58139242c79a1023a243a`,
  );
  expect([...bodies.keys()]).toEqual(["wrapper"]);
  const body = bodies.get("wrapper")!;
  expect(body.nodeId).toBe("wrapper");
  expect(body.owner).toEqual(childOwner);
  expect(body.flowId).toBe(inlineFlowId(childOwner));
  expect(body.nodesCursor.value).toEqual(originalChildNodes.value);
  expect(body.nodesCursor.range).toEqual(originalChildNodes.range!);
  expect(body.output).toBe("${echo.output}");
  expect(body.outputRange).toEqual(originalBody.rangeOf("output")!);
  expect(body.inputRanges).toEqual(originalBody.mappingValueRanges("input"));

  const maskedBody = maskedNodes.child("wrapper");
  expect(maskedBody.value).toEqual({ kind: "call", input: { text: "${input.text}" }, call: body.flowId });
  expect(maskedBody.rangeOf("call")).toEqual(originalBody.rangeOf("nodes")!);
  expect(keysOf(maskedBody)).toEqual(["kind", "input", "call"]);

  expect(originalBody.value).toEqual(originalBodyValue);
  expect(mappingSnapshot(originalBody)).toEqual(originalPairs);
  expect(document.root.value).toEqual(originalDocumentValue);
});

test("an inline projection leaves the original cursor and document unchanged", () => {
  const document = SourceDocument.parse(`nodes:
  wrapper:
    kind: call
    nodes:
      echo:
        kind: code
        run: ./fixtures/compose_codefns.ts:echo
        input:
          value: hello
        output: str
    output: \${echo.output}
`);
  const originalNodes = document.root.child("nodes");
  const originalValue = originalNodes.value;
  const originalPairs = mappingSnapshot(originalNodes);
  const originalDocumentValue = document.root.value;

  const [masked, bodies] = extractInlineBodies(originalNodes, { owner: ["root"] });

  expect(masked.child("wrapper").has("call")).toBe(true);
  expect(masked.child("wrapper").has("nodes")).toBe(false);
  expect(bodies.get("wrapper")!.nodesCursor.child("echo").str("kind")).toBe("code");
  expect(originalNodes.value).toEqual(originalValue);
  expect(mappingSnapshot(originalNodes)).toEqual(originalPairs);
  expect(document.root.value).toEqual(originalDocumentValue);
});

test("alias occurrences get distinct ids and share retained children", () => {
  const document = SourceDocument.parse(`
body: &body
  kind: call
  input:
    text: \${input.text}
  nodes:
    echo:
      kind: code
      run: ./fixtures/compose_codefns.ts:echo
      input:
        value: \${input.text}
      output: str
  output: \${echo.output}
nodes:
  first: *body
  second: *body
`);
  const originalNodes = document.root.child("nodes");
  const anchor = document.root.child("body");
  const anchorChildren = anchor.child("nodes");
  const originalAnchorValue = anchor.value;
  const originalAnchorPairs = mappingSnapshot(anchor);
  const originalAnchorKeys = mappingKeys(anchor);
  const originalDocumentValue = document.root.value;

  const [maskedNodes, bodies] = extractInlineBodies(originalNodes, { owner: ["root"] });

  expect([...bodies.keys()]).toEqual(["first", "second"]);
  expect(bodies.get("first")!.owner).toEqual(["root", "node", "first", "inline"]);
  expect(bodies.get("second")!.owner).toEqual(["root", "node", "second", "inline"]);
  expect(bodies.get("first")!.flowId).not.toBe(bodies.get("second")!.flowId);
  for (const body of bodies.values()) {
    expect(body.nodesCursor.value).toEqual(anchorChildren.value);
    expect(body.nodesCursor.range).toEqual(anchorChildren.range!);
    expect(body.output).toBe("${echo.output}");
    expect(body.outputRange).toEqual(anchor.rangeOf("output")!);
  }

  const first = maskedNodes.child("first");
  const second = maskedNodes.child("second");
  expect(first._valueView()).not.toBe(second._valueView());
  expect((first.value as ValueRecord)["call"]).toBe(bodies.get("first")!.flowId);
  expect((second.value as ValueRecord)["call"]).toBe(bodies.get("second")!.flowId);
  expect(first.rangeOf("call")).toEqual(anchor.rangeOf("nodes")!);
  expect(second.rangeOf("call")).toEqual(anchor.rangeOf("nodes")!);
  expect(keysOf(first)).toEqual(["kind", "input", "call"]);
  expect(keysOf(second)).toEqual(["kind", "input", "call"]);

  expect(anchor.value).toEqual(originalAnchorValue);
  expect(mappingSnapshot(anchor)).toEqual(originalAnchorPairs);
  expect(mappingKeys(anchor)).toEqual(originalAnchorKeys);
  expect(document.root.value).toEqual(originalDocumentValue);
});

test("a merged inline body retains the anchor's values and ranges", () => {
  const document = SourceDocument.parse(`
body: &body
  kind: call
  nodes:
    echo:
      kind: code
      run: ./fixtures/compose_codefns.ts:echo
      input:
        value: hello
      output: str
  output: \${echo.output}
nodes:
  merged:
    <<: *body
    input:
      text: hello
`);
  const anchor = document.root.child("body");
  const anchorChildren = anchor.child("nodes");
  const originalNodes = document.root.child("nodes");
  const mergedBefore = originalNodes.child("merged");
  const originalMergedPairs = mappingSnapshot(mergedBefore);
  const originalMergedKeys = mappingKeys(mergedBefore);
  const originalDocumentValue = document.root.value;

  const [maskedNodes, bodies] = extractInlineBodies(originalNodes, { owner: ["root"] });

  const body = bodies.get("merged")!;
  expect(body.nodesCursor.value).toEqual(anchorChildren.value);
  expect(body.nodesCursor.range).toEqual(anchorChildren.range!);
  expect(body.output).toBe("${echo.output}");
  expect(body.outputRange).toEqual(anchor.rangeOf("output")!);

  const masked = maskedNodes.child("merged");
  expect(masked.value).toEqual({ kind: "call", call: body.flowId, input: { text: "hello" } });
  expect(masked.rangeOf("call")).toEqual(anchor.rangeOf("nodes")!);
  expect(keysOf(masked)).toEqual(["kind", "call", "input"]);
  expect(mappingSnapshot(mergedBefore)).toEqual(originalMergedPairs);
  expect(mappingKeys(mergedBefore)).toEqual(originalMergedKeys);
  expect(document.root.value).toEqual(originalDocumentValue);
});

test("a merged body's local override uses the effective values and ranges", () => {
  const document = SourceDocument.parse(`
body: &body
  kind: call
  nodes:
    anchored:
      kind: code
      run: ./fixtures/compose_codefns.ts:echo
      input:
        value: anchored
      output: str
  output: \${anchored.output}
nodes:
  merged:
    <<: *body
    nodes:
      local:
        kind: code
        run: ./fixtures/compose_codefns.ts:echo
        input:
          value: local
        output: str
    output: \${local.output}
`);
  const anchor = document.root.child("body");
  const originalNodes = document.root.child("nodes");
  const mergedBefore = originalNodes.child("merged");
  const nodesPairs = mergedBefore.mappingOccurrences("nodes");
  const outputPairs = mergedBefore.mappingOccurrences("output");
  expect(nodesPairs.length).toBe(2);
  expect(outputPairs.length).toBe(2);
  const localNodesRange = nodesPairs.at(-1)!.range;
  const localOutputRange = outputPairs.at(-1)!.range;
  const originalMergedPairs = mappingSnapshot(mergedBefore);
  const originalAnchorPairs = mappingSnapshot(anchor);
  const originalMergedKeys = mappingKeys(mergedBefore);
  const originalAnchorKeys = mappingKeys(anchor);
  const originalDocumentValue = document.root.value;

  const mergedValue = mergedBefore.value as ValueRecord;
  expect(Object.keys(mergedValue["nodes"] as ValueRecord)).toEqual(["local"]);
  expect(mergedValue["output"]).toBe("${local.output}");

  const [maskedNodes, bodies] = extractInlineBodies(originalNodes, { owner: ["root"] });

  const body = bodies.get("merged")!;
  expect(body.nodesCursor.value).toEqual(mergedValue["nodes"]!);
  expect(body.nodesCursor.range).toEqual(localNodesRange!);
  expect(keysOf(body.nodesCursor)).toEqual(["local"]);
  expect(body.output).toBe("${local.output}");
  expect(body.outputRange).toEqual(localOutputRange!);

  const masked = maskedNodes.child("merged");
  expect(masked.value).toEqual({ kind: "call", call: body.flowId });
  expect(masked.mappingOccurrences("call").length).toBe(1);
  expect(masked.rangeOf("call")).toEqual(localNodesRange!);

  expect(mappingSnapshot(mergedBefore)).toEqual(originalMergedPairs);
  expect(mappingSnapshot(anchor)).toEqual(originalAnchorPairs);
  expect(mappingKeys(mergedBefore)).toEqual(originalMergedKeys);
  expect(mappingKeys(anchor)).toEqual(originalAnchorKeys);
  expect(document.root.value).toEqual(originalDocumentValue);
});

const RESERVED = (nodeId: string) =>
  `node '${nodeId}' (kind=call): call targets beginning with '__inline_flow_' are reserved for ` +
  "compiler-generated inline child flows";

test("an authored private call target is rejected without mutating the source", () => {
  const document = SourceDocument.parse(`nodes:
  attacker:
    kind: call
    call: ${INLINE_FLOW_PREFIX}not-generated
`);
  const originalNodes = document.root.child("nodes");
  const attacker = originalNodes.child("attacker");
  const originalValue = attacker.value;
  const originalPairs = mappingSnapshot(attacker);
  const originalDocumentValue = document.root.value;

  const error = loadError(() => extractInlineBodies(originalNodes, { owner: ["root"] }));

  expect(error.message).toBe(RESERVED("attacker"));
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(attacker.rangeOf("call")!);
  expect(attacker.value).toEqual(originalValue);
  expect(mappingSnapshot(attacker)).toEqual(originalPairs);
  expect(document.root.value).toEqual(originalDocumentValue);
});

test("a recursive alias the private-target preflight reaches is a located load error", () => {
  const source = `id: recursive-inline-source
name: recursive_inline_source
nodes: &nodes
  kind: call
  call: ${INLINE_FLOW_PREFIX}not-generated
  recursive: *nodes
output: unused
`;

  const error = loadError(() => assembleText(source));

  expect(error.message).toBe("cannot descend through a recursive alias occurrence");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(new SourceRange(3, 8, 7, 1));
});

test.each([
  ["before", true],
  ["after", false],
])("a sibling private id attack %s the victim is rejected before any body is masked", (_name, attackerFirst) => {
  const victimTarget = inlineFlowId(["root", "node", "victim", "inline"]);
  const attacker = `  attacker:\n    kind: call\n    call: ${victimTarget}\n`;
  const victim = `  victim:
    kind: call
    nodes:
      echo:
        kind: code
        run: ./fixtures/compose_codefns.ts:echo
        input:
          value: hello
        output: str
    output: \${echo.output}
`;
  const [first, second] = attackerFirst ? [attacker, victim] : [victim, attacker];
  const document = SourceDocument.parse(`nodes:\n${first}${second}`);
  const originalNodes = document.root.child("nodes");
  const attackerCursor = originalNodes.child("attacker");
  const victimCursor = originalNodes.child("victim");
  const originalAttacker = attackerCursor.value;
  const originalVictim = victimCursor.value;
  const originalAttackerPairs = mappingSnapshot(attackerCursor);
  const originalVictimPairs = mappingSnapshot(victimCursor);
  const originalDocumentValue = document.root.value;
  const projections = spyOn(SourceCursor.prototype, "projectMapping");

  let error: LoadError;
  try {
    error = loadError(() => extractInlineBodies(originalNodes, { owner: ["root"] }));
    expect(projections).not.toHaveBeenCalled();
  } finally {
    projections.mockRestore();
  }

  expect(error.message).toBe(RESERVED("attacker"));
  expect(error.message).not.toContain(victimTarget);
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(attackerCursor.rangeOf("call")!);
  expect(attackerCursor.value).toEqual(originalAttacker);
  expect(victimCursor.value).toEqual(originalVictim);
  expect(mappingSnapshot(attackerCursor)).toEqual(originalAttackerPairs);
  expect(mappingSnapshot(victimCursor)).toEqual(originalVictimPairs);
  expect(document.root.value).toEqual(originalDocumentValue);
});

test("extraction allows the private prefix in a graph node id", () => {
  const root = SourceDocument.parse(`nodes:
  ${INLINE_FLOW_PREFIX}host:
    kind: call
    call: named
`).root;

  const [maskedNodes, bodies] = extractInlineBodies(root.child("nodes"), { owner: ["root"] });

  expect(bodies.size).toBe(0);
  expect((maskedNodes.value as ValueRecord)[`${INLINE_FLOW_PREFIX}host`]).toEqual({ kind: "call", call: "named" });
});

test("the loader allows the private callable prefix in a graph node id", () => {
  const { flow } = assembleText(`id: prefix-node
name: prefix_node
input:
  x: !str
nodes:
  ${INLINE_FLOW_PREFIX}host:
    kind: code
    code: ./fixtures/seeds.ts:one_line_summary
    input:
      x: \${input.x}
output: \${${INLINE_FLOW_PREFIX}host.output}
`);

  expect(flow.compiled.nodes.has(`${INLINE_FLOW_PREFIX}host`)).toBe(true);
});

const INLINE_CHILD = `nodes:
      echo:
        kind: code
        run: ./fixtures/compose_codefns.ts:echo
        input:
          value: hello
        output: str
`;

/**
 * Each malformed body: its name, its kind, the body text, the key its range is at (else the body),
 * and the message.
 */
const INLINE_SHAPE_CASES: [string, string, string, string | null, string][] = [
  [
    "both-call-and-nodes",
    "map",
    `call: named\n    ${INLINE_CHILD}    output: \${echo.output}\n`,
    "nodes",
    "node 'composite' (kind=map): exactly one of `call:` or inline `nodes:` is required",
  ],
  [
    "neither-call-nor-nodes",
    "call",
    "input: {}\n",
    null,
    "node 'composite' (kind=call): exactly one of `call:` or inline `nodes:` is required",
  ],
  ["inline-missing-output", "loop", INLINE_CHILD, null, "node 'composite' (kind=loop): inline `nodes:` requires `output:`"],
  [
    "named-with-output",
    "call",
    "call: named\n    output: ${named.output}\n",
    "output",
    "node 'composite' (kind=call): `output:` is only valid with inline `nodes:`",
  ],
  [
    "empty-nodes",
    "call",
    "nodes: {}\n    output: ${missing.output}\n",
    "nodes",
    "node 'composite' (kind=call): inline `nodes:` must be a non-empty mapping",
  ],
  [
    "non-mapping-nodes",
    "map",
    "nodes: not-a-map\n    output: value\n",
    "nodes",
    "node 'composite' (kind=map): inline `nodes:` must be a non-empty mapping",
  ],
  [
    "non-mapping-input",
    "call",
    `input: not-a-map\n    ${INLINE_CHILD}    output: \${echo.output}\n`,
    "input",
    "node 'composite' (kind=call): inline `input:` must be a mapping with string keys",
  ],
];

test.each(INLINE_SHAPE_CASES)(
  "inline body shape error %s is located without private ids",
  (_name, kind, body, rangeKey, message) => {
    const nodesCursor = SourceDocument.parse(`nodes:\n  composite:\n    kind: ${kind}\n    ${body}`).root.child("nodes");
    const rawBody = nodesCursor.child("composite");

    const error = loadError(() => extractInlineBodies(nodesCursor, { owner: ["root"] }));

    expect(error.message).toBe(message);
    expect(error.message).not.toContain(INLINE_FLOW_PREFIX);
    expect(error.diagnostic.frames.at(-1)!.range).toEqual(rangeKey ? rawBody.rangeOf(rangeKey)! : rawBody.range!);
  },
);

test.each(INLINE_SHAPE_CASES)(
  "inline body shape error %s flows through the loader",
  (_name, kind, body, rangeKey, message) => {
    const source =
      "id: inline-shape-loader\nname: inline_shape_loader\nnodes:\n  composite:\n" + `    kind: ${kind}\n    ${body}`;
    const rawBody = SourceDocument.parse(source).root.child("nodes").child("composite");

    const error = loadError(() => assembleText(source));

    expect(error.message).toBe(message);
    expect(error.message).not.toContain(INLINE_FLOW_PREFIX);
    expect(error.diagnostic.frames.at(-1)!.range).toEqual(rangeKey ? rawBody.rangeOf(rangeKey)! : rawBody.range!);
  },
);

test("an inline body's non-text input key fails at the key through the loader", () => {
  const source =
    "id: inline-shape-loader\n" +
    "name: inline_shape_loader\n" +
    "nodes:\n" +
    "  composite:\n" +
    "    kind: loop\n" +
    "    input:\n" +
    "      1: value\n" +
    `    ${INLINE_CHILD}    output: \${echo.output}\n`;

  const error = loadError(() => assembleText(source));

  expect(error.message).toBe(
    'the key `1` was read by YAML as an integer, but names in a flow must be text. Quote it ("1") or ' +
      "choose another name.",
  );
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(new SourceRange(7, 7, 7, 8));
});

test("the loader rejects an actual sibling's private digest without exposing it", () => {
  const victimId = inlineFlowId(["root", "node", "victim", "inline"]);
  const source = `id: inline-sibling-digest
name: inline_sibling_digest
nodes:
  victim:
    kind: call
    nodes:
      echo:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo_x
        input:
          x: safe
        output: str
    output: \${echo.output}
  attacker:
    kind: call
    call: ${victimId}
`;
  const expectedRange = SourceDocument.parse(source).root.child("nodes").child("attacker").rangeOf("call");

  const error = loadError(() => assembleText(source));

  expect(error.message).toBe(RESERVED("attacker"));
  expect(error.message).not.toContain(victimId);
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(expectedRange!);
});

/** A descriptor's fields, leaving out `childLabel` as Python's `compare=False` does. */
function withoutChildLabel(desc: NodeDescriptor): Record<string, unknown> {
  const { childLabel: _childLabel, ...fields } = desc as NodeDescriptor & { childLabel: unknown };
  return fields;
}

test("registerInlineBodies associates CALL, MAP and LOOP bodies and stays lazy", () => {
  const [descriptors, bodies] = associationFixture();
  const originals = new Map(descriptors);
  const registrar = new RecordingRegistrar();
  const flows = new Map([...bodies.keys()].map((nodeId) => [nodeId, handChild({ output: null })]));
  const buildCalls: [InlineBody, CallDescriptor | LoopDescriptor][] = [];

  const associated = registerInlineBodies(descriptors, bodies, {
    registrar,
    buildChild: (body, descriptor) => {
      buildCalls.push([body, descriptor]);
      return flows.get(body.nodeId)!;
    },
  });

  expect(associated).not.toBe(descriptors);
  expect([...associated.keys()]).toEqual(["call_node", "map_node", "loop_node"]);
  expect(registrar.registrations.length).toBe(3);
  expect(buildCalls).toEqual([]);

  const registrations = registrar.registrations;
  for (const [index, body] of [...bodies.values()].entries()) {
    const [flowId, owner, label, builder] = registrations[index]!;
    const expectedLabel = `inline body of node '${body.nodeId}'`;
    const rewritten = associated.get(body.nodeId) as CallDescriptor | LoopDescriptor;
    const original = originals.get(body.nodeId) as CallDescriptor | LoopDescriptor;

    expect(flowId).toBe(body.flowId);
    expect(owner).toEqual(body.owner);
    expect(label).toBe(expectedLabel);
    expect(rewritten).not.toBe(original);
    expect(rewritten.constructor).toBe(original.constructor);
    expect(withoutChildLabel(rewritten)).toEqual(withoutChildLabel(original));
    expect(rewritten.childLabel).toBe(expectedLabel);
    expect(rewritten.call).toBe(body.flowId);
    expect(descriptors.get(body.nodeId)).toBe(original);
    expect(original.childLabel).toBeNull();

    expect(builder()).toBe(flows.get(body.nodeId)!);
    expect(buildCalls.at(-1)![0]).toBe(body);
    expect(buildCalls.at(-1)![1]).toBe(rewritten);
  }
});

test.each([
  ["missing", "inline body of node 'call_node': no matching composite node descriptor"],
  ["wrong-kind", "inline body of node 'call_node': final node descriptor is not CALL, MAP, or LOOP"],
  [
    "wrong-target",
    "inline body of node 'call_node': final node descriptor does not target its retained inline body",
  ],
])("registerInlineBodies rejects a bad association, located and redacted: %s", (failure, message) => {
  const [descriptors, bodies] = associationFixture();
  const body = bodies.get("call_node")!;
  const call = descriptors.get("call_node") as CallDescriptor;
  const candidate = new Map<string, NodeDescriptor>();
  if (failure === "wrong-kind") candidate.set("call_node", new CodeDescriptor({ id: "call_node", code: "return 1" }));
  if (failure === "wrong-target") candidate.set("call_node", call.replace({ call: "named" }));
  const registrar = new RecordingRegistrar();

  const error = loadError(() =>
    registerInlineBodies(candidate, new Map([["call_node", body]]), {
      registrar,
      buildChild: () => handChild({ output: null }),
    }),
  );

  expect(error.message).toBe(message);
  expect(error.message).not.toContain(body.flowId);
  expect(error.message).not.toContain(INLINE_FLOW_PREFIX);
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(body.nodesCursor.range!);
  expect(registrar.registrations).toEqual([]);
});

test("registerInlineBodies checks every body before the first registration", () => {
  const [descriptors, bodies] = associationFixture();
  const broken = new Map(descriptors);
  broken.delete("loop_node");
  const registrar = new RecordingRegistrar();

  const error = loadError(() =>
    registerInlineBodies(broken, bodies, { registrar, buildChild: () => handChild({ output: null }) }),
  );

  expect(error.message).toBe("inline body of node 'loop_node': no matching composite node descriptor");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(bodies.get("loop_node")!.nodesCursor.range!);
  expect(registrar.registrations).toEqual([]);
});

test.each([
  ["call", "call: named", "asserts, call, depends_on, env, input, kind, node_name, runs_after"],
  [
    "map",
    "call: named\n    over: ${input.items}",
    "asserts, call, depends_on, env, input, kind, node_name, over, parallel, runs_after",
  ],
  [
    "loop",
    "call: named\n    times: 1",
    "asserts, call, depends_on, env, input, kind, max, node_name, runs_after, times, until, while",
  ],
])("child_label is not authorable on %s", (kind, body, allowed) => {
  const nodes = SourceDocument.parse(
    `nodes:\n  composite:\n    kind: ${kind}\n    ${body}\n    child_label: forged\n`,
  ).root.child("nodes");

  const error = loadError(() => parseNodes(nodes));

  expect(error.message).toBe(
    `node 'composite' (kind=${kind}): field 'child_label' is not allowed (allowed: ${allowed})`,
  );
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(nodes.child("composite").rangeOf("child_label")!);
});

/** A build context resolving every callable to a body whose node `constant` is typed `str`. */
function bodyContext(output: string | null): BuildContext {
  const body = handChild({ producers: { constant: new ValueContract(Type.scalar(ValueKind.STRING)) }, output });
  return new BuildContext({ registry: new Map(), resolver: () => body });
}

test("an inline call's undeclared argument names the body by its child label", () => {
  const desc = new CallDescriptor({
    id: "each",
    call: "inline-body",
    inputs: { topic: "${input.topic}" },
    childLabel: "inline body of node 'each'",
  });

  const error = loadError(() => desc.build(bodyContext("${constant.output}")));

  expect(error.message).toBe(
    "call node 'each': binding 'topic' is not a declared input of inline body of node 'each'",
  );
});

// ---------- inline children built at load ----------

/** The generated parameter expected for an inline body input of type `typ`. */
function parameter(name: string, typ: Type): Parameter {
  return new Parameter(name, { contract: new ValueContract(typ), required: !typ.nullable, coerceStrings: true });
}

/** The declared type of a generated parameter. */
function parameterType(param: Parameter): Type {
  return (param.contract as ValueContract).type;
}

/** The compiled node `nodeId` of `flow`: a CALL, MAP or LOOP driver. */
function driver<T extends CallNode | MapNode | LoopNode>(flow: ComposedFlow, nodeId: string): T {
  return flow.compiled.nodes.get(nodeId) as T;
}

/** The authored range of node `nodeId`'s binding `input.<key>` in flow `source`. */
function inputRange(source: string, nodeId: string, key: string): SourceRange {
  return SourceDocument.parse(source).root.child("nodes").child(nodeId).mappingValueRanges("input").get(key)!;
}

/**
 * Run `load` with every `CallableScope`'s private registrations and its resolutions of private
 * ids recorded: `[scope, flowId, owner, label]` and `[scope, flowId]`.
 */
async function recordPrivateScope<T>(load: () => Promise<T>) {
  const register = spyOn(CallableScope.prototype, "registerPrivate");
  const resolve = spyOn(CallableScope.prototype, "resolve");
  try {
    const result = await load();
    const registrations = register.mock.calls.map(([flowId, owner, label], index) => [
      register.mock.contexts[index],
      flowId,
      owner,
      label,
    ]);
    const resolutions = resolve.mock.calls.flatMap(([flowId], index) =>
      flowId.startsWith(INLINE_FLOW_PREFIX) ? [[resolve.mock.contexts[index], flowId]] : [],
    );
    return { result, registrations, resolutions };
  } finally {
    register.mockRestore();
    resolve.mockRestore();
  }
}

/** Load `text` and run it on `inputs`; expect it to succeed; return its output. */
async function runOutput(text: string, inputs: Record<string, unknown> = {}): Promise<unknown> {
  const result = await runFlow(await loadConformanceFlow(text), inputs);
  expect(result.status, result.failure?.message).toBe("succeeded");
  return result.output;
}

/**
 * Load `text` and run it on `inputs` with `numWorkers` (`recordPrivateScope` around the run);
 * expect the run to register and resolve no private flow beyond the load's, and to succeed; return
 * its output.
 */
async function runOutputRegisteringNoMore(
  text: string,
  inputs: Record<string, unknown> = {},
  numWorkers?: number,
): Promise<unknown> {
  const loaded = await loadConformanceFlow(text);
  const { result, registrations, resolutions } = await recordPrivateScope(() =>
    runFlow(loaded, inputs, numWorkers === undefined ? {} : { numWorkers }),
  );
  expect(registrations).toEqual([]);
  expect(resolutions).toEqual([]);
  expect(result.status, result.failure?.message).toBe("succeeded");
  return result.output;
}

const ZERO_INPUT = `id: inline-call-zero-input
name: inline_call_zero_input
nodes:
  wrapper:
    kind: call
    nodes:
      constant:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input:
          topic: inline-ok
        output: str
    output: \${constant.output}
output: \${wrapper.output}
`;

test("a zero-input inline call loads in one file scope", async () => {
  const { result: flow, registrations, resolutions } = await recordPrivateScope(() => loadConformanceFlow(ZERO_INPUT));

  const owner = ["root", "node", "wrapper", "inline"];
  const flowId = inlineFlowId(owner);
  const wrapper = driver<CallNode>(flow, "wrapper");
  expect(wrapper.flowId).toBe(flowId);
  expect(wrapper.childInputs).toEqual([]);
  expect(registrations).toHaveLength(1);
  const scope = registrations[0]![0];
  expect(scope).toBeInstanceOf(CallableScope);
  expect(registrations[0]).toEqual([scope, flowId, owner, "inline body of node 'wrapper'"]);
  expect(resolutions).toHaveLength(1);
  expect(resolutions[0]![0]).toBe(scope);
  expect(resolutions[0]![1]).toBe(flowId);
});

test("a zero-input inline call runs in one file scope", async () => {
  // The run registers and resolves nothing more: the load's one registration and resolution stand.
  expect(await runOutputRegisteringNoMore(ZERO_INPUT)).toBe("inline-ok");
});

const NULLABLE_INPUT = `id: inline-call-nullable-input
name: inline_call_nullable_input
input:
  maybe: !Optional(int)
nodes:
  wrapper:
    kind: call
    input:
      value: \${input.maybe}
    nodes:
      echo:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo_x
        input:
          x: \${input.value}
        output: Optional(int)
    output: \${echo.output}
output: \${wrapper.output}
`;

test("a typed inline call preserves a nullable parent scalar", () => {
  const { flow } = assembleText(NULLABLE_INPUT);

  expect(driver<CallNode>(flow, "wrapper").childInputs).toEqual([
    parameter("value", new Type({ kind: ValueKind.INTEGER, nullable: true })),
  ]);
});

test.each([null, 7n])("a typed inline call runs a nullable parent scalar: %p", async (value) => {
  expect(await runOutput(NULLABLE_INPUT, { maybe: value })).toBe(value);
});

const RECORD_INPUT = `id: inline-call-record-input
name: inline_call_record_input
nodes:
  wrapper:
    kind: call
    input:
      payload:
        z: tail
        a: 3
      marker: before
    nodes:
      echo:
        kind: code
        code: |
          return {"payload": inputs["payload"], "marker": inputs["marker"]}
        input:
          payload: \${input.payload}
          marker: \${input.marker}
        output:
          payload:
            a: int
            z: str
          marker: str
    output: \${echo.output}
output: \${wrapper.output}
`;

test("a typed inline call canonicalizes a literal record and keeps its input order", () => {
  const { flow } = assembleText(RECORD_INPUT);

  const wrapper = driver<CallNode>(flow, "wrapper");
  expect(wrapper.childInputs.map((param) => param.name)).toEqual(["payload", "marker"]);
  const [payload, marker] = wrapper.childInputs as [Parameter, Parameter];
  expect([payload.hasDefault, payload.default, payload.required]).toEqual([false, null, true]);
  expect([...parameterType(payload).fields!.keys()]).toEqual(["a", "z"]);
  expect(parameterType(payload)).toEqual(
    new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([
        ["a", Type.scalar(ValueKind.INTEGER)],
        ["z", Type.scalar(ValueKind.STRING)],
      ]),
      required: new Set(["a", "z"]),
    }),
  );
  expect([marker.hasDefault, marker.default, marker.required]).toEqual([false, null, true]);
  expect(parameterType(marker)).toEqual(Type.scalar(ValueKind.STRING));
});

test("a typed inline call runs a literal record", async () => {
  expect(await runOutput(RECORD_INPUT)).toEqual({ payload: { z: "tail", a: 3n }, marker: "before" });
});

const FORWARD_PRODUCER = `id: inline-call-forward-producer
name: inline_call_forward_producer
nodes:
  consumer:
    kind: call
    input:
      value: \${producer.output}
    nodes:
      echo:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo_x
        input:
          x: \${input.value}
        output: str
    output: \${echo.output}
  producer:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: forward-value
    output: str
output: \${consumer.output}
`;

test("a forward producer's type builds it once and keeps source order", () => {
  const builds = spyOn(CodeDescriptor.prototype, "build");
  let flow: ComposedFlow;
  let producerBuilds: number;
  try {
    ({ flow } = assembleText(FORWARD_PRODUCER));
    producerBuilds = builds.mock.contexts.filter((desc) => (desc as CodeDescriptor).id === "producer").length;
  } finally {
    builds.mockRestore();
  }

  const bodyOrder = [...flow.compiled.nodes.keys()].filter((nodeId) => nodeId !== START_ID && nodeId !== END_ID);
  expect(producerBuilds).toBe(1);
  expect(bodyOrder).toEqual(["consumer", "producer"]);
  expect(driver<CallNode>(flow, "consumer").childInputs).toEqual([
    parameter("value", Type.scalar(ValueKind.STRING)),
  ]);
});

test("a forward producer runs, built once", async () => {
  const builds = spyOn(CodeDescriptor.prototype, "build");
  let output: unknown;
  let producerBuilds: number;
  try {
    output = await runOutput(FORWARD_PRODUCER);
    producerBuilds = builds.mock.contexts.filter((desc) => (desc as CodeDescriptor).id === "producer").length;
  } finally {
    builds.mockRestore();
  }

  expect(producerBuilds).toBe(1); // the run builds nothing again
  expect(output).toBe("forward-value");
});

const NONE_PRODUCER = `id: inline-call-none-producer
name: inline_call_none_producer
nodes:
  consumer:
    kind: call
    input:
      value: \${producer.output}
    nodes:
      echo:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo_x
        input:
          x: \${input.value}
        output: "None"
    output: \${echo.output}
  producer:
    kind: code
    code: |
      return null
    output: "None"
output: \${consumer.output}
`;

test("a producer declared `None` is not treated as a missing output", () => {
  const { flow } = assembleText(NONE_PRODUCER);

  expect(driver<CallNode>(flow, "consumer").childInputs).toEqual([parameter("value", Type.scalar(ValueKind.NONE))]);
});

test("a producer declared `None` runs to null", async () => {
  expect(await runOutput(NONE_PRODUCER)).toBeNull();
});

test("an inline signature self-cycle names the binding path, at its range", () => {
  const source = `id: inline-signature-self-cycle
name: inline_signature_self_cycle
nodes:
  first:
    kind: call
    input:
      value: \${first.output}
    nodes:
      echo:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo_x
        input:
          x: \${input.value}
        output: str
    output: \${echo.output}
output: \${first.output}
`;

  const error = loadError(() => assembleText(source));

  expect(error.message).toBe("node build dependency cycle: first.input.value -> first");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(inputRange(source, "first", "value"));
});

test("an inline signature mutual cycle is located at the closing binding", () => {
  const source = `id: inline-signature-mutual-cycle
name: inline_signature_mutual_cycle
nodes:
  left:
    kind: call
    input:
      from_right: \${right.output}
    nodes:
      echo_left:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo_x
        input:
          x: \${input.from_right}
        output: str
    output: \${echo_left.output}
  right:
    kind: call
    input:
      from_left: \${left.output}
    nodes:
      echo_right:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo_x
        input:
          x: \${input.from_left}
        output: str
    output: \${echo_right.output}
output: \${left.output}
`;

  const error = loadError(() => assembleText(source));

  expect(error.message).toBe(
    "node build dependency cycle: left.input.from_right -> right.input.from_left -> left",
  );
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(inputRange(source, "right", "from_left"));
});

/** A flow whose inline call `consumer` binds `value` to `${<producer>.output}`, then `extra`. */
function consumerOf(producer: string, extra = ""): string {
  return `id: inline-signature-producer
name: inline_signature_producer
nodes:
  consumer:
    kind: call
    input:
      value: \${${producer}.output}
    nodes:
      echo:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo_x
        input:
          x: \${input.value}
        output: str
    output: \${echo.output}
${extra}output: \${consumer.output}
`;
}

test.each([
  ["an unknown producer", "missing", ""],
  [
    "a producer without an output",
    "producer",
    "  producer:\n    kind: code\n    code: ./fixtures/compose_codefns.ts:echo\n    input:\n" +
      "      topic: no-static-output\n",
  ],
])("an inline signature over %s is located and strict", (_name, producer, extra) => {
  const source = consumerOf(producer, extra);

  const error = loadError(() => assembleText(source));

  expect(error.message).toBe(
    `cannot infer type for inline binding 'consumer.input.value': producer '${producer}' has no available ` +
      "static type; declare its type with a tag (`value: !<type> ...`) or use an explicitly typed named " +
      "`defs:` body",
  );
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(inputRange(source, "consumer", "value"));
});

const ALIAS_MERGE = `id: inline-alias-merge
name: inline_alias_merge
nodes:
  first: &body
    kind: call
    input:
      x: base
    nodes:
      echo:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo_x
        input:
          x: \${input.x}
        output: str
    output: \${echo.output}
  second: *body
  merged:
    <<: *body
    input:
      x: merged
output:
  first: \${first.output}
  second: \${second.output}
  merged: \${merged.output}
`;

test("alias and merge occurrences assemble as distinct private children", () => {
  const { flow } = assembleText(ALIAS_MERGE);

  const calls = ["first", "second", "merged"].map((nodeId) => driver<CallNode>(flow, nodeId));
  expect(new Set(calls.map((node) => node.flowId)).size).toBe(3);
  expect(new Set(calls.map((node) => node.child)).size).toBe(3);
});

test("alias and merge occurrences run as distinct private children", async () => {
  expect(await runOutput(ALIAS_MERGE)).toEqual({ first: "base", second: "base", merged: "merged" });
});

const NESTED_CONTAINERS = `id: inline-nested-containers
name: inline_nested_containers
input:
  x: !str
nodes:
  producer:
    kind: code
    code: |
      return "resolved"
    output: str
  wrapper:
    kind: call
    input:
      payload:
        producer: \${producer.output}
        parent: \${input.x}
        dollars: $$
        deep:
          - \${producer.output}
          - got \${input.x}
    nodes:
      echo:
        kind: code
        code: |
          return inputs["payload"]
        input:
          payload: \${input.payload}
        output:
          producer: str
          parent: str
          dollars: str
          deep: list(str)
    output: \${echo.output}
output: \${wrapper.output}
`;

test("nested containers in an inline binding are wired at the edges", () => {
  const { flow } = assembleText(NESTED_CONTAINERS);

  expect(flow.compiled.edges).toContainEqual(expect.objectContaining({ from: "producer", to: "wrapper" }));
});

test("nested containers in an inline binding are evaluated at runtime", async () => {
  expect(await runOutput(NESTED_CONTAINERS, { x: "parent" })).toEqual({
    producer: "resolved",
    parent: "parent",
    dollars: "$",
    deep: ["resolved", "got parent"],
  });
});

const OPTIONAL_RECORD_PATHS = `id: inline-optional-record-paths
name: inline_optional_record_paths
typedefs:
  Inner:
    value: str
  Outer:
    inner: Optional(Inner)
input:
  root: !Optional(Inner)
  outer: !Outer
nodes:
  wrapper:
    kind: call
    input:
      root_value: \${input.root.value}
      nested_value: \${input.outer.inner.value}
    nodes:
      collect:
        kind: code
        code: |
          return {"root": inputs["root_value"], "nested": inputs["nested_value"]}
        input:
          root_value: \${input.root_value}
          nested_value: \${input.nested_value}
        output:
          root: Optional(str)
          nested: Optional(str)
    output: \${collect.output}
output: \${wrapper.output}
`;

test("optional record paths make generated inputs nullable", () => {
  const { flow } = assembleText(OPTIONAL_RECORD_PATHS);

  expect(
    driver<CallNode>(flow, "wrapper").childInputs.map((param) => [
      param.name,
      param.required,
      parameterType(param).nullable,
    ]),
  ).toEqual([
    ["root_value", false, true],
    ["nested_value", false, true],
  ]);
});

test("optional record paths run to null and to their values", async () => {
  expect(await runOutput(OPTIONAL_RECORD_PATHS, { root: null, outer: { inner: null } })).toEqual({
    root: null,
    nested: null,
  });
  expect(
    await runOutput(OPTIONAL_RECORD_PATHS, { root: { value: "root" }, outer: { inner: { value: "nested" } } }),
  ).toEqual({ root: "root", nested: "nested" });
});

test("inline input inference projects a known field of a partial LOOP output", () => {
  const body = handChild({
    inputs: parametersFromYaml("input:\n  n: !int\n  note: !str\n", new Map()),
    producers: { note: null },
    output: { n: "${input.n}", note: "${note.output}" },
  });

  const { flow } = assembleText(
    `id: inline-partial-producer-field
name: inline_partial_producer_field
nodes:
  loop:
    kind: loop
    call: body
    input:
      n: 7
      note: seed
    times: 1
  wrapper:
    kind: call
    input:
      value: \${loop.output.n}
    nodes:
      echo:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo_x
        input:
          x: \${input.value}
        output: int
    output: \${echo.output}
output: \${wrapper.output}
`,
    "flow.yaml",
    { children: { body } },
  );

  expect(driver<CallNode>(flow, "wrapper").childInputs).toEqual([parameter("value", Type.scalar(ValueKind.INTEGER))]);
});

test("inline input inference runs a known field of a partial LOOP output", async () => {
  // The flow with the `body` def Python writes, its `note` an expression output.
  const output = await runOutput(`id: inline-partial-producer-field
name: inline_partial_producer_field
defs:
  body:
    input:
      n: !int
      note: !str
    nodes: {}
    output:
      n: \${input.n}
      note: \${input.note + "!"}
nodes:
  loop:
    kind: loop
    call: body
    input:
      n: 7
      note: seed
    times: 1
  wrapper:
    kind: call
    input:
      value: \${loop.output.n}
    nodes:
      echo:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo_x
        input:
          x: \${input.value}
        output: int
    output: \${echo.output}
output: \${wrapper.output}
`);

  expect(output).toBe(7n);
});

test("an inline multi-node body matches the named def's signature", async () => {
  const body = `    nodes:
      echo:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo_x
        input:
          x: \${input.x}
        output: int
      pack:
        kind: code
        code: |
          return {"value": inputs["value"]}
        input:
          value: \${echo.output}
        output:
          value: int
    output: \${pack.output}
`;
  const loaded = await loadConformanceFlow(`id: inline-named-equivalence
name: inline_named_equivalence
defs:
  named:
    input:
      x: !int
${body}nodes:
  named_call:
    kind: call
    call: named
    input:
      x: 5
  inline_call:
    kind: call
    input:
      x: 5
${body}output:
  named: \${named_call.output}
  inline: \${inline_call.output}
`);

  // A parameter's signature, its diagnostic-only origins left out.
  const semantics = (param: Parameter) => [
    param.name,
    parameterType(param),
    param.required,
    param.hasDefault,
    param.default,
    param.coerceStrings,
  ];
  const named = driver<CallNode>(loaded, "named_call");
  const inline = driver<CallNode>(loaded, "inline_call");
  expect(named.childInputs.map(semantics)).toEqual(inline.childInputs.map(semantics));
  expect((named.outputContract as ValueContract).type).toEqual((inline.outputContract as ValueContract).type);
  const bodyIds = (node: CallNode) =>
    [...(node.child as CompiledFlow).nodes.keys()].filter((id) => id !== START_ID && id !== END_ID);
  expect(bodyIds(named)).toEqual(["echo", "pack"]);
  expect(bodyIds(inline)).toEqual(["echo", "pack"]);
});

const EAGER_DEF_INLINE_CALL = `id: eager-def-inline-call
name: eager_def_inline_call
defs:
  outer:
    nodes:
      wrapper:
        kind: call
        nodes:
          constant:
            kind: code
            code: ./fixtures/compose_codefns.ts:echo
            input:
              topic: inline-ok
            output: str
        output: \${constant.output}
    output: \${wrapper.output}
nodes:
  invoke:
    kind: call
    call: outer
output: \${invoke.output}
`;

test("an eager def's inline call takes its origin and shares the file scope", async () => {
  const { result: loaded, registrations, resolutions } = await recordPrivateScope(() =>
    loadConformanceFlow(EAGER_DEF_INLINE_CALL),
  );

  const owner = ["def", "outer", "node", "wrapper", "inline"];
  const flowId = inlineFlowId(owner);
  const wrapper = (driver<CallNode>(loaded, "invoke").child as CompiledFlow).nodes.get("wrapper") as CallNode;
  const keyRange = SourceDocument.parse(EAGER_DEF_INLINE_CALL)
    .root.child("defs")
    .child("outer")
    .child("nodes")
    .child("wrapper").keyRange!;
  expect(wrapper.flowId).toBe(flowId);
  expect(wrapper.childInputs).toEqual([]);
  expect(wrapper.origin).not.toBeNull();
  expect(loaded.originResolver!.resolve(wrapper.origin!)!.range).toEqual(keyRange);
  expect(registrations).toHaveLength(1);
  expect(resolutions).toHaveLength(1);
  const scope = resolutions[0]![0];
  expect(registrations[0]).toEqual([scope, flowId, owner, "inline body of node 'wrapper'"]);
  expect(registrations[0]![0]).toBe(scope);
  expect(resolutions[0]![1]).toBe(flowId);
});

test("an eager def's inline call runs in the shared file scope", async () => {
  // The run registers and resolves nothing more: the load's one registration and resolution stand.
  expect(await runOutputRegisteringNoMore(EAGER_DEF_INLINE_CALL)).toBe("inline-ok");
});

test("a def's assembly cursor fallback restores only node surfaces", () => {
  const literal = { inputs: "literal-input", outputs: "literal-output" };
  const normalized = {
    wrapper: {
      kind: "call",
      inputs: { payload: literal },
      nodes: {
        constant: { kind: "code", inputs: { topic: "inline-ok" }, outputs: "str", env: { literal } },
      },
      outputs: "${constant.output}",
    },
  };

  const cursor = defAssemblyCursor(null, normalized);

  const wrapper = (cursor.value as ValueRecord)["wrapper"] as ValueRecord;
  const constant = (wrapper["nodes"] as ValueRecord)["constant"] as ValueRecord;
  expect(cursor.document).toBeNull();
  expect(Object.keys(wrapper)).toEqual(["kind", "nodes", "input", "output"]);
  expect(Object.keys(constant)).toEqual(["kind", "env", "input", "output"]);
  expect((wrapper["input"] as ValueRecord)["payload"]).toBe((constant["env"] as ValueRecord)["literal"]);
  expect(literal).toEqual({ inputs: "literal-input", outputs: "literal-output" });
  expect(Object.keys(normalized.wrapper)).toEqual(["kind", "inputs", "nodes", "outputs"]);
});

test("an inline MAP over a literal list loads with its inferred item type", () => {
  const { flow } = assembleText(`id: inline-map-literal-over
name: inline_map_literal_over
nodes:
  each:
    kind: map
    over: [1, 2]
    input:
      value: \${item}
    nodes:
      echo:
        kind: code
        code: |
          return inputs["value"]
        input:
          value: \${input.value}
        output: int
    output: \${echo.output}
output: \${each.output}
`);

  expect(driver<MapNode>(flow, "each").childInputs).toEqual([parameter("value", Type.scalar(ValueKind.INTEGER))]);
});

const OPAQUE_OVER = `id: inline-map-opaque-over-no-item
name: inline_map_opaque_over_no_item
input:
  primary: !list(str)
  fallback: !list(str)
  prefix: !str
nodes:
  each:
    kind: map
    over: \${input.primary | input.fallback}
    input:
      constant: fixed
      parent: \${input.prefix}
      template: item \${item}
      literal:
        nested: \${input.prefix}
    nodes:
      collect:
        kind: code
        code: |
          return {
            "constant": inputs["constant"],
            "parent": inputs["parent"],
            "template": inputs["template"],
            "literal": inputs["literal"],
          }
        input:
          constant: \${input.constant}
          parent: \${input.parent}
          template: \${input.template}
          literal: \${input.literal}
        output:
          constant: str
          parent: str
          template: str
          literal:
            nested: str
    output: \${collect.output}
output: \${each.output}
`;

test("an inline MAP over an opaque list loads when no binding walks the item directly", () => {
  const { flow } = assembleText(OPAQUE_OVER);

  expect(driver<MapNode>(flow, "each").childInputs.map((param) => param.name)).toEqual([
    "constant",
    "parent",
    "template",
    "literal",
  ]);
});

test("an inline MAP over an opaque list runs when no binding walks the item directly", async () => {
  const output = await runOutput(OPAQUE_OVER, { primary: ["first", "second"], fallback: ["unused"], prefix: "P" });

  expect(output).toEqual([
    { constant: "fixed", parent: "P", template: "item first", literal: { nested: "P" } },
    { constant: "fixed", parent: "P", template: "item second", literal: { nested: "P" } },
  ]);
});

test.each(["${item}", "${item.name}", "{nested: '${item}'}"])(
  "a direct item read over an opaque MAP list fails at the requesting binding: %s",
  (itemSource) => {
    const source = `id: inline-map-opaque-over-item
name: inline_map_opaque_over_item
input:
  primary: !list(str)
  fallback: !list(str)
nodes:
  each:
    kind: map
    over: \${input.primary | input.fallback}
    input:
      row: ${itemSource}
    nodes:
      echo:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo_x
        input:
          x: \${input.row}
        output: str
    output: \${echo.output}
output: \${each.output}
`;

    const error = loadError(() => assembleText(source));

    expect(error.message).toBe(
      "cannot infer type for inline binding 'each.input.row': a whole-value expression has no supported " +
        "static scalar type; declare its type with a tag (`row: !<type> ...`) or use an explicitly typed " +
        "named `defs:` body",
    );
    expect(error.diagnostic.frames.at(-1)!.range).toEqual(inputRange(source, "each", "row"));
  },
);

// Spies on a module export, which only Bun allows (Node's ESM namespaces are frozen); the proof
// path calls the function directly, with no object or option to observe it through.
test.skipIf(runtime !== "bun")("an inline MAP proves its item type lazily, once", () => {
  const infer = spyOn(inlineInputs, "inferInlineBindingType");
  const overRequests = () =>
    infer.mock.calls
      .filter(([value]) => value === "${input.items}")
      .map(([, options]) => [options.bindingLabel, options.bindingRange]);
  const directSource = `id: inline-map-cached-item-proof
name: inline_map_cached_item_proof
typedefs:
  Row:
    name: str
input:
  items: !list(Row)
nodes:
  each:
    kind: map
    over: \${input.items}
    input:
      row: \${item}
      name: \${item.name}
    nodes:
      echo:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo_x
        input:
          x: \${input.name}
        output: str
    output: \${echo.output}
output: \${each.output}
`;
  let lazyRequests: unknown[];
  let directRequests: unknown[];
  let flow: ComposedFlow;
  try {
    assembleText(`id: inline-map-lazy-item-proof
name: inline_map_lazy_item_proof
typedefs:
  Row:
    name: str
input:
  items: !list(Row)
  prefix: !str
nodes:
  each:
    kind: map
    over: \${input.items}
    input:
      constant: fixed
      parent: \${input.prefix}
      template: prefix \${input.prefix}
      literal:
        nested: $\${item.name}
    nodes:
      echo:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo_x
        input:
          x: \${input.constant}
        output: str
    output: \${echo.output}
output: \${each.output}
`);
    lazyRequests = overRequests();
    infer.mockClear();
    ({ flow } = assembleText(directSource));
    directRequests = overRequests();
  } finally {
    infer.mockRestore();
  }

  expect(lazyRequests).toEqual([]);
  expect(driver<MapNode>(flow, "each").childInputs.map((param) => param.name)).toEqual(["row", "name"]);
  expect(directRequests).toEqual([["each.input.row", inputRange(directSource, "each", "row")]]);
});

test("an inline MAP over a typed producer types the whole item and a field", () => {
  const { flow } = assembleText(`id: inline-map-typed-records
name: inline_map_typed_records
typedefs:
  Row:
    name: str
    position: int
input:
  rows: !list(Row)
nodes:
  prepare:
    kind: code
    code: |
      return inputs["rows"]
    input:
      rows: \${input.rows}
    output: list(Row)
  each:
    kind: map
    parallel: true
    over: \${prepare.output}
    input:
      row: \${item}
      name: \${item.name}
    nodes:
      render:
        kind: code
        code: |
          return {"row": inputs["row"], "name": inputs["name"]}
        input:
          row: \${input.row}
          name: \${input.name}
        output:
          row: Row
          name: str
    output: \${render.output}
output: \${each.output}
`);

  const each = driver<MapNode>(flow, "each");
  expect(each.parallel).toBe(true);
  expect(each.childInputs.map((param) => param.name)).toEqual(["row", "name"]);
  expect(parameterType(each.childInputs[0]!)).toEqual(parameterType(flow.input[0]!).element!);
  expect(parameterType(each.childInputs[1]!)).toEqual(Type.scalar(ValueKind.STRING));
});

const DATE_ITEMS = `id: inline-map-date-items
name: inline_map_date_items
input:
  days: !list(date)
nodes:
  each:
    kind: map
    over: \${input.days}
    input:
      day: \${item}
    nodes:
      echo:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo_x
        input:
          x: \${input.day}
        output: date
    output: \${echo.output}
output: \${each.output}
`;

test("an inline MAP proves a `list(date)` element, which reads as text", () => {
  const { flow } = assembleText(DATE_ITEMS);

  expect(parameterType(flow.input[0]!)).toEqual(
    new Type({ kind: ValueKind.LIST_ANY, element: Type.scalar(ValueKind.DATE) }),
  );
  expect(parameterType(driver<MapNode>(flow, "each").childInputs[0]!)).toEqual(Type.scalar(ValueKind.DATE));
});

test("an inline MAP over a `list(date)` runs, its dates as text", async () => {
  expect(await runOutput(DATE_ITEMS, { days: ["2026-08-15", "2026-08-16"] })).toEqual(["2026-08-15", "2026-08-16"]);
});

test("a MAP list with no element type fails at the item binding", () => {
  const source = `id: inline-map-missing-element
name: inline_map_missing_element
input:
  items: !list(str)
nodes:
  each:
    kind: map
    over: \${input.items}
    input:
      value: \${item}
    nodes:
      echo:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo_x
        input:
          x: \${input.value}
        output: str
    output: \${echo.output}
output: \${each.output}
`;
  const inputParameters = [parameter("items", new Type({ kind: ValueKind.LIST_ANY }))];
  const error = loadError(() => assembleText(source, "flow.yaml", { inputParameters }));

  expect(error.message).toBe(
    "cannot infer type for inline binding 'each.input.value': list type 'list(Optional(Any))' has no proven " +
      "element type; declare its type with a tag (`value: !<type> ...`) or use an explicitly typed named " +
      "`defs:` body",
  );
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(inputRange(source, "each", "value"));
});

test("an inline MAP over nullable rows infers optional item and field inputs", () => {
  const { flow } = assembleText(`id: inline-map-nullable-items
name: inline_map_nullable_items
typedefs:
  Detail:
    name: str
  Row:
    detail: Optional(Detail)
input:
  items: !list(Optional(Row))
nodes:
  each:
    kind: map
    over: \${input.items}
    input:
      row: \${item}
      name: \${item.detail.name}
    nodes:
      collect:
        kind: code
        code: |
          return {"row": inputs["row"], "name": inputs["name"]}
        input:
          row: \${input.row}
          name: \${input.name}
        output:
          row: Optional(Row)
          name: Optional(str)
    output: \${collect.output}
output: \${each.output}
`);

  const [row, name] = driver<MapNode>(flow, "each").childInputs as [Parameter, Parameter];
  expect([row.name, row.required, parameterType(row).nullable]).toEqual(["row", false, true]);
  expect([name.name, name.required]).toEqual(["name", false]);
  expect(parameterType(name)).toEqual(new Type({ kind: ValueKind.STRING, nullable: true }));
});

test("an inline driver's env stays on the driver and the child's env stays inside", () => {
  const { flow } = assembleText(`id: inline-env-ownership
name: inline_env_ownership
env:
  FLOW_LEVEL: root
nodes:
  wrapper:
    kind: call
    env:
      DRIVER_ONLY: outer
    nodes:
      inner:
        kind: code
        env:
          CHILD_ONLY: inner
        code: ./fixtures/compose_codefns.ts:echo
        input:
          topic: ready
        output: str
    output: \${inner.output}
output: \${wrapper.output}
`);

  const wrapper = driver<CallNode>(flow, "wrapper");
  const inner = (wrapper.child as CompiledFlow).nodes.get("inner")!;
  expect(wrapper.env).toEqual({ FLOW_LEVEL: "root", DRIVER_ONLY: "outer" });
  expect(inner.env).toEqual({ CHILD_ONLY: "inner" });
});

const LOOP_BODY = `    input:
      n: 0
      exited: false
    nodes:
      bump:
        kind: code
        code: ./fixtures/compose_codefns.ts:loop_bump
        input:
          n: \${input.n}
          exited: \${input.exited}
        output:
          n: int
          exited: bool
    output: \${bump.output}
output: \${repeat.output}
`;

test("a basic inline LOOP with `times:` bakes its private child signature", () => {
  const { flow } = assembleText(`id: inline-loop-times
name: inline_loop_times
nodes:
  repeat:
    kind: loop
    times: 2
${LOOP_BODY}`);

  const repeat = driver<LoopNode>(flow, "repeat");
  expect(repeat.flowId).toBe(inlineFlowId(["root", "node", "repeat", "inline"]));
  expect(repeat.childInputs).toEqual([
    parameter("n", Type.scalar(ValueKind.INTEGER)),
    parameter("exited", Type.scalar(ValueKind.BOOLEAN)),
  ]);
  expect((repeat.outputContract as ValueContract).type).toEqual(
    new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([
        ["n", Type.scalar(ValueKind.INTEGER)],
        ["exited", Type.scalar(ValueKind.BOOLEAN)],
      ]),
      required: new Set(["n", "exited"]),
    }),
  );
  expect(repeat.predicateKind).toBe("times");
  expect([repeat.times, repeat.maxIters]).toEqual([2, 2]);
});

test("a basic inline LOOP with `times:` runs", async () => {
  const output = await runOutput(`id: inline-loop-times
name: inline_loop_times
nodes:
  repeat:
    kind: loop
    times: 2
${LOOP_BODY}`);

  expect(output).toEqual({ n: 2n, exited: false });
});

test("an inline LOOP with `until:` loads with that predicate kind", () => {
  const { flow } = assembleText(`id: inline-loop-predicate
name: inline_loop_predicate
nodes:
  repeat:
    kind: loop
    until: \${exited}
    max: 10
${LOOP_BODY}`);

  expect(driver<LoopNode>(flow, "repeat").predicateKind).toBe("until");
});

test.each(["until: ${exited}"])("an inline LOOP predicate runs to the same carried result: %s", async (predicate) => {
  const output = await runOutput(`id: inline-loop-predicate
name: inline_loop_predicate
nodes:
  repeat:
    kind: loop
    ${predicate}
    max: 10
${LOOP_BODY}`);

  expect(output).toEqual({ n: 3n, exited: true });
});

const NESTED_COMPOSITES = `id: nested-inline-composites
name: nested_inline_composites
input:
  items: !list(int)
nodes:
  outer_call:
    kind: call
    input:
      items: \${input.items}
    nodes:
      inner:
        kind: map
        over: \${input.items}
        input:
          value: \${item}
        nodes:
          echo:
            kind: code
            code: ./fixtures/compose_codefns.ts:echo_x
            input:
              x: \${input.value}
            output: int
        output: \${echo.output}
    output: \${inner.output}
  outer_loop:
    kind: loop
    times: 2
    input:
      n: 0
      exited: false
    nodes:
      inner:
        kind: call
        input:
          n: \${input.n}
          exited: \${input.exited}
        nodes:
          bump:
            kind: code
            code: ./fixtures/compose_codefns.ts:loop_bump
            input:
              n: \${input.n}
              exited: \${input.exited}
            output:
              n: int
              exited: bool
        output: \${bump.output}
    output: \${inner.output}
output:
  mapped: \${outer_call.output}
  looped: \${outer_loop.output}
`;

test("nested inline composites register four lexical children once", async () => {
  const { result: flow, registrations, resolutions } = await recordPrivateScope(() =>
    loadConformanceFlow(NESTED_COMPOSITES),
  );

  const outerCallOwner = ["root", "node", "outer_call", "inline"];
  const innerMapOwner = [...outerCallOwner, "node", "inner", "inline"];
  const outerLoopOwner = ["root", "node", "outer_loop", "inline"];
  const innerCallOwner = [...outerLoopOwner, "node", "inner", "inline"];
  const outerCall = driver<CallNode>(flow, "outer_call");
  const outerLoop = driver<LoopNode>(flow, "outer_loop");
  const innerMap = (outerCall.child as CompiledFlow).nodes.get("inner") as MapNode;
  const innerCall = (outerLoop.child as CompiledFlow).nodes.get("inner") as CallNode;
  const flowIds = [outerCall.flowId, innerMap.flowId, outerLoop.flowId, innerCall.flowId];
  expect(flowIds).toEqual([
    inlineFlowId(outerCallOwner),
    inlineFlowId(innerMapOwner),
    inlineFlowId(outerLoopOwner),
    inlineFlowId(innerCallOwner),
  ]);
  expect(new Set(flowIds).size).toBe(4);
  expectRegisteredOnce(registrations, resolutions, [
    [outerCallOwner, "inline body of node 'outer_call'"],
    [innerMapOwner, "inline body of node 'inner'"],
    [outerLoopOwner, "inline body of node 'outer_loop'"],
    [innerCallOwner, "inline body of node 'inner'"],
  ]);
});

test("nested inline composites run without registering again", async () => {
  const output = await runOutputRegisteringNoMore(NESTED_COMPOSITES, { items: [3n, 1n, 2n] }, 4);

  expect(output).toEqual({ mapped: [3n, 1n, 2n], looped: { n: 2n, exited: false } });
});

/**
 * Expect `registrations` and `resolutions` (from `recordPrivateScope`) to register and resolve
 * each of `expected`'s `[owner, label]` once, under its owner's flow id, all in one scope.
 */
function expectRegisteredOnce(
  registrations: unknown[][],
  resolutions: unknown[][],
  expected: [string[], string][],
): void {
  const scope = registrations[0]![0];
  expect(registrations.map(([, flowId, owner, label]) => [flowId, owner, label]).sort()).toEqual(
    expected.map(([owner, label]) => [inlineFlowId(owner), owner, label]).sort(),
  );
  expect(resolutions.map(([, flowId]) => flowId).sort()).toEqual(expected.map(([owner]) => inlineFlowId(owner)).sort());
  for (const [entryScope] of [...registrations, ...resolutions]) expect(entryScope).toBe(scope);
}

const EAGER_DEF_NESTED = `id: eager-def-nested-inline-calls
name: eager_def_nested_inline_calls
defs:
  outer:
    nodes:
      outer_call:
        kind: call
        nodes:
          inner_call:
            kind: call
            nodes:
              constant:
                kind: code
                code: ./fixtures/compose_codefns.ts:echo
                input:
                  topic: nested-inline-ok
                output: str
            output: \${constant.output}
        output: \${inner_call.output}
    output: \${outer_call.output}
nodes:
  invoke:
    kind: call
    call: outer
output: \${invoke.output}
`;

test("an eager def's nested inline calls use their exact lexical owners, once", async () => {
  const { result: loaded, registrations, resolutions } = await recordPrivateScope(() =>
    loadConformanceFlow(EAGER_DEF_NESTED),
  );

  const outerOwner = ["def", "outer", "node", "outer_call", "inline"];
  const innerOwner = [...outerOwner, "node", "inner_call", "inline"];
  const outerCall = (driver<CallNode>(loaded, "invoke").child as CompiledFlow).nodes.get("outer_call") as CallNode;
  const innerCall = (outerCall.child as CompiledFlow).nodes.get("inner_call") as CallNode;
  expect(outerCall.flowId).toBe(inlineFlowId(outerOwner));
  expect(innerCall.flowId).toBe(inlineFlowId(innerOwner));
  expectRegisteredOnce(registrations, resolutions, [
    [outerOwner, "inline body of node 'outer_call'"],
    [innerOwner, "inline body of node 'inner_call'"],
  ]);
});

test("an eager def's nested inline calls run without registering again", async () => {
  expect(await runOutputRegisteringNoMore(EAGER_DEF_NESTED)).toBe("nested-inline-ok");
});

test.each([
  [
    "call",
    "asserts, call, depends_on, env, input, kind, node_name, runs_after",
    `    kind: call
    llm_config: {model: forbidden}
    nodes:
      inner:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input: {topic: ready}
        output: str
    output: \${inner.output}
`,
  ],
  [
    "map",
    "asserts, call, depends_on, env, input, kind, node_name, over, parallel, runs_after",
    `    kind: map
    llm_config: {model: forbidden}
    over: \${[1]}
    nodes:
      inner:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input: {topic: ready}
        output: str
    output: \${inner.output}
`,
  ],
  [
    "loop",
    "asserts, call, depends_on, env, input, kind, max, node_name, runs_after, times, until, while",
    `    kind: loop
    llm_config: {model: forbidden}
    times: 1
    input: {n: 0}
    nodes:
      inner:
        kind: code
        code: |
          return {"n": inputs["n"] + 1n}
        input:
          n: \${input.n}
        output:
          n: int
    output: \${inner.output}
`,
  ],
])("an inline %s rejects an outer llm_config without its private identity", (kind, allowed, nodeBody) => {
  const source = `id: inline-llm-config-rejected
name: inline_llm_config_rejected
nodes:
  composite:
${nodeBody}output: \${composite.output}
`;

  const error = loadError(() => assembleText(source));

  expect(error.message).toBe(
    `node 'composite' (kind=${kind}): field 'llm_config' is not allowed (allowed: ${allowed})`,
  );
});

test.each([
  [
    "call",
    `    kind: call
    asserts:
      - output == 999
    nodes:
      inner:
        kind: code
        code: |
          return 1
        output: int
    output: \${inner.output}
`,
  ],
  [
    "loop",
    `    kind: loop
    times: 1
    input:
      n: 0
      exited: false
    asserts:
      - output.n == 999
    nodes:
      inner:
        kind: code
        code: ./fixtures/compose_codefns.ts:loop_bump
        input:
          n: \${input.n}
          exited: \${input.exited}
        output:
          n: int
          exited: bool
    output: \${inner.output}
`,
  ],
])("an inline %s's outer post asserts still fire", async (_kind, nodeBody) => {
  const source = `id: inline-outer-assert
name: inline_outer_assert
nodes:
  composite:
${nodeBody}output: \${composite.output}
`;

  const result = await runFlow(await loadConformanceFlow(source), {});

  expect(result.status).toBe("failed");
  const failures = result.events.filter((event) => event instanceof RunFailed);
  expect(failures.length).toBeGreaterThan(0);
  expect(failures.at(-1)!.failure.causeType).toBe("NodeAssertFailed");
});

test.each([
  [
    "scalar-output",
    `      step:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo_x
        input:
          x: \${input.n}
        output: int
`,
    "body inline body of node 'repeat' must return a record with at least one carried field (a subset of the " +
      'carried record ["n"]); write `output: {field: ...}` or a whole reference to a node with a record output',
  ],
  [
    "record-fields",
    `      step:
        kind: code
        code: |
          return {"value": inputs["n"]}
        input:
          n: \${input.n}
        output:
          value: int
`,
    'body output field(s) ["value"] are not carried record fields ["n"] (a body returns a subset of the ' +
      "carried record)",
  ],
  [
    "field-type",
    `      step:
        kind: code
        code: |
          return {"n": String(inputs["n"])}
        input:
          n: \${input.n}
        output:
          n: str
`,
    "carried field 'n' — body output is 'str', carried source is 'int' (a returned field must keep its carried type)",
  ],
])("an inline LOOP's contract error is friendly and redacted: %s", (_name, step, message) => {
  const source = `id: inline-loop-bad-contract
name: inline_loop_bad_contract
input:
  seed: !int
nodes:
  repeat:
    kind: loop
    times: 1
    input:
      n: \${input.seed}
    nodes:
${step}    output: \${step.output}
output: \${repeat.output}
`;

  const error = loadError(() => assembleText(source));

  expect(error.message).toBe(`loop node 'repeat': ${message}`);
});

test.each(["value", "count"])("an inline wrapper's bad input %s is located at the outer binding", (badInput) => {
  const source = `id: inline-child-input-ranges
name: inline_child_input_ranges
input:
  value: !str
  count: !int
nodes:
  wrapper:
    kind: call
    input:
      value: \${input.value}
      count: \${input.count}
    nodes:
      echo:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo_x
        input:
          x: \${input.value}
        output: str
    output: \${echo.output}
output: \${wrapper.output}
`.replace(`\${input.${badInput}}`, "${missing.output}");

  const error = loadError(() => assembleText(source));

  expect(error.message).toBe(
    `cannot infer type for inline binding 'wrapper.input.${badInput}': producer 'missing' has no available ` +
      `static type; declare its type with a tag (\`${badInput}: !<type> ...\`) or use an explicitly typed ` +
      "named `defs:` body",
  );
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(inputRange(source, "wrapper", badInput));
});

test("an inline MAP's output error uses the friendly child label", () => {
  const error = loadError(() =>
    assembleText(`id: inline-map-no-output
name: inline_map_no_output
input:
  items: !list(str)
nodes:
  each:
    kind: map
    over: \${input.items}
    nodes:
      constant:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input:
          topic: inline-ok
        output: str
    output:
output: \${each.output}
`),
  );

  expect(error.message).toBe(
    "map node 'each': inline body of node 'each' must declare a non-empty `output:`; " +
      "a mapped call (over:) returns a list of the child's results",
  );
});

test("an inline MAP rejects outer asserts without private identity", () => {
  const error = loadError(() =>
    assembleText(`id: inline-map-asserts
name: inline_map_asserts
input:
  items: !list(str)
nodes:
  each:
    kind: map
    over: \${input.items}
    input:
      value: \${item}
    asserts:
      - output != null
    nodes:
      echo:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo_x
        input:
          x: \${input.value}
        output: str
    output: \${echo.output}
output: \${each.output}
`),
  );

  expect(error.message).toBe(
    "flow has invalid node asserts:\n" +
      "  node 'each': `asserts:` is not allowed on a mapped call (`kind: map`) — assert inside the def, " +
      "or on a downstream node",
  );
  expect(error.message).not.toContain(INLINE_FLOW_PREFIX);
});

// ---------- reference validation ----------

test.each([
  [
    "parent-node",
    "  parent:\n    kind: code\n    code: |\n      return 'parent'\n    output: str\n",
    "${parent.output}",
    "parent",
  ],
  ["item", "", "${item}", "item"],
])("an inline child's scope is closed and redacts its private identity: %s", (_id, outerNodes, badRef, namespace) => {
  const source =
    "id: inline-closed-scope\n" +
    "name: inline_closed_scope\n" +
    "nodes:\n" +
    outerNodes +
    "  wrapper:\n" +
    "    kind: call\n" +
    "    nodes:\n" +
    "      leak:\n" +
    "        kind: code\n" +
    "        code: ./fixtures/compose_codefns.ts:echo_x\n" +
    "        input:\n" +
    `          x: ${badRef}\n` +
    "        output: str\n" +
    "    output: ${leak.output}\n" +
    "output: ${wrapper.output}\n";
  const expected = SourceDocument.parse(source)
    .root.child("nodes")
    .child("wrapper")
    .child("nodes")
    .child("leak")
    .mappingValueRanges("input")
    .get("x");

  const error = loadError(() => assembleText(source));

  expect(error.message).toContain(`unknown namespace '${namespace}'`);
  expect(error.message).not.toContain(INLINE_FLOW_PREFIX);
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(expected!);
});

test("an inline MAP child keeps the item namespace closed", () => {
  const source = `id: inline-map-closed-child
name: inline_map_closed_child
input:
  items: !list(str)
nodes:
  each:
    kind: map
    over: \${input.items}
    input:
      value: \${item}
    nodes:
      leak:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo_x
        input:
          x: \${item}
        output: str
    output: \${leak.output}
output: \${each.output}
`;
  const expected = SourceDocument.parse(source)
    .root.child("nodes")
    .child("each")
    .child("nodes")
    .child("leak")
    .mappingValueRanges("input")
    .get("x");

  const error = loadError(() => assembleText(source));

  expect(error.message).toContain("unknown namespace 'item'");
  expect(error.message).not.toContain(INLINE_FLOW_PREFIX);
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(expected!);
});

test("an inline sibling keeps the outer bad binding's range", () => {
  const source = `id: inline-outer-bad-binding
name: inline_outer_bad_binding
nodes:
  wrapper:
    kind: call
    nodes:
      constant:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input:
          topic: ready
        output: str
    output: \${constant.output}
  broken:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo_x
    input:
      x: \${missing.output}
    output: str
output: \${broken.output}
`;

  const error = loadError(() => assembleText(source));

  expect(error.message).toContain("unknown namespace 'missing'");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(inputRange(source, "broken", "x"));
});

test("an inline child's bad output uses the retained output range", () => {
  const source = `id: inline-child-output-range
name: inline_child_output_range
nodes:
  wrapper:
    kind: call
    nodes:
      constant:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input:
          topic: ready
        output: str
    output: \${missing.output}
output: \${wrapper.output}
`;
  const expected = SourceDocument.parse(source).root.child("nodes").child("wrapper").rangeOf("output")!;

  const error = loadError(() => assembleText(source));

  expect(error.message).toContain("unknown namespace 'missing'");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(expected);
});

test("an eager def's inline child bad output keeps its absolute range", async () => {
  const source = `id: eager-def-inline-output-range
name: eager_def_inline_output_range
defs:
  outer:
    nodes:
      wrapper:
        kind: call
        nodes:
          constant:
            kind: code
            code: ./fixtures/compose_codefns.ts:echo
            input:
              topic: ready
            output: str
        output: \${missing.output}
    output: \${wrapper.output}
nodes:
  invoke:
    kind: call
    call: outer
output: \${invoke.output}
`;
  const expected = SourceDocument.parse(source)
    .root.child("defs")
    .child("outer")
    .child("nodes")
    .child("wrapper")
    .rangeOf("output")!;

  const error = await loadErrorAsync(loadConformanceFlow(source));

  expect(error.message).toBe(
    "defs entry 'outer': flow has unresolved references:\n" +
      "  flow output from: reference ${missing.output} uses unknown namespace 'missing'",
  );
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(expected);
});

test("an inline call's resolver failure uses the friendly child label", async () => {
  const resolved: string[] = [];
  const resolve = spyOn(CallableScope.prototype, "resolve").mockImplementation((flowId) => {
    resolved.push(flowId);
    throw new Error("resolver exploded");
  });
  try {
    const error = await loadErrorAsync(
      loadConformanceFlow(`id: inline-resolver-failure
name: inline_resolver_failure
nodes:
  wrapper:
    kind: call
    nodes:
      constant:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input:
          topic: ready
        output: str
    output: \${constant.output}
output: \${wrapper.output}
`),
    );

    expect(resolved).toEqual([inlineFlowId(["root", "node", "wrapper", "inline"])]);
    expect(error.message).toBe(
      "call node 'wrapper': could not resolve inline body of node 'wrapper': resolver exploded",
    );
  } finally {
    resolve.mockRestore();
  }
});

// ---------- the callable scope's private callables ----------

const WRAPPER_OWNER = ["root", "node", "wrapper", "inline"];

/** A scope over no `defs:` or `uses:`. */
const emptyScope = () => new CallableScope({}, new Map(), null, {});

/** A stand-in child: the scope only caches and returns it. */
const stubChild = () => ({}) as ComposedFlow;

test("the callable scope builds a private once and same-owner registration is idempotent", () => {
  const flowId = `${INLINE_FLOW_PREFIX}opaque-once`;
  const child = stubChild();
  const built: string[] = [];
  const scope = emptyScope();
  scope.registerPrivate(flowId, WRAPPER_OWNER, "inline body `wrapper`", () => {
    built.push("first");
    return child;
  });
  scope.registerPrivate(flowId, WRAPPER_OWNER, "ignored label", () => {
    built.push("ignored");
    return stubChild();
  });

  expect(built).toEqual([]);
  expect(scope.resolve(flowId)).toBe(child);
  expect(scope.resolve(flowId, "ignored-version")).toBe(child);
  expect(built).toEqual(["first"]);
});

test("the callable scope rejects a non-private id without registering it", () => {
  const built: string[] = [];
  const scope = emptyScope();

  const error = loadError(() =>
    scope.registerPrivate("ordinary_name", WRAPPER_OWNER, "inline body `wrapper`", () => {
      built.push("built");
      return stubChild();
    }),
  );

  expect(error.message).toBe(
    "compiler-generated callable 'inline body `wrapper`' owned by " +
      '["root","node","wrapper","inline"] must use the reserved prefix \'__inline_flow_\'',
  );
  expect(built).toEqual([]);
  expect(loadError(() => scope.resolve("ordinary_name")).message).toContain("neither an in-file");
  expect(built).toEqual([]);
});

test("the callable scope rejects a private owner collision without exposing the id", () => {
  const flowId = `${INLINE_FLOW_PREFIX}opaque-owner-collision`;
  const scope = emptyScope();
  scope.registerPrivate(flowId, ["root", "node", "first", "inline"], "inline body `first`", stubChild);

  const error = loadError(() =>
    scope.registerPrivate(flowId, ["root", "node", "second", "inline"], "inline body `second`", stubChild),
  );

  expect(error.message).toBe(
    "compiler-generated callable collision in reserved prefix '__inline_flow_': 'inline body `first`' owned by " +
      '["root","node","first","inline"] conflicts with \'inline body `second`\' owned by ' +
      '["root","node","second","inline"]',
  );
});

test.each(["defs", "uses"])(
  "the callable scope rejects a private collision with an authored %s: name without exposing the id",
  (section) => {
    const flowId = `${INLINE_FLOW_PREFIX}opaque-authored-collision`;
    const defs: ValueRecord = section === "defs" ? { [flowId]: { nodes: {} } } : {};
    const uses: Record<string, string> = section === "uses" ? { [flowId]: "library/child" } : {};
    const scope = new CallableScope(defs, new Map(), null, uses);

    const error = loadError(() => scope.registerPrivate(flowId, WRAPPER_OWNER, "inline body `wrapper`", stubChild));

    expect(error.message).toBe(
      "compiler-generated callable collision in reserved prefix '__inline_flow_': 'inline body `wrapper`' " +
        `owned by ["root","node","wrapper","inline"] conflicts with an authored ${section}: name`,
    );
  },
);

test("the callable scope reports a private recursion without exposing the id", () => {
  const flowId = `${INLINE_FLOW_PREFIX}opaque-recursion`;
  const scope = emptyScope();
  let calls = 0;
  scope.registerPrivate(flowId, WRAPPER_OWNER, "inline body `wrapper`", () => {
    calls += 1;
    return scope.resolve(flowId);
  });

  const error = loadError(() => scope.resolve(flowId));

  expect(error.message).toBe(
    "recursive compiler-generated callable in reserved prefix '__inline_flow_': 'inline body `wrapper`' " +
      'owned by ["root","node","wrapper","inline"] is defined in terms of itself (the baked call graph must ' +
      "be finite)",
  );
  expect(calls).toBe(1);
});

test("the callable scope retries a failed private builder and caches its success", () => {
  const flowId = `${INLINE_FLOW_PREFIX}opaque-retry`;
  const child = stubChild();
  let attempts = 0;
  const scope = emptyScope();
  scope.registerPrivate(flowId, WRAPPER_OWNER, "inline body `wrapper`", () => {
    attempts += 1;
    if (attempts === 1) throw new LoadDiagnostics().error("temporary child build failure");
    return child;
  });

  expect(loadError(() => scope.resolve(flowId)).message).toBe("temporary child build failure");
  expect(scope.resolve(flowId)).toBe(child);
  expect(scope.resolve(flowId)).toBe(child);
  expect(attempts).toBe(2);
});

test("the callable scope reports an unregistered private prefix without exposing the id", () => {
  const error = loadError(() => emptyScope().resolve(`${INLINE_FLOW_PREFIX}opaque-unregistered`));

  expect(error.message).toBe(
    "compiler-generated callable with reserved prefix '__inline_flow_' was not registered in this file scope",
  );
});

test("an eager def can resolve a pre-registered private", () => {
  const flowId = `${INLINE_FLOW_PREFIX}opaque-eager-def`;
  const privateChild = stubChild();
  const defChild = stubChild();
  const events: string[] = [];
  // The def build is module-private, so a subclass fakes it for `outer`.
  class FakeDefScope extends CallableScope {
    override resolve(name: string, version: string | null = null): ComposedFlow {
      if (name !== "outer" || events.includes("def:outer")) return super.resolve(name, version);
      events.push("def:outer");
      expect(this.resolve(flowId)).toBe(privateChild);
      return defChild;
    }
  }
  const scope = new FakeDefScope({ outer: { nodes: {} } }, new Map(), null, {});
  const owner = ["def", "outer", "node", "wrapper", "inline"];
  scope.registerPrivate(flowId, owner, "inline body `wrapper` in defs:`outer`", () => {
    events.push("private");
    return privateChild;
  });

  scope.eagerLoadDefs();

  expect(events).toEqual(["def:outer", "private"]);
  expect(scope.resolve(flowId)).toBe(privateChild);
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  { id: "load-map-call/engine_inline_composite_bodies__llm_config_on_an_inline_call", call: parsedNodes },
  { id: "load-map-call/engine_inline_composite_bodies__llm_config_on_an_inline_loop", call: parsedNodes },
  { id: "load-map-call/engine_inline_composite_bodies__llm_config_on_an_inline_map", call: parsedNodes },
])("parseNodes: $id", runCase);

test.each<CaseRow>([
  { id: "load-map-call/engine_inline_composite_bodies__call_node_naming_a_private_digest", call: assembled },
  { id: "load-map-call/engine_inline_composite_bodies__inline_body_missing_output", call: assembled },
  { id: "load-map-call/engine_inline_composite_bodies__named_call_with_output", call: assembled },
  { id: "load-map-call/engine_inline_composite_bodies__neither_call_nor_nodes", call: assembled },
  { id: "load-map-call/engine_inline_composite_bodies__non_mapping_inline_nodes", call: assembled },
  { id: "load-map-call/engine_inline_composite_bodies__private_prefix_in_a_node_id_loads", call: assembled },
  { id: "load-map-call/engine_inline_composite_bodies__inline_map_body_without_output", call: assembled },
  { id: "load-map-call/engine_inline_composite_bodies__inline_loop_body_returns_an_uncarried_field", call: assembled },
  { id: "load-map-call/engine_inline_composite_bodies__inline_signature_self_cycle", call: assembled },
  { id: "load-map-call/engine_inline_composite_bodies__inline_child_scope_is_closed", call: assembled },
  { id: "load-map-call/engine_inline_composite_bodies__inline_child_unknown_output_ref_is_located", call: assembled },
  { id: "load-map-call/engine_inline_composite_bodies__inline_map_child_keeps_item_namespace_closed", call: assembled },
  { id: "load-types/compose_typed_bindings_downstream__inline_map_item_over_an_opaque_list", call: assembled },
  { id: "load-map-call/engine_templates_load__inline_composites_template_loads", call: assembled },
  { id: "load-map-call/engine_inline_composite_bodies__inline_map_rejects_outer_asserts", call: assembled },
])("assemble: $id", runCase);
