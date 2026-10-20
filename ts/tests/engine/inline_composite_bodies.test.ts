/**
 * Inline composite bodies: a port of `tests/engine/test_inline_composite_bodies.py`, so far the
 * rows the CALL, MAP and LOOP parse decides and the inline body extraction and registration rows.
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
 * `assembleText` (the load half):
 * `test_load_translates_recursive_inline_preflight_access_to_load_error`,
 * `test_loader_allows_private_callable_prefix_in_graph_node_id`,
 * `test_inline_body_shape_matrix_flows_through_loader`,
 * `test_inline_body_non_text_input_key_fails_at_the_key_through_loader`,
 * `test_loader_rejects_actual_sibling_private_digest_without_exposing_it`. Where Python checks
 * a message fragment, the row pins the whole message.
 *
 * TS-only: `inlineFlowId` pinned for two owners, one holding non-ASCII, control, quote and
 * backslash characters (the digest of Python's `json.dumps(..., ensure_ascii=False)` bytes).
 *
 * Changed: the direct shape row drops its `non-string-input-key` param (the TS document gives
 * every key as text, so `extractInlineBodies` sees `"1"`); the non-text key row expects the
 * document's refusal of the key, as Python's loader row does. The sibling-attack row cannot patch
 * the private `maskInlineBody`, so it spies on `SourceCursor.prototype.projectMapping`, which
 * masking calls. A wrong-kind descriptor is a CODE descriptor, not an arbitrary object. Descriptor
 * equality leaves out `childLabel`, as Python's `compare=False` does.
 *
 * TS-only: a CALL or MAP descriptor built with `childLabel: "inline body of node 'each'"` over a
 * hand-built child names the body by that label, reproducing the message composition of
 * `test_inline_map_output_error_uses_friendly_child_label` (a MAP over a body with no output) and
 * of an undeclared argument of an inline call (the inline body itself is T3-17's); and a LOOP
 * descriptor built with `childLabel: "inline body of node 'repeat'"` the same way, reproducing
 * the message composition of the scalar-output and record-fields rows of
 * `test_inline_loop_contract_diagnostics_are_friendly_and_redacted`.
 *
 * Pinned at the end (`parsedNodes`):
 * `load-map-call/engine_inline_composite_bodies__llm_config_on_an_inline_{call,loop,map}`, whose
 * body-key check reaches `llm_config:`, the second key, before the inline `nodes:` key and before
 * the missing-`call:` check; and (`assembled`) the extraction cases
 * `load-map-call/engine_inline_composite_bodies__{call_node_naming_a_private_digest,
 * inline_body_missing_output,named_call_with_output,neither_call_nor_nodes,
 * non_mapping_inline_nodes}` and the ok case
 * `load-map-call/engine_inline_composite_bodies__private_prefix_in_a_node_id_loads`.
 *
 * Later (T3-22, the callable scope and def assembly):
 * `test_def_assembly_cursor_fallback_restores_only_node_surfaces`,
 * `test_callable_scope_builds_private_once_and_same_owner_registration_is_idempotent`,
 * `test_callable_scope_rejects_non_private_id_without_registering_it`,
 * `test_callable_scope_rejects_private_owner_collision_without_exposing_id`,
 * `test_callable_scope_rejects_private_authored_collision_without_exposing_id`,
 * `test_callable_scope_reports_private_recursion_without_exposing_id`,
 * `test_callable_scope_retries_failed_private_builder_and_caches_success`,
 * `test_callable_scope_reports_unregistered_private_prefix_without_exposing_id`,
 * `test_callable_scope_eager_def_can_resolve_pre_registered_private`.
 *
 * Later (T3-17, T3-22, loads only):
 * `test_inline_signature_self_cycle_uses_binding_path_and_range`,
 * `test_inline_signature_mutual_cycle_uses_closing_binding_range`,
 * `test_inline_signature_unknown_producer_is_located_and_strict`,
 * `test_inline_signature_producer_without_output_is_located_and_strict`,
 * `test_inline_child_scope_is_closed_and_redacts_private_identity`,
 * `test_inline_map_direct_item_on_opaque_over_fails_at_requesting_binding`,
 * `test_inline_map_item_proof_is_lazy_and_cached`,
 * `test_inline_map_missing_list_element_fails_at_item_binding`,
 * `test_inline_map_rejects_outer_asserts_without_private_identity`,
 * `test_inline_outer_env_stays_on_driver_and_child_env_stays_inner`,
 * `test_inline_composites_reject_outer_llm_config_without_private_identity`,
 * `test_inline_map_child_keeps_item_namespace_closed`,
 * `test_inline_loop_contract_diagnostics_are_friendly_and_redacted`,
 * `test_inline_sibling_keeps_outer_bad_binding_range`,
 * `test_inline_wrapper_bad_input_uses_outer_binding_range`,
 * `test_inline_child_bad_output_uses_retained_output_range`,
 * `test_eager_def_inline_child_bad_output_preserves_absolute_range`,
 * `test_inline_call_resolver_failure_uses_friendly_child_label`,
 * `test_inline_map_output_error_uses_friendly_child_label`.
 *
 * Later (T4, runs): `test_zero_input_inline_call_loads_and_runs_in_one_file_scope`,
 * `test_typed_inline_call_preserves_nullable_parent_scalar`,
 * `test_typed_inline_call_canonicalizes_literal_record_and_retains_input_order`,
 * `test_forward_producer_type_builds_once_and_keeps_source_order`,
 * `test_producer_declared_none_type_is_not_treated_as_missing_output`,
 * `test_inline_alias_and_merge_occurrences_run_as_distinct_private_children`,
 * `test_inline_nested_containers_are_evaluated_at_edges_and_runtime`,
 * `test_inline_optional_record_paths_make_generated_inputs_nullable`,
 * `test_inline_input_inference_projects_known_partial_loop_field`,
 * `test_inline_multinode_body_matches_named_def_signature_and_runtime`,
 * `test_eager_def_inline_call_uses_origin_and_shared_scope`,
 * `test_inline_map_literal_over_runs_with_inferred_item_type`,
 * `test_inline_map_opaque_over_without_direct_item_walk_loads_and_runs`,
 * `test_inline_map_typed_producer_maps_whole_item_and_field_in_parallel_order`,
 * `test_inline_map_proves_list_any_date_element_and_runs`,
 * `test_inline_map_nullable_root_and_intermediate_infer_optional_and_intermediate_runs_none`,
 * `test_inline_call_and_loop_outer_post_asserts_still_fire`,
 * `test_basic_inline_loop_times_bakes_private_child_signature_and_runs`,
 * `test_inline_loop_predicates_run_to_the_same_carried_result`,
 * `test_nested_inline_composites_register_four_lexical_children_once`,
 * `test_eager_def_nested_inline_calls_use_exact_lexical_owners_once`.
 */

import { expect, spyOn, test } from "bun:test";

import { type ComposedFlow, type PrivateRegistrar, registerInlineBodies } from "../../src/compose/build/pipeline.ts";
import {
  extractInlineBodies,
  INLINE_FLOW_PREFIX,
  type InlineBody,
  inlineFlowId,
} from "../../src/compose/desugar/inline_bodies.ts";
import type { LoadError } from "../../src/compose/helpers/errors.ts";
import { BuildContext, type NodeDescriptor } from "../../src/compose/nodes/base.ts";
import { CallDescriptor } from "../../src/compose/nodes/call.ts";
import { CodeDescriptor } from "../../src/compose/nodes/code.ts";
import { LoopDescriptor } from "../../src/compose/nodes/loop.ts";
import { MapDescriptor } from "../../src/compose/nodes/map.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { Type, type Value, ValueKind, type ValueRecord } from "../../src/typesys/values.ts";
import {
  assembled,
  assembleText,
  type CaseRow,
  handChild,
  loadError,
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
    expect(Object.values(source!._value as ValueRecord).every((value) => value === null)).toBe(true);
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
  expect(keysOf(first).filter((key) => key === "call")).toEqual(["call"]);
  expect(keysOf(second).filter((key) => key === "call")).toEqual(["call"]);

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
  expect(keysOf(masked).filter((key) => key === "call")).toEqual(["call"]);
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
  expect(error.diagnostic.frames.at(-1)!.range).not.toBeNull();
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

/** Each malformed body: its kind, the body text, the key its range is at (else the body), the message. */
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
    expect(rewritten).toBeInstanceOf(original.constructor as typeof CallDescriptor);
    expect(withoutChildLabel(rewritten)).toEqual(withoutChildLabel(original));
    expect(rewritten.childLabel).toBe(expectedLabel);
    expect(rewritten.call).toBe(body.flowId);
    expect(original).toBe(descriptors.get(body.nodeId)! as CallDescriptor);
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

test("an inline map output error uses the friendly child label", () => {
  const desc = new MapDescriptor({
    id: "each",
    call: "inline-body",
    over: "${input.items}",
    childLabel: "inline body of node 'each'",
  });

  const error = loadError(() => desc.build(bodyContext(null)));

  expect(error.message).toBe(
    "map node 'each': inline body of node 'each' must declare a non-empty `output:`; " +
      "a mapped call (over:) returns a list of the child's results",
  );
});

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

test.each([
  [
    "scalar output",
    new ValueContract(Type.scalar(ValueKind.INTEGER)),
    "body inline body of node 'repeat' must return a record with at least one carried field (a subset of the " +
      'carried record ["n"]); write `output: {field: ...}` or a whole reference to a node with a record output',
  ],
  [
    "record fields",
    new ValueContract(
      new Type({
        kind: ValueKind.OBJECT,
        fields: new Map([["value", Type.scalar(ValueKind.INTEGER)]]),
        required: new Set(["value"]),
      }),
    ),
    'body output field(s) ["value"] are not carried record fields ["n"] (a body returns a subset of the ' +
      "carried record)",
  ],
])("an inline loop's contract error names the body by its child label: %s", (_name, step, message) => {
  const body = handChild({
    inputs: parametersFromYaml("input:\n  n: !int\n", new Map()),
    producers: { step },
    output: "${step.output}",
  });
  const desc = new LoopDescriptor({
    id: "repeat",
    call: "inline-body",
    inputs: { n: "${input.seed}" },
    times: 1n,
    childLabel: "inline body of node 'repeat'",
  });

  const error = loadError(() => desc.build(new BuildContext({ registry: new Map(), resolver: () => body })));

  expect(error.message).toBe(`loop node 'repeat': ${message}`);
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
])("assemble: $id", runCase);
