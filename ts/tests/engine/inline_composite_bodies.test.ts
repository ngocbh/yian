/**
 * Inline composite bodies: a port of `tests/engine/test_inline_composite_bodies.py`, so far the
 * rows the CALL and MAP parse decides.
 *
 * Ported: the call and map rows of `test_child_label_is_not_authorable`, through `parseNodes` with
 * the registered kinds; the TS message names the field and lists the allowed keys.
 *
 * TS-only: a CALL or MAP descriptor built with `childLabel: "inline body of node 'each'"` over a
 * hand-built child names the body by that label, reproducing the message composition of
 * `test_inline_map_output_error_uses_friendly_child_label` (a MAP over a body with no output) and
 * of an undeclared argument of an inline call (the inline body itself is T3-17's).
 *
 * Pinned at the end (`parsedNodes`):
 * `load-map-call/engine_inline_composite_bodies__llm_config_on_an_inline_{call,map}`, whose
 * body-key check reaches `llm_config:`, the second key, before the inline `nodes:` key and before
 * the missing-`call:` check.
 *
 * Later (T3-16, the loop kind): the loop row of `test_child_label_is_not_authorable` and
 * `load-map-call/engine_inline_composite_bodies__llm_config_on_an_inline_loop`.
 *
 * Later (T3-17, inline body extraction and registration):
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
 * `test_inline_body_shape_errors_are_located_without_private_ids`,
 * `test_register_inline_bodies_associates_call_map_loop_and_stays_lazy`,
 * `test_register_inline_body_rejects_bad_association_located_and_redacted`,
 * `test_register_inline_bodies_preflights_all_before_first_registration`.
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
 * `test_load_translates_recursive_inline_preflight_access_to_load_error`,
 * `test_loader_allows_private_callable_prefix_in_graph_node_id`,
 * `test_inline_body_shape_matrix_flows_through_loader`,
 * `test_inline_body_non_text_input_key_fails_at_the_key_through_loader`,
 * `test_loader_rejects_actual_sibling_private_digest_without_exposing_it`,
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

import { expect, test } from "bun:test";

import { BuildContext } from "../../src/compose/nodes/base.ts";
import { CallDescriptor } from "../../src/compose/nodes/call.ts";
import { MapDescriptor } from "../../src/compose/nodes/map.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { type CaseRow, handChild, loadError, parsedNodes, runCase } from "../compose/declarations.ts";

test.each([
  ["call", "call: named", "asserts, call, depends_on, env, input, kind, node_name, runs_after"],
  [
    "map",
    "call: named\n    over: ${input.items}",
    "asserts, call, depends_on, env, input, kind, node_name, over, parallel, runs_after",
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

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  { id: "load-map-call/engine_inline_composite_bodies__llm_config_on_an_inline_call", call: parsedNodes },
  { id: "load-map-call/engine_inline_composite_bodies__llm_config_on_an_inline_map", call: parsedNodes },
])("parseNodes: $id", runCase);
