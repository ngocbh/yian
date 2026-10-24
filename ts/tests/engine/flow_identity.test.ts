/**
 * Stable authored-flow manifests and deterministic expansion-topology digests
 * (`compile/manifest.ts`): rows of `tests/engine/test_flow_identity.py`.
 *
 * Ported: `test_flow_manifest_emits_the_literal_node_kind`,
 * `test_flow_fingerprint_covers_each_included_common_node_field`,
 * `test_node_origin_does_not_affect_program_or_expansion_identity`,
 * `test_node_origin_context_does_not_affect_program_or_expansion_identity`,
 * `test_prompts_code_and_tool_ids_keep_uri_text`, `test_flow_fingerprint_covers_each_parameter_field`,
 * `test_flow_fingerprint_covers_parameter_default_independently`,
 * `test_flow_fingerprint_preserves_parameter_order`,
 * `test_parameter_origins_do_not_affect_program_or_expansion_identity`,
 * `test_output_contract_origins_do_not_affect_program_or_expansion_identity`,
 * `test_output_contract_cutover_preserves_manifest_shape_and_fingerprint`,
 * `test_numeric_bounds_are_part_of_type_identity_only_when_present`,
 * `test_numeric_bounds_remain_in_nested_parameter_contract_identity`,
 * `test_output_contract_manifest_rejects_unknown_raw_and_subclass_bypasses`,
 * `test_parameter_manifest_rejects_unknown_fields_and_raw_bypass`,
 * `test_union_free_manifest_has_no_union_key_and_keeps_its_fingerprint`,
 * `test_union_manifest_orders_members_canonically`, `test_deep_untagged_union_manifests_quickly`,
 * `test_flow_fingerprint_covers_each_structural_type_field`,
 * `test_flow_fingerprint_preserves_structural_field_order_used_by_agent_schema` (the fingerprint
 * half), `test_start_manifest_uses_canonical_parameters`,
 * `test_flow_fingerprint_preserves_start_parameter_order`,
 * `test_flow_fingerprint_covers_each_end_field`, `test_value_end_manifest_keeps_the_record_encoding`,
 * `test_zero_and_one_name_record_end_manifests_are_keyed`,
 * `test_two_name_record_end_manifest_is_not_keyed`, `test_flow_fingerprint_covers_each_edge_field`,
 * `test_flow_fingerprint_covers_flow_boundary_ids`,
 * `test_expansion_digest_changes_for_every_builder_topology_component`,
 * `test_expansion_digest_is_stable_across_canonical_mapping_order`,
 * `test_flow_fingerprint_covers_each_included_kind_field`,
 * `test_flow_manifest_covers_remaining_kind_execution_fields` (its END row; its other five rows
 * repeat kind-field rows),
 * `test_flow_manifest_includes_human_input_title_but_ignores_inert_map_parallel`,
 * `test_flow_manifest_allows_shared_child_aliases_but_rejects_child_cycles`,
 * `test_wait_fingerprint_and_expansion_digest_cover_absolute_until_wiring`,
 * `test_case_predicate_origins_do_not_affect_program_or_expansion_identity`,
 * `test_case_manifest_rejects_a_mutated_non_origin_predicate_carrier`,
 * `test_memory_scope_is_not_authored_identity_but_is_expansion_topology`,
 * `test_memory_expansion_projection_covers_keys_and_occurrence_topology`,
 * `test_memory_expansion_projection_fields`,
 * `test_agent_variable_manifest_preserves_the_variable_free_fingerprint` (the same pin as Python's),
 * `test_agent_variable_identity_tracks_names_domains_defaults_and_order_only`,
 * `test_agent_variable_manifest_rejects_noncanonical_containers` and
 * `test_expansion_digest_excludes_effective_but_includes_authored_llm_config`.
 *
 * Rebased on hand-built flows (Python builds the baseline with the loader): `fragment()` mirrors
 * Python's `_fragment()`, and `compiled()` stands in for `_compiled()`'s loaded flow.
 * `test_flow_fingerprint_is_stable_across_reloads` (built twice),
 * `test_flow_fingerprint_preserves_semantic_edge_order`,
 * `test_flow_fingerprint_covers_flow_output_fields_and_order`,
 * `test_node_inputs_and_flow_outputs_keep_literals_alike`,
 * `test_flow_fingerprint_changes_with_wiring_types_asserts_and_env`,
 * `test_flow_manifest_sorts_mapping_keys_without_reordering_semantic_lists`,
 * `test_manifest_rejects_non_json_values_without_repr_fallback`,
 * `test_manifest_rejects_unknown_execution_fields_instead_of_omitting_them` (with the retired
 * node-range fields of `tests/engine/test_node_range.py::test_manifest_rejects_retired_node_range_fields`
 * as rows), `test_manifest_emits_a_node_type_table_only_when_non_empty`,
 * `test_manifest_rejects_a_malformed_type_table`,
 * `test_manifest_has_no_redaction_markers`,
 * `test_credential_named_parameter_without_default_has_the_null_default`, and three of Python's
 * parametrized rows each of `test_node_input_literals_are_identity`,
 * `test_declaration_default_literals_are_identity`, `test_env_literals_are_identity` (node scope;
 * the flow scope is a loader merge into the node's `env`),
 * `test_templates_under_any_name_keep_their_source_text` and
 * `test_env_values_under_credential_like_names_are_identity` (their other parametrized cases test
 * the same rule and aren't ported one by one), `test_total_loop_manifest_omits_returned_fields`,
 * `test_loop_without_max_has_null_max_in_manifest_and_its_own_fingerprint` (built twice instead of
 * loaded twice), `test_memory_expansion_projection_omits_root_id_and_config_but_hashes_keys` (the
 * memory layout's `memoryScope` and the flow `memories` set by hand),
 * `test_expansion_digest_applies_topology_only_policy_to_recursive_children` (a CALL child
 * `CompiledFlow` with a `flowLlmConfig` layer), and
 * `tests/engine/test_call_subgraph_builder.py::test_growth_metadata_and_execution_topology_have_a_separate_digest_policy`
 * (a hand-grown `fragment()` instead of `call_subgraph`'s).
 * `test_flow_fingerprint_includes_authored_llm_layers_but_not_effective_values`,
 * `test_manifest_records_blocks_as_nested_sorted_mappings`,
 * `test_flow_fingerprint_tracks_block_fields_but_not_key_order` and
 * `test_block_field_names_are_not_redacted_as_credentials` build `llmFlow()`, a CompiledFlow with
 * one AGENT step whose layers are the loaded flow's (the loader's `inherit: false` is
 * `llmInherit: false`); and
 * `tests/compose/test_agent_skills_yaml.py::test_skills_are_identity_but_empty_and_order_are_not`
 * and the manifest assertion of
 * `tests/compose/test_agent_mode_removed.py::test_agent_manifest_has_no_mode` use hand-built AGENT
 * nodes.
 * `test_flow_fingerprint_changes_with_node_kind_and_execution_source` and
 * `test_flow_fingerprint_recursively_includes_static_child_flows` are covered by the kind-literal
 * and kind-field rows.
 *
 * Changed:
 * - TS yian has no HUMAN_INPUT or WAIT kind, so their rows are the `ask` and `sleep` TOOL
 *   built-ins'. `test_flow_fingerprint_covers_each_included_kind_field` keeps HUMAN_INPUT's answer
 *   schema as an ask row; an ask's prompt and questions are inputs, identity through its wiring
 *   like the WAIT `until` row's (now a `sleep`'s). In
 *   `test_flow_manifest_includes_human_input_title_but_ignores_inert_map_parallel` an ask's title
 *   (its `node_name:`) is identity, as Python's HUMAN_INPUT title is; another tool's is display
 *   metadata.
 * - Node, wiring and type tables are `Map`s, so "shuffle mapping order" rebuilds `Map`s and
 *   records in reverse insertion order. Carriers are frozen: Python's `replace(param, ...)` builds
 *   a new `Parameter`, and `object.__setattr__(carrier, "future_field", True)` is a forged carrier
 *   (`Object.assign(Object.create(Cls.prototype), carrier, {futureField: 1})`). Edges are frozen
 *   and a Flow's `startId`/`endId` are readonly, so the edge and boundary rows rebuild them.
 * - Ints in values are `bigint`s, since a `number` is a float; Python's pinned hex digests are TS
 *   pins (TS fingerprints are SHA-256 over RFC 8785 text of TS's own manifest). Numeric bounds are
 *   typed values (`{"type": "int", "value": "1"}`), not raw JSON numbers.
 * - Python's `ValueError` is a `RangeError`; error names are TS field names (`outputContract`,
 *   `futureField`) in text form (`["futureField"]`).
 * - `Case` is frozen, so the forged-origin row builds a forged `Case`; `_memory_scope_topology`
 *   isn't exported, so the projection-fields row compares the expansion digest with the SHA-256 of
 *   the expected manifest.
 * - `test_agent_variable_manifest_rejects_noncanonical_containers`: TS has no tuple, so its rows
 *   are `null` and a `Set`.
 * - The non-JSON row uses a `Map`, a `Uint8Array`, a class instance, a function and `undefined`.
 * - The malformed type-table rows are a record instead of a `Map`, a `Map` with a `bigint` key, and
 *   a `Map` whose value is the string `"int"`.
 * - The deep-union row has no call counter (an ES module can't be monkeypatched): a depth-30 union
 *   must manifest within the test's timeout, which an exponential encoding never does.
 * - The census is `Object.keys`, so the trait fields, which a kind fixes, are ignored fields
 *   (`kind` is emitted): an instance override of a trait goes unnoticed (Python would refuse
 *   `node.is_spawner = True` as unknown).
 * - A lone surrogate in a name or value is left to `canonicalJson`'s message, which carries no
 *   path (load refuses one first).
 *
 * TS-only: typed floats (`1.5`, `Infinity`, `-0`), NaN refused, `2n` vs `2`, a cyclic env record vs
 * a shared alias, a `FileRef` encoded as its mapping, `params`, `preAsserts` and `postAsserts` that
 * aren't arrays, a forged END count, a list END's `names`, seed-path indexes as counts, the pinned
 * fingerprint and expansion digest of `fragment()`, a missing node field, a node whose `kind` isn't
 * its class's, a CODE node whose `kind` is `"agent"` refused as not an `AgentNode`, a node value
 * that isn't a `Node`, labels and a CODE or TOOL title or base directory outside identity, the
 * CODE descriptors (`{"inline": ...}`, `{"ref": ...}`, `{"builtin": "identity", "field": ...}`) and
 * their refusals, and TOOL `retries` (absent at 0,
 * identity otherwise, a forged count refused), and the flow fields: a typed wiring binding's
 * `binding_types` and a typed output's `type`, a plain `Flow`'s missing CompiledFlow keys, the flow
 * `llm_config:` layer (literal values are identity and key order isn't; the `ValueTemplate` and
 * `Binding` arms), the LLM value types and adjacency index outside identity, the flow id and flow
 * memories (identity when present, never expansion topology, forged values refused in both modes),
 * the root outputs and LLM layer outside the expansion digest, and an unknown flow field, a `Flow`
 * subclass, a non-Flow, a wiring value that isn't a `Binding`, a node wiring that isn't a `Map`,
 * node and wiring tables that aren't `Map`s, a non-string node, wiring or parameter key, an edge
 * that isn't an `Edge`, a forged `Edge`, a `flowLlmConfig` that isn't a record, edges, outputs and
 * memories that aren't arrays and an output that isn't a `FlowOutput` refused; an ask's answer
 * schema as a typed value and its title left out, a CALL child that isn't a `Flow` refused, a literal in MAP wiring and a
 * CALL argument literal as identity, LOOP `times` as a typed int (`null` for a `while` loop),
 * forged LOOP counts, a CASE branch that isn't a `Case` and CASE `cases` that isn't an array
 * refused, forged memory scopes and segments and a non-`ExecutionScope` scope refused, and the
 * execution-scope projection; and for AGENT: unset optional keys left out, `retries` present at
 * zero and a forged count refused, the run-resolved and derived fields (`llmConfig`,
 * `llmConfigLabel`, `wiresLlmConfig`, `llmValueTypes`, `callTime`, `hostCredentials`) outside
 * identity, and an unknown field, `skills` or `tools` that aren't arrays and a skill that isn't a
 * string refused.
 *
 * Ported through `loadFlow` (at the end): `test_loaded_flow_stores_only_exact_parameter_carriers`,
 * `test_flow_fingerprint_ignores_comments_and_python_object_identity` (the two loads' `step` nodes
 * are distinct objects, for Python's `_code` objects),
 * `test_inline_identity_ignores_formatting_but_tracks_owner_and_child_semantics`,
 * `test_inline_manifest_canonicalizes_generated_structural_input_types`,
 * `test_reference_code_identity_is_the_token_not_the_callable_body` (a TS module in a temporary
 * `baseDir`, `./identity_refmod.ts:fn`, which the load checks exists but never imports),
 * `test_a_flow_file_cannot_put_api_key_in_a_manifest`,
 * `test_flow_without_formerly_redacted_values_keeps_its_fingerprint`,
 * `test_loaded_flow_output_forms_keep_their_fingerprints`,
 * `test_loaded_one_field_record_flow_has_its_pinned_fingerprint`,
 * `test_loaded_zero_and_one_field_records_differ_from_their_value_forms`,
 * `test_call_over_a_single_node_def_keeps_its_fingerprint`,
 * `test_loaded_value_output_and_single_node_def_use_a_value_end` and
 * `test_loop_with_max_keeps_its_fingerprint`; the pinned fingerprints are TS pins: the manifest
 * writes an inline CODE body as `{inline: <body>}` (design 6.7) where Python writes the bare text,
 * and the inline bodies are TS (`+ 1n`); with both put back, Python's pins reproduce.
 *
 * Ported through `loadFlow`, `runFlow` and `advance`:
 * `test_resume_rejects_a_flow_with_an_edited_literal`, the mismatch a `RangeError` where Python
 * raises a `ValueError`; resume compiles the flow the run recorded, so the literal is edited in
 * the recorded text (Python passes the edited flow).
 *
 * Not ported: `test_flow_fingerprint_distinguishes_yaml_literal_types`,
 * `test_flow_fingerprint_supports_binary_nonfinite_and_non_string_mapping_keys` and
 * `test_flow_fingerprint_tags_programmatic_container_types` (TS values have no date, set, tuple or
 * bytes, and record keys are always strings); `test_field_census_supports_declared_object_models`
 * and `test_field_census_detects_pydantic_extra_fields_without_internal_slots` (the census is
 * `Object.keys`; no dataclass, attrs, pydantic or slots); and
 * `test_parameter_manifest_rejects_noncanonical_containers` (TS has no tuple; the TS-only
 * non-array `params` row stands in).
 */

import { expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { expansionDigest, flowFingerprint, flowManifest } from "../../src/compile/manifest.ts";
import { CompiledFlow, Edge, END_ID, Flow, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { LoadError } from "../../src/compose/helpers/errors.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { compareCodeUnits, OriginRef } from "../../src/diagnostics.ts";
import {
  DeclSite,
  MemoryDecl,
  MemoryOccurrenceKind,
  MemoryOccurrenceSegment,
  MemoryScope,
} from "../../src/memory/contracts.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import type { Node } from "../../src/nodes/base.ts";
import {
  Assertion,
  Binding,
  ChildFlowBoundary,
  ContractOrigins,
  ExecutionPosition,
  ExecutionScope,
  isExactly,
  Parameter,
  SeedConnection,
  ValueContract,
  ValueTemplate,
} from "../../src/nodes/contracts.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { Case, CaseNode } from "../../src/nodes/case/node.ts";
import { type BuiltinCode, CodeNode } from "../../src/nodes/code/node.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { advance, runFlow } from "../../src/run.ts";
import { FileRef, Type, typeFor, ValueKind } from "../../src/typesys/index.ts";
import { canonicalJson, type JsonObject, type JsonValue } from "../../src/typesys/json.ts";
import { editRecordedFlow } from "./run_state_fixtures.ts";

const HEX = /^[0-9a-f]{64}$/;

// The fingerprint (and, for this plain Flow, the equal expansion digest) of `fragment()`.
const FRAGMENT_FINGERPRINT = "e0697a4672fd30610ccf14fdf654d54d174c265ad664e06e55f62c7d2808a3b3";

/** An `int` scalar type. */
function intType(): Type {
  return Type.scalar(ValueKind.INTEGER);
}

/** An origin-free `int` value contract. */
function intContract(): ValueContract {
  return new ValueContract(intType());
}

/** A `str` value contract. */
function strContract(): ValueContract {
  return new ValueContract(Type.scalar(ValueKind.STRING));
}

/** Build a small valid fragment with literal wiring and two ordered edges (Python's `_fragment()`). */
function fragment(): Flow {
  const start = new StartNode("entry", { params: [] });
  const body = new CodeNode("body", { code: 'return inputs["x"]' });
  body.params = [new Parameter("x", { contract: intContract() })];
  body.outputContract = intContract();
  const end = EndNode.value("terminal", { outputName: "result" });
  return new Flow(
    new Map<string, Node>([
      ["entry", start],
      ["body", body],
      ["terminal", end],
    ]),
    [
      new Edge("entry->body", "entry", "body", { ordering: true }),
      new Edge("body->terminal", "body", "terminal", { inputGroup: "result" }),
    ],
    {
      wiring: new Map([
        ["entry", new Map()],
        ["body", new Map([["x", 1n]])],
        ["terminal", new Map([["result", "${body.output}"]])],
      ]),
      startId: "entry",
      endId: "terminal",
    },
  );
}

/** The CompiledFlow Python's `_compiled()` loads: an int input, one asserted CODE step and env. */
function compiled(): CompiledFlow {
  const start = new StartNode(START_ID, {
    params: [new Parameter("x", { contract: intContract(), hasDefault: true, default: 1n })],
  });
  const step = new CodeNode("step", { code: 'return inputs["x"] + 1' });
  step.params = [new Parameter("x", { contract: intContract() })];
  step.outputContract = intContract();
  step.preAsserts = [new Assertion("x >= 0", "x >= 0"), new Assertion("x < 100", "x < 100")];
  step.postAsserts = [new Assertion("output > x", "output > x")];
  step.env = { alpha: 1n, beta: 2n };
  const end = EndNode.value(END_ID, { outputName: "result" });
  return CompiledFlow.fromParts(
    new Map<string, Node>([
      [START_ID, start],
      ["step", step],
      [END_ID, end],
    ]),
    [
      new Edge(`${START_ID}->step`, START_ID, "step", { inputGroup: "x" }),
      new Edge(`step->${END_ID}`, "step", END_ID, { inputGroup: "result" }),
    ],
    {
      wiring: new Map([
        [START_ID, new Map()],
        ["step", new Map([["x", "${input.x}"]])],
        [END_ID, new Map([["result", "${step.output}"]])],
      ]),
      flowId: "stable",
    },
  );
}

/** A CompiledFlow whose one CODE step (`return "ok"`) takes `wiring`, with `inputs` and `env`. */
function oneStepFlow(
  options: { inputs?: Parameter[]; wiring?: [string, unknown][]; env?: Record<string, unknown> } = {},
): CompiledFlow {
  const { inputs = [], wiring = [], env = {} } = options;
  const step = new CodeNode("step", { code: 'return "ok"' });
  step.params = wiring.map(([name]) => new Parameter(name));
  step.env = env;
  return CompiledFlow.fromParts(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID, { params: inputs })],
      ["step", step],
      [END_ID, EndNode.value(END_ID, { outputName: "result" })],
    ]),
    [new Edge(`step->${END_ID}`, "step", END_ID, { inputGroup: "result" })],
    {
      wiring: new Map([
        [START_ID, new Map()],
        ["step", new Map(wiring)],
        [END_ID, new Map([["result", "${step.output}"]])],
      ]),
    },
  );
}

/** Wrap one node so kind-specific manifest fields can be compared. */
function singleNodeFlow(node: Node): Flow {
  return new Flow(new Map([[node.id, node]]), [], {
    wiring: new Map([[node.id, new Map()]]),
    startId: node.id,
    endId: node.id,
  });
}

/** The CODE node `id` of `flow`. */
function code(flow: Flow, id: string): CodeNode {
  return flow.nodes.get(id) as CodeNode;
}

/** The manifest of node `id` in `flow`'s manifest. */
function nodeManifest(flow: Flow, id: string): JsonObject {
  return (flowManifest(flow).nodes as JsonObject)[id] as JsonObject;
}

/** Return one string-keyed value from the manifest's typed mapping form. */
function typedMappingValue(mapping: JsonValue, key: string): JsonValue {
  const matches = ((mapping as JsonObject).entries as JsonObject[])
    .filter((entry) => (entry.key as JsonObject).value === key)
    .map((entry) => entry.value!);
  expect(matches).toHaveLength(1);
  return matches[0]!;
}

/** The typed env value `key` of node `id` in `flow`'s manifest. */
function envValue(flow: Flow, id: string, key: string): JsonValue {
  return typedMappingValue(nodeManifest(flow, id).env!, key);
}

/** Assert that `fn` throws exactly `cls` (not a subclass) with exactly `message`. */
function expectError(fn: () => unknown, cls: typeof Error, message: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(cls);
  expect((caught as Error).constructor).toBe(cls);
  expect((caught as Error).message).toBe(message);
}

/** A closed record tagged by a single-value `action` Literal. */
function member(tag: string, fields: Record<string, ValueKind> = {}): Type {
  const recordFields = new Map<string, Type>([["action", new Type({ kind: ValueKind.STRING, tags: new Set([tag]) })]]);
  for (const [name, kind] of Object.entries(fields)) recordFields.set(name, Type.scalar(kind));
  return new Type({ kind: ValueKind.OBJECT, fields: recordFields, required: new Set(recordFields.keys()) });
}

/** A union of record members. */
function union(...members: Type[]): Type {
  return new Type({ kind: ValueKind.OBJECT, variants: members });
}

/** `fragment()` with `type` as the body's output contract. */
function fragmentWithOutput(type: Type): Flow {
  const flow = fragment();
  code(flow, "body").outputContract = new ValueContract(type);
  return flow;
}

class ParameterSubclass extends Parameter {}

class ValueContractSubclass extends ValueContract {}

class Opaque {}

// ---- Stability, kind and mapping order --------------------------------------------------------

test("test_flow_fingerprint_is_stable_across_reloads", () => {
  const first = compiled();
  const second = compiled();

  expect(first).not.toBe(second);
  expect(first.nodes.get("step")).not.toBe(second.nodes.get("step"));
  expect(flowManifest(first)).toEqual(flowManifest(second));
  expect(flowFingerprint(first)).toBe(flowFingerprint(second));
  expect(flowFingerprint(first)).toMatch(HEX);
});

test("the fragment's fingerprint and expansion digest are pinned", () => {
  expect(flowFingerprint(fragment())).toBe(FRAGMENT_FINGERPRINT);
  expect(expansionDigest(fragment())).toBe(FRAGMENT_FINGERPRINT);
});

test("a seed-path index is a count: -0 fingerprints like 0 and an unsafe int is refused", () => {
  /** `fragment()` with its entry as a child START whose wired `x` is seeded from item `index`. */
  const seeded = (index: number): Flow => {
    const base = fragment();
    base.nodes.set("entry", new StartNode("entry", { params: [new Parameter("x", { contract: intContract() })] }));
    return new Flow(base.nodes, base.edges, {
      wiring: new Map<string, ReadonlyMap<string, unknown>>([...base.wiring, ["entry", new Map([["x", 1n]])]]),
      startId: "entry",
      endId: "terminal",
      inputConnections: [new SeedConnection("entry", "x", [index, "x"])],
      childBoundaries: [new ChildFlowBoundary("entry", "terminal")],
    });
  };

  expect(expansionDigest(seeded(-0))).toBe(expansionDigest(seeded(0)));
  expect(expansionDigest(seeded(1))).not.toBe(expansionDigest(seeded(0)));
  expectError(
    () => expansionDigest(seeded(2 ** 60)),
    TypeError,
    "expansion.inputConnections[0].seedPath[0]: expected a safe integer, got 1152921504606847000",
  );
});

test("test_flow_manifest_emits_the_literal_node_kind", () => {
  const codeManifest = nodeManifest(singleNodeFlow(new CodeNode("node", { code: 'return "ok"' })), "node");
  const toolManifest = nodeManifest(singleNodeFlow(new ToolNode("node", { toolId: "t", retries: 0 })), "node");

  expect(codeManifest.kind).toBe("code");
  expect(toolManifest.kind).toBe("tool");
});

test("test_flow_manifest_sorts_mapping_keys_without_reordering_semantic_lists", () => {
  const original = compiled();
  const shuffled = compiled();
  shuffled.nodes = new Map([...shuffled.nodes].reverse());
  shuffled.wiring = new Map([...shuffled.wiring].reverse().map(([id, wiring]) => [id, new Map([...wiring].reverse())]));
  const step = code(shuffled, "step");
  step.env = Object.fromEntries(Object.entries(step.env).reverse());

  const manifest = flowManifest(shuffled);
  expect(manifest).toEqual(flowManifest(original));
  expect(flowFingerprint(shuffled)).toBe(flowFingerprint(original));
  expect(manifest.edges).toEqual(flowManifest(original).edges!);
});

// ---- Node field policy -----------------------------------------------------------------------

test.each([["futureExecutionFlag"], ["range"], ["fieldRanges"], ["preAssertRanges"], ["postAssertRanges"]])(
  "test_manifest_rejects_unknown_execution_fields_instead_of_omitting_them: %s",
  (field) => {
    const flow = compiled();
    Object.assign(code(flow, "step"), { [field]: true });

    expectError(
      () => flowManifest(flow),
      RangeError,
      `flow.nodes.step: unknown noncanonical execution field(s) for code: ["${field}"]`,
    );
  },
);

test("a node missing a field is refused", () => {
  const flow = compiled();
  Reflect.deleteProperty(code(flow, "step"), "env");

  expectError(() => flowManifest(flow), RangeError, 'flow.nodes.step: missing canonical field(s): ["env"]');
});

test.each([
  ["tool", RangeError, "flow.nodes.step: node kind 'tool' requires ToolNode, got CodeNode"],
  ["nope", RangeError, "flow.nodes.step: node kind must be a NodeKind"],
  ["agent", RangeError, "flow.nodes.step: node kind 'agent' requires AgentNode, got CodeNode"],
] as const)("a CODE node whose kind is %p is refused", (kind, cls, message) => {
  const flow = compiled();
  Object.assign(code(flow, "step"), { kind });

  expectError(() => flowManifest(flow), cls, message);
});

test("a node value that isn't a Node is refused", () => {
  const flow = compiled();
  // Flow keeps `nodes` by identity, so this forges a non-Node value under a node id.
  (flow.nodes as Map<string, unknown>).set("step", "not a node");

  expectError(() => flowManifest(flow), TypeError, "flow.nodes.step: expected Node, got string");
});

test("labels, a CODE or TOOL title and a base directory are not identity", () => {
  const original = compiled();
  const changed = compiled();
  const step = code(changed, "step");
  step.wiringLabels = new Map([["x", "cases[1].when"]]);
  step.bindingLabels = new Map([["x", "binding 'x'"]]);
  step.title = "A different title";
  step.baseDir = "/elsewhere";
  const tool = (title: string | null) => singleNodeFlow(new ToolNode("node", { toolId: "t", title, retries: 0 }));

  expect(flowManifest(changed)).toEqual(flowManifest(original));
  expect(flowFingerprint(changed)).toBe(flowFingerprint(original));
  expect(flowFingerprint(tool("Place"))).toBe(flowFingerprint(tool(null)));
});

test.each([
  ["id", (node: CodeNode) => (node.id = "renamed-body")],
  ["output-contract", (node: CodeNode) => (node.outputContract = new ValueContract(Type.scalar(ValueKind.NUMBER)))],
  ["pre-asserts", (node: CodeNode) => (node.preAsserts = [new Assertion("x > 0", "x > 0")])],
  ["post-asserts", (node: CodeNode) => (node.postAsserts = [new Assertion("output > 0", "output > 0")])],
  ["commit-as", (node: CodeNode) => (node.commitAs = "owner")],
  ["origin-id", (node: CodeNode) => (node.originId = "origin")],
  ["env", (node: CodeNode) => (node.env = { threshold: 2n })],
])("test_flow_fingerprint_covers_each_included_common_node_field: %s", (_, change) => {
  const original = fragment();
  const changed = fragment();
  change(code(changed, "body"));

  expect(flowFingerprint(changed)).not.toBe(flowFingerprint(original));
});

test("test_node_origin_does_not_affect_program_or_expansion_identity", () => {
  const first = fragment();
  const second = fragment();
  code(second, "body").origin = new OriginRef("root/node/body");

  expect(flowManifest(first)).toEqual(flowManifest(second));
  expect(flowFingerprint(first)).toBe(flowFingerprint(second));
  expect(expansionDigest(first)).toBe(expansionDigest(second));
});

test("test_node_origin_context_does_not_affect_program_or_expansion_identity", () => {
  const first = fragment();
  const second = fragment();
  code(second, "body").originContext = [new OriginRef("root/nodes/invoke"), new OriginRef("child/nodes/map")];

  expect(flowManifest(first)).toEqual(flowManifest(second));
  expect(flowFingerprint(first)).toBe(flowFingerprint(second));
  expect(expansionDigest(first)).toBe(expansionDigest(second));
});

test("test_manifest_emits_a_node_type_table_only_when_non_empty", () => {
  const tagged = (tag: string) => new Type({ kind: ValueKind.STRING, tags: new Set([tag]) });
  const withTypes = (types: Map<string, Type>) => {
    const flow = compiled();
    code(flow, "step").types = types;
    return flow;
  };
  const original = compiled();
  const typed = withTypes(
    new Map([
      ["Small", tagged("small")],
      ["Big", tagged("big")],
    ]),
  );
  const retyped = withTypes(
    new Map([
      ["Small", tagged("small")],
      ["Big", tagged("huge")],
    ]),
  );

  for (const node of Object.values(flowManifest(original).nodes as JsonObject)) {
    expect("types" in (node as JsonObject)).toBe(false);
  }
  const table = nodeManifest(typed, "step").types as JsonObject[];
  expect(table.map((entry) => entry.name)).toEqual(["Big", "Small"]);
  expect(flowFingerprint(typed)).not.toBe(flowFingerprint(original));
  expect(flowFingerprint(retyped)).not.toBe(flowFingerprint(typed));
  expect(flowFingerprint(withTypes(new Map()))).toBe(flowFingerprint(original));
});

test.each([
  ["record", { Big: Type.scalar(ValueKind.INTEGER) }, "flow.nodes.step.types: expected Map, got Object"],
  [
    "non-string key",
    new Map([[1n, Type.scalar(ValueKind.INTEGER)]]),
    "flow.nodes.step.types: keys must be strings, got bigint",
  ],
  ["non-Type value", new Map([["Big", "int"]]), "flow.nodes.step.types.Big: expected Type, got string"],
])("test_manifest_rejects_a_malformed_type_table: %s", (_, types, message) => {
  const flow = compiled();
  Object.assign(code(flow, "step"), { types });

  expectError(() => flowManifest(flow), TypeError, message);
});

// ---- CODE and TOOL ---------------------------------------------------------------------------

/** The manifest of a lone CODE node `node` whose `code` is `source`. */
function codeNodeManifest(source: string | BuiltinCode): JsonObject {
  return nodeManifest(singleNodeFlow(new CodeNode("node", { code: source })), "node");
}

/** The fingerprint of a lone CODE node `node` whose `code` is `source`. */
function codeFingerprint(source: string | BuiltinCode, options: Partial<CodeNode> = {}): string {
  return flowFingerprint(singleNodeFlow(Object.assign(new CodeNode("node", { code: source }), options)));
}

test("CODE text is described as an inline body or a reference, and a built-in as its form", () => {
  expect(codeNodeManifest('return "ok"').code).toEqual({ inline: 'return "ok"' });
  expect(codeNodeManifest("./h.ts:fn").code).toEqual({ ref: "./h.ts:fn" });
  expect(codeNodeManifest({ builtin: "identity", field: "x" }).code).toEqual({ builtin: "identity", field: "x" });
});

test("different CODE texts and built-in fields fingerprint differently", () => {
  expect(codeFingerprint("return 1")).not.toBe(codeFingerprint("return 2"));
  expect(codeFingerprint("./h.ts:fn")).not.toBe(codeFingerprint("./h.ts:other"));
  expect(codeFingerprint({ builtin: "identity", field: "x" })).not.toBe(
    codeFingerprint({ builtin: "identity", field: "y" }),
  );
  expect(codeFingerprint("./h.ts:fn")).not.toBe(codeFingerprint("return 1"));
});

test("CODE mode, base directory, source file and origins keep the fingerprint", () => {
  const baseline = codeFingerprint("./h.ts:fn");

  expect(codeFingerprint("./h.ts:fn", { mode: "inline" })).toBe(baseline);
  expect(codeFingerprint("./h.ts:fn", { baseDir: "/flows" })).toBe(baseline);
  expect(codeFingerprint("./h.ts:fn", { sourceFile: "flow.yaml" })).toBe(baseline);
  expect(codeFingerprint("./h.ts:fn", { codeOrigin: new OriginRef("code") })).toBe(baseline);
  expect(codeFingerprint("./h.ts:fn", { codeLineOrigins: [new OriginRef("code/line/0")] })).toBe(baseline);
});

test.each([
  ["a number", 42, TypeError, "flow.nodes.node.code: expected code text or a built-in operation, got number"],
  [
    "a built-in with an extra key",
    { builtin: "identity", field: "x", extra: 1n },
    RangeError,
    'flow.nodes.node.code: unknown noncanonical field(s): ["extra"]',
  ],
  [
    "an unknown built-in",
    { builtin: "nope", field: "x" },
    TypeError,
    "flow.nodes.node.code: expected code text or a built-in operation, got Object",
  ],
  [
    "an empty built-in field",
    { builtin: "identity", field: "" },
    RangeError,
    "flow.nodes.node.code: built-in field must be non-empty",
  ],
] as const)("CODE code that is %s is refused", (_, value, cls, message) => {
  const node = new CodeNode("node", { code: 'return "ok"' });
  Object.assign(node, { code: value });

  expectError(() => flowManifest(singleNodeFlow(node)), cls, message);
});

test("test_prompts_code_and_tool_ids_keep_uri_text", () => {
  const firstUri = "https://first-user:first-secret@example.test/path?token=first-token";
  const secondUri = "https://second-user:second-secret@example.test/path?token=second-token";
  const pairs: [Node, Node][] = [
    [
      new CodeNode("node", { code: `return ${JSON.stringify(firstUri)}` }),
      new CodeNode("node", { code: `return ${JSON.stringify(secondUri)}` }),
    ],
    [new ToolNode("node", { toolId: firstUri, retries: 0 }), new ToolNode("node", { toolId: secondUri, retries: 0 })],
    [new AgentNode("node", { prompt: firstUri }), new AgentNode("node", { prompt: secondUri })],
  ];

  for (const [first, second] of pairs) {
    expect(flowFingerprint(singleNodeFlow(first))).not.toBe(flowFingerprint(singleNodeFlow(second)));
    expect(canonicalJson(flowManifest(singleNodeFlow(first)))).toContain("first-secret");
  }
});

test("TOOL retries is absent at zero and identity when non-zero", () => {
  const tool = (retries: number) => singleNodeFlow(new ToolNode("node", { toolId: "t", retries }));
  const zero = nodeManifest(tool(0), "node");

  expect(zero.tool_id).toBe("t");
  expect("retries" in zero).toBe(false);
  expect(nodeManifest(tool(2), "node").retries).toBe(2);
  expect(flowFingerprint(tool(2))).not.toBe(flowFingerprint(tool(0)));
  expect(flowFingerprint(tool(2))).not.toBe(flowFingerprint(tool(1)));
});

test("a forged TOOL retries count is refused with its path", () => {
  const node = new ToolNode("node", { toolId: "t", retries: 0 });
  node.retries = 1.5;

  expectError(
    () => flowManifest(singleNodeFlow(node)),
    TypeError,
    "flow.nodes.node.retries: expected a safe integer, got 1.5",
  );
});

// ---- AGENT -----------------------------------------------------------------------------------

/**
 * A CompiledFlow whose one AGENT step `ask` answers briefly with text, under the flow layer
 * `flowLayer` and its own layer `nodeLayer`.
 */
function llmFlow(
  flowLayer: CompiledFlow["flowLlmConfig"],
  nodeLayer: Record<string, unknown> | null = null,
  llmInherit = true,
): CompiledFlow {
  const ask = new AgentNode("ask", { prompt: "Answer briefly.", llmConfig: nodeLayer, llmInherit });
  ask.outputContract = strContract();
  return CompiledFlow.fromParts(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID, { params: [] })],
      ["ask", ask],
      [END_ID, EndNode.value(END_ID, { outputName: "result" })],
    ]),
    [new Edge(`ask->${END_ID}`, "ask", END_ID, { inputGroup: "result" })],
    {
      wiring: new Map([
        [START_ID, new Map()],
        ["ask", new Map()],
        [END_ID, new Map([["result", "${ask.output}"]])],
      ]),
      flowLlmConfig: flowLayer,
      flowId: "llm",
    },
  );
}

/** The AGENT node `ask` of `flow`. */
function ask(flow: Flow): AgentNode {
  return flow.nodes.get("ask") as AgentNode;
}

/** Unwrap the manifest's typed mapping form into plain nested values, checking keys are sorted. */
function plainMapping(canonical: JsonValue): unknown {
  const value = canonical as JsonObject;
  if (value.type !== "mapping") return value.value;
  const entries = value.entries as JsonObject[];
  const keys = entries.map((entry) => (entry.key as JsonObject).value as string);
  expect(keys).toEqual([...keys].sort(compareCodeUnits));
  return Object.fromEntries(entries.map((entry, index) => [keys[index], plainMapping(entry.value!)]));
}

test("test_flow_fingerprint_includes_authored_llm_layers_but_not_effective_values", () => {
  const flowLayer = () => ({ provider: "openai", openai: { temperature: 0.2 } });
  const original = () => llmFlow(flowLayer(), { model: "gpt-5.5" }, false);
  const baseline = flowFingerprint(original());

  const changedFlowLayer = llmFlow({ provider: "openai", openai: { temperature: 0.7 } }, { model: "gpt-5.5" }, false);
  const ownLayer = llmFlow(flowLayer(), { model: "gpt-5.5-pro" }, false);
  const inherit = llmFlow(flowLayer(), { model: "gpt-5.5" }, true);
  const effective = original();
  ask(effective).llmConfig = { provider: "anthropic", model: "claude-opus-4-8" };

  expect(flowFingerprint(changedFlowLayer)).not.toBe(baseline);
  expect(flowFingerprint(ownLayer)).not.toBe(baseline);
  expect(flowFingerprint(inherit)).not.toBe(baseline);
  expect(flowFingerprint(effective)).toBe(baseline);
});

/** Python's `_BLOCK_FLOW`, with its layers' keys in authored or reversed order. */
function blockFlow(options: { reordered?: boolean; effort?: string } = {}): CompiledFlow {
  const { reordered = false, effort = "low" } = options;
  const anthropic = reordered
    ? { output_config: { effort }, max_tokens: 2048n }
    : { max_tokens: 2048n, output_config: { effort } };
  const openai = { reasoning: { effort: "high" } };
  const flowLayer = reordered
    ? { openai, anthropic, provider: "anthropic" }
    : { provider: "anthropic", anthropic, openai };
  const thinking = reordered ? { budget_tokens: 1024n, type: "enabled" } : { type: "enabled", budget_tokens: 1024n };
  return llmFlow(flowLayer, { anthropic: { max_tokens: 512n, thinking } });
}

test("test_manifest_records_blocks_as_nested_sorted_mappings", () => {
  const manifest = flowManifest(blockFlow());

  expect(plainMapping(manifest.flow_llm_config!)).toEqual({
    anthropic: { max_tokens: "2048", output_config: { effort: "low" } },
    openai: { reasoning: { effort: "high" } },
    provider: "anthropic",
  });
  expect(plainMapping(((manifest.nodes as JsonObject).ask as JsonObject).own_llm_config!)).toEqual({
    anthropic: { max_tokens: "512", thinking: { budget_tokens: "1024", type: "enabled" } },
  });
});

test("test_flow_fingerprint_tracks_block_fields_but_not_key_order", () => {
  const baseline = flowFingerprint(blockFlow());
  const raised = blockFlow({ effort: "high" });
  const reordered = blockFlow({ reordered: true });

  expect(Object.keys(reordered.flowLlmConfig)).not.toEqual(Object.keys(blockFlow().flowLlmConfig));
  const ownLayer = ask(reordered).ownLlmConfig as { anthropic: { thinking: object } };
  expect(Object.keys(ownLayer.anthropic.thinking)).toEqual(["budget_tokens", "type"]);
  expect(flowFingerprint(raised)).not.toBe(baseline);
  expect(flowFingerprint(reordered)).toBe(baseline);
});

test("test_block_field_names_are_not_redacted_as_credentials", () => {
  const flow = llmFlow({
    provider: "openai",
    openai: { max_output_tokens: 7771n, safety_identifier: "safety-id-value", prompt_cache_key: "cache-key-value" },
    anthropic: { max_tokens: 7772n, thinking: { type: "enabled", budget_tokens: 7773n } },
    ollama: { keep_alive: "keep-alive-value" },
    google: { generationConfig: { thinkingConfig: { thinkingBudget: 7774n } } },
  });
  const flowConfig = plainMapping(flowManifest(flow).flow_llm_config!) as Record<string, unknown>;

  expect(flowConfig.openai).toEqual({
    max_output_tokens: "7771",
    prompt_cache_key: "cache-key-value",
    safety_identifier: "safety-id-value",
  });
  expect(flowConfig.anthropic).toEqual({ max_tokens: "7772", thinking: { budget_tokens: "7773", type: "enabled" } });
  expect(flowConfig.ollama).toEqual({ keep_alive: "keep-alive-value" });
  expect(flowConfig.google).toEqual({ generationConfig: { thinkingConfig: { thinkingBudget: "7774" } } });
});

test("test_agent_variable_manifest_preserves_the_variable_free_fingerprint", () => {
  const flow = singleNodeFlow(new AgentNode("node", { prompt: "hi" }));

  expect("variables" in nodeManifest(flow, "node")).toBe(false);
  expect(flowFingerprint(flow)).toBe("4bf7361e46300e1e8c9250fbca0505b9ad985ce4159558c1528ac1bc29bf8bd4");
});

test("test_agent_variable_identity_tracks_names_domains_defaults_and_order_only", () => {
  const countType = (high: bigint) => new Type({ kind: ValueKind.INTEGER, bounds: [1n, high] });
  const count = new Parameter("count", { contract: new ValueContract(countType(12n)), hasDefault: true, default: 4n });
  const tone = new Parameter("tone", {
    contract: new ValueContract(new Type({ kind: ValueKind.STRING, tags: new Set(["brief", "detailed"]) })),
    hasDefault: true,
    default: "brief",
  });
  const declarations = [
    [count, tone],
    [new Parameter("sentences", { contract: count.contract, hasDefault: true, default: 4n }), tone],
    [new Parameter("count", { contract: count.contract, hasDefault: true, default: 5n }), tone],
    [new Parameter("count", { contract: new ValueContract(countType(8n)), hasDefault: true, default: 4n }), tone],
    [tone, count],
  ];
  const flows = declarations.map((variables) => singleNodeFlow(new AgentNode("node", { variables })));

  expect(new Set(flows.map(flowFingerprint)).size).toBe(flows.length);
  expect(new Set(flows.map(expansionDigest)).size).toBe(flows.length);
  expect((nodeManifest(flows[0]!, "node").variables as JsonObject[]).map((variable) => variable.name)).toEqual([
    "count",
    "tone",
  ]);

  const located = singleNodeFlow(
    new AgentNode("node", {
      variables: [
        new Parameter("count", {
          contract: new ValueContract(countType(12n), new ContractOrigins({ default: new OriginRef("type") })),
          hasDefault: true,
          default: 4n,
          origin: new OriginRef("declaration"),
          defaultOrigin: new OriginRef("default"),
        }),
        tone,
      ],
    }),
  );
  expect(flowFingerprint(located)).toBe(flowFingerprint(flows[0]!));
  expect(expansionDigest(located)).toBe(expansionDigest(flows[0]!));
});

test.each([
  [null, "null"],
  [new Set([new Parameter("x")]), "Set"],
])("test_agent_variable_manifest_rejects_noncanonical_containers: %p", (variables, name) => {
  const node = new AgentNode("node");
  Object.assign(node, { variables });

  expectError(
    () => flowManifest(singleNodeFlow(node)),
    TypeError,
    `flow.nodes.node.variables: expected array, got ${name}`,
  );
});

test("test_expansion_digest_excludes_effective_but_includes_authored_llm_config", () => {
  const first = new AgentNode("node", { prompt: "answer", llmConfig: { provider: "openai" } });
  const changedEffective = new AgentNode("node", { prompt: "answer", llmConfig: { provider: "openai" } });
  changedEffective.llmConfig = { provider: "anthropic" };
  const changedAuthored = new AgentNode("node", { prompt: "answer", llmConfig: { provider: "anthropic" } });

  expect(expansionDigest(singleNodeFlow(first))).toBe(expansionDigest(singleNodeFlow(changedEffective)));
  expect(expansionDigest(singleNodeFlow(first))).not.toBe(expansionDigest(singleNodeFlow(changedAuthored)));
});

test("an AGENT leaves out unset variables, skills, steerable, system prompt, memories and mode", () => {
  const manifest = nodeManifest(singleNodeFlow(new AgentNode("node", { prompt: "p" })), "node");

  for (const key of ["variables", "skills", "steerable", "system_prompt", "memories", "mode"]) {
    expect(key in manifest).toBe(false);
  }
  expect(manifest.retries).toBe(2);
});

test("test_skills_are_identity_but_empty_and_order_are_not", () => {
  const skills = (granted: readonly string[] | "all") =>
    flowFingerprint(singleNodeFlow(new AgentNode("node", { prompt: "p", skills: granted })));
  const granted = (names: readonly string[] | "all") =>
    nodeManifest(singleNodeFlow(new AgentNode("node", { skills: names })), "node").skills;
  const base = flowFingerprint(singleNodeFlow(new AgentNode("node", { prompt: "p" })));
  const ab = skills(["a", "b"]);

  expect(skills([])).toBe(base);
  expect(ab).not.toBe(base);
  expect(skills(["b", "a"])).toBe(ab);
  expect([base, ab]).not.toContain(skills("all"));
  expect(granted(["b", "a"])).toEqual(["a", "b"]);
  expect(granted("all")).toBe("all");
});

test("AGENT retries is identity even at zero, and a forged count is refused with its path", () => {
  const agent = (retries: number) => singleNodeFlow(new AgentNode("node", { prompt: "p", retries }));

  expect(nodeManifest(agent(0), "node").retries).toBe(0);
  const forged = new AgentNode("node", { prompt: "p" });
  forged.retries = 1.5;
  expectError(
    () => flowManifest(singleNodeFlow(forged)),
    TypeError,
    "flow.nodes.node.retries: expected a safe integer, got 1.5",
  );
});

test("AGENT run-resolved and derived fields are not identity", () => {
  const original = singleNodeFlow(new AgentNode("node", { prompt: "p", llmConfig: { model: "m" } }));
  const node = new AgentNode("node", { prompt: "p", llmConfig: { model: "m" } });
  Object.assign(node, {
    llmConfig: { model: "other" },
    llmConfigLabel: "openai/other",
    wiresLlmConfig: true,
    llmValueTypes: [[["model"], null]],
    callTime: { model: "other" },
    hostCredentials: { apiKey: "secret" },
  });

  expect(flowManifest(singleNodeFlow(node))).toEqual(flowManifest(original));
});

test.each([
  [
    "an unknown field",
    (node: AgentNode) => Object.assign(node, { futureField: 1 }),
    RangeError,
    'flow.nodes.node: unknown noncanonical execution field(s) for agent: ["futureField"]',
  ],
  [
    "skills that aren't an array",
    (node: AgentNode) => Object.assign(node, { skills: 5 }),
    TypeError,
    'flow.nodes.node.skills: expected "all" or an array, got number',
  ],
  [
    "a skill that isn't a string",
    (node: AgentNode) => Object.assign(node, { skills: ["a", 1] }),
    TypeError,
    "flow.nodes.node.skills[1]: expected a string, got number",
  ],
  [
    "tools that aren't an array",
    (node: AgentNode) => Object.assign(node, { tools: "t" }),
    TypeError,
    "flow.nodes.node.tools: expected array, got string",
  ],
] as const)("an AGENT with %s is refused", (_, mutate, cls, message) => {
  const node = new AgentNode("node", { prompt: "p" });
  mutate(node);

  expectError(() => flowManifest(singleNodeFlow(node)), cls, message);
});

// ---- Values ----------------------------------------------------------------------------------

test.each([
  [new Map(), "Map"],
  [new Uint8Array(), "Uint8Array"],
  [new Opaque(), "Opaque"],
  [() => 1, "function"],
  [undefined, "undefined"],
])("test_manifest_rejects_non_json_values_without_repr_fallback: %p", (value, name) => {
  const flow = compiled();
  code(flow, "step").env = { ...code(flow, "step").env, opaque: value };

  expectError(
    () => flowFingerprint(flow),
    TypeError,
    `flow.nodes.step.env.values[2]: unsupported non-JSON value of type ${name}`,
  );
});

/** `compiled()` with `value` under env key `key` of its step. */
function withEnv(key: string, value: unknown): CompiledFlow {
  const flow = compiled();
  code(flow, "step").env = { ...code(flow, "step").env, [key]: value };
  return flow;
}

test("a float is its IEEE-754 big-endian hex", () => {
  expect(envValue(withEnv("f", 1.5), "step", "f")).toEqual({ type: "float", value: "3ff8000000000000" });
});

test("an int and a float of the same number differ", () => {
  expect(envValue(withEnv("n", 2n), "step", "n")).toEqual({ type: "int", value: "2" });
  expect(flowFingerprint(withEnv("n", 2n))).not.toBe(flowFingerprint(withEnv("n", 2)));
});

test("infinity is a float and fingerprints", () => {
  const flow = withEnv("f", Infinity);

  expect(envValue(flow, "step", "f")).toEqual({ type: "float", value: "7ff0000000000000" });
  expect(envValue(withEnv("f", -Infinity), "step", "f")).toEqual({ type: "float", value: "fff0000000000000" });
  expect(flowFingerprint(flow)).toMatch(HEX);
});

test("NaN is refused with its path", () => {
  expectError(
    () => flowFingerprint(withEnv("f", NaN)),
    TypeError,
    "flow.nodes.step.env.values[2]: NaN is not a yian value",
  );
});

test("negative zero fingerprints as zero", () => {
  expect(envValue(withEnv("z", -0), "step", "z")).toEqual({ type: "float", value: "0000000000000000" });
  expect(flowFingerprint(withEnv("z", -0))).toBe(flowFingerprint(withEnv("z", 0)));
});

test("a cyclic env record is refused while a shared alias encodes twice", () => {
  const cyclic: Record<string, unknown> = {};
  cyclic.self = cyclic;
  expectError(
    () => flowFingerprint(withEnv("loop", cyclic)),
    RangeError,
    "flow.nodes.step.env.values[2].values[0]: cyclic value is not manifestable",
  );

  const shared = { a: 1n };
  const flow = compiled();
  code(flow, "step").env = { first: shared, second: shared };
  expect(envValue(flow, "step", "first")).toEqual(envValue(flow, "step", "second"));
  expect(flowFingerprint(flow)).toMatch(HEX);
});

test("a FileRef encodes as its mapping", () => {
  const flow = withEnv("file", new FileRef({ uri: "s3://bucket/key", mime: "text/plain" }));

  expect(envValue(flow, "step", "file")).toEqual({
    entries: [
      { key: { type: "str", value: "mime" }, value: { type: "str", value: "text/plain" } },
      { key: { type: "str", value: "name" }, value: { type: "null" } },
      { key: { type: "str", value: "uri" }, value: { type: "str", value: "s3://bucket/key" } },
    ],
    type: "mapping",
  });
});

// ---- Parameters and contracts ----------------------------------------------------------------

test.each([
  ["name", () => new Parameter("renamed", { contract: intContract() })],
  ["contract", () => new Parameter("x", { contract: new ValueContract(Type.scalar(ValueKind.NUMBER)) })],
  ["required", () => new Parameter("x", { contract: intContract(), required: true })],
  ["default", () => new Parameter("x", { contract: intContract(), hasDefault: true })],
  ["coerce-strings", () => new Parameter("x", { contract: intContract(), coerceStrings: true })],
])("test_flow_fingerprint_covers_each_parameter_field: %s", (_, changed) => {
  const original = fragment();
  const flow = fragment();
  code(flow, "body").params = [changed()];

  expect(flowFingerprint(flow)).not.toBe(flowFingerprint(original));
});

test("test_flow_fingerprint_covers_parameter_default_independently", () => {
  const original = fragment();
  code(original, "body").params = [new Parameter("x", { contract: intContract(), hasDefault: true, default: 1n })];
  const changed = fragment();
  code(changed, "body").params = [new Parameter("x", { contract: intContract(), hasDefault: true, default: 2n })];

  expect(flowFingerprint(changed)).not.toBe(flowFingerprint(original));
});

test("test_flow_fingerprint_preserves_parameter_order", () => {
  const original = fragment();
  const body = code(original, "body");
  body.params = [...body.params!, new Parameter("y")];
  const changed = fragment();
  code(changed, "body").params = [...body.params].reverse();

  expect(flowFingerprint(changed)).not.toBe(flowFingerprint(original));
});

test("test_parameter_origins_do_not_affect_program_or_expansion_identity", () => {
  const first = fragment();
  const second = fragment();
  code(first, "body").params = [new Parameter("x", { contract: intContract(), hasDefault: true, default: 1n })];
  code(second, "body").params = [
    new Parameter("x", {
      contract: new ValueContract(
        intType(),
        new ContractOrigins({
          default: new OriginRef("second/contract"),
          entries: [[["nested"], new OriginRef("second/contract/nested")]],
        }),
      ),
      hasDefault: true,
      default: 1n,
      origin: new OriginRef("second/parameter"),
      defaultOrigin: new OriginRef("second/default"),
    }),
  ];

  expect(flowManifest(first)).toEqual(flowManifest(second));
  expect(flowFingerprint(first)).toBe(flowFingerprint(second));
  expect(expansionDigest(first)).toBe(expansionDigest(second));
});

test("test_output_contract_origins_do_not_affect_program_or_expansion_identity", () => {
  const first = fragment();
  const second = fragment();
  code(second, "body").outputContract = new ValueContract(
    intType(),
    new ContractOrigins({
      default: new OriginRef("second/output"),
      entries: [[["nested"], new OriginRef("second/output/nested")]],
    }),
  );

  expect(flowManifest(first)).toEqual(flowManifest(second));
  expect(flowFingerprint(first)).toBe(flowFingerprint(second));
  expect(expansionDigest(first)).toBe(expansionDigest(second));
});

test("test_output_contract_cutover_preserves_manifest_shape_and_fingerprint", () => {
  const flow = fragment();
  const manifest = nodeManifest(flow, "body");

  expect("output_contract" in manifest).toBe(false);
  expect(manifest.output_type).toEqual({
    element: null,
    fields: null,
    kind: "int",
    nullable: false,
    required: null,
    tags: null,
  });
  expect(flowFingerprint(flow)).toBe(FRAGMENT_FINGERPRINT);
  expect(expansionDigest(flow)).toBe(flowFingerprint(flow));
});

test("test_numeric_bounds_are_part_of_type_identity_only_when_present", () => {
  const plain = fragment();
  expect("bounds" in (nodeManifest(plain, "body").output_type as JsonObject)).toBe(false);
  const fingerprints = new Set([flowFingerprint(plain)]);
  const digests = new Set([expansionDigest(plain)]);
  const rows: [bigint | null, bigint | null][] = [
    [1n, 12n],
    [2n, 12n],
    [1n, 13n],
    [1n, null],
    [null, 12n],
  ];
  const typed = (end: bigint | null): JsonValue =>
    end === null ? { type: "null" } : { type: "int", value: String(end) };
  for (const bounds of rows) {
    const bounded = fragmentWithOutput(new Type({ kind: ValueKind.INTEGER, bounds }));
    const manifest = flowManifest(bounded);
    // An open end is the null value.
    const outputType = ((manifest.nodes as JsonObject).body as JsonObject).output_type as JsonObject;
    expect(outputType.bounds).toEqual(bounds.map(typed));
    if (bounds[1] === null) {
      expect(canonicalJson(manifest)).toContain('"bounds":[{"type":"int","value":"1"},{"type":"null"}]');
    }
    fingerprints.add(flowFingerprint(bounded));
    digests.add(expansionDigest(bounded));
  }
  expect(fingerprints.size).toBe(6);
  expect(digests.size).toBe(6);
});

test("test_numeric_bounds_remain_in_nested_parameter_contract_identity", () => {
  const withScores = (text: string) => {
    const flow = fragment();
    code(flow, "body").params = [
      new Parameter("scores", {
        contract: new ValueContract(typeFor(text, new Map())),
        hasDefault: true,
        default: [0.5],
      }),
    ];
    flow.wiring.set("body", new Map());
    return flow;
  };
  const first = withScores("list(Optional(float(0..1)))");
  const second = withScores("list(Optional(float(0..2)))");

  const params = nodeManifest(first, "body").params as JsonObject[];
  const nested = ((params[0]!.contract as JsonObject).type as JsonObject).element as JsonObject;
  expect(nested.bounds).toEqual([
    { type: "int", value: "0" },
    { type: "int", value: "1" },
  ]);
  expect(nested.nullable).toBe(true);
  expect(flowFingerprint(first)).not.toBe(flowFingerprint(second));
});

test("test_output_contract_manifest_rejects_unknown_raw_and_subclass_bypasses", () => {
  const unknown = fragment();
  const body = code(unknown, "body");
  body.outputContract = Object.assign(Object.create(ValueContract.prototype), body.outputContract, { futureField: 1 });
  expectError(
    () => flowManifest(unknown),
    RangeError,
    'flow.nodes.body.outputContract: unknown noncanonical field(s): ["futureField"]',
  );

  const raw = fragment();
  code(raw, "body").outputContract = intType() as never;
  expectError(
    () => flowManifest(raw),
    TypeError,
    "flow.nodes.body.outputContract: expected ValueContract or PartialRecordContract, got Type",
  );

  const subclass = fragment();
  code(subclass, "body").outputContract = new ValueContractSubclass(intType());
  expectError(
    () => flowManifest(subclass),
    TypeError,
    "flow.nodes.body.outputContract: expected ValueContract or PartialRecordContract, got ValueContractSubclass",
  );
});

test("test_parameter_manifest_rejects_unknown_fields_and_raw_bypass", () => {
  const withUnknown = fragment();
  const body = code(withUnknown, "body");
  body.params = [Object.assign(Object.create(Parameter.prototype), body.params![0], { futureField: true })];
  expectError(
    () => flowManifest(withUnknown),
    RangeError,
    'flow.nodes.body.params[0]: unknown noncanonical field(s): ["futureField"]',
  );

  const raw = fragment();
  code(raw, "body").params = ["x"] as never;
  expectError(() => flowManifest(raw), TypeError, "flow.nodes.body.params[0]: expected Parameter, got string");

  const subclass = fragment();
  code(subclass, "body").params = [new ParameterSubclass("x")];
  expectError(
    () => flowManifest(subclass),
    TypeError,
    "flow.nodes.body.params[0]: expected Parameter, got ParameterSubclass",
  );
});

test("params that isn't an array is refused", () => {
  const flow = fragment();
  code(flow, "body").params = new Set([new Parameter("x")]) as never;

  expectError(() => flowManifest(flow), TypeError, "flow.nodes.body.params: expected array, got Set");
});

test.each([
  ["preAsserts", {}, "flow.nodes.body.preAsserts: expected array, got Object"],
  ["postAsserts", 5, "flow.nodes.body.postAsserts: expected array, got number"],
] as const)("%s that isn't an array is refused", (field, value, message) => {
  const flow = fragment();
  Object.assign(code(flow, "body"), { [field]: value });

  expectError(() => flowManifest(flow), TypeError, message);
});

// ---- Types -----------------------------------------------------------------------------------

test("test_union_free_manifest_has_no_union_key_and_keeps_its_fingerprint", () => {
  const outputType = nodeManifest(fragmentWithOutput(member("buy", { qty: ValueKind.INTEGER })), "body")
    .output_type as JsonObject;

  expect("union" in outputType).toBe(false);
  for (const field of outputType.fields as JsonObject[]) expect("union" in (field.type as JsonObject)).toBe(false);
  expect(flowFingerprint(fragment())).toBe(FRAGMENT_FINGERPRINT);
});

test("test_union_manifest_orders_members_canonically", () => {
  const buy = member("buy", { qty: ValueKind.INTEGER });
  const sell = member("sell", { price: ValueKind.NUMBER });
  const first = fragmentWithOutput(union(buy, sell));
  const second = fragmentWithOutput(union(sell, buy));

  const members = (nodeManifest(second, "body").output_type as JsonObject).union as JsonObject[];
  const tags = members.map((item) => ((item.fields as JsonObject[])[0]!.type as JsonObject).tags);
  expect(tags).toEqual([["buy"], ["sell"]]);
  expect(flowFingerprint(first)).toBe(flowFingerprint(second));
  expect(expansionDigest(first)).toBe(expansionDigest(second));
});

test("test_deep_untagged_union_manifests_quickly", () => {
  const record = (fields: Record<string, Type>) =>
    new Type({
      kind: ValueKind.OBJECT,
      fields: new Map(Object.entries(fields)),
      required: new Set(Object.keys(fields)),
    });
  let typ = union(record({ a: Type.scalar(ValueKind.STRING) }), record({ b: intType() }));
  for (let level = 0; level < 30; level++) {
    typ = union(record({ inner: typ }), record({ [`leaf${level}`]: intType() }));
  }

  // Ordering each level's members labels that level's subtree, so the work is quadratic in
  // depth; an exponential relabelling of nested members would not finish within the timeout.
  expect(flowFingerprint(fragmentWithOutput(typ))).toMatch(HEX);
});

test.each([
  ["kind", intType(), Type.scalar(ValueKind.NUMBER)],
  [
    "fields",
    new Type({ kind: ValueKind.OBJECT, fields: new Map([["a", intType()]]) }),
    new Type({ kind: ValueKind.OBJECT, fields: new Map([["b", intType()]]) }),
  ],
  [
    "required",
    new Type({ kind: ValueKind.OBJECT, fields: new Map([["a", intType()]]), required: new Set() }),
    new Type({ kind: ValueKind.OBJECT, fields: new Map([["a", intType()]]), required: new Set(["a"]) }),
  ],
  [
    "tags",
    new Type({ kind: ValueKind.STRING, tags: new Set(["first"]) }),
    new Type({ kind: ValueKind.STRING, tags: new Set(["second"]) }),
  ],
  [
    "element",
    new Type({ kind: ValueKind.LIST_ANY, element: intType() }),
    new Type({ kind: ValueKind.LIST_ANY, element: Type.scalar(ValueKind.STRING) }),
  ],
  [
    "nullable",
    new Type({ kind: ValueKind.STRING, nullable: false }),
    new Type({ kind: ValueKind.STRING, nullable: true }),
  ],
  [
    "variants-member-field",
    union(member("buy", { qty: ValueKind.INTEGER }), member("sell")),
    union(member("buy", { qty: ValueKind.NUMBER }), member("sell")),
  ],
  ["variants-tag", union(member("buy"), member("sell")), union(member("buy"), member("short"))],
  [
    "variants-member-added",
    union(member("buy"), member("sell")),
    union(member("buy"), member("sell"), member("hold")),
  ],
])("test_flow_fingerprint_covers_each_structural_type_field: %s", (_, firstType, secondType) => {
  expect(flowFingerprint(fragmentWithOutput(firstType))).not.toBe(flowFingerprint(fragmentWithOutput(secondType)));
});

test("test_flow_fingerprint_preserves_structural_field_order_used_by_agent_schema", () => {
  const first = new Type({
    kind: ValueKind.OBJECT,
    fields: new Map([
      ["first", intType()],
      ["second", Type.scalar(ValueKind.STRING)],
    ]),
  });
  const second = new Type({
    kind: ValueKind.OBJECT,
    fields: new Map([
      ["second", Type.scalar(ValueKind.STRING)],
      ["first", intType()],
    ]),
  });

  expect(flowFingerprint(fragmentWithOutput(first))).not.toBe(flowFingerprint(fragmentWithOutput(second)));
});

test("test_start_manifest_uses_canonical_parameters", () => {
  const parameter = new Parameter("x", { contract: intContract(), required: true, coerceStrings: true });
  const start = new StartNode("node", { params: [parameter] });

  const params = nodeManifest(singleNodeFlow(start), "node").params as JsonObject[];

  expect(params[0]!.name).toBe("x");
  expect(((params[0]!.contract as JsonObject).type as JsonObject).kind).toBe("int");
});

test("test_flow_fingerprint_preserves_start_parameter_order", () => {
  const firstParameter = new Parameter("first");
  const secondParameter = new Parameter("second");
  const first = new StartNode("node", { params: [firstParameter, secondParameter] });
  const second = new StartNode("node", { params: [secondParameter, firstParameter] });

  expect(flowFingerprint(singleNodeFlow(first))).not.toBe(flowFingerprint(singleNodeFlow(second)));
});

// ---- END -------------------------------------------------------------------------------------

test.each([
  [
    "end-record-names",
    () => EndNode.record("node", { outputNames: ["first"] }),
    () => EndNode.record("node", { outputNames: ["second"] }),
  ],
  ["end-list-n", () => EndNode.list("node", { n: 1 }), () => EndNode.list("node", { n: 2 })],
  ["end-mode", () => EndNode.record("node", { outputNames: [] }), () => EndNode.list("node", { n: 0 })],
])("test_flow_fingerprint_covers_each_end_field: %s", (_, makeFirst, makeSecond) => {
  const first = makeFirst();
  const second = makeSecond();
  first.params = second.params = null;

  expect(flowFingerprint(singleNodeFlow(first))).not.toBe(flowFingerprint(singleNodeFlow(second)));
});

test("test_value_end_manifest_keeps_the_record_encoding", () => {
  const manifest = nodeManifest(singleNodeFlow(EndNode.value("t", { outputName: "result" })), "t");

  expect(manifest.mode).toBe("record");
  expect(manifest.n).toBeNull();
  expect(manifest.names).toEqual([{ type: "str", value: "result" }]);
  expect("keyed" in manifest).toBe(false);
});

test.each([
  ["zero", []],
  ["one", ["result"]],
])("test_zero_and_one_name_record_end_manifests_are_keyed: %s", (_, names) => {
  const valueEnd = EndNode.value("t", { outputName: names[0] ?? null });
  const recordEnd = EndNode.record("t", { outputNames: names });

  const valueManifest = nodeManifest(singleNodeFlow(valueEnd), "t");
  const recordManifest = nodeManifest(singleNodeFlow(recordEnd), "t");
  expect(recordManifest.keyed).toBe(true);
  expect(recordManifest).toEqual({ ...valueManifest, keyed: true });
});

test("test_two_name_record_end_manifest_is_not_keyed", () => {
  const end = EndNode.record("t", { outputNames: ["a", "b"] });

  expect("keyed" in nodeManifest(singleNodeFlow(end), "t")).toBe(false);
});

test("a list END writes null names and its count", () => {
  const manifest = nodeManifest(singleNodeFlow(EndNode.list("t", { n: 2 })), "t");

  expect(manifest.mode).toBe("list");
  expect(manifest.names).toBeNull();
  expect(manifest.n).toBe(2);
});

test("a forged END count is refused with its path", () => {
  const end = EndNode.list("t", { n: 1 });
  end.n = 1.5;

  expectError(() => flowManifest(singleNodeFlow(end)), TypeError, "flow.nodes.t.n: expected a safe integer, got 1.5");
});

// ---- No redaction ----------------------------------------------------------------------------

test("test_flow_fingerprint_changes_with_wiring_types_asserts_and_env", () => {
  const original = compiled();
  const literal = (value: bigint) => {
    const flow = compiled();
    flow.wiring.set("step", new Map([["x", new Binding(value)]]));
    return flow;
  };

  const outputType = compiled();
  code(outputType, "step").outputContract = new ValueContract(Type.scalar(ValueKind.NUMBER));

  const inputType = compiled();
  inputType.nodes.get(START_ID)!.params = [
    new Parameter("x", { contract: new ValueContract(Type.scalar(ValueKind.NUMBER)), hasDefault: true, default: 1 }),
  ];

  const asserts = compiled();
  code(asserts, "step").preAsserts = [...code(asserts, "step").preAsserts].reverse();

  const env = compiled();
  code(env, "step").env = { ...code(env, "step").env, alpha: 99n };

  const title = compiled();
  code(title, "step").title = "A different title";

  const baseline = flowFingerprint(original);
  expect(flowFingerprint(literal(7n))).not.toBe(flowFingerprint(literal(8n)));
  expect(flowFingerprint(outputType)).not.toBe(baseline);
  expect(flowFingerprint(inputType)).not.toBe(baseline);
  expect(flowFingerprint(asserts)).not.toBe(baseline);
  expect(flowFingerprint(env)).not.toBe(baseline);
  expect(flowFingerprint(title)).toBe(baseline);
});

test("test_manifest_has_no_redaction_markers", () => {
  const str = (name: string, fallback?: string) =>
    new Parameter(name, {
      contract: strContract(),
      ...(fallback === undefined ? {} : { hasDefault: true, default: fallback }),
    });
  const step = new CodeNode("step", { code: 'return "ok"' });
  const wiring: [string, string][] = [
    ["signature", "q -> a"],
    ["url", "https://alice:pw@db/x?key=apples&user=alice&page_token=1"],
    ["connection_string", "jdbc:sqlserver://h;user=alice;databaseName=app"],
    ["dsn", "host=db user=alice"],
    ["oracle", "jdbc:oracle:thin:alice/pw@h:1521/s"],
    ["authorization", "Bearer ${input.t}"],
    ["credential", "${input.token}"],
    ["api_token", "shared-literal"],
  ];
  step.params = wiring.map(([name]) => new Parameter(name));
  step.env = {
    AUTH_TOKEN: "a",
    region_url: "https://x?key=a",
    sig: "first-value",
    webhook_url: "https://hooks.example/services/T0/B0/first-value",
  };
  const flow = CompiledFlow.fromParts(
    new Map<string, Node>([
      [
        START_ID,
        new StartNode(START_ID, {
          params: [str("t"), str("token"), str("signature", "q -> a"), str("connection_string", "Server=a;User=alice")],
        }),
      ],
      ["step", step],
      [END_ID, EndNode.record(END_ID, { outputNames: ["step", "api_token"] })],
    ]),
    [new Edge(`step->${END_ID}`, "step", END_ID, { inputGroup: "step" })],
    {
      wiring: new Map<string, Map<string, unknown>>([
        [START_ID, new Map()],
        ["step", new Map(wiring)],
        [
          END_ID,
          new Map([
            ["step", "${step.output}"],
            ["api_token", "shared-literal"],
          ]),
        ],
      ]),
      outputs: [new FlowOutput("step", "${step.output}"), new FlowOutput("api_token", "shared-literal")],
      flowId: "no-markers",
    },
  );
  const manifestText = JSON.stringify(flowManifest(flow));

  expect(manifestText).not.toContain("__redacted__");
  expect(manifestText).not.toContain("__redacted_binding__");
  expect(manifestText).not.toContain('"redacted"');
});

test("test_credential_named_parameter_without_default_has_the_null_default", () => {
  const flow = oneStepFlow({
    inputs: [new Parameter("token", { contract: strContract() })],
    wiring: [["token", "${input.token}"]],
  });
  const params = Object.values(flowManifest(flow).nodes as JsonObject)
    .flatMap((node) => ((node as JsonObject).params as JsonObject[] | null) ?? [])
    .filter((param) => param.name === "token");

  expect(params.length).toBeGreaterThan(0);
  for (const param of params) expect(param.default).toEqual({ type: "null" });
});

/** Assert two flows differing in one literal differ in identity and show it. */
function expectLiteralIsIdentity(first: Flow, second: Flow, firstValue: string, secondValue: string): void {
  expect(flowFingerprint(first)).not.toBe(flowFingerprint(second));
  expect(JSON.stringify(flowManifest(first))).toContain(firstValue);
  expect(JSON.stringify(flowManifest(second))).toContain(secondValue);
}

// Three of Python's `_FORMERLY_REDACTED_LITERALS` rows: a credential-like name, a URI's userinfo
// and a JDBC string.
const FORMERLY_REDACTED: [string, string, string, string][] = [
  ["signature", "signature", "q -> a", "q -> s"],
  ["uri-userinfo", "url", "https://alice:pw@db/x", "https://bob:pw@db/x"],
  ["jdbc-oracle", "connection_string", "jdbc:oracle:thin:alice/pw@h:1521/s", "jdbc:oracle:thin:bob/pw@h:1521/s"],
];

test.each(FORMERLY_REDACTED)("test_node_input_literals_are_identity: %s", (_, name, firstValue, secondValue) => {
  expectLiteralIsIdentity(
    oneStepFlow({ wiring: [[name, firstValue]] }),
    oneStepFlow({ wiring: [[name, secondValue]] }),
    firstValue,
    secondValue,
  );
});

test.each(FORMERLY_REDACTED)(
  "test_declaration_default_literals_are_identity: %s",
  (_, name, firstValue, secondValue) => {
    const withDefault = (value: string) =>
      oneStepFlow({ inputs: [new Parameter(name, { contract: strContract(), hasDefault: true, default: value })] });
    expectLiteralIsIdentity(withDefault(firstValue), withDefault(secondValue), firstValue, secondValue);
  },
);

test.each([
  ["token", "token", "first-value", "second-value"],
  ["AUTH_TOKEN", "AUTH_TOKEN", "a", "b"],
  ["region_url", "region_url", "https://x?key=a", "https://x?key=b"],
])("test_env_literals_are_identity: node-%s", (_, name, firstValue, secondValue) => {
  expectLiteralIsIdentity(
    oneStepFlow({ env: { [name]: firstValue } }),
    oneStepFlow({ env: { [name]: secondValue } }),
    firstValue,
    secondValue,
  );
});

test.each([
  ["authorization-scheme", "authorization", "Bearer ${input.t}", "Basic ${input.t}"],
  [
    "connection-uri-host",
    "connection_string",
    "postgres://${input.a}:x@db.example/app",
    "postgres://${input.a}:x@other.example/app",
  ],
  [
    "uri-query-reference",
    "endpoint",
    "https://api.example/v1?token=${input.a}&region=us",
    "https://api.example/v1?token=${input.b}&region=us",
  ],
])("test_templates_under_any_name_keep_their_source_text: %s", (_, name, firstSource, secondSource) => {
  const withTemplate = (source: string) =>
    oneStepFlow({
      inputs: ["a", "b", "t"].map((input) => new Parameter(input, { contract: strContract() })),
      wiring: [[name, source]],
    });
  const first = withTemplate(firstSource);
  const second = withTemplate(secondSource);

  expect(flowFingerprint(first)).not.toBe(flowFingerprint(second));
  for (const [flow, source] of [
    [first, firstSource],
    [second, secondSource],
  ] as const) {
    const wiring = (flowManifest(flow).wiring as JsonObject).step!;
    expect(typedMappingValue(wiring, name)).toEqual({ type: "str", value: source });
    expect(JSON.stringify(flowManifest(flow))).not.toContain("__redacted");
  }
});

test.each([["key"], ["GITHUB_TOKEN"], ["X-Amz-Signature"]])(
  "test_env_values_under_credential_like_names_are_identity: %s",
  (configKey) => {
    const first = withEnv(configKey, "first-semantic-value");
    const second = withEnv(configKey, "second-semantic-value");

    expect(flowFingerprint(first)).not.toBe(flowFingerprint(second));
    expect(JSON.stringify(flowManifest(first))).toContain("first-semantic-value");
  },
);

// ---- Edges, boundary ids and flow outputs ----------------------------------------------------

/** Replace `flow.edges[index]` with a rebuilt edge carrying `change` (edges are frozen). */
function replaceEdge(flow: Flow, index: number, change: Partial<Edge>): void {
  const fields = { ...flow.edges[index]!, ...change };
  flow.edges[index] = new Edge(fields.id, fields.from, fields.to, fields);
}

/** `flow` rebuilt with `ids` as its boundary ids, which are readonly. */
function withBoundary(flow: Flow, ids: { startId?: string; endId?: string }): Flow {
  return new Flow(flow.nodes, flow.edges, {
    wiring: flow.wiring,
    startId: flow.startId,
    endId: flow.endId,
    ...ids,
  });
}

// The manifest of an `int` scalar type.
const INT_TYPE_MANIFEST = { element: null, fields: null, kind: "int", nullable: false, required: null, tags: null };

test("test_flow_fingerprint_preserves_semantic_edge_order", () => {
  const original = compiled();
  const base = compiled();
  const reordered = CompiledFlow.fromParts(base.nodes, [...base.edges].reverse(), {
    outputs: base.outputs,
    wiring: base.wiring,
    flowLlmConfig: base.flowLlmConfig,
    flowId: base.flowId,
  });

  expect(flowFingerprint(original)).not.toBe(flowFingerprint(reordered));
});

test.each([
  ["id", { id: "changed-edge" }],
  ["from", { from: "changed-source" }],
  ["to", { to: "changed-target" }],
  ["source-handle", { sourceHandle: "branch" }],
  ["input-group", { inputGroup: "value" }],
  ["optional", { optional: true }],
  ["ordering", { ordering: false }],
] as const)("test_flow_fingerprint_covers_each_edge_field: %s", (_, change) => {
  const changed = fragment();
  // A dangling endpoint is not the manifest's to validate.
  replaceEdge(changed, 0, change);

  expect(flowFingerprint(changed)).not.toBe(flowFingerprint(fragment()));
});

test.each([["startId"], ["endId"]] as const)("test_flow_fingerprint_covers_flow_boundary_ids: %s", (field) => {
  const changed = withBoundary(fragment(), { [field]: "body" });

  expect(flowFingerprint(changed)).not.toBe(flowFingerprint(fragment()));
});

test.each([
  ["name", [new FlowOutput("first", 1n)], [new FlowOutput("second", 1n)]],
  ["from", [new FlowOutput("value", 1n)], [new FlowOutput("value", 2n)]],
  [
    "order",
    [new FlowOutput("first", 1n), new FlowOutput("second", 2n)],
    [new FlowOutput("second", 2n), new FlowOutput("first", 1n)],
  ],
])("test_flow_fingerprint_covers_flow_output_fields_and_order: %s", (_, firstOutputs, secondOutputs) => {
  const first = compiled();
  const second = compiled();
  first.outputs = firstOutputs;
  second.outputs = secondOutputs;

  expect(flowFingerprint(first)).not.toBe(flowFingerprint(second));
});

test("test_node_inputs_and_flow_outputs_keep_literals_alike", () => {
  const flow = oneStepFlow({ wiring: [["api_token", "shared-literal"]] });
  flow.outputs = [new FlowOutput("step", "${step.output}"), new FlowOutput("api_token", "shared-literal")];
  const manifest = flowManifest(flow);

  expect(typedMappingValue((manifest.wiring as JsonObject).step!, "api_token")).toEqual({
    type: "str",
    value: "shared-literal",
  });
  expect(manifest.outputs).toContainEqual({ from: { type: "str", value: "shared-literal" }, name: "api_token" });
});

test("a typed wiring binding adds binding_types and changes both digests", () => {
  const wired = (contract: ValueContract | null) =>
    oneStepFlow({
      inputs: [new Parameter("x", { contract: intContract() })],
      wiring: [["x", new Binding("${input.x}", { contract })]],
    });
  const untyped = wired(null);
  const typed = wired(intContract());

  expect("binding_types" in flowManifest(untyped)).toBe(false);
  expect(flowManifest(typed).binding_types).toEqual({ step: { x: INT_TYPE_MANIFEST } });
  expect(flowFingerprint(typed)).not.toBe(flowFingerprint(untyped));
  expect(expansionDigest(typed)).not.toBe(expansionDigest(untyped));
});

test("a typed flow output adds its type", () => {
  const flow = compiled();
  flow.outputs = [
    new FlowOutput("plain", "${step.output}"),
    new FlowOutput("typed", new Binding("${step.output}", { contract: intContract() })),
  ];
  const source = { type: "str", value: "${step.output}" };

  expect(flowManifest(flow).outputs).toEqual([
    { from: source, name: "plain" },
    { from: source, name: "typed", type: INT_TYPE_MANIFEST },
  ]);
});

test("a plain Flow has no outputs, flow id, memories or LLM layer", () => {
  expect(Object.keys(flowManifest(fragment())).sort()).toEqual(["edges", "end_id", "nodes", "start_id", "wiring"]);
});

// ---- Compiled flow fields --------------------------------------------------------------------

/** `compiled()` with `layer` as its flow `llm_config:` layer. */
function withLlmConfig(layer: CompiledFlow["flowLlmConfig"]): CompiledFlow {
  const flow = compiled();
  flow.flowLlmConfig = layer;
  return flow;
}

test("flow llm_config literal values are identity and key order isn't", () => {
  const base = withLlmConfig({ openai: { model: "a", temperature: 0.2 }, provider: "openai" });
  const reordered = withLlmConfig({ provider: "openai", openai: { temperature: 0.2, model: "a" } });
  const changed = withLlmConfig({ openai: { model: "a", temperature: 0.7 }, provider: "openai" });

  expect(flowFingerprint(reordered)).toBe(flowFingerprint(base));
  expect(flowFingerprint(changed)).not.toBe(flowFingerprint(base));
  expect(flowFingerprint(base)).not.toBe(flowFingerprint(compiled()));
  expect(flowManifest(compiled()).flow_llm_config).toEqual({ entries: [], type: "mapping" });
});

test("a ValueTemplate flow layer encodes as a value template", () => {
  const flow = withLlmConfig(new ValueTemplate({ model: new Binding("${input.model}") }));

  expect(flowManifest(flow).flow_llm_config).toEqual({
    tree: {
      entries: [
        {
          key: { type: "str", value: "model" },
          value: { contract: null, source: { type: "str", value: "${input.model}" }, type: "binding" },
        },
      ],
      type: "mapping",
    },
    type: "value_template",
  });
});

test("a Binding flow layer encodes as a binding", () => {
  const flow = withLlmConfig(new Binding("${input.llm}"));

  expect(flowManifest(flow).flow_llm_config).toEqual({
    contract: null,
    source: { type: "str", value: "${input.llm}" },
    type: "binding",
  });
});

test("the LLM value types and the adjacency index keep the fingerprint", () => {
  const changed = withLlmConfig({ model: "a" });
  changed.flowLlmValueTypes = [[["model"], Type.scalar(ValueKind.STRING)]];
  changed.incomingEdges = new Map();
  changed.outgoingEdges = new Map();

  expect(flowFingerprint(changed)).toBe(flowFingerprint(withLlmConfig({ model: "a" })));
});

test("the flow id is identity but not expansion topology", () => {
  const named = compiled();
  const unnamed = compiled();
  Object.assign(unnamed, { flowId: null });

  expect(flowManifest(named).flow_id).toBe("stable");
  expect("flow_id" in flowManifest(unnamed)).toBe(false);
  expect(flowFingerprint(named)).not.toBe(flowFingerprint(unnamed));
  expect(expansionDigest(named)).toBe(expansionDigest(unnamed));
});

test("a CompiledFlow's root outputs and LLM layer aren't expansion topology", () => {
  const changed = compiled();
  changed.outputs = [new FlowOutput("other", 1n)];
  changed.flowLlmConfig = { model: "a" };

  expect(flowFingerprint(changed)).not.toBe(flowFingerprint(compiled()));
  expect(expansionDigest(changed)).toBe(expansionDigest(compiled()));
});

test.each([
  ["a number", 5],
  ["an empty string", ""],
])("a forged flow id that is %s is refused in both modes", (_, flowId) => {
  const flow = compiled();
  Object.assign(flow, { flowId });

  expectError(() => flowManifest(flow), RangeError, "flow.flowId: expected a non-empty exact string or null");
  expectError(
    () => expansionDigest(flow),
    RangeError,
    "expansion.flowId: expected a non-empty exact string or null",
  );
});

test("flow memories are identity only when present and never expansion topology", () => {
  const plain = compiled();
  const withMemory = compiled();
  withMemory.memories = [new MemoryDecl("reflective", "reflective", { max_items: 4n }, new DeclSite([], ["root"]))];

  expect("memories" in flowManifest(plain)).toBe(false);
  expect(flowManifest(withMemory).memories).toEqual([
    {
      config: {
        entries: [{ key: { type: "str", value: "max_items" }, value: { type: "int", value: "4" } }],
        type: "mapping",
      },
      name: "reflective",
      site: { node: null, owner: ["root"], scope: [] },
      type: "reflective",
    },
  ]);
  expect(flowFingerprint(withMemory)).not.toBe(flowFingerprint(plain));
  expect(expansionDigest(withMemory)).toBe(expansionDigest(plain));
});

test("forged flow memories are refused in both modes", () => {
  const flow = compiled();
  Object.assign(flow, { memories: ["notes"] });

  expectError(() => flowManifest(flow), TypeError, "flow.memories[0]: expected exact MemoryDecl, got string");
  expectError(() => expansionDigest(flow), TypeError, "expansion.memories[0]: expected exact MemoryDecl, got string");
});

// ---- Flow field policy -----------------------------------------------------------------------

class FlowSubclass extends Flow {}

test("an unknown flow field is refused", () => {
  const flow = compiled();
  Object.assign(flow, { futureField: 1 });

  expectError(() => flowManifest(flow), RangeError, 'flow: unknown noncanonical field(s): ["futureField"]');
});

test("a Flow subclass and a non-Flow are refused", () => {
  const base = fragment();
  const subclass = new FlowSubclass(base.nodes, base.edges, {
    wiring: base.wiring,
    startId: "entry",
    endId: "terminal",
  });

  expectError(() => flowManifest(subclass), TypeError, "flow: unsupported Flow subclass FlowSubclass");
  expectError(() => flowManifest({} as Flow), TypeError, "flow: expected Flow, got Object");
});

test("a wiring value that isn't a Binding is refused", () => {
  const flow = fragment();
  (flow.wiring as Map<string, unknown>).set("body", new Map([["x", 1n]]));

  expectError(() => flowManifest(flow), TypeError, "flow.wiring.body.values[0]: expected Binding, got bigint");
});

test("a node wiring that isn't a Map is refused", () => {
  const flow = fragment();
  (flow.wiring as Map<string, unknown>).set("body", { x: new Binding(1n) });

  expectError(() => flowManifest(flow), TypeError, "flow.wiring.body: expected Map, got Object");
});

test.each([
  [
    "nodes that aren't a Map",
    (flow: CompiledFlow) => Object.assign(flow, { nodes: {} }),
    TypeError,
    "flow.nodes: expected Map, got Object",
  ],
  [
    "a non-string node key",
    (flow: CompiledFlow) => (flow.nodes as Map<unknown, unknown>).set(1, flow.nodes.get("step")),
    TypeError,
    "flow.nodes: keys must be strings, got number",
  ],
  [
    "wiring that isn't a Map",
    (flow: CompiledFlow) => Object.assign(flow, { wiring: {} }),
    TypeError,
    "flow.wiring: expected Map, got Object",
  ],
  [
    "a non-string wiring key",
    (flow: CompiledFlow) => (flow.wiring as Map<unknown, unknown>).set(1, new Map()),
    TypeError,
    "flow.wiring: keys must be strings, got number",
  ],
  [
    "a non-string parameter key",
    (flow: CompiledFlow) => (flow.wiring as Map<string, unknown>).set("step", new Map([[1, new Binding(1n)]])),
    TypeError,
    "flow.wiring.step: mapping keys must be strings, got number",
  ],
  [
    "an edge that isn't an Edge",
    (flow: CompiledFlow) => flow.edges.push({} as Edge),
    TypeError,
    "flow.edges[2]: expected Edge, got Object",
  ],
  [
    "a forged Edge",
    (flow: CompiledFlow) =>
      (flow.edges[0] = Object.assign(Object.create(Edge.prototype), flow.edges[0], { futureField: 1 })),
    RangeError,
    'flow.edges[0]: unknown noncanonical field(s): ["futureField"]',
  ],
  [
    "edges that aren't an array",
    (flow: CompiledFlow) => Object.assign(flow, { edges: {} }),
    TypeError,
    "flow.edges: expected array, got Object",
  ],
  [
    "a flowLlmConfig that isn't a record",
    (flow: CompiledFlow) => Object.assign(flow, { flowLlmConfig: 5 }),
    TypeError,
    "flow.flowLlmConfig: expected a record, got number",
  ],
  [
    "memories that aren't an array",
    (flow: CompiledFlow) => Object.assign(flow, { memories: "x" }),
    TypeError,
    "flow.memories: expected an array of MemoryDecl values",
  ],
  [
    "an output that isn't a FlowOutput",
    (flow: CompiledFlow) => Object.assign(flow, { outputs: [{}] }),
    TypeError,
    "flow.outputs[0]: expected FlowOutput, got Object",
  ],
  [
    "outputs that aren't an array",
    (flow: CompiledFlow) => Object.assign(flow, { outputs: 5 }),
    TypeError,
    "flow.outputs: expected array, got number",
  ],
] as const)("a flow carrier with %s is refused", (_, mutate, cls, message) => {
  const flow = compiled();
  mutate(flow);

  expectError(() => flowManifest(flow), cls, message);
});

// ---- CASE, the ask and sleep built-ins, CALL, MAP and LOOP -----------------------------------

/** A LOOP node `node` over a baked `fragment()` body, with `options` on top. */
function loopNode(options: Partial<ConstructorParameters<typeof LoopNode>[1]> = {}): LoopNode {
  return new LoopNode("node", { flowId: "child", child: fragment(), ...options });
}

/** A flow of the single node `node`, wired with `wiring`. */
function wiredNodeFlow(node: Node, wiring: [string, unknown][]): Flow {
  return new Flow(new Map([[node.id, node]]), [], {
    wiring: new Map([[node.id, new Map(wiring)]]),
    startId: node.id,
    endId: node.id,
  });
}

test.each([
  [
    "code-source",
    () => new CodeNode("node", { code: 'return "first"' }),
    () => new CodeNode("node", { code: 'return "second"' }),
  ],
  ["agent-prompt", () => new AgentNode("node", { prompt: "first" }), () => new AgentNode("node", { prompt: "second" })],
  [
    "agent-tools",
    () => new AgentNode("node", { prompt: "p", tools: ["first"] }),
    () => new AgentNode("node", { prompt: "p", tools: ["second"] }),
  ],
  [
    "agent-retries",
    () => new AgentNode("node", { prompt: "p", retries: 1 }),
    () => new AgentNode("node", { prompt: "p", retries: 2 }),
  ],
  [
    "agent-own-llm-config",
    () => new AgentNode("node", { prompt: "p", llmConfig: { provider: "openai" } }),
    () => new AgentNode("node", { prompt: "p", llmConfig: { provider: "anthropic" } }),
  ],
  [
    "agent-llm-inherit",
    () => new AgentNode("node", { prompt: "p", llmInherit: true }),
    () => new AgentNode("node", { prompt: "p", llmInherit: false }),
  ],
  ["tool-id", () => new ToolNode("node", { toolId: "first" }), () => new ToolNode("node", { toolId: "second" })],
  [
    "case-handle",
    () => new CaseNode("node", [new Case("first", { when: "x" })]),
    () => new CaseNode("node", [new Case("second", { when: "x" })]),
  ],
  [
    "case-when",
    () => new CaseNode("node", [new Case("branch", { when: "first" })]),
    () => new CaseNode("node", [new Case("branch", { when: "second" })]),
  ],
  [
    "case-order",
    () => new CaseNode("node", [new Case("first", { when: "x" }), new Case("second", { when: "y" })]),
    () => new CaseNode("node", [new Case("second", { when: "y" }), new Case("first", { when: "x" })]),
  ],
  [
    "ask-answer-schema",
    () => new ToolNode("node", { toolId: "ask", answerSchema: [{ name: "first" }] }),
    () => new ToolNode("node", { toolId: "ask", answerSchema: [{ name: "second" }] }),
  ],
  [
    "call-flow-id",
    () => new CallNode("node", { flowId: "first", child: fragment() }),
    () => new CallNode("node", { flowId: "second", child: fragment() }),
  ],
  [
    "call-flow-version",
    () => new CallNode("node", { flowId: "child", flowVersion: 1n, child: fragment() }),
    () => new CallNode("node", { flowId: "child", flowVersion: 2n, child: fragment() }),
  ],
  [
    "call-child",
    () => new CallNode("node", { flowId: "child", child: fragment() }),
    () => new CallNode("node", { flowId: "child", child: replaceBody(fragment()) }),
  ],
  [
    "map-flow-id",
    () => new MapNode("node", { flowId: "first", child: fragment() }),
    () => new MapNode("node", { flowId: "second", child: fragment() }),
  ],
  [
    "map-flow-version",
    () => new MapNode("node", { flowId: "child", flowVersion: 1n, child: fragment() }),
    () => new MapNode("node", { flowId: "child", flowVersion: 2n, child: fragment() }),
  ],
  [
    "map-child",
    () => new MapNode("node", { flowId: "child", child: fragment() }),
    () => new MapNode("node", { flowId: "child", child: replaceBody(fragment()) }),
  ],
  ["loop-flow-id", () => loopNode({ flowId: "first" }), () => loopNode({ flowId: "second" })],
  ["loop-flow-version", () => loopNode({ flowVersion: "1" }), () => loopNode({ flowVersion: "2" })],
  ["loop-child", () => loopNode(), () => loopNode({ child: replaceBody(fragment()) })],
  ["loop-predicate-kind", () => loopNode({ predicateKind: "while" }), () => loopNode({ predicateKind: "until" })],
  ["loop-predicate", () => loopNode({ predicate: "first" }), () => loopNode({ predicate: "second" })],
  ["loop-times", () => loopNode({ times: 1 }), () => loopNode({ times: 2 })],
  ["loop-max-iters", () => loopNode({ maxIters: 1 }), () => loopNode({ maxIters: 2 })],
  ["loop-max-iters-none", () => loopNode({ maxIters: null }), () => loopNode({ maxIters: 5 })],
  ["loop-iteration", () => loopNode({ iteration: 1 }), () => loopNode({ iteration: 2 })],
  ["loop-returned-fields", () => loopNode({ returnedFields: null }), () => loopNode({ returnedFields: ["x"] })],
] as const)("test_flow_fingerprint_covers_each_included_kind_field: %s", (_, makeFirst, makeSecond) => {
  expect(flowFingerprint(singleNodeFlow(makeFirst()))).not.toBe(flowFingerprint(singleNodeFlow(makeSecond())));
});

test("test_flow_manifest_covers_remaining_kind_execution_fields", () => {
  const record = EndNode.record("node", { outputNames: ["result"] });
  const list = EndNode.list("node", { n: 1 });

  expect(flowFingerprint(singleNodeFlow(record))).not.toBe(flowFingerprint(singleNodeFlow(list)));
});

test("test_flow_manifest_includes_human_input_title_but_ignores_inert_map_parallel", () => {
  // An ask sends its title (its `node_name:`) to the host, as HUMAN_INPUT did.
  const firstAsk = new ToolNode("node", { toolId: "ask", title: "First title" });
  const secondAsk = new ToolNode("node", { toolId: "ask", title: "Second title" });
  const serialMap = new MapNode("node", { flowId: "child", child: fragment(), parallel: false });
  const parallelMap = new MapNode("node", { flowId: "child", child: fragment(), parallel: true });

  expect(flowFingerprint(singleNodeFlow(firstAsk))).not.toBe(flowFingerprint(singleNodeFlow(secondAsk)));
  expect(flowFingerprint(singleNodeFlow(serialMap))).toBe(flowFingerprint(singleNodeFlow(parallelMap)));
});

test("an ask's prompt and questions are identity through its wiring", () => {
  const ask = (key: string, value: unknown) => wiredNodeFlow(new ToolNode("node", { toolId: "ask" }), [[key, value]]);

  expect(flowFingerprint(ask("prompt", new Binding("first")))).not.toBe(
    flowFingerprint(ask("prompt", new Binding("second"))),
  );
  expect(flowFingerprint(ask("questions", new ValueTemplate([{ question: "first" }])))).not.toBe(
    flowFingerprint(ask("questions", new ValueTemplate([{ question: "second" }]))),
  );
});

test("an ask TOOL node writes its answer schema as a value", () => {
  const node = new ToolNode("node", { toolId: "ask", answerSchema: [{ name: "answer", type: "str" }], title: "T" });

  expect(nodeManifest(singleNodeFlow(node), "node")).toMatchObject({
    tool_id: "ask",
    answer_schema: { type: "list" },
    title: { type: "str", value: "T" },
  });
  // Another tool's title is display metadata.
  const plain = new ToolNode("node", { toolId: "search", title: "T" });
  expect(nodeManifest(singleNodeFlow(plain), "node")).not.toHaveProperty("title");
});

test("test_flow_manifest_allows_shared_child_aliases_but_rejects_child_cycles", () => {
  const shared = fragment();
  const aliased = new Flow(
    new Map<string, Node>([
      ["left", new CallNode("left", { flowId: "child", child: shared })],
      ["right", new CallNode("right", { flowId: "child", child: shared })],
    ]),
    [],
    {
      wiring: new Map([
        ["left", new Map()],
        ["right", new Map()],
      ]),
      startId: "left",
      endId: "right",
    },
  );
  expect(() => flowManifest(aliased)).not.toThrow();

  const cyclic = new Flow(new Map(), [], { wiring: new Map(), startId: "call", endId: "call" });
  cyclic.nodes.set("call", new CallNode("call", { flowId: "self", child: cyclic }));
  cyclic.wiring.set("call", new Map());
  expectError(
    () => flowManifest(cyclic),
    RangeError,
    "flow.nodes.call.child: recursive child-flow cycle is not manifestable",
  );
});

test("a CALL child that isn't a Flow is refused", () => {
  expectError(
    () => flowManifest(singleNodeFlow(new CallNode("c", { flowId: "x" }))),
    TypeError,
    "flow.nodes.c.child: expected Flow, got null",
  );
});

test("test_wait_fingerprint_and_expansion_digest_cover_absolute_until_wiring", () => {
  const sleep = () => new ToolNode("node", { toolId: "sleep" });
  const first = wiredNodeFlow(sleep(), [["until", new Binding("2026-01-01T00:00:00+00:00")]]);
  const second = wiredNodeFlow(sleep(), [["until", new Binding("2026-01-02T00:00:00+00:00")]]);

  expect(flowFingerprint(first)).not.toBe(flowFingerprint(second));
  expect(expansionDigest(first)).not.toBe(expansionDigest(second));
});

test("a literal in MAP wiring and a CALL argument literal are identity", () => {
  const map = (over: unknown) =>
    wiredNodeFlow(new MapNode("node", { flowId: "child", child: fragment() }), [["over", over]]);
  const call = (x: unknown) => wiredNodeFlow(new CallNode("node", { flowId: "child", child: fragment() }), [["x", x]]);

  expect(flowFingerprint(map(["a"]))).not.toBe(flowFingerprint(map(["b"])));
  expect(flowFingerprint(call("first"))).not.toBe(flowFingerprint(call("second")));
  expect((flowManifest(call("first")).wiring as JsonObject).node).toEqual({
    entries: [{ key: { type: "str", value: "x" }, value: { type: "str", value: "first" } }],
    type: "mapping",
  });
});

test("test_total_loop_manifest_omits_returned_fields", () => {
  const total = nodeManifest(singleNodeFlow(loopNode()), "node");
  const partial = nodeManifest(singleNodeFlow(loopNode({ returnedFields: ["x"] })), "node");

  expect(total.kind).toBe("loop");
  expect("returned_fields" in total).toBe(false);
  expect(partial.returned_fields).toEqual(["x"]);
});

test("test_loop_without_max_has_null_max_in_manifest_and_its_own_fingerprint", () => {
  const withMax = singleNodeFlow(loopNode({ predicate: "n < 3", maxIters: 5 }));
  const without = singleNodeFlow(loopNode({ predicate: "n < 3" }));

  expect(nodeManifest(without, "node").max_iters).toBeNull();
  expect(nodeManifest(withMax, "node").max_iters).toBe(5);
  expect(flowFingerprint(without)).not.toBe(flowFingerprint(withMax));
  // Building the same loop again gives the same fingerprint.
  expect(flowFingerprint(without)).toBe(flowFingerprint(singleNodeFlow(loopNode({ predicate: "n < 3" }))));
});

test("a LOOP's times is a typed int, or null for a while loop", () => {
  expect(nodeManifest(singleNodeFlow(loopNode({ predicate: "n < 3" })), "node").times).toEqual({ type: "null" });
  expect(nodeManifest(singleNodeFlow(loopNode({ predicateKind: "times", times: 2 })), "node").times).toEqual({
    type: "int",
    value: "2",
  });
});

test.each([
  ["times", 1.5, "1.5"],
  ["maxIters", Infinity, "Infinity"],
  ["iteration", 0.5, "0.5"],
] as const)("a forged LOOP %s count is refused with its path", (field, value, text) => {
  const node = loopNode();
  Object.assign(node, { [field]: value });

  expectError(
    () => flowManifest(singleNodeFlow(node)),
    TypeError,
    `flow.nodes.node.${field}: expected a safe integer, got ${text}`,
  );
});

test("test_case_predicate_origins_do_not_affect_program_or_expansion_identity", () => {
  const first = singleNodeFlow(new CaseNode("case", [new Case("branch", { when: "true" })]));
  const second = singleNodeFlow(
    new CaseNode("case", [new Case("branch", { when: "true", origin: new OriginRef("case/0/when") })]),
  );

  expect(flowManifest(first)).toEqual(flowManifest(second));
  expect(flowFingerprint(first)).toBe(flowFingerprint(second));
  expect(expansionDigest(first)).toBe(expansionDigest(second));
});

test("test_case_manifest_rejects_a_mutated_non_origin_predicate_carrier", () => {
  const forged = Object.assign(Object.create(Case.prototype), new Case("branch", { when: "true" }), {
    origin: "flow.yaml:12",
  });
  const flow = singleNodeFlow(new CaseNode("case", [forged]));

  expectError(() => flowManifest(flow), TypeError, "flow.nodes.case.cases[0].origin: expected OriginRef or null");
});

test("a CASE branch that isn't a Case is refused", () => {
  const flow = singleNodeFlow(new CaseNode("case", [{ handle: "branch", when: "true", origin: null } as Case]));

  expectError(() => flowManifest(flow), TypeError, "flow.nodes.case.cases[0]: expected Case, got Object");
});

test("CASE cases that isn't an array is refused", () => {
  const node = new CaseNode("case", [new Case("branch", { when: "true" })]);
  Object.assign(node, { cases: "x" });

  expectError(() => flowManifest(singleNodeFlow(node)), TypeError, "flow.nodes.case.cases: expected array, got string");
});

// ---- Expansion digests -----------------------------------------------------------------------

/** Python's `_replace_body`: swap the fragment's body for one with a different CODE body. */
function replaceBody(flow: Flow): Flow {
  const body = new CodeNode("body", { code: 'return inputs["x"] + 1' });
  body.params = [new Parameter("x", { contract: intContract() })];
  body.outputContract = intContract();
  flow.nodes.set("body", body);
  return flow;
}

test.each([
  ["entry", (flow: Flow) => withBoundary(flow, { startId: "body" })],
  ["terminal", (flow: Flow) => withBoundary(flow, { endId: "body" })],
  ["node-fields", replaceBody],
  [
    "edge-order",
    (flow: Flow) => {
      flow.edges.reverse();
      return flow;
    },
  ],
  [
    "wiring",
    (flow: Flow) => {
      flow.wiring.set("body", new Map([["x", new Binding(2n)]]));
      return flow;
    },
  ],
  [
    "topology",
    (flow: Flow) => {
      flow.edges.push(new Edge("entry->terminal", "entry", "terminal", { ordering: true }));
      return flow;
    },
  ],
] as const)("test_expansion_digest_changes_for_every_builder_topology_component: %s", (_, mutate) => {
  const original = fragment();
  const changed = mutate(fragment());

  expect(expansionDigest(changed)).not.toBe(expansionDigest(original));
  expect(expansionDigest(original)).toMatch(HEX);
});

test("test_expansion_digest_is_stable_across_canonical_mapping_order", () => {
  const original = fragment();
  const shuffled = fragment();
  shuffled.nodes = new Map([...shuffled.nodes].reverse());
  shuffled.wiring = new Map([...shuffled.wiring].reverse().map(([id, wiring]) => [id, new Map([...wiring].reverse())]));

  expect(expansionDigest(shuffled)).toBe(expansionDigest(original));
  expect(() => JSON.stringify(flowManifest(shuffled))).not.toThrow();
});

// ---- Expansion topology ----------------------------------------------------------------------

/** Return the lowercase SHA-256 the manifest module takes over `manifest`'s RFC 8785 text. */
function sha256(manifest: JsonValue): string {
  return createHash("sha256").update(canonicalJson(manifest), "utf8").digest("hex");
}

test("test_memory_scope_is_not_authored_identity_but_is_expansion_topology", () => {
  const plain = fragment();
  const stamped = fragment();
  code(stamped, "body").memoryScope = new MemoryScope(["body"]);

  expect(flowManifest(stamped)).toEqual(flowManifest(plain));
  expect(flowFingerprint(stamped)).toBe(flowFingerprint(plain));
  expect(expansionDigest(stamped)).not.toBe(expansionDigest(plain));
});

test("test_memory_expansion_projection_covers_keys_and_occurrence_topology", () => {
  const scopePath = ["nested", "body"];
  const segment = new MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ["nested"], 0);
  const baseline = fragment();
  code(baseline, "body").memoryScope = new MemoryScope(scopePath);
  const variants = [baseline];
  for (const [memories, occurrencePath] of [
    [["notes"], []],
    [["chat"], []],
    [["notes", "chat"], []],
    [["chat", "notes"], []],
    [["notes"], [segment]],
  ] as const) {
    const flow = fragment();
    code(flow, "body").memoryScope = new MemoryScope(scopePath, { memories, occurrencePath });
    variants.push(flow);
  }

  expect(new Set(variants.map(expansionDigest)).size).toBe(6);
});

test("test_memory_expansion_projection_fields", () => {
  const segment = new MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ["nested"], 1);
  const flow = fragment();
  code(flow, "body").memoryScope = new MemoryScope(["nested", "body"], {
    memories: ["notes"],
    occurrencePath: [segment],
  });
  // `fragment()` has no growth metadata, so its authored manifest equals its expansion manifest.
  const expected = flowManifest(fragment());
  ((expected.nodes as JsonObject).body as JsonObject).memory_scope = {
    memories: ["notes"],
    occurrence_path: [{ index: 1, kind: "map_item", static_path: ["nested"] }],
    static_path: ["nested", "body"],
  };

  expect(expansionDigest(flow)).toBe(sha256(expected));
});

test("a forged memory scope or occurrence segment is refused", () => {
  const segment = new MemoryOccurrenceSegment(MemoryOccurrenceKind.MAP_ITEM, ["nested"], 1);
  const scoped = (scope: unknown) => {
    const flow = fragment();
    Object.assign(code(flow, "body"), { memoryScope: scope });
    return flow;
  };
  const forgedSegment = Object.assign(Object.create(MemoryOccurrenceSegment.prototype), segment, { index: 0.5 });
  const forgedScope = Object.assign(Object.create(MemoryScope.prototype), new MemoryScope(["nested", "body"]), {
    occurrencePath: [forgedSegment],
  });

  expectError(
    () => expansionDigest(scoped({ staticPath: ["body"] })),
    TypeError,
    "expansion.nodes.body.memoryScope: expected exact MemoryScope, got Object",
  );
  expectError(() => expansionDigest(scoped(forgedScope)), TypeError, "index must be an integer");
  // The authored manifest never reads the scope.
  expect(flowFingerprint(scoped(forgedScope))).toBe(FRAGMENT_FINGERPRINT);
});

test("test_memory_expansion_projection_omits_root_id_and_config_but_hashes_keys", () => {
  /** `compiled()` with flow id `rootId`, one `policy` memory of config `{limit}`, scoped on its step. */
  const withMemory = (rootId: string, policy: string, limit: bigint): CompiledFlow => {
    const flow = compiled();
    Object.assign(flow, {
      flowId: rootId,
      memories: [new MemoryDecl(policy, policy, { limit }, new DeclSite([], [rootId]))],
    });
    code(flow, "step").memoryScope = new MemoryScope(["step"], { memories: [policy] });
    return flow;
  };
  const first = withMemory("first", "outer", 3n);
  const second = withMemory("second", "outer", 99n);
  const otherKey = withMemory("first", "inner", 3n);

  expect(flowFingerprint(first)).not.toBe(flowFingerprint(second));
  expect(expansionDigest(first)).toBe(expansionDigest(second));
  expect(expansionDigest(first)).not.toBe(expansionDigest(otherKey));
});

test("test_expansion_digest_applies_topology_only_policy_to_recursive_children", () => {
  const parent = (child: CompiledFlow) => singleNodeFlow(new CallNode("invoke", { flowId: "child", child }));
  const original = parent(compiled());
  const changedChild = compiled();
  changedChild.flowLlmConfig = { openai: { temperature: 0.7 } };
  const changed = parent(changedChild);

  expect(expansionDigest(changed)).toBe(expansionDigest(original));
  expect(flowFingerprint(changed)).not.toBe(flowFingerprint(original));
});

/**
 * `fragment()` grown as a child: `scope` on its entry, and its seed connection (at `seedPath`) and
 * child boundary unless `metadata` is false.
 */
function grown(
  options: { seedPath?: (string | number)[]; scope?: ExecutionScope | null; metadata?: boolean } = {},
): Flow {
  const { seedPath = [0, "x"], scope = null, metadata = true } = options;
  const base = fragment();
  const entry = new StartNode("entry", { params: [new Parameter("x", { contract: intContract() })] });
  entry.executionScope = scope;
  base.nodes.set("entry", entry);
  return new Flow(base.nodes, base.edges, {
    wiring: new Map<string, ReadonlyMap<string, unknown>>([...base.wiring, ["entry", new Map([["x", 1n]])]]),
    startId: "entry",
    endId: "terminal",
    inputConnections: metadata ? [new SeedConnection("entry", "x", seedPath)] : [],
    childBoundaries: metadata ? [new ChildFlowBoundary("entry", "terminal")] : [],
  });
}

test("test_growth_metadata_and_execution_topology_have_a_separate_digest_policy", () => {
  const scope = new ExecutionScope(["call", "entry"], { flowPath: ["call"], boundary: "start" });
  const steps = [
    grown(),
    grown({ seedPath: [3, "x"] }),
    grown({ seedPath: [3, "x"], scope }),
    grown({ scope, metadata: false }),
  ];
  const authored = flowFingerprint(grown());

  for (const flow of steps) expect(flowFingerprint(flow)).toBe(authored);
  expect(new Set(steps.map(expansionDigest)).size).toBe(steps.length);
});

test("an execution scope projects its paths, positions and boundary", () => {
  const flow = fragment();
  flow.nodes.get("terminal")!.executionScope = new ExecutionScope(["outer", "terminal"], {
    flowPath: ["outer"],
    occurrencePath: [new ExecutionPosition("call", ["outer"]), new ExecutionPosition("map_item", ["outer"], 2)],
    boundary: "end",
  });
  // `fragment()` has no growth metadata, so its authored manifest equals its expansion manifest.
  const expected = flowManifest(fragment());
  ((expected.nodes as JsonObject).terminal as JsonObject).execution_scope = {
    boundary: "end",
    flow_path: ["outer"],
    occurrence_path: [
      { index: null, kind: "call", static_path: ["outer"] },
      { index: 2, kind: "map_item", static_path: ["outer"] },
    ],
    static_path: ["outer", "terminal"],
  };

  expect(expansionDigest(flow)).toBe(sha256(expected));
  expect(flowFingerprint(flow)).toBe(FRAGMENT_FINGERPRINT);
});

test("an execution scope that isn't an ExecutionScope is refused", () => {
  const flow = fragment();
  Object.assign(code(flow, "body"), { executionScope: { staticPath: ["body"] } });

  expectError(() => expansionDigest(flow), TypeError, "expansion.nodes.body.executionScope must be an ExecutionScope");
});

// ---- Loaded flows ----------------------------------------------------------------------------

const INLINE_FLOW = `
id: stable
name: Stable
input:
  x: !int 1
env:
  beta: 2
  alpha: 1
nodes:
  step:
    kind: code
    input:
      x: \${input.x}
    output: int
    code: |
      return inputs["x"] + 1n
    asserts:
      - x >= 0
      - x < 100
      - output > x
output: \${step.output}
`;

const INLINE_FLOW_WITH_COMMENTS = `
# This comment and the blank lines move every source range and inline-code frame.

id: stable
name: Stable
input:
  x: !int 1
env:
  alpha: 1
  beta: 2  # mapping presentation is not program identity
nodes:
  step:
    kind: code
    input:
      x: \${input.x}
    output: int
    code: |
      return inputs["x"] + 1n
    asserts:
      - x >= 0
      - x < 100
      - output > x
output: \${step.output}
`;

/** Load `source` and return its compiled flow (Python's `_compiled`). */
async function loaded(source: string = INLINE_FLOW): Promise<CompiledFlow> {
  return (await loadFlow(source)).compiled;
}

test("test_loaded_flow_stores_only_exact_parameter_carriers", async () => {
  // Compose builders and synthesized boundaries share one compiled parameter shape.
  const flow = await loaded();

  for (const node of flow.nodes.values()) {
    if (node.params === null) continue;
    expect(Array.isArray(node.params)).toBe(true);
    expect(node.params.every((parameter) => isExactly(parameter, Parameter))).toBe(true);
  }
});

const INLINE_COMPOSITE_IDENTITY = `
id: inline-identity
name: inline_identity
input:
  value: !int
nodes:
  wrapper:
    kind: call
    input:
      value: \${input.value}
    nodes:
      step:
        kind: code
        input:
          value: \${input.value}
        output: int
        code: |
          return inputs["value"] + 1n
    output: \${step.output}
output: \${wrapper.output}
`;

test("test_flow_fingerprint_ignores_comments_and_python_object_identity", async () => {
  const plain = await loaded();
  const moved = await loaded(INLINE_FLOW_WITH_COMMENTS);

  expect(plain.nodes.get("step")).not.toBe(moved.nodes.get("step"));
  expect(flowManifest(plain)).toEqual(flowManifest(moved));
  expect(flowFingerprint(plain)).toBe(flowFingerprint(moved));
});

test("test_inline_identity_ignores_formatting_but_tracks_owner_and_child_semantics", async () => {
  const baseline = await loaded(INLINE_COMPOSITE_IDENTITY);
  const formatted = await loaded(
    "# moves every authored range without changing the program\n\n" + INLINE_COMPOSITE_IDENTITY,
  );
  const changedChild = await loaded(
    INLINE_COMPOSITE_IDENTITY.replace('return inputs["value"] + 1n', 'return inputs["value"] + 2n'),
  );
  const renamedOwner = await loaded(
    INLINE_COMPOSITE_IDENTITY.replace("wrapper:", "renamed:").replace("${wrapper.output}", "${renamed.output}"),
  );

  const flowIdOf = (flow: CompiledFlow, id: string) => (flow.nodes.get(id) as CallNode).flowId;
  const baselineId = flowIdOf(baseline, "wrapper");
  expect(flowIdOf(formatted, "wrapper")).toBe(baselineId);
  expect(flowIdOf(changedChild, "wrapper")).toBe(baselineId);
  expect(flowIdOf(renamedOwner, "renamed")).not.toBe(baselineId);

  expect(flowManifest(formatted)).toEqual(flowManifest(baseline));
  expect(flowFingerprint(formatted)).toBe(flowFingerprint(baseline));
  expect(flowFingerprint(changedChild)).not.toBe(flowFingerprint(baseline));
  expect(flowFingerprint(renamedOwner)).not.toBe(flowFingerprint(baseline));
});

test("test_inline_manifest_canonicalizes_generated_structural_input_types", async () => {
  const flow = await loaded(`id: inline-manifest-structural-input
name: inline_manifest_structural_input
typedefs:
  Row:
    z: int
    a: Optional(str)
input:
  row: !Row
nodes:
  wrapper:
    kind: call
    input:
      row: \${input.row}
    nodes:
      echo:
        kind: code
        code: |
          return inputs["row"]
        input:
          row: \${input.row}
        output: Row
    output: \${echo.output}
output: \${wrapper.output}
`);

  const wrapper = (flowManifest(flow).nodes as JsonObject).wrapper as JsonObject;
  const childStart = ((wrapper.child as JsonObject).nodes as JsonObject)[START_ID] as JsonObject;
  const generated = (childStart.params as JsonObject[])[0]!;
  const fields = ((generated.contract as JsonObject).type as JsonObject).fields as JsonObject[];

  expect(fields.map((field) => field.name)).toEqual(["a", "z"]);
});

test("test_reference_code_identity_is_the_token_not_the_callable_body", async () => {
  const dir = mkdtempSync(join(tmpdir(), "yian-identity-"));
  try {
    const module = join(dir, "identity_refmod.ts");
    writeFileSync(module, "export function fn(inputs) {\n  return 'one';\n}\n");
    const text = `
id: ref
name: Ref
nodes:
  leaf:
    kind: code
    output: str
    code: ./identity_refmod.ts:fn
output: \${leaf.output}
`;
    const fingerprint = async (source: string) => flowFingerprint((await loadFlow(source, { baseDir: dir })).compiled);
    const before = await fingerprint(text);

    const changed = "export function fn(inputs) {\n  return 'a changed body';\n}\n";
    writeFileSync(module, changed);
    writeFileSync(join(dir, "identity_refmod_other.ts"), changed);

    expect(await fingerprint(text)).toBe(before);
    expect(await fingerprint(text.replace("identity_refmod.ts:fn", "identity_refmod_other.ts:fn"))).not.toBe(before);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test.each([
  ["api_key: sk-flow-secret", "llm_config.api_key is a host credential"],
  ["openai: {api_key: sk-flow-secret}", "`openai.api_key` is not a supported field"],
  ["anthropic: {api_key: sk-flow-secret}", "`anthropic.api_key` is not a supported field"],
])("test_a_flow_file_cannot_put_api_key_in_a_manifest: %s", async (layer, message) => {
  let caught: unknown = null;
  try {
    await loaded(`
id: keyed
name: Keyed
llm_config:
  provider: openai
  ${layer}
nodes:
  ask:
    kind: agent
    prompt: Answer briefly.
    output: str
output: \${ask.output}
`);
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(LoadError);
  expect((caught as LoadError).message).toContain(message);
  expect((caught as LoadError).message).not.toContain("sk-flow-secret");
});

/** A two-step flow whose steps both carry the same two literals. */
const resumeLiteralFlow = (url: string, signature: string) => `
id: resume-literal
name: resume_literal
nodes:
  first:
    kind: code
    input:
      url: '${url}'
      signature: '${signature}'
    code: |
      return inputs["url"]
  second:
    kind: code
    input:
      url: '${url}'
      signature: '${signature}'
      after: \${first.output}
    code: |
      return inputs["url"]
output:
  first: \${first.output}
  second: \${second.output}
`;

/** Cut the run log at `path` right after the record that commits `nodeId`. */
function truncateAfterNodeSuccess(path: string, nodeId: string): void {
  const lines = readFileSync(path, "utf8").split(/(?<=\n)/);
  const index = lines.findIndex((line) => {
    const record = JSON.parse(line);
    return record.kind === "node_succeeded" && record.payload.node_id === nodeId;
  });
  if (index < 0) throw new Error(`no node_succeeded record for ${nodeId}`);
  writeFileSync(path, lines.slice(0, index + 1).join(""));
}

test.each([
  [
    "url-query-key",
    ["https://example.com/search?key=apples", "q -> a"],
    ["https://example.com/search?key=pears", "q -> a"],
  ],
  [
    "signature",
    ["https://example.com/search?key=apples", "q -> a"],
    ["https://example.com/search?key=apples", "q -> s"],
  ],
] as const)("test_resume_rejects_a_flow_with_an_edited_literal: %s", async (_id, before, after) => {
  const dir = mkdtempSync(join(tmpdir(), "flow-identity-"));
  try {
    const log = join(dir, "run.jsonl");
    await runFlow(await loadFlow(resumeLiteralFlow(before[0], before[1])), {}, { log, numWorkers: 0 });
    truncateAfterNodeSuccess(log, "first");
    // Resume compiles the recorded flow, so the edit is to the recorded text.
    editRecordedFlow(log, () => resumeLiteralFlow(after[0], after[1]));

    const resumed = advance({ log, numWorkers: 0 });
    await expect(resumed).rejects.toThrow(RangeError);
    await expect(resumed).rejects.toThrow("flow fingerprint mismatch");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

const GOLDEN_FLOW = `
id: golden
name: golden
input:
  topic: !str
  limit: !int 3
  mode: !str 'fast'
llm_config:
  provider: openai
  model: gpt-4o
env:
  max_tool_iterations: 40
long_term_memory:
  notes:
    type: episodic
    config:
      max_episodes: 5
nodes:
  fetch:
    kind: code
    code: |
      return inputs
    env:
      region: eu
    input:
      topic: \${input.topic}
      url: 'https://example.com/search?q=\${input.topic}'
      opts: {depth: 2, tags: [a, b]}
      label: 'plain text'
  write:
    kind: agent
    prompt: Summarize \${fetch.output}.
    llm_config:
      model: gpt-4o-mini
    input:
      fetch: \${fetch.output}
      limit: \${input.limit}
output:
  summary: \${write.output}
  mode: \${input.mode}
`;

test("test_flow_without_formerly_redacted_values_keeps_its_fingerprint", async () => {
  expect(flowFingerprint(await loaded(GOLDEN_FLOW))).toBe(
    "786dc8194fcfbd44f554e4151e29991b1faf373d4f6445bb5a387196a2a28314",
  );
});

/** The pin flow with `outputSection` (Python's `_PIN_FLOW % output_section`). */
const pinFlow = (outputSection: string) => `
id: pin
name: pin
nodes:
  step:
    kind: code
    output: str
    code: |
      return "filled"
${outputSection}`;

const PIN_DEF_FLOW = `
id: pdef
name: pdef
defs:
  greet:
    kind: code
    output: str
    code: |
      return "hi"
nodes:
  c:
    kind: call
    call: greet
output: \${c.output}
`;

// These pins guard that the value form, an absent output, and multi-field records keep their
// encoding when END record semantics change; the END-manifest tests above are the primary guard.
test.each([
  ["value", "output: ${step.output}\n", "62defc93158f3d2e3b0e973c7c51270d98da4a6c4cba0c416e87cff51b7aa36a"],
  ["none", "", "2ce711f32439e55e347dad1e19e608e649cf9bbf12876a12fd3769ea2ebb0ef7"],
  [
    "two-fields",
    "output:\n  order: ${step.output}\n  note: ${step.output}\n",
    "f8dfe98712cff57decbe0a06ce38ce89670c41cf40fce95aef873a53cbd678b4",
  ],
])("test_loaded_flow_output_forms_keep_their_fingerprints: %s", async (_, outputSection, expected) => {
  expect(flowFingerprint(await loaded(pinFlow(outputSection)))).toBe(expected);
});

test("test_loaded_one_field_record_flow_has_its_pinned_fingerprint", async () => {
  expect(flowFingerprint(await loaded(pinFlow("output:\n  order: ${step.output}\n")))).toBe(
    "f882c54e76f793a662ab6967970c58476d90ebfa771f51b006a72ccaa72f8073",
  );
});

test("test_loaded_zero_and_one_field_records_differ_from_their_value_forms", async () => {
  const named = flowFingerprint(await loaded(pinFlow("output:\n  result: ${step.output}\n")));
  const value = flowFingerprint(await loaded(pinFlow("output: ${step.output}\n")));
  const empty = flowFingerprint(await loaded(pinFlow("output: {}\n")));
  const absent = flowFingerprint(await loaded(pinFlow("")));

  expect(named).not.toBe(value);
  expect(empty).not.toBe(absent);
});

test("test_call_over_a_single_node_def_keeps_its_fingerprint", async () => {
  expect(flowFingerprint(await loaded(PIN_DEF_FLOW))).toBe(
    "90a5fb6b7cf746fd492fb240d500d5b5b3f04bb045aaac1faac21d56ae5e7a0f",
  );
});

test("test_loaded_value_output_and_single_node_def_use_a_value_end", async () => {
  const flow = await loaded(pinFlow("output: ${step.output}\n"));
  expect((flow.nodes.get(flow.endId) as EndNode).returnsRecord).toBe(false);

  const call = (await loaded(PIN_DEF_FLOW)).nodes.get("c") as CallNode;
  const child = call.child as Flow;
  expect((child.nodes.get(child.endId) as EndNode).returnsRecord).toBe(false);
});

/** The loop flow with `predicate` and `max: 5` (Python's `_LOOP_IDENTITY % predicate`). */
const loopIdentity = (predicate: string) => `
id: count
name: count
nodes:
  loop:
    kind: loop
    input:
      n: 0
    ${predicate}
    max: 5
    nodes:
      bump:
        kind: code
        input:
          n: \${input.n}
        output:
          n: int
        code: |
          return {"n": inputs["n"] + 1n}
    output: \${bump.output}
output: \${loop.output.n}
`;

test.each([
  ["while: n < 3", "99eeb626fcb47ff8d0bd152d0a1b6eb19f210156f9a91cb528a78783598e99a7"],
  ["until: n >= 3", "e01bdcef87cf52de28135353b42e02f6d15b2154940b976f48f58a72e0497e5b"],
])("test_loop_with_max_keeps_its_fingerprint: %s", async (predicate, digest) => {
  // Pinned before `max:` became optional: a loop that sets `max:` keeps its identity.
  expect(flowFingerprint(await loaded(loopIdentity(predicate)))).toBe(digest);
});
