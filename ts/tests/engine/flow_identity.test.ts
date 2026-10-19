/**
 * Stable authored-flow manifests and deterministic expansion-topology digests
 * (`compile/manifest.ts`): rows of `tests/engine/test_flow_identity.py`.
 *
 * Ported: `test_flow_manifest_emits_the_literal_node_kind`,
 * `test_flow_fingerprint_covers_each_included_common_node_field`,
 * `test_node_origin_does_not_affect_program_or_expansion_identity`,
 * `test_node_origin_context_does_not_affect_program_or_expansion_identity`,
 * `test_prompts_code_and_tool_ids_keep_uri_text` (the CODE and TOOL pairs; the AGENT prompt pair
 * comes with AGENT), `test_flow_fingerprint_covers_each_parameter_field`,
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
 * `test_expansion_digest_changes_for_every_builder_topology_component` and
 * `test_expansion_digest_is_stable_across_canonical_mapping_order`.
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
 * the same rule and aren't ported one by one).
 * `test_flow_fingerprint_changes_with_node_kind_and_execution_source` and
 * `test_flow_fingerprint_recursively_includes_static_child_flows` are covered by the kind-literal
 * and kind-field rows.
 *
 * Changed:
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
 * - The non-JSON row uses a `Map`, a `Uint8Array`, a class instance, a function and `undefined`.
 * - The malformed type-table rows are a record instead of a `Map`, a `Map` with a `bigint` key, and
 *   a `Map` whose value is the string `"int"`.
 * - Setting a CODE node's `kind` to `"agent"` is refused as not ported, since AGENT's manifest
 *   comes with AGENT.
 * - The deep-union row has no call counter (an ES module can't be monkeypatched): a depth-30 union
 *   must manifest within the test's timeout, which an exponential encoding never does.
 * - The census is `Object.keys`, so the trait fields and `kind`, which a kind fixes, are ignored
 *   fields: an instance override of a trait goes unnoticed (Python would refuse
 *   `node.is_spawner = True` as unknown).
 * - A lone surrogate in a name or value is left to `canonicalJson`'s message, which carries no
 *   path (load refuses one first).
 *
 * TS-only: typed floats (`1.5`, `Infinity`, `-0`), NaN refused, `2n` vs `2`, a cyclic env record vs
 * a shared alias, a `FileRef` encoded as its mapping, `params` that isn't an array, a forged END
 * count, a list END's `names`, seed-path indexes as counts, the pinned fingerprint and expansion
 * digest of `fragment()`, a missing node field, a node whose `kind` isn't its class's, labels and a
 * CODE or TOOL title or base directory outside identity, the CODE descriptors (`{"inline": ...}`,
 * `{"ref": ...}`, `{"builtin": "identity", "field": ...}`) and their refusals, and TOOL `retries`
 * (absent at 0, identity otherwise, a forged count refused), and the flow fields: a typed wiring
 * binding's `binding_types` and a typed output's `type`, a plain `Flow`'s missing CompiledFlow keys,
 * the flow `llm_config:` layer (literal values are identity and key order isn't; the
 * `ValueTemplate` and `Binding` arms), the LLM value types and adjacency index outside identity,
 * the flow id and flow memories (identity when present, never expansion topology, forged values
 * refused in both modes), and an unknown flow field, a `Flow` subclass, a non-Flow, a wiring
 * value that isn't a `Binding` and a node wiring that isn't a `Map` refused.
 *
 * Later (T3-22): the loader rows `test_loaded_flow_stores_only_exact_parameter_carriers`,
 * `test_flow_fingerprint_ignores_comments_and_python_object_identity`,
 * `test_inline_identity_ignores_formatting_but_tracks_owner_and_child_semantics`,
 * `test_inline_manifest_canonicalizes_generated_structural_input_types`,
 * `test_reference_code_identity_is_the_token_not_the_callable_body`,
 * `test_a_flow_file_cannot_put_api_key_in_a_manifest`,
 * `test_flow_without_formerly_redacted_values_keeps_its_fingerprint`,
 * `test_loaded_flow_output_forms_keep_their_fingerprints`,
 * `test_loaded_one_field_record_flow_has_its_pinned_fingerprint`,
 * `test_loaded_zero_and_one_field_records_differ_from_their_value_forms`,
 * `test_call_over_a_single_node_def_keeps_its_fingerprint`,
 * `test_loaded_value_output_and_single_node_def_use_a_value_end` and
 * `test_loop_with_max_keeps_its_fingerprint`. With AGENT (T3-22, which ports `AgentNode`'s
 * manifest): `test_flow_fingerprint_includes_authored_llm_layers_but_not_effective_values`,
 * `test_manifest_records_blocks_as_nested_sorted_mappings`,
 * `test_flow_fingerprint_tracks_block_fields_but_not_key_order`,
 * `test_block_field_names_are_not_redacted_as_credentials`, the `test_agent_variable_*` rows and
 * `test_expansion_digest_excludes_effective_but_includes_authored_llm_config`.
 *
 * Later (T4): `test_resume_rejects_a_flow_with_an_edited_literal` (a run and a resume).
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

import { expansionDigest, flowFingerprint, flowManifest } from "../../src/compile/manifest.ts";
import { CompiledFlow, Edge, END_ID, Flow, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { DeclSite, MemoryDecl } from "../../src/memory/contracts.ts";
import type { Node } from "../../src/nodes/base.ts";
import {
  Assertion,
  Binding,
  ChildFlowBoundary,
  ContractOrigins,
  Parameter,
  SeedConnection,
  ValueContract,
  ValueTemplate,
} from "../../src/nodes/contracts.ts";
import { type BuiltinCode, CodeNode } from "../../src/nodes/code/node.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { FileRef, Type, typeFor, ValueKind } from "../../src/typesys/index.ts";
import { canonicalJson, type JsonObject, type JsonValue } from "../../src/typesys/json.ts";

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
  ["agent", Error, "flow.nodes.step: node kind 'agent' manifest is not ported yet (T3-22)"],
] as const)("a CODE node whose kind is %p is refused", (kind, cls, message) => {
  const flow = compiled();
  Object.assign(code(flow, "step"), { kind });

  expectError(() => flowManifest(flow), cls, message);
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

test("CODE mode, base directory and origins keep the fingerprint", () => {
  const baseline = codeFingerprint("./h.ts:fn");

  expect(codeFingerprint("./h.ts:fn", { mode: "inline" })).toBe(baseline);
  expect(codeFingerprint("./h.ts:fn", { baseDir: "/flows" })).toBe(baseline);
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
  const pairs = [
    [
      new CodeNode("node", { code: `return ${JSON.stringify(firstUri)}` }),
      new CodeNode("node", { code: `return ${JSON.stringify(secondUri)}` }),
    ],
    [new ToolNode("node", { toolId: firstUri, retries: 0 }), new ToolNode("node", { toolId: secondUri, retries: 0 })],
  ];

  for (const [first, second] of pairs) {
    expect(flowFingerprint(singleNodeFlow(first!))).not.toBe(flowFingerprint(singleNodeFlow(second!)));
    expect(canonicalJson(flowManifest(singleNodeFlow(first!)))).toContain("first-secret");
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
  flow.wiring.set("body", new Map([["x", 1n]]) as unknown as ReadonlyMap<string, Binding>);

  expectError(() => flowManifest(flow), TypeError, "flow.wiring.body.values[0]: expected Binding, got bigint");
});

test("a node wiring that isn't a Map is refused", () => {
  const flow = fragment();
  flow.wiring.set("body", { x: new Binding(1n) } as unknown as ReadonlyMap<string, Binding>);

  expectError(() => flowManifest(flow), TypeError, "flow.wiring.body: expected Map, got Object");
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
