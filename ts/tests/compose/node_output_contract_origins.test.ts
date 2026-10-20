/**
 * The authored-site origins of a node's `output:` contract: a port of
 * `tests/compose/test_node_output_contract_origins.py`.
 *
 * Changed: the flows assemble with `assembleText`, where Python loads them, and an origin
 * resolves through the scope's catalog builder (`builder.resolveRegistered`) in place of
 * `loaded.origin_resolver`.
 * `test_formatting_only_relocation_preserves_output_origins_and_fingerprint` keeps its origin half.
 * `test_nested_inline_child_output_contract_uses_its_physical_site` reads the CODE node of the
 * CALL's baked child.
 *
 * Pinned in `engine/read_type.test.ts` (the cases of the reader that raises):
 * `test_compact_invalid_output_type_reports_the_top_level_output_site`,
 * `test_invalid_nested_output_type_reports_the_exact_value_site`,
 * `test_unsupported_nested_output_type_reports_the_exact_value_site` and
 * `test_invalid_output_list_item_reports_the_exact_item_site`.
 *
 * Later (T4-9):
 * `test_nested_output_failure_selects_the_exact_contract_field`,
 * `test_root_output_failure_selects_the_contract_root`,
 * `test_scalar_list_output_failure_falls_back_to_the_contract_root` and
 * `test_code_syntax_error_precedes_an_invalid_output_type` (TS `build` compiles no CODE, so T4-9
 * decides that order). Later (T5-11a):
 * `test_agent_structured_failure_selects_exact_authored_output_field`.
 * Later (T3-22): `test_multi_node_def_output_contract_uses_the_nested_node_site`,
 * `test_flat_def_output_contract_uses_the_projected_def_output_site`,
 * `test_external_alias_rebases_every_output_contract_origin` and the fingerprint half of the
 * formatting row.
 * Later (T3-16, T3-22, T4):
 * `test_named_call_output_failure_keeps_call_ancestry_and_exact_child_field`.
 */

import { expect, test } from "bun:test";

import type { CompiledFlow } from "../../src/compile/model.ts";
import { BuildContext } from "../../src/compose/nodes/base.ts";
import { CodeDescriptor } from "../../src/compose/nodes/code.ts";
import type { SourceCatalogBuilder } from "../../src/compose/source/catalog.ts";
import { type SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import { ContractOrigins, ValueContract } from "../../src/nodes/contracts.ts";
import { ValueKind } from "../../src/typesys/values.ts";
import { assembleText, loadError } from "./declarations.ts";

const NESTED_OUTPUT_FLOW = `id: nested-output-contract
name: nested-output-contract
nodes:
  emit:
    kind: code
    code: |-
      return {"profile": {"age": "bad", "name": "Ada"}, "ok": true}
    output:
      profile:
        age: int
        name: str
      ok: bool
output: \${emit.output}
`;

/** Assemble `text` under `label`; return node `nodeId`'s exact output contract and the builder. */
function contractOf(
  text: string,
  nodeId: string,
  label: string,
): { contract: ValueContract; builder: SourceCatalogBuilder } {
  const { flow, builder } = assembleText(text, label);
  const contract = flow.compiled.nodes.get(nodeId)!.outputContract;
  expect(contract?.constructor).toBe(ValueContract);
  return { contract: contract as ValueContract, builder };
}

/** Expect `origin` to resolve to exactly `cursor`'s value occurrence. */
function expectSite(builder: SourceCatalogBuilder, origin: OriginRef | null | undefined, cursor: SourceCursor): void {
  expect(origin).toBeInstanceOf(OriginRef);
  const frame = builder.resolveRegistered(origin!);
  expect(frame).not.toBeNull();
  expect(frame!.text).toBe(cursor.document!.text);
  expect(frame!.range).toEqual(cursor.range!);
}

/** Return `contract`'s nested origins by path, the path joined with `.`. */
function entriesOf(contract: ValueContract): Map<string, OriginRef> {
  return new Map(contract.origins.entries.map(([path, origin]) => [path.join("."), origin]));
}

/** Expect the declaration root and the nested profile fields to keep their sites. */
function expectProfileSites(builder: SourceCatalogBuilder, contract: ValueContract, output: SourceCursor): void {
  const entries = entriesOf(contract);
  expectSite(builder, contract.origins.default, output);
  expectSite(builder, entries.get("profile"), output.child("profile"));
  expectSite(builder, entries.get("profile.age"), output.child("profile").child("age"));
}

test("full code output compiles root and nested contract origins", () => {
  const output = SourceDocument.parse(NESTED_OUTPUT_FLOW, "nested.yaml")
    .root.child("nodes")
    .child("emit")
    .child("output");
  const { contract, builder } = contractOf(NESTED_OUTPUT_FLOW, "emit", "nested.yaml");
  const entries = entriesOf(contract);

  expectProfileSites(builder, contract, output);
  expectSite(builder, entries.get("profile.name"), output.child("profile").child("name"));
  expectSite(builder, entries.get("ok"), output.child("ok"));
});

test("list form output keeps winning sites and record semantics", () => {
  const text = `id: list-output-contract
name: list-output-contract
nodes:
  emit:
    kind: code
    code: |-
      return {"profile": {"age": 7, "active": true}, "status": false}
    output:
      - profile:
          age: str
      - status: bool
      - profile:
          age: int
          active: bool
output: \${emit.output}
`;
  const output = SourceDocument.parse(text, "list.yaml").root.child("nodes").child("emit").child("output");
  const { contract, builder } = contractOf(text, "emit", "list.yaml");
  const entries = entriesOf(contract);

  const fields = contract.type.fields!;
  expect([...fields.keys()]).toEqual(["profile", "status"]);
  const profile = fields.get("profile")!.fields!;
  expect(profile.get("age")!.kind).toBe(ValueKind.INTEGER);
  expect(profile.get("active")!.kind).toBe(ValueKind.BOOLEAN);
  const winning = output.child(2).child("profile");
  expectSite(builder, entries.get("profile"), winning);
  expectSite(builder, entries.get("profile.age"), winning.child("age"));
  expectSite(builder, entries.get("profile.active"), winning.child("active"));
  expectSite(builder, entries.get("status"), output.child(1).child("status"));
});

test("direct descriptor output contract remains source free", () => {
  const descriptor = new CodeDescriptor({
    id: "emit",
    code: 'return {"profile": {"age": 7}}',
    outputs: { profile: { age: "int" } },
  });

  const contract = descriptor.build(new BuildContext({ registry: new Map() })).node.outputContract;

  expect(contract?.constructor).toBe(ValueContract);
  expect((contract as ValueContract).origins).toEqual(new ContractOrigins());
  expect((contract as ValueContract).type.fields!.get("profile")!.fields!.get("age")!.kind).toBe(ValueKind.INTEGER);
});

test("compact root output contract uses the top-level output site", () => {
  const text = `id: compact_output_contract
name: compact_output_contract
kind: code
code: |-
  return {"profile": {"age": 7}}
output:
  profile:
    age: int
`;
  const { contract, builder } = contractOf(text, "compact_output_contract", "compact.yaml");

  expectProfileSites(builder, contract, SourceDocument.parse(text, "compact.yaml").root.child("output"));
});

test("adaptive generated output contracts remain source free", () => {
  // Desugared contracts without authorable type declarations invent no origins.
  const text = `id: adaptive-output-contracts
name: adaptive-output-contracts
input:
  topic: !str
nodes:
  ask:
    kind: human_input
    input:
      topic: \${input.topic}
    adaptive_questions:
      prompt: Ask about \${topic}
output: \${ask.output}
`;
  const nodes = assembleText(text).flow.compiled.nodes;

  for (const nodeId of ["ask__compose", "ask"]) {
    const contract = nodes.get(nodeId)!.outputContract;
    expect(contract?.constructor).toBe(ValueContract);
    expect((contract as ValueContract).origins).toEqual(new ContractOrigins());
  }
});

test("formatting-only relocation preserves output origins", () => {
  const relocatedText = `id: nested-output-contract
name: nested-output-contract

# Formatting alone moves the declaration.
nodes:
  emit:
    kind: code
    code: |-
      return {"profile": {"age": "bad", "name": "Ada"}, "ok": true}
    # This comment moves the whole contract again.
    output:
      profile:
        # Nested fields move independently too.
        age: int
        name: str
      ok: bool
output: \${emit.output}
`;
  const outputOf = (text: string) =>
    SourceDocument.parse(text, "relocated.yaml").root.child("nodes").child("emit").child("output");
  const baseline = contractOf(NESTED_OUTPUT_FLOW, "emit", "relocated.yaml");
  const relocated = contractOf(relocatedText, "emit", "relocated.yaml");
  const baselineOutput = outputOf(NESTED_OUTPUT_FLOW);
  const relocatedOutput = outputOf(relocatedText);

  expect(relocated.contract.origins).toEqual(baseline.contract.origins);
  expectProfileSites(baseline.builder, baseline.contract, baselineOutput);
  expectProfileSites(relocated.builder, relocated.contract, relocatedOutput);
  expect(relocatedOutput.range).not.toEqual(baselineOutput.range);
  expect(relocatedOutput.child("profile").child("age").range).not.toEqual(
    baselineOutput.child("profile").child("age").range,
  );
});

test("a nested inline child's output contract uses its physical site", () => {
  // Inline extraction retains the nested child's output declaration.
  const text = `id: inline-output-contract
name: inline-output-contract
nodes:
  invoke:
    kind: call
    nodes:
      emit:
        kind: code
        code: |-
          return {"profile": {"age": 7}}
        output:
          profile:
            age: int
    output: \${emit.output}
output: \${invoke.output}
`;
  const output = SourceDocument.parse(text, "inline.yaml")
    .root.child("nodes")
    .child("invoke")
    .child("nodes")
    .child("emit")
    .child("output");
  const { flow, builder } = assembleText(text, "inline.yaml");
  const invoke = flow.compiled.nodes.get("invoke") as CallNode;
  const contract = (invoke.child as CompiledFlow).nodes.get("emit")!.outputContract;

  expect(contract?.constructor).toBe(ValueContract);
  expectProfileSites(builder, contract as ValueContract, output);
});

test("non-string nested output key reports the key site", () => {
  const text = `id: invalid-output-key
name: invalid-output-key
nodes:
  emit:
    kind: code
    code: 'return {"profile": {}}'
    output:
      profile:
        7: int
output: \${emit.output}
`;
  const document = SourceDocument.parse(text, "invalid-key.yaml");
  const profile = document.root.child("nodes").child("emit").child("output").child("profile");

  const error = loadError(() => assembleText(text, "invalid-key.yaml"));

  expect(error.message).toContain("the key `7` was read by YAML as an integer");
  const frame = error.diagnostic.frames.at(-1)!;
  expect(frame.text).toBe(document.text);
  expect(frame.range).toEqual(profile.keyRangeOf("7")!);
});

test("authorable leaf kinds receive output root origins on load", () => {
  const text = `id: output-origin-leaves
name: output-origin-leaves
nodes:
  agent_leaf:
    kind: agent
    prompt: hi
    output: str
  code_leaf:
    kind: code
    code: return "ok"
    output: str
  human_leaf:
    kind: human_input
    prompt: continue?
    output: str
output: \${code_leaf.output}
`;
  const nodes = SourceDocument.parse(text, "leaves.yaml").root.child("nodes");
  const { flow, builder } = assembleText(text, "leaves.yaml");

  for (const nodeId of ["agent_leaf", "code_leaf", "human_leaf"]) {
    const contract = flow.compiled.nodes.get(nodeId)!.outputContract as ValueContract;
    expect(contract.constructor).toBe(ValueContract);
    expect(contract.origins.entries).toEqual([]);
    expectSite(builder, contract.origins.default, nodes.child(nodeId).child("output"));
  }
});
