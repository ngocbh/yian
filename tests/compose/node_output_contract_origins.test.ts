/** The authored-site origins of a node's `output:` contract. */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, spyOn, test } from "bun:test";

import { flowFingerprint } from "../../src/compile/manifest.ts";
import type { CompiledFlow } from "../../src/compile/model.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { BuildContext } from "../../src/compose/nodes/base.ts";
import { CodeDescriptor } from "../../src/compose/nodes/code.ts";
import type { SourceCatalogBuilder } from "../../src/compose/source/catalog.ts";
import { type SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { OriginRef, type OriginResolver } from "../../src/diagnostics.ts";
import { Message, Response, TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { ContractOrigins, ValueContract } from "../../src/nodes/contracts.ts";
import { ValueKind } from "../../src/typesys/values.ts";
import { EvaluationFailure } from "../../src/runtime/eval_node.ts";
import { runFlow } from "../../src/run.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { drive, NO_ENV } from "../helpers/run_state_fixtures.ts";
import { assembleText, loadError } from "../helpers/declarations.ts";

const NESTED_OUTPUT_FLOW = `id: nested-output-contract
name: nested-output-contract
nodes:
  emit:
    kind: code
    code: |-
      return {"profile": {"age": "bad", "name": "Ada"}, "ok": true}
    output:
      profile:
        age: !int
        name: !str
      ok: !bool
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

/** Where an origin resolves: an assembly's catalog builder or a loaded flow's resolver. */
type Sites = SourceCatalogBuilder | OriginResolver;

/** Resolve `origin` through `sites`. */
const resolveIn = (sites: Sites, origin: OriginRef) =>
  "resolveRegistered" in sites ? sites.resolveRegistered(origin) : sites.resolve(origin);

/** Expect `origin` to resolve to exactly `cursor`'s value occurrence. */
function expectSite(sites: Sites, origin: OriginRef | null | undefined, cursor: SourceCursor): void {
  expect(origin).toBeInstanceOf(OriginRef);
  const frame = resolveIn(sites, origin!);
  expect(frame).not.toBeNull();
  expect(frame!.text).toBe(cursor.document!.text);
  expect(frame!.range).toEqual(cursor.range!);
}

/** Return `contract`'s nested origins by path, the path joined with `.`. */
function entriesOf(contract: ValueContract): Map<string, OriginRef> {
  return new Map(contract.origins.entries.map(([path, origin]) => [path.join("."), origin]));
}

/** Expect the declaration root and the nested profile fields to keep their sites. */
function expectProfileSites(sites: Sites, contract: ValueContract, output: SourceCursor): void {
  const entries = entriesOf(contract);
  expectSite(sites, contract.origins.default, output);
  expectSite(sites, entries.get("profile"), output.child("profile"));
  expectSite(sites, entries.get("profile.age"), output.child("profile").child("age"));
}

/** Return `node`'s exact output contract. */
function exactContract(node: { outputContract: unknown }): ValueContract {
  expect((node.outputContract as object | null)?.constructor).toBe(ValueContract);
  return node.outputContract as ValueContract;
}

/** Return the child flow CALL node `nodeId` of `loaded` bakes in. */
const childOf = (loaded: ComposedFlow, nodeId: string) =>
  (loaded.compiled.nodes.get(nodeId) as CallNode).child as CompiledFlow;

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

test("an invalid output type precedes a CODE syntax error", () => {
  const text = `id: invalid-code-before-output
name: invalid-code-before-output
nodes:
  broken:
    kind: code
    code: |-
      return (
    output: !DefinitelyNotAType
output: \${broken.output}
`;
  const document = SourceDocument.parse(text, "precedence.yaml");
  const outputCursor = document.root.child("nodes").child("broken").child("output");
  // Nothing in assembly calls `check` today; this guards the order once the loader awaits it.
  const check = spyOn(CodeNode.prototype, "check");

  try {
    const error = loadError(() => assembleText(text, "precedence.yaml"));

    expect(error.diagnostic.message.startsWith("bad type expression 'DefinitelyNotAType': ")).toBe(true);
    expect(error.diagnostic.frames.at(-1)!.range).toEqual(outputCursor.range!);
    expect(check).not.toHaveBeenCalled();
  } finally {
    check.mockRestore();
  }
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
    age: !int
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
    kind: tool
    tool_id: ask
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

test("formatting-only relocation preserves output origins and fingerprint", async () => {
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
        age: !int
        name: !str
      ok: !bool
output: \${emit.output}
`;
  const outputOf = (text: string) =>
    SourceDocument.parse(text, "relocated.yaml").root.child("nodes").child("emit").child("output");
  const baseline = contractOf(NESTED_OUTPUT_FLOW, "emit", "relocated.yaml");
  const relocated = contractOf(relocatedText, "emit", "relocated.yaml");
  const baselineOutput = outputOf(NESTED_OUTPUT_FLOW);
  const relocatedOutput = outputOf(relocatedText);

  expect(relocated.contract.origins).toEqual(baseline.contract.origins);
  const fingerprint = async (text: string) =>
    flowFingerprint((await loadFlow(text, { sourceLabel: "relocated.yaml" })).compiled);
  expect(await fingerprint(relocatedText)).toBe(await fingerprint(NESTED_OUTPUT_FLOW));
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
            age: !int
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

test("a multi-node def's output contract uses the nested node site", async () => {
  // A full definition keeps its leaf node's physical output declaration.
  const text = `id: full-def-output-contract
name: full-def-output-contract
defs:
  child:
    nodes:
      emit:
        kind: code
        code: |-
          return {"profile": {"age": 7}}
        output:
          profile:
            age: !int
    output: \${emit.output}
nodes:
  invoke:
    kind: call
    call: child
output: \${invoke.output}
`;
  const output = SourceDocument.parse(text, "full-def.yaml")
    .root.child("defs")
    .child("child")
    .child("nodes")
    .child("emit")
    .child("output");
  const loaded = await loadFlow(text, { sourceLabel: "full-def.yaml" });
  const emit = childOf(loaded, "invoke").nodes.get("emit")!;

  expectProfileSites(loaded.originResolver!, exactContract(emit), output);
});

test("a flat def's output contract uses the projected def output site", async () => {
  // A compact definition projects its leaf contract to the def's output field.
  const text = `id: flat-def-output-contract
name: flat-def-output-contract
defs:
  child:
    kind: code
    code: |-
      return {"profile": {"age": 7}}
    output:
      profile:
        age: !int
nodes:
  invoke:
    kind: call
    call: child
output: \${invoke.output}
`;
  const output = SourceDocument.parse(text, "flat-def.yaml").root.child("defs").child("child").child("output");
  const loaded = await loadFlow(text, { sourceLabel: "flat-def.yaml" });
  const child = childOf(loaded, "invoke").nodes.get("child")!;

  expectProfileSites(loaded.originResolver!, exactContract(child), output);
});

test("an external alias rebases every output contract origin", async () => {
  // An alias view owns distinct output keys resolving to canonical child sites.
  const childText = `id: external-output-child
name: external-output-child
nodes:
  emit:
    kind: code
    code: |-
      return {"profile": {"age": 7}}
    output:
      profile:
        age: !int
output: \${emit.output}
`;
  const childDocument = SourceDocument.parse(childText, "child.yaml");
  const child = await loadFlow(childText, { sourceLabel: "child.yaml" });
  const parentText = `id: external-output-parent
name: external-output-parent
uses:
  imported: child
nodes:
  invoke:
    kind: call
    call: imported
output: \${invoke.output}
`;
  const parent = await loadFlow(parentText, { sourceLabel: "parent.yaml", childResolver: () => child });
  const canonical = exactContract(child.compiled.nodes.get("emit")!);
  const invoke = parent.compiled.nodes.get("invoke")!;
  const rebased = exactContract(childOf(parent, "invoke").nodes.get("emit")!);
  const originsOf = (contract: ValueContract) => [
    contract.origins.default,
    ...contract.origins.entries.map(([, origin]) => origin),
  ];
  const canonicalOrigins = originsOf(canonical);
  const rebasedOrigins = originsOf(rebased);

  expect(exactContract(invoke).origins).toEqual(rebased.origins);
  expect(rebasedOrigins).toHaveLength(canonicalOrigins.length);
  canonicalOrigins.forEach((canonicalOrigin, index) => {
    const rebasedOrigin = rebasedOrigins[index];
    expect(canonicalOrigin).toBeInstanceOf(OriginRef);
    expect(rebasedOrigin).toBeInstanceOf(OriginRef);
    expect(rebasedOrigin).not.toEqual(canonicalOrigin);
    expect(parent.originResolver!.resolve(rebasedOrigin!)).toEqual(child.originResolver!.resolve(canonicalOrigin!));
  });
  expectProfileSites(parent.originResolver!, rebased, childDocument.root.child("nodes").child("emit").child("output"));
});

const dir = mkdtempSync(join(tmpdir(), "yian-output-contract-origins-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

test.each([
  [
    "root output failure selects the contract root",
    "root.yaml",
    `id: root-output-contract
name: root-output-contract
nodes:
  emit:
    kind: code
    code: |-
      return {"not": "an integer"}
    output: !int
output: \${emit.output}
`,
  ],
  [
    "scalar list output failure falls back to the contract root",
    "scalar-list.yaml",
    `id: scalar-list-output
name: scalar-list-output
typedefs:
  Profile:
    age: int
nodes:
  emit:
    kind: code
    code: |-
      return [{"age": "bad"}]
    output: !list(Profile)
output: \${emit.output}
`,
  ],
])("%s", async (_name, label, text) => {
  const loaded = await loadFlow(text, { sourceLabel: label });
  const contract = exactContract(loaded.compiled.nodes.get("emit")!);
  expect(contract.origins.entries).toEqual([]);

  const result = await runFlow(loaded, {}, { env: hostEnv(dir) });

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("contract.output");
  expect(result.failure!.trace).toEqual([contract.origins.default!]);
  const expected = SourceDocument.parse(text, label).root.child("nodes").child("emit").child("output");
  expect(loaded.diagnose(result.failure!).frames.at(-1)!.range).toEqual(expected.range!);
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
        7: !int
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
    output: !str
  code_leaf:
    kind: code
    code: return "ok"
    output: !str
  ask_leaf:
    kind: tool
    tool_id: ask
    input:
      prompt: continue?
    output: !str
output: \${code_leaf.output}
`;
  const nodes = SourceDocument.parse(text, "leaves.yaml").root.child("nodes");
  const { flow, builder } = assembleText(text, "leaves.yaml");

  for (const nodeId of ["agent_leaf", "code_leaf", "ask_leaf"]) {
    const contract = flow.compiled.nodes.get(nodeId)!.outputContract as ValueContract;
    expect(contract.constructor).toBe(ValueContract);
    expect(contract.origins.entries).toEqual([]);
    expectSite(builder, contract.origins.default, nodes.child(nodeId).child("output"));
  }
});

test("an agent structured failure selects the exact authored output field", async () => {
  const text = `id: agent-output-origin
name: agent-output-origin
nodes:
  writer:
    kind: agent
    prompt: write a profile
    retries: 0
    llm_config:
      provider: openai
      model: gpt-4o
    output:
      profile:
        age: !int
output: \${writer.output}
`;
  const { flow, builder } = assembleText(text, "agent-output.yaml");
  const writer = flow.compiled.nodes.get("writer")!;
  const ageOrigin = entriesOf(writer.outputContract as ValueContract).get("profile.age")!;
  const config = { provider: "openai", model: "gpt-4o" };
  const reply = new Message({ role: "assistant", parts: [new TextPart('{"profile": {"age": "not-an-integer"}}')] });
  const model = new ScriptedModel([new Response({ message: reply, finishReason: "stop" })], config);

  const outcome = await drive(writer, { models: { env: NO_ENV, createModel: () => model } });

  expect(outcome).toBeInstanceOf(EvaluationFailure);
  const { failure } = outcome as EvaluationFailure;
  expect([failure.code, failure.causeType]).toEqual(["agent.output_invalid", "OutputInvalidError"]);
  expect(failure.trace.at(-1)).toBe(ageOrigin);
  const age = SourceDocument.parse(text, "agent-output.yaml")
    .root.child("nodes")
    .child("writer")
    .child("output")
    .child("profile")
    .child("age");
  expectSite(builder, failure.trace.at(-1), age);
});
