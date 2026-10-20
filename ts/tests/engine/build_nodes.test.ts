/**
 * Per-kind descriptor builds: rows of `tests/engine/test_build_nodes.py`.
 *
 * Ported: the leaf-build rows (`test_agent_node_built_with_type_and_bindings`,
 * `test_agent_node_object_output`, `test_agent_invalid_llm_config_uses_the_config_field_range`,
 * `test_agent_non_mapping_llm_config_is_a_ranged_load_error`, `test_agent_node_knobs`,
 * `test_node_name_becomes_title`, `test_code_node_built`, `test_tool_node_built_with_args_bindings`
 * and `test_output_contract_resolves_registry_name`), and
 * `test_call_loop_descriptors_need_a_resolver` (CALL, MAP and LOOP; the LOOP count is `3n`, as
 * YAML reads it).
 *
 * Changed: the seeds are read from their harvested cases (`load-core/seed_00_hello_agent`,
 * `load-agent/seed_01_structured_agent`, `load-leaves/seed_08_tool_news`,
 * `load-agent/seed_14_agent_tools`); seed 01's CODE reference is the TS
 * `./fixtures/seeds.ts:one_line_summary`, which the node keeps as its `code` (Python's `ref`), and
 * YAML ints are `bigint`s. Python's `None` resolver fails as CPython's `'NoneType' object is not
 * callable`; TS's `resolveChild` refuses a `null` resolver with its own text, so the rows pin the
 * message's prefix. The classes come from `compose/nodes/*.ts`, where Python imports them from
 * `compose.parse.file`.
 */

import { afterEach, expect, test } from "bun:test";

import { LoadDiagnostics } from "../../src/compose/helpers/errors.ts";
import { BuildContext, type NodeDescriptor, siteKey } from "../../src/compose/nodes/base.ts";
import { CallDescriptor } from "../../src/compose/nodes/call.ts";
import { LoopDescriptor } from "../../src/compose/nodes/loop.ts";
import { MapDescriptor } from "../../src/compose/nodes/map.ts";
import { flowNodesCursor, parseFile, parseSourceDocument } from "../../src/compose/parse/file.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { type SourceDocument, SourceCursor } from "../../src/compose/source/index.ts";
import { forget } from "../../src/defaults.ts";
import type { Node } from "../../src/nodes/base.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { readTypedefs, type TypeRegistry } from "../../src/typesys/types.ts";
import { ValueKind } from "../../src/typesys/values.ts";
import { caseDocument, loadError } from "../compose/declarations.ts";

afterEach(() => forget());

/** Build a leaf descriptor and return its node and wiring. */
function build(
  desc: NodeDescriptor,
  registry: TypeRegistry = new Map(),
  diagnostics = new LoadDiagnostics(),
): [Node, Map<string, unknown>] {
  const result = desc.build(new BuildContext({ registry, diagnostics }));
  return [result.node, result.wiring];
}

/** Parse the `nodes:` of a parsed flow document as the loader does. */
const nodesOfDocument = (document: SourceDocument) => parseNodes(flowNodesCursor(document, parseFile(document).nodes));

/** Parse the `nodes:` of seed case `id`. */
const seedNodes = (id: string) => nodesOfDocument(caseDocument(id).document);

/** Return the exact contract `node` carries, failing on anything else. */
function exactContract(node: Node): ValueContract {
  expect(node.outputContract?.constructor).toBe(ValueContract);
  return node.outputContract as ValueContract;
}

// ---------- agent ----------

test("an AGENT node is built with its type and bindings", () => {
  // Seed 00: a scalar (str) output, one ${input.topic} sink binding.
  const [node, wiring] = build(seedNodes("load-core/seed_00_hello_agent").get("note")!);

  expect(node).toBeInstanceOf(AgentNode);
  expect(node.id).toBe("note");
  expect(exactContract(node).type.kind).toBe(ValueKind.STRING);
  // The node-side signature is the parameter names; the flow owns the source.
  expect(node.params!.map((param) => param.name)).toEqual(["topic"]);
  expect(wiring).toEqual(new Map([["topic", "${input.topic}"]]));
  // Body-node sink parameters are exact, source-neutral, and untyped.
  expect(node.params![0]!.constructor).toBe(Parameter);
  expect(node.params![0]!.contract).toBeNull();
});

test("an AGENT node's record output", () => {
  // Seed 01's score: a record {rating, rationale} output.
  const [node] = build(seedNodes("load-agent/seed_01_structured_agent").get("score")!);

  expect(node).toBeInstanceOf(AgentNode);
  const typ = exactContract(node).type;
  expect(typ.kind).toBe(ValueKind.OBJECT);
  expect(typ.fields!.get("rating")!.kind).toBe(ValueKind.NUMBER);
  expect(typ.fields!.get("rationale")!.kind).toBe(ValueKind.STRING);
});

/** Return a flow whose AGENT `note` has the `llm_config:` lines `llmConfig`. */
const badConfigFlow = (llmConfig: string) =>
  "id: bad-config\nname: bad\n" +
  "nodes:\n  note:\n    kind: agent\n" +
  `    llm_config:${llmConfig}\n` +
  "    prompt: hi\n" +
  "output: ${note.output}\n";

/** Build `text`'s AGENT `note`, expecting a load error holding `message` at its `llm_config:`. */
function llmConfigErrorAt(text: string, message: string): void {
  const document = parseSourceDocument(text, "flow.yaml");
  const desc = nodesOfDocument(document).get("note")!;

  const error = loadError(() => build(desc, new Map(), LoadDiagnostics.fromDocument(document)));

  expect(error.message).toContain(message);
  const frame = error.diagnostic.frames[0]!;
  expect(frame.range).toEqual(desc.fieldRanges.get(siteKey("field", "llm_config"))!);
  expect(frame.text).toBe(text);
}

test("an invalid llm_config uses the config field's range", () => {
  llmConfigErrorAt(badConfigFlow("\n      provider: bogus_provider"), "llm_config");
});

test.each(["bogus", "1", "false", "[bogus]"])("a non-mapping llm_config is a ranged load error: %s", (value) => {
  llmConfigErrorAt(badConfigFlow(` ${value}`), "must be a mapping");
});

test("an AGENT node's knobs", () => {
  // Seed 14: tools and llm_config carried onto the AgentNode.
  const [node] = build(seedNodes("load-agent/seed_14_agent_tools").get("reviewer")!);

  expect(node).toBeInstanceOf(AgentNode);
  const agent = node as AgentNode;
  expect(agent.tools).toEqual(["get_web_data", "web_search", "ask_user"]);
  // llm_config is a plain record end to end.
  expect(agent.llmConfig).toEqual({ provider: "anthropic", model: "claude-opus-4-8", anthropic: { temperature: 0.3 } });
  expect(agent.prompt).toStartWith("Research ${topic}");
  expect(exactContract(node).type.fields!.get("confidence")!.kind).toBe(ValueKind.NUMBER);
});

test("node_name becomes the title", () => {
  // node_name -> the Node title; absent -> null. A constructed descriptor pins the mapping
  // (the seed leaf nodes carry no node_name).
  const [note] = build(seedNodes("load-core/seed_00_hello_agent").get("note")!);
  expect(note.title).toBeNull();

  const nodes = parseNodes(
    SourceCursor.virtual({
      n: { kind: "code", node_name: "Format verdict", inputs: { x: "${input.x}" }, outputs: "str", code: "m:f" },
    }),
  );
  const [named] = build(nodes.get("n")!);
  expect(named.title).toBe("Format verdict");
});

// ---------- code ----------

test("a CODE node is built", () => {
  // Seed 01's verdict: a CODE node with a file:function reference.
  const [node, wiring] = build(seedNodes("load-agent/seed_01_structured_agent").get("verdict")!);

  expect(node).toBeInstanceOf(CodeNode);
  expect((node as CodeNode).code).toBe("./fixtures/seeds.ts:one_line_summary");
  expect((node as CodeNode).mode).toBe("reference");
  expect(exactContract(node).type.kind).toBe(ValueKind.STRING);
  expect(new Set(node.params!.map((param) => param.name))).toEqual(new Set(["rating", "rationale"]));
  expect(wiring).toEqual(
    new Map([
      ["rating", "${score.output.rating}"],
      ["rationale", "${score.output.rationale}"],
    ]),
  );
});

// ---------- tool ----------

test("a TOOL node is built with its argument bindings", () => {
  // Seed 08's news: tool_id and untyped inputs.
  const [node, wiring] = build(seedNodes("load-leaves/seed_08_tool_news").get("news")!);

  expect(node).toBeInstanceOf(ToolNode);
  expect((node as ToolNode).toolId).toBe("get_facts");
  expect(node.outputContract).toBeNull(); // no `output:` declared
  expect(new Set(node.params!.map((param) => param.name))).toEqual(new Set(["symbol", "limit"]));
  expect(wiring).toEqual(
    new Map<string, unknown>([
      ["symbol", "${input.topic}"],
      ["limit", 10n],
    ]),
  );
  // Inputs are exact untyped sink parameters.
  expect(node.params!.every((param) => param.constructor === Parameter && param.contract === null)).toBe(true);
});

// ---------- call, map and loop ----------

test.each([
  ["call", new CallDescriptor({ id: "c", call: "child" }), "call node 'c'", "child"],
  ["map", new MapDescriptor({ id: "m", call: "child", over: "${input.xs}" }), "call node 'm'", "child"],
  ["loop", new LoopDescriptor({ id: "p", call: "body", times: 3n }), "call node 'p'", "body"],
])("a %s descriptor needs a resolver", (_kind, desc, subject, callable) => {
  // A leaf-only context (no resolver) can't resolve the callable: a LoadError, not a TypeError.
  const error = loadError(() => desc.build(new BuildContext({ registry: new Map() })));

  expect(error.message).toStartWith(`${subject}: could not resolve callable '${callable}'`);
});

// ---------- registry-resolved output type ----------

test("an output contract resolves a registry name", () => {
  const registry = readTypedefs({ Rating: { category: "str", score: "float" } });
  // An AGENT descriptor whose output names a registry record.
  const nodes = parseNodes(
    SourceCursor.virtual({
      synth: { kind: "agent", inputs: { topic: "${input.topic}" }, outputs: "Rating", prompt: "x" },
    }),
  );

  const [node] = build(nodes.get("synth")!, registry);

  expect(node).toBeInstanceOf(AgentNode);
  const typ = exactContract(node).type;
  expect(typ.kind).toBe(ValueKind.OBJECT);
  expect(typ.fields!.get("category")!.kind).toBe(ValueKind.STRING);
});
