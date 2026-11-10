/**
 * Value templates and their nested typed bindings in flow manifests and fingerprints. A template
 * encodes apart from every authored literal, a nested binding keeps its declared type, and every
 * part of a template is identity (credential-named slots too, as for literals).
 */

import { expect, test } from "bun:test";

import { flowFingerprint, flowManifest } from "../../src/compile/manifest.ts";
import { CompiledFlow, Edge, END_ID, Flow, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { Verbatim } from "../../src/expr/template.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import type { Node } from "../../src/nodes/base.ts";
import { Binding, ContractOrigins, Parameter, ValueContract, ValueTemplate } from "../../src/nodes/contracts.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { Type, ValueKind } from "../../src/typesys/index.ts";
import type { JsonObject, JsonValue } from "../../src/typesys/json.ts";

const INTEGER = Type.scalar(ValueKind.INTEGER);
const STRING = Type.scalar(ValueKind.STRING);
const OBJECT = new Type({ kind: ValueKind.OBJECT });

/** Check `fn` throws exactly `cls` with exactly `message`. */
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

/** Build a START -> CODE -> END fragment whose CODE node takes `wiring`. */
function fragment(wiring: Record<string, unknown>): Flow {
  const start = new StartNode("entry", { params: [] });
  const body = new CodeNode("body", { code: 'return inputs["x"]' });
  body.params = Object.keys(wiring).map((name) => new Parameter(name));
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
        ["body", new Map(Object.entries(wiring).map(([name, source]) => [name, new Binding(source)]))],
        ["terminal", new Map([["result", new Binding("${body.output}")]])],
      ]),
      startId: "entry",
      endId: "terminal",
    },
  );
}

/** Return one parameter's canonical wiring value from the body node. */
function wired(flow: Flow, param: string): JsonObject {
  const wiring = (flowManifest(flow).wiring as JsonObject).body as JsonObject;
  const matches = (wiring.entries as JsonObject[]).filter((entry) => {
    const key = entry.key as JsonObject;
    return key.type === "str" && key.value === param;
  });
  expect(matches).toHaveLength(1);
  return matches[0]!.value as JsonObject;
}

/** The first entry value of a template's encoded tree. */
function firstTreeValue(encoded: JsonObject): JsonValue {
  return ((encoded.tree as JsonObject).entries as JsonObject[])[0]!.value!;
}

test("a value template encodes as a tagged tree", () => {
  const flow = fragment({ x: new ValueTemplate({ a: "${n.output}", b: [1n] }) });

  const encoded = wired(flow, "x");

  expect(encoded.type).toBe("value_template");
  expect((encoded.tree as JsonObject).type).toBe("mapping");
  expect((encoded.tree as JsonObject).entries).toContainEqual({
    key: { type: "str", value: "a" },
    value: { type: "str", value: "${n.output}" },
  });
});

test("a nested binding encodes its source and declared type", () => {
  const typed = fragment({
    x: new ValueTemplate({ a: new Binding("${n.output}", { contract: new ValueContract(INTEGER) }) }),
  });
  const untyped = fragment({ x: new ValueTemplate({ a: new Binding("${n.output}") }) });

  const typedValue = firstTreeValue(wired(typed, "x")) as JsonObject;
  const untypedValue = firstTreeValue(wired(untyped, "x"));

  expect(typedValue.type).toBe("binding");
  expect(typedValue.source).toEqual({ type: "str", value: "${n.output}" });
  expect((typedValue.contract as JsonObject).kind).toBe("int");
  expect(untypedValue).toEqual({
    contract: null,
    source: { type: "str", value: "${n.output}" },
    type: "binding",
  });
  expect(flowFingerprint(typed)).not.toBe(flowFingerprint(untyped));
});

test("template identity differs from every authored literal", () => {
  const template = fragment({ x: new ValueTemplate({ a: "${n.output}" }) });
  const plain = fragment({ x: { a: "${n.output}" } });
  const shaped = fragment({ x: wired(template, "x") });

  const fingerprints = new Set([template, plain, shaped].map(flowFingerprint));

  expect(fingerprints.size).toBe(3);
});

test("verbatim text encodes apart from template text, alone or in a template", () => {
  const verbatim = fragment({ x: new Verbatim("a$$b") });
  const template = fragment({ x: "a$$b" });
  const nested = fragment({ x: new ValueTemplate({ a: new Verbatim("a$$b") }) });

  expect(wired(verbatim, "x")).toEqual({ type: "verbatim", value: "a$$b" });
  expect(firstTreeValue(wired(nested, "x"))).toEqual({ type: "verbatim", value: "a$$b" });
  expect(flowFingerprint(verbatim)).not.toBe(flowFingerprint(template));
});

test("a nested binding is not a bare string or an untyped template part", () => {
  const bare = fragment({ x: new ValueTemplate({ a: "${n.output}" }) });
  const nested = fragment({ x: new ValueTemplate({ a: new Binding("${n.output}") }) });

  expect(flowFingerprint(bare)).not.toBe(flowFingerprint(nested));
});

test("template and nested binding origins are not identity", () => {
  /** A template whose leaf, nested binding and contract origins are all keyed by `site`. */
  const template = (site: string) =>
    new ValueTemplate(
      {
        a: "${n.output}",
        b: new Binding("${m.output}", {
          origin: new OriginRef(`${site}/b`),
          contract: new ValueContract(INTEGER, new ContractOrigins({ default: new OriginRef(`${site}/c`) })),
        }),
      },
      [[["a"], new OriginRef(`${site}/a`)]],
    );

  const first = fragment({ x: template("first") });
  const second = fragment({ x: template("second") });

  expect(flowFingerprint(first)).toBe(flowFingerprint(second));
});

test("template parts change identity", () => {
  const first = fragment({ x: new ValueTemplate({ a: "${n.output}", b: [1n, "${m.output}"] }) });
  const changed = fragment({ x: new ValueTemplate({ a: "${n.output}", b: [1n, "${k.output}"] }) });

  expect(flowFingerprint(first)).not.toBe(flowFingerprint(changed));
});

test("a nested binding must have only canonical fields", () => {
  const nested = Object.assign(Object.create(Binding.prototype), new Binding("${n.output}"), { extra: 1 });
  const flow = fragment({ x: new ValueTemplate({ a: nested }) });

  expectError(
    () => flowFingerprint(flow),
    RangeError,
    'flow.wiring.body.values[0].value.values[0]: unknown noncanonical field(s): ["extra"]',
  );
});

test("a template under a credential name keeps every part as identity", () => {
  const first = fragment({ password: new ValueTemplate({ primary: "${input.first}", note: "first", pin: 1n }) });
  const editedText = fragment({
    password: new ValueTemplate({ primary: "${input.first}", note: "second", pin: 1n }),
  });
  const editedNumber = fragment({
    password: new ValueTemplate({ primary: "${input.first}", note: "first", pin: 2n }),
  });
  const rerouted = fragment({ password: new ValueTemplate({ primary: "${input.other}", note: "first", pin: 1n }) });

  const text = JSON.stringify(flowManifest(first));
  expect(text).not.toContain("redacted");
  expect(text).toContain("input.first");
  const fingerprints = new Set([first, editedText, editedNumber, rerouted].map(flowFingerprint));
  expect(fingerprints.size).toBe(4);
});

test("a plain mapping under a credential key is identity", () => {
  const first = fragment({ request: { headers: { Authorization: "Bearer ${input.a}" } } });
  const edited = fragment({ request: { headers: { Authorization: "Bearer ${input.b}" } } });

  expect(flowFingerprint(first)).not.toBe(flowFingerprint(edited));
  expect(JSON.stringify(flowManifest(first))).not.toContain("redacted");
});

/** A one-step CompiledFlow (`step` returns 1) whose one declared output has `source`. */
function outputsFlow(source: unknown): CompiledFlow {
  const step = new CodeNode("step", { code: "return 1" });
  step.outputContract = new ValueContract(INTEGER);
  return new CompiledFlow(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID, { params: [] })],
      ["step", step],
      [END_ID, EndNode.value(END_ID, { outputName: "result" })],
    ]),
    [
      new Edge(`${START_ID}->step`, START_ID, "step", { ordering: true }),
      new Edge(`step->${END_ID}`, "step", END_ID, { inputGroup: "result" }),
    ],
    {
      wiring: new Map([
        [START_ID, new Map()],
        ["step", new Map()],
        [END_ID, new Map([["result", "${step.output}"]])],
      ]),
      outputs: [new FlowOutput("result", source)],
    },
  );
}

test("a flow output template encodes as a tagged tree", () => {
  const changed = outputsFlow(new Binding(new ValueTemplate({ n: "${step.output}" })));
  const plain = outputsFlow(new Binding({ n: "${step.output}" }));

  const manifest = flowManifest(changed);

  expect(((manifest.outputs as JsonObject[])[0]!.from as JsonObject).type).toBe("value_template");
  expect(flowFingerprint(changed)).not.toBe(flowFingerprint(plain));
});

type LlmLayer = Record<string, unknown> | ValueTemplate | Binding;

/**
 * Fingerprint the LLM flow (`llmFlow`): inputs `m`, `n` and `cfg`, and one AGENT `a` with the node layer
 * `nodeLayer`, under the flow layer `flowLayer`.
 */
function llmFingerprint(
  nodeLayer: LlmLayer,
  flowLayer: LlmLayer = { provider: "anthropic", anthropic: { max_tokens: 900n } },
): string {
  const inputs = [
    new Parameter("m", { contract: new ValueContract(STRING) }),
    new Parameter("n", { contract: new ValueContract(STRING) }),
    new Parameter("cfg", { contract: new ValueContract(OBJECT) }),
  ];
  return flowFingerprint(
    new CompiledFlow(
      new Map<string, Node>([
        [START_ID, new StartNode(START_ID, { params: inputs })],
        ["a", new AgentNode("a", { prompt: "Go.", llmConfig: nodeLayer })],
        [END_ID, EndNode.value(END_ID, { outputName: "result" })],
      ]),
      [new Edge(`a->${END_ID}`, "a", END_ID, { inputGroup: "result" })],
      {
        wiring: new Map([
          [START_ID, new Map()],
          ["a", new Map()],
          [END_ID, new Map([["result", "${a.output}"]])],
        ]),
        flowLlmConfig: flowLayer,
        flowId: "f",
      },
    ),
  );
}

/** The layer `{key: ${input.<name>}}`, with `contract` on the value when given. */
function valueLayer(key: string, name: string, contract: ValueContract | null = null): ValueTemplate {
  return new ValueTemplate({ [key]: new Binding(`\${input.${name}}`, { contract }) });
}

test("llm config values are fingerprinted as written", () => {
  expect(llmFingerprint(valueLayer("model", "m"))).not.toBe(llmFingerprint(valueLayer("model", "n")));
  expect(llmFingerprint(valueLayer("model", "m"))).not.toBe(llmFingerprint({ model: "m" }));
});

test("a tag on an llm config value changes the fingerprint", () => {
  expect(llmFingerprint(valueLayer("model", "m", new ValueContract(STRING)))).not.toBe(
    llmFingerprint(valueLayer("model", "m")),
  );
});

test("whole record llm config values are fingerprinted", () => {
  const whole = new Binding("${input.cfg}");
  const tagged = new Binding("${input.cfg}", { contract: new ValueContract(OBJECT) });

  const node = llmFingerprint(whole);
  expect(node).not.toBe(llmFingerprint(tagged));
  expect(node).not.toBe(llmFingerprint(valueLayer("model", "m")));
  const flow = llmFingerprint({ model: "m" }, whole);
  expect(flow).not.toBe(llmFingerprint({ model: "m" }, tagged));
  expect(flow).not.toBe(llmFingerprint({ model: "m" }, { model: "x" }));
});

test("a flow llm config value is fingerprinted as written", () => {
  expect(llmFingerprint({ model: "m" }, valueLayer("provider", "m"))).not.toBe(
    llmFingerprint({ model: "m" }, valueLayer("provider", "n")),
  );
});

// ---------- loaded flows ----------

const UNTYPED_FLOW = `
id: plain
name: Plain
input:
  x: !int
  token: !str
nodes:
  step:
    kind: code
    input:
      x: \${input.x}
      request: {headers: {Authorization: "Bearer $\${input.token}"}, items: [1, "a"]}
      password: "\${input.token}"
    output: int
    code: |
      return inputs["x"] + 1
output: \${step.output}
`;

test("a flow without templates keeps its fingerprint", async () => {
  // A template-free flow must not move; its manifest writes the inline CODE as
  // `{inline: <text>}`.
  expect(flowFingerprint((await loadFlow(UNTYPED_FLOW)).compiled)).toBe(
    "e1abe2e3f331a207810eb89af00120062b61906b7a98c03c0a43240157a96762",
  );
});

/** The LLM flow's text, with the node and flow `llm_config` layers given. */
const llmFlow = (nodeLayer: string, flowLayer: string) => `id: f
name: f
input: {m: !str , n: !str , cfg: !object }
${flowLayer}
nodes:
  a:
    kind: agent
    prompt: Go.
    llm_config: ${nodeLayer}
output: \${a.output}
`;

test("a literal llm config flow keeps its fingerprint", async () => {
  // The fingerprint this flow had before `llm_config` read values.
  const text = `id: f
name: f
llm_config: {provider: anthropic, anthropic: {max_tokens: 900}}
nodes:
  a:
    kind: agent
    prompt: Go.
    llm_config: {model: claude-x, inherit: true, anthropic: {temperature: 0.2}}
output: \${a.output}
`;
  expect(flowFingerprint((await loadFlow(text)).compiled)).toBe(
    "0f2c73ac336010ede82bbbc2f9a6584bbe70a5c7453285d4a2eedbdb81e4376e",
  );
});

test("a literal llm config with a doubled dollar reads unescaped", async () => {
  // A literal layer is read as a template would be, so `$$` is one `$` and the fingerprint is that
  // of the unescaped text.
  const flowLayer = "llm_config: {provider: anthropic, anthropic: {max_tokens: 900}}";
  const fingerprint = async (nodeLayer: string) =>
    flowFingerprint((await loadFlow(llmFlow(nodeLayer, flowLayer))).compiled);
  expect(await fingerprint('{model: "a$$b"}')).toBe(await fingerprint('{model: "a$b"}'));

  const flow = await loadFlow(llmFlow('{model: "a$$b"}', ""));
  expect(((flowManifest(flow.compiled).nodes as JsonObject).a as JsonObject).own_llm_config).toEqual({
    entries: [{ key: { type: "str", value: "model" }, value: { type: "str", value: "a$b" } }],
    type: "mapping",
  });
});
