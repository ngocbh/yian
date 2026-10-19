/**
 * Value templates and their nested typed bindings in flow manifests: rows of
 * `tests/engine/test_value_template_manifest.py`. A template encodes apart from every authored
 * literal, a nested binding keeps its declared type, and every part of a template is identity
 * (credential-named slots too, as for literals).
 *
 * Ported: `test_a_value_template_encodes_as_a_tagged_tree`,
 * `test_a_nested_binding_encodes_its_source_and_declared_type`,
 * `test_template_identity_differs_from_every_authored_literal`,
 * `test_a_nested_binding_is_not_a_bare_string_or_untyped_template_part`,
 * `test_template_and_nested_binding_origins_are_not_identity`,
 * `test_template_parts_change_identity`, `test_a_nested_binding_must_have_only_canonical_fields`,
 * `test_a_template_under_a_credential_name_keeps_every_part_as_identity`,
 * `test_a_plain_mapping_under_a_credential_key_is_identity`,
 * `test_a_flow_output_template_encodes_as_a_tagged_tree`,
 * `test_llm_config_values_are_fingerprinted_as_written`,
 * `test_a_tag_on_an_llm_config_value_changes_the_fingerprint`,
 * `test_whole_record_llm_config_values_are_fingerprinted` and
 * `test_a_flow_llm_config_value_is_fingerprinted_as_written`.
 *
 * Changed:
 * - `test_a_flow_output_template_encodes_as_a_tagged_tree` is hand-built instead of `load_flow`:
 *   `CompiledFlow.fromParts` builds the one-step flow with each output. The `llm_config` rows build
 *   Python's `_LLM_FLOW` the same way (`llmFingerprint`), with each layer as the loader leaves it:
 *   a record, a `ValueTemplate` whose values are `Binding`s, or a `Binding` for the whole record
 *   (a tag is the binding's contract).
 * - Ints in values are `bigint`s. Binding is frozen, so the unknown-field row forges one
 *   (`Object.assign(Object.create(Binding.prototype), binding, {extra: 1})`); Python's
 *   `ValueError` is a `RangeError` and the message lists the field as a JSON array (`["extra"]`
 *   where Python prints `['extra']`).
 * - `json.dumps(manifest)` is `JSON.stringify`.
 *
 * Later (T3-22): the loader rows `test_a_flow_without_templates_keeps_its_fingerprint` and
 * `test_a_literal_llm_config_flow_keeps_its_fingerprint` (pinned fingerprints of loaded flows), and
 * `test_a_literal_llm_config_with_a_doubled_dollar_reads_unescaped` (the loader's `$$` reading).
 */

import { expect, test } from "bun:test";

import { flowFingerprint, flowManifest } from "../../src/compile/manifest.ts";
import { CompiledFlow, Edge, END_ID, Flow, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { OriginRef } from "../../src/diagnostics.ts";
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
  return CompiledFlow.fromParts(
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
 * Fingerprint Python's `_LLM_FLOW`: inputs `m`, `n` and `cfg`, and one AGENT `a` with the node layer
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
    CompiledFlow.fromParts(
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
