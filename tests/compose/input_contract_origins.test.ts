/** Compose provenance contracts for authored flow-input declarations. */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import { flowFingerprint } from "../../src/compile/manifest.ts";
import { type CompiledFlow, START_ID } from "../../src/compile/model.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { type SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { OriginRef, type OriginResolver } from "../../src/diagnostics.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import { ContractOrigins, Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import type { StartNode } from "../../src/nodes/start/node.ts";
import { runFlow } from "../../src/run.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { loadConformanceFlow, loadErrorAsync } from "../helpers/declarations.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-input-contract-origins-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

const ROOT_FLOW = `id: input-contract-origins
name: input-contract-origins
typedefs:
  Profile:
    age: int
    nickname: Optional(str)
  Payload:
    profile: Profile
  Limits:
    count: int
  Flags:
    enabled: bool
    limits: Limits
input:
  topic: !str
  window: !int 30
  payload: !Payload
  flags: !Flags
nodes:
  echo:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo_x
    input:
      x: \${input.topic}
    output: str
output: \${echo.output}
`;

/** One exact parameter selected by name. */
function parameterOf(parameters: readonly Parameter[], name: string): Parameter {
  const matches = parameters.filter((parameter) => parameter.name === name);
  expect(matches).toHaveLength(1);
  expect(matches[0]!.constructor).toBe(Parameter);
  return matches[0]!;
}

/** One compiled flow's canonical START signature. */
function startParameters(compiled: CompiledFlow): readonly Parameter[] {
  const parameters = (compiled.nodes.get(START_ID) as StartNode).params;
  expect(Array.isArray(parameters)).toBe(true);
  expect(parameters!.every((parameter) => parameter.constructor === Parameter)).toBe(true);
  return parameters!;
}

/** One parameter's exact value contract. */
function contractOf(parameter: Parameter): ValueContract {
  expect(parameter.contract?.constructor).toBe(ValueContract);
  return parameter.contract!;
}

/** The compiled child flow baked onto CALL node `nodeId`. */
const childOf = (loaded: ComposedFlow, nodeId: string) =>
  (loaded.compiled.nodes.get(nodeId) as CallNode).child as CompiledFlow;

/** Assert that one semantic origin resolves to the expected source occurrence, and return it. */
function expectSite(
  resolver: OriginResolver | null,
  origin: OriginRef | null | undefined,
  cursor: SourceCursor,
  anchor: "value" | "key" = "value",
): OriginRef {
  expect(resolver).not.toBeNull();
  expect(origin).toBeInstanceOf(OriginRef);
  const frame = resolver!.resolve(origin!);
  expect(frame).not.toBeNull();
  expect(cursor.document).not.toBeNull();
  expect(frame!.text).toBe(cursor.document!.text);
  expect(frame!.range).toEqual((anchor === "key" ? cursor.keyRange : cursor.range)!);
  return origin!;
}

/** The number of distinct origins among `origins`. */
const distinct = (...origins: OriginRef[]) => new Set(origins.map((origin) => origin.key)).size;

/** The cursor at `path` from `document`'s root. */
const cursorAt = (document: SourceDocument, path: readonly string[]) =>
  path.reduce((cursor: SourceCursor, step) => cursor.child(step), document.root);

/** Indent every non-empty line of `text` by `prefix`. */
const indent = (text: string, prefix: string) => text.replace(/^(?=.)/gm, prefix);

test("agent variables keep declaration contract and default source sites", async () => {
  const text = `id: variable-origins
name: variable-origins
typedefs:
  Settings:
    limit: int(1..12)
    tags: list(str)
nodes:
  writer:
    kind: agent
    variables:
      settings: !Settings {limit: 4, tags: [a, b]}
      note: !Optional(str) null
    prompt: \${note}
    output: str
output: \${writer.output}
`;
  const document = SourceDocument.parse(text);
  const loaded = await loadConformanceFlow(text);
  const variables = (loaded.compiled.nodes.get("writer") as AgentNode).variables;
  for (const name of ["settings", "note"]) {
    const variable = parameterOf(variables, name);
    const cursor = cursorAt(document, ["nodes", "writer", "variables", name]);
    const declaration = expectSite(loaded.originResolver, variable.origin, cursor, "key");
    const contract = expectSite(loaded.originResolver, contractOf(variable).origins.default, cursor);
    const fallback = expectSite(loaded.originResolver, variable.defaultOrigin, cursor);
    expect(distinct(declaration, contract, fallback)).toBe(3);
  }
  const settings = parameterOf(variables, "settings");
  expect(contractOf(settings).origins.at(["limit"])).toEqual(contractOf(settings).origins.default);
});

test.each(["compact", "definition", "flat-definition", "inline"])(
  "agent variable locations follow existing node projections (%s)",
  async (placement) => {
    const body = "kind: agent\nvariables:\n  count: !int(1..12) 4\nprompt: ${count}\noutput: str\n";
    const header = "id: projected_variables\nname: projected_variables\n";
    let text: string;
    let cursorPath: string[];
    if (placement === "compact") {
      text = header + body;
      cursorPath = [];
    } else if (placement === "flat-definition") {
      text =
        header +
        "defs:\n  child:\n" +
        indent(body, "    ") +
        "nodes:\n  invoke:\n    kind: call\n    call: child\noutput: ${invoke.output}\n";
      cursorPath = ["defs", "child"];
    } else if (placement === "definition") {
      text =
        header +
        "defs:\n  child:\n    nodes:\n      writer:\n" +
        indent(body, "        ") +
        "    output: ${writer.output}\nnodes:\n  invoke:\n    kind: call\n" +
        "    call: child\noutput: ${invoke.output}\n";
      cursorPath = ["defs", "child", "nodes", "writer"];
    } else {
      text =
        header +
        "nodes:\n  invoke:\n    kind: call\n    nodes:\n      writer:\n" +
        indent(body, "        ") +
        "    output: ${writer.output}\noutput: ${invoke.output}\n";
      cursorPath = ["nodes", "invoke", "nodes", "writer"];
    }

    const cursor = cursorAt(SourceDocument.parse(text), [...cursorPath, "variables", "count"]);
    const loaded = await loadConformanceFlow(text);
    const compiled = placement === "compact" ? loaded.compiled : childOf(loaded, "invoke");
    const agents = [...compiled.nodes.values()].filter((node) => node instanceof AgentNode);
    expect(agents).toHaveLength(1);
    const variable = agents[0]!.variables[0]!;
    expectSite(loaded.originResolver, variable.origin, cursor, "key");
    expectSite(loaded.originResolver, variable.contract!.origins.default, cursor);
    expectSite(loaded.originResolver, variable.defaultOrigin, cursor);

    const invalid = text.replace("!int(1..12) 4", "!int(1..12) 13");
    const error = await loadErrorAsync(loadConformanceFlow(invalid));
    expect(error.message).toContain("'count': `13` is not a valid int(1..12)");
    const invalidCursor = cursorAt(SourceDocument.parse(invalid), [...cursorPath, "variables", "count"]);
    expect(error.diagnostic.frames.at(-1)!.range).toEqual(invalidCursor.range!);
  },
);

test("external aliases rebase agent variables without changing identity", async () => {
  const childText = `id: variable_child
name: variable_child
kind: agent
variables:
  count: !int(1..12) 4
prompt: \${count}
output: str
`;
  const child = await loadConformanceFlow(childText, { sourceLabel: "variable-child.yaml" });
  const parent = await loadConformanceFlow(
    `id: variable-parent
name: variable-parent
uses:
  left: child
  right: child
nodes:
  first:
    kind: call
    call: left
  second:
    kind: call
    call: right
output:
  first: \${first.output}
  second: \${second.output}
`,
    { childResolver: () => child },
  );
  const fragments = ["first", "second"].map((name) => childOf(parent, name));
  const variables = [child.compiled, ...fragments].map(
    (fragment) => ([...fragment.nodes.values()].find((node) => node instanceof AgentNode) as AgentNode).variables[0]!,
  );
  const origins = variables.map((variable) => [...variable.iterOrigins()]);
  expect(origins.every((sites) => sites.length === 3)).toBe(true);
  for (let index = 0; index < 3; index++) {
    expect(distinct(...origins.map((sites) => sites[index]!))).toBe(3);
    const canonical = child.originResolver!.resolve(origins[0]![index]!);
    expect(parent.originResolver!.resolve(origins[1]![index]!)).toEqual(canonical);
    expect(parent.originResolver!.resolve(origins[2]![index]!)).toEqual(canonical);
  }
  expect(flowFingerprint(fragments[0]!)).toBe(flowFingerprint(fragments[1]!));
});

test("root inputs compile parameter contract default and nested origins", async () => {
  // One traversal retains every semantic site on mapping and list-form records.
  const document = SourceDocument.parse(ROOT_FLOW);
  const loaded = await loadConformanceFlow(ROOT_FLOW);
  const inputs = document.root.child("input");

  const topic = parameterOf(loaded.input, "topic");
  const topicCursor = inputs.child("topic");
  expectSite(loaded.originResolver, topic.origin, topicCursor, "key");
  expectSite(loaded.originResolver, contractOf(topic).origins.default, topicCursor);
  expect(topic.origin).not.toEqual(contractOf(topic).origins.default);

  const window = parameterOf(loaded.input, "window");
  const windowCursor = inputs.child("window");
  const parameterOrigin = expectSite(loaded.originResolver, window.origin, windowCursor, "key");
  const contractOrigin = expectSite(loaded.originResolver, contractOf(window).origins.default, windowCursor);
  const defaultOrigin = expectSite(loaded.originResolver, window.defaultOrigin, windowCursor);
  expect(distinct(parameterOrigin, contractOrigin, defaultOrigin)).toBe(3);

  // A record-typed input names its typedef in a tag, so every nested path resolves to that tag.
  for (const [name, paths] of [
    ["payload", [["profile"], ["profile", "age"], ["profile", "nickname"]]],
    ["flags", [["enabled"], ["limits"], ["limits", "count"]]],
  ] as const) {
    const record = parameterOf(loaded.input, name);
    const recordCursor = inputs.child(name);
    expectSite(loaded.originResolver, record.origin, recordCursor, "key");
    const tagOrigin = expectSite(loaded.originResolver, contractOf(record).origins.default, recordCursor);
    expect(contractOf(record).origins.entries).toEqual([]);
    for (const path of paths) expect(contractOf(record).origins.at(path)).toEqual(tagOrigin);
  }
});

test("required flow input failure selects the authored parameter key", async () => {
  // An omitted required value reports the declaration that required it.
  const document = SourceDocument.parse(ROOT_FLOW, "required.yaml");
  const loaded = await loadConformanceFlow(ROOT_FLOW, { sourceLabel: "required.yaml" });
  const topic = parameterOf(loaded.input, "topic");

  const result = await runFlow(
    loaded,
    { payload: { profile: { age: 7 } }, flags: { enabled: true, limits: { count: 1 } } },
    { env: hostEnv(dir) },
  );

  expect(result.status).toBe("failed");
  expect(result.failure).not.toBeNull();
  expect(result.failure!.code).toBe("binding.evaluate");
  expect(result.failure!.trace.at(-1)).toEqual(topic.origin!);
  const diagnostic = loaded.diagnose(result.failure!);
  expect(diagnostic.frames.at(-1)!.range).toEqual(document.root.child("input").child("topic").keyRange!);
});

test("nested flow input failure selects the record tag", async () => {
  // A nested field mismatch resolves to the tag naming the input's record type.
  const document = SourceDocument.parse(ROOT_FLOW, "nested.yaml");
  const loaded = await loadConformanceFlow(ROOT_FLOW, { sourceLabel: "nested.yaml" });
  const payload = parameterOf(loaded.input, "payload");
  const ageOrigin = contractOf(payload).origins.at(["profile", "age"]);
  expect(ageOrigin).toEqual(contractOf(payload).origins.default);

  const result = await runFlow(
    loaded,
    {
      topic: "hello",
      payload: { profile: { age: "not-an-int" } },
      flags: { enabled: true, limits: { count: 1 } },
    },
    { env: hostEnv(dir) },
  );

  expect(result.status).toBe("failed");
  expect(result.failure).not.toBeNull();
  expect(result.failure!.code).toBe("contract.input");
  expect(result.failure!.trace.at(-1)).toEqual(ageOrigin!);
  const diagnostic = loaded.diagnose(result.failure!);
  expect(diagnostic.frames.at(-1)!.range).toEqual(document.root.child("input").child("payload").range!);
});

test("scalar flow input failure selects the contract root", async () => {
  // A root mismatch resolves to the declaration value, not the parameter key.
  const document = SourceDocument.parse(ROOT_FLOW, "scalar.yaml");
  const loaded = await loadConformanceFlow(ROOT_FLOW, { sourceLabel: "scalar.yaml" });
  const window = parameterOf(loaded.input, "window");
  const contractOrigin = contractOf(window).origins.default;

  const result = await runFlow(
    loaded,
    {
      topic: "hello",
      window: "not-an-int",
      payload: { profile: { age: 7 } },
      flags: { enabled: true, limits: { count: 1 } },
    },
    { env: hostEnv(dir) },
  );

  expect(result.status).toBe("failed");
  expect(result.failure).not.toBeNull();
  expect(result.failure!.code).toBe("contract.input");
  expect(result.failure!.trace.at(-1)).toEqual(contractOrigin!);
  const diagnostic = loaded.diagnose(result.failure!);
  expect(diagnostic.frames.at(-1)!.range).toEqual(document.root.child("input").child("window").range!);
});

test("non string record field is a located load error", async () => {
  // Invalid YAML record keys in a typedef never escape the Compose error boundary.
  const text = `id: invalid-record-field
name: invalid-record-field
typedefs:
  Payload:
    true: int
input:
  payload: !Payload
nodes: {}
`;
  const document = SourceDocument.parse(text);

  const error = await loadErrorAsync(loadConformanceFlow(text));

  expect(error.message).toContain("names in a flow must be text");
  const expected = document.root.child("typedefs").child("Payload").keyRangeOf("true");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(expected!);
});

const INVALID_NAMES: [string, string][] = [
  ["true", "names in a flow must be text"],
  ["null", "names in a flow must be text"],
  ["1.5", "names in a flow must be text"],
  ["''", "flow input names"],
];

test.each(
  ["multi-node", "compact"].flatMap((form) => INVALID_NAMES.map(([key, message]) => [form, key, message])),
)("invalid definition input name is a located load error (%s, %s)", async (form, key, message) => {
  // Invalid nested signature keys never escape the Compose error boundary.
  const body =
    form === "compact"
      ? `    kind: code
    code: ./fixtures/compose_codefns.ts:echo_x
    input:
      ${key}: !int
    output: int
`
      : `    input:
      ${key}: !int
    nodes: {}
`;
  const text = `id: invalid-definition-input
name: invalid-definition-input
defs:
  child:
${body}nodes: {}
`;
  const inputCursor = SourceDocument.parse(text).root.child("defs").child("child").child("input");
  const invalidName = Object.keys(inputCursor.value as object)[0]!;

  const error = await loadErrorAsync(loadConformanceFlow(text));

  expect(error.message).toContain(message);
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(inputCursor.keyRangeOf(invalidName)!);
});

test.each(["null", "[]", "false", "0", "''"])(
  "falsy definition input keeps existing empty normalization (%s)",
  async (empty) => {
    // Located compilation preserves the loader's established empty-def shorthand.
    await loadConformanceFlow(`id: empty-definition-input
name: empty-definition-input
defs:
  child:
    input: ${empty}
    nodes: {}
nodes: {}
`);
  },
);

test("compact root and both definition forms retain input contract sites", async () => {
  // Root and definition lowering share the physical declaration reader.
  const compactRoot = `id: compact_input_contract
name: compact_input_contract
input:
  x: !int 3
output: int
kind: code
code: ./fixtures/compose_codefns.ts:echo_x
`;
  const compactDocument = SourceDocument.parse(compactRoot);
  const compact = await loadConformanceFlow(compactRoot);
  const rootParameter = parameterOf(compact.input, "x");
  const rootCursor = compactDocument.root.child("input").child("x");
  expectSite(compact.originResolver, rootParameter.origin, rootCursor, "key");
  expectSite(compact.originResolver, contractOf(rootParameter).origins.default, rootCursor);
  expectSite(compact.originResolver, rootParameter.defaultOrigin, rootCursor);

  const text = `id: definition-input-contracts
name: definition-input-contracts
input:
  x: !int
defs:
  named:
    input:
      x: !int
    nodes:
      echo:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo_x
        input:
          x: \${input.x}
        output: int
    output: \${echo.output}
  compact:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo_x
    input:
      x: !int
    output: int
nodes:
  named_call:
    kind: call
    call: named
    input:
      x: \${input.x}
  compact_call:
    kind: call
    call: compact
    input:
      x: \${input.x}
output:
  named: \${named_call.output}
  compact: \${compact_call.output}
`;
  const document = SourceDocument.parse(text);
  const loaded = await loadConformanceFlow(text);
  const sites: [Parameter, SourceCursor][] = [
    [
      parameterOf(startParameters(childOf(loaded, "named_call")), "x"),
      cursorAt(document, ["defs", "named", "input", "x"]),
    ],
    [
      parameterOf(startParameters(childOf(loaded, "compact_call")), "x"),
      cursorAt(document, ["defs", "compact", "input", "x"]),
    ],
  ];
  for (const [parameter, cursor] of sites) {
    expectSite(loaded.originResolver, parameter.origin, cursor, "key");
    expectSite(loaded.originResolver, contractOf(parameter).origins.default, cursor);
  }
});

test("generated inline child input contract stays source free", async () => {
  // Inferred anonymous-child signatures do not claim an authored declaration.
  const text = `id: generated-inline-input
name: generated-inline-input
input:
  x: !int
nodes:
  invoke:
    kind: call
    input:
      x: \${input.x}
    nodes:
      echo:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo_x
        input:
          x: \${input.x}
        output: int
    output: \${echo.output}
output: \${invoke.output}
`;
  const loaded = await loadConformanceFlow(text);
  const generated = parameterOf(startParameters(childOf(loaded, "invoke")), "x");

  expect(generated.origin).toBeNull();
  expect(generated.defaultOrigin).toBeNull();
  expect(contractOf(generated).origins).toEqual(new ContractOrigins());
});

test("external alias rebases every input contract origin", async () => {
  // An alias view owns distinct keys that resolve to the canonical child sites.
  const childText = `id: external-input-child
name: external-input-child
typedefs:
  Profile:
    age: int
  Payload:
    profile: Profile
input:
  payload: !Payload
  window: !int 30
nodes:
  echo:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo_x
    input:
      x: \${input.payload}
    output:
      profile:
        age: int
output: \${echo.output}
`;
  const child = await loadConformanceFlow(childText, { sourceLabel: "child.yaml" });
  const parentText = `id: external-input-parent
name: external-input-parent
typedefs:
  Profile:
    age: int
  Payload:
    profile: Profile
input:
  payload: !Payload
uses:
  imported: child
nodes:
  invoke:
    kind: call
    call: imported
    input:
      payload: \${input.payload}
output: \${invoke.output}
`;
  const parent = await loadConformanceFlow(parentText, {
    sourceLabel: "parent.yaml",
    childResolver: () => child,
  });
  const embeddedInputs = startParameters(childOf(parent, "invoke"));

  const pairs = [
    [parameterOf(child.input, "payload").origin, parameterOf(embeddedInputs, "payload").origin],
    [
      contractOf(parameterOf(child.input, "payload")).origins.default,
      contractOf(parameterOf(embeddedInputs, "payload")).origins.default,
    ],
    [
      contractOf(parameterOf(child.input, "payload")).origins.at(["profile", "age"]),
      contractOf(parameterOf(embeddedInputs, "payload")).origins.at(["profile", "age"]),
    ],
    [parameterOf(child.input, "window").defaultOrigin, parameterOf(embeddedInputs, "window").defaultOrigin],
  ];
  expect(child.originResolver).not.toBeNull();
  expect(parent.originResolver).not.toBeNull();
  for (const [canonical, rebased] of pairs) {
    expect(canonical).toBeInstanceOf(OriginRef);
    expect(rebased).toBeInstanceOf(OriginRef);
    expect(rebased).not.toEqual(canonical);
    expect(parent.originResolver!.resolve(rebased!)).toEqual(child.originResolver!.resolve(canonical!));
  }
});

test("formatting only relocation keeps keys and moves resolved ranges", async () => {
  // Semantic identities stay stable while the retained document owns coordinates.
  const movedText = "# formatting-only relocation\n\n" + ROOT_FLOW;
  const baseline = await loadConformanceFlow(ROOT_FLOW, { sourceLabel: "flow.yaml" });
  const moved = await loadConformanceFlow(movedText, { sourceLabel: "flow.yaml" });
  const baselinePayload = parameterOf(baseline.input, "payload");
  const movedPayload = parameterOf(moved.input, "payload");
  const baselineAge = contractOf(baselinePayload).origins.at(["profile", "age"]);
  const movedAge = contractOf(movedPayload).origins.at(["profile", "age"]);

  expect(movedPayload.origin).toEqual(baselinePayload.origin);
  expect(contractOf(movedPayload).origins).toEqual(contractOf(baselinePayload).origins);
  expect(movedAge).toEqual(baselineAge);
  expect(baseline.originResolver).not.toBeNull();
  expect(moved.originResolver).not.toBeNull();
  const baselineFrame = baseline.originResolver!.resolve(baselineAge!);
  const movedFrame = moved.originResolver!.resolve(movedAge!);
  expect(baselineFrame).not.toBeNull();
  expect(movedFrame).not.toBeNull();
  expect(movedFrame!.range.startLine).toBe(baselineFrame!.range.startLine + 2);
  expect(movedFrame!.range.endLine).toBe(baselineFrame!.range.endLine + 2);
});
