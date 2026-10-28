/**
 * A flow's `output:` shape decides a child's signature and a run's result: a one-field record
 * stays a record, a value output is the value, and an empty record is refused where a value is
 * needed; also `ChildSignature`'s constructor checks.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import { END_ID, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { combineOutputContracts } from "../../src/compile/output_contracts.ts";
import { ChildSignature, childSignature } from "../../src/compose/nodes/resolver.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { BackwardResult } from "../../src/nodes/base.ts";
import { isExactly, type OutputContract, sameOutputContract, ValueContract } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { runFlow } from "../../src/run.ts";
import { computeBackward } from "../../src/runtime/backward.ts";
import { validateAttributedBackward } from "../../src/runtime/learning.ts";
import { Type, typeEquals, ValueKind } from "../../src/typesys/values.ts";
import { assembled, assembleText, type CaseRow, loadError, loadErrorAsync, runCase } from "../helpers/declarations.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { engine, graph, RecordingBackwardNode } from "../helpers/run_state_fixtures.ts";

const STR = Type.scalar(ValueKind.STRING);
const INT = Type.scalar(ValueKind.INTEGER);

const STEP = `
  step:
    kind: code
    output: str
    code: |
      return "filled"
`;

/** The one-step flow with the `output:` section `section`. */
const shapeFlow = (section: string) => `id: shape\nname: shape\nnodes:${STEP}${section}`;

/** A parent whose def `child` has the `output:` lines `defOutput`. */
const parent = (defOutput: string, nodes: string, output: string) =>
  "id: parent\nname: parent\ndefs:\n" +
  '  child:\n    nodes:\n      step:\n        kind: code\n        output: str\n        code: |\n          return "filled"\n' +
  defOutput +
  "nodes:\n" +
  nodes +
  `output: ${output}\n`;

const ONE_FIELD_DEF_OUTPUT = "    output:\n      order: ${step.output}\n";

const CALL_CHILD = "  c:\n    kind: call\n    call: child\n";

const dir = mkdtempSync(join(tmpdir(), "yian-flow-output-shape-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

/** Load and run `text` on `inputs`, returning the output. */
async function run(text: string, inputs: Record<string, unknown> = {}): Promise<unknown> {
  const result = await runFlow(await loadFlow(text), inputs, { env: hostEnv(dir) });
  expect(result.failure).toBeNull();
  return result.output;
}

/** Return the error `run` throws, failing when it returns. */
function thrown(run: () => unknown): Error {
  try {
    run();
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected an error");
}

/** Expect `run` to throw exactly `cls` with `message`. */
function expectError(run: () => unknown, cls: typeof Error, message: string): void {
  const error = thrown(run);
  expect(error.constructor).toBe(cls);
  expect(error.message).toBe(message);
}

test("the child signature of a one-field record is its exact record", async () => {
  const sig = childSignature(await loadFlow(shapeFlow("output:\n  order: ${step.output}\n")));

  expect(sig.returnsRecord).toBe(true);
  expect(isExactly(sig.output, ValueContract)).toBe(true);
  const output = sig.output as ValueContract;
  expect(output.type.kind).toBe(ValueKind.OBJECT);
  expect(output.type.fields!.size).toBe(1);
  expect(typeEquals(output.type.fields!.get("order")!, STR)).toBe(true);
});

test("the child signature of a value output is the value contract", async () => {
  const sig = childSignature(await loadFlow(shapeFlow("output: ${step.output}\n")));

  expect(sig.returnsRecord).toBe(false);
  expect(isExactly(sig.output, ValueContract)).toBe(true);
  expect(typeEquals((sig.output as ValueContract).type, STR)).toBe(true);
});

test("the child signature of an empty record is the empty record contract", async () => {
  const sig = childSignature(await loadFlow(shapeFlow("output: {}\n")));

  expect(sig.returnsRecord).toBe(true);
  expect(sameOutputContract(sig.output, combineOutputContracts(new Map()))).toBe(true);
});

test("a direct child signature must say it returns a record", () => {
  const text = new ValueContract(STR);
  expectError(
    () =>
      new ChildSignature({
        inputs: [],
        output: combineOutputContracts(
          new Map([
            ["a", text],
            ["b", text],
          ]),
        ),
        declaredOutputs: [
          ["a", text],
          ["b", text],
        ],
        returnsRecord: false,
      }),
    RangeError,
    "a child that does not return a record declares at most one output",
  );
});

test("the ChildSignature constructor rejects malformed output evidence", () => {
  const text = new ValueContract(STR);
  const forged = "x" as unknown as OutputContract;

  expectError(
    () => new ChildSignature({ inputs: [], output: forged }),
    TypeError,
    "child output must be ValueContract, PartialRecordContract, or null, got string",
  );
  expectError(
    () =>
      new ChildSignature({
        inputs: [],
        output: text,
        declaredOutputs: "result" as never,
      }),
    TypeError,
    "declared child outputs must be an array",
  );
  expectError(
    () =>
      new ChildSignature({
        inputs: [],
        output: text,
        declaredOutputs: [["result"]] as never,
      }),
    TypeError,
    "declared child outputs must be (name, optional contract) tuples",
  );
  expectError(
    () => new ChildSignature({ inputs: [], output: text, declaredOutputs: [["", text]] }),
    TypeError,
    "declared child output names must be non-empty strings",
  );
  expectError(
    () =>
      new ChildSignature({
        inputs: [],
        output: text,
        declaredOutputs: [
          ["a", text],
          ["a", text],
        ],
        returnsRecord: true,
      }),
    RangeError,
    "duplicate declared child output 'a'",
  );
  expectError(
    () => new ChildSignature({ inputs: [], output: text, declaredOutputs: [["a", forged]] }),
    TypeError,
    "declared child output contracts must be ValueContract, PartialRecordContract, or null, got string for 'a'",
  );
});

test("a ChildSignature with no declared evidence skips the evidence check", () => {
  const text = new ValueContract(STR);
  const sig = new ChildSignature({ inputs: [], output: text });

  expect(sig.output).toBe(text);
  expect(sig.declaredOutputs).toEqual([]);
  expect(sig.outputNames).toEqual([]);
  expect(sig.returnsRecord).toBe(false);
  expect(Object.isFrozen(sig)).toBe(true);
  expect(Object.isFrozen(sig.inputs)).toBe(true);
});

/** A parent mapping its def `child` (output lines `defOutput`) over `items`. */
const mapParent = (defOutput: string) =>
  "input:\n  items: !list(str)\n" +
  parent(defOutput, "  each:\n    kind: map\n    call: child\n    over: ${input.items}\n", "${each.output}");

test("a map over a one-field record def returns a list of records", async () => {
  const loaded = await loadFlow(mapParent(ONE_FIELD_DEF_OUTPUT));

  const contract = loaded.compiled.nodes.get("each")!.outputContract as ValueContract;
  expect(contract.type.kind).toBe(ValueKind.LIST_OBJECT);
  const fields = contract.type.element!.fields!;
  expect([...fields.keys()]).toEqual(["order"]);
  expect(typeEquals(fields.get("order")!, STR)).toBe(true);
  const result = await runFlow(loaded, { items: ["a", "b"] }, { env: hostEnv(dir) });
  expect(result.output).toEqual([{ order: "filled" }, { order: "filled" }]);
});

test.each([
  ["absent", ""],
  ["empty record", "    output: {}\n"],
])("a map over a def without fields is rejected: %s", async (_name, defOutput) => {
  const error = await loadErrorAsync(loadFlow(mapParent(defOutput)));

  expect(error.message).toBe(
    "map node 'each': callable 'child' must declare a non-empty `output:`; " +
      "a mapped call (over:) returns a list of the child's results",
  );
});

const NO_CARRIED_FIELD =
  "loop node 'lp': body inline body of node 'lp' must return a record with at least one carried field " +
  '(a subset of the carried record ["count"]); write `output: {field: ...}` or a whole reference to a node ' +
  "with a record output";

test("a loop body with an empty record output is rejected", () => {
  const error = loadError(() =>
    assembleText(`
id: l
name: l
nodes:
  lp:
    kind: loop
    times: 2
    input:
      count: 0
    nodes:
      bump:
        kind: code
        output: int
        code: |
          return 1
    output: {}
output: \${lp.output}
`),
  );

  expect(error.message).toBe(NO_CARRIED_FIELD);
});

test("a loop value body re-exporting an empty record is rejected", async () => {
  const error = await loadErrorAsync(
    loadFlow(`
id: l
name: l
defs:
  empty:
    nodes:
      step:
        kind: code
        output: int
        code: |
          return 1
    output: {}
nodes:
  lp:
    kind: loop
    times: 2
    input:
      count: 0
    nodes:
      inner:
        kind: call
        call: empty
    output: \${inner.output}
output: \${lp.output}
`),
  );

  expect(error.message).toBe(NO_CARRIED_FIELD);
});

// ---------- inline bodies, loaded ----------

/** A record of `fields`, each required. */
function record(fields: Record<string, Type>): Type {
  return new Type({
    kind: ValueKind.OBJECT,
    fields: new Map(Object.entries(fields)),
    required: new Set(Object.keys(fields)),
  });
}

test.each<[string, string, EndNode["mode"], string[]]>([
  ["one-field", "output:\n  order: ${step.output}\n", "record", ["order"]],
  ["two-fields", "output:\n  order: ${step.output}\n  note: ${step.output}\n", "record", ["order", "note"]],
  ["value", "output: ${step.output}\n", "value", ["result"]],
  ["absent", "", "value", []],
  ["empty-record", "output: {}\n", "record", []],
])("the flow's END follows the output shape: %s", (_name, section, mode, names) => {
  const { flow } = assembleText(shapeFlow(section));

  const end = flow.compiled.nodes.get(END_ID) as EndNode;
  expect([end.mode, end.names]).toEqual([mode, names]);
});

test.each<[string, string, unknown]>([
  ["one-field", "output:\n  order: ${step.output}\n", { order: "filled" }],
  ["two-fields", "output:\n  order: ${step.output}\n  note: ${step.output}\n", { order: "filled", note: "filled" }],
  ["value", "output: ${step.output}\n", "filled"],
  ["empty-record", "output: {}\n", {}],
])("the flow result follows the output shape: %s", async (_name, section, expected) => {
  expect(await run(shapeFlow(section))).toEqual(expected);
});

const INLINE_CALL = `
id: parent
name: parent
nodes:
  c:
    kind: call
    nodes:${STEP.replaceAll("\n  ", "\n      ")}    output:
      order: \${step.output}
output: \${c.output}
`;

test("an inline call body with a one-field output returns the record", async () => {
  const { flow } = assembleText(INLINE_CALL);

  expect((flow.compiled.nodes.get("c")!.outputContract as ValueContract).type).toEqual(record({ order: STR }));
  expect(await run(INLINE_CALL)).toEqual({ order: "filled" });
});

const LOOP_ONE_FIELD = `
id: l
name: l
nodes:
  lp:
    kind: loop
    times: 3
    input:
      count: 0
    nodes:
      bump:
        kind: code
        input:
          n: \${input.count}
        output: int
        code: |
          return inputs["n"] + 1
    output:
      count: \${bump.output}
output: \${lp.output}
`;

test("a loop body with a one-field record output runs", async () => {
  const { flow } = assembleText(LOOP_ONE_FIELD);

  expect((flow.compiled.nodes.get("lp")!.outputContract as ValueContract).type).toEqual(record({ count: INT }));
  expect(await run(LOOP_ONE_FIELD)).toEqual({ count: 3n });
});

// ---------- backward ----------

test("root feedback on a one-field record reaches its producer", async () => {
  const producer = new RecordingBackwardNode(
    "producer",
    (inputs) => inputs.x,
    (_context, feedback) => new BackwardResult({ inputFeedback: { x: feedback } }),
    { reads: { x: "${input.topic}" } },
  );
  const end = EndNode.record(END_ID, { outputNames: ["order"] });
  const flow = graph([producer, end], [[START_ID, "producer"]], {
    outputs: [new FlowOutput("order", "${producer.output}")],
  });
  const built = engine(flow, { inputs: { topic: "reefs" }, learnable: true });
  await built.run();
  expect(built.state.trace).not.toBeNull();

  const feedback = { order: "Review the order." };
  const [root, results] = await computeBackward(built.state.trace!, feedback);
  validateAttributedBackward(built.state.trace!, feedback, root, results);

  expect(producer.calls).toHaveLength(1);
  expect(producer.calls[0]![1]).toEqual(["Review the order."]);
  expect(root.inputFeedback).toEqual({ topic: ["Review the order."] });
});

test.each([
  ["one-field", ONE_FIELD_DEF_OUTPUT],
  ["empty-record", "    output: {}\n"],
])("reading a missing field of a record child is rejected at load: %s", async (_id, defOutput) => {
  // A field read on a child's statically known record contract is already checked at load, for
  // one field and for none.
  const error = await loadErrorAsync(loadFlow(parent(defOutput, CALL_CHILD, "${c.output.missing}")));

  expect(error.message).toContain("reads unknown field 'missing' on record");
});

test("a non-string output name in a def is a load error", async () => {
  const error = await loadErrorAsync(
    loadFlow(parent("    output:\n      1: ${step.output}\n", CALL_CHILD, "${c.output}")),
  );

  expect(error.message).toBe(
    'the key `1` was read by YAML as an integer, but names in a flow must be text. Quote it ("1") or ' +
      "choose another name.",
  );
});

// ---------- runs ----------

test("a call over a one-field record def returns the record: a field read", async () => {
  expect(await run(parent(ONE_FIELD_DEF_OUTPUT, CALL_CHILD, "${c.output.order}"))).toBe("filled");
});

test("a nested value re-export of a record def keeps the record", async () => {
  const text = `
id: parent
name: parent
defs:
  b:
    nodes:
      step:
        kind: code
        output: str
        code: |
          return "filled"
    output:
      order: \${step.output}
  a:
    nodes:
      inner:
        kind: call
        call: b
    output: \${inner.output}
nodes:
  c:
    kind: call
    call: a
output: \${c.output.order}
`;
  expect(await run(text)).toBe("filled");
});

test("a case then call over a one-field record def returns the record", async () => {
  const nodes = `  fallback:
    kind: code
    output:
      order: str
    code: |
      return {"order": "never"}
  route:
    kind: case
    cases:
      - when: "\${input.go}"
        then: call(child)
    else: fallback
`;
  const text = "input:\n  go: !bool\n" + parent(ONE_FIELD_DEF_OUTPUT, nodes, "${route.output.order}");
  expect(await run(text, { go: true })).toBe("filled");
});

test("a literal flow output is returned as written: value", async () => {
  expect(await run(shapeFlow("output: 5\n"))).toBe(5n);
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>(
  [
    "map_over_a_def_without_output",
    "loop_body_with_an_empty_record_output",
    "loop_body_re_exporting_an_empty_record",
    "missing_field_of_an_empty_record_child",
  ].map((name) => ({ id: `load-core/engine_flow_output_shape__${name}`, call: assembled })),
)("assemble: $id", runCase);
