/**
 * A TOOL node's optional `output:` contract is checked against every value the tool returns, and
 * a run checks each static TOOL node against its tool's declared types before the log exists: a
 * port of `tests/engine/test_typed_tool_node.py`.
 *
 * TS tools are `FunctionTool`s built with `tool()`: a Python parameter annotation is a property
 * of the argument JSON Schema (`int` is `{"type": "integer"}`, an unannotated parameter `{}`,
 * `Field(ge=0)` a `minimum`), a pydantic model parameter an object schema (a `extra="forbid"`
 * model sets `additionalProperties: false`), and a return annotation the `output` JSON Schema (a
 * `TypedDict` a closed object schema, a `Union` of them an `anyOf`); Python's `tool_types` read
 * through a `ToolSet` is `toolTypes(tool)`. A host stop is the tool aborting the run's signal and
 * throwing its reason, where Python raises `_HostStop`; the run rejects with that reason. A
 * run-start refusal is a `RangeError` where Python raises `ValueError`. Tools run on the main
 * thread, so a list the test's tools append to stands for Python's `_calls`. Every run sees only
 * the test's credential store in its environment.
 *
 * Ported: every row but those covered below. The inline CODE bodies are written in JS (decision
 * 11): `on_filled` concatenates where Python uses an f-string. Ints a host tool returns read as
 * `bigint`s in the run's output (decisions 4a, 9A). Changed: lists of names and quoted values in
 * messages are JSON style (`known tools: ["place_order"]`, `"bad" does not match`; decisions 7,
 * 43). `test_an_unread_return_type_skips_the_check_and_the_gate_still_applies`: a TS tool's return
 * type is read only from its output JSON Schema, so its rows are a tool with no output schema
 * (Python's unannotated function and `_run`-only `BaseTool`) and one whose output schema has no
 * yian type, `{}` (Python's model return).
 *
 * Covered by `run-leaves/typed_tool_node__the_gate_runs_on_every_call_inside_a_map`:
 * `test_the_gate_runs_on_every_call_inside_a_map`; by
 * `run-leaves/typed_tool_node__an_input_the_tool_does_not_take_fails_at_run_start`:
 * `test_an_input_the_tool_does_not_take_fails_at_run_start`; by
 * `run-leaves/typed_tool_node__every_missing_required_input_is_named`:
 * `test_every_missing_required_input_is_named`; by
 * `run-leaves/typed_tool_node__an_omitted_optional_input_runs`: `test_an_omitted_optional_input_runs`;
 * by `run-leaves/typed_tool_node__a_binding_disjoint_from_a_schema_maximum_fails_at_run_start`:
 * `test_a_tagged_binding_disjoint_from_a_json_schema_maximum_fails_at_run_start`; by
 * `run-leaves/typed_tool_node__ask_user_with_string_options_runs`:
 * `test_ask_user_with_string_options_runs`.
 */

import { afterAll, afterEach, beforeEach, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { resumeFlow, type RunFlowOptions, type RunResult, runFlow } from "../../src/run.ts";
import { type FunctionTool, tool } from "../../src/tools/function_tools.ts";
import { registerTool, TOOL_REGISTRY } from "../../src/tools/registry.ts";
import { toolTypes } from "../../src/tools/types.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { rejected } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-typed-tool-node-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

// Keep process-global registrations isolated between tests.
const registered = new Set(TOOL_REGISTRY.keys());
afterEach(() => {
  for (const id of TOOL_REGISTRY.keys()) if (!registered.has(id)) TOOL_REGISTRY.delete(id);
});

let logs = 0;
/** A fresh log path. */
const freshLog = () => join(dir, `run-${++logs}.jsonl`);

/** Load `source` in the test's environment. */
const load = (source: string) => loadFlow(source, { env });

/** `runFlow` in the test's environment. */
const run = async (source: string, inputs: Record<string, unknown>, options: RunFlowOptions = {}) =>
  runFlow(await load(source), inputs, { env, ...options });

/** Return the error `promise` rejects with, after checking it is a run-start `RangeError`. */
async function refused(promise: Promise<unknown>): Promise<string> {
  const error = await rejected(promise);
  expect(error).toBeInstanceOf(RangeError);
  return (error as Error).message;
}

/** The arguments of each call the test's recording tools received. */
let calls: Record<string, unknown>[] = [];
beforeEach(() => {
  calls = [];
});

/**
 * A `place_order` tool taking `properties` (required unless `required` says otherwise) and
 * returning what `output` describes, if given; `execute` gets the arguments.
 */
function placeOrder(
  properties: Record<string, JsonObject>,
  execute: (args: Record<string, unknown>) => unknown,
  options: { required?: string[]; output?: JsonObject; additionalProperties?: boolean } = {},
): FunctionTool {
  const parameters: JsonObject = { type: "object", properties, required: options.required ?? Object.keys(properties) };
  if (options.additionalProperties !== undefined) parameters.additionalProperties = options.additionalProperties;
  return tool({
    name: "place_order",
    description: "Place one order.",
    parameters,
    ...(options.output === undefined ? {} : { output: options.output }),
    execute: (args) => execute(args as Record<string, unknown>),
  });
}

/** Return a host-stop pair: a signal, and a call that aborts it and throws the stop. */
function hostStop() {
  const host = new AbortController();
  const stop = new Error("host stop");
  const interrupt = () => {
    host.abort(stop);
    throw stop;
  };
  return { signal: host.signal, stop, interrupt };
}

/** Return the source line of `flow` that `result`'s failure's one diagnostic frame points at. */
function authoredLine(flow: ComposedFlow, result: RunResult): string {
  const frames = flow.diagnose(result.failure!).frames;
  expect(frames).toHaveLength(1);
  const frame = frames[0]!;
  return frame.text.split("\n")[frame.range.startLine - 1]!;
}

const INT_FLOW = `
id: count
name: count
nodes:
  place:
    kind: tool
    tool_id: place_order
    input:
      qty: 5
    output: int
output: \${place.output}
`;

/** An untyped `place_order(qty)` returning `value`. */
const returning = (value: unknown) => placeOrder({ qty: {} }, () => value);

test("a value that fits the declared output succeeds", async () => {
  const result = await run(INT_FLOW, {}, { tools: { place_order: returning(3) } });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe(3n);
});

test("a value that does not fit fails at the output declaration", async () => {
  const flow = await load(INT_FLOW);
  const result = await runFlow(flow, {}, { env, tools: { place_order: returning("x") } });

  expect(result.status).toBe("failed");
  expect(result.failure!.nodeId).toBe("place");
  expect(result.failure!.code).toBe("contract.output");
  expect(result.failure!.message).toContain("does not match declared type int");
  expect(authoredLine(flow, result).trim()).toStartWith("output: int");
});

const ONE_NODE_FLOW = `
id: count
name: count
input:
  qty: !int
kind: tool
tool_id: place_order
output: int
`;

test("a one node tool flow gates its output", async () => {
  const flow = await load(ONE_NODE_FLOW);

  const ok = await runFlow(flow, { qty: 5 }, { env, tools: { place_order: returning(4) } });
  const bad = await runFlow(flow, { qty: 5 }, { env, tools: { place_order: returning("x") } });

  expect(ok.status, ok.failure?.message).toBe("succeeded");
  expect(ok.output).toBe(4n);
  expect(bad.status).toBe("failed");
  expect(bad.failure!.nodeId).toBe("count");
  expect(bad.failure!.code).toBe("contract.output");
  expect(authoredLine(flow, bad).trim()).toStartWith("output: int");
});

const ORDERS_FLOW = `
id: orders
name: orders
typedefs:
  Filled: {type: 'Literal(filled)', qty: int}
  Interrupted: {type: 'Literal(interrupted)', by: str, partial: 'Optional(int)'}
  OrderResult: Union(Filled, Interrupted)
nodes:
  place:
    kind: tool
    tool_id: place_order
    input:
      qty: 5
    output: OrderResult
  route:
    kind: case
    on: \${place.output.type}
    cases:
      - when: filled
        then: on_filled
      - when: interrupted
        then: on_interrupted
  on_filled:
    kind: code
    input:
      qty: \${place.output.qty}
    code: 'return "filled " + inputs["qty"]'
    output: str
  on_interrupted:
    kind: code
    input:
      by: \${place.output.by}
    code: 'return "stopped by " + inputs["by"]'
    output: str
output: \${on_filled.output | on_interrupted.output}
`;

for (const [name, returned, expected] of [
  ["interrupted", { type: "interrupted", by: "owner", partial: 2 }, "stopped by owner"],
  ["filled", { type: "filled", qty: 5 }, "filled 5"],
] as const) {
  test(`a variant result routes to its branch: ${name}`, async () => {
    const result = await run(ORDERS_FLOW, {}, { tools: { place_order: returning(returned) } });

    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(result.output).toBe(expected);
  });
}

test("an unknown variant fails at the gate", async () => {
  const result = await run(ORDERS_FLOW, {}, { tools: { place_order: returning({ type: "rejected" }) } });

  expect(result.status).toBe("failed");
  expect(result.failure!.nodeId).toBe("place");
  expect(result.failure!.code).toBe("contract.output");
  expect(result.failure!.message).toContain('"rejected" is not a tag of the union');
});

test("an unexpected tool error fails the node", async () => {
  const failing = placeOrder({ qty: {} }, () => {
    throw new Error("exchange down");
  });

  const result = await run(INT_FLOW, {}, { tools: { place_order: failing } });

  expect(result.status).toBe("failed");
  expect(result.failure!.nodeId).toBe("place");
  expect(result.failure!.message).toContain("exchange down");
});

test("a host stop leaves the run resumable", async () => {
  const path = freshLog();
  const flow = await load(INT_FLOW);
  const { signal, stop, interrupt } = hostStop();

  const error = await rejected(
    runFlow(flow, {}, { log: path, env, signal, tools: { place_order: placeOrder({ qty: {} }, interrupt) } }),
  );
  expect(error).toBe(stop);

  const resumed = await resumeFlow(flow, { log: path, env, tools: { place_order: returning(7) } });

  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBe(7n);
});

// --- Run-start checks against the tool's declared types ---------------------------------

const INT: JsonObject = { type: "integer" };
const STR: JsonObject = { type: "string" };

/** `place_order(qty: int, sym: str) -> str`, recording each call. */
const placeQtySym = placeOrder(
  { qty: INT, sym: STR },
  (args) => {
    calls.push(args);
    return `${args.qty} ${args.sym}`;
  },
  { output: STR },
);

/** A flow whose one TOOL node `place` calls `place_order` with `inputBlock`'s inputs. */
function oneToolFlow(inputBlock: string): string {
  return `
id: placing
name: placing
nodes:
  place:
    kind: tool
    tool_id: place_order
    input:
${inputBlock}
output: \${place.output}
`;
}

test("a missing required input fails at run start", async () => {
  const path = freshLog();
  const flow = await load(oneToolFlow("      sym: X"));

  const message = await refused(runFlow(flow, {}, { log: path, env, tools: { place_order: placeQtySym } }));

  expect(message).toContain("TOOL node 'place' is missing input 'qty' that tool 'place_order' requires");
  expect(existsSync(path)).toBe(false);
  expect(calls).toEqual([]);
});

const UNREACHED_BRANCH_FLOW = `
id: branching
name: branching
nodes:
  route:
    kind: case
    cases:
      - when: 'true'
        then: fine
    else: place
  fine:
    kind: code
    code: 'return "ok"'
    output: str
  place:
    kind: tool
    tool_id: place_order
    input:
      qtty: 5
      sym: X
output: \${fine.output | place.output}
`;

const DEFS_CHILD_FLOW = `
id: nested
name: nested
defs:
  inner:
    nodes:
      place:
        kind: tool
        tool_id: place_order
        input:
          qtty: 5
          sym: X
    output: \${place.output}
nodes:
  a:
    kind: call
    call: inner
output: \${a.output}
`;

const MAP_BODY_FLOW = `
id: batch
name: batch
nodes:
  each:
    kind: map
    over: [1, 2]
    input:
      x: \${item}
    nodes:
      place:
        kind: tool
        tool_id: place_order
        input:
          qtty: \${input.x}
          sym: X
    output: \${place.output}
output: \${each.output}
`;

const LOOP_BODY_FLOW = `
id: looping
name: looping
defs:
  turn:
    input:
      count: !int
      last: !str
    nodes:
      place:
        kind: tool
        tool_id: place_order
        input:
          qtty: \${input.count}
          sym: X
    output:
      count: \${input.count + 1}
      last: \${place.output}
nodes:
  again:
    kind: loop
    call: turn
    input:
      count: 0
      last: ""
    while: count < 1
    max: 3
output: \${again.output}
`;

for (const [name, source, nodePath] of [
  ["unreached-branch", UNREACHED_BRANCH_FLOW, "place"],
  ["defs-child", DEFS_CHILD_FLOW, "a/place"],
  ["map-body", MAP_BODY_FLOW, "each/place"],
  ["loop-body", LOOP_BODY_FLOW, "again/place"],
] as const) {
  test(`tool nodes that may never run are checked: ${name}`, async () => {
    const path = freshLog();

    const message = await refused(run(source, {}, { log: path, tools: { place_order: placeQtySym } }));

    expect(message).toContain(`TOOL node '${nodePath}' input 'qtty': tool 'place_order' takes no such input`);
    expect(existsSync(path)).toBe(false);
  });
}

/**
 * Return `source` with its TOOL node naming `place_ordr`.
 *
 * It also renames the `qtty` input to `qty`, so each case has only one fault: the unknown tool id.
 */
const misspelled = (source: string) =>
  source.replace("tool_id: place_order", "tool_id: place_ordr").replaceAll("qtty", "qty");

const INLINE_CALL_FLOW = `
id: inline_call
name: inline_call
nodes:
  wrap:
    kind: call
    input:
      n: 5
    nodes:
      place:
        kind: tool
        tool_id: place_order
        input:
          qty: \${input.n}
          sym: X
        output: str
    output: \${place.output}
output: \${wrap.output}
`;

const INLINE_LOOP_FLOW = `
id: inline_loop
name: inline_loop
nodes:
  again:
    kind: loop
    input:
      count: 0
      last: ""
    while: count < 1
    max: 3
    nodes:
      place:
        kind: tool
        tool_id: place_order
        input:
          qty: \${input.count}
          sym: X
        output: str
    output:
      count: \${input.count + 1}
      last: \${place.output}
output: \${again.output}
`;

const MISSPELLED =
  `names tool 'place_ordr', which isn't in this run's tools; known tools: ["place_order"]; ` +
  "did you mean 'place_order'?";

for (const [name, source, nodePath, inputs] of [
  ["root", oneToolFlow("      qty: 5\n      sym: X"), "place", {}],
  ["unreached-branch", UNREACHED_BRANCH_FLOW, "place", {}],
  ["defs-child", DEFS_CHILD_FLOW, "a/place", {}],
  ["empty-map-body", MAP_BODY_FLOW.replace("[1, 2]", "[]").replace("x: ${item}", "x: !int ${item}"), "each/place", {}],
  ["loop-body", LOOP_BODY_FLOW, "again/place", {}],
  ["inline-call-body", INLINE_CALL_FLOW, "wrap/place", {}],
  ["inline-loop-body", INLINE_LOOP_FLOW, "again/place", {}],
  ["compact-flow", ONE_NODE_FLOW, "count", { qty: 5 }],
] as const) {
  test(`an unknown tool id fails at run start: ${name}`, async () => {
    const path = freshLog();

    const message = await refused(run(misspelled(source), inputs, { log: path, tools: { place_order: placeQtySym } }));

    expect(message).toContain(`TOOL node '${nodePath}' ${MISSPELLED}`);
    expect(existsSync(path)).toBe(false);
    expect(calls).toEqual([]);
  });
}

test("an unknown tool id with no close match has no hint", async () => {
  const path = freshLog();
  const source = oneToolFlow("      qty: 5\n      sym: X").replace("place_order", "zzz");

  const message = await refused(run(source, {}, { log: path, tools: { place_order: placeQtySym } }));

  expect(message).toEndWith(
    `TOOL node 'place' names tool 'zzz', which isn't in this run's tools; known tools: ["place_order"]`,
  );
  expect(existsSync(path)).toBe(false);
});

test("a tool node naming a group fails at run start", async () => {
  const path = freshLog();
  const source = oneToolFlow("      qty: 5\n      sym: X");
  const grouped = { box: [placeQtySym] };

  const message = await refused(run(source.replace("place_order", "box"), {}, { log: path, tools: grouped }));
  expect(message).toContain("TOOL node 'place' names 'box': 'box' is a tool group, not a tool");
  expect(existsSync(path)).toBe(false);

  const result = await run(source, {}, { tools: grouped });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe("5 X");
});

const ASK_FLOW = `
id: asking
name: asking
nodes:
  ask:
    kind: tool
    tool_id: ask_user
    input:
      question: Proceed?
      options: "yes, no"
output: \${ask.output}
`;

test("a tool node naming ask user without a handler fails at run start", async () => {
  const path = freshLog();

  const message = await refused(run(ASK_FLOW, {}, { log: path, tools: {} }));

  expect(message).toContain("TOOL node 'ask' names 'ask_user': no ask_user handler for this run");
  expect(existsSync(path)).toBe(false);
});

test("without a tools mapping the registry is checked", async () => {
  const path = freshLog();
  registerTool(
    "typed_tool_registry_probe",
    tool({
      name: "typed_tool_registry_probe",
      description: "Return the order text.",
      parameters: { type: "object", properties: { qty: INT, sym: STR }, required: ["qty", "sym"] },
      output: STR,
      execute: (args) => `${args.qty} ${args.sym}`,
    }),
  );
  const source = oneToolFlow("      qty: 5\n      sym: X");

  const message = await refused(run(source.replace("place_order", "zz_unregistered_probe"), {}, { log: path }));
  expect(message).toStartWith(
    "TOOL node 'place' names tool 'zz_unregistered_probe', which isn't in this run's tools; known tools:",
  );
  expect(message).toContain('"typed_tool_registry_probe"');
  expect(existsSync(path)).toBe(false);

  const result = await run(source.replace("place_order", "typed_tool_registry_probe"), {});
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe("5 X");
});

test("resume rejects an unknown tool id before appending", async () => {
  const path = freshLog();
  const flow = await load(INT_FLOW);
  const { signal, stop, interrupt } = hostStop();
  const error = await rejected(
    runFlow(flow, {}, { log: path, env, signal, tools: { place_order: placeOrder({ qty: {} }, interrupt) } }),
  );
  expect(error).toBe(stop);
  const before = readFileSync(path);

  const message = await refused(resumeFlow(flow, { log: path, env, tools: {} }));

  expect(message).toEndWith(
    "TOOL node 'place' names tool 'place_order', which isn't in this run's tools; known tools: []",
  );
  expect(readFileSync(path).equals(before)).toBe(true);
});

/** `place_order(qty: <qty>) -> str`, recording each call. */
const placeQty = (qty: JsonObject) =>
  placeOrder(
    { qty },
    (args) => {
      calls.push(args);
      return `placed ${args.qty}`;
    },
    { output: STR },
  );

const placeInt = placeQty(INT);
const placeFloat = placeQty({ type: "number" });
const placeStr = placeQty(STR);

/** `place_order(order) -> str` over an `order` object with an int `qty`, closed when `closed`. */
const placeModel = (closed: boolean) =>
  placeOrder(
    {
      order: {
        type: "object",
        properties: { qty: INT },
        required: ["qty"],
        ...(closed ? { additionalProperties: false } : {}),
      },
    },
    (args) => `placed ${(args.order as Record<string, unknown>).qty}`,
    { output: STR },
  );

/** A JSON Schema tool taking an int `qty` and nothing else. */
const jsonSchemaTool = placeOrder({ qty: INT }, (args) => `placed ${args.qty}`, { additionalProperties: false });

/** Return a flow whose TOOL node `place` binds `qty` as `binding`. */
function typedFlow(binding: string, options: { flowInputs?: string; extraNodes?: string } = {}): string {
  return `
id: placing
name: placing
${options.flowInputs ?? ""}
nodes:
${options.extraNodes ?? ""}
  place:
    kind: tool
    tool_id: place_order
    input:
      qty: ${binding}
output: \${place.output}
`;
}

test("a disjoint input type fails at run start", async () => {
  const path = freshLog();
  const source = typedFlow("${input.s}", { flowInputs: "input:\n  s: !str" });

  const message = await refused(run(source, { s: "10" }, { log: path, tools: { place_order: placeInt } }));

  expect(message).toContain("TOOL node 'place' input 'qty': tool 'place_order' takes int, but the binding is str");
  expect(existsSync(path)).toBe(false);
  expect(calls).toEqual([]);
});

test("a string literal into an int fails at run start", async () => {
  const message = await refused(run(typedFlow('"10"'), {}, { tools: { place_order: placeInt } }));

  expect(message).toContain(`TOOL node 'place' input 'qty': tool 'place_order' takes int, but the binding is "10" (str)`);
  expect(calls).toEqual([]);
});

test("a long string literal is truncated in the message", async () => {
  const message = await refused(run(typedFlow(`"${"x".repeat(60)}"`), {}, { tools: { place_order: placeInt } }));

  expect(message).not.toContain("x".repeat(60));
  expect(message).toEndWith('..." (str)');
});

test("an int literal into a str fails at run start and shows its value", async () => {
  const message = await refused(run(typedFlow("5"), {}, { tools: { place_order: placeStr } }));

  expect(message).toMatch(/input 'qty'.*takes str, but the binding is 5 \(int\)$/);
});

test("an int literal into an int runs", async () => {
  const result = await run(typedFlow("10"), {}, { tools: { place_order: placeInt } });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe("placed 10");
});

test("an optional input starts and the tool validates none", async () => {
  const flow = await load(typedFlow("${input.n}", { flowInputs: "input:\n  n: !Optional(int)" }));

  const missing = await runFlow(flow, { n: null }, { env, tools: { place_order: placeInt } });
  const present = await runFlow(flow, { n: 3 }, { env, tools: { place_order: placeInt } });

  expect(missing.status).toBe("failed");
  expect(missing.failure!.nodeId).toBe("place");
  expect(present.status, present.failure?.message).toBe("succeeded");
  expect(present.output).toBe("placed 3");
});

test("an int into a float runs", async () => {
  const source = typedFlow("${input.n}", { flowInputs: "input:\n  n: !int" });

  const result = await run(source, { n: 2 }, { tools: { place_order: placeFloat } });

  expect(result.status, result.failure?.message).toBe("succeeded");
});

test("a typed binding over an unproven source is checked by its declared type", async () => {
  const code = `
  c:
    kind: code
    code: 'return 1'
`;
  const source = typedFlow("!int ${c.output}", { extraNodes: code });

  const message = await refused(run(source, {}, { tools: { place_order: placeStr } }));

  expect(message).toMatch(/input 'qty'.*takes str, but the binding is int$/);
});

const RECORD_FLOW = `
id: placing
name: placing
typedefs:
  Order: {qty: int, note: str}
input:
  order: !Order
nodes:
  place:
    kind: tool
    tool_id: place_order
    input:
      order: \${input.order}
output: \${place.output}
`;

test("a record with an extra field into a default model runs", async () => {
  const result = await run(RECORD_FLOW, { order: { qty: 2, note: "n" } }, { tools: { place_order: placeModel(false) } });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe("placed 2");
});

test("a record with an extra field into a closed model fails at run start", async () => {
  const message = await refused(
    run(RECORD_FLOW, { order: { qty: 2, note: "n" } }, { tools: { place_order: placeModel(true) } }),
  );

  expect(message).toContain('takes {"qty":int}, but the binding is {"note":str,"qty":int}');
});

test("a json schema tool with a disjoint binding fails at run start", async () => {
  const source = typedFlow("${input.s}", { flowInputs: "input:\n  s: !str" });

  const message = await refused(run(source, { s: "x" }, { tools: { place_order: jsonSchemaTool } }));

  expect(message).toMatch(/input 'qty'.*takes int, but the binding is str$/);
});

/** `place_order(qty: Annotated[int, Field(ge=0)]) -> str`, recording each call. */
const placeNonNegative = placeQty({ type: "integer", minimum: 0 });

test("a literal below a one sided minimum fails at run start", async () => {
  const path = freshLog();

  const message = await refused(run(typedFlow("-1"), {}, { log: path, tools: { place_order: placeNonNegative } }));

  expect(message).toContain(
    "TOOL node 'place' input 'qty': tool 'place_order' takes int(0..), but the binding is -1 (int)",
  );
  expect(existsSync(path)).toBe(false);
  expect(calls).toEqual([]);
});

test("a literal inside a one sided minimum runs", async () => {
  const result = await run(typedFlow("0"), {}, { tools: { place_order: placeNonNegative } });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe("placed 0");
});

test("an unproven int into a one sided minimum is left to the tool", async () => {
  // `int` overlaps `int(0..)`; the tool validates each call.
  const source = typedFlow("${input.n}", { flowInputs: "input:\n  n: !int" });

  const result = await run(source, { n: -1 }, { tools: { place_order: placeNonNegative } });

  expect(result.status).toBe("failed");
  expect(result.failure!.nodeId).toBe("place");
  expect(calls).toEqual([]);
});

const COUNTED_FLOW = `
id: counting
name: counting
typedefs:
  Short: {n: int(..-1)}
nodes:
  place:
    kind: tool
    tool_id: place_order
    input:
      qty: 5
    output: Short
output: \${place.output}
`;

test("an output disjoint from a one sided return field fails at run start", async () => {
  const counted: JsonObject = {
    type: "object",
    properties: { n: { type: "integer", minimum: 0 } },
    required: ["n"],
  };
  const placeCounted = placeOrder({ qty: INT }, (args) => ({ n: args.qty }), { output: counted });

  const message = await refused(run(COUNTED_FLOW, {}, { tools: { place_order: placeCounted } }));

  expect(message).toMatch(/output: declared .*, but tool 'place_order' returns \{"n":int\(0\.\.\)\}$/);
});

test("a return type disjoint from the declared output fails at run start", async () => {
  const path = freshLog();
  expect(toolTypes(placeInt).output).not.toBeNull();

  const message = await refused(run(INT_FLOW, {}, { log: path, tools: { place_order: placeInt } }));

  expect(message).toContain("TOOL node 'place' output: declared int, but tool 'place_order' returns str");
  expect(existsSync(path)).toBe(false);
  expect(calls).toEqual([]);
});

/** A closed object schema of a variant tagged `type` with `fields` besides the tag. */
function variant(tag: string, fields: Record<string, JsonObject>): JsonObject {
  return {
    type: "object",
    properties: { type: { type: "string", const: tag }, ...fields },
    required: ["type", ...Object.keys(fields)],
  };
}

/** `place_order(qty: int) -> Union[Filled, Interrupted]`. */
const placeVariant = placeOrder({ qty: INT }, (args) => ({ type: "filled", qty: args.qty }), {
  output: {
    anyOf: [
      variant("filled", { qty: INT }),
      variant("interrupted", { by: STR, partial: { anyOf: [INT, { type: "null" }] } }),
    ],
  },
});

/** `place_order(qty: int) -> Union[Rejected, Cancelled]`. */
const placeRefused = placeOrder({ qty: INT }, () => ({ type: "cancelled" }), {
  output: { anyOf: [variant("rejected", { reason: STR }), variant("cancelled", {})] },
});

test("a union return matching the declared union runs", async () => {
  expect(toolTypes(placeVariant).output).not.toBeNull();

  const result = await run(ORDERS_FLOW, {}, { tools: { place_order: placeVariant } });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe("filled 5");
});

test("a union return with no shared tag fails at run start", async () => {
  expect(toolTypes(placeRefused).output).not.toBeNull();

  const message = await refused(run(ORDERS_FLOW, {}, { tools: { place_order: placeRefused } }));

  expect(message).toMatch(
    /output: declared Union\(.*\), but tool 'place_order' returns Union\(\{"type":Literal\(cancelled\)\},\{"reason":str,"type":Literal\(rejected\)\}\)$/,
  );
});

const RECORD_OUTPUT_FLOW = `
id: placing
name: placing
nodes:
  place:
    kind: tool
    tool_id: place_order
    input:
      qty: 5
    output: {qty: int}
output: \${place.output}
`;

test("a wider return type runs and the gate still checks each value", async () => {
  const placeDict = placeOrder({ qty: INT }, (args) => ({ qty: args.qty }), { output: { type: "object" } });
  const placeBadDict = placeOrder({ qty: INT }, () => ({ qty: "bad" }), { output: { type: "object" } });
  expect(toolTypes(placeDict).output).not.toBeNull();
  const flow = await load(RECORD_OUTPUT_FLOW);

  const ok = await runFlow(flow, {}, { env, tools: { place_order: placeDict } });
  const bad = await runFlow(flow, {}, { env, tools: { place_order: placeBadDict } });

  expect(ok.status, ok.failure?.message).toBe("succeeded");
  expect(ok.output).toEqual({ qty: 5n });
  expect(bad.status).toBe("failed");
  expect(bad.failure!.code).toBe("contract.output");
});

for (const [name, unread] of [
  ["no output schema", placeOrder({ qty: INT }, () => "x")],
  ["an output schema with no yian type", placeOrder({ qty: INT }, () => ({ qty: 5 }), { output: {} })],
] as const) {
  test(`an unread return type skips the check and the gate still applies: ${name}`, async () => {
    expect(toolTypes(unread).output).toBeNull();

    const result = await run(INT_FLOW, {}, { tools: { place_order: unread } });

    expect(result.status).toBe("failed");
    expect(result.failure!.nodeId).toBe("place");
    expect(result.failure!.code).toBe("contract.output");
  });
}

test("a node without output runs any return type", async () => {
  expect(toolTypes(placeInt).output).not.toBeNull();

  const result = await run(INT_FLOW.replace("    output: int\n", ""), {}, { tools: { place_order: placeInt } });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe("placed 5");
});

const TWO_STEP_FLOW = `
id: two_step
name: two_step
nodes:
  first:
    kind: tool
    tool_id: first_step
  place:
    kind: tool
    tool_id: place_order
    depends_on: [first]
    input:
      qty: 5
output: \${place.output}
`;

/** A `first_step()` tool returning what `execute` returns. */
const firstStepTool = (execute: () => unknown) =>
  tool({
    name: "first_step",
    description: "Get ready.",
    parameters: { type: "object", properties: {} },
    output: STR,
    execute,
  });

const firstStep = firstStepTool(() => "ready");

test("resume checks tool types before appending", async () => {
  const path = freshLog();
  const flow = await load(TWO_STEP_FLOW);
  const { signal, stop, interrupt } = hostStop();
  const error = await rejected(
    runFlow(flow, {}, { log: path, env, signal, tools: { first_step: firstStepTool(interrupt), place_order: placeInt } }),
  );
  expect(error).toBe(stop);
  const lines = readFileSync(path, "utf8");

  const message = await refused(
    resumeFlow(flow, { log: path, env, tools: { first_step: firstStep, place_order: placeStr } }),
  );
  expect(message).toMatch(/input 'qty'.*takes str, but the binding is 5 \(int\)$/);
  expect(readFileSync(path, "utf8")).toBe(lines);

  const resumed = await resumeFlow(flow, { log: path, env, tools: { first_step: firstStep, place_order: placeInt } });
  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBe("placed 5");
});

test("a finished run resumes without the tool type check", async () => {
  const path = freshLog();
  const flow = await load(TWO_STEP_FLOW);
  const finished = await runFlow(flow, {}, { log: path, env, tools: { first_step: firstStep, place_order: placeInt } });
  expect(finished.status, finished.failure?.message).toBe("succeeded");

  const replayed = await resumeFlow(flow, { log: path, env, tools: { first_step: firstStep, place_order: placeStr } });

  expect(replayed.status, replayed.failure?.message).toBe("succeeded");
  expect(replayed.output).toBe("placed 5");
});
