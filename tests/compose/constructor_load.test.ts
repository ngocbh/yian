/**
 * `${T(...)}` constructors resolve at load into each node's type table.
 *
 * A constructor name is a `typedefs:` name; `T(x)` checks a value (exactly `!T ${x}`), and
 * `T(a=..., b=...)` builds a record from exactly the given fields and checks it.
 */

import { afterAll, afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { CompiledFlow } from "../../src/compile/model.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import type { Node } from "../../src/nodes/base.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import { runFlow } from "../../src/run.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import { type Type, typeEquals } from "../../src/typesys/values.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import {
  assembled,
  assembleText,
  type CaseRow,
  loadConformanceFlow,
  loadError,
  loadErrorAsync,
  runCase,
} from "../helpers/declarations.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-constructor-load-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

install(BUILTIN);
beforeEach(() => install(BUILTIN));
afterEach(forget);

const REG = readTypedefs({
  Buy: { action: "Literal(buy)", conviction: "float" },
  Sell: { action: "Literal(sell)", price: "float" },
  Decision: "Union(Buy, Sell)",
  Size: "Literal(s, m, l)",
});
const BUY = typeFor("Buy", REG);

const A_BUY = { action: "buy", conviction: 0.5 };
const A_SELL = { action: "sell", price: 1.5 };

const TYPEDEFS = `typedefs:
  Buy: {action: Literal(buy), conviction: float}
  Sell: {action: Literal(sell), price: float}
  Decision: Union(Buy, Sell)
  Size: Literal(s, m, l)
`;

// One CODE node `d` that returns the untyped `order` input, so only a check can narrow it.
const HEAD =
  "id: p\nname: p\n" +
  TYPEDEFS +
  `input:
  order: !object
nodes:
  d:
    kind: code
    input:
      order: \${input.order}
    output: object
    code: |
      return inputs["order"]
`;

const flow = (rest: string) => HEAD + rest;

/** A flow whose node `c` binds `x: <binding>` and returns it. */
const consumer = (binding: string) =>
  flow(`  c:
    kind: code
    input:
      x: ${binding}
    code: |
      return inputs["x"]
output: \${c.output}
`);

/** Return the 1-based line of the first line of `text` containing `marker`. */
function lineOf(text: string, marker: string): number {
  const index = text.split("\n").findIndex((line) => line.includes(marker));
  if (index < 0) throw new Error(`${JSON.stringify(marker)} not in text`);
  return index + 1;
}

/** Return the start line of the error's last frame. */
const errorLine = (error: { diagnostic: { frames: readonly { range: { startLine: number } }[] } }) =>
  error.diagnostic.frames.at(-1)!.range.startLine;

/** Load and run `text` on `inputs`. */
const run = async (text: string, inputs: Record<string, unknown>) =>
  runFlow(await loadConformanceFlow(text), inputs, { env });

/** Run `text` on `inputs` and return its output, expecting it succeeded. */
async function ok(text: string, inputs: Record<string, unknown>): Promise<unknown> {
  const result = await run(text, inputs);
  expect(result.status, result.failure?.message).toBe("succeeded");
  return result.output;
}

/** Assemble `text` as the root flow; return its nodes. */
const loadedNodes = (text: string) => assembleText(text).flow.compiled.nodes;

/** Check that `node`'s type table holds exactly `expected`, in any order. */
function expectTypes(node: Node | undefined, expected: Record<string, Type>): void {
  const types = node!.types;
  expect([...types.keys()].sort()).toEqual(Object.keys(expected).sort());
  for (const [name, typ] of Object.entries(expected)) expect(typeEquals(types.get(name)!, typ)).toBe(true);
}

// ---------- resolution ----------

test("a constructor resolves into the node type table", () => {
  const nodes = loadedNodes(consumer("${Buy(d.output)}"));

  expectTypes(nodes.get("c"), { Buy: BUY });
  expectTypes(nodes.get("d"), {});
});

test.each([
  ["${Buy(${d.output})}", "${Buy(d.output)}"],
  ['${Buy(action="buy", conviction=${d.output.conviction})}', '${Buy(action="buy", conviction=d.output.conviction)}'],
])("a span inside a constructor shows the rewrite: %s", (binding, rewrite) => {
  const error = loadError(() => loadedNodes(consumer(`'${binding}'`)));

  expect(error.diagnostic.message).toBe(`c: \`\${...}\` can't appear inside an expression; write \`${rewrite}\``);
});

test("a union typedef and an enum alias check", async () => {
  const text = flow(`  c:
    kind: code
    input:
      x: \${Decision(d.output.decision)}
      size: \${Size(d.output.size)}
    code: |
      return [inputs["x"]["action"], inputs["size"]]
output: \${c.output}
`);

  expect(await ok(text, { order: { decision: A_SELL, size: "m" } })).toStrictEqual(["sell", "m"]);
  const failed = await run(text, { order: { decision: A_SELL, size: "xl" } });
  expect(failed.failure!.code).toBe("binding.type");
});

test("isinstance with a builtin name does not make it a constructor", () => {
  expectTypes(loadedNodes(consumer("${isinstance(d.output, int)}")).get("c"), { int: typeFor("int", new Map()) });
});

// ---------- positions ----------

const CASE_FLOW = flow(`  route:
    kind: case
    cases:
      - when: Buy(d.output).conviction > 0.4
        then: strong
    else: weak
  strong:
    kind: code
    code: return "strong"
  weak:
    kind: code
    code: return "weak"
output: \${strong.output | weak.output}
`);

test("a constructor in a case when (loads)", () => {
  expectTypes(loadedNodes(CASE_FLOW).get("route"), { Buy: BUY });
});

test.each<[string, object, string]>([
  ["a strong buy", A_BUY, "strong"],
  ["a weak buy", { action: "buy", conviction: 0.1 }, "weak"],
])("a constructor in a case when (runs): %s", async (_name, order, expected) => {
  expect(await ok(CASE_FLOW, { order })).toBe(expected);
});

const ASSERTS_FLOW = flow(`  c:
    kind: code
    input:
      x: \${d.output}
    asserts:
      - Buy(x).conviction > 0.1
    code: |
      return inputs["x"]
output: \${c.output}
asserts:
  - \${Buy(c.output).conviction < 0.9}
`);

test("a constructor in node and flow asserts (loads)", () => {
  expectTypes(loadedNodes(ASSERTS_FLOW).get("c"), { Buy: BUY });
});

test("a constructor in node and flow asserts (runs)", async () => {
  const loaded = await loadConformanceFlow(ASSERTS_FLOW);

  expect((await runFlow(loaded, { order: A_BUY }, { env })).status).toBe("succeeded");
  expect((await runFlow(loaded, { order: { action: "buy", conviction: 0.95 } }, { env })).status).toBe("failed");
  expect((await runFlow(loaded, { order: { action: "buy", conviction: 0.05 } }, { env })).status).toBe("failed");
});

test("a constructor in a flow output (loads)", () => {
  loadedNodes(flow("output:\n  decision: ${Buy(d.output)}\n"));
});

test("a constructor in a flow output (runs)", async () => {
  const text = flow("output:\n  decision: ${Buy(d.output)}\n");

  expect(await ok(text, { order: A_BUY })).toStrictEqual({ decision: A_BUY });
});

const MAP_FLOW =
  "id: m\nname: m\n" +
  TYPEDEFS +
  `input:
  orders: !list(object)
defs:
  one:
    input:
      x: !object
    nodes:
      e:
        kind: code
        input:
          x: \${input.x}
        code: |
          return inputs["x"]["conviction"]
    output: \${e.output}
nodes:
  each:
    kind: map
    call: one
    over: \${input.orders}
    input:
      x: \${Buy(item)}
output: \${each.output}
`;

test("a constructor in a map per-item input (loads)", () => {
  expectTypes(loadedNodes(MAP_FLOW).get("each"), { Buy: BUY });
});

test("a constructor in a map per-item input (runs)", async () => {
  expect(await ok(MAP_FLOW, { orders: [A_BUY, A_BUY] })).toStrictEqual([0.5, 0.5]);
});

// ---------- typed bindings and call types over a constructor ----------

test.each([
  // `test_a_value_lacking_a_field_fails_with_binding_type_naming_it`
  "${Buy(d.output)}",
  // `test_a_bad_built_field_fails_with_binding_type`
  '${Buy(action="buy", conviction=d.output.price)}',
])("a binding whose run fails with binding.type loads: %s", (binding) => {
  loadedNodes(consumer(binding));
});

test("a nested constructor whose run fails with binding.type loads", () => {
  loadedNodes(
    flow(`  c:
    kind: code
    input:
      nested:
        first: \${Buy(d.output)}
    code: |
      return inputs["nested"]
output: \${c.output}
`),
  );
});

test("downstream sees the constructed type", () => {
  // A typed binding over a constructor compares the declared type with the constructor's.
  const error = loadError(() => loadedNodes(consumer("!Sell ${Buy(d.output)}")));

  expect(error.message).toContain('declared Sell, but the value is {"action":Literal(buy),"conviction":float}');
});

test("a call argument is checked with the constructed type", () => {
  const text = flow(`  run:
    kind: call
    call: sell
    input:
      order: \${Buy(d.output)}
output: \${run.output}
`).replace(
    "nodes:\n",
    `defs:
  sell:
    input:
      order: !Sell
    nodes:
      e:
        kind: code
        input:
          order: \${input.order}
        code: |
          return inputs["order"]
    output: \${e.output}
nodes:
`,
  );

  expect(loadError(() => loadedNodes(text)).message).toContain("call node 'run': binding 'order'");
});

const PARENT =
  "id: p\nname: p\n" +
  TYPEDEFS +
  `input:
  order: !object
defs:
  decide:
    input:
      order: !object
    nodes:
      e:
        kind: code
        input:
          order: \${input.order}
        code: |
          return inputs["order"]
    output: \${Buy(e.output)}
nodes:
  run:
    kind: call
    call: decide
    input:
      order: \${input.order}
  use:
    kind: code
    input:
      conviction: \${run.output.conviction}
    code: |
      return inputs["conviction"]
output: \${use.output}
`;

test("a caller sees a child's constructed output type (loads)", async () => {
  await loadFlow(PARENT);
  const text = PARENT.replace("${run.output.conviction}", "${run.output.size}");

  expect((await loadErrorAsync(loadFlow(text))).message).toContain("reads unknown field 'size' on record");
});

test("a caller sees a child's constructed output type (runs)", async () => {
  expect(await ok(PARENT, { order: A_BUY })).toBe(0.5);
});

const INLINE_FLOW = flow(`  inline:
    kind: call
    input:
      decision: \${Buy(d.output)}
    nodes:
      e:
        kind: code
        input:
          c: \${input.decision.conviction}
        code: |
          return inputs["c"]
    output: \${e.output}
output: \${inline.output}
`);

test("an inline body parameter takes the constructed type (loads)", () => {
  const child = (loadedNodes(INLINE_FLOW).get("inline") as CallNode).child as CompiledFlow;

  expect(typeEquals(child.nodes.get(child.startId)!.params![0]!.contract!.type, BUY)).toBe(true);
});

test("an inline body parameter takes the constructed type (runs)", async () => {
  expect(await ok(INLINE_FLOW, { order: A_BUY })).toBe(0.5);
});

const USES_CHILD = `id: child
name: child
typedefs:
  Hold: {action: Literal(hold)}
input:
  order: !object
nodes:
  c:
    kind: code
    input:
      x: \${Hold(input.order)}
    code: |
      return inputs["x"]
output: \${c.output}
`;

const USES_PARENT = `id: parent
name: parent
input:
  order: !object
uses:
  child: child
nodes:
  run:
    kind: call
    call: child
    input:
      order: \${input.order}
output: \${run.output}
`;

/** A resolver that hands back the loaded `USES_CHILD` for `child`. */
async function usesResolver(): Promise<(flowId: string) => ComposedFlow> {
  const loaded: Record<string, ComposedFlow> = { child: await loadFlow(USES_CHILD) };
  return (flowId) => loaded[flowId]!;
}

test("a uses child resolves constructors against its own typedefs (loads)", async () => {
  const loaded = await loadFlow(USES_PARENT, { childResolver: await usesResolver() });

  const child = (loaded.compiled.nodes.get("run") as CallNode).child as CompiledFlow;
  expect(child.nodes.get("c")!.types.has("Hold")).toBe(true);
});

test("a uses child resolves constructors against its own typedefs (runs)", async () => {
  const loaded = await loadConformanceFlow(USES_PARENT, { childResolver: await usesResolver() });

  const result = await runFlow(loaded, { order: { action: "hold" } }, { env });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toStrictEqual({ action: "hold" });
});

test("a call with several arguments to an unknown name is an unknown function, not a constructor", () => {
  const error = loadError(() => loadedNodes(consumer("${review(d.output, d.output)}")));

  expect(error.message).toBe("c: unknown expression function 'review'");
});

test.each([
  ["binding", consumer('${Buy(d.output, action="buy")}')],
  ["case when", CASE_FLOW.replace("Buy(d.output)", "Buy(d.output, d.output)")],
])("a malformed constructor call is a load error: %s", (_, text) => {
  expect(loadError(() => loadedNodes(text)).message).toContain(
    "Buy(...) takes keyword fields, as in Buy(name=value, ...), or one value to check, as in Buy(x)",
  );
});

test("a parent cannot construct a child's typedef", async () => {
  const text = USES_PARENT.replace(
    "      order: ${input.order}\n",
    "      order: ${input.order}\n  probe:\n    kind: code\n    input:\n" +
      "      x: ${Hold(input.order)}\n    code: |\n      return 1\n",
  );

  const error = await loadErrorAsync(loadFlow(text, { childResolver: await usesResolver() }));

  expect(error.message).toContain("unknown expression function 'Hold'");
});

// ---------- builtin arguments ----------

/** A flow whose node `c` binds `x: <binding>`, with typed flow inputs to read. */
const typedConsumer = (binding: string) => `id: p
name: p
typedefs:
  R: {k: int}
input:
  rec: !R
  recs: !list(R)
  orecs: !Optional(list(R))
  n: !int
  s: !str
  xs: !list(str)
  maybe: !Optional(int)
  any: !object
nodes:
  c:
    kind: code
    input:
      x: ${binding}
    code: |
      return inputs["x"]
output: \${c.output}
`;

describe("a float-typed value may hold an int, so it passes where an int goes", () => {
  const text = `id: p
name: p
input:
  f: !float
  xs: !list(int)
nodes:
  c:
    kind: code
    input:
      tail: \${input.xs[input.f:]}
      item: \${input.xs[input.f]}
      json: \${render_as_json(input.xs, input.f)}
      rounded: \${round(2.25, input.f)}
    code: |
      return [inputs["tail"], inputs["item"], inputs["json"], inputs["rounded"]]
output: \${c.output}
`;

  test("the flow loads and runs with an int for the float", async () => {
    expect(await ok(text, { f: 1n, xs: [1n, 2n, 3n] })).toStrictEqual([[2n, 3n], 2n, "[\n 1,\n 2,\n 3\n]", 2.2]);
  });
});

describe("a field read after a subscript or parentheses is checked by its base's type", () => {
  test.each([
    ["${input.recs[0].nope}", "reference ${input.recs[...].nope} reads unknown field 'nope' on record"],
    ["${(input.rec).nope}", "reference ${input.rec.nope} reads unknown field 'nope' on record"],
    ["${input.xs[0].k}", "reference ${input.xs[...].k} reads field 'k' on type str, which has no fields"],
    [
      "${input.orecs[0].k}",
      "${input.orecs[...].k} reads field 'k' of input.orecs[...], which may be null; write input.orecs[...]?.k",
    ],
  ])("%s", (binding, message) => {
    const error = loadError(() => loadedNodes(typedConsumer(binding)));
    expect(error.message).toContain(`node 'c' input 'x' from: ${message}`);
  });

  test.each(["${input.recs[0].k}", "${input.orecs[0]?.k}", "${(input.rec).k + 1}", "${input.any[0].k}"])(
    "%s loads",
    (binding) => {
      expect(() => loadedNodes(typedConsumer(binding))).not.toThrow();
    },
  );
});

describe("a subscript of a kind it never takes is a load error", () => {
  test.each([
    [
      "${input.rec[0]}",
      'indexing takes a list or text, got {"k":int}; read a record\'s field with a dot, as in `r.name`',
    ],
    ["${input.n[1:]}", "slicing takes a list or text, got int"],
    ['${input.xs["a"]}', "an index takes int, got Literal(a)"],
    ["${input.xs[1.5:]}", "slice bounds take int, got float(1.5..1.5)"],
  ])("%s", (binding, message) => {
    const error = loadError(() => loadedNodes(typedConsumer(binding)));
    expect(error.message).toBe(`flow has expressions that can never succeed:\n  node 'c' input 'x' from: ${message}`);
  });

  test.each(["${input.xs[0]}", "${input.s[-1]}", "${input.s[1:]}", "${input.any[0]}", "${input.xs[input.n]}"])(
    "%s loads",
    (binding) => {
      expect(() => loadedNodes(typedConsumer(binding))).not.toThrow();
    },
  );
});

describe("a builtin argument of a kind the builtin never takes is a load error", () => {
  test.each([
    ["${upper(input.n)}", "upper() takes str, got int"],
    ["${upper(input.maybe)}", "upper() takes str, got Optional(int)"],
    ["${upper(null)}", "upper() takes str, got null"],
    ["${sum(input.xs)}", "sum() takes a list of numbers, got list(str)"],
    ["${min(input.n)}", "min() takes a list of numbers or texts, got int"],
    ["${split(input.s, sep=input.n)}", "split() takes str for sep, got int"],
    ["${int(input.xs)}", "int() takes str, int, float or bool, got list(str)"],
    ["${len(5)}", "len() takes str or list, got int(5..5)"],
    ['${round("2.5")}', 'round() takes a number, got Literal("2.5")'],
  ])("%s", (binding, message) => {
    const error = loadError(() => loadedNodes(typedConsumer(binding)));
    expect(error.message).toBe(`flow has expressions that can never succeed:\n  node 'c' input 'x' from: ${message}`);
  });

  test.each([
    "${upper(input.s)}",
    "${upper(input.any.name)}",
    '${upper(input.maybe :- "")}',
    "${round(input.n, null)}",
    "${split(input.s, null)}",
    "${str(input.xs)}",
    "${sum([1, 2.5, true])}",
  ])("%s loads: the argument may be one", (binding) => {
    expect(() => loadedNodes(typedConsumer(binding))).not.toThrow();
  });

  test("in a condition, a literal argument is checked", () => {
    const text = CASE_FLOW.replace("Buy(d.output).conviction > 0.4", 'upper(5) == "5"');
    const error = loadError(() => loadedNodes(text));
    expect(error.message).toBe("case node 'route' cases[0].when: upper() takes str, got int(5..5)");
  });
});

// ---------- names that are not constructors ----------

test("a builtin type name in a binding is not a conversion", () => {
  const text = consumer("${bool(d.output)}");

  const error = loadError(() => loadedNodes(text));

  expect(error.message).toContain(
    "unknown expression function 'bool'; types in expressions are typedef names; `bool(...)` is not a conversion",
  );
  expect(errorLine(error)).toBe(lineOf(text, "x: ${bool"));
});

test("Message, a built-in type, is not a constructor", () => {
  const error = loadError(() => loadedNodes(consumer("${Message(d.output)}")));
  expect(error.message).toContain(
    "unknown expression function 'Message'; types in expressions are typedef names; `Message(...)` is not a conversion",
  );
});

test("a builtin type name in a case when is not a conversion", () => {
  const text = CASE_FLOW.replace("Buy(d.output).conviction > 0.4", "bool(d.output) > 0");

  const error = loadError(() => loadedNodes(text));

  expect(error.message).toContain("case node 'route' cases[0].when: unknown expression function 'bool'; types");
  expect(errorLine(error)).toBe(lineOf(text, "bool(d.output)"));
});

test("an unknown name in a case when is an unknown function", () => {
  const text = CASE_FLOW.replace("Buy(d.output)", "Hold(d.output)");

  expect(loadError(() => loadedNodes(text)).message.split("\n")[0]).toMatch(/unknown expression function 'Hold'$/);
});

test("a constructor in a sleep's until is a binding, checked as a date", () => {
  // A sleep's until: is a TOOL input, where a constructor builds a value; its type must fit.
  const text =
    "id: w\nname: w\n" +
    TYPEDEFS +
    `input:
  when: !date
nodes:
  w:
    kind: tool
    tool_id: sleep
    input:
      until: \${Size(input.when)}
output: \${w.output}
`;

  const error = loadError(() => loadedNodes(text));

  expect(error.message).toContain(
    "node 'w' (tool_id: sleep) input 'until': must be a date or datetime, but the value is Literal(l, m, s)",
  );
  expect(errorLine(error)).toBe(lineOf(text, "until:"));
});

test("a constructor in a call node's asserts is a located load error", () => {
  const text =
    "id: p\nname: p\n" +
    TYPEDEFS +
    `input:
  order: !object
defs:
  check:
    input:
      order: !object
    nodes:
      e:
        kind: code
        input:
          order: \${input.order}
        code: |
          return inputs["order"]
    output: \${e.output}
nodes:
  run:
    kind: call
    call: check
    input:
      order: \${input.order}
    asserts:
      - Buy(output).conviction > 0
output: \${run.output}
`;

  const error = loadError(() => loadedNodes(text));

  expect(error.message).toBe(
    "flow has invalid node asserts:\n" +
      "  node 'run' assert 'Buy(output).conviction > 0': Buy(...) in a call node's asserts is not " +
      "supported; assert inside the child flow",
  );
  expect(errorLine(error)).toBe(lineOf(text, "Buy(output)"));
});

test("a constructor in an agent prompt is a located load error", () => {
  const text =
    "id: a\nname: a\n" +
    TYPEDEFS +
    `input:
  order: !object
nodes:
  ask:
    kind: agent
    input:
      order: \${input.order}
    prompt: "Is it a buy? \${Buy(order)}"
    output: str
output: \${ask.output}
`;

  const error = loadError(() => loadedNodes(text));

  expect(error.message).toContain("unknown prompt function 'Buy'");
  expect(error.message).toContain(
    "build or check a typed value with a `typedefs:` constructor `T(...)` in an input binding",
  );
  expect(errorLine(error)).toBe(lineOf(text, "prompt:"));
});

// ---------- static types ----------

/** Like `HEAD`, with `d` declaring `dOutput`. */
const typedHead = (dOutput: string) => HEAD.replace("    output: object\n", `    output: ${dOutput}\n`);

test("a field read on a constructor result type checks", () => {
  loadedNodes(consumer("${Buy(d.output).conviction}"));
  const text = consumer("${Buy(d.output).size}");

  const error = loadError(() => loadedNodes(text));

  expect(error.message).toContain("reference ${Buy(...).size} reads unknown field 'size' on record");
  expect(errorLine(error)).toBe(lineOf(text, "x: ${Buy"));
});

test("a field read on a constructor in a case when type checks", () => {
  const text = CASE_FLOW.replace("Buy(d.output).conviction", "Buy(d.output).size");

  const error = loadError(() => loadedNodes(text));

  expect(error.message).toContain("reads unknown field 'size' on record");
  expect(errorLine(error)).toBe(lineOf(text, "Buy(d.output).size"));
});

test("a disjoint argument is a load error", () => {
  const text =
    typedHead("Sell") +
    `  c:
    kind: code
    input:
      x: \${Buy(d.output)}
    code: |
      return inputs["x"]
output: \${c.output}
`;

  const error = loadError(() => loadedNodes(text));

  expect(error.message).toContain(
    'Buy(...): the value is {"action":Literal(sell),"price":float}, which can never be a Buy',
  );
  expect(errorLine(error)).toBe(lineOf(text, "x: ${Buy"));
});

test("a narrower or unknown argument loads", () => {
  loadedNodes(typedHead("Decision") + "output: ${Buy(d.output)}\n");
  loadedNodes(typedHead("Optional(Buy)") + "output: ${Buy(d.output)}\n");
});

test("keywords missing a required field are a load error", () => {
  const text = consumer("${Buy(conviction=d.output.conviction)}");

  const error = loadError(() => loadedNodes(text));

  expect(error.message).toContain("Buy(...) is missing required field `action`");
  expect(errorLine(error)).toBe(lineOf(text, "x: ${Buy"));
});

test("keywords missing a field in a case when are a load error", () => {
  const text = CASE_FLOW.replace("Buy(d.output)", "Buy(conviction=d.output.conviction)");

  expect(loadError(() => loadedNodes(text)).message).toContain("Buy(...) is missing required field `action`");
});

test("an unknown keyword field is a load error", () => {
  const text = consumer('${Buy(action="buy", conviction=1, size="m")}');

  expect(loadError(() => loadedNodes(text)).message).toContain("Buy has no field 'size' (fields: action, conviction)");
});

test.each([
  [
    '${Buy(action="sell", conviction=1)}',
    "Buy(...) field 'action': declared Literal(buy), but the value is Literal(sell)",
  ],
  ['${Buy(action="buy", conviction=d.output)}', "Buy(...) field 'conviction': declared float, but the value is str"],
])("a disjoint keyword field is a load error: %s", (binding, message) => {
  const text =
    typedHead("str") +
    `  c:
    kind: code
    input:
      x: ${binding}
    code: |
      return inputs["x"]
output: \${c.output}
`;

  expect(loadError(() => loadedNodes(text)).message).toContain(message);
});

test("keywords cannot build a union", () => {
  const error = loadError(() => loadedNodes(consumer("${Decision(conviction=1)}")));

  expect(error.message).toContain(
    "Decision is a union; construct a member (Buy(...) or Sell(...)), or check a value with Decision(x)",
  );
});

test("keywords cannot build a scalar type", () => {
  const error = loadError(() => loadedNodes(consumer('${Size(name="m")}')));

  expect(error.message).toContain("Size is not a record, so Size(...) takes one value to check, as in Size(x)");
});

describe("constructor messages at a case when", () => {
  test.each([
    [
      "Decision(conviction=1).conviction > 0.4",
      "Decision is a union; construct a member (Buy(...) or Sell(...)), or check a value with Decision(x)",
    ],
    ['Size(name="m") == "m"', "Size is not a record, so Size(...) takes one value to check, as in Size(x)"],
    [
      'Buy(action="buy", conviction=1, size="m").conviction > 0.4',
      "Buy has no field 'size' (fields: action, conviction)",
    ],
    ['Buy("x").conviction > 0.4', "Buy(...): the value is Literal(x), which can never be a Buy"],
    [
      'Buy(d.output).conviction > 0.4 and Size(3) == "m"',
      "Size(...): the value is int(3..3), which can never be a Size",
    ],
    [
      'Buy(action="sell", conviction=1).conviction > 0.4',
      "Buy(...) field 'action': declared Literal(buy), but the value is Literal(sell)",
    ],
    [
      'Buy(conviction=true, action="buy").conviction > 0.4',
      "Buy(...) field 'conviction': declared float, but the value is bool",
    ],
    ["upper(d.output).size > 0", "reference ${upper(...).size} reads field 'size' on type str, which has no fields"],
  ])("%s", (when, message) => {
    const text = CASE_FLOW.replace("Buy(d.output).conviction > 0.4", when);

    const error = loadError(() => loadedNodes(text));

    expect(error.message).toBe(`case node 'route' cases[0].when: ${message}`);
    expect(errorLine(error)).toBe(lineOf(text, "- when:"));
  });
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>(
  [
    "a_builtin_type_name_in_a_case_when_is_not_a_conversion",
    "a_constructor_in_wait_until",
    "a_field_read_on_a_constructor_in_a_case_when",
    "a_malformed_constructor_call_in_a_case_when",
    "an_unknown_name_in_a_case_when_is_an_unknown_function",
    "keywords_missing_a_field_in_a_case_when",
    "a_constructor_in_an_agent_prompt",
    "a_disjoint_argument_is_a_load_error",
    "a_disjoint_keyword_field_is_a_load_error",
    "a_field_read_on_a_constructor_result_type_checks",
    "a_malformed_constructor_call_in_a_binding",
    "an_unknown_keyword_field_is_a_load_error",
    "keywords_cannot_build_a_scalar_type",
    "keywords_cannot_build_a_union",
    "keywords_missing_a_required_field",
    "downstream_sees_the_constructed_type",
  ].map((name) => ({ id: `load-types/compose_constructor_load__${name}`, call: assembled })),
)("assembled: $id", runCase);

test("assembled: load-types/compose_constructor_load__a_constructor_in_a_call_nodes_asserts", () => {
  runCase({ id: "load-types/compose_constructor_load__a_constructor_in_a_call_nodes_asserts", call: assembled });
});
