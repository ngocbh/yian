/**
 * Flows that use `Any`.
 *
 * `Any` is any JSON value except null; `Optional(Any)` adds null. Wiring an `Any` into a narrower
 * child input is a load error unless a tag or an `isinstance` branch narrows it.
 */

import { afterAll, afterEach, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { forget } from "../../src/defaults.ts";
import { runFlow } from "../../src/run.ts";
import { hostEnv } from "../engine/mcp_helpers.ts";
import { assembled, assembleText, loadConformanceFlow, loadError, runCase } from "./declarations.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-any-flows-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

afterEach(() => forget());

/**
 * A flow taking `s: !str` whose CODE node `mk` returns `code`, typed `output` (untyped when
 * `null`), then the `rest` nodes; `head` holds top-level blocks before `input:`.
 */
function flow(
  output: string | null,
  rest: string,
  options: { code?: string; out?: string; head?: string } = {},
): string {
  const { code = "null", out = "${mk.output}", head = "" } = options;
  const typed = output === null ? "" : `    output: ${output}\n`;
  return (
    `id: p\nname: p\n${head}input:\n  s: !str\nnodes:\n  mk:\n    kind: code\n` +
    `    input:\n      s: \${input.s}\n${typed}    code: |\n      return ${code}\n${rest}output: ${out}\n`
  );
}

/** A `defs:` entry `name` taking `x: !<inputType>` and returning its text. */
const childDef = (name: string, inputType: string) =>
  `  ${name}:\n    input:\n      x: !${inputType}\n    nodes:\n      n:\n        kind: code\n` +
  "        input:\n          x: ${input.x}\n        output: str\n" +
  "        code: return JSON.stringify(inputs['x'])\n    output: ${n.output}\n";

/** A CALL node `nodeId` calling `callee` with `x: <binding>`. */
const call = (binding: string, nodeId = "c", callee = "child") =>
  `  ${nodeId}:\n    kind: call\n    call: ${callee}\n    input:\n      x: ${binding}\n`;

/** Load and run `text` with `inputs` (default `{s: "x"}`). */
const run = async (text: string, inputs: Record<string, unknown> = { s: "x" }) =>
  runFlow(await loadConformanceFlow(text), inputs, { env });

/** Run `text` and return its output, expecting it succeeded. */
async function ok(text: string, inputs?: Record<string, unknown>): Promise<unknown> {
  const result = await run(text, inputs);
  expect(result.status, result.failure?.message).toBe("succeeded");
  return result.output;
}

// ---------- node outputs ----------

test.each<[string, unknown]>([
  ['"hello"', "hello"],
  ["3", 3n],
  ['{"a": 1}', { a: 1n }],
  ["[1]", [1n]],
])("an Any output returns any JSON value: %s", async (code, value) => {
  expect(await ok(flow("Any", "", { code }))).toStrictEqual(value);
});

// ---------- field reads on Any ----------

test("a field read on an Any output loads in a binding (loads)", () => {
  const rest = "  use:\n    kind: code\n    input:\n      v: ${mk.output.a.b}\n    code: return inputs['v']\n";

  assembleText(flow("Any", rest, { code: '{"a": {"b": 2}}', out: "${use.output}" }));
});

test("a field read on an Any output loads in a binding (runs)", async () => {
  const rest = "  use:\n    kind: code\n    input:\n      v: ${mk.output.a.b}\n    code: return inputs['v']\n";

  expect(await ok(flow("Any", rest, { code: '{"a": {"b": 2}}', out: "${use.output}" }))).toBe(2n);
});

test("a field read on an Any output loads in a flow output (loads)", () => {
  assembleText(flow("Any", "", { code: '{"a": 1}', out: "\n  a: ${mk.output.a}" }));
});

test("a field read on an Any output loads in a flow output (runs)", async () => {
  expect(await ok(flow("Any", "", { code: '{"a": 1}', out: "\n  a: ${mk.output.a}" }))).toStrictEqual({ a: 1n });
});

/** `route` reading `mk.output.a` (an `Any` output) in a `when:`, taking `hit` or `miss`. */
const FIELD_CASE =
  "  route:\n    kind: case\n    cases:\n" +
  "      - when: ${mk.output.a == 1}\n        then: hit\n" +
  "    else: miss\n" +
  "  hit:\n    kind: code\n    output: str\n    code: return 'hit'\n" +
  "  miss:\n    kind: code\n    output: str\n    code: return 'miss'\n";

test("a field read on an Any output loads in a case condition (loads)", () => {
  assembleText(flow("Any", FIELD_CASE, { code: '{"a": 1}', out: "${hit.output | miss.output}" }));
});

test("a field read on an Any output loads in a case condition (runs)", async () => {
  expect(await ok(flow("Any", FIELD_CASE, { code: '{"a": 1}', out: "${hit.output | miss.output}" }))).toBe("hit");
});

/** A flow with value entry `entry` and a CODE node reading `read`. */
const entryFlow = (entry: string, read: string) =>
  `id: p\nname: p\nnodes:\n  ${entry}\n  use:\n    kind: code\n    input:\n      v: ${read}\n` +
  "    code: return inputs['v']\noutput: ${use.output}\n";

test("a field read on an Any value entry is not checked", () => {
  assembleText(entryFlow("v: !Any {a: 1}", "${v.output.zz}"));
});

test("a field read on an object value entry is still checked", () => {
  const error = loadError(() => assembleText(entryFlow("w: !object {a: 1}", "${w.output.zz}")));

  expect(error.message).toContain("reads unknown field 'zz'");
});

// ---------- CALL wiring ----------

/** `mk` typed `output` returning `code`, passed to `child` (taking `x: !<childType>`) as `binding`. */
const callFlow = (output: string, binding: string, code: string, childType: string) =>
  flow(output, call(binding), { code, out: "${c.output}", head: `defs:\n${childDef("child", childType)}` });

test.each([
  ["fails at runtime for an int", "3"],
  ["runs for text", '"hi"'],
])("a tagged Any binding into a child str input %s (loads)", (_name, code) => {
  assembleText(callFlow("Any", "!str ${mk.output}", code, "str"));
});

test("a tagged Any binding into a child str input runs for text (runs)", async () => {
  expect(await ok(callFlow("Any", "!str ${mk.output}", '"hi"', "str"))).toBe('"hi"');
});

test("a str output into a child Any input loads and runs (loads)", () => {
  assembleText(callFlow("str", "${mk.output}", '"hi"', "Any"));
});

test("a str output into a child Any input loads and runs (runs)", async () => {
  expect(await ok(callFlow("str", "${mk.output}", '"hi"', "Any"))).toBe('"hi"');
});

test("an optional str output into a child Any input is a load error", () => {
  const text = flow("Optional(str)", call("${mk.output}"), {
    out: "${c.output}",
    head: `defs:\n${childDef("child", "Any")}`,
  });

  const error = loadError(() => assembleText(text));

  expect(error.message).toBe("call node 'c': binding 'x' — child expects 'Any', source is 'Optional(str)'");
});

// ---------- narrowing with isinstance ----------

/** A flow routing on `isinstance(mk.output, str)` with `bindingNode` as `go`. */
function narrowed(bindingNode: string): string {
  const rest =
    "  route:\n    kind: case\n    cases:\n" +
    "      - when: isinstance(mk.output, str)\n        then: go\n" +
    "    else: other\n" +
    bindingNode +
    "  other:\n    kind: code\n    output: str\n    code: return 'other'\n";
  return flow("Any", rest, {
    code: '"hi"',
    out: "${go.output | other.output}",
    head: `defs:\n${childDef("child", "str")}`,
  });
}

test("an untagged Any into a child str input loads inside an isinstance branch (loads)", () => {
  assembleText(narrowed(call("${mk.output}", "go")));
});

test("an untagged Any into a child str input loads inside an isinstance branch (runs)", async () => {
  expect(await ok(narrowed(call("${mk.output}", "go")))).toBe('"hi"');
});

test("the same untagged wiring outside the branch is a load error", () => {
  const text = narrowed(call("${mk.output}", "go") + call("${mk.output}", "c2"));

  const message = loadError(() => assembleText(text)).message;

  // `go` is inside the branch, so only `c2` fails.
  expect(message).toBe(
    "call node 'c2': binding 'x' — child expects 'str', source is 'Any'; tag the slot " +
      "(`x: !str ${...}`) to check it at runtime, or narrow it with `isinstance` in a `case`",
  );
});

// ---------- MAP over lists of Any ----------

/** A flow mapping `child` (taking `x: !<childType>`) over `over`. */
function map(overOutput: string | null, childType: string, code: string, over = "${mk.output}"): string {
  const rest = `  m:\n    kind: map\n    over: ${over}\n    call: child\n    input:\n      x: \${item}\n`;
  return flow(overOutput, rest, { code, out: "${m.output}", head: `defs:\n${childDef("child", childType)}` });
}

const MAP_ROWS: [string, string | null, string, string, string | undefined, string[]][] = [
  ["a list of Any runs items into a child Any input", "list(Any)", "Any", '[1, "two"]', undefined, ["1", '"two"']],
  [
    "a list of optional Any into a child optional Any input runs",
    "list(Optional(Any))",
    "Optional(Any)",
    "[1, null]",
    undefined,
    ["1", "null"],
  ],
  ["a mixed literal list into a child Any input runs", null, "Any", "null", "['a', 1]", ['"a"', "1"]],
];

test.each(MAP_ROWS)("a map over %s (loads)", (_name, overOutput, childType, code, over) => {
  assembleText(map(overOutput, childType, code, over));
});

test.each(MAP_ROWS)("a map over %s (runs)", async (_name, overOutput, childType, code, over, value) => {
  expect(await ok(map(overOutput, childType, code, over))).toStrictEqual(value);
});

test("a map over a list of optional Any into a child Any input is a load error", () => {
  const error = loadError(() => assembleText(map("list(Optional(Any))", "Any", "[1]")));

  expect(error.message).toBe("map node 'm': binding 'x' — child expects 'Any', source is 'Optional(Any)'");
});

test("a map over a mixed literal list still checks each element at load", () => {
  const error = loadError(() => assembleText(map(null, "str", "null", "['a', 1]")));

  expect(error.message).toBe(
    "map node 'm': binding 'x' — child expects 'str', but the element 1 does not fit; " +
      'quote it to keep it as text: "1"',
  );
});

// ---------- LOOP carried Any ----------

/** A flow with an inline loop carrying `acc`, seeded `seed`, from a body node typed `bodyOutput`. */
const inlineLoop = (seed: string, bodyOutput: string, code: string) =>
  "id: p\nname: p\nnodes:\n" +
  `  loop:\n    kind: loop\n    times: 2\n    input:\n      acc: ${seed}\n    nodes:\n` +
  "      step:\n        kind: code\n        input:\n          acc: ${input.acc}\n" +
  `        output: ${bodyOutput}\n        code: |\n          return ${code}\n` +
  "    output:\n      acc: ${step.output}\noutput: ${loop.output}\n";

test.each([
  ["a loop carrying an Any seed with an Any body field runs", "Any", "[inputs['acc']]"],
  ["a typed loop seed accepts an optional Any body field", "Optional(Any)", "inputs['acc'] + 1"],
  ["a null from an optional Any body field fails the next turn", "Optional(Any)", "null"],
])("%s (loads)", (_name, bodyOutput, code) => {
  assembleText(inlineLoop("!Any 0", bodyOutput, code));
});

test("a typed loop seed accepts an optional Any body field (runs)", async () => {
  expect(await ok(inlineLoop("!Any 0", "Optional(Any)", "inputs['acc'] + 1"), {})).toStrictEqual({ acc: 2n });
});

test("an untyped seed into an Any body input rejects an optional Any body field", () => {
  const text = inlineLoop("${mk.output}", "Optional(Any)", "inputs['acc']").replace(
    "nodes:\n",
    "nodes:\n  mk:\n    kind: code\n    output: Any\n    code: return 0\n",
  );

  const error = loadError(() => assembleText(text));

  expect(error.message).toBe(
    "loop node 'loop': the body returns 'acc' as Optional(Any), but the carried field is typed Any from its seed; " +
      "declare the seed's type: `acc: !Optional(Any) ${mk.output}`",
  );
});

// ---------- CASE output Any over branches of different types ----------

/** Return a flow whose `route` case picks the value entry `one` or `a`, read by `use`. */
const routed = (caseOutput = "", reader = "!Any ${route.output}") =>
  "id: p\nname: p\ninput:\n  s: !str\nnodes:\n" +
  `  route:\n    kind: case\n${caseOutput}    cases:\n` +
  "      - when: ${input.s == 'i'}\n        then: one\n" +
  "    else: a\n" +
  "  one: !int 1\n" +
  '  a: !str "a"\n' +
  `  use:\n    kind: code\n    input:\n      v: ${reader}\n` +
  "    code: return inputs['v']\n" +
  "output: ${use.output}\n";

test("a reader tagging a case result Any gets the taken branch value (loads)", () => {
  assembleText(routed());
});

test("a reader tagging a case result Any gets the taken branch value (runs)", async () => {
  expect(await ok(routed(), { s: "i" })).toBe(1n);
  expect(await ok(routed(), { s: "x" })).toBe("a");
});

test("a case output Any is a load error suggesting the reader tag", () => {
  const error = loadError(() => assembleText(routed("    output: Any\n", "${route.output}")));

  expect(error.message).toBe(
    "case node 'route': a `case` routes and has no `output:`; tag its result where you read it, " +
      "e.g. `x: !Any ${route.output}`",
  );
});

// ---------- CASE on an Any value ----------

/** Return a flow whose `route` matches `on: ${mk.output}` (`Any`) against `when`. */
function onAny(when: string, code = "1"): string {
  const rest =
    "  route:\n    kind: case\n    on: ${mk.output}\n    cases:\n" +
    `      - when: ${when}\n        then: hit\n` +
    "    else: miss\n" +
    "  hit:\n    kind: code\n    output: str\n    code: return 'hit'\n" +
    "  miss:\n    kind: code\n    output: str\n    code: return 'miss'\n";
  return flow("Any", rest, { code, out: "${hit.output | miss.output}" });
}

test("an int when on an Any value loads and matches (loads)", () => {
  assembleText(onAny("1", "1"));
  assembleText(onAny("1", '"1"'));
});

test("an int when on an Any value loads and matches (runs)", async () => {
  expect(await ok(onAny("1", "1"))).toBe("hit");
  expect(await ok(onAny("1", '"1"'))).toBe("miss");
});

test("a text when on an Any value loads and matches (loads)", () => {
  assembleText(onAny("a", '"a"'));
});

/** `mk` returning `{"a": 1}` as `Any`, with the node assert `output.a == 1`. */
const assertFlow = () =>
  flow("Any", "", { code: '{"a": 1}' }).replace("    code: |", "    asserts:\n      - output.a == 1\n    code: |");

test("a field read on an Any output loads in an assert (loads)", () => {
  // The node assert reads `output.a` of an `Any` output: the walk lets any field through.
  assembleText(assertFlow());
});

test("a field read on an Any output loads in an assert (runs)", async () => {
  expect(await ok(assertFlow())).toStrictEqual({ a: 1n });
});

// ---------- child flow output Any ----------

const ANY_CHILD_DEF =
  "  mk_any:\n    input:\n      x: !str\n    nodes:\n" +
  "      n:\n        kind: code\n        input:\n          x: ${input.x}\n" +
  "        code: return 3\n    output: !Any ${n.output}\n";

/** A flow calling `mk_any`, then `child` (taking `consumerType`) on its output. */
function fromAnyChild(consumerType: string): string {
  const rest = call("${input.s}", "g", "mk_any") + call("${g.output}");
  return flow(null, rest, { out: "${c.output}", head: `defs:\n${ANY_CHILD_DEF}${childDef("child", consumerType)}` });
}

test("a child Any output into a parent Any consumer runs (loads)", () => {
  assembleText(fromAnyChild("Any"));
});

test("a child Any output into a parent Any consumer runs (runs)", async () => {
  expect(await ok(fromAnyChild("Any"))).toBe("3");
});

test("a child Any output into a parent int input is a load error with the hint", () => {
  const error = loadError(() => assembleText(fromAnyChild("int")));

  expect(error.message).toBe(
    "call node 'c': binding 'x' — child expects 'int', source is 'Any'; " +
      "tag the slot (`x: !int ${...}`) to check it at runtime, or narrow it with `isinstance` in a `case`",
  );
});

// ---------- typed slot literals ----------

/** A flow whose CODE node `n` takes `x: <binding>` and returns it. */
const literal = (binding: string) =>
  `id: p\nname: p\nnodes:\n  n:\n    kind: code\n    input:\n      x: ${binding}\n` +
  "    code: return inputs['x']\noutput: ${n.output}\n";

test.each<[string, unknown]>([
  ["!Any 3", 3n],
  ['!Any [1, "two"]', [1n, "two"]],
  ["!Any {a: 1}", { a: 1n }],
  ["!Optional(Any) null", null],
])("a typed Any literal loads and runs: %s", async (binding, value) => {
  expect(await ok(literal(binding), {})).toStrictEqual(value);
});

// ---------- inline bodies ----------

test.each<[string, string, unknown]>([
  ["list(Any)", '[1, "two"]', [1n, "two"]],
  ["list(Optional(Any))", "[1, null]", [1n, null]],
])("a list of Any into an untagged inline body parameter runs: %s", async (output, code, value) => {
  const rest =
    "  c:\n    kind: call\n    input:\n      xs: ${mk.output}\n    nodes:\n" +
    "      inner:\n        kind: code\n        input:\n          xs: ${input.xs}\n" +
    "        code: return inputs['xs']\n    output: ${inner.output}\n";

  expect(await ok(flow(output, rest, { code, out: "${c.output}" }))).toStrictEqual(value);
});

// ---------- harvested conformance cases ----------

test("assembled: load-types/compose_any_flows__a_map_over_a_mixed_literal_list_checks_each_element", () => {
  runCase({ id: "load-types/compose_any_flows__a_map_over_a_mixed_literal_list_checks_each_element", call: assembled });
});

test("assembled: load-types/compose_any_flows__an_untyped_seed_rejects_an_optional_any_body_field", () => {
  runCase({ id: "load-types/compose_any_flows__an_untyped_seed_rejects_an_optional_any_body_field", call: assembled });
});
