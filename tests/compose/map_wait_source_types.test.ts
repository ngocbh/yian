/**
 * A MAP `over:` of the wrong type is a load error at that value.
 *
 * Literals are checked where the node is parsed; references and expressions by their inferred type
 * once the graph is built.
 */

import { afterEach, expect, test } from "bun:test";

import type { LoadError } from "../../src/compose/helpers/errors.ts";
import { valueTypeError } from "../../src/compose/nodes/base.ts";
import { forget } from "../../src/defaults.ts";
import {
  assembled,
  assembleText,
  type CaseRow,
  loadError,
  runCase,
} from "./declarations.ts";

afterEach(() => forget());

const OVER_FIX = "write a list, or a reference such as `over: ${input.items}`";

const overMessage = (readAs: string, node = "m") =>
  `node '${node}' (kind=map): \`over:\` must be a list, but ${readAs}; ${OVER_FIX}`;


/** Load `text`, expecting a load error, and return it. */
const errorOf = (text: string): LoadError => loadError(() => assembleText(text));

/** Load `text`. */
const load = (text: string): void => {
  assembleText(text);
};

/** Return the 1-based line of the first line of `text` holding `marker`. */
function lineOf(text: string, marker: string): number {
  const index = text.split("\n").findIndex((line) => line.includes(marker));
  if (index < 0) throw new Error(`${JSON.stringify(marker)} not in text`);
  return index + 1;
}

/** Return the start line of the error's last frame. */
const startLine = (error: LoadError) => error.diagnostic.frames.at(-1)!.range.startLine;

const ITEMS_DEF = `  items:
    nodes:
      c:
        kind: code
        output: list(str)
        code: |
          return ["a", "b"]
    output: \${c.output}
`;

/**
 * A flow whose MAP node `m` has `over: <value>` and calls `child`. Each flow builder substitutes
 * through a function, since a replacement string reads `$$` as `$`.
 */
function mapFlow(value: string): string {
  return (
    `id: f
name: f
defs:
  child:
    input:
      x: !Any
    nodes:
      echo:
        kind: code
        input:
          x: \${input.x}
        code: |
          return inputs["x"]
    output: \${echo.output}
` +
    ITEMS_DEF +
    `nodes:
  m:
    kind: map
    over: @V
    call: child
    input:
      x: 1
output: \${m.output}
`
  ).replace("@V", () => value);
}

// ---------- literals, checked where the node is parsed ----------

test("value type error wording", () => {
  expect(valueTypeError("over", "a list", 5n, "F")).toBe("`over:` must be a list, but YAML read it as an integer; F");
  expect(valueTypeError("over", "a list", null, "F")).toBe("`over:` must be a list, but it is empty; F");
});

test.each<[string, string]>([
  ["5", "an integer"],
  ["0", "an integer"],
  ["true", "a boolean"],
  ["false", "a boolean"],
  ["1.5", "a float"],
  ["{x: 1}", "a mapping"],
  ["{}", "a mapping"],
  ["2026-01-01", "text"],
  ["2026-01-01T10:00:00", "text"],
])("a non-list over literal %s is a load error", (value, phrase) => {
  const text = mapFlow(value);
  const error = errorOf(text);
  expect(error.diagnostic.message).toBe(overMessage(`YAML read it as ${phrase}`));
  expect(startLine(error)).toBe(lineOf(text, "over: "));
});

test("a null over is empty", () => {
  const text = mapFlow("null");
  const error = errorOf(text);
  expect(error.diagnostic.message).toBe(overMessage("it is empty"));
  expect(startLine(error)).toBe(lineOf(text, "over: "));
});

test.each<[string, string]>([
  ["word", "ok"],
  ["int", "'5'"],
  ["date", "'2026-01-01'"],
  ["escape", "'$${x}'"],
  ["quoted-call", "'call(items)'"],
  ["spaced-call", "' call(items)'"],
  ["block-call", "|\n      call(items)"],
])("a text over is a load error (%s)", (_id, value) => {
  const text = mapFlow(value);
  const error = errorOf(text);
  expect(error.diagnostic.message).toBe(overMessage("YAML read it as text"));
  expect(startLine(error)).toBe(lineOf(text, "over: "));
});

test.each(["[1, 2]", "[]"])("a list over literal %s loads", (value) => {
  load(mapFlow(value));
});

test("a call directive over is left to the graph stage", () => {
  load(mapFlow("call(items)"));
});

test("the literal check reaches defs", () => {
  const text = `id: f
name: f
defs:
  child:
    input:
      x: !Any
    nodes:
      echo:
        kind: code
        input:
          x: \${input.x}
        code: |
          return inputs["x"]
    output: \${echo.output}
  outer:
    nodes:
      inner:
        kind: map
        over: 5
        call: child
        input:
          x: 1
    output: \${inner.output}
nodes:
  a:
    kind: call
    call: outer
output: \${a.output}
`;
  const error = errorOf(text);
  expect(error.message).toContain(overMessage("YAML read it as an integer", "inner"));
  expect(startLine(error)).toBe(lineOf(text, "over: 5"));
});

test("a tagged wrong literal reports the type first", () => {
  expect(errorOf(mapFlow("!int 5")).diagnostic.message).toBe(overMessage("YAML read it as an integer"));
});

// ---------- references and expressions, checked by their inferred type ----------

const MODE = "typedefs:\n  Mode: Literal(ok, no)\n";

const overRefMessage = (label: string, node = "m") =>
  `node '${node}' (kind=map): \`over:\` must be a list, but the value is ${label}; write a reference to a list`;


/**
 * A flow with `inputs` whose MAP node `m` has `over: <over>` and calls `child`. A code node `c`
 * with `output: int` and one `u` with no declared output are available as producers.
 */
function mapRef(over: string, inputs = "  s: !str\n", typedefs = ""): string {
  return (
    "id: f\nname: f\n" +
    typedefs +
    "input:\n" +
    inputs +
    `defs:
  child:
    input:
      x: !Any
    nodes:
      echo:
        kind: code
        input:
          x: \${input.x}
        code: |
          return inputs["x"]
    output: \${echo.output}
  words:
    nodes:
      c:
        kind: code
        output: str
        code: |
          return "a"
    output: \${c.output}
nodes:
  c:
    kind: code
    output: int
    code: |
      return 1
  u:
    kind: code
    code: |
      return [1]
  m:
    kind: map
    over: @V
    call: child
    input:
      x: 1
output: \${m.output}
`
  ).replace("@V", () => over);
}

test.each<[string, string, string, string]>([
  ["str", "${input.a}", "  a: !str\n", "str"],
  ["int", "${input.a}", "  a: !int\n", "int"],
  ["bool", "${input.a}", "  a: !bool\n", "bool"],
  ["object", "${input.a}", "  a: !object\n", "object"],
  ["date", "${input.a}", "  a: !date\n", "date"],
  ["optional-int", "${input.a}", "  a: !Optional(int)\n", "Optional(int)"],
  ["optional-str", "${input.a}", "  a: !Optional(str)\n", "Optional(str)"],
  ["literal", "${input.a}", "  a: !Mode\n", "Literal(no, ok)"],
  ["template", "'a ${input.s}'", "  s: !str\n", "str"],
  ["typed-producer", "${c.output}", "  s: !str\n", "int"],
])("a disjoint over reference is a load error (%s)", (_id, over, inputs, label) => {
  const text = mapRef(over, inputs, MODE);
  const error = errorOf(text);
  expect(error.diagnostic.message).toBe(overRefMessage(label));
  expect(startLine(error)).toBe(lineOf(text, "over: "));
});

test.each<[string, string]>([
  ["over-list", mapRef("${input.a}", "  a: !list(str)\n")],
  ["over-optional-list", mapRef("${input.a}", "  a: !Optional(list(str))\n")],
  ["over-any", mapRef("${input.a}", "  a: !Any\n")],
  ["over-list-expression", mapRef("${[input.s]}")],
  ["over-arithmetic", mapRef("${input.n + 1}", "  n: !int\n")],
  ["over-untyped-producer", mapRef("${u.output}")],
])("a fitting or unproven source loads (%s)", (_id, text) => {
  load(text);
});

test("a call directive over with a str output is a load error", () => {
  const text = mapRef("call(words)");
  const error = errorOf(text);
  expect(error.diagnostic.message).toBe(overRefMessage("str"));
  expect(startLine(error)).toBe(lineOf(text, "over: "));
});

test("isinstance keeps its own error", () => {
  const text = mapRef("${isinstance(input.a, int)}", "  a: !int\n");
  const error = errorOf(text);
  expect(error.message).toContain("isinstance is not supported in a map node's over:");
  expect(startLine(error)).toBe(lineOf(text, "over: "));
});

test("a dangling reference keeps its error", () => {
  expect(errorOf(mapRef("${input.nope}")).diagnostic.message).toBe(
    "flow has unresolved references:\n" +
      "  node 'm' map over: reference ${input.nope} points to unknown flow input 'nope'",
  );
});

test("a record typedef is named", () => {
  const text = mapRef("${input.r}", "  r: !Buy\n", "typedefs:\n  Buy: {action: str}\n");
  const error = errorOf(text);
  expect(error.diagnostic.message).toBe(overRefMessage("Buy"));
  expect(startLine(error)).toBe(lineOf(text, "over: "));
});

test("a None-typed source is a load error", () => {
  expect(errorOf(mapRef("${input.z}", "  z: !None\n")).diagnostic.message).toBe(overRefMessage("None"));
});

/** A flow whose CASE routes to `m` only in union member `branch` of `d`'s output. */
function caseRouted(branch: string): string {
  const node = "  m:\n    kind: map\n    over: ${d.output.items}\n    call: child\n    input:\n      x: 1\n";
  return `id: p
name: p
typedefs:
  A: {tag: 'Literal(a)', items: list(str), when: date}
  B: {tag: 'Literal(b)', items: int, when: int}
  U: Union(A, B)
nodes:
  d:
    kind: code
    output: U
    code: return None
  r:
    kind: case
    on: \${d.output.tag}
    cases:
      - when: ${branch}
        then: m
    else: z
${node}  z:
    kind: code
    code: return 1
output: 1
defs:
  child:
    input:
      x: !Any
    nodes:
      e:
        kind: code
        input:
          x: \${input.x}
        code: return 1
    output: \${e.output}
`;
}

test("a case branch narrows the source", () => {
  load(caseRouted("a"));
  expect(errorOf(caseRouted("b")).diagnostic.message).toBe(overRefMessage("int"));
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  {
    id: "load-map-call/compose_map_wait_source_types__call_directive_over_with_a_str_output",
    call: assembled,
  },
  { id: "load-map-call/compose_map_wait_source_types__int_over_reference", call: assembled },
  {
    id: "load-map-call/compose_map_wait_source_types__record_typedef_over_reference_names_the_typedef",
    call: assembled,
  },
])("assemble: $id", runCase);
