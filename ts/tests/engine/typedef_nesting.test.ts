/**
 * How `readTypedefs` walks typedef references and bounds their nesting, and how deep a type
 * may nest once its typedef names are expanded, a port of `tests/engine/test_typedef_nesting.py`.
 * The flow-loading tests run their typedefs through `readTypedefs` and `typeFor` here, except
 * the located load errors (`test_a_flow_with_a_long_chain_is_a_located_load_error`,
 * `test_a_flow_wrapping_a_deep_typedef_is_a_located_load_error`), which run through `loadFlow`,
 * the CODE node naming the conformance fixture `echo`; a stack overflow (Python's
 * `RecursionError`) is checked for in the error's causes.
 * Not ported: `test_field_parse_follows_the_int_digit_limit` (D4a: no digit limits);
 * `test_measuring_a_diamond_is_memoized` and the measure half of
 * `test_an_empty_record_and_an_enum_load` call private functions;
 * `test_measured_levels_match_the_resolved_type` checks the measured levels through
 * `readTypedefs` instead.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { BUILTIN, forget, install, reload } from "../../src/defaults.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import { TypeCheckError, typeLevels } from "../../src/typesys/values.ts";
import { loadConformanceFlow, loadErrorAsync } from "../compose/declarations.ts";

// Start each test from the built-in values, whatever `YIAN_*` variables the shell sets.
beforeEach(() => install(BUILTIN));
afterEach(forget);

type Raw = Record<string, unknown>;

/** Return the `TypeCheckError` `readTypedefs(raw)` throws. */
function typedefsError(raw: Raw): TypeCheckError {
  try {
    readTypedefs(raw);
  } catch (error) {
    expect(error).toBeInstanceOf(TypeCheckError);
    return error as TypeCheckError;
  }
  throw new Error("readTypedefs did not throw");
}

/** Expect `readTypedefs(raw)` to throw exactly `message` at `path`. */
function expectRejected(raw: Raw, message: string, path: readonly string[]): void {
  const error = typedefsError(raw);
  expect(error.message).toBe(message);
  expect(error.path).toEqual(path);
}

/** Return `n` records, `T0: {a: int}` and each next one `{a: <previous>}`. */
function records(n: number): Raw {
  const raw: Raw = { T0: { a: "int" } };
  for (let i = 1; i < n; i++) raw[`T${i}`] = { a: `T${i - 1}` };
  return raw;
}

/** Return `n` aliases over `bottom`, each the previous name wrapped by `wrap`. */
function aliases(n: number, prefix = "A", bottom: unknown = "int", wrap = (t: string) => t): Raw {
  const raw: Raw = { [`${prefix}0`]: bottom };
  for (let i = 1; i < n; i++) raw[`${prefix}${i}`] = wrap(`${prefix}${i - 1}`);
  return raw;
}

/** Return `name` wrapped in `n` nested `ctor(...)`s. */
function nest(ctor: string, n: number, name: string): string {
  return `${ctor}(`.repeat(n) + name + ")".repeat(n);
}

/** Return a flow declaring `typedefs` whose CODE node `use` has `output: <output>`. */
function flow(typedefs: Record<string, string>, output: string): string {
  const lines = Object.entries(typedefs)
    .map(([name, defn]) => `  ${name}: ${defn}\n`)
    .join("");
  return (
    `id: f\nname: f\ntypedefs:\n${lines}nodes:\n  use:\n    kind: code\n` +
    `    code: ./fixtures/compose_codefns.ts:echo\n    input:\n      topic: x\n` +
    `    output: ${output}\noutput: \${use.output}\n`
  );
}

/** Set the type nesting limit to `limit` through `YIAN_MAX_TYPE_NESTING`. */
function setTypeNesting(limit: number): void {
  reload((name) => (name === "YIAN_MAX_TYPE_NESTING" ? String(limit) : undefined));
}

/** The levels error for record `name`'s `field` of type `text`, which adds `adds` levels. */
function levelsMessage(
  name: string,
  field: string,
  text: string,
  adds: number,
  limit = 100,
): string {
  return (
    `typedef '${name}' field '${field}': type nests deeper than ${limit} levels, the limit for a ` +
    `type (each record and list is a level); its type '${text}' adds ${adds}`
  );
}

/** The chain error for typedef `name` (at `field`, if given) ending a chain of `chain`. */
function chainMessage(name: string, chain: number, field?: string, limit = 100): string {
  const where = field === undefined ? `'${name}'` : `'${name}' field '${field}'`;
  return (
    `typedef ${where}: it ends a chain of ${chain} typedefs, each naming the next, more than ` +
    `${limit}, the limit for a type; write its type with fewer typedefs in between`
  );
}

describe("reference walks", () => {
  test("a long alias chain fails on its length quickly", () => {
    const start = performance.now();
    const error = typedefsError(aliases(5000));
    expect(performance.now() - start).toBeLessThan(5000);
    expect(error.message.startsWith("typedef 'A100': it ends a chain of 101 typedefs")).toBe(true);
  });

  test.each([10, 200, 1000])("a record cycle of %p reports at the last typedef", (n) => {
    const raw: Raw = {};
    for (let i = 0; i < n; i++) raw[`T${i}`] = { a: `T${(i + 1) % n}` };
    const last = `T${n - 1}`;
    expectRejected(raw, `typedef '${last}' field 'a': recursive type reference: '${last}'`, [
      last,
      "a",
    ]);
  });

  test("a self-recursive record", () => {
    expectRejected({ R: { a: "R" } }, "typedef 'R' field 'a': recursive type reference: 'R'", [
      "R",
      "a",
    ]);
  });

  test("a two-record cycle", () => {
    expectRejected(
      { A: { x: "B" }, B: { y: "A" } },
      "typedef 'B' field 'y': recursive type reference: 'B'",
      ["B", "y"],
    );
  });

  test("a cycle through an alias and a record", () => {
    expectRejected(
      { A: "B", B: { x: "A" } },
      "typedef 'B' field 'x': recursive type reference: 'B'",
      ["B", "x"],
    );
  });

  test("an alias cycle", () => {
    expectRejected({ A: "B", B: "A" }, "alias cycle through 'A' in typedefs", ["A"]);
  });

  test("field order decides between a union error and a cycle", () => {
    expectRejected(
      { U1: { x: "int" }, U2: { x: "int" }, R: { a: "Union(U1, U2)", b: "R" } },
      "typedef 'R' field 'a': Union(U1, U2): members 'U1' and 'U2' can match the same " +
        "value; add a tag field every member declares as a single-value Literal, with a " +
        "different value in each member, e.g. kind: Literal(u1)",
      ["R", "a"],
    );
  });

  test("aliases wrapped in Optional resolve", () => {
    readTypedefs(aliases(20, "A", { a: "int" }, (t) => nest("Optional", 99, t)));
  });

  test("aliases wrapped in Union resolve", () => {
    readTypedefs(aliases(100, "A", { a: "int" }, (t) => nest("Union", 99, t)));
  });
});

/**
 * Return `limit` record levels, each above the bottom one holding an optional union of the
 * level below and `S`; the bottom field nests `limit` Optionals. Also return the top name.
 */
function unionLevels(limit: number, tagged: boolean): [Raw, string] {
  const tag = tagged ? { k: "Literal(r)" } : {};
  const raw: Raw = {
    S: tagged ? { k: "Literal(s)" } : { x: "int" },
    T0: { ...tag, a: nest("Optional", limit, "int") },
  };
  for (let i = 1; i < limit; i++) raw[`T${i}`] = { ...tag, a: `Optional(Union(T${i - 1}, S))` };
  return [raw, `T${limit - 1}`];
}

/**
 * Return typedefs like `unionLevels` with `limit - 1` records over a bottom field of `limit`
 * nested unions of `S`, so `S` is the last of `limit` levels, and the top name.
 */
function unionsOverADeepUnion(limit: number): [Raw, string] {
  const raw: Raw = {
    S: { k: "Literal(s)" },
    T0: { k: "Literal(r)", a: nest("Union", limit, "S") },
  };
  for (let i = 1; i < limit - 1; i++) {
    raw[`T${i}`] = { k: "Literal(r)", a: `Optional(Union(T${i - 1}, S))` };
  }
  return [raw, `T${limit - 2}`];
}

/**
 * Return `limit - 1` records, each the only field of the next, over a bottom field of
 * `limit` nested unions of `S`, and the top name.
 */
function recordsOverADeepUnion(limit: number): [Raw, string] {
  const raw: Raw = { S: { x: "int" }, T0: { a: nest("Union", limit, "S") } };
  for (let i = 1; i < limit - 1; i++) raw[`T${i}`] = { a: `T${i - 1}` };
  return [raw, `T${limit - 2}`];
}

/** Return `n` records, each holding the previous one in two fields. */
function diamond(n: number): Raw {
  const raw: Raw = { D0: { x: "int" } };
  for (let i = 1; i < n; i++) raw[`D${i}`] = { x: `D${i - 1}`, y: `D${i - 1}` };
  return raw;
}

describe("types at the limit", () => {
  const builds = [
    ["tagged unions", (limit: number) => unionLevels(limit, true)],
    ["untagged unions", (limit: number) => unionLevels(limit, false)],
    ["unions over a union", unionsOverADeepUnion],
    ["records over a union", recordsOverADeepUnion],
  ] as const;
  const rows = [BUILTIN.maxTypeNesting, 125].flatMap((limit) =>
    builds.map(([id, build]) => [id, limit, build] as const),
  );
  test.each(rows)("%s load at the limit %p", (_id, limit, build) => {
    setTypeNesting(limit);
    const [raw, top] = build(limit);
    const start = performance.now();
    expect(typeLevels(typeFor(top, readTypedefs(raw)))).toBe(limit);
    expect(performance.now() - start).toBeLessThan(5000);
  });

  test("a diamond loads", () => {
    expect(typeLevels(typeFor("D11", readTypedefs(diamond(12))))).toBe(12);
  });

  test("an empty record and an enum load", () => {
    readTypedefs({ E: {}, V: "Literal(a)" });
  });

  test("field parse follows the type nesting limit", () => {
    const raw = { R: { a: nest("list", 6, "int") } };
    readTypedefs(raw);
    setTypeNesting(5);
    expect(() => readTypedefs(raw)).toThrow(TypeCheckError);
  });
});

/** Render each typedef of `raw` as the flow-style YAML `flow` writes. */
function flowLines(raw: Raw): Record<string, string> {
  return Object.fromEntries(
    Object.entries(raw).map(([name, defn]) => [
      name,
      typeof defn === "string"
        ? defn
        : `{${Object.entries(defn as Record<string, string>)
            .map(([field, type]) => `${field}: ${type}`)
            .join(", ")}}`,
    ]),
  );
}

/** Whether `error` or one of its causes is a stack overflow. */
function overflowed(error: unknown): boolean {
  for (let cause = error; cause !== undefined; cause = (cause as Error).cause) {
    if (String(cause).includes("Maximum call stack size exceeded")) return true;
  }
  return false;
}

describe("levels and chains", () => {
  test.each([101, 1000])("a record chain of %p fails on levels", (n) => {
    expectRejected(records(n), levelsMessage("T100", "a", "T99", 100), ["T100", "a"]);
  });

  test("a record chain at the limit loads", () => {
    readTypedefs(records(100));
  });

  test.each([101, 1000])("an alias chain of %p fails on its length", (n) => {
    expectRejected(aliases(n), chainMessage("A100", 101), ["A100"]);
  });

  test("an alias chain at the limit loads", () => {
    readTypedefs(aliases(100));
  });

  test("an Optional alias chain fails on its length", () => {
    const raw = aliases(101, "O", "int", (t) => `Optional(${t})`);
    expectRejected(raw, chainMessage("O100", 101), ["O100"]);
  });

  test("a list alias chain over a scalar fails on its length", () => {
    const raw = aliases(101, "L", "int", (t) => `list(${t})`);
    expectRejected(raw, chainMessage("L100", 101), ["L100"]);
  });

  test("a list alias chain over a record fails on levels", () => {
    const raw = aliases(101, "L", { a: "int" }, (t) => `list(${t})`);
    expectRejected(
      raw,
      "typedef 'L100': type nests deeper than 100 levels, the limit for a type (each record " +
        "and list is a level)",
      ["L100"],
    );
  });

  test("a list over a deep record fails on levels", () => {
    const error = typedefsError({ ...records(100), T: "list(T99)" });
    expect(error.message.startsWith("typedef 'T': type nests deeper than 100 levels")).toBe(true);
    expect(error.path).toEqual(["T"]);
  });

  test("a record field holding a long alias chain fails on the chain", () => {
    expectRejected(
      { ...aliases(100), R: { x: "int", a: "A99" } },
      chainMessage("R", 101, "a"),
      ["R", "a"],
    );
  });

  test("a lower limit bounds records and aliases", () => {
    setTypeNesting(5);
    readTypedefs(records(5));
    readTypedefs(aliases(5));
    expect(typedefsError(records(6)).message).toBe(levelsMessage("T5", "a", "T4", 5, 5));
    expect(typedefsError(aliases(6)).message).toBe(chainMessage("A5", 6, undefined, 5));
  });

  test.each([
    ["1000-records", records(1000), "bad typedefs: typedef 'T100' field 'a'", "  T100: {a: ".length + 1],
    ["300-records", records(300), "bad typedefs: typedef 'T100' field 'a'", "  T100: {a: ".length + 1],
    ["1000-aliases", aliases(1000), "bad typedefs: typedef 'A100'", "  ".length + 1],
  ] as const)("a flow with a long chain is a located load error: %s", async (_id, raw, prefix, column) => {
    const last = Object.keys(raw).at(-1)!;
    const error = await loadErrorAsync(loadConformanceFlow(flow(flowLines(raw), last)));
    expect(error.message.startsWith(prefix)).toBe(true);
    // Line 4 holds the first typedef, so the 101st is on line 104.
    const range = error.diagnostic.frames.at(-1)!.range;
    expect([range.startLine, range.startCol]).toEqual([104, column]);
    expect(overflowed(error)).toBe(false);
  });

  // Each typedef's levels as Python's `_measure_typedef` measures them.
  const MEASURED: Raw = {
    E: {},
    V: "Literal(a, b)",
    B: "int(1..5)",
    R: { a: "int", b: "list(str)", c: "Optional(E)" },
    S: { k: "Literal(s)", r: "R" },
    U: "Union(R, S)",
    W: { u: "U", t: "topics", d: "dict(str, int)", o: "object", v: "V" },
    L: "list(list(W))",
    M: { l: "Optional(L)", b: "B", e: "list(E)" },
  };
  test.each([
    ["E", 1],
    ["V", 0],
    ["B", 0],
    ["R", 2],
    ["S", 3],
    ["U", 3],
    ["W", 4],
    ["L", 6],
    ["M", 7],
  ] as const)("the measured levels of %s match the resolved type", (name, levels) => {
    expect(typeLevels(typeFor(name, readTypedefs(MEASURED)))).toBe(levels);
    readTypedefs({ ...MEASURED, Probe: { f: nest("list", 99 - levels, name) } });
    const deeper = nest("list", 100 - levels, name);
    expectRejected(
      { ...MEASURED, Probe: { f: deeper } },
      levelsMessage("Probe", "f", deeper, 100),
      ["Probe", "f"],
    );
  });
});

/** Return `name` wrapped in `lists` nested `list(...)`s. */
function wrapped(name: string, lists = 100): string {
  return nest("list", lists, name);
}

const EXPANDED =
  "type nests deeper than 100 levels once its typedef names are expanded, the limit for a " +
  "type (each record and list is a level)";

/** Return the 1-based line of the first `needle` in `text`. */
function lineOf(text: string, needle: string): number {
  return text.slice(0, text.indexOf(needle)).split("\n").length;
}

/** Return a flow whose CODE node passes `topic` an empty list tagged `!<tag>`. */
function nodeInputTagFlow(typedefs: Record<string, string>, tag: string): string {
  return flow(typedefs, "T0").replace("      topic: x\n", `      topic: !${tag} []\n`);
}

/** Return a flow declaring a flow input `t` with the tag `!<tag>`. */
function flowInputTagFlow(typedefs: Record<string, string>, tag: string): string {
  return flow(typedefs, "T0").replace("name: f\n", `name: f\ninput:\n  t: !${tag}\n`);
}

describe("typeFor nesting", () => {
  test("bounds a type with its typedefs expanded", () => {
    const registry = readTypedefs(records(100));
    for (const text of [wrapped("T99"), wrapped("T99", 1)]) {
      expect(() => typeFor(text, registry)).toThrow(TypeCheckError);
      expect(() => typeFor(text, registry)).toThrow(EXPANDED);
    }
  });

  test("a type at the limit resolves", () => {
    expect(typeLevels(typeFor(wrapped("T98", 1), readTypedefs(records(99))))).toBe(100);
    typeFor(wrapped("int"), new Map());
  });

  test.each([
    ["output", (typedefs: Record<string, string>, t: string) => flow(typedefs, `'${t}'`), "    output: 'list("],
    ["node-input-tag", nodeInputTagFlow, "      topic: !list("],
    ["flow-input-tag", flowInputTagFlow, "  t: !list("],
  ] as const)("a flow wrapping a deep typedef is a located load error: %s", async (_id, build, needle) => {
    const text = build(flowLines(records(100)), wrapped("T99"));
    const error = await loadErrorAsync(loadConformanceFlow(text));
    expect(error.message).toContain(EXPANDED);
    expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(lineOf(text, needle));
    expect(overflowed(error)).toBe(false);
  });
});
