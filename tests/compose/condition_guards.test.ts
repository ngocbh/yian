/** Guards: what a condition proves, and what holds at each of its reads. */

import { describe, expect, test } from "bun:test";

import {
  AllOf,
  AnyOf,
  allOf,
  anyOf,
  type Check,
  checkIn,
  conjuncts,
  falsified,
  type Guard,
  guardChecks,
  guardedReads,
  guardedRefs,
  Holds,
  IsinstanceCheck,
  literalsInContext,
  literalText,
  masked,
  NEVER,
  NOTHING,
  NotNullCheck,
  proves,
  sameCheck,
  sameGuard,
  TagCheck,
  unknownCheck,
} from "../../src/compose/helpers/condition_guards.ts";
import { referenceText } from "../../src/expr/expressions.ts";
import { conditionReads, exprRefs, parseCondition } from "../../src/expr/index.ts";

const A = new IsinstanceCheck("v", "Buy");
const B = new TagCheck("v", "action", "sell");
const C = new IsinstanceCheck("w", "Done");

// The source of each of A, B and C as a condition.
const [SA, SB, SC] = ["isinstance(v, Buy)", "v.action == 'sell'", "isinstance(w, Done)"];

/** What the condition `text` proves when it evaluates to `value`. */
function p(text: string, value = true): Guard {
  return proves(parseCondition(text), value);
}

/** Each reference occurrence of `text` with its guard. */
function guards(text: string): [string, Guard][] {
  return guardedRefs(parseCondition(text));
}

/** The guard of the only occurrence of `ref` in `text`. */
function guardOf(text: string, ref: string): Guard {
  const found = guards(text)
    .filter(([path]) => path === ref)
    .map(([, guard]) => guard);
  expect(found).toHaveLength(1);
  return found[0]!;
}

// --------------------------------------------------------------------------- //
// Guard constructors
// --------------------------------------------------------------------------- //

test("constructors flatten and drop nothing", () => {
  const [a, b, c] = [new Holds(A, true), new Holds(B, false), new Holds(C, true)];
  expect(allOf(a, allOf(b, c))).toStrictEqual(new AllOf([a, b, c]));
  expect(allOf(a, NOTHING, b)).toStrictEqual(new AllOf([a, b]));
  expect(allOf(a)).toStrictEqual(a);
  expect(allOf()).toStrictEqual(NOTHING);
  expect(anyOf(a, anyOf(b, c))).toStrictEqual(new AnyOf([a, b, c]));
  expect(anyOf(a, NOTHING)).toStrictEqual(NOTHING);
  expect(anyOf(a)).toStrictEqual(a);
});

// --------------------------------------------------------------------------- //
// checkIn
// --------------------------------------------------------------------------- //

describe("checkIn recognizes isinstance, tag and null comparisons", () => {
  const rows: [string, [Check, boolean] | null][] = [
    ["isinstance(d.output, Buy)", [new IsinstanceCheck("d.output", "Buy"), true]],
    ["${isinstance(d.output, Buy)}", [new IsinstanceCheck("d.output", "Buy"), true]],
    ['v.action == "buy"', [new TagCheck("v", "action", "buy"), true]],
    ['"buy" == v.action', [new TagCheck("v", "action", "buy"), true]],
    ["v.action != 'buy'", [new TagCheck("v", "action", "buy"), false]],
    ["d.output.leg.type == 'limit'", [new TagCheck("d.output.leg", "type", "limit"), true]],
    ["v.action == 'a\\b'", [new TagCheck("v", "action", "a\\b"), true]],
    ["v.qty > 0", null],
    ['v == "x"', null],
    ["isinstance(f(x), Buy)", null],
    ["v.action == other.action", null],
    ["v.action == 1", null],
    ["v.action == null", [new NotNullCheck("v.action"), false]],
    ["v.action == true", null],
    ["upper(v).action == 'buy'", null],
    ["not isinstance(v, Buy)", null],
    ["n > 1", null],
  ];
  for (const [text, expected] of rows) {
    test(text, () => {
      expect(checkIn(parseCondition(text))).toStrictEqual(expected);
    });
  }
});

// --------------------------------------------------------------------------- //
// proves
// --------------------------------------------------------------------------- //

test("proves a check and its negation", () => {
  expect(p(SA)).toStrictEqual(new Holds(A, true));
  expect(p(SA, false)).toStrictEqual(new Holds(A, false));
  expect(p(`not ${SA}`)).toStrictEqual(new Holds(A, false));
  expect(p(`not ${SA}`, false)).toStrictEqual(new Holds(A, true));
  expect(p(`not not ${SA}`)).toStrictEqual(new Holds(A, true));
  expect(p("v.action != 'sell'")).toStrictEqual(new Holds(B, false));
  expect(p("v.action != 'sell'", false)).toStrictEqual(new Holds(B, true));
});

test("proves and", () => {
  expect(p(`${SA} and ${SB}`)).toStrictEqual(new AllOf([new Holds(A, true), new Holds(B, true)]));
  expect(p(`${SA} and ${SB}`, false)).toStrictEqual(
    new AnyOf([new Holds(A, false), new AllOf([new Holds(A, true), new Holds(B, false)])]),
  );
});

test("proves or", () => {
  expect(p(`${SA} or ${SB}`)).toStrictEqual(
    new AnyOf([new Holds(A, true), new AllOf([new Holds(A, false), new Holds(B, true)])]),
  );
  expect(p(`${SA} or ${SB}`, false)).toStrictEqual(new AllOf([new Holds(A, false), new Holds(B, false)]));
});

test("proves not of and and or", () => {
  expect(p(`not (${SA} and ${SB})`)).toStrictEqual(p(`${SA} and ${SB}`, false));
  expect(p(`not (${SA} or ${SB})`)).toStrictEqual(new AllOf([new Holds(A, false), new Holds(B, false)]));
});

test("proves nested groups", () => {
  expect(p(`(${SA} or ${SB}) and ${SC}`)).toStrictEqual(
    new AllOf([
      new AnyOf([new Holds(A, true), new AllOf([new Holds(A, false), new Holds(B, true)])]),
      new Holds(C, true),
    ]),
  );
  expect(p(`${SA} and (${SB} or ${SC})`)).toStrictEqual(
    new AllOf([
      new Holds(A, true),
      new AnyOf([new Holds(B, true), new AllOf([new Holds(B, false), new Holds(C, true)])]),
    ]),
  );
});

test("proves a chain", () => {
  expect(p(`${SA} and ${SB} and ${SC}`)).toStrictEqual(
    new AllOf([new Holds(A, true), new Holds(B, true), new Holds(C, true)]),
  );
  expect(p(`${SA} and ${SB} and ${SC}`, false)).toStrictEqual(
    new AnyOf([
      new Holds(A, false),
      new AllOf([new Holds(A, true), new Holds(B, false)]),
      new AllOf([new Holds(A, true), new Holds(B, true), new Holds(C, false)]),
    ]),
  );
});

test("proves nothing for other expressions", () => {
  expect(p("n > 1")).toStrictEqual(NOTHING);
  expect(p("n > 1", false)).toStrictEqual(NOTHING);
  expect(p(`n > 1 and ${SA}`)).toStrictEqual(new Holds(A, true));
  // `n > 1` may be what made the `or` true, so nothing is proven.
  expect(p(`n > 1 or ${SA}`)).toStrictEqual(NOTHING);
  expect(p(`(${SA}) :- false`)).toStrictEqual(NOTHING);
  expect(p("true")).toStrictEqual(NOTHING);
  // A false `and` (a true `or`) may be decided by an operand that proves nothing.
  expect(p(`${SA} and n > 1`, false)).toStrictEqual(NOTHING);
  expect(p(`${SA} and n > 1 and ${SB}`, false)).toStrictEqual(NOTHING);
  expect(p(`not ${SA} or n > 1`)).toStrictEqual(NOTHING);
  expect(p(`${SA} and n > 1`)).toStrictEqual(new Holds(A, true));
  expect(p(`not ${SA} or n > 1`, false)).toStrictEqual(new Holds(A, true));
});

// --------------------------------------------------------------------------- //
// guardedRefs
// --------------------------------------------------------------------------- //

test("a read after an and check is guarded by it", () => {
  expect(guardOf("isinstance(v, Buy) and v.qty > 0", "v.qty")).toStrictEqual(new Holds(A, true));
});

test("a read after an or check is guarded by its falsity", () => {
  expect(guardOf("not isinstance(v, Buy) or v.qty > 0", "v.qty")).toStrictEqual(new Holds(A, true));
  expect(guardOf('v.action != "sell" or v.qty > 0', "v.qty")).toStrictEqual(new Holds(B, true));
  expect(guardOf("isinstance(v, Buy) or v.qty > 0", "v.qty")).toStrictEqual(new Holds(A, false));
});

test("a read after an and with a non check is unguarded", () => {
  expect(guardOf("(isinstance(v, Buy) and n > 1) or v.qty > 0", "v.qty")).toStrictEqual(NOTHING);
  expect(guardOf("(not isinstance(v, Buy) or n > 1) and v.qty > 0", "v.qty")).toStrictEqual(NOTHING);
});

test("a check after the read does not guard it", () => {
  expect(guardOf("v.qty > 0 and isinstance(v, Buy)", "v.qty")).toStrictEqual(NOTHING);
});

test("a non check before the read proves nothing", () => {
  expect(guardOf("n > 5 or isinstance(r, Done) and r.answer == 'ok'", "r.answer")).toStrictEqual(
    new Holds(new IsinstanceCheck("r", "Done"), true),
  );
});

test("not and escapes pass the context down", () => {
  expect(guardOf("not (isinstance(r, Done) and r.answer == 'ok')", "r.answer")).toStrictEqual(
    new Holds(new IsinstanceCheck("r", "Done"), true),
  );
  expect(guardOf("(isinstance(v, Buy) and v.qty > 0) :- false", "v.qty")).toStrictEqual(new Holds(A, true));
});

test("every read inside a guarded operand is guarded", () => {
  expect(guards("isinstance(v, Buy) and (v.qty > 0 or v.qty < -1)")).toStrictEqual([
    ["v", NOTHING],
    ["v.qty", new Holds(A, true)],
    ["v.qty", new Holds(A, true)],
  ]);
});

test("a builtin argument read is guarded", () => {
  expect(guardOf('isinstance(v, Buy) and join(v.tags, ",") == "x"', "v.tags")).toStrictEqual(new Holds(A, true));
});

test("a conditional's branches are read under what its test proves", () => {
  const text = "v.qty if isinstance(v, Buy) else w.qty if w != null else 0";
  expect(guards(text)).toStrictEqual([
    ["v.qty", new Holds(A, true)],
    ["v", NOTHING],
    ["w.qty", new AllOf([new Holds(A, false), new Holds(new NotNullCheck("w"), true)])],
    ["w", new Holds(A, false)],
  ]);
});

test("a conditional proves what the branch its test picked proves", () => {
  expect(p(`${SB} if ${SA} else ${SC}`)).toStrictEqual(
    new AnyOf([
      new AllOf([new Holds(A, true), new Holds(B, true)]),
      new AllOf([new Holds(A, false), new Holds(C, true)]),
    ]),
  );
  // A branch that proves nothing still tells which way the test went.
  expect(p(`${SB} if ${SA} else n > 1`)).toStrictEqual(
    new AnyOf([new AllOf([new Holds(A, true), new Holds(B, true)]), new Holds(A, false)]),
  );
});

test("guards of a nested or inside an and", () => {
  const pairs = guards(`${SA} and (v.action == 'sell' or v.qty > 0)`);
  expect(pairs.at(-1)).toStrictEqual(["v.qty", new AllOf([new Holds(A, true), new Holds(B, false)])]);
});

describe("guardedRefs walks the reads in exprRefs order", () => {
  const cases = [
    "isinstance(v, Buy) and v.qty > 0",
    "not isinstance(v, Buy) or v.qty > 0",
    "n > 5 or isinstance(r, Done) and r.answer == 'ok'",
    "(isinstance(v, Buy) and v.qty > 0) :- false",
    "(a.output | b.output) == 2 and c.x > 1",
    "a.output :? 'need a' == 'x' or b",
    "v.action in [w.x, 'buy'] and not v.qty in [1, u.y]",
    'isinstance(v, Buy) and join(v.tags, ",") == upper(w.name)',
    "${isinstance(v, Buy) and lower(w.name) == v.action}",
    "not (isinstance(r, Done) and r.answer == 'ok') or f(k).z > m",
    "v.qty if isinstance(v, Buy) else w.n > (v.price if v.action == 'sell' else 0)",
  ];
  for (const text of cases) {
    test(text, () => {
      const tree = parseCondition(text);
      const paths = guardedRefs(tree).map(([path]) => path);
      expect(paths).toEqual(exprRefs(tree));
      expect(paths).toEqual(conditionReads(tree).map((read) => read.path));
    });
  }
});

// --------------------------------------------------------------------------- //
// guardChecks / conjuncts
// --------------------------------------------------------------------------- //

test("guardChecks and conjuncts", () => {
  const guard = p(`${SA} and (${SB} or ${SC}) and ${SA}`);
  expect(guardChecks(guard)).toStrictEqual([A, B, C]);
  expect(guardChecks(NOTHING)).toStrictEqual([]);
  expect(guard).toBeInstanceOf(AllOf);
  expect(conjuncts(guard)).toBe((guard as AllOf).parts);
  expect(conjuncts(new Holds(A, true))).toStrictEqual([new Holds(A, true)]);
  expect(conjuncts(NOTHING)).toStrictEqual([]);
});

// --------------------------------------------------------------------------- //
// literalsInContext
// --------------------------------------------------------------------------- //

/** Each literal of `text` with its context, as `[context, check, value, negated]`. */
function literals(text: string): [Guard, Check, boolean, boolean][] {
  return literalsInContext(parseCondition(text)).map((literal) => [
    literal.context,
    literal.check,
    literal.value,
    literal.negated,
  ]);
}

test("a literal carries the context before it", () => {
  expect(literals("isinstance(v, Buy) and v.action == 'hold'")).toStrictEqual([
    [NOTHING, A, true, false],
    [new Holds(A, true), new TagCheck("v", "action", "hold"), true, false],
  ]);
});

test("a negated check is one literal", () => {
  expect(literals("not isinstance(v, Buy) or x")).toStrictEqual([[NOTHING, A, false, true]]);
  expect(literals("not not isinstance(v, T)")).toStrictEqual([[NOTHING, new IsinstanceCheck("v", "T"), true, false]]);
  expect(literals("v.action != 'buy'")).toStrictEqual([[NOTHING, new TagCheck("v", "action", "buy"), false, false]]);
});

test("literals are only in boolean positions", () => {
  expect(literals("(isinstance(v, Buy)) == true")).toStrictEqual([]);
  expect(literals("upper(v.action == 'buy')")).toStrictEqual([]);
  expect(literals("[isinstance(v, Buy)] == x")).toStrictEqual([]);
  expect(literals("(v.action == 'buy') | false")).toStrictEqual([]);
  expect(literals("(isinstance(v, Buy) and n > 1) :- false")).toStrictEqual([[NOTHING, A, true, false]]);
  expect(literals("isinstance(v, Buy) :? 'need v'")).toStrictEqual([[NOTHING, A, true, false]]);
});

test("a conditional's test is a boolean position, and so are its branches in one", () => {
  expect(literals(`${SB} if ${SA} else ${SC}`)).toStrictEqual([
    [new Holds(A, true), B, true, false],
    [NOTHING, A, true, false],
    [new Holds(A, false), C, true, false],
  ]);
  expect(literals(`len(v.tags if ${SA} else []) > 0`)).toStrictEqual([[NOTHING, A, true, false]]);
});

test("literals under not of a group see their own context", () => {
  expect(literals(`not (${SA} and ${SB})`)).toStrictEqual([
    [NOTHING, A, true, false],
    [new Holds(A, true), B, true, false],
  ]);
});

test("literalText is canonical", () => {
  const texts = literalsInContext(
    parseCondition(
      "not not isinstance( v , Buy ) and not v.action == 'hold' and v.action != 'a\"b' and 'sell' == v.action",
    ),
  ).map(literalText);
  expect(texts).toEqual(["isinstance(v, Buy)", 'not v.action == "hold"', "v.action != 'a\"b'", 'v.action == "sell"']);
});

test("masked checks prove nothing and keep the reads", () => {
  const tree = parseCondition("isinstance(v, Buy) and not isinstance(w, Done) and v.qty > w.n");
  const literal = literalsInContext(tree)[0]!;
  expect(literal.node).not.toBeNull();
  const copy = masked(tree, [literal.node!]);
  expect(guardedRefs(copy)).toStrictEqual([
    ["v", NOTHING],
    ["w", NOTHING],
    ["v.qty", new Holds(C, false)],
    ["w.n", new Holds(C, false)],
  ]);
  expect(literalsInContext(copy).map((lit) => lit.check)).toStrictEqual([C]);
  // The original is unchanged.
  expect(guardedRefs(tree)[2]).toStrictEqual(["v.qty", allOf(new Holds(A, true), new Holds(C, false))]);
});

test("NEVER absorbs an allOf and drops out of an anyOf", () => {
  const [a, b] = [new Holds(A, true), new Holds(B, false)];
  expect(allOf(a, NEVER, b)).toStrictEqual(NEVER);
  expect(anyOf(a, NEVER)).toStrictEqual(a);
  expect(anyOf(NEVER, NEVER)).toStrictEqual(NEVER);
  expect(anyOf(NEVER, NOTHING)).toStrictEqual(NOTHING);
});

test("falsified checks prove NEVER for the value they never have", () => {
  const tree = parseCondition("isinstance(v, Buy) and v.action != 'sell' and v.qty > 0");
  const nodes = literalsInContext(tree).map((literal) => literal.node!);
  expect(nodes).toHaveLength(2);
  const [first, second] = nodes;
  let reads = guardedRefs(falsified(tree, [[first!, true]]));
  expect(reads.at(-1)).toStrictEqual(["v.qty", NEVER]);
  // The other value proves nothing, and the reads keep their order.
  reads = guardedRefs(falsified(tree, [[first!, false]]));
  expect(reads.map(([read]) => read)).toEqual(guardedRefs(tree).map(([read]) => read));
  expect(reads.at(-1)).toStrictEqual(["v.qty", new Holds(B, false)]);
  // A `!=` sub-tree that is never false: the read after it still runs.
  expect(guardedRefs(falsified(tree, [[second!, false]])).at(-1)).toStrictEqual(["v.qty", new Holds(A, true)]);
  expect(guardedRefs(falsified(tree, [[second!, true]])).at(-1)).toStrictEqual(["v.qty", NEVER]);
});

test("sameGuard, sameCheck and unknownCheck", () => {
  const [a, b] = [new Holds(A, true), new Holds(B, false)];
  expect(sameGuard(NOTHING, NEVER)).toBe(false);
  expect(sameGuard(new AllOf([a, b]), new AllOf([b, a]))).toBe(false);
  expect(sameGuard(new AllOf([a, b]), new AllOf([a, b]))).toBe(true);
  expect(sameCheck(new TagCheck("v", "f", "x"), new TagCheck("v", "f", "y"))).toBe(false);
  class OtherCheck {}
  const error = unknownCheck(new OtherCheck());
  expect(error).toBeInstanceOf(TypeError);
  expect(error.message).toBe("unhandled check kind OtherCheck");
});

test("a comprehension's reference reads match conditionReads, in order, around its variable reads", () => {
  const when = 'len(join([m.text for t in input.history for m in t], "")) > input.budget';
  const paths = guards(when).map(([path]) => path);
  const references = paths.filter((path) => !path.split(".")[0]!.includes("@"));
  expect(references).toEqual(conditionReads(parseCondition(when)).map((r) => r.path));
  expect(paths.map((path) => path.replace(/@\d+/, "@"))).toEqual(["m@.text", "input.history", "t@", "input.budget"]);
});

test("a comprehension proves nothing", () => {
  expect(p("[x for x in xs if x.type == 'a']")).toStrictEqual(NOTHING);
  expect(p("[x for x in xs if x.type == 'a']", false)).toStrictEqual(NOTHING);
});

test("a ?. read is a guarded read and a check subject", () => {
  expect(guardedRefs(parseCondition("x?.k != null and x?.k.m")).map(([path]) => path)).toEqual(["x.k", "x.k.m"]);
  expect(checkIn(parseCondition("x?.k != null"))).toStrictEqual([new NotNullCheck("x.k"), true]);
});

test("guardedReads gives each read occurrence its node, path and guard", () => {
  const reads = guardedReads(parseCondition("x != null and x?.k.m"));
  expect(reads.map((read) => read.path)).toEqual(["x", "x.k.m"]);
  expect(referenceText(reads[1]!.node)).toBe("x?.k.m");
  expect(reads[1]!.guard).toEqual(new Holds(new NotNullCheck("x"), true));
});

// ---------- comprehension variables ----------

test("an if clause guards the elem and later clauses, by binding occurrence", () => {
  const reads = guardedReads(parseCondition('[p.text for m in ms for p in m.parts if p.type == "text"]'));
  const text = reads.find((r) => r.path.startsWith("p@") && r.path.endsWith(".text"))!;
  expect(text.guard).toEqual(new Holds(new TagCheck(text.path.split(".")[0]!, "type", "text"), true));
  // `m.parts` is read in the clause after `m`'s for, before the if.
  expect(reads.find((r) => r.path.startsWith("m@"))!.guard).toEqual(NOTHING);
});

test("an if null check narrows its variable, and another comprehension's m is another key", () => {
  const reads = guardedReads(parseCondition("[[m.k for m in xs if m != null], [m.k for m in ys]]"));
  const [first, second] = reads.filter((r) => r.path.endsWith(".k"));
  expect(first!.guard).toEqual(new Holds(new NotNullCheck(first!.path.split(".")[0]!), true));
  expect(second!.guard).toEqual(NOTHING);
  expect(first!.path).not.toBe(second!.path);
});

test("an if clause narrows an outer reference read after it", () => {
  const reads = guardedReads(parseCondition("[input.maybe.a for m in ms if input.maybe != null]"));
  const outer = reads.find((r) => r.path === "input.maybe.a")!;
  expect(outer.guard).toEqual(new Holds(new NotNullCheck("input.maybe"), true));
});

test("clause guards build on the context around the comprehension", () => {
  const reads = guardedReads(parseCondition("x != null and len([m.k for m in xs if m != null]) > 0"));
  const read = reads.find((r) => r.path.endsWith(".k"))!;
  expect(read.guard).toEqual(
    allOf(new Holds(new NotNullCheck("x"), true), new Holds(new NotNullCheck(read.path.split(".")[0]!), true)),
  );
});

test("masked and falsified copies keep comprehension reads", () => {
  const tree = parseCondition("isinstance(x, Text) and len([m.k for m in xs]) > 0");
  const literal = literalsInContext(tree)[0]!.node!;
  const paths = guardedRefs(tree).map(([path]) => path);
  expect(paths.some((path) => path.startsWith("m@"))).toBe(true);
  expect(guardedRefs(masked(tree, [literal])).map(([path]) => path)).toEqual(paths);
  expect(guardedRefs(falsified(tree, [[literal, true]])).map(([path]) => path)).toEqual(paths);
});

test("a comprehension's if clause is a boolean position, under the clauses before it", () => {
  const literals = literalsInContext(parseCondition("len([m.k for m in xs if m != null if isinstance(m.k, T)]) > 0"));
  expect(literals.map(literalText).map((text) => text.replace(/@\d+/g, "@"))).toEqual([
    "m@ != null",
    "isinstance(m@.k, T)",
  ]);
  expect(literals[1]!.context).toEqual(new Holds(literals[0]!.check, true));
});

test("guards of deeply alternating and/or nesting take time linear in the depth", () => {
  // Each level proves its inner chain for both values; without sharing that work doubles per level.
  let text = "isinstance(v, T0)";
  for (let i = 1; i <= 40; i++) text = `(${text} ${i % 2 === 1 ? "and" : "or"} isinstance(v, T${i}))`;
  const tree = parseCondition(text);
  const start = performance.now();
  expect(proves(tree, true)).not.toStrictEqual(NOTHING);
  expect(proves(tree, false)).not.toStrictEqual(NOTHING);
  // The guard's tree grows about 1.6x per level; `guardChecks` visits each shared sub-guard once.
  expect(guardChecks(proves(tree, true))).toHaveLength(41);
  expect(guardedRefs(tree)).toHaveLength(41);
  expect(literalsInContext(tree)).toHaveLength(41);
  expect(performance.now() - start).toBeLessThan(5000);
});
