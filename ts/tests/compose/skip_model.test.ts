/**
 * The skip model: which nodes of a flow level CASE routing may skip, and together. A port of
 * `tests/compose/test_skip_model.py`; every row is ported.
 *
 * Changed: node and CASE id sets are `Set`s, producer and input tables `Map`s.
 */

import { expect, test } from "bun:test";

import { Edge } from "../../src/compile/model.ts";
import { CaseDescriptor } from "../../src/compose/nodes/case.ts";
import { caseIsExhaustive, exhaustiveCases } from "../../src/compose/validate/case_exhaustive.ts";
import { SkipModel } from "../../src/compose/validate/skips.ts";
import { ELSE_HANDLE } from "../../src/nodes/base.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";

let ids = 0;

/** A CASE control edge `case -[handle]-> target`. */
function ctl(caseId: string, handle: string, target: string): Edge {
  return new Edge(`e${ids++}`, caseId, target, { sourceHandle: handle });
}

/** A data edge feeding input group `group` of `dst`. */
function data(src: string, dst: string, group = "x", options: { optional?: boolean } = {}): Edge {
  return new Edge(`e${ids++}`, src, dst, { inputGroup: group, optional: options.optional ?? false });
}

/** An ordering edge: `depends_on` when required, `runs_after` when optional. */
function order(src: string, dst: string, options: { optional?: boolean } = {}): Edge {
  return new Edge(`e${ids++}`, src, dst, { ordering: true, optional: options.optional ?? false });
}

/** A skip model over the space-separated node ids `nodes`. */
function model(
  nodes: string,
  edges: Edge[],
  exhaustive: Iterable<string> = [],
  options: { limit?: number | null } = {},
): SkipModel {
  return new SkipModel(nodes.split(" "), edges, new Set(exhaustive), options);
}

/** `caseId` routes `then -> a` and `else -> b`. */
function route(caseId = "c"): Edge[] {
  return [ctl(caseId, "then", "a"), ctl(caseId, "else", "b")];
}

test("a case target may skip and a root may not", () => {
  // A CASE target may be skipped; the CASE itself, a root, may not.
  const m = model("c a b", route(), ["c"]);
  expect(m.mayAllSkip(["a"])).toBe(true);
  expect(m.mayAllSkip(["c"])).toBe(false);
  expect(m.mayAllSkip([])).toBe(false);
});

test("a producer outside the level never skips", () => {
  // An id outside the level, such as START, never skips.
  const m = model("c a b", route());
  expect(m.mayAllSkip(["a", "__start__"])).toBe(false);
  expect(m.maySkipWhileRunning(["a", "__start__"], null)).toBe(false);
});

test("joint targets skip only when the case is not exhaustive", () => {
  // Both targets of a CASE skip together only when it may match nothing.
  expect(model("c a b", route(), ["c"]).mayAllSkip(["a", "b"])).toBe(false);
  expect(model("c a b", route()).mayAllSkip(["a", "b"])).toBe(true);
});

/** `outer` routes `inner`/`rejected`; `inner` routes `order`/`hold`. */
function nested(): Edge[] {
  return [
    ctl("outer", "then", "inner"),
    ctl("outer", "else", "rejected"),
    ctl("inner", "then", "order"),
    ctl("inner", "else", "hold"),
  ];
}

test("nested cases", () => {
  // Leaves of nested exhaustive CASEs never skip together; one leaf may.
  const both = model("outer inner rejected order hold", nested(), ["outer", "inner"]);
  expect(both.mayAllSkip(["order", "hold", "rejected"])).toBe(false);
  expect(both.mayAllSkip(["order"])).toBe(true);
  const innerOnly = model("outer inner rejected order hold", nested(), ["inner"]);
  expect(innerOnly.mayAllSkip(["order", "hold", "rejected"])).toBe(true);
});

test("a required data group", () => {
  // A required group skips its node only when all producers skip; an optional one never.
  const edges = [
    ctl("c", "then", "pro"),
    ctl("c", "else", "caut"),
    data("pro", "both"),
    data("caut", "both"),
    data("pro", "one"),
    data("pro", "opt", "x", { optional: true }),
  ];
  const m = model("c pro caut both one opt", edges, ["c"]);
  expect(m.mayAllSkip(["both"])).toBe(false);
  expect(m.mayAllSkip(["one"])).toBe(true);
  expect(m.mayAllSkip(["opt"])).toBe(false);
});

test("ordering edges", () => {
  // `depends_on` propagates a skip; `runs_after` does not.
  const edges = [...route(), order("a", "dep"), order("a", "after", { optional: true })];
  const m = model("c a b dep after", edges, ["c"]);
  expect(m.mayAllSkip(["dep"])).toBe(true);
  expect(m.mayAllSkip(["after"])).toBe(false);
});

test("a case reading a maybe-skipped node may skip with all its targets", () => {
  // A CASE that may itself be skipped takes all its targets with it.
  const edges = [...route(), data("a", "c2"), ctl("c2", "then", "x"), ctl("c2", "else", "y")];
  const m = model("c a b c2 x y", edges, ["c", "c2"]);
  expect(m.mayAllSkip(["x", "y"])).toBe(true);
});

test("a node targeted by two cases may skip", () => {
  // A node two CASEs target dies when both skip their edge to it.
  expect(model("c1 c2 n", [ctl("c1", "then", "n"), ctl("c2", "then", "n")]).mayAllSkip(["n"])).toBe(true);
});

test("a node targeted by an exhaustive case may still skip", () => {
  // `c2` always takes a branch, but `n` dies when `c2` picks `m` and `c1` matches nothing.
  const edges = [ctl("c1", "then", "n"), ctl("c2", "then", "n"), ctl("c2", "else", "m")];
  const m = model("c1 c2 n m", edges, ["c2"]);
  expect(m.mayAllSkip(["n"])).toBe(true);
  expect(m.mayAllSkip(["n", "m"])).toBe(false);
});

test("a node targeted by a sole branch of an exhaustive case never skips", () => {
  // An exhaustive CASE with one branch always keeps that edge taken.
  const m = model("c1 c2 n", [ctl("c1", "then", "n"), ctl("c2", "then", "n")], ["c1"]);
  expect(m.mayAllSkip(["n"])).toBe(false);
});

test("depends_on a case runs on every branch", () => {
  // An ordering edge from a CASE is not a branch: it skips only when the CASE is skipped.
  const edges = [ctl("gate", "then", "a"), ctl("gate", ELSE_HANDLE, "b"), order("gate", "after")];
  const m = model("gate a b after", edges, ["gate"]);
  expect(m.mayAllSkip(["after"])).toBe(false);
  expect(m.maySkipWhileRunning(["a"], "after")).toBe(true);
});

test("depends_on a skipped case skips", () => {
  // An ordering edge from a CASE that an outer CASE skips co-skips its node.
  const edges = [ctl("outer", "then", "gate"), ctl("gate", "then", "a"), order("gate", "after")];
  const m = model("outer gate a after", edges);
  expect(m.mayAllSkip(["after"])).toBe(true);
  expect(m.maySkipWhileRunning(["gate"], "after")).toBe(false);
});

test("runs_after a case never skips", () => {
  // An optional ordering edge from a CASE never skips its node.
  const edges = [ctl("gate", "then", "a"), ctl("gate", ELSE_HANDLE, "b")];
  const m = model("gate a b after", [...edges, order("gate", "after", { optional: true })], ["gate"]);
  expect(m.mayAllSkip(["after"])).toBe(false);
});

test("required data from a case runs on every branch", () => {
  // A required data edge from a CASE is taken on every route the CASE picks.
  const edges = [ctl("gate", "then", "a"), data("gate", "reader")];
  const m = model("gate a reader", edges);
  expect(m.mayAllSkip(["reader"])).toBe(false);
  expect(m.maySkipWhileRunning(["a"], "reader")).toBe(true);
});

test("may skip while running", () => {
  // A consumer that co-skips on the producer never runs while it is skipped.
  const edges = [
    ctl("c", "then", "order"),
    ctl("c", "else", "hold"),
    data("order", "req"),
    data("order", "mid"),
    data("mid", "chain"),
    data("order", "opt", "x", { optional: true }),
  ];
  const m = model("c order hold req mid chain opt", edges, ["c"]);
  expect(m.maySkipWhileRunning(["order"], "req")).toBe(false);
  expect(m.maySkipWhileRunning(["order"], "chain")).toBe(false);
  expect(m.maySkipWhileRunning(["order"], "opt")).toBe(true);
  expect(m.maySkipWhileRunning(["order"], null)).toBe(true);
  expect(m.maySkipWhileRunning(["order", "hold"], null)).toBe(false);
});

test("a cycle reports nothing", () => {
  // A cyclic level reports nothing skippable.
  const edges = [ctl("c", "then", "a"), data("a", "b"), data("b", "a")];
  const m = model("c a b", edges);
  expect(m.mayAllSkip(["a"])).toBe(false);
  expect(m.maySkipWhileRunning(["a"], null)).toBe(false);
});

/**
 * `c` (exhaustive) routes `a`/`b`; `x` is `g`'s only route; ten more CASEs feed both.
 *
 * Each `t{i}` shares a required group of `a` and `x` with the root `r`, so the ten CASEs are
 * branched on but can never skip `a` or `x`.
 */
function manyCases(options: { limit?: number | null } = {}): SkipModel {
  const edges = [...route(), ctl("g", "then", "x")];
  const nodes = ["r", "g", "c", "a", "b", "x"];
  for (let i = 0; i < 10; i++) {
    edges.push(ctl(`k${i}`, "then", `t${i}`));
    for (const n of ["a", "x"]) for (const p of [`t${i}`, "r"]) edges.push(data(p, n, `in${i}`));
    nodes.push(`k${i}`, `t${i}`);
  }
  return new SkipModel(nodes, edges, new Set(["c"]), options);
}

test("cases behind edges that cannot skip are not branched on", () => {
  // CASEs reached only by an optional group or `runs_after` add no assignments.
  const edges = [...route(), ctl("k1", "then", "t1"), ctl("k2", "then", "t2")];
  edges.push(data("t1", "a", "x", { optional: true }), order("t2", "b", { optional: true }));
  const m = model("c a b k1 t1 k2 t2", edges, ["c"], { limit: 2 });
  expect(m.mayAllSkip(["a", "b"])).toBe(false);
});

test("past the world limit the queries fall back safely", () => {
  // Past the bound, `mayAllSkip` says true and `maySkipWhileRunning` false.
  const exact = manyCases();
  expect(exact.mayAllSkip(["a", "b"])).toBe(false);
  expect(exact.maySkipWhileRunning(["x"], null)).toBe(true);
  const bounded = manyCases({ limit: 1 });
  expect(bounded.mayAllSkip(["a", "b"])).toBe(true);
  expect(bounded.maySkipWhileRunning(["x"], null)).toBe(false);
});

/** An `on: ${cls.output}` CASE listing tags `a` and `b`. */
function onCase(options: { else_?: string } = {}): CaseDescriptor {
  return new CaseDescriptor({
    id: "route",
    on: "${cls.output}",
    cases: [
      { when: "a", then: "na" },
      { when: "b", then: "nb" },
    ],
    ...options,
  });
}

/** A producer map with `cls` of type `typ`. */
function cls(typ: Type): Map<string, ValueContract> {
  return new Map([["cls", new ValueContract(typ)]]);
}

const AB = new Type({ kind: ValueKind.STRING, tags: new Set(["a", "b"]) });

test("a case is exhaustive", () => {
  // `else:`, or `on:` over a non-nullable Literal with every tag listed.
  const searched = new CaseDescriptor({ id: "s", cases: [{ when: "cls.output == 'a'", then: "na" }] });
  const string = Type.scalar(ValueKind.STRING);
  const nullable = new Type({ kind: ValueKind.STRING, tags: new Set(["a", "b"]), nullable: true });
  expect(caseIsExhaustive(onCase({ else_: "other" }), cls(string))).toBe(true);
  expect(caseIsExhaustive(onCase(), cls(AB))).toBe(true);
  expect(caseIsExhaustive(onCase(), cls(nullable))).toBe(false);
  expect(caseIsExhaustive(onCase(), cls(string))).toBe(false);
  expect(caseIsExhaustive(searched, cls(AB))).toBe(false);
  const abc = new Type({ kind: ValueKind.STRING, tags: new Set(["a", "b", "c"]) });
  expect(caseIsExhaustive(onCase(), cls(abc))).toBe(false);
});

test("exhaustive cases", () => {
  // Only the exhaustive CASE descriptors are returned.
  const searched = new CaseDescriptor({ id: "s", cases: [{ when: "cls.output == 'a'", then: "na" }] });
  const descriptors = new Map<string, CaseDescriptor>([
    ["route", onCase()],
    ["s", searched],
  ]);
  expect(exhaustiveCases(descriptors, cls(AB))).toEqual(new Set(["route"]));
});

test("nodes and hasCases", () => {
  // The model exposes its node ids and whether any CASE routes within the level.
  const m = model("c a b", route());
  expect(m.nodes).toEqual(new Set(["c", "a", "b"]));
  expect(m.hasCases).toBe(true);
  expect(model("a b", [data("a", "b")]).hasCases).toBe(false);
});

/** An `on:` CASE over a flow input, listing tags `a` and `b`. */
function onInput(on = "${input.k}"): CaseDescriptor {
  return new CaseDescriptor({
    id: "route",
    on,
    cases: [
      { when: "a", then: "na" },
      { when: "b", then: "nb" },
    ],
  });
}

test("a case over a Literal input is exhaustive", () => {
  // `on: ${input.k}` over a fully listed non-nullable Literal input always takes a branch.
  const nullable = new Type({ kind: ValueKind.STRING, tags: new Set(["a", "b"]), nullable: true });
  const abc = new Type({ kind: ValueKind.STRING, tags: new Set(["a", "b", "c"]) });
  const record = new Type({ kind: ValueKind.OBJECT, fields: new Map([["kind", AB]]), required: new Set(["kind"]) });
  const input = (name: string, typ: Type) => new Map([[name, new ValueContract(typ)]]);
  expect(caseIsExhaustive(onInput(), new Map(), input("k", AB))).toBe(true);
  expect(caseIsExhaustive(onInput("${input.r.kind}"), new Map(), input("r", record))).toBe(true);
  expect(caseIsExhaustive(onInput(), new Map(), input("k", nullable))).toBe(false);
  expect(caseIsExhaustive(onInput(), new Map(), input("k", abc))).toBe(false);
  expect(caseIsExhaustive(onInput(), new Map())).toBe(false);
  expect(exhaustiveCases(new Map([["route", onInput()]]), new Map(), input("k", AB))).toEqual(new Set(["route"]));
});
