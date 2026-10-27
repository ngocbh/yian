/** `hostControlEdges`: a lifted node takes its host's CASE routes, ordering, and data gates. */

import { expect, test } from "bun:test";

import { Edge } from "../../src/compile/model.ts";
import { hostControlEdges } from "../../src/compose/edges/hosts.ts";
import type { NodeDescriptor } from "../../src/compose/nodes/base.ts";
import { CodeDescriptor } from "../../src/compose/nodes/code.ts";
import { ELSE_HANDLE } from "../../src/nodes/base.ts";

/** A CODE descriptor with the given common fields. */
function code(
  nodeId: string,
  fields: { host?: string; dependsOn?: string[]; runsAfter?: string[] } = {},
): CodeDescriptor {
  return new CodeDescriptor({ id: nodeId, code: "return 1", ...fields });
}

/** Descriptors keyed by id, in the given order. */
function nodes(...descriptors: NodeDescriptor[]): Map<string, NodeDescriptor> {
  return new Map(descriptors.map((desc) => [desc.id, desc]));
}

/** A data edge from `src` into `dst`'s input `group`. */
function data(src: string, dst: string, group: string, optional = false, i = 0): Edge {
  return new Edge(`${src}->${dst}#${i}`, src, dst, { inputGroup: group, optional });
}

/** A CASE route edge from `source` to `target`. */
function route(source: string, target: string, handle: string, index = 0): Edge {
  return new Edge(`${source}->${target}#${index}`, source, target, { sourceHandle: handle });
}

/** Return the `RangeError` `run` throws. */
function rangeError(run: () => unknown): RangeError {
  try {
    run();
  } catch (error) {
    if (error instanceof RangeError) return error;
    throw error;
  }
  throw new Error("expected a RangeError");
}

test("a route into the host is copied", () => {
  const descs = nodes(code("c"), code("h"), code("L", { host: "h" }));
  expect(hostControlEdges(descs, [route("c", "h", "h")])).toStrictEqual([
    new Edge("c->L#0", "c", "L", { sourceHandle: "h" }),
  ]);
});

test("an else route keeps its handle", () => {
  const descs = nodes(code("c"), code("h"), code("L", { host: "h" }));
  const edges = hostControlEdges(descs, [route("c", "h", ELSE_HANDLE)]);
  expect(edges.length).toBe(1);
  expect(edges[0]!.sourceHandle).toBe(ELSE_HANDLE);
});

test("each route into the host gets a copy", () => {
  const descs = nodes(code("c"), code("d"), code("h"), code("L", { host: "h" }));
  const edges = [route("c", "h", "h", 0), route("c", "h", ELSE_HANDLE, 1), route("d", "h", "h", 0)];
  expect(hostControlEdges(descs, edges)).toStrictEqual([
    new Edge("c->L#0", "c", "L", { sourceHandle: "h" }),
    new Edge("c->L#1", "c", "L", { sourceHandle: ELSE_HANDLE }),
    new Edge("d->L#0", "d", "L", { sourceHandle: "h" }),
  ]);
});

test("depends_on and runs_after become ordering edges", () => {
  const descs = nodes(
    code("p"),
    code("q"),
    code("h", { dependsOn: ["p"], runsAfter: ["q"] }),
    code("L", { host: "h" }),
  );
  expect(hostControlEdges(descs, [])).toStrictEqual([
    new Edge("p~>L#0", "p", "L", { ordering: true, optional: false }),
    new Edge("q~>L#0", "q", "L", { ordering: true, optional: true }),
  ]);
});

test("routes come before depends_on, then runs_after", () => {
  const descs = nodes(
    code("c"),
    code("p"),
    code("q"),
    code("h", { dependsOn: ["p"], runsAfter: ["q"] }),
    code("L", { host: "h" }),
  );
  const edges = hostControlEdges(descs, [route("c", "h", "h")]);
  expect(edges.map((edge) => edge.id)).toEqual(["c->L#0", "p~>L#0", "q~>L#0"]);
});

test("unknown or self sources are skipped", () => {
  // The host reports an unknown source itself; a host ordering its own lifted node would make a
  // self-loop.
  const descs = nodes(code("h", { dependsOn: ["missing"], runsAfter: ["L"] }), code("L", { host: "h" }));
  expect(hostControlEdges(descs, [])).toStrictEqual([]);
});

test("host chains are followed", () => {
  const descs = nodes(
    code("c"),
    code("p"),
    code("h", { dependsOn: ["p"] }),
    code("L1", { host: "h" }),
    code("L2", { host: "L1" }),
  );
  const edges = hostControlEdges(descs, [route("c", "h", "h")]);
  expect(edges.map((edge) => [edge.from, edge.to, edge.sourceHandle, edge.ordering])).toEqual([
    ["c", "L1", "h", false],
    ["p", "L1", null, true],
    ["c", "L2", "h", false],
    ["p", "L2", null, true],
  ]);
});

test("ids skip ones already used", () => {
  const descs = nodes(code("c"), code("h"), code("L", { host: "h" }));
  const taken = new Edge("c->L#0", "c", "L", { inputGroup: "x" });
  const edges = hostControlEdges(descs, [taken, route("c", "h", "h")]);
  expect(edges.map((edge) => edge.id)).toEqual(["c->L#1"]);
});

test("ordering edges are not copied as routes, and data edges become gates", () => {
  const descs = nodes(code("a"), code("h"), code("L", { host: "h" }));
  const edges = [
    new Edge("a->h#0", "a", "h", { inputGroup: "x" }),
    new Edge("a~>h#0", "a", "h", { ordering: true }),
  ];
  expect(hostControlEdges(descs, edges)).toStrictEqual([new Edge("a->L#0", "a", "L", { inputGroup: "h.x" })]);
});

test("no host or an ungated host gives no edges", () => {
  const descs = nodes(code("c"), code("h"), code("free"), code("L", { host: "free" }));
  expect(hostControlEdges(descs, [route("c", "h", "h")])).toStrictEqual([]);
});

test.each<[string, CodeDescriptor, Edge[]]>([
  ["its own route", code("L", { host: "h" }), [route("c", "L", "L")]],
  ["its own depends_on", code("L", { host: "h", dependsOn: ["c"] }), []],
  ["its own runs_after", code("L", { host: "h", runsAfter: ["c"] }), []],
])("a lifted node with its own control is a compiler bug: %s", (_label, lifted, edges) => {
  const descs = nodes(code("c"), code("h"), lifted);
  expect(rangeError(() => hostControlEdges(descs, edges)).message).toBe(
    "lifted node 'L' has its own route or ordering edges; it may only inherit its host's",
  );
});

test("a missing host or a host cycle is a compiler bug", () => {
  expect(rangeError(() => hostControlEdges(nodes(code("L", { host: "nowhere" })), [])).message).toBe(
    "lifted node 'L' names host 'nowhere', which is not a node",
  );
  expect(
    rangeError(() => hostControlEdges(nodes(code("A", { host: "B" }), code("B", { host: "A" })), [])).message,
  ).toBe("lifted-node hosts form a cycle: A -> B -> A");
  // A host's data producer whose own hosts cycle is met before that producer's own turn.
  const descs = nodes(code("h"), code("L", { host: "h" }), code("A", { host: "B" }), code("B", { host: "A" }));
  expect(rangeError(() => hostControlEdges(descs, [data("A", "h", "x")])).message).toBe(
    "lifted-node hosts form a cycle: A -> B -> A",
  );
});

test("a host data input becomes a gate", () => {
  const descs = nodes(code("m"), code("h"), code("L", { host: "h" }));
  const edges = [data("L", "h", "draft"), data("m", "h", "notes")];
  expect(hostControlEdges(descs, edges)).toStrictEqual([new Edge("m->L#0", "m", "L", { inputGroup: "h.notes" })]);
});

test("a defaulted host input gives an optional gate", () => {
  const descs = nodes(code("m"), code("h"), code("L", { host: "h" }));
  const edges = [data("L", "h", "draft"), data("m", "h", "notes", true)];
  expect(hostControlEdges(descs, edges)).toStrictEqual([
    new Edge("m->L#0", "m", "L", { inputGroup: "h.notes", optional: true }),
  ]);
});

test("coalesce producers share one gate group", () => {
  const descs = nodes(code("a"), code("b"), code("h"), code("L", { host: "h" }));
  const edges = [data("L", "h", "draft"), data("a", "h", "n"), data("b", "h", "n")];
  expect(hostControlEdges(descs, edges)).toStrictEqual([
    new Edge("a->L#0", "a", "L", { inputGroup: "h.n" }),
    new Edge("b->L#0", "b", "L", { inputGroup: "h.n" }),
  ]);
});

test("siblings do not wait on each other", () => {
  const descs = nodes(code("m"), code("h"), code("L0", { host: "h" }), code("L1", { host: "h" }));
  const edges = [data("L0", "h", "a"), data("L1", "h", "b"), data("m", "h", "notes")];
  expect(hostControlEdges(descs, edges)).toStrictEqual([
    new Edge("m->L0#0", "m", "L0", { inputGroup: "h.notes" }),
    new Edge("m->L1#0", "m", "L1", { inputGroup: "h.notes" }),
  ]);
});

test("a sibling read by the host is looked through", () => {
  const descs = nodes(code("m"), code("m2"), code("h"), code("outer", { host: "h" }), code("inner", { host: "h" }));
  const edges = [
    data("outer", "h", "draft"),
    data("m", "h", "notes"),
    data("inner", "outer", "x"),
    data("m2", "outer", "y"),
  ];
  expect(hostControlEdges(descs, edges)).toStrictEqual([
    new Edge("m->outer#0", "m", "outer", { inputGroup: "h.notes" }),
    new Edge("m->inner#0", "m", "inner", { inputGroup: "h.notes" }),
    new Edge("m2->inner#0", "m2", "inner", { inputGroup: "outer.y" }),
  ]);
});

test("a case target's inner call waits on the target's other inputs", () => {
  const descs = nodes(code("d"), code("outer"), code("inner", { host: "outer" }));
  const edges = [data("inner", "outer", "x"), data("d", "outer", "y")];
  expect(hostControlEdges(descs, edges)).toStrictEqual([
    new Edge("d->inner#0", "d", "inner", { inputGroup: "outer.y" }),
  ]);
});

test("transitive lifts gate on every ancestor", () => {
  const descs = nodes(code("m"), code("n"), code("h"), code("L1", { host: "h" }), code("L2", { host: "L1" }));
  const edges = [data("L1", "h", "a"), data("m", "h", "b"), data("L2", "L1", "x"), data("n", "L1", "y")];
  expect(hostControlEdges(descs, edges)).toStrictEqual([
    new Edge("m->L1#0", "m", "L1", { inputGroup: "h.b" }),
    new Edge("n->L2#0", "n", "L2", { inputGroup: "L1.y" }),
    new Edge("m->L2#0", "m", "L2", { inputGroup: "h.b" }),
  ]);
});

test("the adaptive gate shape", () => {
  const descs = nodes(
    code("p"),
    code("m"),
    code("gate"),
    code("call", { host: "gate" }),
    code("comp", { host: "gate" }),
  );
  const edges = [
    data("call", "gate", "ctx"),
    data("m", "gate", "notes"),
    data("comp", "gate", "__questions"),
    data("call", "comp", "ctx"),
    data("m", "comp", "notes"),
    data("p", "call", "tag"),
  ];
  expect(hostControlEdges(descs, edges)).toStrictEqual([
    new Edge("m->call#0", "m", "call", { inputGroup: "gate.notes" }),
  ]);
});

test("a source the lifted node already requires gives no gate", () => {
  const descs = nodes(code("m"), code("h"), code("L", { host: "h" }));
  const edges = [data("m", "L", "v"), data("L", "h", "draft"), data("m", "h", "notes")];
  expect(hostControlEdges(descs, edges)).toStrictEqual([]);
});

test("a source the lifted node reads optionally still gates", () => {
  const descs = nodes(code("m"), code("h"), code("L", { host: "h" }));
  const edges = [data("m", "L", "v", true), data("L", "h", "draft"), data("m", "h", "notes")];
  expect(hostControlEdges(descs, edges)).toStrictEqual([new Edge("m->L#1", "m", "L", { inputGroup: "h.notes" })]);
});

test("a group mixing a sibling and another source only waits", () => {
  const descs = nodes(code("t"), code("h"), code("L0", { host: "h" }), code("L1", { host: "h" }));
  const edges = [data("L0", "h", "a"), data("L1", "h", "n"), data("t", "h", "n")];
  expect(hostControlEdges(descs, edges)).toStrictEqual([
    new Edge("t->L0#0", "t", "L0", { inputGroup: "h.n", optional: true }),
    new Edge("t->L1#0", "t", "L1", { inputGroup: "h.n", optional: true }),
  ]);
});

test("an unknown producer gives no gate", () => {
  const descs = nodes(code("h"), code("L", { host: "h" }));
  const edges = [data("L", "h", "draft"), data("nope", "h", "x")];
  expect(hostControlEdges(descs, edges)).toStrictEqual([]);
});

test("route copies and data gates share ids", () => {
  const descs = nodes(code("m"), code("h"), code("L", { host: "h" }));
  const edges = [route("m", "h", "h"), data("m", "h", "notes", false, 1)];
  expect(hostControlEdges(descs, edges)).toStrictEqual([
    new Edge("m->L#0", "m", "L", { sourceHandle: "h" }),
    new Edge("m->L#1", "m", "L", { inputGroup: "h.notes" }),
  ]);
});

test("routes, then ordering, then data gates", () => {
  const descs = nodes(code("c"), code("q"), code("m"), code("h", { runsAfter: ["q"] }), code("L", { host: "h" }));
  const edges = hostControlEdges(descs, [route("c", "h", "h"), data("m", "h", "notes")]);
  expect(edges.map((edge) => edge.id)).toEqual(["c->L#0", "q~>L#0", "m->L#0"]);
});
