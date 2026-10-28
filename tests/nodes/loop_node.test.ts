/** `LoopNode`, the self-respawning `while:`/`until:`/`times:` driver. */

import { expect, test } from "bun:test";

import { buildGrow } from "../../src/compile/expand.ts";
import { expansionDigest } from "../../src/compile/manifest.ts";
import { type CompiledFlow, END_ID, START_ID } from "../../src/compile/model.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { NodeFailed } from "../../src/events.ts";
import { ExpressionError, evaluateWhenRecord } from "../../src/expr/expressions.ts";
import { MemoryScope } from "../../src/memory/contracts.ts";
import { Grow, NodeKind, Output } from "../../src/nodes/base.ts";
import {
  Assertion,
  ContractOrigins,
  ExecutionPosition,
  ExecutionScope,
  Parameter,
  PartialRecordContract,
  ValueContract,
} from "../../src/nodes/contracts.ts";
import { mapCallsite, ns } from "../../src/nodes/ids.ts";
import { LoopMaxExceeded, LoopNode } from "../../src/nodes/loop/node.ts";
import type { JsonValue } from "../../src/typesys/json.ts";
import { FileRef, Type, ValueKind } from "../../src/typesys/values.ts";
import { childFlow, engine, graph, rejected } from "../helpers/run_state_fixtures.ts";

/** Assert that `fn` throws exactly `cls` with exactly `message`. */
function expectError(fn: () => unknown, cls: typeof TypeError | typeof RangeError, message: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(cls);
  expect((caught as Error).message).toBe(message);
}

/** A `while` driver `loop` over a stub body, guarded at 3 runs. */
function whileLoop(): LoopNode {
  return new LoopNode("loop", { flowId: "body", child: {}, predicateKind: "while", predicate: "${x}", maxIters: 3 });
}

/** A `times` loop `lp` whose body returns only `n` of its carried `n` and `person`. */
function partialLoop(): LoopNode {
  const node = new LoopNode("lp", {
    flowId: "f",
    child: {},
    predicateKind: "times",
    times: 1,
    maxIters: 1,
    returnedFields: ["n"],
  });
  node.params = [new Parameter("n"), new Parameter("person")];
  return node;
}

test("loop is a spawner kind", () => {
  const node = new LoopNode("lp", { flowId: "f", child: {} });
  expect(node.isSpawner).toBe(true);
  expect(NodeKind.LOOP).toBe("loop");
});

test("compiled loop origin is self", () => {
  const node = new LoopNode("lp", {
    flowId: "f",
    child: {},
    predicateKind: "while",
    predicate: "${x}",
    maxIters: 10,
  });
  expect(node.originId).toBe("lp");
  expect(node.iteration).toBe(0);
});

test("respawn preserves assertion carriers", () => {
  const node = whileLoop();
  const origin = new OriginRef("nodes/loop/asserts/0");
  node.preAsserts = [new Assertion("x > 0", "x must be positive", { origin })];

  const clone = node.respawn(1);

  expect(clone.preAsserts).toEqual(node.preAsserts);
  expect(clone.preAsserts).not.toBe(node.preAsserts);
  expect(clone.preAsserts[0]!.origin).toBe(origin);
});

test("respawn preserves the full output contract carrier", () => {
  const node = whileLoop();
  const origins = new ContractOrigins({
    default: new OriginRef("nodes/loop/output"),
    entries: [[["answer"], new OriginRef("nodes/loop/output/answer")]],
  });
  const contract = new ValueContract(
    new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([["answer", Type.scalar(ValueKind.INTEGER)]]),
      required: new Set(["answer"]),
    }),
    origins,
  );
  node.outputContract = contract;

  const clone = node.respawn(1);

  expect(clone.outputContract).toBe(contract);
  expect((clone.outputContract as ValueContract).origins).toBe(origins);
  expect((clone.outputContract as ValueContract).type).toBe(contract.type);
});

test("respawn preserves the derived base memory scope", () => {
  const node = whileLoop();
  node.memoryScope = new MemoryScope(["loop"], { memories: ["outer"] });

  const clone = node.respawn(1);

  expect(clone.memoryScope).toBe(node.memoryScope);
});

test("respawn defaults omitted fields from the carried record", () => {
  const node = partialLoop();
  const originParams = node.params;
  const originSnapshot = [...node.params!];

  const clone = node.respawn(1, { n: 1n, person: { name: "A" } });

  const person = clone.params!.find((param) => param.name === "person")!;
  const n = clone.params!.find((param) => param.name === "n")!;
  expect(person.hasDefault).toBe(true);
  expect(person.default).toEqual({ name: "A" });
  expect(person.required).toBe(false);
  expect(n.hasDefault).toBe(false);
  expect(clone.returnedFields).toEqual(["n"]);
  expect(node.params).toBe(originParams);
  expect(node.params).toEqual(originSnapshot);
  expect(node.params!.every((param) => param.hasDefault === false)).toBe(true);
});

test("partial respawn requires the carried record", () => {
  expectError(
    () => partialLoop().respawn(1),
    RangeError,
    "loop 'lp' returns a subset of its record; respawn needs the carried record",
  );
});

test("respawn keeps a null omitted value as its default", () => {
  const clone = partialLoop().respawn(1, { n: 1n, person: null });
  const person = clone.params!.find((param) => param.name === "person")!;
  expect(person.hasDefault).toBe(true);
  expect(person.default).toBeNull();
});

test("total respawn keeps params", () => {
  const node = new LoopNode("lp", { flowId: "f", child: {}, predicateKind: "times", times: 1, maxIters: 1 });
  node.params = [new Parameter("n"), new Parameter("person")];

  const clone = node.respawn(1, { n: 1n, person: { name: "A" } });

  expect(clone.params).toBe(node.params);
  expect(clone.returnedFields).toBeNull();
});

test("a respawned driver is named for its origin and targets it", () => {
  const clone = whileLoop().respawn(2).respawn(3);
  expect(clone.id).toBe("loop~3");
  expect(clone.iteration).toBe(3);
  expect(clone.originId).toBe("loop");
  expect(clone.skippedOutputTarget()).toBe("loop");
});

test("respawn moves a scope that ends in the driver's position to index k", () => {
  const node = whileLoop();
  const outer = new ExecutionPosition("call", ["c"]);
  node.executionScope = new ExecutionScope(["c", "loop"], {
    flowPath: ["c"],
    occurrencePath: [outer, new ExecutionPosition("loop_driver", ["c", "loop"], 0)],
  });

  const scope = node.respawn(2).executionScope!;

  expect(scope).toBeInstanceOf(ExecutionScope);
  expect(scope.staticPath).toEqual(["c", "loop"]);
  expect(scope.flowPath).toEqual(["c"]);
  expect(scope.boundary).toBeNull();
  expect(scope.occurrencePath[0]).toBe(outer);
  expect(scope.occurrencePath[1]).toEqual(new ExecutionPosition("loop_driver", ["c", "loop"], 2));
});

test("respawn refuses an execution scope of another class", () => {
  class Subscope extends ExecutionScope {}
  const node = whileLoop();
  node.executionScope = new Subscope(["loop"], { occurrencePath: [new ExecutionPosition("loop_driver", ["loop"], 0)] });
  expectError(() => node.respawn(1), TypeError, "loop execution scope must be an ExecutionScope");
});

test("respawn refuses a scope that doesn't end in the driver's position", () => {
  const node = whileLoop();
  for (const occurrencePath of [
    [],
    [new ExecutionPosition("loop_driver", ["loop"], 1)],
    [new ExecutionPosition("loop_body", ["loop"], 0)],
    [new ExecutionPosition("loop_driver", ["other"], 0)],
  ]) {
    node.executionScope = new ExecutionScope(["loop"], { occurrencePath });
    expectError(() => node.respawn(1), RangeError, "loop scope must end in its current driver position");
  }
});

/** A `while` driver `lp` over `childFlow()`, its params the child START's. */
function grownLoop(): LoopNode {
  const child = childFlow();
  const node = new LoopNode("lp", { flowId: "f", child, predicateKind: "while", predicate: "${x}", maxIters: 10 });
  node.params = child.nodes.get(child.startId)!.params;
  return node;
}

test("buildGrow rebuilds the body and the next driver on the origin", () => {
  const node = grownLoop();
  const child = node.child as CompiledFlow;

  // The seed as the log's JSON gives it: body 2 and `lp~3`.
  const fragment = buildGrow(node, [{ x: 1n }, 2n]);

  expect(fragment.nodes.has(ns(mapCallsite("lp", 2), child.startId))).toBe(true);
  expect(fragment.nodes.has("lp~3")).toBe(true);
});

test("later loop growth does not duplicate the loop origin", () => {
  const outer = new OriginRef("root/nodes/call");
  const loopOrigin = new OriginRef("child/nodes/loop");
  const origin = grownLoop();
  origin.originContext = [outer];
  origin.origin = loopOrigin;

  const first = buildGrow(origin, [{ x: 1n }, 0n]);
  const driver = first.nodes.get("lp~1")!;
  const second = buildGrow(driver, [{ x: 1n }, 1n]);

  expect(driver.failureTrace()).toEqual([outer, loopOrigin]);
  expect(second.nodes.get("lp~2")!.failureTrace()).toEqual([outer, loopOrigin]);
  const bodyPrefix = ns(mapCallsite("lp", 1), "");
  const contexts = [...second.nodes]
    .filter(([nodeId]) => nodeId.startsWith(bodyPrefix))
    .map(([, node]) => node.originContext);
  expect(contexts.length).toBeGreaterThan(0);
  for (const context of contexts) expect(context).toEqual([outer, loopOrigin]);
});

test("should stop is the max iters budget", () => {
  const node = new LoopNode("l", { flowId: "f", child: {}, predicateKind: "while", predicate: "${x}", maxIters: 3 });
  expect(node.shouldStop(0)).toBe(false);
  expect(node.shouldStop(2)).toBe(false);
  expect(node.shouldStop(3)).toBe(true);
  expect(node.shouldStop(4)).toBe(true);
});

test("should stop without max never stops", () => {
  const node = new LoopNode("lp", {
    flowId: "f",
    child: {},
    predicateKind: "while",
    predicate: "${x}",
    maxIters: null,
  });
  expect(node.shouldStop(0)).toBe(false);
  expect(node.shouldStop(10 ** 9)).toBe(false);
});

test("a loop's other traits are read on an instance", () => {
  const node = new LoopNode("lp", { flowId: "f" });
  expect(node.kind).toBe(NodeKind.LOOP);
  expect(node.isLoop).toBe(true);
  expect(node.growDepthDelta).toBe(0);
});

test("the runaway guard is classified loop.max_exceeded", () => {
  const node = new LoopNode("lp", { flowId: "f" });
  const guard = new LoopMaxExceeded("x");
  expect(guard).toBeInstanceOf(Error);
  expect(guard.name).toBe("LoopMaxExceeded");
  expect(node.exceptionCode(guard)).toBe("loop.max_exceeded");
  expect(node.exceptionCode(new Error("x"))).toBe("node.execute");
});

test("a driver's skipped output targets its origin", () => {
  const clone = new LoopNode("lp~2", { flowId: "f", iteration: 2, originId: "lp" });
  expect(clone.originId).toBe("lp");
  expect(clone.skippedOutputTarget()).toBe("lp");
  expect(new LoopNode("lp", { flowId: "f" }).skippedOutputTarget()).toBe("lp");
});

/** A `while` driver `chat_loop` over `childFlow()` on `not exited`, its params the child START's. */
function chatLoop(): LoopNode {
  const child = childFlow();
  const node = new LoopNode("chat_loop", {
    flowId: "ct",
    child,
    predicateKind: "while",
    predicate: "not exited",
    maxIters: 1000,
  });
  node.params = child.nodes.get(child.startId)!.params;
  return node;
}

/** Return what `fn` throws, or fail when it returns. */
function thrown(fn: () => unknown): unknown {
  try {
    fn();
  } catch (error) {
    return error;
  }
  throw new Error("expected a throw");
}

test("run grows iteration zero with next driver when continue", async () => {
  const seed = { messages: [], exited: false };
  const result = await chatLoop().run({ ...seed });
  expect(result).toBeInstanceOf(Grow);
  expect((result as Grow).seed).toEqual([seed, 0n]);
  // k is 0: no previous body, and the origin never prunes itself.
  expect((result as Grow).prune.size).toBe(0);
  expect("subgraph" in (result as Grow)).toBe(false);
});

test("run commits seed when while predicate false", async () => {
  const node = new LoopNode("chat_loop", {
    flowId: "ct",
    child: {},
    predicateKind: "while",
    predicate: "${exited}",
    maxIters: 1000,
  });
  const seed = { messages: [], exited: false };
  const result = await node.run({ ...seed });
  expect(result).toBeInstanceOf(Output);
  expect((result as Output).value).toEqual(seed);
  expect((result as Output).commitAs).toBe("chat_loop");
});

test("clone driver self prunes previous body and self", async () => {
  const node = chatLoop();
  const clone = node.respawn(1);
  clone.params = node.params;
  expect([clone.id, clone.originId, clone.iteration]).toEqual(["chat_loop~1", "chat_loop", 1]);

  const result = await clone.run({ messages: ["hi"], exited: false });

  expect(result).toBeInstanceOf(Grow);
  const { prune, seed } = result as Grow;
  // The previous body (`chat_loop#0/...`) and this driver.
  expect(prune.has("chat_loop~1")).toBe(true);
  expect([...prune].some((id) => id.startsWith("chat_loop#0/"))).toBe(true);
  expect(seed).toEqual([{ messages: ["hi"], exited: false }, 1n]);
  expect("subgraph" in (result as Grow)).toBe(false);
});

test("respawned driver binds merged record", async () => {
  const clone = partialLoop().respawn(1, { n: 1n, person: { name: "A" } });

  const merged = await clone.run({ n: 2n });
  expect(merged).toBeInstanceOf(Output);
  expect((merged as Output).value).toEqual({ n: 2n, person: { name: "A" } });

  const explicitNull = await clone.run({ n: 2n, person: null });
  expect(explicitNull).toBeInstanceOf(Output);
  expect((explicitNull as Output).value).toEqual({ n: 2n, person: null });
});

test("until do while continues on iteration zero", async () => {
  const child = childFlow();
  const node = new LoopNode("lp", { flowId: "f", child, predicateKind: "until", predicate: "x <= 0", maxIters: 10 });
  node.params = child.nodes.get(child.startId)!.params;
  // `until` stops on a true post-check; k 0 always continues.
  expect(await node.run({ x: 5n })).toBeInstanceOf(Grow);
});

test("times stops at count", async () => {
  const child = childFlow();
  const node = new LoopNode("lp", { flowId: "f", child, predicateKind: "times", times: 3, maxIters: 3 });
  node.params = child.nodes.get(child.startId)!.params;
  const node3 = node.respawn(3);
  node3.params = node.params;

  const result = await node3.run({ x: 1n });

  expect(result).toBeInstanceOf(Output);
  expect((result as Output).commitAs).toBe("lp");
});

test("runaway guard raises loop max exceeded", async () => {
  const child = childFlow();
  // `max: 1` permits exactly one body run.
  const origin = new LoopNode("lp", { flowId: "f", child, predicateKind: "while", predicate: "${x}", maxIters: 1 });
  origin.params = child.nodes.get(child.startId)!.params;
  // driver@0 grows body_0, within budget.
  expect(await origin.run({ x: 1n })).toBeInstanceOf(Grow);
  // driver@1 would grow the second body: over budget.
  const driver = origin.respawn(1);
  const error = await rejected(driver.run({ x: 1n }));
  expect(error).toBeInstanceOf(LoopMaxExceeded);
  expect((error as Error).message).toBe("loop 'lp' exceeded max (1)");
  expect(driver.exceptionCode(error)).toBe("loop.max_exceeded");
});

test.each([["until", "${exited}"]] as const)(
  "unlimited driver far past any cap still grows: %s %s",
  async (kind, predicate) => {
    const child = childFlow();
    const origin = new LoopNode("chat_loop", { flowId: "ct", child, predicateKind: kind, predicate, maxIters: null });
    origin.params = child.nodes.get(child.startId)!.params;
    const driver = origin.respawn(100_000);
    expect(driver.maxIters).toBeNull();

    const result = await driver.run({ messages: [], exited: false });

    expect(result).toBeInstanceOf(Grow);
    expect(((result as Grow).seed as JsonValue[])[1]).toBe(100_000n);
  },
);

test("buildGrow matches live clone and origin replay byte for byte", async () => {
  // A live clone and the compiled origin's replay rebuild the same origin-keyed window.
  const origin = grownLoop();
  const clone = origin.respawn(2);

  const liveOutcome = await clone.run({ x: 1n });
  expect(liveOutcome).toBeInstanceOf(Grow);
  const live = buildGrow(clone, (liveOutcome as Grow).seed as JsonValue);
  // The log's canonical JSON turns the pair into a list.
  const replay = buildGrow(origin, [{ x: 1n }, 2n]);

  expect(live.startId).toBe(replay.startId);
  expect(live.endId).toBe(replay.endId);
  expect(new Set(live.nodes.keys())).toEqual(new Set(replay.nodes.keys()));
  expect(live.edges).toEqual(replay.edges);
  expect(live.wiring).toEqual(replay.wiring);
  const [liveDriver, replayDriver] = [live.nodes.get("lp~3") as LoopNode, replay.nodes.get("lp~3") as LoopNode];
  expect([liveDriver.originId, replayDriver.originId]).toEqual(["lp", "lp"]);
  expect([liveDriver.iteration, replayDriver.iteration]).toEqual([3, 3]);
});

test("buildGrow partial loop matches live and replay digest", async () => {
  // Live and replayed partial-loop growth build one fragment whose next driver keeps omitted fields.
  const origin = new LoopNode("lp", {
    flowId: "f",
    child: childFlow(),
    predicateKind: "while",
    predicate: "${x}",
    maxIters: 10,
    returnedFields: ["x"],
  });
  origin.params = [new Parameter("x"), new Parameter("exited")];
  const carried = { x: 1n, exited: false };
  const clone = origin.respawn(2, carried);

  const liveOutcome = await clone.run({ ...carried });
  expect(liveOutcome).toBeInstanceOf(Grow);
  const live = buildGrow(clone, (liveOutcome as Grow).seed as JsonValue);
  const replay = buildGrow(origin, [{ x: 1n, exited: false }, 2n]);

  expect(expansionDigest(live)).toBe(expansionDigest(replay));
  const exited = live.nodes.get("lp~3")!.params!.find((param) => param.name === "exited")!;
  expect(exited.hasDefault).toBe(true);
  expect(exited.default).toBe(carried.exited);
});

test("a while: that throws while evaluating is rethrown unchanged as condition.evaluate", async () => {
  const node = new LoopNode("lp", { flowId: "f", child: {}, predicateKind: "while", predicate: "n / 0 > 0" });
  const direct = thrown(() => evaluateWhenRecord("n / 0 > 0", { n: 1n }));

  const error = await rejected(node.run({ n: 1n }));

  expect(error).toBeInstanceOf(ExpressionError);
  expect((error as Error).message).toBe((direct as Error).message);
  expect(node.exceptionCode(error)).toBe("condition.evaluate");
  // The code follows the thrown error itself, not its text.
  expect(node.exceptionCode(new ExpressionError((error as Error).message))).toBe("node.execute");
});

test("an until: that throws at k >= 1 is rethrown unchanged as condition.evaluate", async () => {
  const origin = new LoopNode("lp", { flowId: "f", child: {}, predicateKind: "until", predicate: "x > 0" });
  const direct = thrown(() => evaluateWhenRecord("x > 0", { x: { a: 1n } }));
  const driver = origin.respawn(1);

  const error = await rejected(driver.run({ x: { a: 1n } }));

  expect(error).toBeInstanceOf(ExpressionError);
  expect((error as Error).message).toBe((direct as Error).message);
  expect(driver.exceptionCode(error)).toBe("condition.evaluate");
});

test("a failing predicate fails the run with condition.evaluate at the loop's origin", async () => {
  const origin = new OriginRef("nodes/lp");
  const loop = new LoopNode("lp", { flowId: "f", child: childFlow(), predicateKind: "while", predicate: "n / 0 > 0" });
  loop.params = [new Parameter("n")];
  loop.origin = origin;
  const flow = graph([loop], [[START_ID, "lp"], ["lp", END_ID]], { wiring: { lp: { n: "${input.n}" } } });

  const events = await engine(flow, { inputs: { n: 1n }, growBuilder: buildGrow }).run();

  const failed = events.filter((event) => event instanceof NodeFailed);
  expect(failed).toHaveLength(1);
  const { failure } = failed[0] as NodeFailed;
  expect(failure.code).toBe("condition.evaluate");
  expect(failure.causeType).toBe("ExpressionError");
  expect(failure.trace.at(-1)).toEqual(origin);
});

/** The object type `{f: file, n: int}`. */
function fileRecordType(): Type {
  return new Type({
    kind: ValueKind.OBJECT,
    fields: new Map([
      ["f", Type.scalar(ValueKind.FILE)],
      ["n", Type.scalar(ValueKind.INTEGER)],
    ]),
    required: new Set(["f", "n"]),
  });
}

/** A file's JSON form. */
function fileForm(): Record<string, string | null> {
  return { uri: "f", mime: null, name: null };
}

test("a stopping driver reads a file's JSON form as a FileRef under an exact contract", async () => {
  const node = new LoopNode("lp", { flowId: "f", child: {}, predicateKind: "while", predicate: "false" });
  node.outputContract = new ValueContract(fileRecordType());

  const result = await node.run({ f: fileForm(), n: 1n });

  expect(result).toBeInstanceOf(Output);
  expect((result as Output).value).toStrictEqual({ f: new FileRef({ uri: "f" }), n: 1n });
});

test("a stopping driver reads a partial contract's typed field and leaves an untyped one", async () => {
  const node = new LoopNode("lp", { flowId: "f", child: {}, predicateKind: "times", times: 0, maxIters: 0 });
  node.outputContract = new PartialRecordContract([
    ["f", new ValueContract(Type.scalar(ValueKind.FILE))],
    ["g", null],
  ]);

  const result = await node.run({ f: fileForm(), g: fileForm() });

  expect(result).toBeInstanceOf(Output);
  expect((result as Output).value).toStrictEqual({ f: new FileRef({ uri: "f" }), g: fileForm() });
});
