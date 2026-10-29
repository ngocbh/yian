/**
 * The raw `evalNode` read/execution boundary. The evaluator binds from flow-owned wiring, runs the
 * node, and resolves to exactly one `Output | Route | Grow | EvaluationFailure`.
 */

import { describe, expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { processEnv } from "../../src/defaults.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { RunSucceeded } from "../../src/events.ts";
import { RouteModel } from "../../src/llm/executor.ts";
import { Message, replayRecord, Request, TextPart } from "../../src/llm/messages.ts";
import { defaultModelAccess, type ModelAccess } from "../../src/llm_clients/factory.ts";
import {
  AssertionStageError,
  BindingStageError,
  CapabilityFault,
  Grow,
  Node,
  NodeAssertFailed,
  type NodeCaps,
  NodeKind,
  type NodeResult,
  Output,
  Route,
  type SteerInbox,
  type SteerRead,
} from "../../src/nodes/base.ts";
import { BindingError } from "../../src/nodes/binding.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { Case, CaseNode } from "../../src/nodes/case/node.ts";
import {
  Assertion,
  Binding,
  ContractOrigins,
  NodeSteps,
  Parameter,
  ValueContract,
} from "../../src/nodes/contracts.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
import { virtualScheduler } from "../../src/platform.ts";
import { EvaluationFailure, evalNode } from "../../src/runtime/eval_node.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { Type, type Value, ValueKind } from "../../src/typesys/values.ts";
import { drive, engine, FuncNode, graph, NO_ENV, nodeFlow, rejected, thrownBy as thrown } from "../helpers/run_state_fixtures.ts";

/** Return the failure `drive` resolves to, or fail when it resolves to anything else. */
async function driveFailure(
  node: Node,
  options: Parameters<typeof drive>[1] = {},
): Promise<EvaluationFailure["failure"]> {
  const outcome = await drive(node, options);
  expect(outcome).toBeInstanceOf(EvaluationFailure);
  return (outcome as EvaluationFailure).failure;
}

/** Return one configured result from its body. */
class ResultNode extends Node {
  readonly kind: NodeKind = NodeKind.CODE;
  readonly result: unknown;

  constructor(nodeId: string, result: unknown, options: { origin?: OriginRef | null } = {}) {
    super(nodeId, options);
    this.result = result;
  }

  override forward(): NodeResult {
    return this.result as NodeResult;
  }
}

/** Return one configured route handle (a CASE-shaped router). */
class RouteNode extends ResultNode {
  override readonly kind = NodeKind.CASE;
}

test("map over accepts a binding carrier", async () => {
  // MAP's reserved iteration source uses the ordinary binding read seam.
  const node = new MapNode("m", { flowId: "child", child: {} });
  const source = new Binding(["first", "second"], { origin: new OriginRef("map-over") });
  const wiring = new Map([["over", source]]);

  expect(node.bindReserved(wiring, new VariablePool())).toEqual({ over: ["first", "second"] });
  expect(await drive(node, { flow: nodeFlow(node, { over: source }) })).toEqual(new Grow([{}, {}]));
});

for (const [label, make, inputs] of [
  ["call", () => new CallNode("call", { flowId: "child" }), {}],
  ["map", () => new MapNode("map", { flowId: "child" }), { over: [] }],
  ["loop", () => new LoopNode("loop", { flowId: "child", predicate: "true", maxIters: 1 }), {}],
] as const) {
  test(`unbaked composite is a setup fault (${label})`, async () => {
    // A missing compiled child is malformed setup, not expected node rejection.
    const node: Node = make();
    node.preAsserts = [new Assertion("false", "must not mask malformed setup")];

    await expect(node.run(inputs)).rejects.toThrow("baked");
  });
}

test("map over not a list message is byte identical", () => {
  // Raw and carrier-wrapped sources produce the same origin-neutral error text.
  const pool = new VariablePool();
  pool.set("__start__", { bad: "notalist" });
  const rawSource = "${input.bad}";
  const origin = new OriginRef("map-over");
  for (const source of [rawSource, new Binding(rawSource, { origin })]) {
    const node = new MapNode("m", { flowId: "f" });
    let caught: unknown = null;
    try {
      node.bindReserved(new Map([["over", source]]), pool);
    } catch (error) {
      caught = error;
    }
    expect(caught).toBeInstanceOf(BindingError);
    expect((caught as BindingError).message).toBe("MAP node 'm': `over` (${input.bad}) did not resolve to a list");
    expect((caught as BindingError).origin).toBe(source instanceof Binding ? origin : null);
  }
});

test("case without predicate is a setup fault", async () => {
  // A compiled CASE branch must have its validated predicate.
  const node = new CaseNode("case", [new Case("branch")]);

  await expect(node.run({})).rejects.toThrow(RangeError);
  await expect(node.run({})).rejects.toThrow("node 'case' case 'branch' has no `when` expression");
});

test("case with non string predicate is a setup fault", async () => {
  // Malformed code-built CASE state must not look like node execution.
  const malformed = new Case("branch", { when: 42 as unknown as string });
  const node = new CaseNode("case", [malformed]);

  await expect(node.run({})).rejects.toThrow(TypeError);
  await expect(node.run({})).rejects.toThrow("node 'case' case 'branch' `when` must be a string");
});

test("case rejects a non origin predicate carrier", () => {
  // Code-built CASE provenance must stay source-neutral.
  const make = () => new Case("branch", { when: "true", origin: "flow.yaml:12" as unknown as OriginRef });

  expect(make).toThrow(TypeError);
  expect(make).toThrow("CASE predicate origin");
});

test("case predicate failure selects the active ordered origin", async () => {
  // The failing arm wins even when an earlier ordered predicate was evaluated.
  const firstOrigin = new OriginRef("case/0/when");
  const secondOrigin = new OriginRef("case/1/when");
  const node = new CaseNode("case", [
    new Case("first", { when: "1 == 2", origin: firstOrigin }),
    new Case("second", { when: "1 / 0 > 0", origin: secondOrigin }),
  ]);
  node.origin = new OriginRef("case/node");

  const failure = await driveFailure(node);

  expect(failure.code).toBe("condition.evaluate");
  expect(failure.causeType).toBe("ExpressionError");
  expect(failure.trace).toEqual([secondOrigin]);
});

test("source free case predicate failure falls back to the node origin", async () => {
  // A predicate without its own site retains the common node fallback.
  const nodeOrigin = new OriginRef("case/node");
  const node = new CaseNode("case", [new Case("broken", { when: "1 / 0 > 0" })]);
  node.origin = nodeOrigin;

  const failure = await driveFailure(node);

  expect(failure.code).toBe("condition.evaluate");
  expect(failure.causeType).toBe("ExpressionError");
  expect(failure.trace).toEqual([nodeOrigin]);
});

test("success returns output without observer events", async () => {
  expect(await drive(new FuncNode("n", () => ({ output: "ok" })))).toEqual(new Output({ output: "ok" }));
});

test("baked commit target remains engine owned", async () => {
  const node = new ResultNode("n", new Output("v"));
  node.commitAs = "spawner";

  const result = await drive(node);

  expect(result).toEqual(new Output("v"));
  expect((result as Output).commitAs).toBeNull();
});

test("node chosen commit target remains on output", async () => {
  const node = new ResultNode("n", new Output("v", { commitAs: "chosen" }));
  node.commitAs = "spawner";

  expect(await drive(node)).toEqual(new Output("v", { commitAs: "chosen" }));
});

test("binding failure returns diagnostic evaluation failure", async () => {
  const node = new FuncNode("n", (inputs) => inputs, { origin: new OriginRef("node:n") });
  node.params = [new Parameter("x", { required: true })];

  const failure = await driveFailure(node);

  expect(failure.code).toBe("binding.evaluate");
  expect(failure.trace).toEqual([node.origin!]);
  expect(failure.causeType).toBe("BindingError");
});

test("binding carrier resolves at the node read boundary", async () => {
  // Ordinary parameter binding evaluates the carrier source without changing it.
  const pool = new VariablePool();
  pool.set(START_ID, { value: "resolved" });
  const origin = new OriginRef("binding");
  const wrapped = new Binding("${input.value}", { origin });
  const node = new FuncNode("n", (inputs) => inputs.x, { reads: { x: wrapped } });

  expect(await drive(node, { pool })).toEqual(new Output("resolved"));
  expect(node.reads.x).toBe(wrapped);
  expect(wrapped.origin).toBe(origin);
});

test("binding carrier unwraps exactly one layer", async () => {
  // A nested carrier stays an opaque payload after the outer layer, never a second source.
  const pool = new VariablePool();
  pool.set(START_ID, { value: "must-not-resolve" });
  const inner = new Binding("${input.value}", { origin: new OriginRef("inner") });
  const outer = new Binding(inner, { origin: new OriginRef("outer") });
  const node = new FuncNode("n", (inputs) => inputs.x, { reads: { x: outer } });

  const failure = await driveFailure(node, { pool });

  expect(failure.code).toBe("binding.evaluate");
  expect(failure.message).not.toContain("must-not-resolve");
});

test("routing returns route without output value", async () => {
  const result = await drive(new RouteNode("n", new Route("case_a")));

  expect(result).toEqual(new Route("case_a"));
  expect("value" in (result as object)).toBe(false);
});

test("non spawner grow returns clear evaluation failure", async () => {
  const node = new ResultNode("n", new Grow({}), { origin: new OriginRef("node:n") });

  const failure = await driveFailure(node);

  expect(failure.code).toBe("contract.node_result");
  expect(failure.trace).toEqual([node.origin!]);
  expect(failure.causeType).toBe("Error");
  expect(failure.message).toBe(
    "node 'n' (kind code) returned a Grow but is not a spawner (CALL/MAP/LOOP); only spawner kinds may grow the graph",
  );
});

test("bad return type returns clear evaluation failure", async () => {
  const failure = await driveFailure(new ResultNode("n", { not: "a NodeResult" }));

  expect(failure.code).toBe("contract.node_result");
  expect(failure.trace).toEqual([]);
  expect(failure.causeType).toBe("Error");
  expect(failure.message).toBe("node 'n' run() returned Object, not a NodeResult");
});

for (const [label, make, message] of [
  ["map", () => new MapNode("map", { flowId: "child", child: {} }), "MAP node 'map' has no `over` wiring"],
] as const) {
  test(`missing reserved wiring is an ir fault (${label})`, async () => {
    // A validated compiled graph has every trait-declared reserved key.
    expect(await rejected(drive(make()))).toEqual(new Error(message));
  });
}

test("malformed parameter carrier is an ir fault", async () => {
  // Malformed compiled setup escapes instead of becoming a flow failure.
  const node = new FuncNode("n", (inputs) => inputs);
  node.params = { 0: new Parameter("x"), length: 1 } as unknown as Parameter[];

  expect(await rejected(drive(node))).toEqual(new TypeError("node params must be an exact list, got Object"));
});

for (const [label, make, field, value] of [
  ["map", () => new MapNode("map", { flowId: "child", child: {} }), "over", "not-a-list"],
] as const) {
  test(`reserved input rejection is an input contract failure (${label})`, async () => {
    // An expected reserved-value rejection keeps its authored binding origin.
    const node: Node = make();
    const origin = new OriginRef(`${node.id}:${field}`);

    const failure = await driveFailure(node, { flow: nodeFlow(node, { [field]: new Binding(value, { origin }) }) });

    expect(failure.code).toBe("contract.input");
    expect(failure.trace).toEqual([origin]);
  });
}

test("map over not a list message is byte identical through the evaluator", async () => {
  const pool = new VariablePool();
  pool.set(START_ID, { bad: "notalist" });
  const rawSource = "${input.bad}";
  for (const source of [rawSource, new Binding(rawSource, { origin: new OriginRef("map-over") })]) {
    const node = new MapNode("m", { flowId: "f" });
    const failure = await driveFailure(node, { pool, flow: nodeFlow(node, { over: source }) });

    expect(failure.message).toBe("MAP node 'm': `over` (${input.bad}) did not resolve to a list");
  }
});

/** Reserve key `k`, whose binding always fails, and record steps. */
class StepsReservedNode extends ResultNode {
  override readonly needsSteps: boolean = true;

  override bindReserved(): Record<string, unknown> {
    throw new BindingError("node 'n' reserved key failed", { failureCode: "contract.input" });
  }
}

test("a reserved binding failure after an earlier start recorded steps is a host fault", async () => {
  // The earlier start resolved the key from the same restored pool, so the flow or yian changed.
  const node = new StepsReservedNode("n", new Output("unused"));
  const steps = new NodeSteps(2, [{ step: 1n }], () => 1);

  expect(await rejected(drive(node, { steps }))).toEqual(
    new RangeError(
      "node 'n' reserved key failed; an earlier start of this node resolved it, so resume the run with the " +
        "flow and yian version that started it",
    ),
  );
});

test("a reserved binding failure with no prior steps stays a node failure", async () => {
  const node = new StepsReservedNode("n", new Output("unused"));

  for (const steps of [null, new NodeSteps(1, [], () => 1)]) {
    expect((await driveFailure(node, { steps })).code).toBe("contract.input");
  }
});

test("a reserved binding failure of a node that doesn't read its steps stays a node failure", async () => {
  // Only a node reading its steps saw the earlier start's resolution.
  class NoStepsReservedNode extends StepsReservedNode {
    override readonly needsSteps = false;
  }
  const steps = new NodeSteps(2, [{ step: 1n }], () => 1);

  expect((await driveFailure(new NoStepsReservedNode("n", new Output("unused")), { steps })).code).toBe(
    "contract.input",
  );
});

/** A model access whose builder is never called. */
const MODELS: ModelAccess = {
  env: NO_ENV,
  createModel: () => {
    throw new Error("unused");
  },
};

/** Record the capabilities each run receives, and return `null`. */
class CapsNode extends Node {
  readonly kind: NodeKind = NodeKind.CODE;
  readonly seen: NodeCaps[];

  constructor(nodeId: string, seen: NodeCaps[]) {
    super(nodeId);
    this.seen = seen;
  }

  override forward(_inputs: Record<string, unknown>, caps: NodeCaps): NodeResult {
    this.seen.push(caps);
    return new Output(null);
  }
}

/** A `CapsNode` whose traits ask for tools, steps and steering. */
class TraitCapsNode extends CapsNode {
  override readonly kind = NodeKind.AGENT;
  override readonly needsTools = true;
  override readonly needsSteps = true;
  override readonly readsSteerInbox = true;
}

describe("evalNode gives every node the scheduler, the random draw and the run folder", () => {
  const pool = new VariablePool();

  for (const make of [
    (seen: NodeCaps[]) => new CapsNode("n", seen),
    (seen: NodeCaps[]) => new TraitCapsNode("n", seen),
  ]) {
    test(`exactly the given objects (${make([]).constructor.name})`, async () => {
      const seen: NodeCaps[] = [];
      const scheduler = virtualScheduler();
      const random = (): number => 0.25;

      await evalNode(make(seen), null, pool, { scheduler, random, runDir: "/runs/r1" });

      expect(seen).toHaveLength(1);
      expect(seen[0]!.scheduler).toBe(scheduler);
      expect(seen[0]!.random).toBe(random);
      expect(seen[0]!.runDir).toBe("/runs/r1");
    });

    test(`neither key when not given (${make([]).constructor.name})`, async () => {
      const seen: NodeCaps[] = [];

      await evalNode(make(seen), null, pool, { scheduler: null, random: null, runDir: null });
      await evalNode(make(seen), null, pool);

      expect(seen).toHaveLength(2);
      for (const caps of seen) {
        expect("scheduler" in caps).toBe(false);
        expect("random" in caps).toBe(false);
        expect("runDir" in caps).toBe(false);
      }
    });
  }
});

describe("FlowEngine gives every node the scheduler, the random draw and the run folder", () => {
  test("each evaluated node receives the given objects", async () => {
    const seen: NodeCaps[] = [];
    const flow = graph([new CapsNode("a", seen), new TraitCapsNode("b", seen)], [
      [START_ID, "a"],
      ["a", "b"],
      ["b", END_ID],
    ]);
    const scheduler = virtualScheduler();
    const random = (): number => 0.75;

    expect((await engine(flow, { scheduler, random, runDir: "/runs/r1" }).run()).at(-1)).toBeInstanceOf(RunSucceeded);
    expect(seen).toHaveLength(2);
    for (const caps of seen) {
      expect(caps.scheduler).toBe(scheduler);
      expect(caps.random).toBe(random);
      expect(caps.runDir).toBe("/runs/r1");
    }
  });
});

/** Record the capabilities each evaluation supplies, by node id. */
class CapsProbe extends Node {
  readonly kind = NodeKind.CODE;
  readonly seen: Map<string, NodeCaps>;

  constructor(nodeId: string, seen: Map<string, NodeCaps>) {
    super(nodeId);
    this.seen = seen;
  }

  override forward(_inputs: Record<string, unknown>, caps: NodeCaps): NodeResult {
    this.seen.set(this.id, caps);
    return new Output("ok");
  }
}

/** A probe that declares it calls models. */
class ModelsProbe extends CapsProbe {
  override readonly needsModels = true;
}

/** A probe that declares it reads a steer inbox. */
class SteerReaderProbe extends CapsProbe {
  override readsSteerInbox = true;
}

describe("the steer inbox capability", () => {
  /** An opener whose inbox `m` reads `read()`'s answer, counting reads. */
  function opener(read: () => SteerRead | null = () => null) {
    const calls: string[] = [];
    const open = (inbox: string): SteerInbox => {
      calls.push(`open ${inbox}`);
      return {
        inbox,
        queued: () => (calls.push("queued"), false),
        read: () => (calls.push("read"), read()),
        interrupt: new AbortController().signal, interruptedBy: () => null,
      };
    };
    return { open, calls };
  }

  test("openSteerInbox reaches only a node declaring readsSteerInbox", async () => {
    const seen = new Map<string, NodeCaps>();
    const { open } = opener();
    await drive(new CapsProbe("plain", seen), { steerInbox: open });
    await drive(new SteerReaderProbe("chat", seen), { steerInbox: open });
    expect("openSteerInbox" in seen.get("plain")!).toBe(false);
    expect(typeof seen.get("chat")!.openSteerInbox).toBe("function");
  });

  test("a steer read after the evaluation settled is refused", async () => {
    const seen = new Map<string, NodeCaps>();
    const { open, calls } = opener();
    await drive(new SteerReaderProbe("chat", seen), { steerInbox: open });
    const kept = seen.get("chat")!.openSteerInbox!;
    const error = thrown(() => kept("m"));
    expect(error).toBeInstanceOf(CapabilityFault);
    expect((error as CapabilityFault).original).toEqual(new Error("steer inbox capability is closed"));
    expect(calls).toEqual([]);
  });

  test("a read opened while running is refused once the evaluation settled", async () => {
    let inbox: SteerInbox | null = null;
    const { open, calls } = opener();
    class Keeper extends SteerReaderProbe {
      override forward(inputs: Record<string, unknown>, caps: NodeCaps): NodeResult {
        inbox = caps.openSteerInbox!("m");
        return super.forward(inputs, caps);
      }
    }
    await drive(new Keeper("chat", new Map()), { steerInbox: open });
    expect(thrown(() => inbox!.read())).toBeInstanceOf(CapabilityFault);
    expect(thrown(() => inbox!.queued())).toBeInstanceOf(CapabilityFault);
    expect(calls).toEqual(["open m"]);
  });

  test("a refused read's fault reaches onFault and rejects the evaluation, even when the node catches it", async () => {
    const original = new RangeError("refused");
    const { open } = opener(() => {
      throw new CapabilityFault(original);
    });
    const faults: unknown[] = [];
    class Catcher extends SteerReaderProbe {
      override forward(inputs: Record<string, unknown>, caps: NodeCaps): NodeResult {
        try {
          caps.openSteerInbox!("m").read();
        } catch {
          // The node hides the fault; the evaluator still sees it.
        }
        return super.forward(inputs, caps);
      }
    }
    const error = await rejected(drive(new Catcher("chat", new Map()), { steerInbox: open, onFault: (e) => faults.push(e) }));
    expect(error).toBe(original);
    expect(faults).toEqual([original]);
  });
});

describe("the models capability", () => {
  test("reaches a node declaring needsModels through evalNode, and not a node without it", async () => {
    const seen = new Map<string, NodeCaps>();

    expect(await drive(new CapsProbe("plain", seen), { models: MODELS })).toEqual(new Output("ok"));
    expect(await drive(new ModelsProbe("agent", seen), { models: MODELS })).toEqual(new Output("ok"));

    expect("models" in seen.get("plain")!).toBe(false);
    expect(seen.get("agent")!.models).toBe(MODELS);
  });

  test("reaches a node declaring needsModels through FlowEngine, and not a node without it", async () => {
    const seen = new Map<string, NodeCaps>();
    const flow = graph([new CapsProbe("plain", seen), new ModelsProbe("agent", seen)], [
      [START_ID, "plain"],
      [START_ID, "agent"],
      ["plain", END_ID],
      ["agent", END_ID],
    ]);

    await engine(flow, { models: MODELS }).run();

    expect("models" in seen.get("plain")!).toBe(false);
    expect(seen.get("agent")!.models).toBe(MODELS);
  });

  test("without models, a node declaring needsModels gets no caps.models", async () => {
    const seen = new Map<string, NodeCaps>();

    await drive(new ModelsProbe("absent", seen));
    await drive(new ModelsProbe("null", seen), { models: null });

    expect("models" in seen.get("absent")!).toBe(false);
    expect("models" in seen.get("null")!).toBe(false);
  });

  test("defaultModelAccess keeps the given env and builds through chatModelFromConfig", async () => {
    const env = (name: string): string | undefined => (name === "OPENAI_API_KEY" ? "sk-test" : NO_ENV(name));
    const access = defaultModelAccess(env);
    const signal = new AbortController().signal;

    expect(access.env).toBe(env);
    expect(defaultModelAccess().env).toBe(processEnv);
    // The key comes only from `env`, so the build succeeds only when `env` reaches the factory.
    const model = await access.createModel({ provider: "openai", model: "gpt-5.5" }, { nodeId: "n", signal });
    expect(model).toBeInstanceOf(RouteModel);
    expect(await (model as RouteModel).route.auth.headers()).toEqual({ Authorization: "Bearer sk-test" });
  });

  test("defaultModelAccess passes onReplayDegraded to chatModelFromConfig", async () => {
    const env = (name: string): string | undefined => (name === "OPENAI_API_KEY" ? "sk-test" : NO_ENV(name));
    const reasons: string[] = [];
    const onReplayDegraded = (reason: string) => void reasons.push(reason);
    const signal = new AbortController().signal;
    const config = { provider: "openai", model: "gpt-5.5" };
    const model = await defaultModelAccess(env).createModel(config, { nodeId: "n", signal, onReplayDegraded });
    // Text another provider's reply recorded degrades when this model replays it.
    const key = { protocol: (model as RouteModel).protocol.name, provider: "azure", model: "gpt-5.5" };
    const text = new TextPart("Hello.", replayRecord(key, { id: "msg_1" }));
    const messages = [new Message({ role: "user", parts: [new TextPart("hi")] }), new Message({ role: "assistant", parts: [text] })];
    (model as RouteModel).protocol.buildBody(new Request({ messages }));
    expect(reasons).toEqual(["other_provider"]);
  });
});

/** Throw `error` from its body, after an `await` when `async`. */
class RaiseNode extends Node {
  readonly kind = NodeKind.CODE;
  readonly error: unknown;
  readonly async: boolean;

  constructor(nodeId: string, error: unknown, options: { origin?: OriginRef | null; async?: boolean } = {}) {
    const { async = false, ...rest } = options;
    super(nodeId, rest);
    this.error = error;
    this.async = async;
  }

  override forward(): NodeResult | Promise<NodeResult> {
    if (!this.async) throw this.error;
    return Promise.resolve().then(() => {
      throw this.error;
    });
  }
}

test("node raise returns evaluation failure", async () => {
  const node = new RaiseNode("n", new Error("boom"), { origin: new OriginRef("node:n") });

  const failure = await driveFailure(node);

  expect(failure.code).toBe("node.execute");
  expect(failure.trace).toEqual([node.origin!]);
  expect([failure.message, failure.causeType]).toEqual(["boom", "Error"]);
});

/** Change its input record, to test the template's post-assert snapshot. */
class MutatingNode extends Node {
  readonly kind = NodeKind.CODE;

  constructor(nodeId: string) {
    super(nodeId);
    this.params = [new Parameter("x")];
  }

  override forward(inputs: Record<string, unknown>): NodeResult {
    inputs.x = 999n;
    return new Output("done");
  }
}

test("post assert isolated from node input mutation", async () => {
  const pool = new VariablePool();
  pool.set(START_ID, { x: 1n });
  const node = new MutatingNode("n");
  node.postAsserts = [new Assertion("x == 1", "x == 1")];

  expect(await drive(node, { pool, flow: nodeFlow(node, { x: "${input.x}" }) })).toEqual(new Output("done"));
});

test("wrapper assertion failure uses assertion stage and origin", async () => {
  // Only the template's own authored assertion gets `assertion.failed`.
  const assertionOrigin = new OriginRef("assert:n:0");
  const node = new FuncNode("n", () => "never", { origin: new OriginRef("node:n") });
  node.preAsserts = [new Assertion("1 == 2", "must hold", { origin: assertionOrigin })];

  const failure = await driveFailure(node);

  expect(failure.code).toBe("assertion.failed");
  expect(failure.trace).toEqual([assertionOrigin]);
  expect(failure.causeType).toBe("NodeAssertFailed");
  expect(failure.traceback).toStartWith("NodeAssertFailed: node 'n' pre-assert failed: must hold\n");
  expect(failure.traceback).not.toContain("AssertionStageError");
});

test("wrapper post assertion uses assertion stage and origin", async () => {
  // Post-asserts keep the same template-owned classification as pre-asserts.
  const assertionOrigin = new OriginRef("assert:n:post:0");
  const node = new FuncNode("n", () => "value", { origin: new OriginRef("node:n") });
  node.postAsserts = [new Assertion("output == 'other'", "output must match", { origin: assertionOrigin })];

  const failure = await driveFailure(node);

  expect(failure.code).toBe("assertion.failed");
  expect(failure.trace).toEqual([assertionOrigin]);
  expect(failure.causeType).toBe("NodeAssertFailed");
});

test("body raised assertion exception remains node execution", async () => {
  // The classification follows the body stage, not the error's class.
  const node = new RaiseNode("n", new NodeAssertFailed("body chose this exception type"), {
    origin: new OriginRef("node:n"),
  });

  const failure = await driveFailure(node);

  expect(failure.code).toBe("node.execute");
  expect(failure.trace).toEqual([node.origin!]);
  expect(failure.causeType).toBe("NodeAssertFailed");
});

for (const async of [false, true]) {
  test(`body raised binding exception remains node execution (${async ? "async" : "body"})`, async () => {
    // Only evaluator-owned binding operations may keep the binder's classification.
    const error = new BindingError("body chose this exception type", {
      failureCode: "contract.input",
      origin: new OriginRef("spoofed-binding-origin"),
    });
    const node = new RaiseNode("n", error, { origin: new OriginRef("node:n"), async });

    const failure = await driveFailure(node);

    expect(failure.code).toBe("node.execute");
    expect(failure.trace).toEqual([node.origin!]);
    expect(failure.causeType).toBe("BindingError");
  });
}

/** Throw from the body while naming a finer operation origin. */
class OperationOriginNode extends RaiseNode {
  readonly operationOrigin: OriginRef | null;

  constructor(nodeId: string, operationOrigin: OriginRef | null) {
    super(nodeId, new Error("operation failed"), { origin: new OriginRef(`node:${nodeId}`) });
    this.operationOrigin = operationOrigin;
  }

  override exceptionOrigin(): OriginRef | null {
    return this.operationOrigin;
  }
}

for (const [operationOrigin, expectedKey] of [
  [new OriginRef("operation:n"), "operation:n"],
  [null, "node:n"],
] as const) {
  test(`node execution prefers operation origin then node fallback (${expectedKey})`, async () => {
    // A body failure appends the selected local origin after the outer ancestry.
    const outer = new OriginRef("outer:call");
    const node = new OperationOriginNode("n", operationOrigin);
    node.originContext = [outer];

    expect((await driveFailure(node)).trace).toEqual([outer, new OriginRef(expectedKey)]);
  });
}

/** Return or throw a configured body outcome while classifying every body error as `provider.auth`. */
class ExceptionCodeNode extends ResultNode {
  constructor(nodeId: string, result: unknown) {
    super(nodeId, result, { origin: new OriginRef(`node:${nodeId}`) });
  }

  override forward(): NodeResult {
    if (this.result instanceof Error) throw this.result;
    return this.result as NodeResult;
  }

  override exceptionCode(): "provider.auth" {
    return "provider.auth";
  }
}

/** Bypass the base template with an `async` `run` that rejects with `error`. */
class OverrideRaisesNode extends Node {
  readonly kind = NodeKind.CODE;
  readonly error: unknown;

  constructor(nodeId: string, error: unknown) {
    super(nodeId, { origin: new OriginRef(`node:${nodeId}`) });
    this.error = error;
  }

  override forward(): NodeResult {
    return new Output("unused");
  }

  override async run(): Promise<NodeResult> {
    throw this.error;
  }
}

/** Bypass the base template while classifying every body error as `provider.auth`. */
class OverrideRaisesWithCode extends OverrideRaisesNode {
  override exceptionCode(): "provider.auth" {
    return "provider.auth";
  }
}

for (const [label, make] of [
  ["body", () => new ExceptionCodeNode("n", new Error("unauthorized"))],
  ["run override", () => new OverrideRaisesWithCode("n", new Error("unauthorized"))],
] as const) {
  test(`node execution uses the node exception code (${label})`, async () => {
    // A body error takes the code the node's hook selects.
    const node: Node = make();

    const failure = await driveFailure(node);

    expect(failure.code).toBe("provider.auth");
    expect(failure.trace).toEqual([node.origin!]);
    expect(failure.causeType).toBe("Error");
  });
}

test("base exception code is node execution", () => {
  // A node without a finer classification reports `node.execute`.
  expect(new ResultNode("n", null).exceptionCode(new Error("boom"))).toBe("node.execute");
});

for (const [label, result, configure, expectedCode] of [
  [
    "unbound parameter",
    new Output("unused"),
    (node: Node) => {
      node.params = [new Parameter("x", { required: true })];
    },
    "binding.evaluate",
  ],
  [
    "failing pre-assert",
    new Output("unused"),
    (node: Node) => {
      node.preAsserts = [new Assertion("1 == 2", "must hold")];
    },
    "assertion.failed",
  ],
  ["bad result", { not: "a NodeResult" }, () => {}, "contract.node_result"],
] as const) {
  test(`exception code is asked only for the body (${label})`, async () => {
    // The binding, assertion and result-contract stages keep their own codes.
    const node = new ExceptionCodeNode("n", result);
    configure(node);

    expect((await driveFailure(node)).code).toBe(expectedCode);
  });
}

for (const [label, error, expectedCauseType] of [
  ["ordinary", new Error("ordinary override failure"), "Error"],
  ["binding error", new BindingError("spoofed", { failureCode: "binding.evaluate" }), "BindingError"],
  [
    "assertion error",
    new NodeAssertFailed("spoofed", { assertion: new Assertion("true", "true", { origin: new OriginRef("spoofed") }) }),
    "NodeAssertFailed",
  ],
  [
    "binding stage marker",
    new BindingStageError(new BindingError("wrapped", { failureCode: "binding.evaluate" })),
    "BindingError",
  ],
  [
    "assertion stage marker",
    new AssertionStageError(new NodeAssertFailed("wrapped", { assertion: new Assertion("true", "true") })),
    "NodeAssertFailed",
  ],
] as const) {
  test(`run override exception is node execution (${label})`, async () => {
    // A custom run is one undivided node-body stage.
    const node = new OverrideRaisesNode("n", error);

    const failure = await driveFailure(node);

    expect(failure.code).toBe("node.execute");
    expect(failure.trace).toEqual([node.origin!]);
    expect(failure.causeType).toBe(expectedCauseType);
    expect(failure.traceback).not.toContain("BindingStageError");
    expect(failure.traceback).not.toContain("AssertionStageError");
  });
}

/** A node class that defines no `forward`. */
class NoForwardNode extends Node {
  readonly kind = NodeKind.CODE;
}

test("missing node implementation is a setup fault", async () => {
  // The base placeholder body is malformed executable setup.
  expect(await rejected(drive(new NoForwardNode("n")))).toEqual(
    new Error("node 'n' (NoForwardNode) defines no forward"),
  );
});

test("malformed assertion carrier is an ir fault", async () => {
  // An unexpected template-state fault is outside expected flow failure.
  const node = new FuncNode("n", () => "never");
  node.preAsserts = [{} as Assertion];

  expect(await rejected(drive(node))).toBeInstanceOf(TypeError);
});

test("map deferred item binding keeps input contract stage", async () => {
  // The per-item binder capability doesn't become a node-body failure.
  const contractOrigin = new OriginRef("map:x:contract");
  const node = new MapNode("m", { flowId: "child", child: {} });
  node.params = [
    new Parameter("x", {
      contract: new ValueContract(Type.scalar(ValueKind.INTEGER), new ContractOrigins({ default: contractOrigin })),
    }),
  ];

  const failure = await driveFailure(node, {
    flow: nodeFlow(node, { over: ["bad"], x: new Binding("${item}", { origin: new OriginRef("map:x:binding") }) }),
  });

  expect(failure.code).toBe("contract.input");
  expect(failure.trace).toEqual([contractOrigin]);
});

test("map deferred expression keeps binding stage", async () => {
  // A per-item expression failure stays evaluator-owned binding work.
  const bindingOrigin = new OriginRef("map:x:binding");
  const node = new MapNode("m", { flowId: "child", child: {} });
  node.params = [new Parameter("x")];

  const failure = await driveFailure(node, {
    flow: nodeFlow(node, {
      over: [{}],
      x: new Binding('${item.missing :? "x is required"}', { origin: bindingOrigin }),
    }),
  });

  expect(failure.code).toBe("binding.evaluate");
  expect(failure.trace).toEqual([bindingOrigin]);
});

test("map deferred unexpected binder fault escapes", async () => {
  // Deferred binding keeps the ordinary binder's host and setup boundary.
  const expected = new Error("item read failed");
  const item = Object.defineProperty({}, "x", {
    enumerable: true,
    get() {
      throw expected;
    },
  }) as Value;
  const node = new MapNode("m", { flowId: "child", child: {} });
  node.params = [new Parameter("x")];

  expect(await rejected(drive(node, { flow: nodeFlow(node, { over: [item], x: "${item}" }) }))).toBe(expected);
});

test("per item wrapper default keeps binding stage", async () => {
  // The template's per-item default backstop keeps the default's origin.
  const defaultOrigin = new OriginRef("map:x:default");
  const node = new MapNode("m", { flowId: "child", child: {} });
  node.params = [
    new Parameter("x", {
      contract: new ValueContract(Type.scalar(ValueKind.INTEGER)),
      hasDefault: true,
      default: "bad",
      defaultOrigin,
    }),
  ];

  const failure = await driveFailure(node, { flow: nodeFlow(node, { over: [] }) });

  expect(failure.code).toBe("binding.evaluate");
  expect(failure.trace).toEqual([defaultOrigin]);
});
