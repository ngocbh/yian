/**
 * Source-neutral, execution-scoped access to recalled memory notes. The local opener records each
 * entry and resolves to an empty context.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { StreamChunk } from "../../src/events.ts";
import { MemoryContext, MemoryScope, MemorySection, type RecallContext } from "../../src/memory/contracts.ts";
import { CapabilityFault, Node, type NodeCaps, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import { EvaluationFailure, evalNode, type EvalNodeOptions, type MemoryContextOpener } from "../../src/runtime/eval_node.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { FuncNode, rejected } from "../helpers/run_state_fixtures.ts";

type Open = NonNullable<NodeCaps["openMemoryContext"]>;

/** A scoped memory consumer whose body is `body`. */
class Consumer extends Node {
  readonly kind = NodeKind.CODE;
  override readonly needsMemoryContext = true;
  override readonly needsLiveEvents = true;
  override readonly readsSteerInbox = true;
  readonly body: (open: Open, caps: NodeCaps) => NodeResult | Promise<NodeResult>;

  constructor(body: (open: Open, caps: NodeCaps) => NodeResult | Promise<NodeResult>) {
    super("work");
    this.memoryScope = new MemoryScope(["work"], { memories: ["notes"] });
    this.body = body;
  }

  override forward(_inputs: Record<string, unknown>, caps: NodeCaps): NodeResult | Promise<NodeResult> {
    return this.body(caps.openMemoryContext!, caps);
  }
}

/** One event the local opener or the node recorded. */
type Event = ["enter", MemoryScope, RecallContext] | [string];

/** An opener recording its entry and resolving to an empty context. */
function opener(events: Event[]): MemoryContextOpener {
  return async (scope, recall) => {
    events.push(["enter", scope, recall]);
    return new MemoryContext();
  };
}

/** Evaluate `node` with `open`, resolving to its outcome. */
async function evaluate(node: Node, open: MemoryContextOpener | null, options: EvalNodeOptions = {}): Promise<unknown> {
  const evaluated = await evalNode(node, null, new VariablePool(), { memoryContextOpener: open, ...options });
  return evaluated.outcome;
}

/** The event names, in order. */
function names(events: readonly Event[]): string[] {
  return events.map((event) => event[0]);
}

test("lazy single entry binds the recall context and closes after execution", async () => {
  const events: Event[] = [];
  const retained: Open[] = [];
  const node = new Consumer(async (open) => {
    expect(events).toEqual([]);
    retained.push(open);
    expect(await open("system", "prompt")).toEqual(new MemoryContext());
    events.push(["body"]);
    return new Output("done");
  });
  expect(await evaluate(node, opener(events))).toEqual(new Output("done"));
  expect(names(events)).toEqual(["enter", "body"]);
  const recall = events[0]![2] as RecallContext;
  expect([recall.baseSystemPrompt, recall.prompt, recall.consumerStaticPath]).toEqual(["system", "prompt", ["work"]]);
  const error = await rejected(retained[0]!("system", "prompt"));
  expect(error).not.toBeInstanceOf(CapabilityFault);
  expect((error as Error).message).toBe("memory context capability is closed");
});

test("an unused capability and no memory remain dormant", async () => {
  const forbidden: MemoryContextOpener = () => {
    throw new Error("an unused capability must not open a context");
  };
  expect(await evaluate(new Consumer(() => new Output(7n)), forbidden)).toEqual(new Output(7n));

  let ran = false;
  const opening = new Consumer(async (open) => {
    ran = true;
    return new Output(await open("system", "prompt"));
  });
  const error = await rejected(evaluate(opening, null));
  expect(error).toBeInstanceOf(Error);
  expect((error as Error).message).toBe("scoped node 'work' requires an injected memory context callback");
  expect(ran).toBe(false);

  const contexts: MemoryContext[] = [];
  const unscoped = new Consumer(async (open) => {
    contexts.push(await open("system", "prompt"));
    return new Output(7n);
  });
  unscoped.memoryScope = null;
  expect(await evaluate(unscoped, forbidden)).toEqual(new Output(7n));
  expect(contexts).toEqual([new MemoryContext()]);

  const ordinary = new FuncNode("ordinary", () => 7n);
  expect(ordinary.needsMemoryContext).toBe(false);
  expect((await evalNode(ordinary, null, new VariablePool(), { memoryContextOpener: forbidden })).outcome).toEqual(
    new Output(7n),
  );
});

test("repeated entry is a host fault", async () => {
  const events: Event[] = [];
  const node = new Consumer(async (open) => {
    await open("system", "prompt");
    await open("system", "prompt");
    return new Output(7n);
  });
  const error = await rejected(evaluate(node, opener(events)));
  expect(error).not.toBeInstanceOf(CapabilityFault);
  expect((error as Error).message).toBe("memory context capability was already entered");
  expect(names(events)).toEqual(["enter"]);
});

test.each([false, true])("opener host fault keeps identity in both execution shapes: streaming %p", async (streaming) => {
  const error = new Error("opening failed");
  const failing: MemoryContextOpener = () => {
    throw error;
  };
  const node = new Consumer(async (open, caps) => {
    await open("system", "prompt");
    if (streaming) caps.emitStream!(new StreamChunk({ nodeId: "work", key: "output", chunk: "never" }));
    return new Output(7n);
  });
  const chunks: StreamChunk[] = [];
  expect(await rejected(evaluate(node, failing, { emitStream: (chunk) => chunks.push(chunk) }))).toBe(error);
  expect(chunks).toEqual([]);
});

test("the scope is frozen before node execution", async () => {
  const events: Event[] = [];
  const node = new Consumer(async (open) => {
    node.memoryScope = new MemoryScope(["replacement"], { memories: ["other"] });
    await open("system", "prompt");
    return new Output(7n);
  });
  await evaluate(node, opener(events));
  expect((events[0]![1] as MemoryScope).staticPath).toEqual(["work"]);
  expect((events[0]![2] as RecallContext).consumerStaticPath).toEqual(["work"]);
});

test("malformed scope is a host fault before the node runs", async () => {
  const node = new Consumer(() => {
    throw new Error("the node must not run");
  });
  node.memoryScope = "invalid" as unknown as MemoryScope;
  const error = await rejected(evaluate(node, opener([])));
  expect(error).toBeInstanceOf(TypeError);
  expect((error as Error).message).toBe("node memory_scope must be an exact MemoryScope or null");
});

test("a node that returns without awaiting the opener doesn't settle before the open", async () => {
  const events: Event[] = [];
  let admit!: () => void;
  const admitted = new Promise<void>((resolve) => {
    admit = resolve;
  });
  const slow: MemoryContextOpener = async (scope, recall) => {
    events.push(["enter", scope, recall]);
    await admitted;
    events.push(["opened"]);
    return new MemoryContext();
  };
  const node = new Consumer((open) => {
    void open("system", "prompt");
    return new Output(7n);
  });
  let settled = false;
  const evaluation = evaluate(node, slow).then((outcome) => {
    settled = true;
    events.push(["settled"]);
    return outcome;
  });
  await new Promise((resolve) => setTimeout(resolve, 0));
  expect(settled).toBe(false);
  admit();
  expect(await evaluation).toEqual(new Output(7n));
  expect(names(events)).toEqual(["enter", "opened", "settled"]);
});

const unhandled: unknown[] = [];
const collect = (reason: unknown): void => {
  unhandled.push(reason);
};
beforeEach(() => {
  unhandled.length = 0;
  process.on("unhandledRejection", collect);
});
afterEach(() => {
  process.off("unhandledRejection", collect);
});

test("a dropped rejecting opener promise, or an open after close, rejects nothing unhandled", async () => {
  const error = new Error("opening failed");
  const failing: MemoryContextOpener = async () => {
    throw error;
  };
  const retained: Open[] = [];
  const dropsOpener = new Consumer((open) => {
    retained.push(open);
    void open("system", "prompt");
    return new Output(7n);
  });
  expect(await rejected(evaluate(dropsOpener, failing))).toBe(error);
  void retained[0]!("system", "prompt");
  await new Promise((resolve) => setTimeout(resolve, 0));
  expect(unhandled).toEqual([]);
});

test("invalid recall inputs, results and sections are recall failures", async () => {
  /** Evaluate a consumer opening with `prompt` through `open`, resolving to its failure. */
  const failure = async (open: MemoryContextOpener, prompt = "prompt"): Promise<EvaluationFailure> => {
    const node = new Consumer(async (begin) => new Output(await begin("system", prompt)));
    return (await evaluate(node, open)) as EvaluationFailure;
  };
  const events: Event[] = [];
  const invalidPrompt = await failure(opener(events), "\ud800");
  expect(invalidPrompt.failure.message).toBe("memory recall context is invalid");
  expect(invalidPrompt.failure.causeType).toBe("MemoryRecallError");
  expect(events).toEqual([]);

  /** A consumer with a bound input canonicalization refuses. */
  class SetInput extends Consumer {
    override bindReserved(): Record<string, unknown> {
      return { unordered: new Set([1n]) };
    }
  }
  const setInput = new SetInput(async (begin) => new Output(await begin("system", "prompt")));
  const invalidInputs = (await evaluate(setInput, opener(events))) as EvaluationFailure;
  expect(invalidInputs.failure.message).toBe("memory recall context is invalid");
  expect(events).toEqual([]);

  const notContext = await failure(async () => ({ sections: [] }) as unknown as MemoryContext);
  expect(notContext.failure.message).toBe("memory recall returned invalid context");

  const outside = await failure(async () => new MemoryContext({ sections: [new MemorySection("elsewhere", "notes")] }));
  expect(outside.failure.message).toBe("memory recall returned invalid context");
  expect(outside.failure.causeType).toBe("MemoryRecallError");

  const empty = await failure(async () => new MemoryContext({ sections: [new MemorySection("notes", "")] }));
  expect(empty.failure.message).toBe("memory recall returned invalid context");
});
