/**
 * Source-neutral, execution-scoped access to optional conversation turns: a port of
 * `tests/engine/test_memory_context_capability.py`. The local opener resolves to a context whose
 * fake transcript records how it was closed; Python's context manager exit is `close(error)`.
 *
 * Ported: `test_lazy_single_entry_binds_detached_identity_and_closes_after_execution` (the closed
 * capability rejects instead of raising), `test_no_eager_turn_and_no_memory_remains_dormant`,
 * `test_postassertion_failure_closes_failed`, `test_repeated_entry_is_host_fault_and_closes_first_turn`,
 * `test_malformed_scope_is_host_fault_even_for_custom_run_override` and
 * `test_missing_scoped_opener_remains_setup_fault_for_custom_run`.
 *
 * Ported (adapted): `test_invalid_node_result_closes_failed_before_failure_conversion` (6 rows):
 * the cleanup note is in the `notes` of the error the turn closed with, which is a plain `Error`
 * carrying the failure's message, and ends the failure's traceback, as in Python's traceback text.
 * The `(host_stop=False, real_backend=False)` cell of
 * `test_callback_error_retains_conversation_cleanup_notes_without_throwing_into_generator`: a sink
 * throws inside the node, so there is no generator to close; the turn closes with the sink's error,
 * which carries the one cleanup note; the opener also receives the node's signal.
 * `test_opener_host_fault_keeps_identity_in_both_execution_shapes` and
 * `test_generator_and_body_errors_close_failed_without_masking_node_failure`: the streaming shape
 * streams through `caps.emitStream` (the node declares `needsLiveEvents`).
 * `test_stream_lifetime_includes_drain_and_callback_failure_cleanup`: a sink throws inside the
 * node, so there is no generator to close; the node's `finally` runs before the turn closes with
 * the sink's error. `test_scope_and_invocation_are_frozen_before_node_execution`: the invocation is
 * frozen, so only the node's scope is replaced.
 *
 * TS-only: a node throwing `undefined` closes the turn with an `Error`; a `close` returning a
 * rejected promise rejects a successful evaluation and becomes a note on a failed one; a node that
 * starts `openMemoryContext` and returns without awaiting it still has its turn closed before
 * `evalNode` settles, also when its dropped code starts the open only after a steering `drain`,
 * while the evaluator waits; a node that drops a rejecting opener or steering `drain` promise, or
 * an open after the evaluation, causes no unhandled rejection; invalid recall inputs or request, an
 * opener result that isn't a `MemoryContext`, and sections outside the scope's memories or without
 * text are recall failures.
 *
 * Ported (adapted): `test_dispatch_binds_accepted_identity_before_worker_submission` (serial and
 * two workers): Python's guard that no worker thread reads `accepted_starts` is dropped, since TS
 * workers share the dispatcher's thread; the opener sees the accepted start's identity.
 *
 * The real-backend rows open `MemoryRuntime.openContext` over an in-memory `SessionMemory`.
 * Ported: `test_real_turn_rethrow_is_not_reported_as_secondary_cleanup_failure` (3 rows).
 * Changed: `test_real_backend_cleanup_note_survives_node_error_unwrapping`: a spy on the private
 * `Turn.publish` replaces the monkeypatched `_Turn._publish`, and the session's
 * `Conversation cleanup also failed` note is on the body's error and ends the traceback. The
 * `(host_stop=False, real_backend=True)` cell of
 * `test_callback_error_retains_conversation_cleanup_notes_without_throwing_into_generator`: the same
 * spy, and a sink throwing inside the node instead of a generator.
 * `test_fifo_turn_covers_stream_drain_and_shape_check_in_bounded_child`: no subprocess or threads;
 * two concurrent evaluations over one runtime, and the first pauses in its body after its chunk,
 * since the TS sink is synchronous; Bun's per-test timeout bounds it.
 *
 * Not ported: TS has no `_HostStop` (a stop is the node's aborted signal) and no `add_note` hook:
 * `test_callback_primary_stop_marks_signal_before_turn_unwind` and
 * `test_broken_cleanup_note_hook_does_not_replace_primary_host_error`.
 *
 * Ported (adapted), through the engine: `test_primary_stop_is_set_before_turn_cleanup_and_keeps_identity`
 * (both shapes): the primary stop is the host's signal, aborted by the node after it opened its
 * turn; the turn closes with the signal's reason while the node's own signal (the one the opener
 * received, not the run's) is aborted, and the cleanup note lands on the reason. The
 * `(host_stop=True, real_backend=False)` and `(host_stop=True, real_backend=True)` cells of
 * `test_callback_error_retains_conversation_cleanup_notes_without_throwing_into_generator` (serial
 * and two workers): a throwing `onStream` stops the run, so the turn closes with the callback's
 * error, which carries the one cleanup note, after the node's signal aborted with it; the real
 * backend uses the `failClose` spy.
 */

import { afterEach, beforeEach, expect, spyOn, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { StreamChunk } from "../../src/events.ts";
import {
  type ConversationCheckpoint,
  ConversationClosed,
  type ConversationCompletion,
  ConversationInvocation,
  type ConversationTurn,
  DeclSite,
  type JsonObject,
  MemoryContext,
  MemoryContextRequest,
  MemoryRecallError,
  MemoryScope,
  MemorySection,
  MemorySpec,
} from "../../src/memory/contracts.ts";
import { SessionMemory, Turn } from "../../src/memory/session.ts";
import {
  CapabilityFault,
  Grow,
  Node,
  type NodeCaps,
  NodeKind,
  type NodeResult,
  Output,
  type Steering,
} from "../../src/nodes/base.ts";
import { Assertion } from "../../src/nodes/contracts.ts";
import {
  EvaluationFailure,
  evalNode,
  type EvalNodeOptions,
  type MemoryContextOpener,
} from "../../src/runtime/eval_node.ts";
import { NodeStartedRecord, RunStoppedRecord } from "../../src/runtime/log.ts";
import { MemoryRuntime } from "../../src/runtime/memory_runtime.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { engine, FuncNode, graph, rejected } from "./run_state_fixtures.ts";

type Open = NonNullable<NodeCaps["openMemoryContext"]>;

/** A scoped memory consumer whose body is `body`. */
class Consumer extends Node {
  readonly kind = NodeKind.CODE;
  override readonly needsMemoryContext = true;
  override readonly needsLiveEvents = true;
  override readonly needsSteering = true;
  readonly body: (open: Open, caps: NodeCaps) => NodeResult | Promise<NodeResult>;

  constructor(body: (open: Open, caps: NodeCaps) => NodeResult | Promise<NodeResult>) {
    super("work");
    this.memoryScope = new MemoryScope(["work"], { memories: ["session"] });
    this.body = body;
  }

  override forward(_inputs: Record<string, unknown>, caps: NodeCaps): NodeResult | Promise<NodeResult> {
    return this.body(caps.openMemoryContext!, caps);
  }
}

/** One event the local opener or its turn recorded. */
type Event =
  | ["enter", MemoryScope, MemoryContextRequest]
  | ["completed"]
  | ["failed", unknown]
  | [string];

/** A fake turn recording its close; `close` throws `cleanup` when given. */
function fakeTurn(events: Event[], cleanup: unknown = undefined): ConversationTurn {
  return {
    close(error?: unknown): void {
      events.push(error === undefined ? ["completed"] : ["failed", error]);
      if (cleanup !== undefined) throw cleanup;
    },
  } as unknown as ConversationTurn;
}

/** An opener recording its entry and resolving to a context holding `turn`. */
function opener(events: Event[], turn: ConversationTurn = fakeTurn(events)): MemoryContextOpener {
  return async (scope, request) => {
    events.push(["enter", scope, request]);
    return new MemoryContext({ transcript: turn });
  };
}

/** Evaluate `node` with `open` and the invocation `("run", 3)`, resolving to its outcome. */
async function evaluate(
  node: Node,
  open: MemoryContextOpener | null,
  options: EvalNodeOptions = {},
): Promise<unknown> {
  const evaluated = await evalNode(node, null, new VariablePool(), {
    memoryContextOpener: open,
    memoryInvocation: new ConversationInvocation("run", 3),
    ...options,
  });
  return evaluated.outcome;
}

/** The event names, in order. */
function names(events: readonly Event[]): string[] {
  return events.map((event) => event[0]);
}

/** The error the turn last closed with. */
function closedWith(events: readonly Event[]): unknown {
  const last = events.at(-1)!;
  expect(last[0]).toBe("failed");
  return last[1];
}

test("lazy single entry binds detached identity and closes after execution", async () => {
  const events: Event[] = [];
  const retained: Open[] = [];
  const descriptor = { format: { version: 1n } };
  const turn = fakeTurn(events);
  const node = new Consumer(async (open) => {
    expect(events).toEqual([]);
    retained.push(open);
    expect((await open("system", "prompt", descriptor)).transcript).toBe(turn);
    descriptor.format.version = 2n;
    events.push(["body"]);
    return new Output("done");
  });
  expect(await evaluate(node, opener(events, turn))).toEqual(new Output("done"));
  expect(names(events)).toEqual(["enter", "body", "completed"]);
  const request = events[0]![2] as MemoryContextRequest;
  expect(request.invocation).toEqual(new ConversationInvocation("run", 3));
  expect(request.recall.consumerStaticPath).toEqual(["work"]);
  expect(request.compatibility).toEqual({ format: { version: 1n } });
  const error = await rejected(retained[0]!("system", "prompt", {}));
  expect(error).not.toBeInstanceOf(CapabilityFault);
  expect((error as Error).message).toBe("memory context capability is closed");
});

test("no eager turn and no memory remains dormant", async () => {
  const forbidden: MemoryContextOpener = () => {
    throw new Error("an unused capability must not open a turn");
  };
  expect(await evaluate(new Consumer(() => new Output(7n)), forbidden)).toEqual(new Output(7n));

  let ran = false;
  const opening = new Consumer(async (open) => {
    ran = true;
    return new Output(await open("system", "prompt", {}));
  });
  const error = await rejected(evaluate(opening, null));
  expect(error).toBeInstanceOf(Error);
  expect((error as Error).message).toBe("scoped node 'work' requires an injected memory context callback");
  expect(ran).toBe(false);

  const contexts: MemoryContext[] = [];
  const unscoped = new Consumer(async (open) => {
    contexts.push(await open("system", "prompt", {}));
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

const invalidResults: [string, unknown, boolean][] = [
  ["null", null, false],
  ["7", 7n, false],
  ["Grow", new Grow({}), false],
  ["null, cleanup fails", null, true],
  ["7, cleanup fails", 7n, true],
  ["Grow, cleanup fails", new Grow({}), true],
];

test.each(invalidResults)(
  "invalid node result closes failed before failure conversion: %s",
  async (_, result, fails) => {
    const events: Event[] = [];
    const node = new Consumer(async (open) => {
      await open("system", "prompt", {});
      return result as NodeResult;
    });
    const cleanup = fails ? new RangeError("result cleanup failed") : undefined;
    const outcome = await evaluate(node, opener(events, fakeTurn(events, cleanup)));
    expect(outcome).toBeInstanceOf(EvaluationFailure);
    const { failure } = outcome as EvaluationFailure;
    expect(failure.code).toBe("contract.node_result");
    expect(failure.causeType).toBe("Error");
    const error = closedWith(events) as Error & { notes?: string[] };
    expect(error.constructor).toBe(Error);
    expect(error.message).toBe(failure.message);
    expect(error.notes).toEqual(fails ? ["Memory context cleanup failed: RangeError: result cleanup failed"] : undefined);
    const note = "\nMemory context cleanup failed: RangeError: result cleanup failed";
    expect(failure.traceback!.endsWith(note)).toBe(fails);
  },
);

test("postassertion failure closes failed", async () => {
  const events: Event[] = [];
  const node = new Consumer(async (open) => {
    await open("system", "prompt", {});
    return new Output(7n);
  });
  node.postAsserts = [new Assertion("false", "must fail")];
  const outcome = (await evaluate(node, opener(events))) as EvaluationFailure;
  expect(outcome.failure.code).toBe("assertion.failed");
  expect(names(events).at(-1)).toBe("failed");
});

test("repeated entry is host fault and closes first turn", async () => {
  const events: Event[] = [];
  const node = new Consumer(async (open) => {
    await open("system", "prompt", {});
    await open("system", "prompt", {});
    return new Output(7n);
  });
  const error = await rejected(evaluate(node, opener(events)));
  expect(error).not.toBeInstanceOf(CapabilityFault);
  expect((error as Error).message).toBe("memory context capability was already entered");
  expect(names(events)).toEqual(["enter", "failed"]);
  expect(closedWith(events)).toBe(error);
});

test.each([false, true])(
  "opener host fault keeps identity in both execution shapes: streaming %p",
  async (streaming) => {
    const error = new Error("opening failed");
    const failing: MemoryContextOpener = () => {
      throw error;
    };
    const node = new Consumer(async (open, caps) => {
      await open("system", "prompt", {});
      if (streaming) caps.emitStream!(new StreamChunk({ nodeId: "work", key: "output", chunk: "never" }));
      return new Output(7n);
    });
    const chunks: StreamChunk[] = [];
    expect(await rejected(evaluate(node, failing, { emitStream: (chunk) => chunks.push(chunk) }))).toBe(error);
    expect(chunks).toEqual([]);
  },
);

test("stream lifetime includes the node's cleanup and callback failure cleanup", async () => {
  const events: Event[] = [];
  const error = new Error("stream callback");
  const node = new Consumer(async (open, caps) => {
    await open("system", "prompt", {});
    try {
      caps.emitStream!(new StreamChunk({ nodeId: "work", key: "output", chunk: "one" }));
      throw new Error("callback should abort the body");
    } finally {
      events.push(["node finally"]);
    }
  });
  const emit = (): void => {
    expect(names(events)).toEqual(["enter"]);
    throw error;
  };
  expect(await rejected(evaluate(node, opener(events), { emitStream: emit }))).toBe(error);
  expect(names(events)).toEqual(["enter", "node finally", "failed"]);
  expect(closedWith(events)).toBe(error);
});

test("callback error retains conversation cleanup notes", async () => {
  const events: Event[] = [];
  const error = new Error("callback failure");
  const signals: AbortSignal[] = [];
  const turn = fakeTurn(events, new RangeError("conversation close failed"));
  const open: MemoryContextOpener = async (scope, request, { signal }) => {
    signals.push(signal);
    return opener(events, turn)(scope, request, { signal });
  };
  const node = new Consumer(async (begin, caps) => {
    await begin("system", "prompt", {});
    caps.emitStream!(new StreamChunk({ nodeId: "work", key: "output", chunk: "one" }));
    return new Output(7n);
  });
  const controller = new AbortController();
  const emit = (): void => {
    throw error;
  };
  expect(await rejected(evaluate(node, open, { emitStream: emit, signal: controller.signal }))).toBe(error);
  expect(signals.length).toBe(1);
  expect(signals[0]).toBe(controller.signal);
  expect(closedWith(events)).toBe(error);
  expect((error as Error & { notes?: string[] }).notes).toEqual([
    "Memory context cleanup failed: RangeError: conversation close failed",
  ]);
});

test("an open the node's dropped code starts while the evaluator waits still closes its turn", async () => {
  const events: Event[] = [];
  let release!: (messages: readonly string[]) => void;
  const drained = new Promise<readonly string[]>((resolve) => {
    release = resolve;
  });
  const steering: Steering = { interruptSignal: () => new AbortController().signal, drain: () => drained };
  const slow: MemoryContextOpener = async (scope, request) => {
    events.push(["enter", scope, request]);
    for (let turn = 0; turn < 10; turn++) await null;
    return new MemoryContext({ transcript: fakeTurn(events) });
  };
  const node = new Consumer((open, caps) => {
    void (async () => {
      await caps.steering!.drain("work");
      await open("system", "prompt", {});
    })().catch(() => {});
    return new Output(7n);
  });
  const evaluation = evaluate(node, slow, { steering }).then((outcome) => {
    events.push(["settled"]);
    return outcome;
  });
  release([]);
  expect(await evaluation).toEqual(new Output(7n));
  expect(names(events)).toEqual(["enter", "completed", "settled"]);
});

test("scope and invocation are frozen before node execution", async () => {
  const events: Event[] = [];
  const invocation = new ConversationInvocation("original", 7);
  const node = new Consumer(async (open) => {
    node.memoryScope = new MemoryScope(["replacement"], { memories: ["other"] });
    await open("system", "prompt", {});
    return new Output(7n);
  });
  await evalNode(node, null, new VariablePool(), { memoryContextOpener: opener(events), memoryInvocation: invocation });
  expect(Object.isFrozen(invocation)).toBe(true);
  expect((events[0]![1] as MemoryScope).staticPath).toEqual(["work"]);
  expect((events[0]![2] as MemoryContextRequest).invocation).toEqual(new ConversationInvocation("original", 7));
});

/** A consumer whose custom `run` must not be reached. */
class CustomRun extends Consumer {
  constructor() {
    super(() => new Output(null));
  }

  override async run(): Promise<NodeResult> {
    throw new Error("the custom run must not be reached");
  }
}

test("malformed scope is host fault even for custom run override", async () => {
  const node = new CustomRun();
  node.memoryScope = "invalid" as unknown as MemoryScope;
  const error = await rejected(evaluate(node, opener([])));
  expect(error).toBeInstanceOf(TypeError);
  expect((error as Error).message).toBe("node memory_scope must be an exact MemoryScope or null");
});

test.each([false, true])(
  "generator and body errors close failed without masking node failure: streaming %p",
  async (streaming) => {
    const events: Event[] = [];
    const node = new Consumer(async (open, caps) => {
      await open("system", "prompt", {});
      if (streaming) caps.emitStream!(new StreamChunk({ nodeId: "work", key: "output", chunk: "one" }));
      throw new RangeError("body rejected");
    });
    const outcome = (await evaluate(node, opener(events), { emitStream: () => {} })) as EvaluationFailure;
    expect(outcome.failure.code).toBe("node.execute");
    expect(outcome.failure.message).toBe("body rejected");
    expect(names(events).at(-1)).toBe("failed");
  },
);

test("missing scoped opener remains setup fault for custom run", async () => {
  const error = await rejected(evaluate(new CustomRun(), null));
  expect(error).toBeInstanceOf(Error);
  expect((error as Error).message).toBe("scoped node 'work' requires an injected memory context callback");
});

test("a node throwing undefined closes the turn with an Error", async () => {
  const events: Event[] = [];
  const node = new Consumer(async (open) => {
    await open("system", "prompt", {});
    throw undefined;
  });
  const outcome = (await evaluate(node, opener(events))) as EvaluationFailure;
  expect(outcome.failure.code).toBe("node.execute");
  expect(closedWith(events)).toBeInstanceOf(Error);
});

test("a close returning a rejected promise rejects a success and is a note on a failure", async () => {
  const cleanup = new Error("close storage failed");
  const rejecting = (events: Event[]): ConversationTurn =>
    ({
      async close(error?: unknown): Promise<void> {
        events.push(error === undefined ? ["completed"] : ["failed", error]);
        throw cleanup;
      },
    }) as unknown as ConversationTurn;

  const succeeded: Event[] = [];
  const success = new Consumer(async (open) => {
    await open("system", "prompt", {});
    return new Output(7n);
  });
  expect(await rejected(evaluate(success, opener(succeeded, rejecting(succeeded))))).toBe(cleanup);
  expect(names(succeeded)).toEqual(["enter", "completed"]);

  const failedEvents: Event[] = [];
  const bodyError = new RangeError("body rejected");
  const failing = new Consumer(async (open) => {
    await open("system", "prompt", {});
    throw bodyError;
  });
  const outcome = (await evaluate(failing, opener(failedEvents, rejecting(failedEvents)))) as EvaluationFailure;
  expect(outcome.failure.message).toBe("body rejected");
  expect(closedWith(failedEvents)).toBe(bodyError);
  expect((bodyError as Error & { notes?: string[] }).notes).toEqual([
    "Memory context cleanup failed: Error: close storage failed",
  ]);
});

test("a node that returns without awaiting the opener still has its turn closed before evalNode settles", async () => {
  const events: Event[] = [];
  let admit!: () => void;
  const admitted = new Promise<void>((resolve) => {
    admit = resolve;
  });
  const slow: MemoryContextOpener = async (scope, request) => {
    events.push(["enter", scope, request]);
    await admitted;
    return new MemoryContext({ transcript: fakeTurn(events) });
  };
  const node = new Consumer((open) => {
    void open("system", "prompt", {});
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
  expect(names(events)).toEqual(["enter", "completed", "settled"]);
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

test("a dropped rejecting opener or drain promise, or an open after close, rejects nothing unhandled", async () => {
  const error = new Error("opening failed");
  const failing: MemoryContextOpener = async () => {
    throw error;
  };
  const retained: Open[] = [];
  const dropsOpener = new Consumer((open) => {
    retained.push(open);
    void open("system", "prompt", {});
    return new Output(7n);
  });
  expect(await rejected(evaluate(dropsOpener, failing))).toBe(error);
  void retained[0]!("system", "prompt", {});

  const drainError = new Error("inbox failed");
  const steering: Steering = {
    interruptSignal: () => new AbortController().signal,
    drain: () => Promise.reject(drainError),
  };
  const dropsDrain = new Consumer((_open, caps) => {
    void caps.steering!.drain("work");
    return new Output(7n);
  });
  expect(await rejected(evaluate(dropsDrain, opener([]), { steering }))).toBe(drainError);
  await new Promise((resolve) => setTimeout(resolve, 0));
  expect(unhandled).toEqual([]);
});

test("invalid recall inputs, a request, results and sections are recall failures", async () => {
  /** Evaluate a consumer opening with `compatibility` through `open`, resolving to its failure. */
  const failure = async (open: MemoryContextOpener, compatibility: unknown = {}): Promise<EvaluationFailure> => {
    const node = new Consumer(
      async (begin) => new Output(await begin("system", "prompt", compatibility as JsonObject)),
    );
    return (await evaluate(node, open)) as EvaluationFailure;
  };
  const events: Event[] = [];
  const invalidInput = await failure(opener(events), { bad: NaN });
  expect(invalidInput.failure.message).toBe("memory recall context is invalid");
  expect(invalidInput.failure.causeType).toBe("MemoryRecallError");
  expect(events).toEqual([]);

  /** A consumer with a bound input canonicalization refuses. */
  class SetInput extends Consumer {
    override bindReserved(): Record<string, unknown> {
      return { unordered: new Set([1n]) };
    }
  }
  const setInput = new SetInput(async (begin) => new Output(await begin("system", "prompt", {})));
  const invalidInputs = (await evaluate(setInput, opener(events))) as EvaluationFailure;
  expect(invalidInputs.failure.message).toBe("memory recall context is invalid");
  expect(events).toEqual([]);

  const notContext = await failure(async () => ({ sections: [], transcript: null }) as unknown as MemoryContext);
  expect(notContext.failure.message).toBe("memory recall returned invalid context");

  const sections = [new MemorySection("elsewhere", "notes")];
  const outside = await failure(async () => new MemoryContext({ sections, transcript: fakeTurn(events) }));
  expect(outside.failure.message).toBe("memory recall returned invalid context");
  expect(closedWith(events)).toBeInstanceOf(MemoryRecallError);

  const empty = await failure(async () => new MemoryContext({ sections: [new MemorySection("session", "")] }));
  expect(empty.failure.message).toBe("memory recall returned invalid context");
});

test.each([0, 2])("dispatch binds the accepted identity before worker submission (numWorkers %p)", async (numWorkers) => {
  const events: Event[] = [];
  const node = new Consumer(async (open) => {
    await open("system", "prompt", {});
    return new Output(7n);
  });
  const built = engine(graph([node], [
    [START_ID, "work"],
    ["work", END_ID],
  ]), { numWorkers, memoryContextOpener: opener(events) });

  await built.run();

  const start = built.log
    .records()
    .find((envelope) => envelope.record instanceof NodeStartedRecord && envelope.record.nodeId === "work")!;
  expect((events[0]![2] as MemoryContextRequest).invocation).toEqual(new ConversationInvocation(start.runId, start.seq));
  expect(events.at(-1)![0]).toBe("completed");
  expect("conversationState" in built.state).toBe(false);
});

/** Open a runtime whose one AGENT-owned key `session` is backed by `memory`. */
function openSessionRuntime(memory: SessionMemory): Promise<MemoryRuntime> {
  const spec = new MemorySpec("session", "session", {}, [new DeclSite([], ["root"], "work")], true);
  return MemoryRuntime.open({
    memories: [spec],
    origins: new Map([["session", null]]),
    memory: new Map([["session", memory]]),
  });
}

/** The runtime's `openContext` as the evaluator's opener. */
function contextOpener(runtime: MemoryRuntime): MemoryContextOpener {
  return (scope, request, options) => runtime.openContext(scope, request, options);
}

/** Make every real turn's `ConversationClosed` checkpoint throw `cleanup`; restore the spy after. */
function failClose(cleanup: Error): { mockRestore(): void } {
  const publish = Turn.prototype["publish"];
  return spyOn(Turn.prototype as never, "publish").mockImplementation(function (
    this: Turn,
    checkpoint: ConversationCheckpoint,
  ): void {
    if (checkpoint instanceof ConversationClosed) throw cleanup;
    publish.call(this, checkpoint);
  } as never);
}

test.each(["body", "postassertion", "result"])(
  "real turn rethrow is not reported as secondary cleanup failure: %s",
  async (stage) => {
    const error: Error & { notes?: string[] } = new RangeError("body rejected");
    const node = new Consumer(async (open) => {
      await open("system", "prompt", {});
      if (stage === "body") throw error;
      return stage === "result" ? (null as unknown as NodeResult) : new Output(7n);
    });
    if (stage === "postassertion") node.postAsserts = [new Assertion("false", "must fail")];
    const memory = new SessionMemory();
    const runtime = await openSessionRuntime(memory);
    try {
      const outcome = await evaluate(node, contextOpener(runtime));
      expect(outcome).toBeInstanceOf(EvaluationFailure);
      expect((outcome as EvaluationFailure).failure.traceback).not.toContain("Memory context cleanup");
      expect(error.notes).toBeUndefined();
      const opened = memory.openRun();
      try {
        const conversation = opened.snapshot.conversation as JsonObject;
        expect((conversation.closed_invocations as JsonObject[]).at(-1)!.status).toBe("failed");
      } finally {
        opened.close();
      }
    } finally {
      await runtime.close();
    }
  },
);

test("real backend cleanup note survives node error unwrapping", async () => {
  const error: Error & { notes?: string[] } = new RangeError("body rejected");
  const node = new Consumer(async (open) => {
    await open("system", "prompt", {});
    throw error;
  });
  const memory = new SessionMemory();
  const runtime = await openSessionRuntime(memory);
  const spy = failClose(new Error("close storage failed"));
  try {
    const outcome = (await evaluate(node, contextOpener(runtime))) as EvaluationFailure;
    expect(outcome.failure.code).toBe("node.execute");
    const note = "Conversation cleanup also failed: Error: close storage failed";
    expect(outcome.failure.traceback!.endsWith(note)).toBe(true);
    expect(error.notes).toEqual([note]);
  } finally {
    spy.mockRestore();
    await runtime.close();
  }
});

test("callback error retains conversation cleanup notes: real backend", async () => {
  const error: Error & { notes?: string[] } = new Error("callback failure");
  const node = new Consumer(async (begin, caps) => {
    await begin("system", "prompt", {});
    caps.emitStream!(new StreamChunk({ nodeId: "work", key: "output", chunk: "one" }));
    return new Output(7n);
  });
  const emit = (): void => {
    throw error;
  };
  const memory = new SessionMemory();
  const runtime = await openSessionRuntime(memory);
  const spy = failClose(new RangeError("conversation close failed"));
  try {
    expect(await rejected(evaluate(node, contextOpener(runtime), { emitStream: emit }))).toBe(error);
    expect((error.notes ?? []).filter((note) => note.includes("conversation close failed")).length).toBe(1);
  } finally {
    spy.mockRestore();
    await runtime.close();
  }
});

test("fifo turn covers stream drain and shape check", async () => {
  const memory = new SessionMemory();
  const runtime = await openSessionRuntime(memory);
  let pause!: () => void;
  const paused = new Promise<void>((resolve) => {
    pause = resolve;
  });
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  let secondEntered = false;
  const prior: ConversationCompletion[] = [];
  const first = new Consumer(async (open, caps) => {
    await open("system", "prompt", {});
    caps.emitStream!(new StreamChunk({ nodeId: "work", key: "output", chunk: "one" }));
    pause();
    await gate;
    return null as unknown as NodeResult; // The shape check must happen before the lease is released.
  });
  const second = new Consumer(async (open) => {
    const context = await open("system", "prompt", {});
    prior.push(...context.transcript!.view.closedInvocations);
    secondEntered = true;
    return new Output(7n);
  });
  const run = (node: Node, id: string) =>
    evalNode(node, null, new VariablePool(), {
      memoryContextOpener: contextOpener(runtime),
      memoryInvocation: new ConversationInvocation(id, 1),
      emitStream: () => {},
    });
  try {
    const a = run(first, "a");
    let b: ReturnType<typeof run>;
    try {
      await paused;
      b = run(second, "b");
      for (let turn = 0; turn < 1000 && memory["queue"].length !== 1; turn++) await Bun.sleep(0);
      expect(memory["queue"].length).toBe(1);
      expect(secondEntered).toBe(false);
    } finally {
      release();
    }
    expect(((await a).outcome as EvaluationFailure).failure.code).toBe("contract.node_result");
    expect((await b).outcome).toEqual(new Output(7n));
    expect(prior.length).toBe(1);
    expect(prior[0]!.invocation).toEqual(new ConversationInvocation("a", 1));
    expect(prior[0]!.status).toBe("failed");
    memory.reset();
  } finally {
    await runtime.close();
  }
});

/**
 * Return an opener whose turn records how it closed and whether the signal the opener received
 * had aborted by then, then throws `cleanup`; `signals` receives that signal.
 */
function stopAwareOpener(events: Event[], stopped: boolean[], signals: AbortSignal[], cleanup: unknown): MemoryContextOpener {
  return async (scope, request, { signal }) => {
    signals.push(signal);
    events.push(["enter", scope, request]);
    const turn = {
      close(error?: unknown): void {
        events.push(error === undefined ? ["completed"] : ["failed", error]);
        stopped.push(signal.aborted);
        throw cleanup;
      },
    } as unknown as ConversationTurn;
    return new MemoryContext({ transcript: turn });
  };
}

/** Return an engine running `node` alone between START and END. */
function single(node: Node, options: Parameters<typeof engine>[1] = {}) {
  return engine(graph([node], [
    [START_ID, "work"],
    ["work", END_ID],
  ]), options);
}

test.each([0, 2])("primary stop is set before turn cleanup and keeps identity (numWorkers %p)", async (numWorkers) => {
  for (const streaming of [false, true]) {
    const events: Event[] = [];
    const stopped: boolean[] = [];
    const signals: AbortSignal[] = [];
    const host = new AbortController();
    const reason = new Error("requested");
    const node = new Consumer(async (open, caps) => {
      await open("system", "prompt", {});
      if (streaming) caps.emitStream!(new StreamChunk({ nodeId: "work", key: "output", chunk: "before" }));
      host.abort(reason);
      throw new Error("interrupted");
    });
    const built = single(node, {
      numWorkers,
      signal: host.signal,
      onStream: () => {},
      memoryContextOpener: stopAwareOpener(events, stopped, signals, new Error("cleanup")),
    });

    expect(await rejected(built.run())).toBe(reason);
    expect(closedWith(events)).toBe(reason);
    expect(stopped).toEqual([true]);
    expect(signals[0]).not.toBe(built.signal);
    expect(signals[0]!.reason).toBe(reason);
    expect((reason as Error & { notes?: string[] }).notes).toEqual(["Memory context cleanup failed: Error: cleanup"]);
  }
});

test.each([0, 2])("a callback stop retains conversation cleanup notes (numWorkers %p)", async (numWorkers) => {
  const events: Event[] = [];
  const stopped: boolean[] = [];
  const signals: AbortSignal[] = [];
  const error = new Error("callback stop");
  const node = new Consumer(async (open, caps) => {
    await open("system", "prompt", {});
    caps.emitStream!(new StreamChunk({ nodeId: "work", key: "output", chunk: "one" }));
    return new Output(7n);
  });
  const built = single(node, {
    numWorkers,
    onStream: () => {
      throw error;
    },
    memoryContextOpener: stopAwareOpener(events, stopped, signals, new RangeError("conversation close failed")),
  });

  expect(await rejected(built.run())).toBe(error);
  expect(closedWith(events)).toBe(error);
  expect(stopped).toEqual([true]);
  expect(signals[0]).not.toBe(built.signal);
  expect(signals[0]!.reason).toBe(error);
  expect((error as Error & { notes?: string[] }).notes).toEqual([
    "Memory context cleanup failed: RangeError: conversation close failed",
  ]);
  expect(built.log.records().at(-1)!.record).toEqual(new RunStoppedRecord("callback", "callback stop", ["work"]));
});

test.each([0, 2])("a callback stop retains conversation cleanup notes: real backend (numWorkers %p)", async (numWorkers) => {
  const error: Error & { notes?: string[] } = new Error("callback stop");
  const node = new Consumer(async (open, caps) => {
    await open("system", "prompt", {});
    caps.emitStream!(new StreamChunk({ nodeId: "work", key: "output", chunk: "one" }));
    return new Output(7n);
  });
  const memory = new SessionMemory();
  const runtime = await openSessionRuntime(memory);
  const spy = failClose(new RangeError("conversation close failed"));
  try {
    const built = single(node, {
      numWorkers,
      onStream: () => {
        throw error;
      },
      memoryContextOpener: contextOpener(runtime),
    });

    expect(await rejected(built.run())).toBe(error);
    expect((error.notes ?? []).filter((note) => note.includes("conversation close failed")).length).toBe(1);
    expect(built.log.records().at(-1)!.record).toEqual(new RunStoppedRecord("callback", "callback stop", ["work"]));
  } finally {
    spy.mockRestore();
    await runtime.close();
  }
});
