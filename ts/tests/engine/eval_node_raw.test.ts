/**
 * Raw node evaluation for the append-only dispatcher: a port of
 * `tests/engine/test_eval_node_raw.py`.
 *
 * Ported: `test_raw_evaluation_returns_closed_success_route_and_grow_values`,
 * `test_raw_node_raise_and_invalid_grow_are_failures_not_observers` (cause type `Error` for
 * Python's `RuntimeError`), `test_raw_streaming_uses_only_the_injected_transient_callback`,
 * `test_raw_stream_callback_failure_propagates_unchanged` and
 * `test_render_capture_is_opt_in_single_write_and_detached_until_close`.
 *
 * Ported (adapted): `test_public_evaluation_always_returns_frozen_slotted_wrapper` and
 * `test_raw_failure_is_frozen_slotted_and_preserves_diagnostics`: the field names are the own
 * keys, frozen means `Object.isFrozen` and a throwing write, and the traceback is the binder
 * error's stack. `test_stream_body_exception_and_invalid_return_use_distinct_stages`: a node
 * streams through `caps.emitStream` (it declares `needsLiveEvents`), so the stream body is the
 * node's `forward`; the body's `ValueError` is a `RangeError`, and the invalid return gets the
 * ordinary non-NodeResult message (TS has no generator to name).
 *
 * Changed: `test_invalid_render_capture_is_a_host_fault` publishes NaN for Python's infinity,
 * which is a value here. `test_render_capture_preserves_original_host_exception_and_closes`
 * monkeypatches `canonical_seed`; in its place a payload with a throwing getter rejects `evalNode`
 * with the getter's error, and the capture closes.
 * `test_scoped_render_capture_requires_recorded_model_map_entry` uses a local scoped node that
 * publishes its render before its model call, for Python's AGENT.
 * The shared `drive` evaluates the node in a one-node `CompiledFlow` wired from a
 * `FuncNode`'s `reads`, as Python's stub flow derives it.
 * `test_raw_stream_callback_failure_closes_the_node_generator_before_returning` and
 * `test_raw_stream_cleanup_failure_does_not_replace_callback_failure`: a sink throws inside the
 * node, so there is no generator to close. In their place: the node's (async) `finally` has run
 * before `evalNode` settles, and a node that throws its own error from `finally`, or catches the
 * fault and returns, still rejects `evalNode` with the sink's error: the first capability fault
 * wins over whatever the node then did, a stop rejection included.
 *
 * Not ported: the `__slots__` (no `__dict__`) checks; a TS class has no slots.
 * `test_raw_host_stop_propagates_from_node_and_preset_event` and
 * `test_raw_generator_host_stop_propagates_by_identity`: TS has no `_HostStop`, since a stop is the
 * node's aborted signal. In their place: an aborted signal rejects with its reason before binding;
 * a node that rejects after its signal aborted (with its own error or the reason, from `forward`
 * or a custom `run`) rejects `evalNode` with the reason, not a failure; a node that returns after
 * the abort still resolves.
 *
 * TS-only: the `EvaluatedOutcome` constructor checks, `requireSteering`, the steering
 * capability's checks on what the host inbox returns, a steering fault the node catches (thrown or
 * rejected) or a `drain` promise it drops still rejecting `evalNode`, the first of two faults
 * the node catches winning (and the only one `onFault` sees, when it is raised), a render capture
 * fault the node catches still rejecting `evalNode`,
 * and a render write by the node's dropped code, landing while the evaluator waits on a steering
 * `drain`, rejecting it.
 *
 * Later (T5-11a): `test_scoped_render_capture_requires_recorded_model_map_entry` through AGENT.
 */

import { describe, expect, test } from "bun:test";

import { Failure, OriginRef } from "../../src/diagnostics.ts";
import { StreamChunk } from "../../src/events.ts";
import {
  CapabilityFault,
  Grow,
  Node,
  type NodeCaps,
  NodeKind,
  type NodeResult,
  Output,
  Route,
  type Steering,
} from "../../src/nodes/base.ts";
import { BindingCollector } from "../../src/nodes/binding.ts";
import { ExecutionScope, Parameter } from "../../src/nodes/contracts.ts";
import { BackwardCapture } from "../../src/runtime/contracts.ts";
import { EvaluationEvidence } from "../../src/memory/contracts.ts";
import {
  EvaluatedOutcome,
  EvaluationFailure,
  evalNode,
  requireSteering,
  SteeringCapability,
} from "../../src/runtime/eval_node.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { drive, FailNode, FuncNode, nodeFlow, rejected, StreamNode } from "./run_state_fixtures.ts";

/** Return one configured result from its body. */
class ResultNode extends Node {
  readonly kind: NodeKind = NodeKind.CODE;
  readonly result: unknown;

  constructor(nodeId: string, result: unknown) {
    super(nodeId);
    this.result = result;
  }

  override forward(): NodeResult {
    return this.result as NodeResult;
  }
}

/** A minimal spawner for the raw `Grow` result. */
class Spawner extends ResultNode {
  override readonly isSpawner = true;
}

/** Return the value `run` throws, or fail when it returns. */
function thrown(run: () => unknown): unknown {
  try {
    run();
  } catch (error) {
    return error;
  }
  throw new Error("expected a throw");
}

/** Return a `node.execute` failure for node `work`. */
function failure(): Failure {
  return new Failure({ code: "node.execute", message: "boom", nodeId: "work" });
}

test("public evaluation always returns frozen slotted wrapper", async () => {
  const node = new FuncNode("work", () => "done");
  const evaluated = await evalNode(node, nodeFlow(node), new VariablePool());

  expect(evaluated).toBeInstanceOf(EvaluatedOutcome);
  expect(evaluated).toEqual(new EvaluatedOutcome(new Output("done"), null));
  expect(Object.keys(evaluated)).toEqual(["outcome", "evidence", "bindings", "backwardCapture"]);
  expect(Object.isFrozen(evaluated)).toBe(true);
  expect(() => {
    (evaluated as { evidence: unknown }).evidence = null;
  }).toThrow(TypeError);
});

test("raw evaluation returns closed success route and grow values", async () => {
  expect(await drive(new FuncNode("work", () => "done"))).toEqual(new Output("done"));
  expect(await drive(new ResultNode("branch", new Route("chosen")))).toEqual(new Route("chosen"));
  expect(await drive(new Spawner("spawn", new Grow({ value: 1n })))).toEqual(new Grow({ value: 1n }));
});

describe("render capture", () => {
  type CaptureRender = NonNullable<NodeCaps["captureRender"]>;

  /** A node opting into render capture: its body publishes `publish`'s records. */
  class Capturing extends Node {
    readonly kind = NodeKind.CODE;
    override readonly needsRenderCapture = true;
    readonly publish: (captureRender: CaptureRender) => void;
    readonly retained: CaptureRender[] = [];

    constructor(publish: (captureRender: CaptureRender) => void) {
      super("custom");
      this.publish = publish;
    }

    override captureBackwardContext(): Record<string, unknown> {
      return { static: "recorded" };
    }

    override forward(_inputs: Record<string, unknown>, caps: NodeCaps): NodeResult {
      if (caps.captureRender !== undefined) {
        this.retained.push(caps.captureRender);
        this.publish(caps.captureRender);
      }
      return new Output("done");
    }
  }

  /** Return the plain error a closed capture throws. */
  function closedError(captureRender: CaptureRender): unknown {
    const error = thrown(() => captureRender({}));
    expect(error).not.toBeInstanceOf(CapabilityFault);
    expect((error as Error).message).toMatch(/closed/);
    return error;
  }

  test("render capture is opt-in, single-write and detached until close", async () => {
    const supplied = { local: { items: [1n] as unknown[] } };
    const node = new Capturing((captureRender) => {
      captureRender(supplied);
      supplied.local.items.push(2n);
    });
    expect((await evalNode(node, null, new VariablePool())).backwardCapture).toBeNull();
    expect(node.retained).toEqual([]);
    const evaluated = await evalNode(node, null, new VariablePool(), { captureBindings: true });
    expect(evaluated.outcome).toEqual(new Output("done"));
    expect(evaluated.backwardCapture!.nodeData).toEqual({
      static: "recorded",
      local: { items: [1n] },
      effective_llm_config: null,
    });
    closedError(node.retained[0]!);
    expect(evaluated.backwardCapture!.nodeData["local"]).toEqual({ items: [1n] });
  });

  const invalidPublications: [string, RegExp, ErrorConstructor][] = [
    ["duplicate", /already recorded/, Error],
    ["reserved", /recorded runtime authority/, RangeError],
    ["static", /replace static/, RangeError],
    ["NaN", /NaN/, RangeError],
  ];

  test.each(invalidPublications)("invalid render capture is a host fault: %s", async (publication, match, kind) => {
    const node = new Capturing((captureRender) => {
      if (publication === "duplicate") {
        captureRender({ first: 1n });
        captureRender({ second: 2n });
      } else if (publication === "reserved") {
        captureRender({ effective_llm_config: { model: "invented" } });
      } else if (publication === "static") {
        captureRender({ static: "changed" });
      } else {
        captureRender({ variables: { nested: [NaN] } });
      }
    });
    const error = await rejected(evalNode(node, null, new VariablePool(), { captureBindings: true }));
    expect(error).toBeInstanceOf(kind);
    expect(error).not.toBeInstanceOf(CapabilityFault);
    expect((error as Error).message).toMatch(match);
    closedError(node.retained[0]!);
  });

  test("a payload whose canonicalization throws keeps the host's error and the capture closes", async () => {
    const expected = new Error("capture allocation failed");
    const payload = {};
    Object.defineProperty(payload, "dynamic", {
      enumerable: true,
      get() {
        throw expected;
      },
    });
    const node = new Capturing((captureRender) => captureRender(payload));
    expect(await rejected(evalNode(node, null, new VariablePool(), { captureBindings: true }))).toBe(expected);
    closedError(node.retained[0]!);
  });

  test("a node catching a render capture fault still rejects evalNode with it", async () => {
    const node = new Capturing((captureRender) => {
      try {
        captureRender({ static: "changed" });
      } catch {}
    });
    const error = await rejected(evalNode(node, null, new VariablePool(), { captureBindings: true }));
    expect(error).toBeInstanceOf(RangeError);
    expect(error).not.toBeInstanceOf(CapabilityFault);
    expect((error as Error).message).toMatch(/replace static/);
    closedError(node.retained[0]!);
  });

  test("a dropped write that lands while the evaluator waits decides the outcome", async () => {
    let release!: (messages: readonly string[]) => void;
    const drained = new Promise<readonly string[]>((resolve) => {
      release = resolve;
    });
    const steering: Steering = { interruptSignal: () => new AbortController().signal, drain: () => drained };

    /** A capturing node whose dropped code writes once the test releases its steering drain. */
    class Steered extends Capturing {
      override readonly needsSteering = true;

      override forward(_inputs: Record<string, unknown>, caps: NodeCaps): NodeResult {
        void (async () => {
          await caps.steering!.drain("custom");
          caps.captureRender!({ static: "changed" });
        })().catch(() => {});
        return new Output("done");
      }
    }

    const options = { captureBindings: true, steering };
    const evaluation = rejected(evalNode(new Steered(() => {}), null, new VariablePool(), options));
    release([]);
    const error = await evaluation;
    expect(error).toBeInstanceOf(RangeError);
    expect((error as Error).message).toMatch(/replace static/);
  });

  test("scoped render capture requires the recorded model map entry", async () => {
    const calls: string[] = [];

    /** A scoped node publishing its render before its model call. */
    class Scoped extends Node {
      readonly kind = NodeKind.AGENT;
      override readonly needsRenderCapture = true;

      override forward(_inputs: Record<string, unknown>, caps: NodeCaps): NodeResult {
        caps.captureRender?.({ prompt: "hello" });
        calls.push("hello");
        return new Output("hello");
      }
    }

    const node = new Scoped("a/runtime/id");
    node.executionScope = new ExecutionScope(["authored", "agent"]);
    const error = await rejected(
      evalNode(node, null, new VariablePool(), { captureBindings: true, effectiveLlmByStaticPath: {} }),
    );
    expect(error).toBeInstanceOf(RangeError);
    expect((error as Error).message).toMatch(/missing static path 'authored\/agent'/);
    expect(calls).toEqual([]);
    const config = { provider: "openai", model: "recorded", openai: { temperature: 0.5 } };
    const evaluated = await evalNode(node, null, new VariablePool(), {
      captureBindings: true,
      effectiveLlmByStaticPath: { "authored/agent": config },
    });
    config.model = "later mutation";
    const recorded = evaluated.backwardCapture!.nodeData["effective_llm_config"] as Record<string, unknown>;
    expect(recorded["model"]).toBe("recorded");
    const ordinary = await evalNode(node, null, new VariablePool(), { effectiveLlmByStaticPath: {} });
    expect(ordinary.outcome).toEqual(evaluated.outcome);
    expect(ordinary.backwardCapture).toBeNull();
  });
});

test("raw failure is frozen slotted and preserves diagnostics", async () => {
  const node = new FuncNode("work", (inputs) => inputs);
  node.params = [new Parameter("required", { required: true })];

  const result = await drive(node);

  expect(result).toBeInstanceOf(EvaluationFailure);
  const retained = (result as EvaluationFailure).failure;
  expect(retained.nodeId).toBe("work");
  expect(retained.causeType).toBe("BindingError");
  expect(retained.traceback).toStartWith("BindingError: required input 'required' is unbound");
  expect(Object.keys(result)).toEqual(["failure"]);
  for (const alias of ["nodeId", "error", "errorType", "traceback"]) expect(alias in result).toBe(false);
  expect(Object.isFrozen(result)).toBe(true);
  expect(() => {
    (result as { failure: unknown }).failure = retained;
  }).toThrow(TypeError);
});

test("raw node raise and invalid grow are failures not observers", async () => {
  const raised = await drive(new FailNode("raise"));
  const invalidGrow = await drive(new ResultNode("leaf", new Grow({})));

  expect(raised).toBeInstanceOf(EvaluationFailure);
  const failure = (raised as EvaluationFailure).failure;
  expect(failure.code).toBe("node.execute");
  expect([failure.message, failure.causeType]).toEqual(["boom", "Error"]);
  expect(invalidGrow).toBeInstanceOf(EvaluationFailure);
  expect((invalidGrow as EvaluationFailure).failure.code).toBe("contract.node_result");
  expect((invalidGrow as EvaluationFailure).failure.message).toContain("not a spawner");
});

describe("a stop is the node's aborted signal", () => {
  /** Await `gate`, then settle with `settle`, recording that `forward` ran. */
  class GatedNode extends Node {
    readonly kind = NodeKind.CODE;
    readonly gate: Promise<void>;
    readonly settle: (caps: NodeCaps) => NodeResult;
    ran = false;

    constructor(nodeId: string, gate: Promise<void>, settle: (caps: NodeCaps) => NodeResult) {
      super(nodeId);
      this.gate = gate;
      this.settle = settle;
    }

    override async forward(_inputs: Record<string, unknown>, caps: NodeCaps): Promise<NodeResult> {
      this.ran = true;
      await this.gate;
      return this.settle(caps);
    }
  }

  test("an aborted signal rejects with its reason before binding", async () => {
    const controller = new AbortController();
    const reason = new Error("stopped");
    controller.abort(reason);
    const node = new GatedNode("never", Promise.resolve(), () => new Output("unexpected"));
    // Binding this parameter would fail, so a rejection proves binding never ran.
    node.params = [new Parameter("required", { required: true })];

    expect(await rejected(drive(node, { signal: controller.signal }))).toBe(reason);
    expect(node.ran).toBe(false);
  });

  /** The same node with a custom `run` that awaits the gate itself, outside the template. */
  class GatedRunNode extends GatedNode {
    override async run(_inputs: Readonly<Record<string, unknown>>, caps: NodeCaps = {}): Promise<NodeResult> {
      this.ran = true;
      await this.gate;
      return this.settle(caps);
    }
  }

  for (const [label, Gated, settle] of [
    [
      "its own error",
      GatedNode,
      () => {
        throw new Error("interrupted");
      },
    ],
    [
      "the reason",
      GatedNode,
      (caps: NodeCaps) => {
        throw caps.signal!.reason;
      },
    ],
    [
      "its own error from a custom run",
      GatedRunNode,
      () => {
        throw new Error("interrupted");
      },
    ],
  ] as const) {
    test(`a node rejecting after its signal aborted rejects with the reason (${label})`, async () => {
      const controller = new AbortController();
      const reason = new Error("stopped");
      const gate = Promise.withResolvers<void>();
      const evaluation = drive(new Gated("work", gate.promise, settle), { signal: controller.signal });

      controller.abort(reason);
      gate.resolve();

      expect(await rejected(evaluation)).toBe(reason);
    });
  }

  test("a node returning after its signal aborted still resolves", async () => {
    // The engine discards it; the evaluator only reports what the node did.
    const controller = new AbortController();
    const gate = Promise.withResolvers<void>();
    const evaluation = drive(new GatedNode("work", gate.promise, () => new Output("late")), {
      signal: controller.signal,
    });

    controller.abort(new Error("stopped"));
    gate.resolve();

    expect(await evaluation).toEqual(new Output("late"));
  });
});

describe("streaming", () => {
  /** A sink that throws `error` for every chunk. */
  function reject(error: unknown): (chunk: StreamChunk) => void {
    return () => {
      throw error;
    };
  }

  /** Stream `chunks` through `caps.emitStream`, then settle with `settle`. */
  class ScriptedStreamNode extends Node {
    readonly kind = NodeKind.CODE;
    override readonly needsLiveEvents = true;
    readonly chunks: readonly string[];
    readonly settle: () => unknown;

    constructor(nodeId: string, chunks: readonly string[], settle: () => unknown, origin: OriginRef | null = null) {
      super(nodeId, { origin });
      this.chunks = chunks;
      this.settle = settle;
    }

    override forward(_inputs: Record<string, unknown>, caps: NodeCaps): NodeResult {
      for (const chunk of this.chunks) caps.emitStream?.(new StreamChunk({ nodeId: this.id, key: "output", chunk }));
      return this.settle() as NodeResult;
    }
  }

  test("raw streaming uses only the injected transient callback", async () => {
    const chunks: StreamChunk[] = [];

    const result = await drive(new StreamNode("agent", ["a", "b"]), { emitStream: (chunk) => chunks.push(chunk) });

    expect(result).toEqual(new Output({ output: "ab" }));
    expect(chunks.map((chunk) => chunk.chunk)).toEqual(["a", "b"]);
    expect(chunks.every((chunk) => chunk instanceof StreamChunk)).toBe(true);
  });

  test("stream body exception and invalid return use distinct stages", async () => {
    // Streaming execution is node code; its terminal value is a result contract.
    const body = new ScriptedStreamNode(
      "body",
      ["first"],
      () => {
        throw new RangeError("stream body failed");
      },
      new OriginRef("node:body"),
    );
    const invalid = new ScriptedStreamNode(
      "invalid",
      ["before-invalid-return"],
      () => "invalid",
      new OriginRef("node:invalid"),
    );
    const chunks: StreamChunk[] = [];
    const invalidChunks: StreamChunk[] = [];

    const bodyFailure = await drive(body, { emitStream: (chunk) => chunks.push(chunk) });
    const resultFailure = await drive(invalid, { emitStream: (chunk) => invalidChunks.push(chunk) });

    expect(chunks.map((chunk) => chunk.chunk)).toEqual(["first"]);
    expect(bodyFailure).toBeInstanceOf(EvaluationFailure);
    const executed = (bodyFailure as EvaluationFailure).failure;
    expect(executed.code).toBe("node.execute");
    expect(executed.trace).toEqual([body.origin!]);
    expect(executed.causeType).toBe("RangeError");
    expect(executed.traceback).toContain("RangeError: stream body failed");
    expect(resultFailure).toBeInstanceOf(EvaluationFailure);
    expect(invalidChunks.map((chunk) => chunk.chunk)).toEqual(["before-invalid-return"]);
    const contract = (resultFailure as EvaluationFailure).failure;
    expect(contract.code).toBe("contract.node_result");
    expect(contract.trace).toEqual([invalid.origin!]);
    expect(contract.causeType).toBe("Error");
    expect(contract.message).toBe("node 'invalid' run() returned string, not a NodeResult");
  });

  test("raw stream callback failure propagates unchanged", async () => {
    const expected = new Error("callback failed");

    expect(await rejected(drive(new StreamNode("agent", ["a"]), { emitStream: reject(expected) }))).toBe(expected);
  });

  test("the node's finally has run before evalNode settles", async () => {
    const closed: string[] = [];
    const sent: string[] = [];

    class ResourceStreamNode extends Node {
      readonly kind = NodeKind.CODE;
      override readonly needsLiveEvents = true;

      override async forward(_inputs: Record<string, unknown>, caps: NodeCaps): Promise<NodeResult> {
        try {
          for (const chunk of ["first", "second"]) {
            sent.push(chunk);
            caps.emitStream!(new StreamChunk({ nodeId: this.id, key: "output", chunk }));
          }
          return new Output("done");
        } finally {
          // An asynchronous cleanup step, which evalNode must wait for.
          await Promise.resolve();
          closed.push(this.id);
        }
      }
    }
    const expected = new Error("callback failed");

    expect(await rejected(drive(new ResourceStreamNode("resource"), { emitStream: reject(expected) }))).toBe(expected);
    expect([sent, closed]).toEqual([["first"], ["resource"]]);
  });

  for (const [label, forward] of [
    [
      "throws its own error from finally",
      (emit: () => void): NodeResult => {
        try {
          emit();
          return new Output("unreachable");
        } finally {
          throw new RangeError("cleanup failed");
        }
      },
    ],
    [
      "catches the fault and returns",
      (emit: () => void): NodeResult => {
        try {
          emit();
        } catch {
          // A node that swallows the fault still can't hide the host's failure.
        }
        return new Output("swallowed");
      },
    ],
    [
      "catches the fault and throws its own error",
      (emit: () => void): NodeResult => {
        try {
          emit();
        } catch {
          throw new Error("replaced");
        }
        return new Output("unreachable");
      },
    ],
  ] as const) {
    test(`a node that ${label} still yields the sink's error`, async () => {
      class CleanupNode extends Node {
        readonly kind = NodeKind.CODE;
        override readonly needsLiveEvents = true;

        override forward(_inputs: Record<string, unknown>, caps: NodeCaps): NodeResult {
          return forward(() => caps.emitStream!(new StreamChunk({ nodeId: this.id, key: "output", chunk: "first" })));
        }
      }
      const expected = new Error("callback failed");

      expect(await rejected(drive(new CleanupNode("resource"), { emitStream: reject(expected) }))).toBe(expected);
    });
  }

  for (const settle of ["rejects", "returns"] as const) {
    test(`a capability fault beats a stop (the node ${settle} after the abort)`, async () => {
      const controller = new AbortController();
      const gate = Promise.withResolvers<void>();

      class CatchThenWaitNode extends Node {
        readonly kind = NodeKind.CODE;
        override readonly needsLiveEvents = true;

        override async forward(_inputs: Record<string, unknown>, caps: NodeCaps): Promise<NodeResult> {
          try {
            caps.emitStream!(new StreamChunk({ nodeId: this.id, key: "output", chunk: "first" }));
          } catch {
            // Swallowed here; evalNode still reports it.
          }
          await gate.promise;
          if (settle === "rejects") throw caps.signal!.reason;
          return new Output("late");
        }
      }
      const expected = new Error("callback failed");
      const evaluation = drive(new CatchThenWaitNode("work"), {
        emitStream: reject(expected),
        signal: controller.signal,
      });

      controller.abort(new Error("stopped"));
      gate.resolve();

      expect(await rejected(evaluation)).toBe(expected);
    });
  }
});

describe("EvaluatedOutcome checks its fields", () => {
  test("every worker outcome is accepted", () => {
    for (const outcome of [new Output("v"), new Route("a"), new Grow({}), new EvaluationFailure(failure())]) {
      expect(new EvaluatedOutcome(outcome, null).outcome).toBe(outcome);
    }
  });

  test("an outcome outside the closed sum is refused", () => {
    for (const outcome of [{ value: "v" }, "done", null]) {
      expect(thrown(() => new EvaluatedOutcome(outcome as Output, null))).toEqual(
        new TypeError("outcome must be a WorkerOutcome"),
      );
    }
  });

  test("evidence must be exactly an EvaluationEvidence", () => {
    const evidence = new EvaluationEvidence(null, null, null, null);
    expect(new EvaluatedOutcome(new Output(), evidence).evidence).toBe(evidence);

    class Subclass extends EvaluationEvidence {}
    for (const bad of [new Subclass(null, null, null, null), {}]) {
      expect(thrown(() => new EvaluatedOutcome(new Output(), bad as EvaluationEvidence))).toEqual(
        new TypeError("evidence must be an EvaluationEvidence or null"),
      );
    }
  });

  test("bindings and backward capture travel together", () => {
    const bindings = new BindingCollector().freeze();
    const capture = new BackwardCapture({ inputNames: ["x"] });

    const evaluated = new EvaluatedOutcome(new Output(), null, bindings, capture);
    expect(evaluated.bindings).toBe(bindings);
    expect(evaluated.backwardCapture).toBe(capture);

    const together = new RangeError("binding capture and backward invocation metadata must be retained together");
    expect(thrown(() => new EvaluatedOutcome(new Output(), null, bindings, null))).toEqual(together);
    expect(thrown(() => new EvaluatedOutcome(new Output(), null, null, capture))).toEqual(together);
  });

  test("bindings must be exactly a BindingCapture", () => {
    const capture = new BackwardCapture();
    expect(thrown(() => new EvaluatedOutcome(new Output(), null, {} as never, capture))).toEqual(
      new TypeError("bindings must be a BindingCapture or null"),
    );
  });

  test("backward capture must be exactly a BackwardCapture and is checked again", () => {
    const bindings = new BindingCollector().freeze();
    expect(thrown(() => new EvaluatedOutcome(new Output(), null, bindings, {} as never))).toEqual(
      new TypeError("backward_capture must be a BackwardCapture or null"),
    );

    // A capture built without its constructor is checked as the constructor would.
    const forged = Object.assign(Object.create(BackwardCapture.prototype) as BackwardCapture, {
      inputNames: ["x", "x"],
      variableNames: [],
      scope: null,
      nodeData: {},
    });
    expect(thrown(() => new EvaluatedOutcome(new Output(), null, bindings, forged))).toEqual(
      new RangeError("input_names must contain unique names"),
    );
  });
});

describe("requireSteering", () => {
  const inbox: Steering = {
    interruptSignal: () => new AbortController().signal,
    drain: () => [],
  };

  test("returns null or the inbox itself", () => {
    expect(requireSteering(null)).toBeNull();
    expect(requireSteering(inbox)).toBe(inbox);
  });

  test("refuses anything without both methods", () => {
    const message = "steering must provide interruptSignal(nodeId) and drain(nodeId), or be null";
    for (const bad of [
      undefined,
      {},
      "steering",
      { drain: inbox.drain },
      { interruptSignal: inbox.interruptSignal },
      { interruptSignal: inbox.interruptSignal, drain: [] },
    ]) {
      expect(thrown(() => requireSteering(bad))).toEqual(new TypeError(message));
    }
  });
});

describe("the steering capability", () => {
  /** Wrap an inbox whose methods are `interruptSignal` and `drain`. */
  function wrap(methods: { interruptSignal?: (id: string) => unknown; drain?: (id: string) => unknown }): Steering {
    return new SteeringCapability({
      interruptSignal: methods.interruptSignal ?? (() => new AbortController().signal),
      drain: methods.drain ?? (() => []),
    } as Steering, () => {}, [], () => false);
  }

  test("passes the node id and returns the host's signal", () => {
    const signal = new AbortController().signal;
    const seen: string[] = [];
    const steering = wrap({ interruptSignal: (id) => (seen.push(id), signal) });

    expect(steering.interruptSignal("n")).toBe(signal);
    expect(seen).toEqual(["n"]);
  });

  test("an interrupt signal that isn't an AbortSignal is a capability fault", () => {
    for (const bad of [true, null, {}]) {
      const fault = thrown(() => wrap({ interruptSignal: () => bad }).interruptSignal("n"));
      expect(fault).toBeInstanceOf(CapabilityFault);
      expect((fault as CapabilityFault).original).toEqual(
        new TypeError("steering.interruptSignal must return an AbortSignal"),
      );
    }
  });

  test("drain returns a fresh array of the host's messages", () => {
    const messages = ["a", "b"];
    const drained = wrap({ drain: () => messages }).drain("n");

    expect(drained).toEqual(["a", "b"]);
    expect(drained).not.toBe(messages);
  });

  test("drain may return a promise of the messages", async () => {
    const drained = wrap({ drain: async () => ["a"] }).drain("n");

    expect(drained).toBeInstanceOf(Promise);
    expect(await drained).toEqual(["a"]);
  });

  test("drain messages that aren't an array of strings are a capability fault", async () => {
    const shape = new TypeError("steering.drain must return an array of strings");
    for (const bad of ["a", ["a", 1], null, new Set(["a"])]) {
      const fault = thrown(() => wrap({ drain: () => bad }).drain("n"));
      expect(fault).toBeInstanceOf(CapabilityFault);
      expect((fault as CapabilityFault).original).toEqual(shape);

      const late = await rejected(wrap({ drain: async () => bad }).drain("n"));
      expect(late).toBeInstanceOf(CapabilityFault);
      expect((late as CapabilityFault).original).toEqual(shape);
    }
  });

  test("a host throw or rejection is a capability fault keeping the original", async () => {
    const expected = new Error("inbox failed");
    for (const fault of [
      thrown(() => wrap({ interruptSignal: () => { throw expected; } }).interruptSignal("n")),
      thrown(() => wrap({ drain: () => { throw expected; } }).drain("n")),
      await rejected(wrap({ drain: () => Promise.reject(expected) }).drain("n")),
    ]) {
      expect(fault).toBeInstanceOf(CapabilityFault);
      expect((fault as CapabilityFault).original).toBe(expected);
    }
  });

  test("a node catching an inbox fault still rejects evalNode with the host's error", async () => {
    const expected = new Error("inbox failed");

    class SteeredNode extends Node {
      readonly kind = NodeKind.CODE;
      override needsSteering = true;

      override forward(_inputs: Record<string, unknown>, caps: NodeCaps): NodeResult {
        try {
          caps.steering!.drain(this.id);
        } catch {
          // Swallowed here; evalNode still reports it.
        }
        return new Output("swallowed");
      }
    }
    const steering: Steering = {
      interruptSignal: () => new AbortController().signal,
      drain: () => {
        throw expected;
      },
    };

    expect(await rejected(drive(new SteeredNode("work"), { steering }))).toBe(expected);
  });

  /** A host inbox whose `drain` answers with `answer`. */
  function inbox(answer: () => unknown): Steering {
    return { interruptSignal: () => new AbortController().signal, drain: answer as Steering["drain"] };
  }

  /** A node that runs `body` over its steering, then returns. */
  class DrainNode extends Node {
    readonly kind = NodeKind.CODE;
    override needsSteering = true;
    override needsLiveEvents = true;
    readonly body: (caps: NodeCaps, nodeId: string) => Promise<void> | void;

    constructor(nodeId: string, body: (caps: NodeCaps, nodeId: string) => Promise<void> | void) {
      super(nodeId);
      this.body = body;
    }

    override async forward(_inputs: Record<string, unknown>, caps: NodeCaps): Promise<NodeResult> {
      await this.body(caps, this.id);
      return new Output("returned");
    }
  }

  test("a node catching an async drain fault still rejects evalNode with the host's error", async () => {
    const expected = new Error("inbox failed");
    const node = new DrainNode("work", async (caps, nodeId) => {
      try {
        await caps.steering!.drain(nodeId);
      } catch {
        // Swallowed here; evalNode still reports it.
      }
    });

    expect(await rejected(drive(node, { steering: inbox(() => Promise.reject(expected)) }))).toBe(expected);
    expect(await rejected(drive(node, { steering: inbox(async () => [1]) }))).toEqual(
      new TypeError("steering.drain must return an array of strings"),
    );
  });

  test("a drain the node drops and the host rejects after the node returns still rejects evalNode", async () => {
    const expected = new Error("inbox failed");
    const host = Promise.withResolvers<string[]>();
    const unhandled: unknown[] = [];
    const guard = (reason: unknown) => void unhandled.push(reason);
    process.on("unhandledRejection", guard);
    try {
      const node = new DrainNode("work", (caps, nodeId) => {
        caps.steering!.drain(nodeId);
      });
      let settled = false;
      const evaluation = drive(node, { steering: inbox(() => host.promise) });
      evaluation.then(
        () => (settled = true),
        () => (settled = true),
      );

      // A macrotask turn: the node has returned, and evalNode waits for the dropped drain.
      await new Promise((resolve) => setTimeout(resolve, 0));
      expect(settled).toBe(false);
      host.reject(expected);

      expect(await rejected(evaluation)).toBe(expected);
      await new Promise((resolve) => setTimeout(resolve, 0));
    } finally {
      process.off("unhandledRejection", guard);
    }
    expect(unhandled).toEqual([]);
  });

  test("of two faults the node catches, the first is thrown", async () => {
    const first = new Error("sink failed");
    const node = new DrainNode("work", (caps, nodeId) => {
      for (const fail of [
        () => caps.emitStream!(new StreamChunk({ nodeId, key: "output", chunk: "a" })),
        () => caps.steering!.drain(nodeId),
      ]) {
        try {
          fail();
        } catch {
          // Swallowed here; evalNode still reports the first.
        }
      }
    });
    const steering = inbox(() => {
      throw new Error("inbox failed");
    });

    const error = await rejected(
      drive(node, {
        steering,
        emitStream: () => {
          throw first;
        },
      }),
    );

    expect(error).toBe(first);
  });

  test("onFault sees the first fault's original once, when it is raised", async () => {
    const first = new Error("sink failed");
    const seen: unknown[] = [];
    const node = new DrainNode("work", (caps, nodeId) => {
      for (const fail of [
        () => caps.emitStream!(new StreamChunk({ nodeId, key: "output", chunk: "a" })),
        () => caps.steering!.drain(nodeId),
      ]) {
        try {
          fail();
        } catch {}
      }
      // Seen before the node settles.
      seen.push("returned");
    });
    const steering = inbox(() => {
      throw new Error("inbox failed");
    });
    const emitStream = (): void => {
      throw first;
    };

    expect(await rejected(drive(node, { steering, emitStream, onFault: (original) => void seen.push(original) }))).toBe(
      first,
    );
    expect(seen).toEqual([first, "returned"]);
  });

  test("a capability fault from the host passes through", async () => {
    const carried = new CapabilityFault(new Error("inner"));
    expect(thrown(() => wrap({ drain: () => { throw carried; } }).drain("n"))).toBe(carried);
    expect(await rejected(wrap({ drain: () => Promise.reject(carried) }).drain("n"))).toBe(carried);
  });
});
