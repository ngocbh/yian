/**
 * Raw node evaluation for the append-only dispatcher: a port of
 * `tests/engine/test_eval_node_raw.py`.
 *
 * Ported: `test_raw_evaluation_returns_closed_success_route_and_grow_values`,
 * `test_raw_node_raise_and_invalid_grow_are_failures_not_observers` (cause type `Error` for
 * Python's `RuntimeError`), `test_raw_streaming_uses_only_the_injected_transient_callback`,
 * `test_raw_stream_callback_failure_propagates_unchanged`.
 *
 * Ported (adapted): `test_scoped_render_capture_requires_recorded_model_map_entry`, on the node
 * data learning rebuilds (`rebuildNodeData`), since TS records none: a missing map entry fails the
 * rebuild, and the rebuilt config is a copy.
 *
 * Ported (adapted): `test_public_evaluation_always_returns_frozen_slotted_wrapper` and
 * `test_raw_failure_is_frozen_slotted_and_preserves_diagnostics`: the field names are the own
 * keys, frozen means `Object.isFrozen` and a throwing write, and the traceback is the binder
 * error's stack. `test_stream_body_exception_and_invalid_return_use_distinct_stages`: a node
 * streams through `caps.emitStream` (it declares `needsLiveEvents`), so the stream body is the
 * node's `forward`; the body's `ValueError` is a `RangeError`, and the invalid return gets the
 * ordinary non-NodeResult message (TS has no generator to name).
 *
 * Not ported: `test_render_capture_is_opt_in_single_write_and_detached_until_close`,
 * `test_invalid_render_capture_is_a_host_fault` and
 * `test_render_capture_preserves_original_host_exception_and_closes`: an evaluation captures no
 * render; learning renders the occurrence again.
 *
 * Changed: the shared `drive` evaluates the node in a one-node `CompiledFlow` wired from a
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
 * capability's checks on what the host returns, a steering fault the node catches still rejecting
 * `evalNode`, the first of two faults the node catches winning (and the only one `onFault` sees,
 * when it is raised).
 */

import { describe, expect, test } from "bun:test";

import { Failure, OriginRef } from "../../src/diagnostics.ts";
import { StreamChunk } from "../../src/events.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
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
import {
  EvaluatedOutcome,
  EvaluationFailure,
  evalNode,
  rebuildNodeData,
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
  expect(evaluated).toEqual(new EvaluatedOutcome(new Output("done")));
  expect(Object.keys(evaluated)).toEqual(["outcome", "bindings", "backwardCapture"]);
  expect(Object.isFrozen(evaluated)).toBe(true);
  expect(() => {
    (evaluated as { bindings: unknown }).bindings = null;
  }).toThrow(TypeError);
});

test("raw evaluation returns closed success route and grow values", async () => {
  expect(await drive(new FuncNode("work", () => "done"))).toEqual(new Output("done"));
  expect(await drive(new ResultNode("branch", new Route("chosen")))).toEqual(new Route("chosen"));
  expect(await drive(new Spawner("spawn", new Grow({ value: 1n })))).toEqual(new Grow({ value: 1n }));
});

test("the rebuilt node data of a scoped agent requires the recorded model map entry", () => {
  const config = { provider: "openai", model: "ambient" };
  const node = new AgentNode("a/runtime/id", { prompt: "hello", llmConfig: config });
  node.executionScope = new ExecutionScope(["authored", "agent"]);
  expect(() => rebuildNodeData(node, {}, {})).toThrow(
    new RangeError("recorded effective model configuration is missing static path 'authored/agent'"),
  );
  const recorded = { provider: "openai", model: "recorded", openai: { temperature: 0.5 } };
  const data = rebuildNodeData(node, {}, { "authored/agent": recorded });
  recorded.model = "later mutation";
  expect(data["effective_llm_config"]).toEqual({ provider: "openai", model: "recorded", openai: { temperature: 0.5 } });
  expect(data["rendered_prompt"]).toBe("hello");
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

  test("bindings and backward capture travel together", () => {
    const bindings = new BindingCollector().freeze();
    const capture = new BackwardCapture({ inputNames: ["x"] });

    const evaluated = new EvaluatedOutcome(new Output(), bindings, capture);
    expect(evaluated.bindings).toBe(bindings);
    expect(evaluated.backwardCapture).toBe(capture);

    const together = new RangeError("binding capture and backward invocation metadata must be retained together");
    expect(thrown(() => new EvaluatedOutcome(new Output(), bindings, null))).toEqual(together);
    expect(thrown(() => new EvaluatedOutcome(new Output(), null, capture))).toEqual(together);
  });

  test("bindings must be exactly a BindingCapture", () => {
    const capture = new BackwardCapture();
    expect(thrown(() => new EvaluatedOutcome(new Output(), {} as never, capture))).toEqual(
      new TypeError("bindings must be a BindingCapture or null"),
    );
  });

  test("backward capture must be exactly a BackwardCapture and is checked again", () => {
    const bindings = new BindingCollector().freeze();
    expect(thrown(() => new EvaluatedOutcome(new Output(), bindings, {} as never))).toEqual(
      new TypeError("backward_capture must be a BackwardCapture or null"),
    );

    // A capture built without its constructor is checked as the constructor would.
    const forged = Object.assign(Object.create(BackwardCapture.prototype) as BackwardCapture, {
      inputNames: ["x", "x"],
      variableNames: [],
      scope: null,
      nodeData: {},
    });
    expect(thrown(() => new EvaluatedOutcome(new Output(), bindings, forged))).toEqual(
      new RangeError("input_names must contain unique names"),
    );
  });
});

describe("requireSteering", () => {
  const inbox: Steering = { interruptSignal: () => new AbortController().signal };

  test("returns null or the steering itself", () => {
    expect(requireSteering(null)).toBeNull();
    expect(requireSteering(inbox)).toBe(inbox);
  });

  test("refuses anything without interruptSignal", () => {
    const message = "steering must provide interruptSignal(nodeId), or be null";
    for (const bad of [undefined, {}, "steering", { drain: () => [] }, { interruptSignal: true }]) {
      expect(thrown(() => requireSteering(bad))).toEqual(new TypeError(message));
    }
  });
});

describe("the steering capability", () => {
  /** Wrap a host whose interrupt signal is `interruptSignal`'s answer. */
  function wrap(interruptSignal: (id: string) => unknown = () => new AbortController().signal): Steering {
    return new SteeringCapability({ interruptSignal } as Steering, () => {}, () => false);
  }

  test("passes the node id and returns the host's signal", () => {
    const signal = new AbortController().signal;
    const seen: string[] = [];
    const steering = wrap((id) => (seen.push(id), signal));

    expect(steering.interruptSignal("n")).toBe(signal);
    expect(seen).toEqual(["n"]);
  });

  test("an interrupt signal that isn't an AbortSignal is a capability fault", () => {
    for (const bad of [true, null, {}]) {
      const fault = thrown(() => wrap(() => bad).interruptSignal("n"));
      expect(fault).toBeInstanceOf(CapabilityFault);
      expect((fault as CapabilityFault).original).toEqual(
        new TypeError("steering.interruptSignal must return an AbortSignal"),
      );
    }
  });

  test("a host throw is a capability fault keeping the original", () => {
    const expected = new Error("inbox failed");
    const fault = thrown(() => wrap(() => { throw expected; }).interruptSignal("n"));
    expect(fault).toBeInstanceOf(CapabilityFault);
    expect((fault as CapabilityFault).original).toBe(expected);
  });

  /** A host whose `interruptSignal` throws `error`. */
  function failing(error: unknown): Steering {
    return {
      interruptSignal: () => {
        throw error;
      },
    };
  }

  /** A node that runs `body` over its capabilities, then returns. */
  class SteeredNode extends Node {
    readonly kind = NodeKind.CODE;
    override readsSteerInbox = true;
    override needsLiveEvents = true;
    readonly body: (caps: NodeCaps, nodeId: string) => void;

    constructor(nodeId: string, body: (caps: NodeCaps, nodeId: string) => void) {
      super(nodeId);
      this.body = body;
    }

    override forward(_inputs: Record<string, unknown>, caps: NodeCaps): NodeResult {
      this.body(caps, this.id);
      return new Output("returned");
    }
  }

  test("a node catching a steering fault still rejects evalNode with the host's error", async () => {
    const expected = new Error("inbox failed");
    const node = new SteeredNode("work", (caps, nodeId) => {
      try {
        caps.steering!.interruptSignal(nodeId);
      } catch {
        // Swallowed here; evalNode still reports it.
      }
    });

    expect(await rejected(drive(node, { steering: failing(expected) }))).toBe(expected);
  });

  test("of two faults the node catches, the first is thrown", async () => {
    const first = new Error("sink failed");
    const node = new SteeredNode("work", (caps, nodeId) => {
      for (const fail of [
        () => caps.emitStream!(new StreamChunk({ nodeId, key: "output", chunk: "a" })),
        () => caps.steering!.interruptSignal(nodeId),
      ]) {
        try {
          fail();
        } catch {
          // Swallowed here; evalNode still reports the first.
        }
      }
    });

    const error = await rejected(
      drive(node, {
        steering: failing(new Error("inbox failed")),
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
    const node = new SteeredNode("work", (caps, nodeId) => {
      for (const fail of [
        () => caps.emitStream!(new StreamChunk({ nodeId, key: "output", chunk: "a" })),
        () => caps.steering!.interruptSignal(nodeId),
      ]) {
        try {
          fail();
        } catch {}
      }
      // Seen before the node settles.
      seen.push("returned");
    });
    const emitStream = (): void => {
      throw first;
    };

    expect(
      await rejected(
        drive(node, { steering: failing(new Error("inbox failed")), emitStream, onFault: (original) => void seen.push(original) }),
      ),
    ).toBe(first);
    expect(seen).toEqual([first, "returned"]);
  });

  test("a capability fault from the host passes through", () => {
    const carried = new CapabilityFault(new Error("inner"));
    expect(thrown(() => wrap(() => { throw carried; }).interruptSignal("n"))).toBe(carried);
  });
});
