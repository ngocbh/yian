/**
 * Raw node evaluation for the append-only dispatcher: closed success, route and grow values,
 * failures, stream callbacks, frozen outcomes, and stops (an aborted signal rejects with its
 * reason).
 */

import { describe, expect, test } from "bun:test";

import { withEnv } from "../../src/defaults.ts";
import { OriginRef } from "../../src/diagnostics.ts";
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
  type SteerInbox,
} from "../../src/nodes/base.ts";
import { ExecutionScope, Parameter } from "../../src/nodes/contracts.ts";
import {
  EvaluatedOutcome,
  EvaluationFailure,
  evalNode,
  rebuildNodeData,
} from "../../src/runtime/eval_node.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { drive, FailNode, FuncNode, nodeFlow, rejected, StreamNode } from "../helpers/run_state_fixtures.ts";

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

test("a failure's message and traceback are cut to the failure cap, saying how much was cut", async () => {
  const raised = await withEnv(
    (name) => (name === "YIAN_MAX_FAILURE_CHARS" ? "50" : undefined),
    () => drive(new FailNode("raise", "😀".repeat(3_000))),
  );
  const { message, traceback } = (raised as EvaluationFailure).failure;
  expect(message).toBe("😀".repeat(50) + "… (2950 more characters)");
  // The stack starts with `Error: ` and the message; its frames are cut too.
  const cut = /^Error: (?:😀){43}… \((\d+) more characters\)$/u.exec(traceback!);
  expect(cut).not.toBeNull();
  expect(Number(cut![1])).toBeGreaterThan(2_957);
});

test("a failure's message and traceback within the failure cap are kept whole", async () => {
  const raised = await drive(new FailNode("raise", "x".repeat(9_000)));
  const { message, traceback } = (raised as EvaluationFailure).failure;
  expect(message).toBe("x".repeat(9_000));
  expect(traceback!.startsWith(`Error: ${message}\n`)).toBe(true);
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

  for (const [label, settle] of [
    [
      "its own error",
      () => {
        throw new Error("interrupted");
      },
    ],
    [
      "the reason",
      (caps: NodeCaps) => {
        throw caps.signal!.reason;
      },
    ],
  ] as const) {
    test(`a node rejecting after its signal aborted rejects with the reason (${label})`, async () => {
      const controller = new AbortController();
      const reason = new Error("stopped");
      const gate = Promise.withResolvers<void>();
      const evaluation = drive(new GatedNode("work", gate.promise, settle), { signal: controller.signal });

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

describe("the steer inbox capability's faults", () => {
  /** An engine steer inbox opener whose opening fails with `error`, as a capability fault. */
  function failing(error: unknown): (inbox: string) => SteerInbox {
    return () => {
      throw new CapabilityFault(error);
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

  test("a node catching a steer inbox fault still rejects evalNode with the host's error", async () => {
    const expected = new Error("inbox failed");
    const node = new SteeredNode("work", (caps, nodeId) => {
      try {
        caps.openSteerInbox!(nodeId);
      } catch {
        // Swallowed here; evalNode still reports it.
      }
    });

    expect(await rejected(drive(node, { steerInbox: failing(expected) }))).toBe(expected);
  });

  test("of two faults the node catches, the first is thrown", async () => {
    const first = new Error("sink failed");
    const node = new SteeredNode("work", (caps, nodeId) => {
      for (const fail of [
        () => caps.emitStream!(new StreamChunk({ nodeId, key: "output", chunk: "a" })),
        () => caps.openSteerInbox!(nodeId),
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
        steerInbox: failing(new Error("inbox failed")),
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
        () => caps.openSteerInbox!(nodeId),
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
        drive(node, { steerInbox: failing(new Error("inbox failed")), emitStream, onFault: (original) => void seen.push(original) }),
      ),
    ).toBe(first);
    expect(seen).toEqual([first, "returned"]);
  });
});
