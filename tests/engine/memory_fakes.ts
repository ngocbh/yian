/** Observable injected memory sessions shared by runtime tests. */

import {
  freezeJsonObject,
  type JsonObject,
  type LearnContext,
  type Memory,
  type MemoryMutation,
  type MemorySession,
  type RecallContext,
} from "../../src/memory/contracts.ts";

/** One immutable-snapshot fake session. */
export class FakeSession implements MemorySession {
  readonly memory: FakeMemory;
  closed = false;
  readonly #snapshot: JsonObject;

  constructor(memory: FakeMemory, snapshot: JsonObject) {
    this.memory = memory;
    this.#snapshot = freezeJsonObject(snapshot, "snapshot", { limitNesting: false });
  }

  get snapshot(): JsonObject {
    return this.#snapshot;
  }

  recall(context: RecallContext | null = null): string {
    this.memory.recallContexts.push(context);
    return this.memory.recallText;
  }

  /** Pop the next learn outcome: an `Error` is thrown, a function is called with `context`. */
  learn(context: LearnContext): readonly MemoryMutation[] {
    this.memory.learnContexts.push(context);
    const outcome = this.memory.learnOutcomes.length > 0 ? this.memory.learnOutcomes.shift() : [];
    if (outcome instanceof Error) throw outcome;
    if (typeof outcome === "function") {
      return (outcome as (context: LearnContext) => readonly MemoryMutation[])(context);
    }
    // A test may queue a malformed result on purpose.
    return outcome as readonly MemoryMutation[];
  }

  /** Pop the next commit outcome: an `Error` is thrown, anything else is returned. */
  commit(commitId: string, mutations: readonly MemoryMutation[]): void {
    this.memory.commitCalls.push([commitId, mutations]);
    const outcome = this.memory.commitOutcomes.shift();
    if (outcome instanceof Error) throw outcome;
    return outcome as void;
  }

  close(): void {
    if (!this.closed) {
      this.closed = true;
      this.memory.closeCount += 1;
    }
  }
}

/** Observable structural `Memory` implementation for public-boundary tests. */
export class FakeMemory implements Memory {
  recallText: string;
  readonly learnOutcomes: unknown[];
  readonly commitOutcomes: unknown[];
  readonly openSnapshots: (JsonObject | null)[] = [];
  readonly sessions: FakeSession[] = [];
  readonly recallContexts: (RecallContext | null)[] = [];
  readonly learnContexts: LearnContext[] = [];
  readonly commitCalls: [string, readonly MemoryMutation[]][] = [];
  closeCount = 0;
  resetCount = 0;

  constructor(
    options: { recallText?: string; learnOutcomes?: readonly unknown[]; commitOutcomes?: readonly unknown[] } = {},
  ) {
    this.recallText = options.recallText ?? "";
    this.learnOutcomes = [...(options.learnOutcomes ?? [])];
    this.commitOutcomes = [...(options.commitOutcomes ?? [])];
  }

  openRun(options: { snapshot?: JsonObject | null } = {}): FakeSession {
    // The runtime passes a recorded envelope's deep-frozen snapshot, so it is kept without a copy.
    const restored = options.snapshot ?? null;
    this.openSnapshots.push(restored);
    const session = new FakeSession(this, restored ?? { namespace: "test" });
    this.sessions.push(session);
    return session;
  }

  reset(): void {
    this.resetCount += 1;
  }
}
