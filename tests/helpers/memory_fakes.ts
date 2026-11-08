/**
 * Fake memories for tests: `RecordingSession`, a `MemorySession` that records the calls it gets and
 * answers as its test sets, and `FakeMemory`, a `MemoryBase` over one that counts its opens.
 */

import {
  type Applied,
  type Author,
  type ChangeEntry,
  type Erased,
  MemoryBase,
  type MemoryContract,
  type MemoryInfo,
  type MemorySession,
  type Recall,
} from "../../src/memory/base.ts";
import { FileMemory } from "../../src/memory/file.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { MemoryRef } from "../../src/typesys/values.ts";

/** The info of a FileMemory-shaped memory with identity `identity`. */
export function fileInfo(identity: string, settings: JsonObject = {}): MemoryInfo {
  return { type: "FileMemory", identity, settings, tools: FileMemory.contract.tools };
}

/** A FileMemory-shaped memory value with identity `identity`. */
export function fileRef(identity: string, recall: Partial<Recall> = {}): MemoryRef {
  return new MemoryRef({
    identity,
    type: "FileMemory",
    settings: {},
    version: "v1",
    recall: { text: "", cut: null, hint: null, ...recall },
    tools: FileMemory.contract.tools,
  });
}

/** A session that records each call and answers with what its test sets. */
export class RecordingSession implements MemorySession {
  readonly info: MemoryInfo;
  readonly created: boolean;
  /** Each `apply`'s id, change and author, in call order. */
  readonly applied: { id: string; change: JsonObject; author: Author }[] = [];
  /** Each read tool call's tool and arguments, in call order. */
  readonly calls: { tool: string; args: JsonObject }[] = [];
  snapshots = 0;
  closed = 0;
  /** What `apply` gives: an outcome, or an error it throws. */
  onApply: (change: JsonObject) => Applied | Promise<Applied> = () => ({ ok: true, version: "v2", result: {} });
  /** What `call` gives: a result, or an error it throws. */
  onCall: (tool: string, args: JsonObject) => JsonObject | Promise<JsonObject> = () => ({});
  /** What `snapshot` gives. */
  onSnapshot: () => { version: string; recall: Recall } | Promise<{ version: string; recall: Recall }> = () => ({
    version: "v1",
    recall: { text: "", cut: null, hint: null },
  });

  constructor(info: MemoryInfo = fileInfo("t/lessons"), created = false) {
    this.info = info;
    this.created = created;
  }

  async snapshot(): Promise<{ version: string; recall: Recall }> {
    this.snapshots += 1;
    return await this.onSnapshot();
  }

  async call(tool: string, args: JsonObject): Promise<JsonObject> {
    this.calls.push({ tool, args });
    return await this.onCall(tool, args);
  }

  async apply(c: { id: string; change: JsonObject; author: Author }): Promise<Applied> {
    this.applied.push({ id: c.id, change: c.change, author: c.author });
    return await this.onApply(c.change);
  }

  undo(): Promise<Applied> {
    throw new Error("undo isn't faked");
  }

  erase(): Promise<Erased> {
    throw new Error("erase isn't faked");
  }

  history(): Promise<ChangeEntry[]> {
    throw new Error("history isn't faked");
  }

  async close(): Promise<void> {
    this.closed += 1;
  }
}

/**
 * A FileMemory-contract memory whose `open` gives `session()` and counts the opens; `failOpen`
 * makes `open` throw it instead.
 */
export class FakeMemory extends MemoryBase {
  static override readonly contract: MemoryContract = FileMemory.contract;
  opens = 0;
  readonly sessions: RecordingSession[] = [];
  /** The options of each `open`, in order. */
  readonly openedWith: { create: boolean; itemType: JsonObject | null }[] = [];
  failOpen: unknown = null;
  readonly #session: () => RecordingSession;

  constructor(session: () => RecordingSession = () => new RecordingSession()) {
    super();
    this.#session = session;
  }

  override async open(o: { create: boolean; itemType: JsonObject | null }): Promise<MemorySession> {
    this.opens += 1;
    this.openedWith.push({ create: o.create, itemType: o.itemType });
    if (this.failOpen !== null) throw this.failOpen;
    const session = this.#session();
    this.sessions.push(session);
    return session;
  }

  override reset(): Promise<void> {
    throw new Error("reset isn't faked");
  }
}
