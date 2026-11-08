/**
 * Fake memories for tests: `RecordingSession`, a `MemorySession` that records the calls it gets and
 * answers as its test sets, `FakeMemory`, a `MemoryBase` over one that counts its opens,
 * `wordEmbed`, a deterministic embedder for a VectorDBMemory, and `foreignWal`, another program's
 * database a VectorDBMemory must leave alone.
 */

import fs from "node:fs";
import { join } from "node:path";

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
import type { EmbeddingModel } from "../../src/llm/embeddings.ts";
import { FileMemory } from "../../src/memory/file.ts";
import { openSqlite } from "../../src/platform.ts";
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

/** The words that each own one dimension of a `wordEmbed` vector. */
const WORDS = ["alpha", "bravo", "charlie", "delta"];

/**
 * A deterministic embedder: each text's vector counts the words `alpha` to `delta` in it, one
 * dimension each, plus one so that no vector is zero. `calls` holds each call's texts and model.
 */
export function wordEmbed(): ((
  texts: readonly string[],
  o: EmbeddingModel & { signal: AbortSignal },
) => Promise<Float32Array[]>) & { calls: { texts: string[]; model: string }[] } {
  const calls: { texts: string[]; model: string }[] = [];
  const embed = async (texts: readonly string[], o: EmbeddingModel & { signal: AbortSignal }) => {
    calls.push({ texts: [...texts], model: `${o.provider}/${o.model}` });
    return texts.map((text) => {
      const words = text.toLowerCase().split(/[^a-z]+/);
      return Float32Array.from([...WORDS.map((word) => words.filter((each) => each === word).length), 1]);
    });
  };
  return Object.assign(embed, { calls });
}

/**
 * Make `<dir>/<name>`, another program's database in WAL mode whose last insert is still only in
 * its `-wal` (copied while the writer had it open); return the database's path.
 */
export function foreignWal(dir: string, name: string): string {
  const source = join(dir, `${name}.source`);
  const db = openSqlite(source);
  db.exec("PRAGMA journal_mode = WAL");
  db.exec("CREATE TABLE notes(x); INSERT INTO notes VALUES (1)");
  db.exec("PRAGMA wal_checkpoint(TRUNCATE)");
  db.exec("INSERT INTO notes VALUES (2)");
  const copy = join(dir, name);
  fs.copyFileSync(source, copy);
  fs.copyFileSync(`${source}-wal`, `${copy}-wal`);
  db.close();
  fs.rmSync(source);
  return copy;
}
