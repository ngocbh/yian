/**
 * VectorDBMemory: its store, identity and settings, recall, read tools, writes, undo, erase and
 * reset, and the host actions re-embed and compact.
 */

import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";
import { type ChildProcess, spawn } from "node:child_process";
import { createHash } from "node:crypto";
import fs from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { withEnv } from "../../src/defaults.ts";
import type { EmbeddingModel } from "../../src/llm/embeddings.ts";
import {
  type Applied,
  type Author,
  BUILTIN_MEMORY_TOOLS,
  ChangeIdReusedError,
  checkContract,
  MemoryBusyError,
  memoryErrorKind,
  type MemorySession,
} from "../../src/memory/base.ts";
import { openStore, VectorDBMemory, type VectorDBMemoryOptions } from "../../src/memory/vectordb.ts";
import { isLocked, lockFile, openSqlite, virtualScheduler } from "../../src/platform.ts";
import { isPermanentToolError, isReportedToolError } from "../../src/tools/errors.ts";
import { type JsonObject, logJson } from "../../src/typesys/json.ts";

let dir: string;
let path: string;
const signal = new AbortController().signal;
const o = { create: true, itemType: null, signal };
const model: EmbeddingModel = { provider: "openai", model: "a" };

/** The words that each own one of a fake vector's four dimensions. */
const WORDS = ["alpha", "bravo", "charlie", "delta"];

/** The texts each `embed` call got, with its model. */
let embedded: { texts: string[]; provider: string; model: string }[];

/**
 * A fake embedder: each text's vector counts the words `alpha` to `delta` in it, one dimension
 * each, and adds any other word to the dimension its characters' sum picks.
 */
async function embed(texts: readonly string[], o: EmbeddingModel & { signal: AbortSignal }): Promise<Float32Array[]> {
  embedded.push({ texts: [...texts], provider: o.provider, model: o.model });
  return texts.map((text) => {
    const vector = new Float32Array(4);
    for (const word of text.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean)) {
      const known = WORDS.indexOf(word);
      vector[known === -1 ? [...word].reduce((sum, c) => sum + c.charCodeAt(0), 0) % 4 : known]! += 1;
    }
    return vector;
  });
}

beforeEach(() => {
  dir = fs.mkdtempSync(join(tmpdir(), "yian-vm-"));
  path = join(dir, "store", "cases.db");
  embedded = [];
});

afterEach(() => {
  mock.restore();
  fs.rmSync(dir, { recursive: true, force: true });
});

/** Return the sha256 hex digest of `data`. */
function sha256(data: string): string {
  return createHash("sha256").update(data).digest("hex");
}

/** Return a VectorDBMemory on `path` with the fake embedder and `options`. */
function memory(options: Partial<VectorDBMemoryOptions> = {}): VectorDBMemory {
  return new VectorDBMemory({ path, embedding: model, embed, id: "t/cases", ...options });
}

/** Open a VectorDBMemory on `path`, creating it. */
function openAt(options: Partial<VectorDBMemoryOptions> = {}, itemType: JsonObject | null = null) {
  return memory(options).open({ ...o, itemType });
}

/** Return the store's `meta` rows as an object, read straight from the file. */
function metaRows(at = path): Record<string, string> {
  const db = openSqlite(at, { readonly: true });
  try {
    const rows = db.prepare("SELECT key, value FROM meta").all();
    return Object.fromEntries(rows.map((row) => [row.key as string, row.value as string]));
  } finally {
    db.close();
  }
}

/** Return what `promise` rejects with. */
async function rejection(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new Error("expected a rejection");
}

/** The JSON Schema of a `Case` item: a title and a list of tags. */
const CASE: JsonObject = {
  type: "object",
  properties: { title: { type: "string" }, tags: { type: "array", items: { type: "string" } } },
  required: ["title", "tags"],
  additionalProperties: false,
};

test("a new store is private, in a private folder; meta holds its identity, type, settings and version", async () => {
  const s = await openAt();
  expect(s.created).toBe(true);
  expect(fs.statSync(path).mode & 0o777).toBe(0o600);
  expect(fs.statSync(join(dir, "store")).mode & 0o777).toBe(0o700);
  expect(fs.statSync(`${path}.lock`).mode & 0o777).toBe(0o600);
  expect(metaRows()).toEqual({
    identity: "t/cases",
    type: "VectorDBMemory",
    settings: '{"embedding":{"model":"a","provider":"openai"},"item_schema":null}',
    store_version: "1",
    version: "0",
  });
  await s.close();
});

test("a new store with no id takes at-<16 hex> of its real path, and stores it", async () => {
  fs.mkdirSync(join(dir, "real"));
  fs.symlinkSync(join(dir, "real"), join(dir, "link"));
  const s = await memory({ path: join(dir, "link", "m.db"), id: undefined }).open(o);
  const real = join(fs.realpathSync(join(dir, "real")), "m.db");
  expect(s.info.identity).toBe(`at-${sha256(real).slice(0, 16)}`);
  expect(metaRows(real).identity).toBe(s.info.identity);
  await s.close();
});

test("a stored identity wins over none; another id is refused", async () => {
  await (await openAt()).close();
  await expect(openAt({ id: "t/other" })).rejects.toThrow("this memory is `t/cases`, not `t/other`");
  const again = await openAt({ id: undefined });
  expect(again.created).toBe(false);
  expect(again.info.identity).toBe("t/cases");
  await again.close();
  const resumed = await memory().open({ ...o, create: false });
  expect(resumed.created).toBe(false);
  await resumed.close();
});

test("open({create: false}) refuses a missing store as missing, naming no location, and creates nothing", async () => {
  fs.mkdirSync(join(dir, "store"));
  const error = await rejection(memory().open({ ...o, create: false }));
  expect(memoryErrorKind(error)).toBe("missing");
  expect((error as Error).message).toBe("the memory isn't at its location: there is no store");
  expect(fs.readdirSync(join(dir, "store"))).toEqual([]);
  await expect(memory({ path: join(dir, "none", "m.db") }).open({ ...o, create: false })).rejects.toThrow(
    "the memory isn't at its location: there is no store",
  );
  expect(fs.existsSync(join(dir, "none"))).toBe(false);
});

test("an empty file left where a store was being made becomes the store; create: false refuses it", async () => {
  fs.mkdirSync(join(dir, "store"), { mode: 0o700 });
  fs.writeFileSync(path, "", { mode: 0o600 });
  const error = await rejection(memory().open({ ...o, create: false }));
  expect(memoryErrorKind(error)).toBe("missing");
  expect((error as Error).message).toBe("the memory isn't at its location: its store isn't a memory yet");
  const s = await openAt();
  expect(s.created).toBe(true);
  expect(metaRows().identity).toBe("t/cases");
  await s.close();
});

test("opening with another embedding model is refused; with none, the stored one is used", async () => {
  await (await openAt()).close();
  await expect(openAt({ embedding: { provider: "openai", model: "b" } })).rejects.toThrow(
    "this memory was embedded with openai/a; re-embed it with `yian memory reembed`",
  );
  const s = await openAt({ embedding: undefined });
  expect(s.info.settings.embedding).toEqual({ provider: "openai", model: "a" });
  await s.close();
});

test("a new store needs an embedding model", async () => {
  await expect(openAt({ embedding: undefined })).rejects.toThrow("a new VectorDBMemory needs an embedding model");
  expect(fs.existsSync(join(dir, "store"))).toBe(false);
});

test("the store keeps the item schema it was made with, and refuses another at open", async () => {
  const typed = join(dir, "typed.db");
  await (await openAt({ path: typed }, CASE)).close();
  expect(JSON.parse(metaRows(typed).settings!).item_schema).toEqual(CASE);
  const other = { ...CASE, required: ["title"] };
  await expect(openAt({ path: typed }, other)).rejects.toThrow("this memory's items have another type");
  await expect(openAt({ path: typed }, null)).rejects.toThrow(
    "this memory's items have typed fields; give its input their type",
  );
  // Key order doesn't count: the schemas are compared as canonical JSON.
  const reordered = Object.fromEntries(Object.entries(CASE).reverse());
  await (await openAt({ path: typed }, reordered)).close();

  await (await openAt()).close();
  await expect(openAt({}, CASE)).rejects.toThrow("this memory's items are text only; its input can't give them a type");
});

test("an item type that isn't a JSON object is refused", async () => {
  await expect(memory().open({ ...o, itemType: [] as unknown as JsonObject })).rejects.toThrow(TypeError);
});

test("the store's connection has busy_timeout 0, secure_delete, a rollback journal and extra syncs", async () => {
  await (await openAt()).close();
  const db = openStore(path);
  try {
    expect(db.prepare("PRAGMA busy_timeout").get()).toEqual({ timeout: 0 });
    expect(db.prepare("PRAGMA secure_delete").get()).toEqual({ secure_delete: 1 });
    expect(db.prepare("PRAGMA journal_mode").get()).toEqual({ journal_mode: "delete" });
    // EXTRA (3) syncs the folder once the journal is deleted, the commit point in this mode.
    expect(db.prepare("PRAGMA synchronous").get()).toEqual({ synchronous: 3 });
    expect(db.prepare("PRAGMA temp_store").get()).toEqual({ temp_store: 2 });
  } finally {
    db.close();
  }
});

test("checkContract passes: the tools are the VectorDBMemory contract's, in spec order", () => {
  expect(() => checkContract(memory())).not.toThrow();
  expect(VectorDBMemory.contract.type).toBe("VectorDBMemory");
  expect(VectorDBMemory.contract.tools.map((tool) => tool.name)).toEqual([...BUILTIN_MEMORY_TOOLS.VectorDBMemory]);
});

test("the session's info holds the type, identity, settings with no location, and its tools", async () => {
  const s = await openAt({ maxK: 7, recallRecent: 3 });
  expect(s.info.type).toBe("VectorDBMemory");
  expect(s.info.identity).toBe("t/cases");
  expect(s.info.settings).toEqual({
    embedding: { provider: "openai", model: "a" },
    item_schema: null,
    max_items: 20_000n,
    max_k: 7n,
    max_text_chars: 8_000n,
    max_recall_chars: 4_000n,
    recall_recent: 3n,
    max_undo: 100n,
    max_change_ids: 10_000n,
  });
  const shownInfo = JSON.stringify(s.info, (_, value: unknown) => (typeof value === "bigint" ? String(value) : value));
  expect(shownInfo).not.toContain(dir);
  expect(s.info.tools.map((tool) => tool.name)).toEqual([...BUILTIN_MEMORY_TOOLS.VectorDBMemory]);
  for (const [i, tool] of s.info.tools.entries()) {
    expect(tool.description).toBe(VectorDBMemory.contract.tools[i]!.description);
    expect(tool.write).toBe(VectorDBMemory.contract.tools[i]!.write);
  }
  await s.close();
});

/** Return the parameters of the tool `name` among `tools`. */
function parametersOf(s: MemorySession, name: string): JsonObject {
  return s.info.tools.find((tool) => tool.name === name)!.parameters;
}

test("a session's tools take k up to maxK, and fields typed by the item schema or none for text only", async () => {
  const text = await openAt({ maxK: 7 });
  const search = parametersOf(text, "search");
  expect((search.properties as JsonObject).k).toMatchObject({ type: "integer", minimum: 1, maximum: 7 });
  expect(Object.keys(search.properties as JsonObject)).toEqual(["query", "k"]);
  expect(Object.keys(parametersOf(text, "add").properties as JsonObject)).toEqual(["text"]);
  expect(Object.keys(parametersOf(text, "edit").properties as JsonObject)).toEqual(["id", "if_version", "text"]);
  await text.close();

  const nested: JsonObject = {
    type: "object",
    properties: { where: { $ref: "#/$defs/Place" } },
    required: ["where"],
    additionalProperties: false,
    $defs: { Place: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } },
  };
  const typed = await openAt({ path: join(dir, "typed.db") }, nested);
  const add = parametersOf(typed, "add");
  // The item schema's definitions move to the parameters' root, where its references point.
  expect(add.$defs).toEqual(nested.$defs!);
  expect((add.properties as JsonObject).fields).toMatchObject({
    type: "object",
    properties: { where: { $ref: "#/$defs/Place" } },
    required: ["where"],
  });
  expect(add.required).toEqual(["text", "fields"]);
  expect(Object.keys(parametersOf(typed, "search").properties as JsonObject)).toEqual(["query", "k", "filter"]);
  expect(parametersOf(typed, "edit").$defs).toEqual(nested.$defs!);
  await typed.close();
});

test("the open that makes a store gives later opens' info, which the caller's item type can't change", async () => {
  // Key order counts: a model sees the properties in this order.
  const ordered = (value: unknown) => JSON.stringify(value, (_, x: unknown) => (typeof x === "bigint" ? `${x}n` : x));
  const itemType = structuredClone(CASE);
  const first = await openAt({}, itemType);
  const later = await openAt({}, CASE);
  expect(ordered(first.info)).toBe(ordered(later.info));
  // The host changes its own object afterwards: the session's settings and specs stay.
  (itemType.properties as JsonObject).title = { type: "integer" };
  expect(ordered(first.info)).toBe(ordered(later.info));
  await first.close();
  await later.close();
});

test("bad settings are refused at construction", () => {
  expect(() => memory({ path: "" })).toThrow("path");
  expect(() => memory({ id: "" })).toThrow("id");
  expect(() => memory({ id: "a\nb" })).toThrow("id");
  expect(() => memory({ embedding: { provider: "", model: "a" } })).toThrow("embedding");
  expect(() => memory({ embedding: { provider: "openai" } as EmbeddingModel })).toThrow("embedding");
  expect(() => memory({ maxItems: 0 })).toThrow("maxItems");
  expect(() => memory({ maxK: 1.5 })).toThrow("maxK");
  expect(() => memory({ recallRecent: -1 })).toThrow("recallRecent");
  expect(() => memory({ lockTimeoutMs: -1 })).toThrow("lockTimeoutMs");
  expect(() => memory({ embed: "x" as unknown as VectorDBMemoryOptions["embed"] })).toThrow("embed");
  expect(() => memory({ maxUndo: 0, recallRecent: 0 })).not.toThrow();
});

test("a folder at the path is refused", async () => {
  fs.mkdirSync(path, { recursive: true });
  await expect(openAt()).rejects.toThrow(`\`${path}\` is a folder, not a store file`);
});

test("a file that isn't a VectorDBMemory store is refused, and left as it was", async () => {
  fs.mkdirSync(join(dir, "store"));
  fs.writeFileSync(path, "not a database at all, just some text that is long enough to have a header");
  await expect(openAt()).rejects.toThrow(`\`${path}\` isn't a VectorDBMemory store: it isn't a SQLite database`);

  const other = join(dir, "other.db");
  const db = openSqlite(other);
  db.exec("CREATE TABLE runs(id TEXT)");
  db.close();
  await expect(openAt({ path: other })).rejects.toThrow(`\`${other}\` isn't a VectorDBMemory store`);
  const check = openSqlite(other, { readonly: true });
  expect(check.prepare("SELECT name FROM sqlite_master WHERE type = 'table'").all()).toEqual([{ name: "runs" }]);
  check.close();
});

test("a store of a newer layout is refused, naming its version", async () => {
  await (await openAt()).close();
  const db = openSqlite(path);
  db.prepare("UPDATE meta SET value = '2' WHERE key = 'store_version'").run();
  db.close();
  await expect(openAt()).rejects.toThrow("this memory's store is version 2; this yian reads version 1");
});

test.each([
  ["", "store file", "replace it with a copy"],
  [".lock", "lock file", "remove it"],
  ["-journal", "rollback journal", "replace it with a copy"],
])("a store's %s swapped for a symlink or a hard link after construction is refused", async (suffix, role, fix) => {
  await (await openAt()).close();
  // Built first: construction resolves a symlinked store path once.
  const m = memory();
  const target = `${path}${suffix}`;
  fs.copyFileSync(path, join(dir, "elsewhere"));
  fs.rmSync(target, { force: true });
  fs.symlinkSync(join(dir, "elsewhere"), target);
  expect(((await rejection(m.open(o))) as Error).message).toBe(
    `this memory's ${role} isn't a plain file (a symlink); ${fix}`,
  );
  fs.rmSync(target);
  fs.linkSync(join(dir, "elsewhere"), target);
  expect(((await rejection(m.open(o))) as Error).message).toBe(
    `this memory's ${role} isn't a plain file (a hard link); ${fix}`,
  );
});

test("a symlinked store path is resolved once, at construction", async () => {
  await (await openAt()).close();
  fs.symlinkSync(path, join(dir, "alias.db"));
  const s = await openAt({ path: join(dir, "alias.db"), id: undefined });
  expect(s.info.identity).toBe("t/cases");
  await s.close();
  expect(fs.existsSync(join(dir, "alias.db.lock"))).toBe(false);
});

test("a closed session's methods return rejected promises; none throws at the call", async () => {
  const s = await openAt();
  await s.close();
  const host: Author = { kind: "host", name: "h" };
  const calls: (() => Promise<unknown>)[] = [
    () => s.snapshot(),
    () => s.call("list", {}, { signal }),
    () => s.apply({ id: "x", change: { op: "add", text: "x" }, author: host, signal }),
    () => s.undo({ id: "u", changeId: "x", author: host, signal }),
    () => s.erase({ id: "e", target: { item: 1n }, author: host, signal }),
    () => s.history({ limit: 1 }),
  ];
  for (const call of calls) {
    let promise: Promise<unknown> | undefined;
    expect(() => {
      promise = call();
    }).not.toThrow();
    await expect(promise!).rejects.toThrow("this VectorDBMemory session is closed");
  }
});

/** One item to seed: its text, and its fields and why it is retired when it has them. */
type Seed = { text: string; fields?: JsonObject; retired?: string };

/**
 * Write `items` straight into the store, as changes 1, 2, ... after its current version would,
 * each embedded by the fake embedder; return their ids.
 */
async function seed(items: Seed[], at = path): Promise<number[]> {
  const vectors = await embed(
    items.map((item) => item.text),
    { ...model, signal },
  );
  embedded = [];
  const db = openSqlite(at);
  try {
    let version = Number(db.prepare("SELECT value FROM meta WHERE key = 'version'").get()!.value);
    const insert = db.prepare(
      "INSERT INTO items(text, fields, version, author, retired, vector) VALUES (?, ?, ?, ?, ?, ?)",
    );
    const ids: number[] = [];
    for (const [i, item] of items.entries()) {
      version += 1;
      const vector = vectors[i]!;
      const bytes = new Uint8Array(vector.buffer, vector.byteOffset, vector.byteLength);
      const fields = item.fields === undefined ? null : logJson(item.fields);
      const author = '{"kind":"host","name":"seed"}';
      ids.push(Number(insert.run(item.text, fields, version, author, item.retired ?? null, bytes).lastInsertRowid));
    }
    db.prepare("UPDATE meta SET value = ? WHERE key = 'version'").run(String(version));
    return ids;
  } finally {
    db.close();
  }
}

/** Call the read tool `tool` on `s` with `args`. */
function call(s: MemorySession, tool: string, args: JsonObject): Promise<JsonObject> {
  return s.call(tool, args, { signal });
}

const HINT = { tool: "search", text: "Search them with {tool}." };

describe("recall and snapshot", () => {
  test("the recall is the count of items, with a hint to search them", async () => {
    const s = await openAt();
    await seed([{ text: "alpha" }, { text: "bravo" }, { text: "charlie", retired: "wrong" }]);
    expect(await s.snapshot()).toEqual({ version: "3", recall: { text: "2 items", cut: null, hint: HINT } });
    await s.close();
  });

  test("an empty memory recalls no items, and no hint", async () => {
    const s = await openAt();
    expect(await s.snapshot()).toEqual({ version: "0", recall: { text: "0 items", cut: null, hint: null } });
    await s.close();
  });

  test("recallRecent shows the newest items after the count, newest first, retired ones left out", async () => {
    const s = await openAt({ recallRecent: 2 });
    const [first] = await seed([{ text: "alpha" }, { text: "bravo\nsecond line" }, { text: "charlie", retired: "x" }]);
    await seed([{ text: "delta" }]);
    const { recall } = await s.snapshot();
    expect(recall).toEqual({ text: "3 items\n[4] delta\n[2] bravo\n  second line", cut: null, hint: HINT });
    expect(first).toBe(1);
    await s.close();
  });

  test("the recall keeps within maxRecallChars, skipping an item too long, and says how many it left out", async () => {
    const s = await openAt({ recallRecent: 3, maxRecallChars: 26 });
    await seed([{ text: "short" }, { text: "a text far too long to fit" }, { text: "tiny" }]);
    const { recall } = await s.snapshot();
    expect(recall).toEqual({
      text: "3 items\n[3] tiny\n[1] short",
      cut: "1 of the 3 newest items not shown",
      hint: HINT,
    });
    expect(recall.text.length).toBe(26);
    await s.close();
  });

  test("a writer holding the lock makes snapshot() wait until it lets go", async () => {
    const s = await openAt({ lockTimeoutMs: 10_000 });
    const lock = lockFile(`${path}.lock`, { private: true });
    let settled = false;
    const snapshot = s.snapshot().finally(() => {
      settled = true;
    });
    await new Promise((resolve) => setTimeout(resolve, 150));
    expect(settled).toBe(false);
    lock.release();
    expect((await snapshot).version).toBe("0");
    await s.close();
  });

  test("with lockTimeoutMs: 0 a held lock makes snapshot() fail busy, naming no location", async () => {
    const s = await openAt({ lockTimeoutMs: 0, scheduler: virtualScheduler() });
    const lock = lockFile(`${path}.lock`, { private: true });
    try {
      const error = await rejection(s.snapshot());
      expect(error).toBeInstanceOf(MemoryBusyError);
      expect(memoryErrorKind(error)).toBe("busy");
      expect(`${(error as Error).message} ${String((error as Error).cause ?? "")}`).not.toContain(dir);
    } finally {
      lock.release();
    }
    await s.close();
  });
});

/** Return `error` when it is a permanent reported tool error, the refusal a model reads. */
function expectRefusal(error: unknown, message: string): void {
  expect(isReportedToolError(error)).toBe(true);
  expect(isPermanentToolError(error)).toBe(true);
  expect((error as Error).message).toBe(message);
}

describe("search", () => {
  test("search embeds the query before taking the lock, then returns the top k by cosine", async () => {
    const s = await openAt();
    await seed([{ text: "alpha alpha" }, { text: "bravo" }, { text: "alpha bravo" }, { text: "charlie" }]);
    const lockedWhileEmbedding: boolean[] = [];
    const watched = await openAt({
      embed: (texts, o) => {
        lockedWhileEmbedding.push(isLocked(`${path}.lock`));
        return embed(texts, o);
      },
    });
    const found = await call(watched, "search", { query: "alpha", k: 2n });
    expect(lockedWhileEmbedding).toEqual([false]);
    expect(embedded).toEqual([{ texts: ["alpha"], provider: "openai", model: "a" }]);
    expect(found).toEqual({
      items: [
        { id: 1n, version: 1n, score: 1, text: "alpha alpha" },
        { id: 3n, version: 3n, score: 0.7071, text: "alpha bravo" },
      ],
    });
    await s.close();
    await watched.close();
  });

  test("retired items aren't searched; equal scores keep the lower id first", async () => {
    const s = await openAt();
    await seed([{ text: "bravo", retired: "old" }, { text: "bravo" }, { text: "bravo bravo" }]);
    const found = await call(s, "search", { query: "bravo", k: 5n });
    expect((found.items as JsonObject[]).map((item) => item.id)).toEqual([2n, 3n]);
    await s.close();
  });

  test("the best items are found across chunks of the scan", async () => {
    const s = await openAt();
    const items: Seed[] = Array.from({ length: 2_500 }, (_, i) => ({ text: i % 2 === 0 ? "bravo" : "charlie" }));
    items[1_200] = { text: "delta" };
    items[2_400] = { text: "delta delta" };
    await seed(items);
    const found = await call(s, "search", { query: "delta", k: 2n });
    expect((found.items as JsonObject[]).map((item) => item.id)).toEqual([1_201n, 2_401n]);
    await s.close();
  });

  test("k past maxK, an empty query or one over maxTextChars is refused before anything is embedded", async () => {
    const s = await openAt({ maxK: 3, maxTextChars: 10 });
    const many = await rejection(call(s, "search", { query: "alpha", k: 4n }));
    expectRefusal(many, "k is 4; search returns at most 3 items");
    expectRefusal(await rejection(call(s, "search", { query: "", k: 1n })), "query is empty");
    expectRefusal(
      await rejection(call(s, "search", { query: "alpha bravo charlie", k: 1n })),
      "the query is 19 characters; it may be at most 10",
    );
    expect(embedded).toEqual([]);
    await expect(call(s, "search", { query: "alpha", k: 0n })).rejects.toThrow(TypeError);
    await expect(call(s, "search", { query: "alpha", k: "2" })).rejects.toThrow(TypeError);
    await s.close();
  });

  test("filter searches only the items whose fields equal its values", async () => {
    const typed = join(dir, "typed.db");
    const s = await openAt({ path: typed }, CASE);
    await seed(
      [
        { text: "alpha", fields: { title: "one", tags: ["x"] } },
        { text: "alpha alpha", fields: { title: "two", tags: ["x", "y"] } },
        { text: "alpha bravo", fields: { title: "one", tags: ["y"] } },
      ],
      typed,
    );
    const byTitle = await call(s, "search", { query: "alpha", k: 5n, filter: { title: "one" } });
    expect(byTitle.items).toEqual([
      { id: 1n, version: 1n, score: 1, text: "alpha", fields: { title: "one", tags: ["x"] } },
      { id: 3n, version: 3n, score: 0.7071, text: "alpha bravo", fields: { title: "one", tags: ["y"] } },
    ]);
    const byTags = await call(s, "search", { query: "alpha", k: 5n, filter: { tags: ["x", "y"] } });
    expect((byTags.items as JsonObject[]).map((item) => item.id)).toEqual([2n]);
    await s.close();
    const text = await openAt();
    await expect(call(text, "search", { query: "alpha", k: 1n, filter: { title: "one" } })).rejects.toThrow(
      "this memory's items are text only; they have no fields to filter",
    );
    await text.close();
  });

  test("a query embedded with another length than the items' is a fault naming both", async () => {
    await (await openAt()).close();
    await seed([{ text: "alpha" }]);
    const s = await openAt({ embed: async (texts) => texts.map(() => new Float32Array(3)) });
    await expect(call(s, "search", { query: "alpha", k: 1n })).rejects.toThrow(
      "the query's embedding has 3 dimensions, but this memory's items have 4; re-embed it with `yian memory reembed`",
    );
    await s.close();
  });

  test("an embedder that returns no vector, or a vector that isn't finite, is a fault", async () => {
    const empty = await openAt({ embed: async () => [] });
    await expect(call(empty, "search", { query: "alpha", k: 1n })).rejects.toThrow(
      "the embedder returned 0 vectors for 1 text",
    );
    await empty.close();
    const nan = await openAt({ embed: async () => [new Float32Array([Number.NaN])] });
    await expect(call(nan, "search", { query: "alpha", k: 1n })).rejects.toThrow(
      "the embedder returned a vector that isn't a non-empty list of finite numbers",
    );
    await nan.close();
  });

  test("a failed embed rejects the search with its error; without embed, search says how to give one", async () => {
    const failing = await openAt({ embed: () => Promise.reject(new Error("429 slow down")) });
    await expect(call(failing, "search", { query: "alpha", k: 1n })).rejects.toThrow("429 slow down");
    await failing.close();
    const none = await openAt({ embed: undefined });
    await expect(call(none, "search", { query: "alpha", k: 1n })).rejects.toThrow(
      "this VectorDBMemory has no embed function; build it with one to search or write",
    );
    await none.close();
  });
});

describe("read and list", () => {
  test("read returns one item with its author, and why it was retired", async () => {
    const typed = join(dir, "typed.db");
    const s = await openAt({ path: typed }, CASE);
    await seed([{ text: "alpha", fields: { title: "one", tags: [] }, retired: "a duplicate" }], typed);
    expect(await call(s, "read", { id: 1n })).toEqual({
      id: 1n,
      version: 1n,
      text: "alpha",
      fields: { title: "one", tags: [] },
      author: { kind: "host", name: "seed" },
      retired: "a duplicate",
    });
    expectRefusal(await rejection(call(s, "read", { id: 9n })), "item 9 doesn't exist");
    await expect(call(s, "read", {})).rejects.toThrow(TypeError);
    await s.close();
  });

  test("list lists the newest live items first, at most maxK, saying how many more there are", async () => {
    const s = await openAt({ maxK: 2 });
    await seed([{ text: "alpha" }, { text: "bravo", retired: "x" }, { text: "charlie" }, { text: "delta" }]);
    expect(await call(s, "list", {})).toEqual({
      items: [
        { id: 4n, version: 4n, text: "delta" },
        { id: 3n, version: 3n, text: "charlie" },
      ],
      cut: "1 more; search them",
    });
    expect(await call(s, "list", { recent: 1n })).toEqual({
      items: [{ id: 4n, version: 4n, text: "delta" }],
      cut: "2 more; search them",
    });
    expect(await call(s, "list", { recent: 9n })).toMatchObject({ cut: "1 more; search them" });
    await s.close();
  });

  test("a session refuses a store older than one it saw, as one restored from a copy is", async () => {
    const s = await openAt();
    await apply(s, { op: "add", text: "alpha" }, "a1");
    fs.copyFileSync(path, join(dir, "copy.db"));
    // Its own write is the newest the session saw, with no read after it.
    await apply(s, { op: "add", text: "bravo" }, "a2");
    fs.copyFileSync(join(dir, "copy.db"), path);
    await expect(call(s, "list", {})).rejects.toThrow(
      "this memory's store is older than this session saw; it may have been restored from a copy",
    );
    // A session opened on the restored store takes it as it is.
    const fresh = await openAt();
    expect((await fresh.snapshot()).version).toBe("1");
    await fresh.close();
    await s.close();
  });

  test("a write tool or an unknown tool isn't a read tool", async () => {
    const s = await openAt();
    await expect(call(s, "add", { text: "x" })).rejects.toThrow("add is a write tool: apply it as a change");
    await expect(call(s, "upsert", {})).rejects.toThrow(
      "VectorDBMemory has no tool upsert; it has search, read, list, add, edit, retire, delete",
    );
    await s.close();
  });

  test("a session notices a store re-embedded or replaced while it was open, and names no location", async () => {
    const s = await openAt();
    const db = openSqlite(path);
    db.prepare("UPDATE meta SET value = ? WHERE key = 'settings'").run(
      '{"embedding":{"model":"b","provider":"openai"},"item_schema":null}',
    );
    db.close();
    await expect(call(s, "list", {})).rejects.toThrow(
      "this memory was re-embedded with openai/b while this session had it open; open it again",
    );
    fs.rmSync(path);
    await expect(call(s, "list", {})).rejects.toThrow("this memory's store file is gone");
    await (await openAt({ id: "t/new" })).close();
    const replaced = (await rejection(call(s, "list", {}))) as Error;
    expect(replaced.message).toBe(
      "this memory's store now holds `t/new`, not `t/cases`: it was replaced while this session had it open",
    );
    expect(`${replaced.message} ${replaced.stack}`).not.toContain(dir);
    await s.close();
  });
});

const author: Author = { kind: "tool", name: "save", run: "r", node: "save" };

/** Apply `change` as change `id` on `s`. */
function apply(s: MemorySession, change: JsonObject, id = "c1"): Promise<Applied> {
  return s.apply({ id, change, author, signal });
}

/** Return the rows of `table` in the store, read straight from the file. */
function rows(table: string, at = path): JsonObject[] {
  const db = openSqlite(at, { readonly: true });
  try {
    return db.prepare(`SELECT * FROM ${table}`).all() as unknown as JsonObject[];
  } finally {
    db.close();
  }
}

/** Return whether any of the store's files holds `text` in its bytes. */
function storeHolds(text: string): boolean {
  return fs
    .readdirSync(join(dir, "store"))
    .some((name) => fs.readFileSync(join(dir, "store", name)).includes(Buffer.from(text, "utf8")));
}

describe("writes", () => {
  test("add never conflicts and returns {id, version}; the item carries its author", async () => {
    const s = await openAt();
    expect(await apply(s, { op: "add", text: "alpha" }, "a1")).toEqual({
      ok: true,
      version: "1",
      result: { id: 1n, version: 1n },
    });
    expect(await apply(s, { op: "add", text: "bravo" }, "a2")).toEqual({
      ok: true,
      version: "2",
      result: { id: 2n, version: 2n },
    });
    expect(await call(s, "read", { id: 2n })).toEqual({ id: 2n, version: 2n, text: "bravo", author });
    expect(embedded.map((each) => each.texts)).toEqual([["alpha"], ["bravo"]]);
    expect((await s.snapshot()).version).toBe("2");
    await s.close();
  });

  test("a typed memory's add needs fields that fit the item type; a text-only one takes none", async () => {
    const typed = await openAt({ path: join(dir, "typed.db") }, CASE);
    const fields = { title: "flaky", tags: ["ci"] };
    expect((await apply(typed, { op: "add", text: "alpha", fields }, "t1")).ok).toBe(true);
    expect(await call(typed, "read", { id: 1n })).toMatchObject({ fields });
    await expect(apply(typed, { op: "add", text: "alpha" }, "t2")).rejects.toThrow(
      "an add to this memory needs fields, since its items are typed",
    );
    await expect(apply(typed, { op: "add", text: "alpha", fields: { title: 3n, tags: [] } }, "t3")).rejects.toThrow(
      "a VectorDBMemory add's fields don't fit the item type",
    );
    await typed.close();
    const text = await openAt();
    await expect(apply(text, { op: "add", text: "alpha", fields }, "t4")).rejects.toThrow(
      "this memory's items are text only; an add takes no fields",
    );
    await text.close();
  });

  test("edit changes the text, the fields or both, and embeds only a new text", async () => {
    const s = await openAt({ path: join(dir, "typed.db") }, CASE);
    await apply(s, { op: "add", text: "alpha", fields: { title: "a", tags: [] } }, "a1");
    embedded = [];
    const fields = { title: "b", tags: ["x"] };
    expect(await apply(s, { op: "edit", id: 1n, if_version: 1n, fields }, "e1")).toEqual({
      ok: true,
      version: "2",
      result: { id: 1n, version: 2n },
    });
    expect(embedded).toEqual([]);
    expect(await apply(s, { op: "edit", id: 1n, if_version: 2n, text: "bravo" }, "e2")).toMatchObject({ ok: true });
    expect(embedded.map((each) => each.texts)).toEqual([["bravo"]]);
    expect(await call(s, "read", { id: 1n })).toMatchObject({ version: 3n, text: "bravo", fields });
    const found = await call(s, "search", { query: "bravo", k: 1n });
    expect(found.items).toEqual([{ id: 1n, version: 3n, score: 1, text: "bravo", fields }]);
    await expect(apply(s, { op: "edit", id: 1n, if_version: 3n }, "e3")).rejects.toThrow(
      "an edit needs text, fields or both",
    );
    await s.close();
  });

  test("edit, retire and delete with a stale if_version conflict with the item's current state", async () => {
    const s = await openAt();
    await apply(s, { op: "add", text: "alpha" }, "a1");
    await apply(s, { op: "edit", id: 1n, if_version: 1n, text: "alpha bravo" }, "e1");
    embedded = [];
    const current = { id: 1n, version: 2n, text: "alpha bravo", author };
    const reason = "item 1 changed since it was read: its version is 2, not 1";
    const stale: JsonObject[] = [
      { op: "edit", id: 1n, if_version: 1n, text: "charlie" },
      { op: "retire", id: 1n, if_version: 1n, why: "old" },
      { op: "delete", id: 1n, if_version: 1n },
    ];
    for (const change of stale) {
      expect(await apply(s, change, `stale-${String(change.op)}`)).toEqual({ ok: false, reason, current });
    }
    // A conflict is found before anything is embedded, and writes nothing.
    expect(embedded).toEqual([]);
    expect((await s.snapshot()).version).toBe("2");
    expect(await apply(s, { op: "edit", id: 9n, if_version: 1n, text: "x" }, "missing")).toEqual({
      ok: false,
      reason: "item 9 doesn't exist",
      current: null,
    });
    await s.close();
  });

  test("retire hides an item from search, list and the recall, and keeps it for read", async () => {
    const s = await openAt({ recallRecent: 5 });
    await apply(s, { op: "add", text: "alpha" }, "a1");
    await apply(s, { op: "add", text: "alpha bravo" }, "a2");
    expect(await apply(s, { op: "retire", id: 1n, if_version: 1n, why: "a duplicate" }, "r1")).toMatchObject({
      ok: true,
      result: { id: 1n, version: 3n },
    });
    expect(((await call(s, "search", { query: "alpha", k: 5n })).items as JsonObject[]).map((i) => i.id)).toEqual([2n]);
    expect(((await call(s, "list", {})).items as JsonObject[]).map((i) => i.id)).toEqual([2n]);
    expect((await s.snapshot()).recall.text).toBe("1 item\n[2] alpha bravo");
    expect(await call(s, "read", { id: 1n })).toMatchObject({ version: 3n, retired: "a duplicate" });
    expect(await apply(s, { op: "retire", id: 1n, if_version: 3n, why: "again" }, "r2")).toMatchObject({
      ok: false,
      reason: "item 1 is already retired",
    });
    await s.close();
  });

  test("delete erases the item's text, its vector and its history rows from the store", async () => {
    const s = await openAt();
    await apply(s, { op: "add", text: "secret alpha" }, "a1");
    await apply(s, { op: "edit", id: 1n, if_version: 1n, text: "secret bravo" }, "e1");
    await apply(s, { op: "add", text: "kept" }, "a2");
    expect(storeHolds("secret alpha")).toBe(true);
    expect(await apply(s, { op: "delete", id: 1n, if_version: 2n }, "d1")).toEqual({
      ok: true,
      version: "4",
      result: { id: 1n, version: 4n },
    });
    expectRefusal(await rejection(call(s, "read", { id: 1n })), "item 1 doesn't exist");
    expect(rows("items").map((row) => row.id)).toEqual([2]);
    expect(rows("history").filter((row) => row.item === 1).map((row) => row.op)).toEqual(["delete"]);
    expect(storeHolds("secret alpha")).toBe(false);
    expect(storeHolds("secret bravo")).toBe(false);
    expect(storeHolds("kept")).toBe(true);
    await s.close();
  });

  test("an add over maxItems, or a text over maxTextChars, is refused with the reason before embedding", async () => {
    const s = await openAt({ maxItems: 2, maxTextChars: 5 });
    await apply(s, { op: "add", text: "alpha" }, "a1");
    await apply(s, { op: "add", text: "bravo" }, "a2");
    await apply(s, { op: "retire", id: 2n, if_version: 2n, why: "x" }, "r1");
    embedded = [];
    // A retired item still takes its place.
    expect(await apply(s, { op: "add", text: "delta" }, "a3")).toEqual({
      ok: false,
      reason: "the memory holds 2 items; it may hold 2",
      current: null,
    });
    expect(await apply(s, { op: "edit", id: 1n, if_version: 1n, text: "charlie" }, "e1")).toEqual({
      ok: false,
      reason: "the text is 7 characters; an item may hold 5",
      current: null,
    });
    expect(await apply(s, { op: "edit", id: 1n, if_version: 1n, text: "" }, "e2")).toEqual({
      ok: false,
      reason: "an item's text can't be empty",
      current: null,
    });
    expect(embedded).toEqual([]);
    expect((await s.snapshot()).version).toBe("3");
    await s.close();
  });

  test("a failed embed writes nothing, and the same id applies once embedding works", async () => {
    let fail = true;
    const s = await openAt({
      embed: (texts, o) => (fail ? Promise.reject(new Error("503 unavailable")) : embed(texts, o)),
    });
    await expect(apply(s, { op: "add", text: "alpha" }, "a1")).rejects.toThrow("503 unavailable");
    expect(rows("items")).toEqual([]);
    expect(rows("changes")).toEqual([]);
    expect((await s.snapshot()).version).toBe("0");
    fail = false;
    expect(await apply(s, { op: "add", text: "alpha" }, "a1")).toMatchObject({ ok: true, version: "1" });
    await s.close();
  });

  test("a vector of another length than the store's is a fault that writes nothing", async () => {
    await (await openAt()).close();
    const s = await openAt();
    await apply(s, { op: "add", text: "alpha" }, "a1");
    const short = await openAt({ embed: async (texts) => texts.map(() => new Float32Array([1, 0, 0])) });
    await expect(apply(short, { op: "add", text: "bravo" }, "a2")).rejects.toThrow(
      "the embedder returned vectors of 3 dimensions, but this memory's items have 4; " +
        "re-embed it with `yian memory reembed`",
    );
    expect(rows("items")).toHaveLength(1);
    await s.close();
    await short.close();
  });

  test("a known id returns its stored result without embedding again; another change under it is refused", async () => {
    const s = await openAt();
    const first = await apply(s, { op: "add", text: "alpha" }, "a1");
    embedded = [];
    expect(await apply(s, { op: "add", text: "alpha" }, "a1")).toEqual(first);
    expect(embedded).toEqual([]);
    expect(rows("items")).toHaveLength(1);
    const reused = await rejection(apply(s, { op: "add", text: "bravo" }, "a1"));
    expect(reused).toBeInstanceOf(ChangeIdReusedError);
    expect(memoryErrorKind(reused)).toBe("change_id_reused");
    expect(embedded).toEqual([]);
    // An int given as a number is the same change as one given as a bigint.
    const retire = { op: "retire", id: 1n, if_version: 1n, why: "x" };
    const retired = await apply(s, retire, "r1");
    expect(await apply(s, { ...retire, id: 1, if_version: 1 }, "r1")).toEqual(retired);
    await s.close();
  });

  test("a conflict records no id: the same id applies once its precondition holds", async () => {
    const s = await openAt();
    await apply(s, { op: "add", text: "alpha" }, "a1");
    await apply(s, { op: "edit", id: 1n, if_version: 1n, text: "bravo" }, "e1");
    const change = { op: "retire", id: 1n, if_version: 3n, why: "x" };
    expect((await apply(s, change, "r1")).ok).toBe(false);
    await apply(s, { op: "edit", id: 1n, if_version: 2n, text: "charlie" }, "e2");
    expect(await apply(s, change, "r1")).toMatchObject({ ok: true, version: "4" });
    await s.close();
  });

  test("after maxChangeIds later changes, an old id applies again", async () => {
    const s = await openAt({ maxChangeIds: 2 });
    await apply(s, { op: "add", text: "alpha" }, "a1");
    await apply(s, { op: "add", text: "bravo" }, "a2");
    expect(rows("changes").map((row) => row.id)).toEqual(["a1", "a2"]);
    await apply(s, { op: "add", text: "charlie" }, "a3");
    expect(rows("changes").map((row) => row.id)).toEqual(["a2", "a3"]);
    expect(await apply(s, { op: "add", text: "alpha" }, "a1")).toMatchObject({ ok: true, result: { id: 4n } });
    await s.close();
  });

  test("the lock is released while a change embeds, and its precondition is checked again after", async () => {
    const s = await openAt();
    const other = await openAt();
    await apply(s, { op: "add", text: "alpha" }, "a1");
    const locked: boolean[] = [];
    const racing = await openAt({
      embed: async (texts, o) => {
        locked.push(isLocked(`${path}.lock`));
        // Another writer gets the lock while this change embeds, and moves the item on.
        await apply(other, { op: "edit", id: 1n, if_version: 1n, text: "bravo" }, "e-other");
        return embed(texts, o);
      },
    });
    expect(await apply(racing, { op: "edit", id: 1n, if_version: 1n, text: "charlie" }, "e-mine")).toEqual({
      ok: false,
      reason: "item 1 changed since it was read: its version is 2, not 1",
      current: { id: 1n, version: 2n, text: "bravo", author },
    });
    expect(locked).toEqual([false]);
    for (const each of [s, other, racing]) await each.close();
  });

  test("one id sent by two sessions at once is applied once; both get its result", async () => {
    const a = await openAt();
    const b = await openAt();
    const change = { op: "add", text: "alpha" };
    const [first, second] = await Promise.all([apply(a, change, "same"), apply(b, change, "same")]);
    expect(first).toEqual(second!);
    expect(rows("items")).toHaveLength(1);
    // Both embedded, since neither had applied when the other checked the id.
    expect(embedded).toHaveLength(2);
    await a.close();
    await b.close();
  });

  test("a change writes its item, history row and id row in one transaction; a failure writes none", async () => {
    const s = await openAt();
    await apply(s, { op: "add", text: "alpha" }, "a1");
    expect(rows("history")).toEqual([
      {
        seq: 1,
        change: "a1",
        item: 1,
        op: "add",
        before: '{"absent":true}',
        after: 1,
        author: '{"kind":"tool","name":"save","node":"save","run":"r"}',
        at: expect.any(String),
        undoes: null,
      },
    ]);
    expect(rows("changes")).toEqual([
      { id: "a1", digest: expect.any(String), result: expect.stringContaining('"ok":true'), seq: 1 },
    ]);
    // A store whose history table can't take the row: the item and id rows roll back with it.
    const db = openSqlite(path);
    db.exec("CREATE TRIGGER no_history BEFORE INSERT ON history BEGIN SELECT RAISE(ABORT, 'no history'); END");
    db.close();
    await expect(apply(s, { op: "add", text: "bravo" }, "a2")).rejects.toThrow("no history");
    expect(rows("items")).toHaveLength(1);
    expect(rows("changes")).toHaveLength(1);
    expect(metaRows().version).toBe("1");
    await s.close();
  });

  test("a change of the wrong shape is a fault, not a refusal", async () => {
    const s = await openAt();
    await expect(apply(s, { op: "upsert", text: "x" })).rejects.toThrow(
      "VectorDBMemory has no change upsert; its changes are add, edit, retire, delete",
    );
    await expect(apply(s, { op: "add", text: "x", id: 1n })).rejects.toThrow(
      "a VectorDBMemory add takes text and fields, not id",
    );
    await expect(apply(s, { op: "retire", id: 1n, if_version: 1n })).rejects.toThrow(
      "a VectorDBMemory retire needs why as a string",
    );
    await expect(apply(s, { op: "delete", id: "1", if_version: 1n })).rejects.toThrow(TypeError);
    await expect(s.apply({ id: "", change: { op: "add", text: "x" }, author, signal })).rejects.toThrow(TypeError);
    await expect(
      s.apply({ id: "x", change: { op: "add", text: "x" }, author: { kind: "robot" } as unknown as Author, signal }),
    ).rejects.toThrow(TypeError);
    expect(rows("changes")).toEqual([]);
    await s.close();
  });

  test("close waits for a change that is embedding, which then lands", async () => {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const s = await openAt({
      embed: async (texts, o) => {
        await gate;
        return embed(texts, o);
      },
    });
    const adding = apply(s, { op: "add", text: "alpha" }, "a1");
    let closed = false;
    const closing = s.close().then(() => {
      closed = true;
    });
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(closed).toBe(false);
    release();
    await closing;
    expect(await adding).toMatchObject({ ok: true });
    expect(rows("items")).toHaveLength(1);
  });
});

/** The children a test started, killed after it if still running. */
const children: ChildProcess[] = [];

afterEach(() => {
  for (const child of children.splice(0)) {
    if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
  }
});

/**
 * Run a Bun child that opens the store at `path` and adds `count` items as ids `<who>-<i>`, with an
 * embedder that gives every text the same vector, and resolve with its exit code and stderr. A
 * child still running after 60 s is killed, so a hung lock can't leave it behind.
 */
function adder(who: string, count: number): Promise<{ code: number | null; stderr: string }> {
  const source = new URL("../../src/memory/vectordb.ts", import.meta.url).href;
  const script = `
const { VectorDBMemory } = await import(${JSON.stringify(source)});
const signal = new AbortController().signal;
const embed = async (texts) => texts.map(() => new Float32Array([1, 0, 0, 0]));
const m = new VectorDBMemory({ path: ${JSON.stringify(path)}, embed });
const s = await m.open({ create: false, itemType: null, signal });
const who = ${JSON.stringify(who)};
for (let i = 0; i < ${count}; i++) {
  const change = { op: "add", text: who + " " + i };
  const applied = await s.apply({ id: who + "-" + i, change, author: { kind: "host", name: who }, signal });
  if (!applied.ok) throw new Error(JSON.stringify(applied));
}
await s.close();
`;
  const child = spawn(process.execPath, ["-e", script], { stdio: ["ignore", "ignore", "pipe"], env: {} });
  children.push(child);
  let stderr = "";
  child.stderr!.setEncoding("utf8").on("data", (chunk: string) => {
    stderr += chunk;
  });
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error(`${who} still ran after 60 s; stderr:\n${stderr}`));
    }, 60_000);
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      resolve({ code, stderr });
    });
  });
}

test("writers in separate processes serialize through the lock: every add lands once, none is busy", async () => {
  await (await openAt()).close();
  const ran = await Promise.all([adder("p", 8), adder("q", 8)]);
  for (const { code, stderr } of ran) expect({ code, stderr }).toEqual({ code: 0, stderr: "" });
  const texts = rows("items").map((row) => row.text as string);
  const expected = ["p", "q"].flatMap((who) => Array.from({ length: 8 }, (_, i) => `${who} ${i}`));
  expect([...texts].sort()).toEqual([...expected].sort());
  expect(rows("changes")).toHaveLength(16);
  expect(metaRows().version).toBe("16");
}, 120_000);

const user: Author = { kind: "user", name: "ada" };

/** Undo change `changeId` on `s` as change `id`, by a person at the CLI. */
function undo(s: MemorySession, changeId: string, id = `undo-${changeId}`): Promise<Applied> {
  return s.undo({ id, changeId, author: user, signal });
}

/** Erase item `item` on `s` as change `id`, by a person at the CLI. */
function erase(s: MemorySession, item: bigint, id = `erase-${item}`) {
  return s.erase({ id, target: { item }, author: user, signal });
}

describe("undo", () => {
  test("undo restores an item when its version is still the change's, as a change of its own", async () => {
    const s = await openAt();
    await apply(s, { op: "add", text: "alpha" }, "a1");
    await apply(s, { op: "edit", id: 1n, if_version: 1n, text: "bravo" }, "e1");
    embedded = [];
    expect(await undo(s, "e1")).toEqual({ ok: true, version: "3", result: { id: 1n, version: 3n } });
    expect(await call(s, "read", { id: 1n })).toEqual({ id: 1n, version: 3n, text: "alpha", author: user });
    // The restored text is embedded again, outside the lock, and search finds it by it.
    expect(embedded.map((each) => each.texts)).toEqual([["alpha"]]);
    expect(((await call(s, "search", { query: "alpha", k: 1n })).items as JsonObject[])[0]).toMatchObject({ score: 1 });
    // An undo is a change: it can be undone too.
    expect(await undo(s, "undo-e1")).toMatchObject({ ok: true, version: "4" });
    expect(await call(s, "read", { id: 1n })).toMatchObject({ text: "bravo" });
    await s.close();
  });

  test("undo of a retire brings the item back with the vector it kept", async () => {
    const s = await openAt();
    await apply(s, { op: "add", text: "alpha" }, "a1");
    await apply(s, { op: "retire", id: 1n, if_version: 1n, why: "old" }, "r1");
    embedded = [];
    expect((await undo(s, "r1")).ok).toBe(true);
    expect(embedded).toEqual([]);
    expect(await call(s, "read", { id: 1n })).not.toHaveProperty("retired");
    expect(((await call(s, "search", { query: "alpha", k: 1n })).items as JsonObject[]).length).toBe(1);
    await s.close();
  });

  test("undo of an add removes the item; undoing that adds it back under its id", async () => {
    const s = await openAt({ path: join(dir, "store", "typed.db") }, CASE);
    const fields = { title: "t", tags: ["x"] };
    await apply(s, { op: "add", text: "alpha", fields }, "a1");
    await apply(s, { op: "add", text: "bravo", fields }, "a2");
    expect(await undo(s, "a1")).toEqual({ ok: true, version: "3", result: { id: 1n, version: 3n } });
    expectRefusal(await rejection(call(s, "read", { id: 1n })), "item 1 doesn't exist");
    expect(await undo(s, "undo-a1")).toMatchObject({ ok: true, version: "4" });
    expect(await call(s, "read", { id: 1n })).toEqual({ id: 1n, version: 4n, text: "alpha", fields, author: user });
    // A new item still gets a new id: ids are never used twice.
    expect(await apply(s, { op: "add", text: "charlie", fields }, "a3")).toMatchObject({ result: { id: 3n } });
    await s.close();
  });

  test("undo conflicts when the item changed after that change, writing nothing", async () => {
    const s = await openAt();
    await apply(s, { op: "add", text: "alpha" }, "a1");
    await apply(s, { op: "edit", id: 1n, if_version: 1n, text: "bravo" }, "e1");
    await apply(s, { op: "edit", id: 1n, if_version: 2n, text: "charlie" }, "e2");
    expect(await undo(s, "e1")).toEqual({
      ok: false,
      reason: "item 1 changed after change `e1`; undoing it would lose that",
      current: { id: 1n, version: 3n, text: "charlie", author },
    });
    expect((await s.snapshot()).version).toBe("3");
    await s.close();
  });

  test("undo of a change the history doesn't hold, a delete, or one past maxUndo is refused", async () => {
    const s = await openAt({ maxUndo: 2 });
    await apply(s, { op: "add", text: "alpha" }, "a1");
    await apply(s, { op: "add", text: "bravo" }, "a2");
    await apply(s, { op: "add", text: "charlie" }, "a3");
    await apply(s, { op: "delete", id: 3n, if_version: 3n }, "d1");
    expect(await undo(s, "nope")).toEqual(refusedWith("change `nope` isn't in the memory's history"));
    expect(await undo(s, "d1")).toEqual(refusedWith("change `d1` was a delete, which can't be undone"));
    // Only the last two changes keep what they replaced.
    expect(await undo(s, "a1")).toEqual(
      refusedWith("change `a1` can't be undone: the memory no longer keeps what it replaced"),
    );
    expect(rows("history").map((row) => [row.change, row.before === null])).toEqual([
      ["a1", true],
      ["a2", true],
      ["d1", true],
    ]);
    await s.close();
  });

  test("an undo just past maxUndo says the memory no longer keeps what it replaced, with few ids kept", async () => {
    const s = await openAt({ maxUndo: 1, maxChangeIds: 1 });
    await apply(s, { op: "add", text: "alpha" }, "a1");
    await apply(s, { op: "edit", id: 1n, if_version: 1n, text: "bravo" }, "e1");
    await apply(s, { op: "add", text: "charlie" }, "a2");
    expect(await undo(s, "e1")).toEqual(
      refusedWith("change `e1` can't be undone: the memory no longer keeps what it replaced"),
    );
    // Past both windows, the change is gone from the history.
    await apply(s, { op: "add", text: "delta" }, "a3");
    expect(await undo(s, "e1")).toEqual(refusedWith("change `e1` isn't in the memory's history"));
    expect(rows("history").map((row) => row.change)).toEqual(["a2", "a3"]);
    await s.close();
  });

  test("with maxUndo 0 no old text is ever written, and nothing can be undone", async () => {
    const s = await openAt({ maxUndo: 0 });
    await apply(s, { op: "add", text: "alpha" }, "a1");
    await apply(s, { op: "edit", id: 1n, if_version: 1n, text: "secret old" }, "e1");
    await apply(s, { op: "edit", id: 1n, if_version: 2n, text: "bravo" }, "e2");
    expect(rows("history").every((row) => row.before === null)).toBe(true);
    expect(storeHolds("secret old")).toBe(false);
    expect((await undo(s, "e2")).ok).toBe(false);
    await s.close();
  });

  test("an undo's id answers again with its stored result; another undo under it is refused", async () => {
    const s = await openAt();
    await apply(s, { op: "add", text: "alpha" }, "a1");
    await apply(s, { op: "edit", id: 1n, if_version: 1n, text: "bravo" }, "e1");
    const first = await undo(s, "e1", "u1");
    embedded = [];
    expect(await undo(s, "e1", "u1")).toEqual(first);
    expect(embedded).toEqual([]);
    await expect(undo(s, "a1", "u1")).rejects.toBeInstanceOf(ChangeIdReusedError);
    await s.close();
  });

  test("an undo writes nothing when the text to restore changed while it embedded the old one", async () => {
    const s = await openAt();
    await apply(s, { op: "add", text: "alpha" }, "a1");
    await apply(s, { op: "edit", id: 1n, if_version: 1n, text: "bravo" }, "e1");
    const edited = await openAt({
      embed: async (texts, o) => {
        // A hand edit of the store between the passes: the change now restores another text.
        const db = openSqlite(path);
        db.prepare("UPDATE history SET before = ? WHERE change = 'e1'").run(
          '{"fields":null,"retired":null,"text":"charlie"}',
        );
        db.close();
        return embed(texts, o);
      },
    });
    expect(await undo(edited, "e1")).toEqual(
      refusedWith("item 1 changed while its old text was embedded; undo change `e1` again"),
    );
    expect(await call(s, "read", { id: 1n })).toMatchObject({ version: 2n, text: "bravo" });
    await s.close();
    await edited.close();
  });

  test("an undo releases the lock while it embeds, and checks the item again after", async () => {
    const s = await openAt();
    const other = await openAt();
    await apply(s, { op: "add", text: "alpha" }, "a1");
    await apply(s, { op: "edit", id: 1n, if_version: 1n, text: "bravo" }, "e1");
    const racing = await openAt({
      embed: async (texts, o) => {
        await apply(other, { op: "edit", id: 1n, if_version: 2n, text: "charlie" }, "e-other");
        return embed(texts, o);
      },
    });
    expect(await undo(racing, "e1")).toMatchObject({
      ok: false,
      reason: "item 1 changed after change `e1`; undoing it would lose that",
    });
    for (const each of [s, other, racing]) await each.close();
  });
});

/** Return the refusal `reason`, which wrote nothing. */
function refusedWith(reason: string): Applied {
  return { ok: false, reason, current: null };
}

describe("erase, history and reset", () => {
  test("erase removes the item and its history rows, leaving no copy of its text; it names the undo lost", async () => {
    const s = await openAt();
    await apply(s, { op: "add", text: "secret alpha" }, "a1");
    await apply(s, { op: "edit", id: 1n, if_version: 1n, text: "secret bravo" }, "e1");
    await apply(s, { op: "add", text: "kept" }, "a2");
    expect(await erase(s, 1n)).toEqual({ version: "4", disabledUndos: ["e1"] });
    expect(rows("items").map((row) => row.text)).toEqual(["kept"]);
    expect(rows("history").map((row) => [row.change, row.op, row.item])).toEqual([
      ["a2", "add", 2],
      ["erase-1", "erase", 1],
    ]);
    expect(storeHolds("secret")).toBe(false);
    expect(fs.existsSync(`${path}-journal`)).toBe(false);
    await s.close();
  });

  test("erase scrubs the copy an undo of an add keeps, and disables that undo", async () => {
    const s = await openAt();
    await apply(s, { op: "add", text: "secret alpha" }, "a1");
    await undo(s, "a1");
    expect(storeHolds("secret alpha")).toBe(true);
    expect(await erase(s, 1n)).toEqual({ version: "3", disabledUndos: ["undo-a1"] });
    expect(storeHolds("secret alpha")).toBe(false);
    expect(await undo(s, "undo-a1")).toEqual(refusedWith("change `undo-a1` isn't in the memory's history"));
    await s.close();
  });

  test("an erase is in history, can't be undone, and its id answers again with its stored result", async () => {
    const s = await openAt();
    await apply(s, { op: "add", text: "alpha" }, "a1");
    const erased = await erase(s, 1n, "x1");
    expect(await erase(s, 1n, "x1")).toEqual(erased);
    await expect(erase(s, 2n, "x1")).rejects.toBeInstanceOf(ChangeIdReusedError);
    expect(await undo(s, "x1")).toEqual(refusedWith("change `x1` was an erase, which can't be undone"));
    expect((await s.history({ limit: 1 }))[0]).toMatchObject({ id: "x1", op: "erase", target: { item: 1n } });
    await s.close();
  });

  test("erasing an item the memory holds nothing of is refused, recording nothing", async () => {
    const s = await openAt();
    await apply(s, { op: "add", text: "alpha" }, "a1");
    await expect(erase(s, 42n, "x1")).rejects.toThrow("item 42 isn't in the memory: there is nothing to erase");
    expect((await s.history({ limit: 5 })).map((entry) => entry.id)).toEqual(["a1"]);
    expect((await s.snapshot()).version).toBe("1");
    // The id wasn't used, so it erases an item that is there.
    expect(await erase(s, 1n, "x1")).toEqual({ version: "2", disabledUndos: ["a1"] });
    // A deleted item's delete is still in the history, so erasing it goes on.
    await apply(s, { op: "add", text: "bravo" }, "a2");
    await apply(s, { op: "delete", id: 2n, if_version: 3n }, "d2");
    expect(await erase(s, 2n, "x2")).toEqual({ version: "5", disabledUndos: [] });
    await s.close();
  });

  test("an erase target of the wrong shape is a fault", async () => {
    const s = await openAt();
    const bad = (target: JsonObject) => s.erase({ id: "x", target, author: user, signal });
    await expect(bad({})).rejects.toThrow("an erase needs item as an integer of at least 1");
    await expect(bad({ item: 1n, text: "x" })).rejects.toThrow("an erase takes item, not text");
    await expect(bad({ item: 0n })).rejects.toThrow(TypeError);
    await s.close();
  });

  test("history lists changes newest first, with author, run, node and what each undid", async () => {
    const s = await openAt({ scheduler: virtualScheduler(Date.UTC(2026, 10, 7)) });
    await apply(s, { op: "add", text: "alpha" }, "a1");
    await apply(s, { op: "retire", id: 1n, if_version: 1n, why: "x" }, "r1");
    await undo(s, "r1");
    const at = "2026-11-07T00:00:00.000Z";
    expect(await s.history({ limit: 10 })).toEqual([
      { id: "undo-r1", op: "undo", target: { item: 1n, undoes: "r1" }, author: user, at },
      { id: "r1", op: "retire", target: { item: 1n }, author, at },
      { id: "a1", op: "add", target: { item: 1n }, author, at },
    ]);
    expect((await s.history({ limit: 1 })).map((entry) => entry.id)).toEqual(["undo-r1"]);
    expect(await s.history({ limit: 0 })).toEqual([]);
    await expect(s.history({ limit: -1 })).rejects.toThrow(RangeError);
    await s.close();
  });

  test("reset empties the items, history and change ids, keeps meta, and moves the version on", async () => {
    const s = await openAt();
    await apply(s, { op: "add", text: "secret alpha" }, "a1");
    await apply(s, { op: "add", text: "bravo" }, "a2");
    const before = metaRows();
    await memory().reset({ signal });
    expect(rows("items")).toEqual([]);
    expect(rows("history")).toEqual([]);
    expect(rows("changes")).toEqual([]);
    expect(storeHolds("secret alpha")).toBe(false);
    expect(metaRows()).toEqual({ ...before, version: "3" });
    // A session open across the reset goes on, and an old id applies again.
    expect(await s.snapshot()).toEqual({ version: "3", recall: { text: "0 items", cut: null, hint: null } });
    expect(await apply(s, { op: "add", text: "alpha" }, "a1")).toMatchObject({ ok: true, result: { id: 3n } });
    await s.close();
  });

  test("reset refuses a store that isn't a memory, or is another one", async () => {
    await expect(memory().reset({ signal })).rejects.toThrow(`\`${path}\` isn't a memory yet`);
    expect(fs.existsSync(path)).toBe(false);
    await (await openAt()).close();
    await expect(memory({ id: "t/other" }).reset({ signal })).rejects.toThrow(
      "this memory is `t/cases`, not `t/other`",
    );
  });
});

/** An embedder for model `b`: two dimensions, `alpha` words in the second. */
async function embedB(texts: readonly string[], o: EmbeddingModel & { signal: AbortSignal }): Promise<Float32Array[]> {
  embedded.push({ texts: [...texts], provider: o.provider, model: o.model });
  return texts.map((text) => new Float32Array([1, text.split(" ").filter((word) => word === "alpha").length]));
}

const modelB: EmbeddingModel = { provider: "openai", model: "b" };

describe("host actions", () => {
  test("reembed embeds every item again in batches under the lock, then records the new model", async () => {
    const s = await openAt();
    for (const [i, text] of ["alpha", "bravo", "alpha alpha", "charlie", "delta"].entries()) {
      await apply(s, { op: "add", text }, `a${i}`);
    }
    await apply(s, { op: "retire", id: 4n, if_version: 4n, why: "x" }, "r1");
    await s.close();
    embedded = [];
    const locked: boolean[] = [];
    const watched = (texts: readonly string[], o: EmbeddingModel & { signal: AbortSignal }) => {
      locked.push(isLocked(`${path}.lock`));
      return embedB(texts, o);
    };
    const env = (name: string) => (name === "YIAN_EMBEDDING_BATCH_ITEMS" ? "2" : undefined);
    await withEnv(env, () => memory().reembed({ embedding: modelB, embed: watched, signal }));
    // Retired items too: an undo or a later edit may bring them back.
    expect(embedded).toEqual([
      { texts: ["alpha", "bravo"], provider: "openai", model: "b" },
      { texts: ["alpha alpha", "charlie"], provider: "openai", model: "b" },
      { texts: ["delta"], provider: "openai", model: "b" },
    ]);
    expect(locked).toEqual([true, true, true]);
    expect(JSON.parse(metaRows().settings!).embedding).toEqual({ provider: "openai", model: "b" });
    expect(rows("items").every((row) => (row.vector as unknown as Uint8Array).byteLength === 8)).toBe(true);
    // The version doesn't move: no item changed.
    expect(metaRows().version).toBe("6");

    await expect(openAt()).rejects.toThrow("this memory was embedded with openai/b");
    const reopened = await openAt({ embedding: modelB, embed: embedB });
    const found = await call(reopened, "search", { query: "alpha", k: 2n });
    expect((found.items as JsonObject[]).map((item) => item.id)).toEqual([1n, 3n]);
    await reopened.close();
  });

  test("a session open across a reembed refuses its next call, rather than mix two models' vectors", async () => {
    const s = await openAt();
    await apply(s, { op: "add", text: "alpha" }, "a1");
    await memory().reembed({ embedding: modelB, embed: embedB, signal });
    await expect(apply(s, { op: "add", text: "bravo" }, "a2")).rejects.toThrow(
      "this memory was re-embedded with openai/b while this session had it open; open it again",
    );
    await s.close();
  });

  test("a reembed whose embedder fails part way leaves the store as it was", async () => {
    const s = await openAt();
    await apply(s, { op: "add", text: "alpha" }, "a1");
    await apply(s, { op: "add", text: "bravo" }, "a2");
    await apply(s, { op: "add", text: "charlie" }, "a3");
    const before = rows("items");
    let calls = 0;
    const failing = async (texts: readonly string[], o: EmbeddingModel & { signal: AbortSignal }) => {
      if (++calls === 2) throw new Error("503 unavailable");
      return embedB(texts, o);
    };
    const env = (name: string) => (name === "YIAN_EMBEDDING_BATCH_ITEMS" ? "2" : undefined);
    await expect(withEnv(env, () => memory().reembed({ embedding: modelB, embed: failing, signal }))).rejects.toThrow(
      "503 unavailable",
    );
    expect(rows("items")).toEqual(before);
    expect(JSON.parse(metaRows().settings!).embedding).toEqual(model);
    // The session opened before goes on with the old model.
    expect(((await call(s, "search", { query: "alpha", k: 1n })).items as JsonObject[])[0]).toMatchObject({ id: 1n });
    // A later reembed starts over and finishes.
    await memory().reembed({ embedding: modelB, embed: embedB, signal });
    expect(JSON.parse(metaRows().settings!).embedding).toEqual(modelB);
    await s.close();
  });

  /** Leave a staging table holding a vector for each of `ids`, as a reembed a crash cut off does. */
  function crashedReembed(ids: number[]): void {
    const db = openSqlite(path);
    db.exec("CREATE TABLE reembed(id INTEGER PRIMARY KEY, vector BLOB NOT NULL)");
    for (const id of ids) db.prepare("INSERT INTO reembed(id, vector) VALUES (?, ?)").run(id, new Uint8Array(8));
    db.close();
  }

  /** Return whether the store holds a staging table. */
  function staged(): boolean {
    return openSqliteRows("SELECT name FROM sqlite_master WHERE name = 'reembed'").length > 0;
  }

  /** Return the rows `sql` reads from the store. */
  function openSqliteRows(sql: string): JsonObject[] {
    const db = openSqlite(path, { readonly: true });
    try {
      return db.prepare(sql).all() as unknown as JsonObject[];
    } finally {
      db.close();
    }
  }

  test.each([
    ["an erase", (s: MemorySession) => erase(s, 1n)],
    ["a delete", (s: MemorySession) => apply(s, { op: "delete", id: 1n, if_version: 1n }, "d1")],
    ["an add", (s: MemorySession) => apply(s, { op: "add", text: "bravo" }, "a2")],
  ] as const)("%s drops the staging table a crashed reembed left, with the vectors it held", async (_, write) => {
    const s = await openAt();
    await apply(s, { op: "add", text: "alpha" }, "a1");
    crashedReembed([1]);
    expect(staged()).toBe(true);
    await write(s);
    expect(staged()).toBe(false);
    await s.close();
  });

  test("reset drops the staging table a crashed reembed left", async () => {
    const s = await openAt();
    await apply(s, { op: "add", text: "alpha" }, "a1");
    crashedReembed([1]);
    await memory().reset({ signal });
    expect(staged()).toBe(false);
    await s.close();
  });

  test("an embedder that breaks its contract fails the reembed, writing nothing", async () => {
    await (await openAt()).close();
    const s = await openAt();
    await apply(s, { op: "add", text: "alpha" }, "a1");
    await s.close();
    await expect(memory().reembed({ embedding: modelB, embed: async () => [], signal })).rejects.toThrow(
      "the embedder returned 0 vectors for 1 text",
    );
    expect(JSON.parse(metaRows().settings!).embedding).toEqual(model);
  });

  test("reembed refuses a store that isn't a memory, or is another one, and bad arguments", async () => {
    await expect(memory().reembed({ embedding: modelB, embed: embedB, signal })).rejects.toThrow(
      `\`${path}\` isn't a memory yet`,
    );
    await (await openAt()).close();
    await expect(memory({ id: "t/other" }).reembed({ embedding: modelB, embed: embedB, signal })).rejects.toThrow(
      "this memory is `t/cases`, not `t/other`",
    );
    await expect(
      memory().reembed({ embedding: { provider: "openai" } as EmbeddingModel, embed: embedB, signal }),
    ).rejects.toThrow(TypeError);
  });

  test("compact runs VACUUM under the lock, so a store shrinks after its items go", async () => {
    const s = await openAt();
    const big = "alpha ".repeat(1_000);
    for (let i = 0; i < 30; i++) await apply(s, { op: "add", text: `${big}${i}` }, `a${i}`);
    for (let i = 1; i <= 30; i++) await apply(s, { op: "delete", id: BigInt(i), if_version: BigInt(i) }, `d${i}`);
    const before = fs.statSync(path).size;
    await memory().compact({ signal });
    expect(fs.statSync(path).size).toBeLessThan(before / 2);
    expect(fs.existsSync(`${path}-journal`)).toBe(false);
    expect((await s.snapshot()).version).toBe("60");

    const lock = lockFile(`${path}.lock`, { private: true });
    try {
      await expect(memory({ lockTimeoutMs: 0 }).compact({ signal })).rejects.toBeInstanceOf(MemoryBusyError);
    } finally {
      lock.release();
    }
    await s.close();
  });

  test("compact refuses a store that isn't a memory; neither host action is a session method", async () => {
    await expect(memory().compact({ signal })).rejects.toThrow(`\`${path}\` isn't a memory yet`);
    const s = await openAt();
    expect("reembed" in s).toBe(false);
    expect("compact" in s).toBe(false);
    await s.close();
  });
});

describe("the lock and the queue", () => {
  test("an abort while a call is queued rejects it at once; the calls before and after it still run", async () => {
    const s = await openAt({ lockTimeoutMs: 10_000 });
    const lock = lockFile(`${path}.lock`, { private: true });
    const first = s.snapshot();
    const controller = new AbortController();
    const queued = call(s, "list", {}).catch(() => "failed");
    const aborted = s.call("list", {}, { signal: controller.signal });
    controller.abort(new Error("stopped"));
    await expect(aborted).rejects.toThrow("stopped");
    lock.release();
    expect((await first).version).toBe("0");
    expect(await queued).toEqual({ items: [] });
    await s.close();
  });

  test("close waits for every call the session queued, even when its last one was aborted while queued", async () => {
    const s = await openAt({ lockTimeoutMs: 10_000 });
    const lock = lockFile(`${path}.lock`, { private: true });
    const first = s.snapshot();
    const controller = new AbortController();
    const aborted = s.call("list", {}, { signal: controller.signal }).catch(() => "aborted");
    controller.abort();
    expect(await aborted).toBe("aborted");
    let closed = false;
    const closing = s.close().then(() => {
      closed = true;
    });
    await new Promise((resolve) => setTimeout(resolve, 150));
    expect(closed).toBe(false);
    lock.release();
    await first;
    await closing;
    expect(closed).toBe(true);
  });

  test("close doesn't wait for the turn of a call that left the queue busy", async () => {
    const s = await openAt({ lockTimeoutMs: 200 });
    await apply(s, { op: "add", text: "alpha" }, "a1");
    const stuck = new AbortController();
    const never = () => new Promise<Float32Array[]>(() => {});
    const reembedding = memory().reembed({ embedding: modelB, embed: never, signal: stuck.signal });
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(await rejection(s.snapshot())).toBeInstanceOf(MemoryBusyError);
    // Every call of the session has returned; the reembed ahead of its busy call still holds the lock.
    const closed = await Promise.race([
      s.close().then(() => true),
      new Promise<boolean>((resolve) => setTimeout(() => resolve(false), 1_000)),
    ]);
    stuck.abort(new Error("stopped"));
    await expect(reembedding).rejects.toThrow("stopped");
    expect(closed).toBe(true);
  });

  test("with lockTimeoutMs: 0 one session's concurrent calls take turns, and none of them is busy", async () => {
    const s = await openAt({ lockTimeoutMs: 0 });
    const writes = ["alpha", "bravo", "charlie", "delta"].map((text, i) => apply(s, { op: "add", text }, `a${i}`));
    const reads = [s.snapshot(), call(s, "list", {}), s.history({ limit: 1 })];
    const results = await Promise.all([...writes, ...reads]);
    expect(results.slice(0, 4).every((result) => (result as Applied).ok)).toBe(true);
    await s.close();
  });

  test("a call queued behind a stuck reembed is busy at its deadline; aborting the reembed frees it", async () => {
    const s = await openAt({ lockTimeoutMs: 200 });
    await apply(s, { op: "add", text: "alpha" }, "a1");
    const stuck = new AbortController();
    const never = () => new Promise<Float32Array[]>(() => {});
    const host = memory({ lockTimeoutMs: 200 });
    const reembedding = host.reembed({ embedding: modelB, embed: never, signal: stuck.signal });
    await new Promise((resolve) => setTimeout(resolve, 50));
    const started = performance.now();
    const queued = await rejection(s.snapshot());
    expect(queued).toBeInstanceOf(MemoryBusyError);
    expect(performance.now() - started).toBeLessThan(2_000);
    // The embedder never answers, but the abort ends the reembed and lets go of the lock.
    stuck.abort(new Error("stopped"));
    await expect(reembedding).rejects.toThrow("stopped");
    expect(await s.snapshot()).toMatchObject({ version: "1" });
    expect(JSON.parse(metaRows().settings!).embedding).toEqual(model);
    await s.close();
  });

  test("an abort ends a change whose embedder never answers, so close isn't held up", async () => {
    const controller = new AbortController();
    const s = await openAt({ embed: () => new Promise<Float32Array[]>(() => {}) });
    const adding = s.apply({ id: "a1", change: { op: "add", text: "alpha" }, author, signal: controller.signal });
    await new Promise((resolve) => setTimeout(resolve, 20));
    controller.abort(new Error("stopped"));
    await expect(adding).rejects.toThrow("stopped");
    await s.close();
    expect(rows("items")).toEqual([]);
  });

  const asUser = process.platform !== "win32" && process.getuid!() !== 0;

  test.if(asUser)("an OS error from a call names no location", async () => {
    const s = await openAt();
    fs.chmodSync(join(dir, "store"), 0o600);
    let error: unknown;
    try {
      error = await rejection(call(s, "list", {}));
    } finally {
      fs.chmodSync(join(dir, "store"), 0o700);
    }
    expect((error as NodeJS.ErrnoException).code).toBe("EACCES");
    expect(`${(error as Error).message} ${(error as Error).stack}`).not.toContain(dir);
    expect(Object.values(error as object).join(" ")).not.toContain(dir);
    expect((error as Error).message).toContain("the memory's store");
    await s.close();
  });
});

describe("numbers that aren't ints", () => {
  /** An item type with a float field and a float bound. */
  const SCORED: JsonObject = {
    type: "object",
    properties: { title: { type: "string" }, score: { type: "number", minimum: 0.5 } },
    required: ["title", "score"],
    additionalProperties: false,
  };

  test("float fields add, edit, resend and filter; a float bound in the item schema opens again", async () => {
    const at = join(dir, "store", "scored.db");
    const s = await openAt({ path: at }, SCORED);
    const added = await apply(s, { op: "add", text: "alpha", fields: { title: "a", score: 0.5 } }, "a1");
    expect(added).toMatchObject({ ok: true, result: { id: 1n } });
    embedded = [];
    // The same change under its id answers with its stored result, embedding nothing.
    expect(await apply(s, { op: "add", text: "alpha", fields: { title: "a", score: 0.5 } }, "a1")).toEqual(added);
    expect(embedded).toEqual([]);
    const other = { op: "add", text: "alpha", fields: { title: "a", score: 0.75 } };
    await expect(apply(s, other, "a1")).rejects.toBeInstanceOf(ChangeIdReusedError);
    await apply(s, { op: "add", text: "alpha bravo", fields: { title: "b", score: 2n } }, "a2");
    const edit = { op: "edit", id: 1n, if_version: 1n, fields: { title: "a", score: 1.5 } };
    const edited = await apply(s, edit, "e1");
    expect(edited).toMatchObject({ ok: true });
    expect(await apply(s, edit, "e1")).toEqual(edited);
    expect(await call(s, "read", { id: 1n })).toMatchObject({ fields: { title: "a", score: 1.5 } });
    // A filter matches by value: an int equals the same number as a float, and floats compare as numbers.
    const ids = async (filter: JsonObject) =>
      ((await call(s, "search", { query: "alpha", k: 5n, filter })).items as JsonObject[]).map((item) => item.id);
    expect(await ids({ score: 1.5 })).toEqual([1n]);
    expect(await ids({ score: 2 })).toEqual([2n]);
    expect(await ids({ score: 2n })).toEqual([2n]);
    expect(await ids({ score: 0.5 })).toEqual([]);
    await s.close();
    await (await openAt({ path: at }, SCORED)).close();
    expect(JSON.parse(metaRows(at).settings!).item_schema.properties.score.minimum).toBe(0.5);
  });

  test("an edit whose fields are undefined changes its text alone", async () => {
    const s = await openAt();
    await apply(s, { op: "add", text: "alpha" }, "a1");
    const change = { op: "edit", id: 1n, if_version: 1n, text: "bravo", fields: undefined } as unknown as JsonObject;
    expect(await apply(s, change, "e1")).toMatchObject({
      ok: true,
    });
    expect(await call(s, "read", { id: 1n })).toMatchObject({ text: "bravo" });
    await s.close();
  });
});

describe("sizes", () => {
  const NOTED: JsonObject = {
    type: "object",
    properties: { note: { type: "string" } },
    required: ["note"],
    additionalProperties: false,
  };

  test("an item's text and fields count together against maxTextChars, written as the store keeps them", async () => {
    const s = await openAt({ path: join(dir, "store", "noted.db"), maxTextChars: 20 }, NOTED);
    // `{"note":"abc"}` is 14 characters, and the text 5: 19 fit.
    expect(await apply(s, { op: "add", text: "alpha", fields: { note: "abc" } }, "a1")).toMatchObject({ ok: true });
    embedded = [];
    expect(await apply(s, { op: "add", text: "alpha", fields: { note: "f".repeat(10) } }, "a2")).toEqual(
      refusedWith("the item's text and fields are 26 characters; an item may hold 20"),
    );
    expect(await apply(s, { op: "edit", id: 1n, if_version: 1n, fields: { note: "abcdef" } }, "e1")).toEqual(
      refusedWith("the item's text and fields are 22 characters; an item may hold 20"),
    );
    // The fields the item keeps count toward a new text too.
    expect(await apply(s, { op: "edit", id: 1n, if_version: 1n, text: "alpha bravo" }, "e2")).toEqual(
      refusedWith("the item's text and fields are 25 characters; an item may hold 20"),
    );
    expect(embedded).toEqual([]);
    expect((await s.snapshot()).version).toBe("1");
    await s.close();
  });

  test("a retire's reason may be at most maxTextChars", async () => {
    const s = await openAt({ maxTextChars: 10 });
    await apply(s, { op: "add", text: "alpha" }, "a1");
    expect(await apply(s, { op: "retire", id: 1n, if_version: 1n, why: "r".repeat(11) }, "r1")).toEqual(
      refusedWith("the reason is 11 characters; it may be at most 10"),
    );
    expect(await apply(s, { op: "retire", id: 1n, if_version: 1n, why: "r".repeat(10) }, "r2")).toMatchObject({
      ok: true,
    });
    await s.close();
  });
});
