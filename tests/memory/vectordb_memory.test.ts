/** VectorDBMemory: its store, identity and settings, recall and read tools. */

import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";
import { createHash } from "node:crypto";
import fs from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { EmbeddingModel } from "../../src/llm/embeddings.ts";
import {
  type Author,
  BUILTIN_MEMORY_TOOLS,
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

test("open({create: false}) refuses a missing store and creates nothing", async () => {
  fs.mkdirSync(join(dir, "store"));
  await expect(memory().open({ ...o, create: false })).rejects.toThrow(`\`${path}\` isn't a memory yet`);
  expect(fs.readdirSync(join(dir, "store"))).toEqual([]);
  await expect(memory({ path: join(dir, "none", "m.db") }).open({ ...o, create: false })).rejects.toThrow(
    "isn't a memory yet",
  );
  expect(fs.existsSync(join(dir, "none"))).toBe(false);
});

test("an empty file left where a store was being made becomes the store; create: false refuses it", async () => {
  fs.mkdirSync(join(dir, "store"), { mode: 0o700 });
  fs.writeFileSync(path, "", { mode: 0o600 });
  await expect(memory().open({ ...o, create: false })).rejects.toThrow(`\`${path}\` isn't a memory yet`);
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

test("the store's connection has busy_timeout 0 and secure_delete on, with a rollback journal", async () => {
  await (await openAt()).close();
  const db = openStore(path);
  try {
    expect(db.prepare("PRAGMA busy_timeout").get()).toEqual({ timeout: 0 });
    expect(db.prepare("PRAGMA secure_delete").get()).toEqual({ secure_delete: 1 });
    expect(db.prepare("PRAGMA journal_mode").get()).toEqual({ journal_mode: "delete" });
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
