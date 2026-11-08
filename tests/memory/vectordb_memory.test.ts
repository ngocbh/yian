/** VectorDBMemory: its store, identity and settings. */

import { afterEach, beforeEach, expect, mock, test } from "bun:test";
import { createHash } from "node:crypto";
import fs from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { EmbeddingModel } from "../../src/llm/embeddings.ts";
import { type Author, BUILTIN_MEMORY_TOOLS, checkContract, type MemorySession } from "../../src/memory/base.ts";
import { openStore, VectorDBMemory, type VectorDBMemoryOptions } from "../../src/memory/vectordb.ts";
import { openSqlite } from "../../src/platform.ts";
import type { JsonObject } from "../../src/typesys/json.ts";

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

test("a new store is a private file in a private folder, and meta holds its identity, type, settings and version", async () => {
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
  expect(JSON.stringify(s.info, (_, value: unknown) => (typeof value === "bigint" ? String(value) : value))).not.toContain(
    dir,
  );
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

test("a session's tools take k up to maxK, and fields typed by the item schema, or no fields for text only", async () => {
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
