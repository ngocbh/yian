/**
 * VectorDBMemory: a memory of items, each a text with optional typed fields, found by the
 * similarity of their texts' embeddings.
 *
 * Its store is one SQLite file with a rollback journal:
 * - `meta` holds the identity, the type, the settings (the embedding model and the item schema),
 *   the store layout's version and the change counter, which is the memory's version;
 * - `items` holds each item: its text, fields, version (the counter at its last change), author,
 *   why it was retired, and its text's vector;
 * - `history` holds each change: its id, item, op, author and time, the item's version after it,
 *   and what undoing it writes back;
 * - `changes` is the change-id table: each id's digest and stored result.
 *
 * Every access holds the memory's lock, `<path>.lock`, and runs synchronously inside it, so SQLite
 * never finds the store busy and its busy timeout stays 0. Embedding happens outside the lock.
 *
 * It imports `defaults`, `platform`, `tools`, `typesys`, the embedding types of `llm`, and
 * `memory/base.ts` and `memory/file_journal.ts`.
 */

import { createHash } from "node:crypto";
// The default import keeps calls patchable by test spies.
import fs from "node:fs";
import { basename, dirname, join, resolve } from "node:path";

import { current } from "../defaults.ts";
import type { Embed, EmbeddingModel } from "../llm/embeddings.ts";
import {
  type FileLock,
  FileLockError,
  isSystemError,
  openSqlite,
  realScheduler,
  type Scheduler,
  type Sqlite,
  sqliteErrorCode,
  syncFolder,
  waitForLock,
} from "../platform.ts";
import { ReportedToolError } from "../tools/errors.ts";
import { canonicalJson, type JsonObject, type JsonValue, logJson, parseJson } from "../typesys/json.ts";
import {
  type Applied,
  type Author,
  type ChangeEntry,
  type Erased,
  MemoryBase,
  MemoryBusyError,
  type MemoryContract,
  type MemoryInfo,
  type MemorySession,
  type MemoryToolSpec,
  type Recall,
} from "./base.ts";
import { charCount, isObject, openRegular } from "./file_journal.ts";

/** The layout of the store's tables; a store of another layout is refused. */
const STORE_VERSION = 1;

/** The rows a search reads at once, so its memory use stays bounded however many items there are. */
const SEARCH_CHUNK_ROWS = 1_000;

/** The store's tables, made in one transaction when the store is created. */
const SCHEMA = `
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE items(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  text TEXT NOT NULL,
  fields TEXT,
  version INTEGER NOT NULL,
  author TEXT NOT NULL,
  retired TEXT,
  vector BLOB NOT NULL
);
CREATE INDEX items_version ON items(version);
CREATE TABLE history(
  seq INTEGER PRIMARY KEY,
  change TEXT NOT NULL,
  item INTEGER NOT NULL,
  op TEXT NOT NULL,
  before TEXT,
  after INTEGER,
  author TEXT NOT NULL,
  at TEXT NOT NULL,
  undoes TEXT
);
CREATE INDEX history_item ON history(item);
CREATE INDEX history_change ON history(change);
CREATE TABLE changes(id TEXT PRIMARY KEY, digest TEXT NOT NULL, result TEXT NOT NULL, seq INTEGER NOT NULL);
CREATE INDEX changes_seq ON changes(seq);
`;

/** The parameter that names an item. */
const ID_PARAMETER: JsonObject = {
  type: "integer",
  minimum: 1,
  description: "The item's id, as search, list or read gives it.",
};

/** The precondition every change to an existing item takes. */
const IF_VERSION_PARAMETER: JsonObject = {
  type: "integer",
  minimum: 1,
  description: "The item's version, as search, list or read gave it.",
};

/** The `fields` of an item, as the contract declares them: the bound instance's item type narrows them. */
const ANY_FIELDS: JsonObject = { type: "object", description: "The item's typed fields." };

/** What `fields` takes in a session's tools: the item schema, and the definitions its references point at. */
interface FieldsSchema {
  readonly schema: JsonObject;
  readonly defs: JsonObject | null;
}

/**
 * Return the VectorDBMemory tools' specs, in the order a whole grant shows them. `search`'s `k`
 * is at most `maxK` when given. With `fields`, `add` and `edit` take `fields` by its schema and
 * `search` takes a `filter`, and `fields.defs` go at each parameters' root, where the schema's
 * references point; with none, an item is text only. Descriptions are fixed text.
 */
function toolSpecs(maxK: number | null, fields: FieldsSchema | null): readonly MemoryToolSpec[] {
  const spec = (
    name: string,
    write: boolean,
    description: string,
    properties: JsonObject,
    required: string[],
  ): MemoryToolSpec => {
    const parameters: JsonObject = { type: "object", properties, required, additionalProperties: false };
    if (fields?.defs != null && Object.hasOwn(properties, "fields")) parameters.$defs = fields.defs;
    return Object.freeze({ name, write, description, parameters });
  };
  const typed = (properties: JsonObject, key: string, schema: JsonObject): JsonObject =>
    fields === null ? properties : { ...properties, [key]: schema };
  const k: JsonObject = { type: "integer", minimum: 1, description: "How many items to return, best first." };
  if (maxK !== null) k.maximum = maxK;
  const filter: JsonObject = {
    type: "object",
    description: "Field values an item's fields must equal, to search only those items.",
  };
  const text: JsonObject = { type: "string", description: "The item's text, which search compares." };
  return Object.freeze([
    spec(
      "search",
      false,
      "Find the items whose text is most like query, best first, each with its id, version and similarity " +
        "score. Retired items aren't searched. Pass filter to search only the items whose fields equal its values.",
      typed({ query: { type: "string", description: "What to look for." }, k }, "filter", filter),
      ["query", "k"],
    ),
    spec("read", false, "Read one item: its text, fields, version and author.", { id: ID_PARAMETER }, ["id"]),
    spec(
      "list",
      false,
      "List the newest items first, each with its id, version, text and fields. Pass recent to list fewer.",
      { recent: { type: "integer", minimum: 1, description: "How many of the newest items to list." } },
      [],
    ),
    spec(
      "add",
      true,
      "Add an item: its text and, when the memory's items are typed, its fields. Never conflicts.",
      typed({ text }, "fields", fields?.schema ?? ANY_FIELDS),
      fields === null || fields.schema === ANY_FIELDS ? ["text"] : ["text", "fields"],
    ),
    spec(
      "edit",
      true,
      "Change an item's text, its fields or both. Conflicts unless the item's version is if_version, as read or " +
        "search gives it.",
      typed(
        { id: ID_PARAMETER, if_version: IF_VERSION_PARAMETER, text: { ...text, description: "The item's new text." } },
        "fields",
        fields?.schema ?? ANY_FIELDS,
      ),
      ["id", "if_version"],
    ),
    spec(
      "retire",
      true,
      "Retire an item: search, list and the recall no longer show it, and its history is kept. Conflicts unless " +
        "the item's version is if_version.",
      {
        id: ID_PARAMETER,
        if_version: IF_VERSION_PARAMETER,
        why: { type: "string", description: "Why the item is retired." },
      },
      ["id", "if_version", "why"],
    ),
    spec(
      "delete",
      true,
      "Delete an item, with its text, its vector and its history. Conflicts unless the item's version is " +
        "if_version.",
      { id: ID_PARAMETER, if_version: IF_VERSION_PARAMETER },
      ["id", "if_version"],
    ),
  ]);
}

/** The settings and caps a `VectorDBMemory` is built with; each omitted cap comes from `src/defaults.ts`. */
export interface VectorDBMemoryOptions {
  /** The store's SQLite file. A symlinked path is resolved once, here. */
  path: string;
  /**
   * The model that embeds the items' texts. A new store needs it; an existing store's stored model
   * must equal it, and is used when it is omitted.
   */
  embedding?: EmbeddingModel;
  /** The embeddings call that searches and writes use; without it, only reads that need no embedding work. */
  embed?: Embed;
  /** The identity a new memory takes; an existing memory's stored identity must equal it. */
  id?: string;
  /** Items the memory may hold, retired ones included. */
  maxItems?: number;
  /** Items one `search` returns and one `list` shows. */
  maxK?: number;
  /** Characters of one item's text, and of a search query. */
  maxTextChars?: number;
  /** Characters of the recall. */
  maxRecallChars?: number;
  /** The newest items the recall shows after its count (default 0: the count alone). */
  recallRecent?: number;
  /** Changes the memory can undo. */
  maxUndo?: number;
  /**
   * Change ids the memory keeps to answer a repeated change. Every writer on one store must use
   * the same value: one that keeps fewer ids may apply again a change another still answers.
   */
  maxChangeIds?: number;
  /** Longest wait in milliseconds for the memory's lock; 0 fails at once when it is held. */
  lockTimeoutMs?: number;
  /**
   * The sleep the lock wait uses between tries, and the clock that stamps changes. A wait's
   * deadline runs on a monotonic clock, not this one.
   */
  scheduler?: Scheduler;
}

/** The caps a VectorDBMemory checks, as numbers. */
interface Limits {
  readonly maxItems: number;
  readonly maxK: number;
  readonly maxTextChars: number;
  readonly maxRecallChars: number;
  readonly recallRecent: number;
  readonly maxUndo: number;
  readonly maxChangeIds: number;
}

/**
 * Return `value` when it is a safe integer of at least `minimum`, else its default.
 *
 * @throws {RangeError} If `value` is given and isn't such an integer.
 */
function capOption(name: string, value: number | undefined, fallback: number, minimum: number): number {
  if (value === undefined) return fallback;
  if (!Number.isSafeInteger(value) || value < minimum) {
    throw new RangeError(`VectorDBMemory ${name} must be an integer of at least ${minimum}, not ${String(value)}`);
  }
  return value;
}

/** Return whether `value` is a non-empty string on one line, with no lone surrogate. */
function isLine(value: unknown): value is string {
  return typeof value === "string" && value !== "" && value.isWellFormed() && !/[\r\n\0]/.test(value);
}

/**
 * Return `value` as an embedding model with only its own fields.
 *
 * @throws {TypeError} If it isn't `{provider, model}`, each a non-empty string on one line.
 */
function checkEmbedding(value: unknown): EmbeddingModel {
  if (!isObject(value) || !isLine(value.provider) || !isLine(value.model)) {
    throw new TypeError("VectorDBMemory embedding must be {provider, model}, each a non-empty string on one line");
  }
  return Object.freeze({ provider: value.provider, model: value.model });
}

/** Return `model` as messages show it: `provider/model`. */
function modelName(model: EmbeddingModel): string {
  return `${model.provider}/${model.model}`;
}

/** Return `path` in backticks for a message, a control character escaped. */
function shown(path: string): string {
  return `\`${JSON.stringify(path).slice(1, -1)}\``;
}

/** Return the sha256 hex digest of `data`. */
function sha256(data: string | Uint8Array): string {
  return createHash("sha256").update(data).digest("hex");
}

/** `fs.lstatSync`, or `null` when nothing is at `path`. */
function lstatOrNull(path: string): fs.Stats | null {
  try {
    return fs.lstatSync(path);
  } catch (error) {
    if (isSystemError(error) && error.code === "ENOENT") return null;
    throw error;
  }
}

/** Return `path`'s real path, resolving the symlinks of the part of it that exists. */
function realPath(path: string): string {
  const rest: string[] = [];
  let head = resolve(path);
  for (;;) {
    try {
      return join(fs.realpathSync(head), ...rest.reverse());
    } catch (error) {
      if (!isSystemError(error) || error.code !== "ENOENT" || dirname(head) === head) throw error;
      rest.push(basename(head));
      head = dirname(head);
    }
  }
}

/** Make the missing folder `folder` and its missing parents, private, syncing each new folder's parent. */
function makeFolder(folder: string): void {
  const first = fs.mkdirSync(folder, { recursive: true, mode: 0o700 });
  if (first === undefined) return;
  for (let made = folder; made !== dirname(first); made = dirname(made)) {
    // The mode passed to mkdir is masked by the umask; chmod isn't.
    fs.chmodSync(made, 0o700);
    syncFolder(dirname(made));
  }
}

/**
 * Make the empty store file `path`, private, and sync it into its folder. SQLite reads an empty
 * file as an empty database.
 */
function makeStoreFile(path: string): void {
  const { O_CREAT, O_EXCL, O_WRONLY } = fs.constants;
  const fd = openRegular(path, O_CREAT | O_EXCL | O_WRONLY, 0o600);
  try {
    // The open's mode is masked by the umask; fchmod isn't. SQLite gives its journal this mode too.
    fs.fchmodSync(fd, 0o600);
    fs.fsyncSync(fd);
  } finally {
    fs.closeSync(fd);
  }
  syncFolder(dirname(path));
}

/**
 * The files the store is made of, what each is called in a message, and what to do when one isn't
 * a plain file. A symlink would let SQLite read or write elsewhere, and another hard link to the
 * store would be a second name with a lock of its own.
 */
const STORE_FILES: readonly { suffix: string; role: string; fix: string }[] = [
  { suffix: "", role: "store file", fix: "replace it with a copy" },
  { suffix: ".lock", role: "lock file", fix: "remove it" },
  { suffix: "-journal", role: "rollback journal", fix: "replace it with a copy" },
];

/**
 * Check, with `lstat`, that each of the store's files that exists is a regular file with one link.
 *
 * @throws Error naming the file by its role, what it is instead, and what to do.
 */
function checkFiles(path: string): void {
  for (const { suffix, role, fix } of STORE_FILES) {
    const stats = lstatOrNull(`${path}${suffix}`);
    if (stats === null) continue;
    let why: string | null = null;
    if (stats.isSymbolicLink()) why = "a symlink";
    else if (!stats.isFile()) why = "not a regular file";
    else if (stats.nlink > 1) why = "a hard link";
    if (why !== null) throw new Error(`this memory's ${role} isn't a plain file (${why}); ${fix}`);
  }
}

/**
 * @internal Open the store at `path` with the settings every access uses: no busy timeout, since
 * the memory's lock serializes all access; `secure_delete`, so deleted text doesn't stay in free
 * pages; a rollback journal deleted after each commit; full syncs; and temporary tables in memory,
 * so no copy of a text lands in a temporary file.
 */
export function openStore(path: string): Sqlite {
  const db = openSqlite(path);
  try {
    db.exec("PRAGMA busy_timeout = 0");
    db.exec("PRAGMA secure_delete = 1");
    db.exec("PRAGMA journal_mode = DELETE");
    db.exec("PRAGMA synchronous = FULL");
    db.exec("PRAGMA temp_store = MEMORY");
  } catch (error) {
    db.close();
    throw error;
  }
  return db;
}

/**
 * Run `fn` on a connection to the store at `path`, closed after it.
 *
 * @throws Error when the store file is gone, rather than letting SQLite make an empty one.
 */
function withDb<T>(path: string, fn: (db: Sqlite) => T): T {
  if (lstatOrNull(path) === null) throw new Error("this memory's store file is gone");
  const db = openStore(path);
  try {
    return fn(db);
  } finally {
    db.close();
  }
}

/** Run `fn` in one `BEGIN IMMEDIATE` transaction on `db`, rolled back when it throws. */
function transaction<T>(db: Sqlite, fn: () => T): T {
  db.exec("BEGIN IMMEDIATE");
  try {
    const result = fn();
    db.exec("COMMIT");
    return result;
  } catch (error) {
    if (db.inTransaction) db.exec("ROLLBACK");
    throw error;
  }
}

/** The settings a store keeps: the model its vectors come from, and the schema of its items' fields. */
interface StoredSettings {
  readonly embedding: EmbeddingModel;
  readonly item_schema: JsonObject | null;
}

/** What a store's `meta` table holds. */
interface Meta {
  readonly identity: string;
  readonly settings: StoredSettings;
  /** The change counter: the memory's version. */
  readonly version: number;
}

/** Return the store's settings as `meta` holds them: yian JSON, so ints and floats read back as written. */
function settingsText(settings: StoredSettings): string {
  return logJson({ embedding: { ...settings.embedding }, item_schema: settings.item_schema });
}

/**
 * Return the store's `meta`, or `null` for a store with no tables yet (one just made, or one whose
 * making a crash rolled back).
 *
 * @throws Error when the file holds other tables, another type of memory, another layout, or a
 *   `meta` that isn't one this code wrote.
 */
function readMeta(db: Sqlite, path: string): Meta | null {
  const tables = db.prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'").all();
  if (tables.length === 0) return null;
  if (!tables.some((table) => table.name === "meta")) {
    throw new Error(`${shown(path)} isn't a VectorDBMemory store: it holds other tables`);
  }
  const rows = new Map(db.prepare("SELECT key, value FROM meta").all().map((row) => [row.key, row.value]));
  if (rows.get("type") !== "VectorDBMemory") throw new Error(`${shown(path)} isn't a VectorDBMemory store`);
  const layout = rows.get("store_version");
  if (layout !== String(STORE_VERSION)) {
    throw new Error(`this memory's store is version ${String(layout)}; this yian reads version ${STORE_VERSION}`);
  }
  const identity = rows.get("identity");
  const version = Number(rows.get("version"));
  let settings: unknown;
  try {
    settings = parseJson(String(rows.get("settings")), "codec");
  } catch {
    settings = null;
  }
  const schema = isObject(settings) ? settings.item_schema : undefined;
  if (
    !isLine(identity) ||
    !Number.isSafeInteger(version) ||
    version < 0 ||
    !isObject(settings) ||
    !(schema === null || isObject(schema))
  ) {
    throw new Error("this memory's store has a damaged meta table");
  }
  return { identity, settings: { embedding: checkEmbedding(settings.embedding), item_schema: schema }, version };
}

/** Make the store's tables in `db`, which has none, and record its identity and settings. */
function initStore(db: Sqlite, identity: string, settings: StoredSettings): Meta {
  transaction(db, () => {
    db.exec(SCHEMA);
    const insert = db.prepare("INSERT INTO meta(key, value) VALUES (?, ?)");
    insert.run("identity", identity);
    insert.run("type", "VectorDBMemory");
    insert.run("settings", settingsText(settings));
    insert.run("store_version", String(STORE_VERSION));
    insert.run("version", "0");
  });
  return { identity, settings, version: 0 };
}

/** Return why an input typed `given` can't bind a store whose items are typed `stored`, or `null` when it can. */
function itemTypeProblem(stored: JsonObject | null, given: JsonObject | null): string | null {
  if (canonicalJson(stored) === canonicalJson(given)) return null;
  if (stored === null) return "this memory's items are text only; its input can't give them a type";
  if (given === null) return "this memory's items have typed fields; give its input their type";
  return "this memory's items have another type than its input's; bind a memory made with that type";
}

/** Return `count` and `noun`, plural unless `count` is 1: `plural(2, "item")` is `"2 items"`. */
function plural(count: number, noun: string): string {
  return `${count} ${count === 1 ? noun : `${noun}s`}`;
}

/** Return the first `limit` code points of `text`. */
function firstChars(text: string, limit: number): string {
  let index = 0;
  for (let n = 0; n < limit && index < text.length; n++) index += text.codePointAt(index)! > 0xffff ? 2 : 1;
  return text.slice(0, index);
}

/** A refusal the calling model reads: a TOOL node doesn't retry it, since it can't change. */
function refusal(message: string): ReportedToolError {
  return new ReportedToolError(message, { permanent: true });
}

/** Return `args[name]` when it is a string, or `undefined` when it is absent and not `required`. */
function stringArg(args: JsonObject, name: string, required: true): string;
function stringArg(args: JsonObject, name: string, required: false): string | undefined;
function stringArg(args: JsonObject, name: string, required: boolean): string | undefined {
  const value = Object.hasOwn(args, name) ? args[name] : undefined;
  if (value === undefined && !required) return undefined;
  if (typeof value !== "string" || !value.isWellFormed()) {
    throw new TypeError(`VectorDBMemory ${name} must be a string`);
  }
  return value;
}

/**
 * Return `args[name]` as a number when it is an integer of at least 1, or `undefined` when it is
 * absent and not `required`.
 */
function intArg(args: JsonObject, name: string, required: true): number;
function intArg(args: JsonObject, name: string, required: false): number | undefined;
function intArg(args: JsonObject, name: string, required: boolean): number | undefined {
  const value = Object.hasOwn(args, name) ? args[name] : undefined;
  if (value === undefined && !required) return undefined;
  const number = typeof value === "bigint" ? Number(value) : value;
  if (typeof number !== "number" || !Number.isSafeInteger(number) || number < 1) {
    throw new TypeError(`VectorDBMemory ${name} must be an integer of at least 1`);
  }
  return number;
}

/** Return the vector a store row's `bytes` hold: float32s, in the byte order every platform yian runs on uses. */
function vectorOf(bytes: Uint8Array): Float32Array {
  const size = Float32Array.BYTES_PER_ELEMENT;
  // A view needs aligned bytes; the drivers don't promise them.
  if (bytes.byteOffset % size === 0) return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / size);
  return new Float32Array(bytes.slice().buffer);
}

/** Return the bytes a store row keeps for `vector`. */
function bytesOf(vector: Float32Array): Uint8Array {
  return new Uint8Array(vector.buffer.slice(vector.byteOffset, vector.byteOffset + vector.byteLength));
}

/** Return the Euclidean length of `vector`. */
function norm(vector: Float32Array): number {
  let sum = 0;
  for (const x of vector) sum += x * x;
  return Math.sqrt(sum);
}

/** Return the cosine similarity of `query`, whose length is `queryNorm`, and `vector`; 0 when either is zero. */
function cosine(query: Float32Array, queryNorm: number, vector: Float32Array): number {
  let dot = 0;
  let sum = 0;
  for (let i = 0; i < query.length; i++) {
    dot += query[i]! * vector[i]!;
    sum += vector[i]! * vector[i]!;
  }
  return queryNorm === 0 || sum === 0 ? 0 : dot / (queryNorm * Math.sqrt(sum));
}

/** Return the yian value a store column holds as yian JSON, or `null` for none. */
function storedJson(text: unknown): JsonValue {
  return typeof text === "string" ? parseJson(text, "codec") : null;
}

/** An item row as the store holds it. */
interface ItemRow {
  readonly id: number;
  readonly version: number;
  readonly text: string;
  /** The fields as yian JSON, or `null` for an item that is text only. */
  readonly fields: string | null;
  /** The author as JSON. */
  readonly author: string;
  /** Why it was retired, or `null` while it is live. */
  readonly retired: string | null;
}

/** Return the item `id`'s row, or `undefined` when there is none. */
function itemRow(db: Sqlite, id: number): ItemRow | undefined {
  return db.prepare("SELECT id, text, fields, version, author, retired FROM items WHERE id = ?").get(id) as
    | ItemRow
    | undefined;
}

/**
 * Return `row` as a tool result shows an item: its id, version, text and fields, and with `full`
 * its author and why it was retired.
 */
function itemResult(row: ItemRow, full: boolean): JsonObject {
  const item: JsonObject = { id: BigInt(row.id), version: BigInt(row.version), text: row.text };
  if (row.fields !== null) item.fields = storedJson(row.fields);
  if (!full) return item;
  item.author = storedJson(row.author);
  if (row.retired !== null) item.retired = row.retired;
  return item;
}

/** Return the store's change counter: the memory's version. */
function storedVersion(db: Sqlite): number {
  return Number(db.prepare("SELECT value FROM meta WHERE key = 'version'").get()!.value);
}

/** Return the number of items that aren't retired. */
function liveCount(db: Sqlite): number {
  return Number(db.prepare("SELECT count(*) AS n FROM items WHERE retired IS NULL").get()!.n);
}

/** Return `text` as one entry of the recall: its later lines indented under the first. */
function recallLine(id: number, text: string): string {
  return `[${id}] ${text.replaceAll("\n", "\n  ")}`;
}

/**
 * Return the recall: the count of items that aren't retired, then the `recallRecent` newest of them
 * (by their last change), all within `maxRecallChars`. An item that doesn't fit is left out whole
 * and counted in the cut note, and an older one that fits still shows.
 */
function recallOf(db: Sqlite, limits: Limits): Recall {
  const count = liveCount(db);
  const header = firstChars(plural(count, "item"), limits.maxRecallChars);
  const lines = [header];
  let used = charCount(header);
  let unshown = 0;
  const newest = db
    .prepare("SELECT id, text FROM items WHERE retired IS NULL ORDER BY version DESC, id DESC LIMIT ?")
    .all(limits.recallRecent);
  for (const row of newest) {
    const line = recallLine(row.id as number, row.text as string);
    const cost = charCount(line) + 1;
    if (used + cost > limits.maxRecallChars) {
      unshown++;
      continue;
    }
    lines.push(line);
    used += cost;
  }
  return {
    text: lines.join("\n"),
    cut: unshown > 0 ? `${unshown} of the ${newest.length} newest items not shown` : null,
    hint: count > 0 ? { tool: "search", text: "Search them with {tool}." } : null,
  };
}

/**
 * Check that `vectors` is one non-empty vector of finite numbers per text, `count` in all, each of
 * one length.
 *
 * @throws Error naming what is wrong, for an embedder that breaks its contract.
 */
function checkVectors(vectors: unknown, count: number): Float32Array[] {
  if (!Array.isArray(vectors) || vectors.length !== count) {
    const got = Array.isArray(vectors) ? vectors.length : 0;
    throw new Error(`the embedder returned ${plural(got, "vector")} for ${plural(count, "text")}`);
  }
  for (const vector of vectors) {
    if (!(vector instanceof Float32Array) || vector.length === 0 || !vector.every(Number.isFinite)) {
      throw new Error("the embedder returned a vector that isn't a non-empty list of finite numbers");
    }
    if (vector.length !== (vectors[0] as Float32Array).length) {
      throw new Error("the embedder returned vectors of different lengths");
    }
  }
  return vectors as Float32Array[];
}

/** Return the length of the store's vectors, or `null` while it holds no item. */
function storedDimensions(db: Sqlite): number | null {
  const row = db.prepare("SELECT length(vector) AS bytes FROM items LIMIT 1").get();
  return row === undefined ? null : (row.bytes as number) / Float32Array.BYTES_PER_ELEMENT;
}

/** What a session needs of its memory: the store, its caps, its lock and its clock. */
interface Store {
  /** The store's SQLite file, symlinks resolved. */
  readonly path: string;
  readonly limits: Limits;
  readonly scheduler: Scheduler;
  /** The model the instance was built with, or `undefined` to take the stored one. */
  readonly embedding: EmbeddingModel | undefined;
  readonly embed: Embed | undefined;
  /**
   * Run `body` holding the memory's lock, after this process's earlier calls on the store: they
   * take turns, so only another process's writer makes it wait for the lock. It waits at most the
   * lock timeout from when it was called, its time in the queue included, and an abort while it
   * is queued rejects it at once.
   *
   * Returns the call's `result`, which rejects with `MemoryBusyError` if the lock is still held by
   * another writer at the deadline, and `done`, which settles once its turn has finished, later
   * than `result` when an abort let the call go before its turn came.
   */
  locked<T>(signal: AbortSignal | undefined, body: Held<T>): Turn<T>;
}

/** A call's place in a store's queue: its result, and when its turn has finished. */
interface Turn<T> {
  readonly result: Promise<T>;
  /** Settles, never rejecting, once the turn has run or been passed over. */
  readonly done: Promise<void>;
}

/**
 * A body that runs holding a memory's lock. It must finish before it returns, so it can't be async:
 * the lock is released when it returns, and an async body would go on without it.
 */
type Held<T> = (() => T) & (T extends PromiseLike<unknown> ? never : unknown);

/**
 * This process's calls that hold a store's lock, keyed by the store's real path and chained, so
 * its sessions and opens take turns rather than poll the lock against each other.
 */
const TURNS = new Map<string, Promise<void>>();

/**
 * Run `fn` once this process's earlier turns on the store at `path` have finished.
 *
 * An abort of `signal` while the call is queued rejects it at once with the signal's reason; its
 * turn, when it comes, does nothing, so the calls after it keep their order. Once its turn has
 * started, `fn` answers the abort itself, so a change it applies is never reported as aborted.
 */
function inTurn<T>(path: string, signal: AbortSignal | undefined, fn: () => Promise<T>): Turn<T> {
  let started = false;
  const run = (TURNS.get(path) ?? Promise.resolve()).then(() => {
    started = true;
    signal?.throwIfAborted();
    return fn();
  });
  const done = run.then(
    () => undefined,
    () => undefined,
  );
  TURNS.set(path, done);
  void done.then(() => {
    if (TURNS.get(path) === done) TURNS.delete(path);
  });
  if (signal === undefined) return { result: run, done };
  const result = new Promise<T>((resolve, reject) => {
    const onAbort = () => {
      if (!started) reject(signal.reason);
    };
    signal.addEventListener("abort", onAbort, { once: true });
    // A signal aborted already never fires its listeners.
    if (signal.aborted) onAbort();
    void run.then(resolve, reject).finally(() => signal.removeEventListener("abort", onAbort));
  });
  return { result, done };
}

/**
 * Return `error` with the store's location taken out of its message: an OS error names absolute
 * paths, and a session's errors can reach a model and a run log, which never hold a location. A
 * system error keeps its code.
 */
function withoutLocation(error: unknown, path: string): unknown {
  const folder = dirname(path);
  if (!(error instanceof Error) || !error.message.includes(folder)) return error;
  const copy: Error & { code?: unknown; syscall?: unknown; errno?: unknown } = new Error(
    error.message.replaceAll(path, "the memory's store").replaceAll(folder, "the memory's folder"),
  );
  copy.name = error.name;
  // Not `path`, `dest` or `cause`, which hold the location too.
  const { code, syscall, errno } = error as NodeJS.ErrnoException;
  if (code !== undefined) copy.code = code;
  if (syscall !== undefined) copy.syscall = syscall;
  if (errno !== undefined) copy.errno = errno;
  return copy;
}

/** The SQLite primary result code for a file that isn't a database. */
const SQLITE_NOTADB = 26;

/**
 * A memory kept as a SQLite store of items, searched by embedding.
 *
 * @example
 * const cases = new VectorDBMemory({
 *   path: "/srv/memory/cases.db",
 *   embedding: { provider: "openai", model: "text-embedding-3-small" },
 *   embed: defaultEmbed(env),
 * });
 */
export class VectorDBMemory extends MemoryBase {
  static override readonly contract: MemoryContract = Object.freeze({
    type: "VectorDBMemory",
    tools: toolSpecs(null, { schema: ANY_FIELDS, defs: null }),
  });

  /** The store's SQLite file, symlinks resolved. */
  readonly path: string;
  readonly #id: string | undefined;
  readonly #store: Store;

  /**
   * Bind a VectorDBMemory to the store file `path`, creating nothing.
   *
   * @throws {TypeError} If `path` or `id` isn't a non-empty string, `embedding` isn't
   *   `{provider, model}`, or `embed` isn't a function.
   * @throws {RangeError} If a cap isn't an integer in its range.
   */
  constructor(options: VectorDBMemoryOptions) {
    super();
    const { path, id } = options;
    if (typeof path !== "string" || path === "" || path.includes("\0")) {
      throw new TypeError("VectorDBMemory path must be a file path");
    }
    if (id !== undefined && !isLine(id)) {
      throw new TypeError("VectorDBMemory id must be a non-empty string on one line");
    }
    const embedding = options.embedding === undefined ? undefined : checkEmbedding(options.embedding);
    if (options.embed !== undefined && typeof options.embed !== "function") {
      throw new TypeError("VectorDBMemory embed must be a function");
    }
    const defaults = current();
    const limits: Limits = Object.freeze({
      maxItems: capOption("maxItems", options.maxItems, defaults.vectorMemoryMaxItems, 1),
      maxK: capOption("maxK", options.maxK, defaults.vectorMemoryMaxK, 1),
      maxTextChars: capOption("maxTextChars", options.maxTextChars, defaults.vectorMemoryMaxTextChars, 1),
      maxRecallChars: capOption("maxRecallChars", options.maxRecallChars, defaults.vectorMemoryMaxRecallChars, 1),
      recallRecent: capOption("recallRecent", options.recallRecent, 0, 0),
      maxUndo: capOption("maxUndo", options.maxUndo, defaults.memoryMaxUndo, 0),
      maxChangeIds: capOption("maxChangeIds", options.maxChangeIds, defaults.memoryMaxChangeIds, 1),
    });
    const defaultTimeoutMs = defaults.memoryLockTimeoutSeconds * 1000;
    const timeoutMs = options.lockTimeoutMs ?? defaultTimeoutMs;
    if (typeof timeoutMs !== "number" || Number.isNaN(timeoutMs) || timeoutMs < 0) {
      throw new RangeError(`VectorDBMemory lockTimeoutMs must be 0 or more milliseconds, not ${String(timeoutMs)}`);
    }
    // Resolved now, so a later chdir or a swapped symlink doesn't move the memory.
    const real = realPath(path);
    const scheduler = options.scheduler ?? realScheduler;
    this.path = real;
    this.#id = id;
    this.#store = Object.freeze({
      path: real,
      limits,
      scheduler,
      embedding,
      embed: options.embed,
      locked<T>(signal: AbortSignal | undefined, body: Held<T>): Turn<T> {
        // A monotonic clock, so a wall-clock step while queued neither stretches nor cuts the wait.
        const deadline = performance.now() + timeoutMs;
        return inTurn(real, signal, async () => {
          // Checked before the lock file is opened, and on every call: the files may change.
          checkFiles(real);
          // Only what is left of the timeout: a call waits it out once, queued or not.
          const left = Math.max(0, deadline - performance.now());
          let lock: FileLock;
          try {
            lock = await waitForLock(`${real}.lock`, { timeoutMs: left, signal, scheduler, private: true });
          } catch (error) {
            if (error instanceof FileLockError && error.reason === "busy") {
              // No cause: the lock's error names its path, and this one can reach a model.
              throw new MemoryBusyError("this memory is busy: another writer holds its lock");
            }
            throw error;
          }
          try {
            // Again holding the lock: the files may have changed while the call waited for it.
            checkFiles(real);
            return body();
          } finally {
            lock.release();
          }
        });
      },
    });
  }

  /**
   * Open a session on the memory. With `create`, a missing store is made (0600, in a folder made
   * 0700 when missing) and records the identity, the embedding model and `itemType`, and the
   * session reports `created`.
   *
   * @param o.itemType The JSON Schema of the items' fields, or `null` for items that are text only.
   * @throws {TypeError} If `itemType` isn't a JSON object or `null`.
   * @throws Error when the store doesn't exist and `create` is false, isn't a VectorDBMemory store,
   *   or its stored identity, embedding model or item schema isn't the given one.
   * @throws {MemoryBusyError} If it waits for the lock longer than the timeout.
   */
  override async open(o: {
    create: boolean;
    itemType: JsonObject | null;
    signal: AbortSignal;
  }): Promise<MemorySession> {
    o.signal.throwIfAborted();
    const { itemType } = o;
    if (itemType !== null && !isObject(itemType)) {
      throw new TypeError("a VectorDBMemory's item type must be a JSON Schema object, or null for text only");
    }
    const { path, embedding } = this.#store;
    // Checked before anything is made, so a refused open leaves no folder or file behind.
    const canCreate = () => {
      if (!o.create) throw new Error(`${shown(path)} isn't a memory yet: there is no store`);
      if (embedding === undefined) throw new Error("a new VectorDBMemory needs an embedding model");
    };
    const stats = lstatOrNull(path);
    if (stats === null) {
      canCreate();
      if (lstatOrNull(dirname(path)) === null) makeFolder(dirname(path));
    } else if (stats.isDirectory()) {
      throw new Error(`${shown(path)} is a folder, not a store file`);
    }
    let created = false;
    let meta: Meta;
    try {
      meta = await this.#store.locked(o.signal, () => {
        if (lstatOrNull(path) === null) {
          canCreate();
          makeStoreFile(path);
        }
        return withDb(path, (db) => {
          const stored = readMeta(db, path);
          if (stored !== null) return stored;
          if (!o.create) throw new Error(`${shown(path)} isn't a memory yet`);
          if (embedding === undefined) throw new Error("a new VectorDBMemory needs an embedding model");
          created = true;
          return initStore(db, this.#id ?? `at-${sha256(path).slice(0, 16)}`, { embedding, item_schema: itemType });
        });
      }).result;
    } catch (error) {
      const code = sqliteErrorCode(error);
      if (code !== undefined && (code & 0xff) === SQLITE_NOTADB) {
        throw new Error(`${shown(path)} isn't a VectorDBMemory store: it isn't a SQLite database`);
      }
      throw error;
    }
    if (this.#id !== undefined && meta.identity !== this.#id) {
      throw new Error(`this memory is ${shown(meta.identity)}, not ${shown(this.#id)}`);
    }
    const stored = meta.settings;
    if (embedding !== undefined && modelName(stored.embedding) !== modelName(embedding)) {
      throw new Error(
        `this memory was embedded with ${modelName(stored.embedding)}; re-embed it with \`yian memory reembed\``,
      );
    }
    const problem = itemTypeProblem(stored.item_schema, itemType);
    if (problem !== null) throw new Error(problem);
    return new VectorDBSession(this.#store, sessionInfo(meta, this.#store.limits), stored, created);
  }

  /** Empty the memory, keeping its identity and settings. */
  override reset(_o: { signal: AbortSignal }): Promise<void> {
    return Promise.reject(new Error("VectorDBMemory reset isn't available yet"));
  }
}

/** Return a session's info for the store whose `meta` is `meta`: its settings hold no location. */
function sessionInfo(meta: Meta, limits: Limits): MemoryInfo {
  const { embedding, item_schema: schema } = meta.settings;
  let fields: FieldsSchema | null = null;
  if (schema !== null) {
    const { $defs: defs, ...rest } = schema;
    fields = { schema: { ...rest, description: ANY_FIELDS.description! }, defs: isObject(defs) ? defs : null };
  }
  return Object.freeze({
    type: "VectorDBMemory",
    identity: meta.identity,
    settings: Object.freeze({
      embedding: { provider: embedding.provider, model: embedding.model },
      item_schema: schema,
      max_items: BigInt(limits.maxItems),
      max_k: BigInt(limits.maxK),
      max_text_chars: BigInt(limits.maxTextChars),
      max_recall_chars: BigInt(limits.maxRecallChars),
      recall_recent: BigInt(limits.recallRecent),
      max_undo: BigInt(limits.maxUndo),
      max_change_ids: BigInt(limits.maxChangeIds),
    }),
    tools: toolSpecs(limits.maxK, fields),
  });
}

/** A body that runs on the store holding its lock; like `Held`, it can't be async. */
type OnStore<T> = ((db: Sqlite) => T) & (T extends PromiseLike<unknown> ? never : unknown);

/** One open VectorDBMemory. Every call takes the memory's lock; embedding happens outside it. */
class VectorDBSession implements MemorySession {
  readonly info: MemoryInfo;
  readonly created: boolean;
  readonly #store: Store;
  /** The settings the store held when the session opened, which every access checks it still holds. */
  readonly #stored: StoredSettings;
  /** The session's calls and turns not yet settled: `close` waits for them. */
  readonly #pending = new Set<Promise<void>>();
  #closed = false;

  /**
   * @param store The memory it reads and writes.
   * @param info What the memory is, as its recorded value holds it.
   * @param stored The settings its store held when it opened.
   * @param created Whether the open that made this session created the memory.
   */
  constructor(store: Store, info: MemoryInfo, stored: StoredSettings, created: boolean) {
    this.#store = store;
    this.#stored = stored;
    this.info = info;
    this.created = created;
  }

  /**
   * Read the version and the recall in one read under the lock.
   *
   * @throws {MemoryBusyError} If another writer holds the lock past the timeout.
   */
  async snapshot(): Promise<{ version: string; recall: Recall }> {
    this.#checkOpen();
    return this.#locked(undefined, (db) => ({
      version: String(storedVersion(db)),
      recall: recallOf(db, this.#store.limits),
    }));
  }

  /**
   * Call the read tool `tool`: `search`, `read` or `list`. A search embeds its query before it
   * takes the lock.
   *
   * @throws {ReportedToolError} (permanent) For a missing item, an empty or too long query, or a
   *   `k` past `maxK`, with the reason.
   * @throws {TypeError} For a tool that isn't a read tool, or an argument of the wrong type.
   * @throws Error when the embedder fails or breaks its contract, and the embedder's own errors.
   */
  async call(tool: string, args: JsonObject, o: { signal: AbortSignal }): Promise<JsonObject> {
    this.#checkOpen();
    o.signal.throwIfAborted();
    const spec = this.info.tools.find((each) => each.name === tool);
    if (spec === undefined) {
      const names = this.info.tools.map((each) => each.name).join(", ");
      throw new TypeError(`VectorDBMemory has no tool ${tool}; it has ${names}`);
    }
    if (spec.write) throw new TypeError(`${tool} is a write tool: apply it as a change`);
    if (!isObject(args)) throw new TypeError(`VectorDBMemory ${tool} takes its arguments as an object`);
    if (tool === "search") return this.#track(this.#search(args, o.signal));
    if (tool === "read") {
      const id = intArg(args, "id", true);
      return this.#locked(o.signal, (db) => {
        const row = itemRow(db, id);
        if (row === undefined) throw refusal(`item ${id} doesn't exist`);
        return itemResult(row, true);
      });
    }
    const recent = Math.min(intArg(args, "recent", false) ?? Number.POSITIVE_INFINITY, this.#store.limits.maxK);
    return this.#locked(o.signal, (db) => {
      const rows = db
        .prepare(
          "SELECT id, text, fields, version, author, retired FROM items WHERE retired IS NULL " +
            "ORDER BY version DESC, id DESC LIMIT ?",
        )
        .all(recent) as unknown as ItemRow[];
      const result: JsonObject = { items: rows.map((row) => itemResult(row, false)) };
      const more = liveCount(db) - rows.length;
      if (more > 0) result.cut = `${more} more; search them`;
      return result;
    });
  }

  async apply(_c: { id: string; change: JsonObject; author: Author; signal: AbortSignal }): Promise<Applied> {
    this.#checkOpen();
    throw new Error("VectorDBMemory changes aren't available yet");
  }

  async undo(_o: { id: string; changeId: string; author: Author; signal: AbortSignal }): Promise<Applied> {
    this.#checkOpen();
    throw new Error("VectorDBMemory undo isn't available yet");
  }

  async erase(_o: { id: string; target: JsonObject; author: Author; signal: AbortSignal }): Promise<Erased> {
    this.#checkOpen();
    throw new Error("VectorDBMemory erase isn't available yet");
  }

  async history(_o: { limit: number }): Promise<ChangeEntry[]> {
    this.#checkOpen();
    throw new Error("VectorDBMemory history isn't available yet");
  }

  /** Close the session once every call and turn it started has settled; later calls reject. */
  async close(): Promise<void> {
    this.#closed = true;
    // A change queues its second turn after its first settles, so wait until none is left.
    while (this.#pending.size > 0) await Promise.all([...this.#pending]);
  }

  /**
   * Embed `query`, then return the `k` items most like it, best first, among those that aren't
   * retired and whose fields equal `filter`'s values. The store is read `SEARCH_CHUNK_ROWS` rows
   * at a time.
   */
  async #search(args: JsonObject, signal: AbortSignal): Promise<JsonObject> {
    const { limits } = this.#store;
    const query = stringArg(args, "query", true);
    const k = intArg(args, "k", true);
    const filter = Object.hasOwn(args, "filter") ? args.filter : undefined;
    if (filter !== undefined && !isObject(filter)) throw new TypeError("VectorDBMemory filter must be an object");
    if (filter !== undefined && this.#stored.item_schema === null) {
      throw new TypeError("this memory's items are text only; they have no fields to filter");
    }
    if (k > limits.maxK) throw refusal(`k is ${k}; search returns at most ${limits.maxK} items`);
    if (query === "") throw refusal("query is empty");
    const chars = charCount(query);
    if (chars > limits.maxTextChars) {
      throw refusal(`the query is ${chars} characters; it may be at most ${limits.maxTextChars}`);
    }
    const wanted = Object.entries(filter ?? {}).map(([key, value]) => [key, canonicalJson(value)] as const);
    const [vector] = await this.#embed([query], signal);
    const queryNorm = norm(vector!);
    return this.#locked(signal, (db) => {
      const best: { id: number; score: number }[] = [];
      const page = db.prepare(
        "SELECT id, fields, vector FROM items WHERE retired IS NULL AND id > ? ORDER BY id LIMIT ?",
      );
      for (let after = 0; ; ) {
        const rows = page.all(after, SEARCH_CHUNK_ROWS);
        for (const row of rows) {
          const id = row.id as number;
          after = id;
          const item = vectorOf(row.vector as Uint8Array);
          if (item.length !== vector!.length) {
            throw new Error(
              `the query's embedding has ${vector!.length} dimensions, but this memory's items have ${item.length}; ` +
                "re-embed it with `yian memory reembed`",
            );
          }
          if (wanted.length > 0) {
            const fields = storedJson(row.fields);
            const matches = (key: string, value: string) =>
              isObject(fields) && Object.hasOwn(fields, key) && canonicalJson(fields[key]) === value;
            if (!wanted.every(([key, value]) => matches(key, value))) continue;
          }
          const score = cosine(vector!, queryNorm, item);
          // Kept best first; an equal score keeps the lower id first, since rows come in id order.
          let at = best.length;
          while (at > 0 && best[at - 1]!.score < score) at--;
          if (at < k) best.splice(at, 0, { id, score });
          if (best.length > k) best.pop();
        }
        if (rows.length < SEARCH_CHUNK_ROWS) break;
      }
      const items = best.map(({ id, score }) => {
        const { version, text, fields } = itemResult(itemRow(db, id)!, false);
        // Rounded for the model and the run log; the ranking used the exact score.
        const item: JsonObject = { id: BigInt(id), version: version!, score: Math.round(score * 10_000) / 10_000 };
        item.text = text!;
        if (fields !== undefined) item.fields = fields;
        return item;
      });
      return { items };
    });
  }

  /**
   * Return the vectors of `texts` from the embedding model the store holds.
   *
   * @throws Error when the memory was built with no embed function, or the embedder breaks its
   *   contract; the embedder's own errors pass through.
   */
  async #embed(texts: readonly string[], signal: AbortSignal): Promise<Float32Array[]> {
    const { embed } = this.#store;
    if (embed === undefined) {
      throw new Error("this VectorDBMemory has no embed function; build it with one to search or write");
    }
    const { provider, model } = this.#stored.embedding;
    return checkVectors(await embed(texts, { provider, model, signal }), texts.length);
  }

  /**
   * Check that the store still holds the memory this session opened: its identity, and the
   * settings its vectors depend on.
   *
   * @throws Error when it was replaced or re-embedded since.
   */
  #checkStore(db: Sqlite): void {
    const { identity } = this.info;
    const meta = readMeta(db, "the memory's store");
    if (meta === null || meta.identity !== identity) {
      const now = meta === null ? "nothing" : shown(meta.identity);
      throw new Error(
        `this memory's store now holds ${now}, not ${shown(identity)}: it was replaced while this session had it open`,
      );
    }
    if (modelName(meta.settings.embedding) !== modelName(this.#stored.embedding)) {
      throw new Error(
        `this memory was re-embedded with ${modelName(meta.settings.embedding)} while this session had it open; ` +
          "open it again",
      );
    }
    if (settingsText(meta.settings) !== settingsText(this.#stored)) {
      throw new Error("this memory's settings changed while this session had it open; open it again");
    }
  }

  /** Throw when the session is closed. */
  #checkOpen(): void {
    if (this.#closed) throw new Error("this VectorDBMemory session is closed");
  }

  /** Return `promise`, keeping it among the pending calls `close` waits for until it settles. */
  #track<T>(promise: Promise<T>): Promise<T> {
    const settled = promise.then(
      () => undefined,
      () => undefined,
    );
    this.#pending.add(settled);
    void settled.then(() => this.#pending.delete(settled));
    return promise;
  }

  /**
   * Run `body` on the store holding the memory's lock, in this process's turn on the store, after
   * checking the store is still the session's. Its errors name no location.
   */
  #locked<T>(signal: AbortSignal | undefined, body: OnStore<T>): Promise<T> {
    const { path } = this.#store;
    const run = () =>
      withDb(path, (db) => {
        this.#checkStore(db);
        return body(db);
      });
    const { result, done } = this.#store.locked(signal, run as Held<T>);
    // The turn itself, not the call: an abort while queued lets the call go before its turn runs.
    this.#track(done);
    return this.#track(
      result.catch((error: unknown) => {
        throw withoutLocation(error, path);
      }),
    );
  }
}
