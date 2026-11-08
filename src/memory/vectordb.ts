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
 * Each change is one `IMMEDIATE` transaction that writes its item, its history row and its id row
 * together, so a crash leaves it whole or not at all, and a change sent again with its id returns
 * the stored result.
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
import { inputTypeFromJsonSchema } from "../tools/types.ts";
import { type JsonObject, type JsonValue, logJson, parseJson } from "../typesys/json.ts";
import { buildValueAs, TypeCheckError } from "../typesys/values.ts";
import {
  type Applied,
  type Author,
  type ChangeEntry,
  ChangeIdReusedError,
  type Erased,
  MemoryBase,
  MemoryBusyError,
  type MemoryContract,
  type MemoryInfo,
  type MemorySession,
  type MemoryToolSpec,
  type Recall,
} from "./base.ts";
import { asAuthor, charCount, isObject, openRegular } from "./file_journal.ts";

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
  /**
   * Characters of one item's text and fields together (the fields as yian JSON), of a retire's
   * reason, and of a search query.
   */
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
 * pages; a rollback journal deleted after each commit; `EXTRA` syncs; and temporary tables in
 * memory, so no copy of a text lands in a temporary file.
 *
 * With a rollback journal, a transaction commits when its journal is deleted. `FULL` doesn't sync
 * the folder after that, so a power loss could bring the journal back and roll back a change
 * already reported as applied; `EXTRA` syncs it.
 */
export function openStore(path: string): Sqlite {
  const db = openSqlite(path);
  try {
    db.exec("PRAGMA busy_timeout = 0");
    db.exec("PRAGMA secure_delete = 1");
    db.exec("PRAGMA journal_mode = DELETE");
    db.exec("PRAGMA synchronous = EXTRA");
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

/**
 * Make the store's tables in `db`, which has none, record its identity and settings, and return
 * its `meta` read back. A session takes its settings from that copy, as every later open does, so
 * the open that makes a store gives the same specs as the rest, and a caller that changes its
 * own objects afterwards changes nothing.
 */
function initStore(db: Sqlite, path: string, identity: string, settings: StoredSettings): Meta {
  transaction(db, () => {
    db.exec(SCHEMA);
    const insert = db.prepare("INSERT INTO meta(key, value) VALUES (?, ?)");
    insert.run("identity", identity);
    insert.run("type", "VectorDBMemory");
    insert.run("settings", settingsText(settings));
    insert.run("store_version", String(STORE_VERSION));
    insert.run("version", "0");
  });
  return readMeta(db, path)!;
}

/** Return why an input typed `given` can't bind a store whose items are typed `stored`, or `null` when it can. */
function itemTypeProblem(stored: JsonObject | null, given: JsonObject | null): string | null {
  // Compared as the store writes them: keys in order, and floats written as floats.
  if (logJson(stored) === logJson(given)) return null;
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

/** A VectorDBMemory change, checked for shape: a write tool's arguments plus `op`, its name. */
type Change =
  | { op: "add"; text: string; fields: JsonObject | null }
  | { op: "edit"; id: number; if_version: number; text: string | undefined; fields: JsonObject | undefined }
  | { op: "retire"; id: number; if_version: number; why: string }
  | { op: "delete"; id: number; if_version: number };

/** The fields each change takes besides `op`, in order. */
const CHANGE_FIELDS: Readonly<Record<Change["op"], readonly string[]>> = Object.freeze({
  add: ["text", "fields"],
  edit: ["id", "if_version", "text", "fields"],
  retire: ["id", "if_version", "why"],
  delete: ["id", "if_version"],
});

/** Return `names` joined for a message: `a`, `a and b`, `a, b and c`. */
function listed(names: readonly string[]): string {
  return names.length < 2 ? names.join("") : `${names.slice(0, -1).join(", ")} and ${names.at(-1)}`;
}

/**
 * Return `fields` as the item schema `schema` types them: a copy of the value the check built.
 *
 * @throws {TypeError} If it isn't an object, or doesn't fit, naming where.
 */
function checkFields(op: string, fields: unknown, schema: JsonObject): JsonObject {
  if (!isObject(fields)) throw new TypeError(`a VectorDBMemory ${op}'s fields must be an object`);
  // The check a tool call's arguments get; a schema too deep to translate isn't checked.
  const type = inputTypeFromJsonSchema(schema);
  if (type === null) return fields;
  try {
    return buildValueAs(type, fields).value as JsonObject;
  } catch (error) {
    if (!(error instanceof TypeCheckError)) throw error;
    throw new TypeError(`a VectorDBMemory ${op}'s fields don't fit the item type: ${error.message}`);
  }
}

/**
 * Return the digest of the checked `change`, which a repeated change id must match: the sha256 of
 * its yian JSON, so ints and floats both write, an int given as a number or a bigint digests the
 * same, and an absent field is left out.
 */
function changeDigest(change: Change): string {
  const record: JsonObject = { op: change.op };
  for (const [key, value] of Object.entries(change)) {
    if (key === "op" || value === undefined || value === null) continue;
    record[key] = typeof value === "number" ? BigInt(value) : (value as JsonValue);
  }
  return sha256(logJson(record));
}

/**
 * Return whether the JSON values `a` and `b` are equal: numbers by value, whether ints or floats,
 * lists item by item and objects key by key, in any key order.
 */
function sameValue(a: JsonValue, b: JsonValue): boolean {
  if ((typeof a === "number" || typeof a === "bigint") && (typeof b === "number" || typeof b === "bigint")) {
    if (typeof a === typeof b) return a === b;
    const [int, float] = typeof a === "bigint" ? [a, b as number] : [b as bigint, a as number];
    return Number.isInteger(float) && BigInt(float) === int;
  }
  if (Array.isArray(a)) return Array.isArray(b) && a.length === b.length && a.every((x, i) => sameValue(x, b[i]!));
  if (isObject(a)) {
    if (!isObject(b)) return false;
    const keys = Object.keys(a);
    if (keys.length !== Object.keys(b).length) return false;
    return keys.every((key) => Object.hasOwn(b, key) && sameValue(a[key]!, b[key]!));
  }
  return a === b;
}

/**
 * Return `change` as a VectorDBMemory change, its ints as numbers, on a memory whose items are
 * typed by `schema` (`null` for text only).
 *
 * @throws {TypeError} If its `op` isn't a VectorDBMemory change, or its fields aren't that
 *   change's, each of its type; or its item fields are given to a memory of text, missing from an
 *   add to a typed memory, or don't fit the item type.
 */
function checkChange(change: JsonObject, schema: JsonObject | null): Change {
  const op: unknown = isObject(change) ? change.op : undefined;
  if (typeof op !== "string" || !Object.hasOwn(CHANGE_FIELDS, op)) {
    const ops = Object.keys(CHANGE_FIELDS).join(", ");
    throw new TypeError(`VectorDBMemory has no change ${String(op)}; its changes are ${ops}`);
  }
  const allowed = CHANGE_FIELDS[op as Change["op"]];
  const extra = Object.keys(change).filter((key) => key !== "op" && !allowed.includes(key));
  if (extra.length > 0) throw new TypeError(`a VectorDBMemory ${op} takes ${listed(allowed)}, not ${listed(extra)}`);
  const has = (key: string) => change[key] !== undefined;
  const text = (key: string): string => {
    const value = change[key];
    if (typeof value !== "string" || !value.isWellFormed()) {
      throw new TypeError(`a VectorDBMemory ${op} needs ${key} as a string`);
    }
    return value;
  };
  const int = (key: string): number => {
    const value = change[key];
    const number = typeof value === "bigint" ? Number(value) : value;
    if (typeof number !== "number" || !Number.isSafeInteger(number) || number < 1) {
      throw new TypeError(`a VectorDBMemory ${op} needs ${key} as an integer of at least 1`);
    }
    return number;
  };
  const fields = (): JsonObject | undefined => {
    if (!has("fields")) return undefined;
    if (schema === null) throw new TypeError(`this memory's items are text only; an ${op} takes no fields`);
    return checkFields(op, change.fields, schema);
  };
  switch (op as Change["op"]) {
    case "add": {
      const checked = { op: "add" as const, text: text("text"), fields: fields() ?? null };
      if (checked.fields === null && schema !== null) {
        throw new TypeError("an add to this memory needs fields, since its items are typed");
      }
      return checked;
    }
    case "edit": {
      const checked = {
        op: "edit" as const,
        id: int("id"),
        if_version: int("if_version"),
        text: has("text") ? text("text") : undefined,
        fields: fields(),
      };
      if (checked.text === undefined && checked.fields === undefined) {
        throw new TypeError("an edit needs text, fields or both");
      }
      return checked;
    }
    case "retire":
      return { op: "retire", id: int("id"), if_version: int("if_version"), why: text("why") };
    case "delete":
      return { op: "delete", id: int("id"), if_version: int("if_version") };
  }
}

/**
 * Return `author` with only its own fields.
 *
 * @throws {TypeError} If it isn't `{kind, name, run?, node?}` with a known kind.
 */
function checkAuthor(author: Author): Author {
  const checked = asAuthor(author);
  if (checked === null) {
    throw new TypeError("a change's author must be {kind, name, run?, node?}, its kind agent, tool, user or host");
  }
  return checked;
}

/** Return a refusal that wrote nothing. */
function refused(reason: string): Applied {
  return { ok: false, reason, current: null };
}

/** Return a conflict over the item `row`, holding its current state as `read` shows it. */
function conflict(reason: string, row: ItemRow): Applied {
  return { ok: false, reason, current: itemResult(row, true) };
}

/**
 * Return why an item can't hold `text` and `fields` (as the store writes them, `null` for none)
 * under `limits`, or `null` when it can. They count together, so what a read shows stays bounded.
 */
function itemProblem(text: string, fields: string | null, limits: Limits): string | null {
  if (text === "") return "an item's text can't be empty";
  const chars = charCount(text) + (fields === null ? 0 : charCount(fields));
  if (chars <= limits.maxTextChars) return null;
  const what = fields === null ? "the text is" : "the item's text and fields are";
  return `${what} ${chars} characters; an item may hold ${limits.maxTextChars}`;
}

/** What `history.before` holds for a change that made its item: undoing it removes the item. */
const ABSENT_STATE = '{"absent":true}';

/** Return what `history.before` holds for an item that was `row`: undoing the change writes it back. */
function stateOf(row: ItemRow): string {
  return logJson({ text: row.text, fields: storedJson(row.fields), retired: row.retired });
}

/**
 * Return the stored result of change `id`, or `null` when the table doesn't hold it.
 *
 * @throws {ChangeIdReusedError} If it holds `id` for a change with another digest.
 */
function knownChange(db: Sqlite, id: string, digest: string): Applied | null {
  const row = db.prepare("SELECT digest, result FROM changes WHERE id = ?").get(id);
  if (row === undefined) return null;
  if (row.digest !== digest) throw new ChangeIdReusedError(id);
  const stored = storedJson(row.result);
  if (!isObject(stored) || stored.ok !== true || typeof stored.version !== "string" || !isObject(stored.result)) {
    throw new Error("this memory's store has a damaged change-id table");
  }
  return { ok: true, version: stored.version, result: stored.result };
}

/** One change as `history` records it. */
interface HistoryEntry {
  /** The memory's version after the change. */
  readonly seq: number;
  readonly item: number;
  readonly op: string;
  /** What undoing it writes back, or `null` when it can't be undone. */
  readonly before: string | null;
  /** The item's version after it, or `null` when it left no item. */
  readonly after: number | null;
  /** For an undo, the id of the change it undid. */
  readonly undoes: string | null;
}

/** A change being written: its id, its digest and its author. */
interface Writing {
  readonly id: string;
  readonly digest: string;
  readonly author: Author;
}

/**
 * Drop the table a reembed keeps its new vectors in. Every caller holds the memory's lock, which a
 * reembed holds until it ends, so a table found then is one a crash left part way: its vectors are
 * derived from texts, including ones a delete or an erase has since removed.
 */
function dropStaging(db: Sqlite): void {
  db.exec("DROP TABLE IF EXISTS reembed");
}

/**
 * Record change `writing` in the transaction open on `db`: its history row, its id row with the
 * result it returns, and the new version. Then drop the old text, history and ids the caps no
 * longer keep, and what a crashed reembed left. Return the outcome.
 */
function record(db: Sqlite, limits: Limits, at: string, writing: Writing, entry: HistoryEntry, result: JsonObject) {
  const { seq } = entry;
  dropStaging(db);
  const applied: Applied = { ok: true, version: String(seq), result };
  db.prepare(
    "INSERT INTO history(seq, change, item, op, before, after, author, at, undoes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
  ).run(
    seq,
    writing.id,
    entry.item,
    entry.op,
    // With no undos kept, the old text is never written.
    limits.maxUndo === 0 ? null : entry.before,
    entry.after,
    logJson(writing.author),
    at,
    entry.undoes,
  );
  db.prepare("INSERT INTO changes(id, digest, result, seq) VALUES (?, ?, ?, ?)").run(
    writing.id,
    writing.digest,
    logJson(applied),
    seq,
  );
  db.prepare("UPDATE meta SET value = ? WHERE key = 'version'").run(String(seq));
  db.prepare("DELETE FROM changes WHERE seq <= ?").run(seq - limits.maxChangeIds);
  db.prepare("UPDATE history SET before = NULL WHERE seq <= ? AND before IS NOT NULL").run(seq - limits.maxUndo);
  db.prepare("DELETE FROM history WHERE seq <= ?").run(seq - Math.max(limits.maxUndo, limits.maxChangeIds));
  return applied;
}

/**
 * Check that `vector` is as long as the store's vectors.
 *
 * @throws Error naming both lengths, since the store's vectors came from another model.
 */
function checkDimensions(db: Sqlite, vector: Float32Array): void {
  const dimensions = storedDimensions(db);
  if (dimensions !== null && dimensions !== vector.length) {
    throw new Error(
      `the embedder returned vectors of ${vector.length} dimensions, but this memory's items have ${dimensions}; ` +
        "re-embed it with `yian memory reembed`",
    );
  }
}

/** A history row as the store holds it. */
interface HistoryRow {
  readonly seq: number;
  readonly change: string;
  readonly item: number;
  readonly op: string;
  readonly before: string | null;
  readonly after: number | null;
}

/** Return the latest history row of change `changeId`, or of any change to item `item`; `undefined` when none. */
function latestChange(db: Sqlite, where: { change: string } | { item: number }): HistoryRow | undefined {
  const [column, value] = "change" in where ? ["change", where.change] : ["item", where.item];
  return db
    .prepare(`SELECT seq, change, item, op, before, after FROM history WHERE ${column} = ? ORDER BY seq DESC LIMIT 1`)
    .get(value) as HistoryRow | undefined;
}

/**
 * Return whether `change` can still be undone: it keeps what it replaced, no later change touched
 * its item, and the item is what it left (at its version, or gone).
 */
function undoable(db: Sqlite, change: HistoryRow, row: ItemRow | undefined): boolean {
  if (change.before === null) return false;
  if (latestChange(db, { item: change.item })?.seq !== change.seq) return false;
  return change.after === null ? row === undefined : row?.version === change.after;
}

/** An item's state as `history.before` holds it: `null` for no item. */
type ItemState = { text: string; fields: JsonValue; retired: string | null } | null;

/**
 * Return the state `history.before` text holds.
 *
 * @throws Error when it isn't one this code wrote.
 */
function parseState(text: string): ItemState {
  const state = storedJson(text);
  if (isObject(state) && state.absent === true) return null;
  if (
    !isObject(state) ||
    typeof state.text !== "string" ||
    !(state.retired === null || typeof state.retired === "string") ||
    state.fields === undefined
  ) {
    throw new Error("this memory's store has a damaged history row");
  }
  return { text: state.text, fields: state.fields, retired: state.retired };
}

/** What `item` is under `target` for an erase: the item id. */
function checkEraseTarget(target: JsonObject): number {
  if (!isObject(target)) throw new TypeError("an erase needs item as an integer of at least 1");
  const extra = Object.keys(target).filter((key) => key !== "item");
  if (extra.length > 0) throw new TypeError(`an erase takes item, not ${listed(extra)}`);
  const value = target.item;
  const item = typeof value === "bigint" ? Number(value) : value;
  if (typeof item !== "number" || !Number.isSafeInteger(item) || item < 1) {
    throw new TypeError("an erase needs item as an integer of at least 1");
  }
  return item;
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
  /**
   * Run the async `body` holding the memory's lock for as long as it runs, as `locked` runs a
   * synchronous one. Only host actions do this: every other writer waits for it, or times out.
   */
  hold<T>(signal: AbortSignal, body: () => Promise<T>): Promise<T>;
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

/** The error a call gets when its wait for the memory's lock runs past its timeout. */
function busy(): MemoryBusyError {
  // No cause: a lock's error names its path, and this one can reach a model.
  return new MemoryBusyError("this memory is busy: another writer holds its lock");
}

/**
 * Call `fn` once `ms` real milliseconds have passed, unless `signal` aborts first; never for an
 * infinite `ms`. Even 0 ms waits for a macrotask: the turns ahead of a queued call settle in
 * microtasks unless one of them waits, so only a real wait makes a queued call late.
 */
function afterMs(ms: number, signal: AbortSignal, fn: () => void): void {
  if (!Number.isFinite(ms)) return;
  const wait = ms > 0 ? realScheduler.sleep(ms, signal) : new Promise<void>((resolve) => setTimeout(resolve, 0));
  wait.then(
    () => {
      if (!signal.aborted) fn();
    },
    () => undefined,
  );
}

/**
 * Run `fn` once this process's earlier turns on the store at `path` have finished.
 *
 * A call leaves the queue before its turn when `signal` aborts, rejecting with its reason, or
 * when it has waited `timeoutMs`, rejecting with `MemoryBusyError`, even while the turn ahead
 * still runs. Its turn, when it comes, does nothing, so the calls after it keep their order. Once
 * its turn has started, `fn` answers the abort and the timeout itself, so a change it applies is
 * never reported as aborted.
 */
function inTurn<T>(path: string, signal: AbortSignal | undefined, timeoutMs: number, fn: () => Promise<T>): Turn<T> {
  let started = false;
  /** Why the call left the queue before its turn, once it has. */
  let left: { reason: unknown } | null = null;
  // Aborted when the turn starts or the call settles, which stops the queue's timer.
  const waiting = new AbortController();
  const run = (TURNS.get(path) ?? Promise.resolve()).then(() => {
    started = true;
    waiting.abort();
    if (left !== null) throw left.reason;
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
  const result = new Promise<T>((resolve, reject) => {
    const leave = (reason: unknown) => {
      if (started || left !== null) return;
      left = { reason };
      reject(reason);
    };
    const onAbort = () => leave(signal!.reason);
    signal?.addEventListener("abort", onAbort, { once: true });
    // A signal aborted already never fires its listeners.
    if (signal?.aborted) onAbort();
    afterMs(timeoutMs, waiting.signal, () => leave(busy()));
    void run.then(resolve, reject).finally(() => {
      signal?.removeEventListener("abort", onAbort);
      waiting.abort();
    });
  });
  return { result, done };
}

/**
 * Return `promise`, or reject with `signal`'s reason as soon as it aborts, for an embedder that
 * doesn't heed its signal: a call waiting on it would otherwise never end, and a reembed would
 * keep the memory's lock.
 */
function untilAborted<T>(promise: Promise<T>, signal: AbortSignal): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(signal.reason);
    signal.addEventListener("abort", onAbort, { once: true });
    if (signal.aborted) onAbort();
    promise.then(resolve, reject).finally(() => signal.removeEventListener("abort", onAbort));
  });
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
    /** Run `body` holding the lock, in this process's turn on the store, as `Store.locked` describes. */
    const turn = <T>(signal: AbortSignal | undefined, body: () => T | Promise<T>): Turn<T> => {
      // A monotonic clock, so a wall-clock step while queued neither stretches nor cuts the wait.
      const deadline = performance.now() + timeoutMs;
      return inTurn(real, signal, timeoutMs, async () => {
        // Checked before the lock file is opened, and on every call: the files may change.
        checkFiles(real);
        // Only what is left of the timeout: a call waits it out once, queued or not.
        const left = Math.max(0, deadline - performance.now());
        let lock: FileLock;
        try {
          lock = await waitForLock(`${real}.lock`, { timeoutMs: left, signal, scheduler, private: true });
        } catch (error) {
          if (error instanceof FileLockError && error.reason === "busy") throw busy();
          throw error;
        }
        try {
          // Again holding the lock: the files may have changed while the call waited for it.
          checkFiles(real);
          return await body();
        } finally {
          lock.release();
        }
      });
    };
    this.#store = Object.freeze({
      path: real,
      limits,
      scheduler,
      embedding,
      embed: options.embed,
      locked: <T>(signal: AbortSignal | undefined, body: Held<T>): Turn<T> => turn(signal, body),
      hold: <T>(signal: AbortSignal, body: () => Promise<T>): Promise<T> => turn(signal, body).result,
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
          const identity = this.#id ?? `at-${sha256(path).slice(0, 16)}`;
          return initStore(db, path, identity, { embedding, item_schema: itemType });
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
    return new VectorDBSession(this.#store, sessionInfo(meta, this.#store.limits), stored, meta.version, created);
  }

  /**
   * Empty the memory, keeping its identity and settings: delete every item, the history and the
   * change ids in one transaction, and move the version on. Item ids aren't used again. Only a
   * host or the CLI calls it, outside runs; a session open on the memory goes on with it empty.
   *
   * @throws Error when the store isn't a memory, or holds another identity than the constructor's `id`.
   * @throws {MemoryBusyError} If another writer holds the lock past the timeout.
   */
  override async reset(o: { signal: AbortSignal }): Promise<void> {
    o.signal.throwIfAborted();
    const { path } = this.#store;
    if (lstatOrNull(path) === null) throw new Error(`${shown(path)} isn't a memory yet`);
    await this.#store.locked(o.signal, (): void => {
      withDb(path, (db) => {
        const meta = memoryMeta(db, path, this.#id);
        transaction(db, () => {
          // With secure_delete on, the deleted texts are overwritten, not left in free pages.
          db.exec("DELETE FROM items; DELETE FROM history; DELETE FROM changes");
          dropStaging(db);
          db.prepare("UPDATE meta SET value = ? WHERE key = 'version'").run(String(meta.version + 1));
        });
      });
    }).result;
  }

  /**
   * Embed every item's text again with `embedding` through `embed`, retired items included, then
   * record the new model in `meta`. It holds the memory's lock throughout, so no writer adds an
   * item with the old model meanwhile; the texts go in batches of `current().embeddingBatchItems`.
   * The new vectors wait in a table of their own until the last batch is in, then replace the old
   * ones in one transaction with the new model, so a failure or a crash part way leaves the store
   * as it was. Items keep their versions. A session open on the memory refuses its next call; open
   * it again with the new model.
   *
   * It is a host action (`yian memory reembed`), never a session method: no run re-embeds.
   *
   * @throws {TypeError} If `embedding` isn't `{provider, model}` or `embed` isn't a function.
   * @throws Error when the store isn't a memory or holds another identity, or the embedder fails or
   *   breaks its contract.
   * @throws {MemoryBusyError} If another writer holds the lock past the timeout.
   */
  async reembed(o: { embedding: EmbeddingModel; embed: Embed; signal: AbortSignal }): Promise<void> {
    o.signal.throwIfAborted();
    const embedding = checkEmbedding(o.embedding);
    if (typeof o.embed !== "function") throw new TypeError("reembed needs embed as a function");
    const { path } = this.#store;
    if (lstatOrNull(path) === null) throw new Error(`${shown(path)} isn't a memory yet`);
    await this.#store.hold(o.signal, async () => {
      if (lstatOrNull(path) === null) throw new Error("this memory's store file is gone");
      // One connection for the whole action: it holds no transaction while it waits for the embedder.
      const db = openStore(path);
      let staged = false;
      try {
        const meta = memoryMeta(db, path, this.#id);
        // A table a crash left part way through an earlier reembed goes first.
        db.exec("DROP TABLE IF EXISTS reembed; CREATE TABLE reembed(id INTEGER PRIMARY KEY, vector BLOB NOT NULL)");
        staged = true;
        const batch = db.prepare("SELECT id, text FROM items WHERE id > ? ORDER BY id LIMIT ?");
        let length: number | null = null;
        for (let after = 0; ; ) {
          // Read per batch, so a run's scoped defaults apply, as the embeddings client reads them.
          const rows = batch.all(after, current().embeddingBatchItems);
          if (rows.length === 0) break;
          const texts = rows.map((row) => row.text as string);
          const vectors = checkVectors(
            await untilAborted(o.embed(texts, { ...embedding, signal: o.signal }), o.signal),
            texts.length,
          );
          length ??= vectors[0]!.length;
          if (vectors[0]!.length !== length) throw new Error("the embedder returned vectors of different lengths");
          transaction(db, () => {
            const insert = db.prepare("INSERT INTO reembed(id, vector) VALUES (?, ?)");
            for (const [i, row] of rows.entries()) insert.run(row.id as number, bytesOf(vectors[i]!));
          });
          after = rows.at(-1)!.id as number;
        }
        transaction(db, () => {
          // Every item has its new vector: the lock kept any from being added or removed meanwhile.
          db.exec("UPDATE items SET vector = (SELECT vector FROM reembed WHERE reembed.id = items.id)");
          db.prepare("UPDATE meta SET value = ? WHERE key = 'settings'").run(
            settingsText({ embedding, item_schema: meta.settings.item_schema }),
          );
          db.exec("DROP TABLE reembed");
        });
      } finally {
        // A failed reembed's half-filled table goes, so the store is as it was.
        try {
          if (staged && !db.inTransaction) dropStaging(db);
        } finally {
          db.close();
        }
      }
    });
  }

  /**
   * Rebuild the store file with SQLite's `VACUUM`, holding the memory's lock, so the space deleted
   * items and history took is given back. A host action (`yian memory compact`), never a session
   * method. Temporary tables stay in memory, so no copy of a text lands in a temporary file.
   *
   * @throws Error when the store isn't a memory, or holds another identity than the constructor's `id`.
   * @throws {MemoryBusyError} If another writer holds the lock past the timeout.
   */
  async compact(o: { signal: AbortSignal }): Promise<void> {
    o.signal.throwIfAborted();
    const { path } = this.#store;
    if (lstatOrNull(path) === null) throw new Error(`${shown(path)} isn't a memory yet`);
    await this.#store.locked(o.signal, (): void => {
      withDb(path, (db) => {
        memoryMeta(db, path, this.#id);
        dropStaging(db);
        db.exec("VACUUM");
      });
    }).result;
  }
}

/**
 * Return the store's `meta` after checking it is a memory, and the one `id` names when given.
 *
 * @throws Error when it isn't a memory yet, or holds another identity.
 */
function memoryMeta(db: Sqlite, path: string, id: string | undefined): Meta {
  const meta = readMeta(db, path);
  if (meta === null) throw new Error(`${shown(path)} isn't a memory yet`);
  if (id !== undefined && meta.identity !== id) {
    throw new Error(`this memory is ${shown(meta.identity)}, not ${shown(id)}`);
  }
  return meta;
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
  /** The highest version the session has seen the store at; a store found older was restored. */
  #seen: number;
  #closed = false;

  /**
   * @param store The memory it reads and writes.
   * @param info What the memory is, as its recorded value holds it.
   * @param stored The settings its store held when it opened.
   * @param version The store's version when it opened.
   * @param created Whether the open that made this session created the memory.
   */
  constructor(store: Store, info: MemoryInfo, stored: StoredSettings, version: number, created: boolean) {
    this.#store = store;
    this.#stored = stored;
    this.#seen = version;
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

  /**
   * Apply one change, unless its id is known: then return the stored outcome without embedding.
   *
   * Holding the lock, it checks the id, the precondition and the caps; a refusal (`ok: false`)
   * writes nothing. A change that needs its text embedded releases the lock, embeds, and takes it
   * again to check the id and the precondition once more and write. Each write is one transaction:
   * the item, its history row and its id row.
   *
   * @throws {ChangeIdReusedError} If `id` was applied with a different change.
   * @throws {MemoryBusyError} If another writer holds the lock past the timeout.
   * @throws {TypeError} If `id`, `change` or `author` has the wrong shape.
   * @throws Error when the embedder fails, and nothing is written.
   */
  async apply(c: { id: string; change: JsonObject; author: Author; signal: AbortSignal }): Promise<Applied> {
    this.#checkOpen();
    c.signal.throwIfAborted();
    if (typeof c.id !== "string" || c.id === "") throw new TypeError("a change id must be a non-empty string");
    const author = checkAuthor(c.author);
    const change = checkChange(c.change, this.#stored.item_schema);
    const writing: Writing = { id: c.id, digest: changeDigest(change), author };
    return this.#track(this.#apply(writing, change, c.signal));
  }

  /**
   * Undo change `changeId` as a new change `id`: write back the item it replaced, if no later
   * change touched the item and the memory still keeps what it replaced. Otherwise it is refused
   * or conflicts, writing nothing. A restored text is embedded again outside the lock, as an edit's
   * is. An undo is in history and can itself be undone; a delete or an erase can't be.
   *
   * @throws {ChangeIdReusedError} If `id` was applied with a different change.
   * @throws {MemoryBusyError} If another writer holds the lock past the timeout.
   * @throws {TypeError} If `id`, `changeId` or `author` has the wrong shape.
   * @throws Error when the embedder fails, and nothing is written.
   */
  async undo(o: { id: string; changeId: string; author: Author; signal: AbortSignal }): Promise<Applied> {
    this.#checkOpen();
    o.signal.throwIfAborted();
    if (typeof o.id !== "string" || o.id === "") throw new TypeError("a change id must be a non-empty string");
    if (typeof o.changeId !== "string") throw new TypeError("undo needs the change id to undo as a string");
    const author = checkAuthor(o.author);
    const writing: Writing = { id: o.id, digest: sha256(logJson({ op: "undo", change: o.changeId })), author };
    return this.#track(this.#undo(writing, o.changeId, o.signal));
  }

  /**
   * Erase as change `id`: remove the item `target.item` and every history row of it, which hold
   * its old texts, in one transaction. It can't be undone, and it reports the change ids whose
   * undo it disabled: the item's last change, when it could still be undone. The ids' digests stay,
   * but they hold no text.
   *
   * @throws Error when the memory holds nothing of the item, neither it nor a history row of it;
   *   nothing is recorded, so the id stays free.
   * @throws {ChangeIdReusedError} If `id` was applied with a different change.
   * @throws {MemoryBusyError} If another writer holds the lock past the timeout.
   * @throws {TypeError} If `id`, `target` or `author` has the wrong shape.
   */
  async erase(o: { id: string; target: JsonObject; author: Author; signal: AbortSignal }): Promise<Erased> {
    this.#checkOpen();
    o.signal.throwIfAborted();
    if (typeof o.id !== "string" || o.id === "") throw new TypeError("a change id must be a non-empty string");
    const author = checkAuthor(o.author);
    const item = checkEraseTarget(o.target);
    const writing: Writing = { id: o.id, digest: sha256(logJson({ op: "erase", item: BigInt(item) })), author };
    const applied = await this.#locked(o.signal, (db) => {
      const known = knownChange(db, writing.id, writing.digest);
      if (known !== null) return known;
      const row = itemRow(db, item);
      const last = latestChange(db, { item });
      // An erase that removes nothing isn't a change: as for a path that isn't a member of a folder.
      if (row === undefined && last === undefined) {
        throw new Error(`item ${item} isn't in the memory: there is nothing to erase`);
      }
      const disabled = last !== undefined && undoable(db, last, row) ? [last.change] : [];
      const at = new Date(this.#store.scheduler.now()).toISOString();
      return transaction(db, () => {
        const seq = storedVersion(db) + 1;
        db.prepare("DELETE FROM items WHERE id = ?").run(item);
        db.prepare("DELETE FROM history WHERE item = ?").run(item);
        const entry: HistoryEntry = { seq, item, op: "erase", before: null, after: null, undoes: null };
        const result = { item: BigInt(item), version: BigInt(seq), disabled_undos: disabled };
        return record(db, this.#store.limits, at, writing, entry, result);
      });
    });
    if (!applied.ok) throw new Error(applied.reason);
    return { version: applied.version, disabledUndos: applied.result.disabled_undos as string[] };
  }

  /**
   * List the last `limit` changes the history holds, newest first.
   *
   * @throws {RangeError} If `limit` isn't an integer of at least 0.
   * @throws {MemoryBusyError} If another writer holds the lock past the timeout.
   */
  async history(o: { limit: number }): Promise<ChangeEntry[]> {
    this.#checkOpen();
    if (!Number.isSafeInteger(o.limit) || o.limit < 0) {
      throw new RangeError(`history needs a limit of 0 or more changes, not ${String(o.limit)}`);
    }
    return this.#locked(undefined, (db) =>
      db
        .prepare("SELECT change, item, op, author, at, undoes FROM history ORDER BY seq DESC LIMIT ?")
        .all(o.limit)
        .map((row) => {
          const target: JsonObject = { item: BigInt(row.item as number) };
          if (typeof row.undoes === "string") target.undoes = row.undoes;
          const author = asAuthor(storedJson(row.author));
          if (author === null) throw new Error("this memory's store has a damaged history row");
          return { id: row.change as string, op: row.op as string, target, author, at: row.at as string };
        }),
    );
  }

  /** Close the session once every call and turn it started has settled; later calls reject. */
  async close(): Promise<void> {
    this.#closed = true;
    // A change queues its second turn after its first settles, so wait until none is left.
    while (this.#pending.size > 0) await Promise.all([...this.#pending]);
  }

  /** Apply `change` as `writing`: check and write at once, or check, embed, then check again and write. */
  async #apply(writing: Writing, change: Change, signal: AbortSignal): Promise<Applied> {
    const first = await this.#locked(signal, (db): Applied | { embed: string } => {
      const known = knownChange(db, writing.id, writing.digest);
      if (known !== null) return known;
      const plan = this.#plan(db, change);
      if ("ok" in plan) return plan;
      return plan.embed === null ? this.#write(db, writing, change, plan, null) : { embed: plan.embed };
    });
    if ("ok" in first) return first;
    // Embedded outside the lock: a slow embedding call mustn't hold up the memory's other writers.
    const [vector] = await this.#embed([first.embed], signal);
    return this.#locked(signal, (db) => {
      const known = knownChange(db, writing.id, writing.digest);
      if (known !== null) return known;
      const plan = this.#plan(db, change);
      if ("ok" in plan) return plan;
      return this.#write(db, writing, change, plan, plan.embed === null ? null : vector!);
    });
  }

  /**
   * Check `change` against the store: the caps, the item it names and its precondition. Return
   * the refusal or conflict, or the item it changes and the text to embed for it (`null` when its
   * vector stays).
   */
  #plan(db: Sqlite, change: Change): Applied | { row: ItemRow | null; embed: string | null } {
    const { limits } = this.#store;
    if (change.op === "add") {
      const why = itemProblem(change.text, change.fields === null ? null : logJson(change.fields), limits);
      if (why !== null) return refused(why);
      const count = Number(db.prepare("SELECT count(*) AS n FROM items").get()!.n);
      if (count >= limits.maxItems) return refused(`the memory holds ${count} items; it may hold ${limits.maxItems}`);
      return { row: null, embed: change.text };
    }
    if (change.op === "edit" && change.text === "") return refused(itemProblem("", null, limits)!);
    if (change.op === "retire" && charCount(change.why) > limits.maxTextChars) {
      return refused(`the reason is ${charCount(change.why)} characters; it may be at most ${limits.maxTextChars}`);
    }
    const text = change.op === "edit" ? change.text : undefined;
    const row = itemRow(db, change.id);
    if (row === undefined) return refused(`item ${change.id} doesn't exist`);
    if (row.version !== change.if_version) {
      return conflict(
        `item ${change.id} changed since it was read: its version is ${row.version}, not ${change.if_version}`,
        row,
      );
    }
    if (change.op === "retire" && row.retired !== null) return conflict(`item ${change.id} is already retired`, row);
    if (change.op === "edit") {
      const fields = change.fields === undefined ? row.fields : logJson(change.fields);
      const why = itemProblem(text ?? row.text, fields, limits);
      if (why !== null) return refused(why);
    }
    return { row, embed: text !== undefined && text !== row.text ? text : null };
  }

  /** Write `change`, which `#plan` passed, with `vector` as its item's new vector, in one transaction. */
  #write(db: Sqlite, writing: Writing, change: Change, plan: { row: ItemRow | null }, vector: Float32Array | null) {
    const { limits, scheduler } = this.#store;
    if (vector !== null) checkDimensions(db, vector);
    const at = new Date(scheduler.now()).toISOString();
    const author = logJson(writing.author);
    return transaction(db, () => {
      const seq = storedVersion(db) + 1;
      const { row } = plan;
      let entry: HistoryEntry;
      switch (change.op) {
        case "add": {
          const fields = change.fields === null ? null : logJson(change.fields);
          const inserted = db
            .prepare("INSERT INTO items(text, fields, version, author, retired, vector) VALUES (?, ?, ?, ?, NULL, ?)")
            .run(change.text, fields, seq, author, bytesOf(vector!));
          const item = Number(inserted.lastInsertRowid);
          entry = { seq, item, op: "add", before: ABSENT_STATE, after: seq, undoes: null };
          break;
        }
        case "edit": {
          const fields = change.fields === undefined ? row!.fields : logJson(change.fields);
          db.prepare("UPDATE items SET text = ?, fields = ?, version = ?, author = ? WHERE id = ?").run(
            change.text ?? row!.text,
            fields,
            seq,
            author,
            row!.id,
          );
          if (vector !== null) db.prepare("UPDATE items SET vector = ? WHERE id = ?").run(bytesOf(vector), row!.id);
          entry = { seq, item: row!.id, op: "edit", before: stateOf(row!), after: seq, undoes: null };
          break;
        }
        case "retire":
          db.prepare("UPDATE items SET retired = ?, version = ?, author = ? WHERE id = ?").run(
            change.why,
            seq,
            author,
            row!.id,
          );
          entry = { seq, item: row!.id, op: "retire", before: stateOf(row!), after: seq, undoes: null };
          break;
        case "delete":
          // The item's text goes from its row and from every history row that holds it.
          db.prepare("DELETE FROM items WHERE id = ?").run(row!.id);
          db.prepare("DELETE FROM history WHERE item = ?").run(row!.id);
          entry = { seq, item: row!.id, op: "delete", before: null, after: null, undoes: null };
          break;
      }
      return record(db, limits, at, writing, entry, { id: BigInt(entry.item), version: BigInt(seq) });
    });
  }

  /**
   * Undo `changeId` as `writing`: check and write at once, or check, embed the restored text, then
   * check again and write.
   */
  async #undo(writing: Writing, changeId: string, signal: AbortSignal): Promise<Applied> {
    const first = await this.#locked(signal, (db): Applied | { embed: string } => {
      const known = knownChange(db, writing.id, writing.digest);
      if (known !== null) return known;
      const plan = this.#undoPlan(db, changeId);
      if ("ok" in plan) return plan;
      return plan.embed === null ? this.#undoWrite(db, writing, plan, null) : { embed: plan.embed };
    });
    if ("ok" in first) return first;
    const [vector] = await this.#embed([first.embed], signal);
    return this.#locked(signal, (db) => {
      const known = knownChange(db, writing.id, writing.digest);
      if (known !== null) return known;
      const again = this.#undoPlan(db, changeId);
      if ("ok" in again) return again;
      // The vector must be the text's it is written with; only a hand edit of the store can make
      // the text to restore another one between the passes.
      if (again.embed !== null && again.embed !== first.embed) {
        const { item } = again.change;
        return refused(`item ${item} changed while its old text was embedded; undo change ${shown(changeId)} again`);
      }
      return this.#undoWrite(db, writing, again, again.embed === null ? null : vector!);
    });
  }

  /**
   * Check that change `changeId` can be undone now. Return the refusal or conflict, or what undoing
   * it writes: the change, the item as it is, the state to restore (`null` to remove the item), and
   * the text to embed for it (`null` when the item's vector stays or it goes).
   */
  #undoPlan(
    db: Sqlite,
    changeId: string,
  ): Applied | { change: HistoryRow; row: ItemRow | undefined; restore: ItemState; embed: string | null } {
    const shownId = shown(changeId);
    const change = latestChange(db, { change: changeId });
    if (change === undefined) return refused(`change ${shownId} isn't in the memory's history`);
    if (change.op === "delete" || change.op === "erase") {
      return refused(`change ${shownId} was ${change.op === "erase" ? "an erase" : "a delete"}, which can't be undone`);
    }
    if (change.before === null) {
      return refused(`change ${shownId} can't be undone: the memory no longer keeps what it replaced`);
    }
    const row = itemRow(db, change.item);
    if (!undoable(db, change, row)) {
      const reason = `item ${change.item} changed after change ${shownId}; undoing it would lose that`;
      return row === undefined ? refused(reason) : conflict(reason, row);
    }
    const restore = parseState(change.before);
    if (restore !== null && row === undefined) {
      const { maxItems } = this.#store.limits;
      const count = Number(db.prepare("SELECT count(*) AS n FROM items").get()!.n);
      if (count >= maxItems) return refused(`the memory holds ${count} items; it may hold ${maxItems}`);
    }
    const embed = restore !== null && (row === undefined || row.text !== restore.text) ? restore.text : null;
    return { change, row, restore, embed };
  }

  /** Write the undo `plan` passed, with `vector` as the restored item's vector, in one transaction. */
  #undoWrite(
    db: Sqlite,
    writing: Writing,
    plan: { change: HistoryRow; row: ItemRow | undefined; restore: ItemState },
    vector: Float32Array | null,
  ): Applied {
    const { limits, scheduler } = this.#store;
    if (vector !== null) checkDimensions(db, vector);
    const at = new Date(scheduler.now()).toISOString();
    const author = logJson(writing.author);
    const { change, row, restore } = plan;
    const item = change.item;
    return transaction(db, () => {
      const seq = storedVersion(db) + 1;
      if (restore === null) {
        db.prepare("DELETE FROM items WHERE id = ?").run(item);
      } else {
        const fields = restore.fields === null ? null : logJson(restore.fields);
        if (row === undefined) {
          db.prepare(
            "INSERT INTO items(id, text, fields, version, author, retired, vector) VALUES (?, ?, ?, ?, ?, ?, ?)",
          ).run(item, restore.text, fields, seq, author, restore.retired, bytesOf(vector!));
        } else {
          db.prepare("UPDATE items SET text = ?, fields = ?, retired = ?, version = ?, author = ? WHERE id = ?").run(
            restore.text,
            fields,
            restore.retired,
            seq,
            author,
            item,
          );
          if (vector !== null) db.prepare("UPDATE items SET vector = ? WHERE id = ?").run(bytesOf(vector), item);
        }
      }
      const entry: HistoryEntry = {
        seq,
        item,
        op: "undo",
        before: row === undefined ? ABSENT_STATE : stateOf(row),
        after: restore === null ? null : seq,
        undoes: change.change,
      };
      return record(db, limits, at, writing, entry, { id: BigInt(item), version: BigInt(seq) });
    });
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
    const wanted = Object.entries(filter ?? {});
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
            const matches = (key: string, value: JsonValue) =>
              isObject(fields) && Object.hasOwn(fields, key) && sameValue(fields[key]!, value);
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
    return checkVectors(await untilAborted(embed(texts, { provider, model, signal }), signal), texts.length);
  }

  /**
   * Check that the store still holds the memory this session opened: its identity, the settings
   * its vectors depend on, and a version no older than the session has seen.
   *
   * @throws Error when it was replaced, re-embedded or restored from a copy since.
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
    // A copy put back would answer for change ids it never saw, and apply them a second time.
    if (meta.version < this.#seen) {
      throw new Error("this memory's store is older than this session saw; it may have been restored from a copy");
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
        const result = body(db);
        // Read again after the body, so the session's own writes count as seen.
        this.#seen = Math.max(this.#seen, storedVersion(db));
        return result;
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
