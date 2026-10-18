/**
 * The one module holding runtime-specific APIs, so the rest of yian runs unchanged
 * on Node and Bun: runtime detection and a SQLite wrapper over `bun:sqlite` or
 * `node:sqlite`, loaded lazily for the runtime in use.
 */

import type { Database } from "bun:sqlite";
import { createRequire } from "node:module";
import type { DatabaseSync } from "node:sqlite";

/** The JavaScript runtime yian is running on, detected once at load. */
export const runtime: "bun" | "node" = typeof Bun !== "undefined" ? "bun" : "node";

/** A value bound to a positional `?` parameter. */
export type SqliteParam = null | number | bigint | string | Uint8Array;

/** A value read from a column; integers are always safe `number`s. */
export type SqliteValue = null | number | string | Uint8Array;

/** One result row: a plain object from column name to value. */
export type SqliteRow = Record<string, SqliteValue>;

/** A prepared statement taking positional `?` parameters. */
export interface SqliteStatement {
  /** Return every result row, in order. */
  all(...params: SqliteParam[]): SqliteRow[];
  /** Return the first result row, or `undefined` when there's none. */
  get(...params: SqliteParam[]): SqliteRow | undefined;
  /** Run the statement, discarding any rows. */
  run(...params: SqliteParam[]): void;
}

/** An open SQLite connection. */
export interface Sqlite {
  /** Run one or more `;`-separated statements without parameters. */
  exec(sql: string): void;
  /** Return the statement for `sql`, prepared once per connection and cached by its text. */
  prepare(sql: string): SqliteStatement;
  /** Finalize the cached statements and close the connection; a second call does nothing. */
  close(): void;
}

/** The part of either driver's statement the wrapper uses; only Bun's has `finalize`. */
interface DriverStatement {
  all(...params: SqliteParam[]): unknown[];
  get(...params: SqliteParam[]): unknown;
  run(...params: SqliteParam[]): unknown;
  finalize?(): void;
}

/** The part of either driver's connection the wrapper uses. */
interface DriverDatabase {
  exec(sql: string): unknown;
  prepare(sql: string): DriverStatement;
  close(): void;
}

const require = createRequire(import.meta.url);

let openDriver: ((path: string, readonly: boolean) => DriverDatabase) | undefined;

/** Return the opener for this runtime's SQLite driver, loading the driver on first use. */
function driver(): (path: string, readonly: boolean) => DriverDatabase {
  if (openDriver) return openDriver;
  if (runtime === "bun") {
    const { Database: BunDatabase } = require("bun:sqlite") as { Database: typeof Database };
    openDriver = (path, readonly) =>
      new BunDatabase(path, { create: !readonly, readwrite: !readonly, readonly, strict: true, safeIntegers: true });
  } else {
    const { DatabaseSync: NodeDatabase } = requireNodeSqlite();
    openDriver = (path, readonly) => new NodeDatabase(path, { readOnly: readonly });
  }
  return openDriver;
}

/** Load `node:sqlite`, dropping only the experimental-feature warning Node 22 emits for it. */
function requireNodeSqlite(): { DatabaseSync: typeof DatabaseSync } {
  const emitWarning = process.emitWarning;
  process.emitWarning = ((warning: string | Error, ...rest: unknown[]) => {
    const option = rest[0];
    const type = typeof option === "string" ? option : (option as { type?: unknown } | undefined)?.type;
    const text = typeof warning === "string" ? warning : warning.message;
    if (type === "ExperimentalWarning" && text.startsWith("SQLite")) return;
    Reflect.apply(emitWarning, process, [warning, ...rest]);
  }) as typeof process.emitWarning;
  try {
    return require("node:sqlite") as { DatabaseSync: typeof DatabaseSync };
  } finally {
    process.emitWarning = emitWarning;
  }
}

/** Copy a driver row into a plain object, turning Bun's `bigint`s into safe numbers. */
function plainRow(row: object): SqliteRow {
  // Spread defines each column as an own property, even one named `__proto__`.
  const copy = { ...row } as Record<string, unknown>;
  for (const [key, value] of Object.entries(copy)) {
    if (typeof value !== "bigint") continue;
    if (value > BigInt(Number.MAX_SAFE_INTEGER) || value < BigInt(Number.MIN_SAFE_INTEGER)) {
      throw new RangeError(`SQLite integer ${value} in column ${key} is outside the safe integer range`);
    }
    copy[key] = Number(value);
  }
  return copy as SqliteRow;
}

/**
 * Open the SQLite database file at `path`, creating it unless `readonly`.
 *
 * Integers read back as `number` on both runtimes, and one outside
 * `Number.isSafeInteger` throws `RangeError`. Driver errors pass through; read their
 * result code with `sqliteErrorCode`.
 */
export function openSqlite(path: string, { readonly = false }: { readonly?: boolean } = {}): Sqlite {
  const db = driver()(path, readonly);
  const statements = new Map<string, { statement: SqliteStatement; raw: DriverStatement }>();
  let closed = false;
  return {
    exec(sql) {
      db.exec(sql);
    },
    prepare(sql) {
      const cached = statements.get(sql);
      if (cached) return cached.statement;
      const raw = db.prepare(sql);
      const statement: SqliteStatement = {
        all: (...params) => raw.all(...params).map((row) => plainRow(row as object)),
        get: (...params) => {
          // Bun returns null for no row, Node undefined.
          const row = raw.get(...params);
          return row == null ? undefined : plainRow(row);
        },
        run: (...params) => {
          raw.run(...params);
        },
      };
      statements.set(sql, { statement, raw });
      return statement;
    },
    close() {
      if (closed) return;
      closed = true;
      // Bun keeps the connection and its locks open while any statement is unfinalized;
      // Node's close finalizes them itself.
      for (const { raw } of statements.values()) raw.finalize?.();
      db.close();
    },
  };
}

/**
 * Return the SQLite extended result code a driver error carries, or `undefined` for
 * any other value. Apply `& 0xff` for the primary code.
 */
export function sqliteErrorCode(error: unknown): number | undefined {
  if (typeof error !== "object" || error === null) return undefined;
  const { name, code, errno, errcode } = error as { name?: unknown; code?: unknown; errno?: unknown; errcode?: unknown };
  const result = name === "SQLiteError" ? errno : code === "ERR_SQLITE_ERROR" ? errcode : undefined;
  return typeof result === "number" ? result : undefined;
}
