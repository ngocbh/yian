/**
 * `VariablePool`: the runtime state primitive for a flow run.
 *
 * Each node id has at most one stored value: an `Output` commits one `TypedValue` (scalar,
 * object, or list), while a CASE `Route` commits none. "Multiple outputs" are fields of one
 * object. `${input.X}` resolves to `store[<startId>].X`: the synthesized START node's committed
 * bound-input record is the flow's run-arguments object. The one side namespace is `system` (host-injected ambient: run id / clock / tenant;
 * reserved, run-global). There is no engine-private key-addressed namespace; live
 * AGENT and TOOL effects all return the leaf's ordinary Output. An AGENT's success also stores
 * its new messages under their own pool name, `<id>.new_messages` (`messagesKey`).
 *
 * Two responsibilities:
 * - `readRoot(names)` backs the `${...}` expression evaluator: it reads a reference's root
 *   (`${x.output}` is the node's whole value), and the evaluator walks the typed value's fields
 *   after it, so `${x.output.ratio}` traverses into an object output.
 * - `toWire()`/`fromWire()` and `dumps()`/`loads()` are a lossless whole-pool snapshot codec
 *   (each value's `{kind, value}` wire form keeps its exact type).
 */

import { logJson, parseJson } from "./json.ts";
import {
  buildValue,
  buildValueAs,
  fieldOf,
  isRecord,
  requireFields,
  type Type,
  TypedValue,
  type Value,
  type ValueKind,
  type ValueRecord,
} from "./values.ts";

type TypedWire = ReturnType<TypedValue["toWire"]>;

/** The field an AGENT's new messages are read under: `${<id>.new_messages}`. */
export const NEW_MESSAGES = "new_messages";

/**
 * Return how many of a pool reference's names its root takes: two for `<node>.output`,
 * `<node>.new_messages`, `input.<k>` and `system.<k>`; one for `item` and any other head.
 */
export function poolRootLength(names: readonly string[]): number {
  if (names.length < 2 || names[0] === "item") return 1;
  if (names[0] === "input" || names[0] === "system") return 2;
  return names[1] === "output" || names[1] === NEW_MESSAGES ? 2 : 1;
}

/** The pool name of node `nodeId`'s new messages. A node id holds no `.`, so it never names a node. */
export function messagesKey(nodeId: string): string {
  return `${nodeId}.${NEW_MESSAGES}`;
}

/**
 * A reference root's pool entry: `name`, or `null` when the root names no entry here; how many
 * names the root used; and the START record's key for `input.<k>`.
 */
export interface PoolRoot {
  readonly name: string | null;
  readonly used: number;
  readonly key?: string;
}

/**
 * Return the pool entry a reference's root reads: `<node>.output` reads `<node>`,
 * `<node>.new_messages` reads `messagesKey(<node>)`, `input.<k>` reads key `k` of the START
 * record (`startId`), and `system.<k>` reads the system value `system.<k>`. A node root names
 * its entry only when `holds` says the pool holds it; any other head names none.
 */
export function referenceRoot(names: readonly string[], startId: string, holds: (name: string) => boolean): PoolRoot {
  const used = poolRootLength(names);
  if (used === 1) return { name: null, used };
  const [head, second] = names as [string, string];
  if (head === "input") return { name: startId, used, key: second };
  if (head === "system") return { name: `system.${second}`, used };
  const name = second === "output" ? head : messagesKey(head);
  return { name: holds(name) ? name : null, used };
}

/**
 * Return why a reference's root is absent, for a read's error: its node was skipped, its input
 * was not given, or its system value is not set; `null` for a head that names no pool root.
 */
export function absentRootWhy(names: readonly string[]): string | null {
  if (poolRootLength(names) === 1) return null;
  const [head, second] = names as [string, string];
  if (head === "input") return `input '${second}' was not given`;
  if (head === "system") return `system value '${second}' is not set`;
  return `node '${head}' was skipped`;
}

/**
 * One root read without its source: how many names the root used, whether it is there, its
 * value (`null` when absent), and why it is absent (`null` when present or unknown).
 */
export interface RootValue {
  readonly used: number;
  readonly present: boolean;
  readonly value: Value;
  readonly why: string | null;
}

/** A variable pool's wire object: each namespace's values as `{kind, value}`, and the START id. */
export interface PoolWire {
  store: Record<string, TypedWire>;
  system: Record<string, TypedWire>;
  start_id: string;
}

/**
 * The runtime variable store for one flow run.
 *
 * Each node id has at most one stored typed value (`store.get(nodeId)`); a routing-only node
 * can have none. "Multiple outputs" are fields of one object. `${input.X}` reads the
 * synthesized START node's committed record; `${system.X}` reads host-injected ambients. Backs
 * the `${...}` root reads and snapshots the whole pool losslessly (`toWire`/`dumps`).
 */
export class VariablePool {
  /** Node id to that node's single produced value. Read via `${<id>.output[.path]}`. */
  readonly store = new Map<string, TypedValue>();
  /**
   * Host-injected ambient namespace (run id / clock / tenant); reserved and run-global. Read
   * via `${system.<key>}`.
   */
  readonly system = new Map<string, TypedValue>();
  /**
   * The enclosing flow's START node id, so `${input.X}` resolves to `store[startId].X`. The
   * engine sets this per flow; the default is the top-level convention for standalone pools.
   */
  readonly startId: string;

  /** Build an empty pool whose `${input.X}` reads `store[startId]`. */
  constructor(startId: string = "__start__") {
    // The default mirrors `StartNode.ID` but isn't imported from it: `typesys` is a leaf below
    // `nodes`, so it can't import the node class. A test pins the two together.
    this.startId = startId;
  }

  // --- writes ------------------------------------------------------------- //

  /**
   * Store a node's single produced value, type-checking it if a type is declared.
   *
   * `value` is the node's plain value. When `declared` (the node's declared output type) is
   * given, the value is validated against it (`buildValueAs`); when `null`, the type is
   * inferred (`buildValue`). A failed check leaves the node's previous value in place.
   *
   * @throws TypeCheckError when the value doesn't fit `declared`, or isn't a yian value.
   */
  set(nodeId: string, value: unknown, declared: ValueKind | Type | null = null): void {
    this.store.set(nodeId, declared !== null ? buildValueAs(declared, value) : buildValue(value));
  }

  /** Store a host-injected ambient under `key`, read via `${system.<key>}`. */
  addSystem(key: string, value: unknown): void {
    this.system.set(key, buildValue(value));
  }

  // --- reads -------------------------------------------------------------- //

  /** Return a node's stored `TypedValue`, or `null` if it has not produced yet. */
  getValue(nodeId: string): TypedValue | null {
    return this.store.get(nodeId) ?? null;
  }

  /** Return a node's plain (unwrapped) value, or `fallback` if it is unset. */
  get(nodeId: string, fallback: Value = null): Value {
    const seg = this.store.get(nodeId);
    return seg !== undefined ? seg.toObject() : fallback;
  }

  /**
   * Read a reference's root (`referenceRoot`) from the pool; the evaluator walks the names after
   * it. A root that names no entry, an `input.<k>` the START record lacks, or an unset system
   * value is absent; a stored `null` is present.
   */
  readRoot(names: readonly string[]): RootValue {
    const root = referenceRoot(names, this.startId, (name) => this.store.has(name));
    const absent: RootValue = { used: root.used, present: false, value: null, why: absentRootWhy(names) };
    if (root.name === null) return absent;
    let typed: TypedValue | undefined;
    if (names[0] === "system") typed = this.system.get(names[1]!);
    else typed = this.store.get(root.name);
    if (typed === undefined) return absent;
    if (root.key === undefined) return { used: root.used, present: true, value: typed.toObject(), why: null };
    const [present, value] = fieldOf(typed.toObject(), root.key);
    return present ? { used: root.used, present, value, why: null } : absent;
  }

  // --- serialization ------------------------------------------------------ //

  /**
   * Return the pool's wire object `{store, system, start_id}`, each value as its `{kind,
   * value}` wire form.
   *
   * @throws TypeCheckError when a stored value isn't a JSON value (see `TypedValue.toWire`).
   */
  toWire(): PoolWire {
    // `Object.fromEntries` defines own data properties, so a `__proto__` node id stays a key.
    return {
      store: Object.fromEntries([...this.store].map(([key, typed]) => [key, typed.toWire()])),
      system: Object.fromEntries([...this.system].map(([key, typed]) => [key, typed.toWire()])),
      start_id: this.startId,
    };
  }

  /**
   * Decode and strictly validate a variable-pool wire object.
   *
   * `data` must be a plain record with exactly `store`, `system` and `start_id`; `store` and
   * `system` records whose every entry is a typed value's wire form (`TypedValue.fromWire`);
   * and `start_id` text. Entries keep the record's key order. Values aren't walked: the
   * lossless reader made the data.
   *
   * @throws RangeError when `data` has missing or extra fields, a namespace or entry of the
   *   wrong shape, an invalid typed value, or a `start_id` that isn't text.
   */
  static fromWire(data: unknown): VariablePool {
    if (!isRecord(data)) throw new RangeError("variable-pool wire must be a JSON object");
    requireFields(data, ["store", "system", "start_id"], "variable pool");
    const store = data["store"];
    const system = data["system"];
    if (!isRecord(store) || !isRecord(system)) {
      throw new RangeError("variable-pool store and system must be JSON objects");
    }
    const storeEntries = decodeEntries(store, "store");
    const systemEntries = decodeEntries(system, "system");
    const startId = data["start_id"];
    if (typeof startId !== "string") {
      throw new RangeError("variable-pool start_id must be a string");
    }
    const pool = new VariablePool(startId);
    for (const [key, typed] of storeEntries) pool.store.set(key, typed);
    for (const [key, typed] of systemEntries) pool.system.set(key, typed);
    return pool;
  }

  /** Serialize the whole pool with its exact typed-value kinds, as yian's JSON codec writes it. */
  dumps(): string {
    return logJson(this.toWire());
  }

  /**
   * Reconstruct a pool from `dumps` output, decoding each value to its exact type.
   *
   * @throws JSONDecodeError when `text` isn't JSON.
   * @throws RangeError when the JSON isn't a pool's wire object (see `fromWire`).
   */
  static loads(text: string): VariablePool {
    return VariablePool.fromWire(parseJson(text, "codec"));
  }
}

/**
 * Decode a namespace record's entries, in key order.
 *
 * @throws RangeError when an entry isn't a record or isn't a typed value's wire form.
 */
function decodeEntries(record: ValueRecord, namespace: string): [string, TypedValue][] {
  return Object.entries(record).map(([key, typedWire]) => {
    if (!isRecord(typedWire)) {
      throw new RangeError(`variable-pool ${namespace} entries must be typed objects`);
    }
    return [key, TypedValue.fromWire(typedWire)];
  });
}
