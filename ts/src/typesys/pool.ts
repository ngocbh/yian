/**
 * `VariablePool`: the runtime state primitive for a flow run.
 *
 * Each node id has at most one stored value: an `Output` commits one `TypedValue` (scalar,
 * object, or list), while a CASE `Route` commits none. "Multiple outputs" are fields of one
 * object. `${input.X}` resolves to `store[<startId>].X`: the synthesized START node's committed
 * bound-input record is the flow's run-arguments object (the standalone `inputs` namespace is
 * retired). The one side namespace is `system` (host-injected ambient: run id / clock / tenant;
 * reserved, run-global). There is no engine-private key-addressed namespace; live
 * AGENT/HUMAN_INPUT/WAIT effects all return the leaf's ordinary Output.
 *
 * Two responsibilities:
 * - `resolve(head, rest)` backs the `${...}` expression evaluator. Because values are typed
 *   objects (not stringified), `${x.output.ratio}` traverses into an object output;
 *   `${x.output}` is the node's whole value.
 * - `toWire()`/`fromWire()` and `dumps()`/`loads()` serialize the whole pool losslessly (each
 *   value's `{kind, value}` wire form keeps its exact type), matching the pool wire
 *   representation recorded at run initialization and reconstructed during replay.
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
 * the `${...}` resolver and serializes losslessly for canonical run-record payloads and
 * recovery.
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
    // `nodes`, so it can't import the node class. A consistency test pins the two together.
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
   * Resolve a parsed `${head.rest...}` reference to a plain value.
   *
   * Recognizes three namespaces: `input.<key>[.<path>]` reads the START record,
   * `<node>.output[.<path>]` reads a node's value (`.output` is a syntactic discriminator the
   * resolver skips), and `system.<key>` reads a host ambient. An unrecognized head resolves to
   * `null` (the missing-ref-is-falsy contract); legacy plural surfaces
   * (`${outputs.X}`/`${inputs.X}`) are rejected at load time, so the load-time guard, not this
   * method, is the authoritative defense.
   *
   * `head` is the first reference segment (`input`, `system`, or a node id) and `rest` the
   * remaining dotted segments (e.g. `["output", "ratio"]`). Returns the resolved plain value,
   * or `null` if any step is absent.
   */
  resolve(head: string, rest: readonly string[]): Value {
    // Node-first head: `${<node>.output[.path]}` reads `store[<node>]`'s interior. The literal
    // `output` token is a syntactic discriminator and is skipped.
    if (this.store.has(head) && rest[0] === "output") {
      return traverse(this.store.get(head), rest.slice(1));
    }
    // Singular input head: `${input.k}` ≡ `store[startId].k`.
    if (head === "input") {
      if (rest.length === 0) return null;
      return traverse(this.store.get(this.startId), rest);
    }
    if (head === "system") {
      const seg = rest.length > 0 ? this.system.get(rest[0]!) : undefined;
      return seg !== undefined ? seg.toObject() : null;
    }
    return null;
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
   * and `start_id` text. Entries keep the record's key order.
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
    if (typeof startId !== "string") throw new RangeError("variable-pool start_id must be a string");
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
 * Object-walk a stored typed value along `path` (the `${<id>.output.a.b}` / `${input.x.y}`
 * dotted read); a missing step resolves to `null` (propagates falsy).
 */
function traverse(seg: TypedValue | undefined, path: readonly string[]): Value {
  if (seg === undefined) return null;
  let value = seg.toObject();
  for (const step of path) {
    value = fieldOf(value, step)[1];
    if (value === null) return null;
  }
  return value;
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
