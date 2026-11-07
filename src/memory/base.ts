/**
 * The one interface yian talks to memory through: `MemoryBase`, the contract a memory class
 * declares, the run session it opens, and the values a session's calls return.
 *
 * A memory type is a contract: the tag a flow writes (`!FileMemory`), its tools, its change
 * shapes and its recall. Each class implements one contract, and an instance binds to an input
 * whose type is its class's `contract.type`. A host class may declare a built-in contract; it then
 * keeps that contract's tool names, which `checkContract` checks at bind.
 *
 * It imports only `typesys`.
 */

import type { JsonObject } from "../typesys/json.ts";
import type { Recall } from "../typesys/memory.ts";

export type { Recall, RecordedMemoryTool } from "../typesys/memory.ts";

/** The tag a flow writes for a memory type, and its tools, which are fixed per type. */
export interface MemoryContract {
  /** The tag: `"FileMemory"`, `"VectorDBMemory"` or a host's own. */
  readonly type: string;
  readonly tools: readonly MemoryToolSpec[];
}

/** One tool of a memory type. */
export interface MemoryToolSpec {
  /** The tool's own name, such as `"read"`; a model sees it as `<input>_<name>`. */
  readonly name: string;
  /** Whether a call goes through `apply`, as a change. */
  readonly write: boolean;
  /** Fixed text, with no counts or sizes: a resume compares it with the recorded one. */
  readonly description: string;
  /** The JSON Schema of the tool's arguments. */
  readonly parameters: JsonObject;
}

/**
 * A memory a host binds to a flow input. The location is a constructor argument, and never
 * reaches a flow or a run log.
 */
export abstract class MemoryBase {
  /** The contract this class implements. */
  static readonly contract: MemoryContract;

  /**
   * Open a session on the memory for one run.
   *
   * @param o.create `false` refuses a memory that doesn't exist yet, as a resume does.
   * @param o.itemType The JSON Schema of a typed memory's items, or `null`.
   */
  abstract open(o: {
    create: boolean;
    itemType: JsonObject | null;
    signal: AbortSignal;
  }): Promise<MemorySession>;

  /**
   * Empty the memory by the type's own policy, keeping its identity. Only a host or the CLI calls
   * it, outside runs.
   */
  abstract reset(o: { signal: AbortSignal }): Promise<void>;
}

/**
 * One open memory. All nodes of a run bound to one identity share one session, so it accepts
 * concurrent calls; the type serializes writes.
 *
 * A method throws `MemoryBusyError` when its lock wait times out, `ChangeIdReusedError` for a
 * known change id with a different change, and other errors for I/O faults.
 */
export interface MemorySession {
  readonly info: MemoryInfo;
  /** Whether this open created the memory. */
  readonly created: boolean;
  /** Read the version and the recall in one consistent read. */
  snapshot(): Promise<{ version: string; recall: Recall }>;
  /** Call the read tool `tool`. */
  call(tool: string, args: JsonObject, o: { signal: AbortSignal }): Promise<JsonObject>;
  /**
   * Apply one change: a write tool's arguments plus `op`, its name. A known `id` with the same
   * change returns the stored result and applies nothing.
   */
  apply(c: { id: string; change: JsonObject; author: Author; signal: AbortSignal }): Promise<Applied>;
  /** Undo the change `changeId`, if its target still holds what that change wrote. */
  undo(o: { id: string; changeId: string; author: Author; signal: AbortSignal }): Promise<Applied>;
  /** Remove `target`'s content and every copy of it in the memory's history. It can't be undone. */
  erase(o: { id: string; target: JsonObject; author: Author; signal: AbortSignal }): Promise<Erased>;
  /** List the last `limit` changes, newest first. */
  history(o: { limit: number }): Promise<ChangeEntry[]>;
  close(): Promise<void>;
}

/** What a session's memory is, as its recorded value holds it. */
export interface MemoryInfo {
  /** The contract type. */
  readonly type: string;
  /** Fixed when the memory is created and stored with it. */
  readonly identity: string;
  /** The settings a resume compares; never a location or a credential. */
  readonly settings: JsonObject;
  /** The bound instance's tool specs, in the order a whole grant shows them. */
  readonly tools: readonly MemoryToolSpec[];
}

/** Who made a change: an agent, a TOOL node, a person through the CLI, or a host program. */
export type Author = {
  kind: "agent" | "tool" | "user" | "host";
  name: string;
  run?: string;
  node?: string;
};

/**
 * A change's outcome. `ok: false` is a conflict or a cap refusal, which wrote nothing, or the
 * stored result of a change whose outcome is unknown, which is never applied again.
 */
export type Applied =
  | { ok: true; version: string; result: JsonObject }
  | { ok: false; reason: string; current: JsonObject | null };

/** An erase's outcome: the new version and the change ids that can no longer be undone. */
export type Erased = { version: string; disabledUndos: string[] };

/** One change in a memory's history. */
export type ChangeEntry = {
  /** The change id; `undo` takes it. */
  id: string;
  /** The change's `op`, such as `"append"`, or `"undo"` and `"erase"`. */
  op: string;
  /** What it changed, such as `{path}` in a folder or `{item}` in a store. */
  target: JsonObject;
  author: Author;
  /** When it was applied, as an ISO 8601 time. */
  at: string;
};

/** A session's wait for its memory's lock timed out; nothing was written. */
export class MemoryBusyError extends Error {
  override name = "MemoryBusyError";
}

/** A change id already applied was sent with a different change; nothing was written. */
export class ChangeIdReusedError extends Error {
  override name = "ChangeIdReusedError";
  /** The reused change id. */
  readonly id: string;

  constructor(id: string, options?: ErrorOptions) {
    super(`change id \`${id}\` was used for another change`, options);
    this.id = id;
  }
}

/** The tool names of each built-in contract, in spec order. */
export const BUILTIN_MEMORY_TOOLS: Readonly<Record<"FileMemory" | "VectorDBMemory", readonly string[]>> =
  Object.freeze({
    FileMemory: Object.freeze(["list", "read", "search", "append", "create", "edit", "delete", "rewrite"]),
    VectorDBMemory: Object.freeze(["search", "read", "list", "add", "edit", "retire", "delete"]),
  });

/**
 * Check that `instance`'s class declares a contract that names each tool once and, for a
 * built-in contract type, exactly that contract's tools.
 *
 * @throws {TypeError} If it declares no contract, names a tool twice, or lacks or adds a
 *   built-in contract's tool.
 */
export function checkContract(instance: MemoryBase): void {
  const cls = instance.constructor as typeof MemoryBase;
  const contract = cls.contract as MemoryContract | undefined;
  if (contract === undefined) {
    throw new TypeError(`${cls.name} declares no memory contract; give it a static contract`);
  }
  const names = contract.tools.map((tool) => tool.name);
  const twice = names.find((name, i) => names.indexOf(name) !== i);
  if (twice !== undefined) throw new TypeError(`${cls.name}'s contract lists the tool ${twice} twice`);
  const expected = Object.hasOwn(BUILTIN_MEMORY_TOOLS, contract.type)
    ? BUILTIN_MEMORY_TOOLS[contract.type as keyof typeof BUILTIN_MEMORY_TOOLS]
    : null;
  if (expected === null) return;
  const declares = `${cls.name} declares ${contract.type}`;
  const missing = expected.filter((name) => !names.includes(name));
  if (missing.length > 0) throw new TypeError(`${declares} but lacks its tools: ${missing.join(", ")}`);
  const added = names.filter((name) => !expected.includes(name));
  if (added.length > 0) {
    throw new TypeError(`${declares} but adds tools ${contract.type} doesn't have: ${added.join(", ")}`);
  }
}
