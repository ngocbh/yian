/**
 * A run's memories: the sessions on the instances a host binds to its memory inputs, one per
 * instance, and the memory value each input records.
 *
 * Knows about: memory (the contract, sessions and their errors), typesys (`MemoryRef`) and
 * diagnostics.
 */

import { noteCleanup } from "../diagnostics.ts";
import {
  BUILTIN_MEMORY_TOOLS,
  checkContract,
  type MemoryBase,
  MemoryBusyError,
  type MemoryContract,
  memoryErrorKind,
  type MemorySession,
  type Recall,
} from "../memory/base.ts";
import { type JsonObject, logJson } from "../typesys/json.ts";
import { MemoryRef } from "../typesys/values.ts";

/** One memory input of a run and the instance a host bound to it. */
export interface MemoryBinding {
  /** The input's name. */
  readonly input: string;
  /** The memory the host bound to the input; one instance bound to several inputs opens once. */
  readonly instance: MemoryBase;
  /** The input's memory type: the contract type its instance must declare. */
  readonly type: string;
  /** The JSON Schema of the input's item type, or `null` for a memory without typed items. */
  readonly itemType: JsonObject | null;
}

/** How `RunMemories.open` opens a run's memories. */
export interface OpenMemoriesOptions {
  /** `false` refuses a memory that doesn't exist yet, as a resume does. */
  readonly create: boolean;
  /** Stops the opens and snapshots: a call rejects with its reason, once it aborts. */
  readonly signal: AbortSignal;
  /**
   * The tool names of each host memory type the flow was loaded with, by contract type; a bound
   * instance of such a type must declare the same tools.
   */
  readonly hostTypes?: ReadonlyMap<string, readonly string[]>;
}

/** How `RunMemories.reopen` opens the memories a resumed run started with. */
export interface ReopenMemoriesOptions {
  /** As `OpenMemoriesOptions.signal`. */
  readonly signal: AbortSignal;
  /** As `OpenMemoriesOptions.hostTypes`. */
  readonly hostTypes?: ReadonlyMap<string, readonly string[]>;
  /** Whether to read each memory's version and report drift: only a crash resume does. */
  readonly drift: boolean;
}

/**
 * A memory that changed since its run started (including the run's own writes), as a crash resume
 * found it: its version isn't the one recorded at the run's start.
 */
export interface MemoryDrift {
  /** The first input bound to the memory. */
  readonly input: string;
  readonly identity: string;
  /** The version the run recorded when it started. */
  readonly recorded: string;
  /** The version the memory holds now. */
  readonly current: string;
}

/** One open session and the inputs bound to its instance. */
interface Opened {
  readonly instance: MemoryBase;
  readonly inputs: string[];
  readonly session: MemorySession;
}

/**
 * The sessions of one run segment on its memories, one per bound instance, and the value each
 * memory input records. Every node of the run bound to a memory reaches it through its session.
 */
export class RunMemories {
  /** The memory value of each bound input, in binding order. */
  readonly values: ReadonlyMap<string, MemoryRef>;
  readonly #sessions: ReadonlyMap<string, MemorySession>;

  private constructor(values: ReadonlyMap<string, MemoryRef>, sessions: ReadonlyMap<string, MemorySession>) {
    this.values = values;
    this.#sessions = sessions;
  }

  /**
   * Open each distinct bound instance once, check it, and snapshot it into its inputs' values.
   *
   * Each instance's class must declare a contract (`checkContract`) of its input's type, and a host
   * type the tools of the class the flow was loaded with. Each session's info must name its
   * contract's type and tools, and no two instances may hold one identity. If any step fails, the
   * sessions opened so far are closed.
   *
   * @throws TypeError when an instance's contract or its session's info is malformed or differs.
   * @throws RangeError when an instance's contract type isn't its input's, one instance is bound to
   *   inputs of different item types, or two instances hold one identity.
   * @throws MemoryBusyError when a memory's lock stays held past its timeout, naming the input.
   * @throws Error naming the input for another failed `open` or `snapshot`, its cause the error.
   * @throws The signal's reason once it aborted.
   */
  static async open(bindings: readonly MemoryBinding[], options: OpenMemoriesOptions): Promise<RunMemories> {
    const opened = await openSessions(bindings, options);
    try {
      const snapshots = new Map<MemorySession, { version: string; recall: Recall }>();
      const values = new Map<string, MemoryRef>();
      for (const { inputs, session } of opened) {
        const snapshot = await sessionStep(inputs[0]!, options.signal, () => session.snapshot());
        snapshots.set(session, snapshot);
        for (const input of inputs) values.set(input, memoryValue(input, session, snapshot));
      }
      const ordered = new Map(bindings.map(({ input }) => [input, values.get(input)!]));
      return new RunMemories(ordered, sessionsByIdentity(opened));
    } catch (error) {
      await closeAll(opened, error);
      throw error;
    }
  }

  /**
   * Open again, never creating one, the memories a resumed run started with: each input's
   * instance is checked as `open` checks it, then its identity, type, settings and tool specs
   * against the value the run recorded. With `drift`, each memory's version is read and compared
   * with the recorded one. The recorded values stay the run's: a resume never takes a new recall.
   *
   * @param recorded The value each memory input recorded at the run's start; an Optional one left
   *   out has none and takes no binding.
   * @returns The sessions, and one drift entry per memory whose version isn't the recorded one
   *   (none without `drift`), frozen.
   * @throws RangeError when a recorded memory has no binding or isn't at its location ("bind it
   *   with --memory"), or its identity, type, settings or tool specs differ from the recorded ones.
   * @throws As `open` throws, for the instance checks and a failed open or snapshot.
   */
  static async reopen(
    bindings: readonly MemoryBinding[],
    recorded: ReadonlyMap<string, MemoryRef>,
    options: ReopenMemoriesOptions,
  ): Promise<{ memories: RunMemories; drift: readonly MemoryDrift[] }> {
    for (const [input, value] of recorded) {
      if (!bindings.some((binding) => binding.input === input)) throw missingMemory(input, value.identity);
    }
    const opened = await openSessions(bindings, { ...options, create: false }, recorded);
    try {
      const drift: MemoryDrift[] = [];
      for (const { inputs, session } of opened) {
        for (const input of inputs) checkRecorded(input, session, recorded.get(input)!);
        if (!options.drift) continue;
        const [input] = inputs;
        const { version } = await sessionStep(input!, options.signal, () => session.snapshot());
        const was = recorded.get(input!)!.version;
        if (version !== was) {
          drift.push(Object.freeze({ input: input!, identity: session.info.identity, recorded: was, current: version }));
        }
      }
      const values = new Map(bindings.map(({ input }) => [input, recorded.get(input)!]));
      return { memories: new RunMemories(values, sessionsByIdentity(opened)), drift: Object.freeze(drift) };
    } catch (error) {
      await closeAll(opened, error);
      throw error;
    }
  }

  /**
   * Return the session on the memory `identity`.
   *
   * @throws Error when this run has none: a node bound to a memory the run didn't bind.
   */
  session(identity: string): MemorySession {
    const session = this.#sessions.get(identity);
    if (session === undefined) throw new Error(`this run has no session on memory \`${identity}\``);
    return session;
  }

  /**
   * Close every session. Never rejects: a run's memories close after its records are written, so
   * a failed close can't change its outcome.
   *
   * @returns Each memory whose session failed to close, by identity, with its error.
   */
  async close(): Promise<{ identity: string; error: unknown }[]> {
    const failed: { identity: string; error: unknown }[] = [];
    for (const [identity, session] of this.#sessions) {
      try {
        await session.close();
      } catch (error) {
        failed.push({ identity, error });
      }
    }
    return failed;
  }
}

/**
 * Check each binding's instance, then open each distinct instance once and check its session.
 * The sessions are closed again if a check fails.
 */
async function openSessions(
  bindings: readonly MemoryBinding[],
  options: OpenMemoriesOptions,
  recorded: ReadonlyMap<string, MemoryRef> = new Map(),
): Promise<Opened[]> {
  const { create, signal } = options;
  for (const binding of bindings) checkInstance(binding, options.hostTypes ?? new Map());
  checkSharedItemTypes(bindings);
  const opened: Opened[] = [];
  try {
    for (const { input, instance, itemType } of bindings) {
      const known = opened.find((each) => each.instance === instance);
      if (known !== undefined) {
        known.inputs.push(input);
        continue;
      }
      const identity = recorded.get(input)?.identity ?? null;
      const session = await sessionStep(input, signal, () => instance.open({ create, itemType, signal }), identity);
      opened.push({ instance, inputs: [input], session });
      checkSession(input, contractOf(instance), session);
      const other = opened.find((each) => each.session !== session && each.session.info.identity === session.info.identity);
      if (other !== undefined) {
        throw new RangeError(
          `\`${input}\` and \`${other.inputs[0]}\` are two memories with identity ` +
            `\`${session.info.identity}\`; bind one instance`,
        );
      }
    }
  } catch (error) {
    await closeAll(opened, error);
    throw error;
  }
  return opened;
}

/**
 * Refuse one instance bound to inputs of different item types: it opens once, with the first
 * input's item type, so another's would never reach it.
 *
 * @throws RangeError naming the two inputs.
 */
function checkSharedItemTypes(bindings: readonly MemoryBinding[]): void {
  for (const binding of bindings) {
    const first = bindings.find((each) => each.instance === binding.instance)!;
    // By value, keys sorted, so one schema written in another key order is the same type.
    if (first === binding || logJson(first.itemType) === logJson(binding.itemType)) continue;
    throw new RangeError(
      `memory inputs \`${first.input}\` and \`${binding.input}\` share one instance but declare different item types`,
    );
  }
}

/** Return the contract `instance`'s class declares, which `checkContract` has checked. */
function contractOf(instance: MemoryBase): MemoryContract {
  return (instance.constructor as typeof MemoryBase).contract;
}

/**
 * Check a binding's instance before it opens: its class declares a well-formed contract of the
 * input's type, and a host type's contract has the tools of the class the flow was loaded with.
 *
 * @throws TypeError naming the input when the contract is malformed or its tools differ.
 * @throws RangeError when its type isn't the input's.
 */
function checkInstance(binding: MemoryBinding, hostTypes: ReadonlyMap<string, readonly string[]>): void {
  const { input, instance, type } = binding;
  try {
    checkContract(instance);
  } catch (error) {
    if (!(error instanceof TypeError)) throw error;
    throw new TypeError(`memory \`${input}\`: ${error.message}`, { cause: error });
  }
  const contract = contractOf(instance);
  if (contract.type !== type) {
    throw new RangeError(`\`${input}\` is \`${type}\`, but its instance is \`${contract.type}\``);
  }
  const loaded = hostTypes.get(type);
  if (loaded === undefined || Object.hasOwn(BUILTIN_MEMORY_TOOLS, type)) return;
  const names = contract.tools.map((tool) => tool.name);
  if (!sameNames(names, loaded)) {
    const cls = (instance.constructor as { name: string }).name;
    throw new TypeError(
      `memory \`${input}\`: ${cls}'s tools (${names.join(", ")}) aren't those of the \`${type}\` the flow was ` +
        `loaded with (${loaded.join(", ")})`,
    );
  }
}

/** Whether two lists of tool names hold the same names in the same order. */
function sameNames(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((name, i) => name === b[i]);
}

/**
 * Check that a session's info names its contract's type and tools, in the contract's order.
 *
 * @throws TypeError naming the input when they differ.
 */
function checkSession(input: string, contract: MemoryContract, session: MemorySession): void {
  const { info } = session;
  if (info.type !== contract.type) {
    throw new TypeError(
      `memory \`${input}\`: its session is a \`${info.type}\`, not the \`${contract.type}\` its class declares`,
    );
  }
  const names = info.tools.map((tool) => tool.name);
  const declared = contract.tools.map((tool) => tool.name);
  if (!sameNames(names, declared)) {
    throw new TypeError(
      `memory \`${input}\`: its session's tools (${names.join(", ")}) aren't its contract's (${declared.join(", ")})`,
    );
  }
}

/**
 * Run one session step for input `input`, naming the input in its failure: a busy memory stays
 * busy, a missing one says to bind it, and any other error is wrapped. A stop's reason passes
 * through.
 *
 * @param identity The memory's recorded identity, for a missing memory's message, or `null`.
 * @throws MemoryBusyError, RangeError or Error naming the input, the original its cause.
 */
async function sessionStep<T>(
  input: string,
  signal: AbortSignal,
  step: () => Promise<T>,
  identity: string | null = null,
): Promise<T> {
  try {
    return await step();
  } catch (error) {
    if (signal.aborted) throw error;
    if (memoryErrorKind(error) === "missing") throw missingMemory(input, identity, error);
    if (memoryErrorKind(error) === "busy") {
      throw new MemoryBusyError(`memory \`${input}\` is busy: another writer holds its lock; try again`, {
        cause: error,
      });
    }
    throw new Error(`memory \`${input}\`: ${errorText(error)}`, { cause: error });
  }
}

/** The refusal of a memory input whose memory isn't bound, or isn't at its instance's location. */
function missingMemory(input: string, identity: string | null, cause?: unknown): RangeError {
  const which = identity === null ? "" : ` (\`${identity}\`)`;
  return new RangeError(`memory \`${input}\`${which} isn't at its location; bind it with --memory`, { cause });
}

/**
 * Check a reopened session against the value input `input` recorded: its identity and type, then
 * its settings and tool specs, compared as a value built from the session's info records them.
 *
 * @throws RangeError naming the input and what differs.
 */
function checkRecorded(input: string, session: MemorySession, recorded: MemoryRef): void {
  const { identity, type } = session.info;
  if (identity !== recorded.identity) {
    throw new RangeError(`resume refused: memory ${input} is \`${identity}\`, not the recorded \`${recorded.identity}\``);
  }
  if (type !== recorded.type) {
    throw new RangeError(`resume refused: memory ${input} is a \`${type}\`, not the recorded \`${recorded.type}\``);
  }
  // Built as the run built the recorded one, so both are in the form the run log reads back.
  const live = memoryValue(input, session, { version: recorded.version, recall: recorded.recall });
  if (logJson(live.settings) !== logJson(recorded.settings)) {
    throw new RangeError(`resume refused: memory ${input} settings differ from the recorded ones`);
  }
  if (logJson(live.tools) !== logJson(recorded.tools)) {
    throw new RangeError(`resume refused: memory ${input} tools differ from the recorded ones`);
  }
}

/** Return an error's message, or the thrown value as text; never throws. */
function errorText(error: unknown): string {
  try {
    return error instanceof Error ? error.message : String(error);
  } catch {
    return "an error that can't be shown";
  }
}

/**
 * Return input `input`'s memory value: the session's identity, type, settings and tools, and the
 * snapshot's version and recall.
 *
 * @throws TypeError naming the input when the info or snapshot can't be recorded.
 */
function memoryValue(
  input: string,
  session: MemorySession,
  snapshot: { version: string; recall: Recall },
): MemoryRef {
  const { identity, type, settings, tools } = session.info;
  try {
    return new MemoryRef({
      identity,
      type,
      settings,
      version: snapshot.version,
      recall: snapshot.recall,
      tools: tools.map(({ name, write, description, parameters }) => ({ name, write, description, parameters })),
    });
  } catch (error) {
    if (!(error instanceof RangeError)) throw error;
    throw new TypeError(`memory \`${input}\`: ${error.message}`, { cause: error });
  }
}

/** Return the sessions of `opened` by their memories' identities. */
function sessionsByIdentity(opened: readonly Opened[]): Map<string, MemorySession> {
  return new Map(opened.map(({ session }) => [session.info.identity, session]));
}

/** Close every session of `opened` after `primary` failed, noting each close error on it. */
async function closeAll(opened: readonly Opened[], primary: unknown): Promise<void> {
  for (const { session } of opened) {
    try {
      await session.close();
    } catch (error) {
      noteCleanup(primary, error, "closing a memory also raised");
    }
  }
}
