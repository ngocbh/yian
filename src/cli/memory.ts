/**
 * Turn `yian run`'s memory flags into one host-built memory registry.
 *
 * The CLI builds only built-in memory types. A flow with a custom type fails here, before any run
 * or resume, even a terminal `--resume` that would open no memory; such flows run through the API
 * (`buildMemories` with `factories` or `supplied`).
 */

import { resolveMemoryLayout } from "../compile/memory_layout.ts";
import type { ComposedFlow } from "../compose/build/pipeline.ts";
import { type Env, processEnv } from "../defaults.ts";
import { compareCodeUnits, reprText } from "../diagnostics.ts";
import { BUILTIN_MEMORY_CONFIG_KEYS, MemorySetupError } from "../memory/contracts.ts";
import { buildMemories, type MemoryRegistry } from "../memory_host.ts";
import { writeAsJson } from "../typesys/values.ts";

/** A memory flag value is invalid; `paramHint` names the flag. */
export class MemoryOptionError extends RangeError {
  override name = "MemoryOptionError";
  readonly paramHint: string;

  constructor(message: string, { paramHint }: { paramHint: string }) {
    super(message);
    this.paramHint = paramHint;
  }
}

/**
 * Parse repeated `KEY=PATH` options into a key → path record with a null prototype.
 *
 * Surrounding whitespace is trimmed from KEY; PATH is kept as given.
 *
 * @throws {MemoryOptionError} When an option is malformed or repeats a key.
 */
export function parseMemoryStorage(options: Iterable<string>): Record<string, string> {
  const paths: Record<string, string> = Object.create(null);
  for (const option of options) {
    const separator = option.indexOf("=");
    const key = separator < 0 ? "" : option.slice(0, separator).trim();
    const path = option.slice(separator + 1);
    if (!key || !path.trim()) {
      throw new MemoryOptionError(`expected KEY=PATH, got ${reprText(option)}`, { paramHint: "--memory-storage" });
    }
    if (Object.hasOwn(paths, key)) {
      throw new MemoryOptionError(`duplicate key ${reprText(key)}`, { paramHint: "--memory-storage" });
    }
    paths[key] = path;
  }
  return paths;
}

/**
 * Build every memory `flow` declares from `--memory-root` and the repeated `--memory-storage
 * KEY=PATH` values.
 *
 * @param flow Loaded flow.
 * @param options.root The `--memory-root` value.
 * @param options.storageOptions The repeated `--memory-storage` values.
 * @param options.env The environment reflective and profile memory read; default `processEnv`.
 * @returns Registry to pass to the run and close afterwards.
 * @throws {MemoryOptionError} When a flag value is invalid for this flow.
 * @throws {MemorySetupError} When the flow declares a custom type, or building fails.
 */
export async function buildCliMemories(
  flow: ComposedFlow,
  {
    root,
    storageOptions,
    env = processEnv,
  }: { root: string | undefined; storageOptions: readonly string[]; env?: Env },
): Promise<MemoryRegistry> {
  const storage = parseMemoryStorage(storageOptions);
  const types = new Map(resolveMemoryLayout(flow.compiled).memories.map((memory) => [memory.key, memory.type]));
  const builtins = [...BUILTIN_MEMORY_CONFIG_KEYS.keys()].sort(compareCodeUnits).join(", ");
  for (const key of Object.keys(storage)) {
    const type = types.get(key);
    if (type === undefined) {
      const declared = [...types.keys()].sort(compareCodeUnits).join(", ") || "none";
      throw new MemoryOptionError(
        `key ${reprText(key)} is not a memory declared by this flow (declared: ${declared})`,
        { paramHint: "--memory-storage" },
      );
    }
    if (!BUILTIN_MEMORY_CONFIG_KEYS.has(type)) {
      throw new MemoryOptionError(
        `key ${reprText(key)} has custom type ${reprText(type)}; the CLI stores only built-in types (${builtins})`,
        { paramHint: "--memory-storage" },
      );
    }
  }
  const custom = [...types]
    .filter(([, type]) => !BUILTIN_MEMORY_CONFIG_KEYS.has(type))
    .map(([key]) => key)
    .sort(compareCodeUnits);
  if (custom.length > 0) {
    throw new MemorySetupError(
      `this flow declares custom memory type(s) for ${writeAsJson(custom)}; yian run builds only built-in ` +
        `types (${builtins}). Use the API: buildMemories(flow, { factories }) or { supplied }, then ` +
        "runFlow(flow, inputs, { memory }).",
    );
  }
  if (root !== undefined && flow.compiled.flowId === null) {
    throw new MemoryOptionError("--memory-root needs a flow with an id: field; memory lives under DIR/<flow id>/", {
      paramHint: "--memory-root",
    });
  }
  return buildMemories(flow, { root, storage, env });
}
