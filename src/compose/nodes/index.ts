/**
 * The per-kind axis of the Compose front end: one descriptor class per author-written `kind:`.
 *
 * Each `compose/nodes/<kind>.ts` module owns its descriptor, a `NodeDescriptor` subclass whose
 * static `parse` reads one node body through a located `SourceCursor`, whose
 * `withSourceOrigins` attaches kind-owned source sites, and whose `build` constructs the runtime
 * node. `KINDS` maps the `kind:` text straight to that class, so the shared dispatch looks the
 * class up and calls it, and adding a kind is a local edit here. `BINDING_FIELDS` is the sibling
 * table: per kind, the descriptor fields that carry `${...}` binding text. Both derive from the
 * one `DESCRIPTORS` list; `BINDING_FIELDS` also holds the TOOL node's, which a `call`, `map` or
 * `loop` naming a tool builds, though no `kind:` names it. `REMOVED_KINDS` maps each retired kind
 * to what to write instead, and `removedKindFix` shows a `kind: tool` body its own rewrite.
 *
 * Knows about: diagnostics, compose source, helpers and peer phases, compile model, nodes,
 *              typesys and expr.
 */

import { isRecord } from "../../typesys/values.ts";
import { parseYamlLiteral, SourceSyntaxError } from "../source/index.ts";
import type { NodeDescriptor } from "./base.ts";
import { AgentDescriptor } from "./agent.ts";
import { CodeDescriptor } from "./code.ts";
import { ToolDescriptor } from "./tool.ts";
import { CallDescriptor } from "./call.ts";
import { CaseDescriptor } from "./case.ts";
import { LoopDescriptor } from "./loop.ts";
import { MapDescriptor } from "./map.ts";

/** A descriptor class: `NodeDescriptor` or a kind's subclass. */
export type DescriptorClass = typeof NodeDescriptor;

/**
 * One entry per author-written kind. `start` and `end` are boundaries the loader makes, never
 * authored, so they are absent, and so is the TOOL node, which a call naming a tool builds.
 */
const DESCRIPTORS: readonly DescriptorClass[] = [
  AgentDescriptor,
  CodeDescriptor,
  CaseDescriptor,
  CallDescriptor,
  MapDescriptor,
  LoopDescriptor,
];

/** The `kind:` text to its descriptor class, for the shared parse and build dispatch. */
export const KINDS: ReadonlyMap<string, DescriptorClass> = new Map(DESCRIPTORS.map((d) => [d.kind, d]));

/** The `kind:` text, and the TOOL node's kind, to the descriptor fields that carry `${...}` binding text. */
export const BINDING_FIELDS: ReadonlyMap<string, readonly string[]> = new Map(
  [...DESCRIPTORS, ToolDescriptor].map((d) => [d.kind, d.bindingFields]),
);

// Stands for the tool in a fix that can't name it.
const TOOL_PLACEHOLDER = "<tool>";
// The characters of a name a fix can show as written, on one line.
const NAME_CHARS = /^[A-Za-z0-9_.-]+$/;

/**
 * Each removed kind to what to write instead, so loading one names the replacement rather than
 * the generic unknown-kind error.
 */
export const REMOVED_KINDS: ReadonlyMap<string, string> = new Map([
  ["model", "serve the model behind a host tool, then write `kind: call` with `call: <tool>` and `input:`"],
  ["human_input", "to ask a person, write `kind: call` with `call: ask` (input: prompt, questions, timeout)"],
  ["wait", "to wait, write `kind: call` with `call: sleep` (input: until or for)"],
  ["tool", `write \`kind: call\` with \`call: ${TOOL_PLACEHOLDER}\``],
]);

/**
 * What to write in place of the removed kind `kind`, or `undefined` for a kind that wasn't
 * removed. A `kind: tool` body is shown the `call:` its `tool_id:` (`toolId`, `null` when absent)
 * becomes; any other kind's fix is its `REMOVED_KINDS` entry.
 */
export function removedKindFix(kind: string, toolId: unknown): string | undefined {
  const fix = REMOVED_KINDS.get(kind);
  return kind === "tool" ? fix!.replace(TOOL_PLACEHOLDER, toolIdAsCall(toolId)) : fix;
}

/**
 * The value of the `call:` that the removed `tool_id:` value `value` becomes: a plain name, or
 * `{group: tool}` for a one-key mapping of plain names; `<tool>` for any other value.
 */
export function toolIdAsCall(value: unknown): string {
  if (plainName(value)) return value;
  const entries = isRecord(value) ? Object.entries(value) : [];
  if (entries.length !== 1) return TOOL_PLACEHOLDER;
  const [group, name] = entries[0]!;
  return plainName(group) && plainName(name) ? `{${group}: ${name}}` : TOOL_PLACEHOLDER;
}

/**
 * Whether `value` is a name a fix can show as written: name characters on one line, which YAML
 * reads back as the same text, not as `true`, `null` or `1e3`.
 */
function plainName(value: unknown): value is string {
  if (typeof value !== "string" || !NAME_CHARS.test(value)) return false;
  try {
    return parseYamlLiteral(value) === value;
  } catch (error) {
    if (error instanceof SourceSyntaxError) return false;
    throw error;
  }
}
