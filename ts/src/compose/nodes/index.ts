/**
 * The per-kind axis of the Compose front end: one descriptor class per author-written `kind:`.
 *
 * Each `compose/nodes/<kind>.ts` module owns its descriptor, a `NodeDescriptor` subclass whose
 * static `parse` reads one node body through a located `SourceCursor`, whose
 * `withSourceOrigins` attaches kind-owned source sites, and whose `build` constructs the runtime
 * node. `KINDS` maps the `kind:` text straight to that class, so the shared dispatch looks the
 * class up and calls it, and adding a kind is a local edit here. `BINDING_FIELDS` is the sibling
 * table: per kind, the descriptor fields that carry `${...}` binding text. Both derive from the
 * one `DESCRIPTORS` list. `REMOVED_KINDS` maps each retired kind to what to write instead.
 *
 * Knows about: diagnostics, compose source, helpers and peer phases, compile model, nodes,
 *              typesys and expr.
 * Never imports: the runtime, the run boundary, the CLI or provider SDKs.
 */

import type { NodeDescriptor } from "./base.ts";
// T3-15: agent, code, tool, wait, human_input
import { AgentDescriptor } from "./agent.ts";
import { CodeDescriptor } from "./code.ts";
import { ToolDescriptor } from "./tool.ts";
import { WaitDescriptor } from "./wait.ts";
// T3-16: case, call, map, loop

/** A descriptor class: `NodeDescriptor` or a kind's subclass. */
export type DescriptorClass = typeof NodeDescriptor;

/**
 * One entry per author-written kind. `start` and `end` are boundaries the loader makes, never
 * authored, so they are absent.
 */
export const DESCRIPTORS: readonly DescriptorClass[] = [
  // T3-15: agent, code, tool, wait, human_input
  AgentDescriptor,
  CodeDescriptor,
  ToolDescriptor,
  WaitDescriptor,
  // T3-16: case, call, map, loop
];

/** The `kind:` text to its descriptor class, for the shared parse and build dispatch. */
export const KINDS: ReadonlyMap<string, DescriptorClass> = new Map(DESCRIPTORS.map((d) => [d.kind, d]));

/** The `kind:` text to the descriptor fields that carry `${...}` binding text. */
export const BINDING_FIELDS: ReadonlyMap<string, readonly string[]> = new Map(
  DESCRIPTORS.map((d) => [d.kind, d.bindingFields]),
);

/**
 * Each removed kind to what to write instead, so loading one names the replacement rather than
 * the generic unknown-kind error.
 */
export const REMOVED_KINDS: ReadonlyMap<string, string> = new Map([
  ["model", "serve the model behind a host tool and call it with kind: tool (set tool_id: and input:)"],
]);
