/**
 * Tool grants: the entries of an AGENT's `tools:` list, and the names a model sees for them.
 *
 * A grant is a tool or a group by name (`search`, `github`), or some tools of one group
 * (`{github: [search, create_issue]}`). A tool reached through a group shows to the model as
 * `<group>_<tool>`; a tool granted by its own name keeps it.
 *
 * Knows about: `diagnostics` and typesys.
 */

import { reprText } from "../diagnostics.ts";
import { isRecord } from "../typesys/values.ts";

/** One authored grant: a tool or a group by name, or some tools of a group. */
export type ToolGrant =
  | { readonly kind: "name"; readonly name: string }
  | { readonly kind: "group"; readonly group: string; readonly tools: readonly string[] };

/** A grant that can't be parsed or resolved; its message names the fix. */
export class GrantError extends RangeError {
  /** The index of the `tools:` entry at fault, or `null` when no one entry is. */
  readonly index: number | null;

  constructor(message: string, index: number | null = null) {
    super(message);
    this.name = "GrantError";
    this.index = index;
  }
}

/** The name a model sees for `tool` of `group`; a tool granted by its own name keeps it. */
export function visibleToolName(group: string | null, tool: string): string {
  return group === null ? tool : `${group}_${tool}`;
}

/**
 * Parse an authored `tools:` list.
 *
 * @param raw The list's entries as YAML read them.
 * @param where The node, for the message (`node 'a' (kind=agent)`).
 * @returns The grants, in order.
 * @throws {GrantError} An entry is neither a non-empty name nor a one-key mapping to a non-empty
 *   list of tool names, or the list grants one name, or one group's tool, twice. The message names
 *   the fix, and `index` the entry.
 */
export function parseToolGrants(raw: readonly unknown[], where: string): ToolGrant[] {
  const fail = (message: string, index: number): never => {
    throw new GrantError(`${where}: \`tools:\` ${message}`, index);
  };
  const grants: ToolGrant[] = [];
  const seen = new Set<string>();
  raw.forEach((entry, index) => {
    if (typeof entry === "string") {
      if (entry === "") fail("has an empty tool name", index);
      if (seen.has(`name:${entry}`)) fail(`grants ${reprText(entry)} twice`, index);
      seen.add(`name:${entry}`);
      grants.push(Object.freeze({ kind: "name", name: entry }));
      return;
    }
    if (!isRecord(entry)) {
      fail(`entry ${flowText(entry)} must be a tool name or a group grant; write \`search\` or \`{github: [search]}\``, index);
    }
    const keys = Object.keys(entry as Record<string, unknown>);
    if (keys.length === 0) fail("entry {} names no group; write `{github: [search]}`", index);
    const group = keys[0]!;
    const value = (entry as Record<string, unknown>)[group];
    if (keys.length > 1) {
      const count = keys.length === 2 ? "two" : String(keys.length);
      const first = Array.isArray(value) ? flowText(value) : `[${flowText(value)}]`;
      fail(`entry with ${count} keys; write one group per entry: \`- ${group}: ${first}\``, index);
    }
    if (group === "") fail("entry names an empty group", index);
    const shown = flowText(entry);
    if (!Array.isArray(value)) {
      const example = typeof value === "string" && value !== "" ? value : "read";
      fail(`entry ${shown} must list tools; write \`{${group}: [${example}]}\``, index);
    }
    const tools = value as unknown[];
    if (tools.length === 0) fail(`entry ${shown} lists no tools; write \`{${group}: [read]}\` or \`${group}\``, index);
    if (!tools.every((tool) => typeof tool === "string" && tool !== "")) {
      fail(`entry ${shown} must list tool names; write \`{${group}: [read]}\``, index);
    }
    for (const tool of tools as string[]) {
      const key = `group:${JSON.stringify([group, tool])}`;
      if (seen.has(key)) fail(`grants ${group}.${tool} twice`, index);
      seen.add(key);
    }
    grants.push(Object.freeze({ kind: "group", group, tools: Object.freeze([...(tools as string[])]) }));
  });
  return grants;
}

/** The grant as an author writes it: `lessons` or `{lessons: [read, search]}`. */
export function grantText(grant: ToolGrant): string {
  return grant.kind === "name" ? grant.name : `{${grant.group}: [${grant.tools.join(", ")}]}`;
}

/** The grant as the manifest encodes it: a name grant is its text, a group grant `{group: [tools]}`. */
export function authoredGrant(grant: ToolGrant): string | Readonly<Record<string, readonly string[]>> {
  // `fromEntries` defines an own data property, so a `__proto__` group stays a key.
  return grant.kind === "name" ? grant.name : Object.fromEntries([[grant.group, [...grant.tools]]]);
}

/** Return a YAML value in flow style, for a message: `x`, `[x, y]`, `{a: [x]}`. */
function flowText(value: unknown): string {
  if (typeof value === "string") return value === "" ? "''" : value;
  if (Array.isArray(value)) return `[${value.map(flowText).join(", ")}]`;
  if (isRecord(value)) {
    return `{${Object.entries(value)
      .map(([key, item]) => `${key}: ${flowText(item)}`)
      .join(", ")}}`;
  }
  return String(value);
}
