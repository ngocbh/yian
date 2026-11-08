/**
 * Tool grants: the entries of an AGENT's `tools:` list, and the names a model sees for them.
 *
 * A grant is a tool or a group by name (`search`, `github`), or some tools of one group
 * (`{github: [search, create_issue]}`). A tool reached through a group shows to the model as
 * `<group>_<tool>`; a tool granted by its own name keeps it.
 *
 * Knows about: `diagnostics`, typesys and `llm/messages` (a request's tool specs).
 */

import { reprText } from "../diagnostics.ts";
import { ToolSpec } from "../llm/messages.ts";
import { isRecord } from "../typesys/values.ts";

/** One authored grant: a tool or a group by name, or some tools of a group. */
export type ToolGrant =
  | { readonly kind: "name"; readonly name: string }
  | { readonly kind: "group"; readonly group: string; readonly tools: readonly string[] };

/**
 * A grant of a memory's tools, by the memory input of the node's own flow: `lessons` grants all
 * its tools (`tools: null`), `{lessons: [read]}` some of them.
 */
export type MemoryGrant = { readonly input: string; readonly tools: readonly string[] | null };

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
      const fix = "write `search` or `{github: [search]}`";
      fail(`entry ${flowText(entry)} must be a tool name or a group grant; ${fix}`, index);
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
    if (tools.length === 0) {
      fail(`entry ${shown} lists no tools; write \`{${group}: [read]}\` or \`${group}\``, index);
    }
    if (!tools.every((tool) => typeof tool === "string" && tool !== "")) {
      fail(`entry ${shown} must list tool names; write \`{${group}: [read]}\``, index);
    }
    for (const tool of tools as string[]) {
      const key = `group:${JSON.stringify([group, tool])}`;
      if (seen.has(key)) fail(`grants ${group}.${tool} twice`, index);
      seen.add(key);
    }
    const names = Object.freeze([...(tools as string[])]);
    grants.push(Object.freeze({ kind: "group", group, tools: names }));
  });
  return grants;
}

/** Return `item` as a grant: a text is a name grant, as a node built in code may pass. */
export function asToolGrant(item: string | ToolGrant): ToolGrant {
  return typeof item === "string" ? Object.freeze({ kind: "name", name: item }) : item;
}

/** The grant as an author writes it: `lessons` or `{lessons: [read, search]}`. */
export function grantText(grant: ToolGrant): string {
  return grant.kind === "name" ? grant.name : `{${grant.group}: [${grant.tools.join(", ")}]}`;
}

/**
 * The grant as the manifest encodes it: a name grant is its text, a group grant
 * `{group: [tools]}`.
 */
export function authoredGrant(
  grant: ToolGrant,
): string | Readonly<Record<string, readonly string[]>> {
  // `fromEntries` defines an own data property, so a `__proto__` group stays a key.
  return grant.kind === "name" ? grant.name : Object.fromEntries([[grant.group, [...grant.tools]]]);
}

/** Where grants resolve; `expandGrants` tries scopes in order. */
export interface GrantScope {
  /** What the scope is, for messages: `this run`. */
  readonly label: string;
  /** Group `name`'s tool names in spec order, or `null` when this scope has no such group. */
  group(name: string): readonly string[] | null;
  /** Whether this scope has a top-level tool `name`. */
  hasTool(name: string): boolean;
  /** The groups of this scope that hold a tool named `name`, one per distinct tool. */
  groupsHolding(name: string): readonly string[];
  /** This scope's group names, in order, for messages. */
  groupNames(): readonly string[];
  /** A key equal for two names of one tool: `group` is `null` for a top-level tool. */
  identity(group: string | null, tool: string): string;
  /** Whether a bare name may reach a group's tool here (never for memory scopes). */
  readonly bareMembers: boolean;
}

/**
 * Return a scope of groups local to one node, such as its memories: each group's tool names, in
 * order. Only a grant naming a group reaches one; a bare name never does. A tool's identity is its
 * group and name, so two groups' tools that would show one name stay two tools.
 */
export function localGroupScope(label: string, groups: ReadonlyMap<string, readonly string[]>): GrantScope {
  return {
    label,
    bareMembers: false,
    group: (name) => groups.get(name) ?? null,
    hasTool: () => false,
    groupsHolding: () => [],
    groupNames: () => [...groups.keys()],
    identity: (group, tool) => JSON.stringify([group, tool]),
  };
}

/** One granted tool: the name the model sees, the tool it names, and that tool's identity. */
export interface GrantedTool {
  /** The name the model sees: `<group>_<tool>` through a group grant, else the tool's own name. */
  readonly visible: string;
  /** The group the tool was reached through, or `null` for a top-level tool. */
  readonly group: string | null;
  /** The tool's own name in its group, or its top-level name. */
  readonly tool: string;
  /** The index of the scope it resolved in. */
  readonly scope: number;
  /** The tool's identity in that scope (`GrantScope.identity`). */
  readonly key: string;
}

/**
 * Expand grants over ordered scopes into the tools a model is given, in grant order.
 *
 * A bare name is a whole group when a scope has that group (each tool shown as `<group>_<tool>`),
 * else a scope's top-level tool, else, where the scope allows bare members, the one group tool by
 * that name, shown under its own name. A group grant takes the tools it lists, in its order, from
 * the first scope with that group. A tool reached twice is bound once, at its first place and
 * under that place's name.
 *
 * @throws {GrantError} A name is unknown, or a tool of two groups; a group grant names a group no
 *   scope has, or a tool its group lacks; or two different tools would show one name. The message
 *   starts with `grants`, so a caller prefixes the node (`AGENT node 'a' grants ...`).
 */
export function expandGrants(
  grants: readonly ToolGrant[],
  scopes: readonly GrantScope[],
): GrantedTool[] {
  const granted: GrantedTool[] = [];
  const bound = new Set<string>();
  const byVisible = new Map<string, GrantedTool>();
  for (const grant of grants) {
    const { scope, group, tools, grouped } = resolveGrant(grant, scopes);
    for (const name of tools) {
      const visible = grouped ? visibleToolName(group, name) : name;
      const key = scopes[scope]!.identity(group, name);
      const tool: GrantedTool = { visible, group, tool: name, scope, key };
      const id = JSON.stringify([tool.scope, tool.key]);
      if (bound.has(id)) continue;
      const other = byVisible.get(tool.visible);
      if (other !== undefined) {
        const first = `${reprText(other.visible)} (${placeText(other)})`;
        const second = `${reprText(tool.visible)} (${placeText(tool)})`;
        throw new GrantError(
          `grants ${grantText(grant)}: tools ${first} and ${second} have one name; grant one`,
        );
      }
      bound.add(id);
      byVisible.set(tool.visible, tool);
      granted.push(Object.freeze(tool));
    }
  }
  return granted;
}

/**
 * Return the groups `grants` reach, once each, in grant order: a group granted whole or in group
 * form, and the group holding a tool granted by its bare name. Grants resolve as `expandGrants`
 * resolves them.
 *
 * @throws {GrantError} As `expandGrants` throws, but for names that clash.
 */
export function reachedGroups(
  grants: readonly ToolGrant[],
  scopes: readonly GrantScope[],
): { readonly scope: number; readonly group: string }[] {
  const reached = new Map<string, { readonly scope: number; readonly group: string }>();
  for (const grant of grants) {
    const { scope, group } = resolveGrant(grant, scopes);
    if (group === null) continue;
    reached.set(JSON.stringify([scope, group]), Object.freeze({ scope, group }));
  }
  return [...reached.values()];
}

/**
 * Where one grant resolves: a scope, the group its tools are in (`null` for a top-level tool), and
 * those tools.
 */
interface ResolvedGrant {
  readonly scope: number;
  readonly group: string | null;
  readonly tools: readonly string[];
  /** Whether the tools show as `<group>_<tool>`: a group granted whole or in group form. */
  readonly grouped: boolean;
}

/**
 * Resolve one grant over `scopes`, as `expandGrants` describes.
 *
 * @throws {GrantError} The grant names an unknown tool or group, a tool of two groups by its bare
 *   name, or a tool its group lacks.
 */
function resolveGrant(grant: ToolGrant, scopes: readonly GrantScope[]): ResolvedGrant {
  if (grant.kind === "group") {
    const scope = scopes.findIndex((each) => each.group(grant.group) !== null);
    if (scope < 0) {
      const known = groupsText(scopes);
      throw new GrantError(`grants unknown tool group ${reprText(grant.group)}; ${known}`);
    }
    const members = scopes[scope]!.group(grant.group)!;
    const missing = grant.tools.find((tool) => !members.includes(tool));
    if (missing !== undefined) {
      const has = members.length === 0 ? "no tools" : members.join(", ");
      throw new GrantError(
        `grants ${grantText(grant)}: \`${grant.group}\` has no tool \`${missing}\`; it has ${has}`,
      );
    }
    return { scope, group: grant.group, tools: grant.tools, grouped: true };
  }
  const name = grant.name;
  const groupAt = scopes.findIndex((each) => each.group(name) !== null);
  if (groupAt >= 0) {
    return { scope: groupAt, group: name, tools: scopes[groupAt]!.group(name)!, grouped: true };
  }
  const toolAt = scopes.findIndex((each) => each.hasTool(name));
  if (toolAt >= 0) return { scope: toolAt, group: null, tools: [name], grouped: false };
  const holders = scopes.flatMap((each, at) =>
    each.bareMembers ? each.groupsHolding(name).map((group) => [at, group] as const) : [],
  );
  if (holders.length === 0) {
    const shown = groupToolShownAs(name, scopes);
    const hint = shown === null ? "" : `; did you mean \`{${shown.group}: [${shown.tool}]}\`?`;
    throw new GrantError(
      `grants unknown tool or group ${reprText(name)}; ${groupsText(scopes)}${hint}`,
    );
  }
  if (holders.length > 1) {
    const groups = holders.map(([, group]) => reprText(group));
    throw new GrantError(
      `grants ${reprText(name)}: ${reprText(name)} is a tool of groups ${andText(groups)}; ` +
        `grant one as \`{${holders[0]![1]}: [${name}]}\``,
    );
  }
  const [scope, group] = holders[0]!;
  return { scope, group, tools: [name], grouped: false };
}

/**
 * Return a request's tool list: each granted tool's spec under its visible name, in grant order,
 * then `loadSkill` when given.
 *
 * @param specOf The spec of a granted tool as its host describes it, or `null` when it has none.
 * @returns The specs, empty when nothing is granted; `null` when `specOf` has no spec for a tool.
 */
export function requestToolSpecs(
  granted: readonly GrantedTool[],
  specOf: (tool: GrantedTool) => ToolSpec | null,
  loadSkill: ToolSpec | null,
): ToolSpec[] | null {
  const specs: ToolSpec[] = [];
  for (const tool of granted) {
    const spec = specOf(tool);
    if (spec === null) return null;
    const { description, parameters, strict } = spec;
    specs.push(new ToolSpec({ name: tool.visible, description, parameters, strict }));
  }
  if (loadSkill !== null) specs.push(loadSkill);
  return specs;
}

/**
 * Return the group tool whose visible name is `name` (`github_search` is `github`'s `search`), or
 * `null` when no group of `scopes` shows a tool under it.
 */
export function groupToolShownAs(
  name: string,
  scopes: readonly GrantScope[],
): { readonly group: string; readonly tool: string } | null {
  for (const scope of scopes) {
    for (const group of scope.groupNames()) {
      const tool = name.slice(group.length + 1);
      if (!name.startsWith(`${group}_`)) continue;
      if (scope.group(group)!.includes(tool)) return { group, tool };
    }
  }
  return null;
}

// The strictest of the providers' tool-name rules, so a flow runs on any of them: OpenAI and
// Anthropic allow up to 64 of [A-Za-z0-9_-]; Gemini also wants a letter or `_` first.
const CALLABLE_NAME = /^[A-Za-z_][A-Za-z0-9_-]{0,63}$/;
const CALLABLE_RULE =
  "a model's tool names take letters, digits, `_` and `-`, start with a letter or `_`, " +
  "and are at most 64 characters";

/**
 * Refuse a granted tool whose visible name a model can't call: a group or tool name with other
 * characters, or a `<group>_<tool>` past 64 characters.
 *
 * @throws {GrantError} Naming the first such tool and what to rename; the message starts with
 *   `grants`, as `expandGrants`' do.
 */
export function checkVisibleNames(granted: readonly GrantedTool[]): void {
  for (const tool of granted) {
    if (CALLABLE_NAME.test(tool.visible)) continue;
    const place =
      tool.group === null ? `tool \`${tool.tool}\`` : `group \`${tool.group}\`'s \`${tool.tool}\``;
    let fix: string;
    if (tool.group === null || tool.visible === tool.tool) fix = "rename the tool";
    else if (!CALLABLE_NAME.test(tool.group)) fix = "rename the group";
    else if (!/^[A-Za-z0-9_-]+$/.test(tool.tool)) fix = "rename the tool";
    else fix = "shorten the group or the tool";
    throw new GrantError(
      `grants ${place}, which would show as \`${tool.visible}\`; ${CALLABLE_RULE}; ${fix}`,
    );
  }
}

/** Return where a granted tool comes from, for a message: `tool x` or `github's search`. */
function placeText(tool: GrantedTool): string {
  return tool.group === null ? `tool ${tool.tool}` : `${tool.group}'s ${tool.tool}`;
}

/** Return the scopes' group names for a message: `tool groups in this run: [a, b]`. */
function groupsText(scopes: readonly GrantScope[]): string {
  const where = scopes.length === 1 ? `in ${scopes[0]!.label}` : "here";
  return `tool groups ${where}: [${scopes.flatMap((scope) => scope.groupNames()).join(", ")}]`;
}

/** Join items as `a`, `a and b`, or `a, b and c`. */
function andText(items: readonly string[]): string {
  if (items.length < 2) return items.join("");
  return `${items.slice(0, -1).join(", ")} and ${items.at(-1)!}`;
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
