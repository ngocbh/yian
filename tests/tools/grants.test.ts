import { expect, test } from "bun:test";
import {
  expandGrants,
  GrantError,
  type GrantScope,
  grantText,
  parseToolGrants,
  type ToolGrant,
  visibleToolName,
} from "../../src/tools/grants.ts";

test("each grant form parses, in order", () => {
  expect(parseToolGrants(["ask_user", "github", { lessons: ["read", "search"] }], "node 'a'")).toEqual([
    { kind: "name", name: "ask_user" },
    { kind: "name", name: "github" },
    { kind: "group", group: "lessons", tools: ["read", "search"] },
  ]);
});

test.each([
  [[{ a: [] }], "`tools:` entry {a: []} lists no tools; write `{a: [read]}` or `a`"],
  [[{ a: ["x"], b: ["y"] }], "`tools:` entry with two keys; write one group per entry: `- a: [x]`"],
  [[{ a: "x" }], "`tools:` entry {a: x} must list tools; write `{a: [x]}`"],
  [[{ a: ["x", "x"] }], "`tools:` grants a.x twice"],
  [["a", "a"], "`tools:` grants 'a' twice"],
])("a bad grant is refused with its fix: %j", (raw, message) => {
  expect(() => parseToolGrants(raw, "node 'n'")).toThrow(message);
});

test("visible names follow <group>_<tool>; a bare tool keeps its name", () => {
  expect(visibleToolName("lessons", "read")).toBe("lessons_read");
  expect(visibleToolName(null, "search")).toBe("search");
  expect(grantText({ kind: "group", group: "lessons", tools: ["read"] })).toBe("{lessons: [read]}");
});

const run: GrantScope = {
  label: "this run",
  group: (name) => ({ github: ["search", "create_issue"], docs: ["search"] })[name] ?? null,
  hasTool: (name) => name === "ask_user" || name === "lookup",
  groupsHolding: (name) => (name === "search" ? ["github", "docs"] : name === "create_issue" ? ["github"] : []),
  groupNames: () => ["github", "docs"],
  identity: (group, tool) => (group === null ? tool : `${group}_${tool}`),
  bareMembers: true,
};
const local: GrantScope = {
  label: "this flow's memories",
  group: (name) => (name === "lessons" ? ["list", "read", "append"] : null),
  hasTool: () => false,
  groupsHolding: () => [],
  groupNames: () => ["lessons"],
  identity: (group, tool) => `${group}_${tool}`,
  bareMembers: false,
};
// One object `web` sits in both groups; its identity is its first group's name for it.
const shared: GrantScope = {
  label: "this run",
  group: (name) => ({ toolbox: ["web", "a"], company: ["web", "b"] })[name] ?? null,
  hasTool: () => false,
  groupsHolding: (name) => (name === "web" ? ["toolbox"] : name === "a" ? ["toolbox"] : name === "b" ? ["company"] : []),
  groupNames: () => ["toolbox", "company"],
  identity: (group, tool) => (tool === "web" ? "toolbox_web" : `${group}_${tool}`),
  bareMembers: true,
};

test("grants expand in grant order, groups in their own order, local scope first", () => {
  const granted = expandGrants(parseToolGrants(["lookup", "lessons", { github: ["create_issue"] }], "n"), [local, run]);
  expect(granted.map((each) => each.visible)).toEqual(["lookup", "lessons_list", "lessons_read", "lessons_append", "github_create_issue"]);
  expect(granted[1]).toEqual({ visible: "lessons_list", group: "lessons", tool: "list", scope: 0, key: "lessons_list" });
});

test.each([
  [["nope"], "grants unknown tool or group 'nope'; tool groups here: [lessons, github, docs]"],
  [[{ lessons: ["upsert"] }], "`lessons` has no tool `upsert`; it has list, read, append"],
  [["search"], "'search' is a tool of groups 'github' and 'docs'; grant one as `{github: [search]}`"],
  [["read"], "grants unknown tool or group 'read'"], // a bare name never reaches a memory tool
])("%j is refused", (raw, message) => {
  expect(() => expandGrants(parseToolGrants(raw, "n"), [local, run])).toThrow(message);
});

test("a refused grant is a GrantError; one scope names itself in the unknown message", () => {
  expect(() => expandGrants(parseToolGrants(["nope"], "n"), [run])).toThrow(GrantError);
  expect(() => expandGrants(parseToolGrants(["nope"], "n"), [run])).toThrow(
    "grants unknown tool or group 'nope'; tool groups in this run: [github, docs]",
  );
});

test("two grants that give one visible name to different tools clash", () => {
  const clash: GrantScope = { ...run, hasTool: (n) => n === "github_search", identity: (g, t) => (g === null ? `top:${t}` : `${g}.${t}`) };
  expect(() => expandGrants(parseToolGrants(["github_search", { github: ["search"] }], "n"), [clash])).toThrow(
    "tools 'github_search' (tool github_search) and 'github_search' (github's search) have one name; grant one",
  );
});

test("a tool granted twice the same way is bound once", () => {
  expect(expandGrants(parseToolGrants(["github", { github: ["search"] }], "n"), [run]).map((g) => g.visible))
    .toEqual(["github_search", "github_create_issue"]);
});

// Built directly: a node built in code may repeat a grant, which `tools:` refuses at load.
const names = (raw: readonly string[]): ToolGrant[] => raw.map((name) => ({ kind: "name", name }));

test.each([
  [["toolbox", "company", "toolbox"], ["toolbox_web", "toolbox_a", "company_b"]],
  [["company"], ["company_web", "company_b"]],
  [["company", "toolbox"], ["company_web", "company_b", "toolbox_a"]],
  [["web", "company"], ["web", "company_b"]],
])("one tool object reached through several grants is bound once, under the first one's name: %j", (raw, visible) => {
  expect(expandGrants(names(raw), [shared]).map((g) => g.visible)).toEqual(visible);
});
