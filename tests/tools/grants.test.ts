import { expect, test } from "bun:test";
import { ToolSpec } from "../../src/llm/messages.ts";
import {
  checkVisibleNames,
  expandGrants,
  GrantError,
  type GrantedTool,
  type GrantScope,
  grantText,
  parseToolGrants,
  requestToolSpecs,
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

const PARAMS = { type: "object", properties: { q: { type: "string" } } };
const docsSearch: GrantedTool = { visible: "docs_search", group: "docs", tool: "search", scope: 0, key: "docs_search" };
const lookup: GrantedTool = { visible: "lookup", group: null, tool: "lookup", scope: 0, key: "lookup" };
const specOf = (tool: GrantedTool) => new ToolSpec({ name: tool.tool, description: `The ${tool.tool} tool.`, parameters: PARAMS });
const loadSkill = new ToolSpec({ name: "load_skill", description: "Load a skill.", parameters: PARAMS });

test("a request lists each granted tool under its visible name, in grant order, then load_skill", () => {
  const specs = requestToolSpecs([docsSearch, lookup], specOf, loadSkill)!;
  expect(specs.map((spec) => [spec.name, spec.description])).toEqual([
    ["docs_search", "The search tool."],
    ["lookup", "The lookup tool."],
    ["load_skill", "Load a skill."],
  ]);
  expect(specs[0]!.parameters).toEqual(PARAMS);
});

test("a request with nothing granted lists no tools; a tool with no spec gives no list", () => {
  expect(requestToolSpecs([], specOf, null)).toEqual([]);
  expect(requestToolSpecs([docsSearch, lookup], (tool) => (tool === lookup ? null : specOf(tool)), loadSkill)).toBeNull();
});

const shownAs = (group: string | null, tool: string): GrantedTool => ({
  visible: group === null ? tool : `${group}_${tool}`,
  group,
  tool,
  scope: 0,
  key: tool,
});
const RULE = "a model's tool names take letters, digits, `_` and `-`, start with a letter or `_`, and are at most 64 characters";

test.each<[GrantedTool, string]>([
  [shownAs("files.v2", "read"), `grants group \`files.v2\`'s \`read\`, which would show as \`files.v2_read\`; ${RULE}; rename the group`],
  [shownAs("2files", "read"), `grants group \`2files\`'s \`read\`, which would show as \`2files_read\`; ${RULE}; rename the group`],
  [shownAs("files", "read.all"), `grants group \`files\`'s \`read.all\`, which would show as \`files_read.all\`; ${RULE}; rename the tool`],
  [shownAs("a".repeat(40), "b".repeat(30)), `which would show as \`${"a".repeat(40)}_${"b".repeat(30)}\`; ${RULE}; shorten the group or the tool`],
  [shownAs(null, "x.y"), `grants tool \`x.y\`, which would show as \`x.y\`; ${RULE}; rename the tool`],
])("a name a model can't call is refused: %#", (granted, message) => {
  expect(() => checkVisibleNames([shownAs("ok", "fine"), granted])).toThrow(message);
});

test("names a model can call pass", () => {
  expect(() => checkVisibleNames([shownAs("my-server", "get-file"), shownAs(null, "_x"), shownAs("a".repeat(32), "b".repeat(31))])).not.toThrow();
});
