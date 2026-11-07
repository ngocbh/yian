import { expect, test } from "bun:test";
import { grantText, parseToolGrants, visibleToolName } from "../../src/tools/grants.ts";

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
