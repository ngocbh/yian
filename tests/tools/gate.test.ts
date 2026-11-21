/**
 * The host's check before each agent tool call: the tool sources a run records, the bound on its
 * texts, the internal seam that carries the hook, and the run tool set that holds it.
 */

import { expect, test } from "bun:test";

import {
  type BeforeToolCall,
  checkGateText,
  GATE_TEXT_MAX_BYTES,
  gateOf,
  isToolSource,
  withGate,
} from "../../src/tools/gate.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { YianRangeError, YianTypeError } from "../../src/typesys/errors.ts";
import { gated } from "../helpers/gate.ts";

test.each([
  { kind: "host", group: null, tool: "lookup" },
  { kind: "host", group: "crm", tool: "find" },
  { kind: "mcp", server: "github", tool: "search" },
  { kind: "module", alias: "crm", ref: "./crm.ts", export: "find" },
  { kind: "memory", input: "lessons", tool: "read" },
  { kind: "flow", flow: "triage", library: null },
  { kind: "flow", flow: "triage", library: "ops" },
])("a tool source of each kind passes: %o", (source) => {
  expect(isToolSource(source)).toBe(true);
});

test.each<[string, unknown]>([
  ["null", null],
  ["a string", "host"],
  ["an unknown kind", { kind: "skill", tool: "x" }],
  ["a missing field", { kind: "host", tool: "lookup" }],
  ["an extra field", { kind: "mcp", server: "s", tool: "t", group: null }],
  ["an empty name", { kind: "memory", input: "", tool: "read" }],
  ["a null server", { kind: "mcp", server: null, tool: "t" }],
  ["a non-text export", { kind: "module", alias: "a", ref: "./a.ts", export: 1 }],
  ["an empty group", { kind: "host", group: "", tool: "t" }],
  ["an array", [{ kind: "host", group: null, tool: "t" }]],
])("isToolSource refuses %s", (_name, value) => {
  expect(isToolSource(value)).toBe(false);
});

test("checkGateText returns a text within the bound, counted in UTF-8 bytes", () => {
  expect(checkGateText("ok", "note")).toBe("ok");
  expect(checkGateText("a".repeat(GATE_TEXT_MAX_BYTES), "note")).toHaveLength(GATE_TEXT_MAX_BYTES);
  // Two bytes each in UTF-8: one past the bound in bytes though not in code units.
  const wide = "é".repeat(GATE_TEXT_MAX_BYTES / 2);
  expect(checkGateText(wide, "note")).toBe(wide);
  expect(() => checkGateText(`${wide}a`, "note")).toThrow(
    new YianRangeError(`note is ${GATE_TEXT_MAX_BYTES + 1} UTF-8 bytes, over the limit of ${GATE_TEXT_MAX_BYTES}`),
  );
});

test.each<[string, unknown, Error]>([
  ["a number", 7, new YianTypeError("refusal must be a string")],
  ["an empty text", "", new YianRangeError("refusal must be a non-empty string")],
  ["a lone surrogate", "a\uD800", new YianRangeError("refusal has a lone surrogate")],
])("checkGateText refuses %s", (_name, value, error) => {
  expect(() => checkGateText(value, "refusal")).toThrow(error);
});

test("the seam carries a hook on the options object it registers, and nothing else does", () => {
  const hook: BeforeToolCall = () => undefined;
  const options = { log: "x" };
  expect(withGate(options, hook)).toBe(options);
  expect(gateOf(options)).toBe(hook);
  // A copy isn't registered.
  expect(gateOf({ ...options })).toBeNull();
  expect(gateOf({})).toBeNull();
  expect(gateOf(withGate(options, null))).toBeNull();
});

test("the seam refuses a hook that isn't a function or null", () => {
  expect(() => withGate({}, 5 as never)).toThrow(new YianTypeError("beforeToolCall must be a function or null"));
});

test("the test helper reaches the seam", () => {
  const hook: BeforeToolCall = () => undefined;
  expect(gateOf(gated({ log: "x" }, hook))).toBe(hook);
});

test("a tool set holds its gate across every copy", () => {
  const hook: BeforeToolCall = () => undefined;
  const tools = new ToolSet({});
  expect(tools.gate).toBeNull();
  const gatedTools = tools.withGate(hook);
  expect([tools.gate, gatedTools.gate]).toEqual([null, hook]);
  expect(gatedTools.forRun("run").gate).toBe(hook);
  expect(gatedTools.withModules([]).gate).toBe(hook);
  expect(gatedTools.withNodeScope({ groups: new Map(), tools: [], identities: new Map() }).gate).toBe(hook);
  expect(gatedTools.withGate(null).gate).toBeNull();
});
