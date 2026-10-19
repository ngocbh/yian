/**
 * The kind dispatch tables `KINDS` and `BINDING_FIELDS`, and `REMOVED_KINDS`: a port of
 * `tests/compose/test_kinds_table.py`.
 *
 * Ported: `test_kinds_tables_present`, `test_kinds_maps_string_to_descriptor_class`,
 * `test_all_authored_kinds_registered`, `test_removed_kinds_are_not_registered`,
 * `test_kinds_and_binding_fields_agree`.
 *
 * TS-only: every kind's class in `KINDS`; the `case`, `call`, `map` and `loop` binding fields;
 * `withKinds` restores the tables after its function returns or throws.
 *
 * Dropped: `test_retired_kindspec_is_not_public` (a Python import).
 */

import { expect, test } from "bun:test";

import { AgentDescriptor } from "../../src/compose/nodes/agent.ts";
import { CallDescriptor } from "../../src/compose/nodes/call.ts";
import { CaseDescriptor } from "../../src/compose/nodes/case.ts";
import { CodeDescriptor } from "../../src/compose/nodes/code.ts";
import { HumanInputDescriptor } from "../../src/compose/nodes/human_input.ts";
import { BINDING_FIELDS, KINDS, REMOVED_KINDS } from "../../src/compose/nodes/index.ts";
import { LoopDescriptor } from "../../src/compose/nodes/loop.ts";
import { MapDescriptor } from "../../src/compose/nodes/map.ts";
import { ToolDescriptor } from "../../src/compose/nodes/tool.ts";
import { WaitDescriptor } from "../../src/compose/nodes/wait.ts";
import { StandInCode, standIn, withKinds } from "./stand_ins.ts";

// The author-written kinds (as `docs/syntax.md` lists them; the loader makes `start` and `end`).
const AUTHORED_KINDS: readonly string[] = [
  "agent",
  "call",
  "case",
  "code",
  "human_input",
  "loop",
  "map",
  "tool",
  "wait",
];

/** Expect `KINDS` and `BINDING_FIELDS` to name the same kinds, each with its class's fields. */
function expectTablesAgree(): void {
  expect([...KINDS.keys()]).toEqual([...BINDING_FIELDS.keys()]);
  for (const [kind, cls] of KINDS) {
    expect(cls.kind).toBe(kind);
    expect(BINDING_FIELDS.get(kind)).toBe(cls.bindingFields);
  }
}

test("the kind tables are maps", () => {
  expect(KINDS).toBeInstanceOf(Map);
  expect(BINDING_FIELDS).toBeInstanceOf(Map);
});

test("KINDS maps the kind text to its descriptor class", () => {
  expect(KINDS.get("agent")).toBe(AgentDescriptor);
  expect(KINDS.get("code")).toBe(CodeDescriptor);
  expect(KINDS.get("tool")).toBe(ToolDescriptor);
  expect(KINDS.get("human_input")).toBe(HumanInputDescriptor);
  expect(KINDS.get("wait")).toBe(WaitDescriptor);
  expect(KINDS.get("case")).toBe(CaseDescriptor);
  expect(KINDS.get("call")).toBe(CallDescriptor);
  expect(KINDS.get("map")).toBe(MapDescriptor);
  expect(KINDS.get("loop")).toBe(LoopDescriptor);
  for (const [kind, cls] of KINDS) expect(cls.kind).toBe(kind);
});

test("all authored kinds are registered", () => {
  expect(AUTHORED_KINDS.filter((kind) => !KINDS.has(kind))).toEqual([]);
  expect(AUTHORED_KINDS.filter((kind) => !BINDING_FIELDS.has(kind))).toEqual([]);
});

test("a case carries no binding fields", () => {
  expect(BINDING_FIELDS.get("case")).toEqual([]);
});

test("a call and a map bind their inputs and over", () => {
  expect(BINDING_FIELDS.get("call")).toEqual(["inputs", "over"]);
  expect(BINDING_FIELDS.get("map")).toEqual(["inputs", "over"]);
});

test("a loop binds its inputs", () => {
  expect(BINDING_FIELDS.get("loop")).toEqual(["inputs"]);
});

test("removed kinds are not registered", () => {
  expect([...REMOVED_KINDS.keys()].filter((kind) => KINDS.has(kind))).toEqual([]);
  expect(REMOVED_KINDS.get("model")).toBe(
    "serve the model behind a host tool and call it with kind: tool (set tool_id: and input:)",
  );
});

test("KINDS and BINDING_FIELDS agree", () => {
  expectTablesAgree();
});

test("withKinds restores the tables after a return or a throw", () => {
  const kinds = [...KINDS];
  const bindingFields = [...BINDING_FIELDS];
  const first = standIn("code");
  withKinds([first], () => {
    withKinds([StandInCode], () => {
      expect(KINDS.get("code")).toBe(StandInCode);
    });
    expect(KINDS.get("code")).toBe(first);
    expect(BINDING_FIELDS.get("code")).toBe(first.bindingFields);
    expect(() =>
      withKinds([StandInCode], () => {
        throw new Error("boom");
      }),
    ).toThrow("boom");
    expect(KINDS.get("code")).toBe(first);
  });
  expect([...KINDS]).toEqual(kinds);
  expect([...BINDING_FIELDS]).toEqual(bindingFields);
});
