/**
 * The kind dispatch tables `KINDS` and `BINDING_FIELDS`, and `REMOVED_KINDS`: a port of
 * `tests/compose/test_kinds_table.py`. TS-era: only some kinds are ported, so the tables hold
 * kinds of `ALL_KIND_NAMES`, and the agreement row also runs on the nine stand-ins registered
 * with `withKinds`.
 *
 * Ported: `test_kinds_maps_string_to_descriptor_class`.
 *
 * TS-only: the `code`, `human_input` and `wait` rows; `withKinds` restores the tables after its
 * function returns or throws.
 *
 * Dropped: `test_retired_kindspec_is_not_public` (a Python import).
 *
 * Later (T3-16): `test_all_authored_kinds_registered`.
 */

import { expect, test } from "bun:test";

import { AgentDescriptor } from "../../src/compose/nodes/agent.ts";
import { CodeDescriptor } from "../../src/compose/nodes/code.ts";
import { HumanInputDescriptor } from "../../src/compose/nodes/human_input.ts";
import { BINDING_FIELDS, KINDS, REMOVED_KINDS } from "../../src/compose/nodes/index.ts";
import { ToolDescriptor } from "../../src/compose/nodes/tool.ts";
import { WaitDescriptor } from "../../src/compose/nodes/wait.ts";
import { ALL_KIND_NAMES, StandInCode, standIn, withKinds } from "./stand_ins.ts";

/** Expect `KINDS` and `BINDING_FIELDS` to name the same kinds, each with its class's fields. */
function expectTablesAgree(): void {
  expect([...KINDS.keys()]).toEqual([...BINDING_FIELDS.keys()]);
  for (const [kind, cls] of KINDS) {
    expect(cls.kind).toBe(kind);
    expect(BINDING_FIELDS.get(kind)).toBe(cls.bindingFields);
  }
}

test("the kind tables are maps of authored kinds", () => {
  expect(KINDS).toBeInstanceOf(Map);
  expect(BINDING_FIELDS).toBeInstanceOf(Map);
  expect([...KINDS.keys()].filter((kind) => !ALL_KIND_NAMES.includes(kind))).toEqual([]);
});

test("KINDS maps the kind text to its descriptor class", () => {
  expect(KINDS.get("agent")).toBe(AgentDescriptor);
  expect(KINDS.get("code")).toBe(CodeDescriptor);
  expect(KINDS.get("tool")).toBe(ToolDescriptor);
  expect(KINDS.get("human_input")).toBe(HumanInputDescriptor);
  expect(KINDS.get("wait")).toBe(WaitDescriptor);
  for (const [kind, cls] of KINDS) expect(cls.kind).toBe(kind);
});

test("removed kinds are not registered", () => {
  expect([...REMOVED_KINDS.keys()].filter((kind) => KINDS.has(kind))).toEqual([]);
  expect(REMOVED_KINDS.get("model")).toBe(
    "serve the model behind a host tool and call it with kind: tool (set tool_id: and input:)",
  );
});

test("KINDS and BINDING_FIELDS agree", () => {
  expectTablesAgree();
  const classes = ALL_KIND_NAMES.map((kind) => standIn(kind, { bindingFields: ["inputs"] }));
  withKinds(classes, () => {
    expect(KINDS.size).toBe(9);
    expectTablesAgree();
  });
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
