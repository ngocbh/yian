/**
 * The CODE node's whole-code and exact-line origins: a port of
 * `tests/compose/test_code_origins.py`.
 *
 * Ported: `test_source_free_code_accepts_empty_origin_carriers` and
 * `test_code_origins_participate_in_local_traversal_and_rewrite`.
 *
 * Ported (adapted): `test_code_origin_carriers_reject_invalid_shapes`, its node rows
 * `whole-not-origin` and `line-item-not-origin`.
 *
 * Changed: the line origins are an array, not a tuple; a rewrite assigns a fresh array.
 *
 * TS-only: the node keeps a frozen copy of the line origins it is given, and a rewrite assigns a
 * fresh frozen array; a line-origins value that isn't an array is rejected.
 *
 * Not ported: the `lines-not-tuple` row of `test_code_origin_carriers_reject_invalid_shapes` (TS
 * line origins are arrays only).
 *
 * Later (T3-15): the descriptor rows of `test_code_origin_carriers_reject_invalid_shapes`,
 * `test_source_free_attachment_preserves_explicit_code_origins` and
 * `test_changed_projection_keeps_explicit_lines_without_claiming_exact_yaml_lines`.
 *
 * Later (T3-15, T3-22): `test_literal_code_carries_whole_scalar_and_exact_constructed_lines`,
 * `test_compact_root_code_origins_retain_the_projected_authored_scalar`,
 * `test_nonliteral_code_carries_only_whole_scalar_fallback`,
 * `test_multi_node_def_code_origins_use_the_definition_lexical_path`,
 * `test_flat_def_code_origins_retain_the_projected_authored_scalar`,
 * `test_inline_child_code_origins_include_the_enclosing_node_path`,
 * `test_external_alias_rebases_every_code_origin_without_mutating_child`,
 * `test_generated_child_clone_preserves_code_origins_for_original_resolver` and
 * `test_formatting_moves_resolved_ranges_without_changing_semantic_keys`.
 *
 * Later (T3-21): `test_code_origins_are_noncanonical_manifest_and_fingerprint_metadata`.
 *
 * Later (T4-9): `test_runtime_failure_diagnoses_exact_line_across_composition_paths`.
 */

import { expect, test } from "bun:test";

import { OriginRef } from "../../src/diagnostics.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";

test("source-free code accepts empty origin carriers", () => {
  const node = new CodeNode("worker", { code: "return 1", codeOrigin: null, codeLineOrigins: [] });

  expect(node.codeOrigin).toBeNull();
  expect(node.codeLineOrigins).toEqual([]);
  expect([...node.iterLocalOrigins()]).toEqual([]);
});

test("code origins participate in local traversal and rewrite", () => {
  const nodeOrigin = new OriginRef("node");
  const codeOrigin = new OriginRef("code");
  const lineOrigins = [new OriginRef("line/0"), new OriginRef("line/1")];
  const node = new CodeNode("worker", {
    code: "const value = 1\nreturn value",
    codeOrigin,
    codeLineOrigins: lineOrigins,
  });
  node.origin = nodeOrigin;

  expect([...node.iterLocalOrigins()]).toEqual([nodeOrigin, codeOrigin, ...lineOrigins]);

  node.rewriteLocalOrigins((origin) => new OriginRef(`rewritten/${origin.key}`));

  expect([...node.iterLocalOrigins()]).toEqual([
    new OriginRef("rewritten/node"),
    new OriginRef("rewritten/code"),
    new OriginRef("rewritten/line/0"),
    new OriginRef("rewritten/line/1"),
  ]);
  expect(node.codeOrigin).toEqual(new OriginRef("rewritten/code"));
  expect(node.codeLineOrigins).toEqual([new OriginRef("rewritten/line/0"), new OriginRef("rewritten/line/1")]);
});

test.each([
  ["whole-not-origin", { codeOrigin: "not-an-origin" }, "CODE source origin must be an OriginRef or null"],
  [
    "line-item-not-origin",
    { codeLineOrigins: [new OriginRef("line/0"), "not-an-origin"] },
    "CODE line origins must contain only OriginRef values",
  ],
])("code origin carriers reject invalid shapes: node, %s", (_id, origins, message) => {
  expect(() => new CodeNode("worker", { code: "return 1", ...(origins as object) })).toThrow(new TypeError(message));
});

// --- TS-only rows --- //

test("a line-origins value that isn't an array is rejected", () => {
  const origins = new OriginRef("line/0") as unknown as OriginRef[];

  expect(() => new CodeNode("worker", { code: "return 1", codeLineOrigins: origins })).toThrow(
    new TypeError("CODE line origins must be an array"),
  );
});

test("the line origins are a frozen copy, and a rewrite assigns a fresh frozen array", () => {
  const given = [new OriginRef("line/0")];
  const node = new CodeNode("worker", { code: "return 1", codeLineOrigins: given });
  given.push(new OriginRef("line/1"));

  expect(node.codeLineOrigins).toEqual([new OriginRef("line/0")]);
  expect(Object.isFrozen(node.codeLineOrigins)).toBe(true);

  const before = node.codeLineOrigins;
  node.rewriteLocalOrigins((origin) => origin);

  expect(node.codeLineOrigins).not.toBe(before);
  expect(Object.isFrozen(node.codeLineOrigins)).toBe(true);
  expect(before).toEqual([new OriginRef("line/0")]);
});
