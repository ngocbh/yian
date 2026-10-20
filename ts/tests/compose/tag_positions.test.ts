/**
 * Tags on a CASE: the `case` rows of `tests/compose/test_tag_positions.py`, loading each flow with
 * `assembleText` where Python calls `load_flow`.
 *
 * A tag on a CASE branch makes the branch a value, even a word naming a node, so its error shows
 * the value entry to write; a tagged case `output:` is rejected at the `output:`, like an untagged
 * one.
 *
 * Ported: `test_a_tagged_case_branch_naming_a_node_is_a_value`,
 * `test_tagged_case_output_is_rejected_at_its_line`, each pinning the whole message where Python
 * matches its end; and their harvested `load-types/compose_tag_positions__*` cases, pinned at the
 * end.
 *
 * Later (T3-22, the loader, whose tag checks and whole-file load these rows exercise; the
 * harvested cases of those checks are pinned in `tags.test.ts`): `test_tags_in_typed_slots_load`,
 * `test_tag_on_kind_is_rejected_at_its_line`, `test_tag_on_prompt_is_rejected_at_its_line`,
 * `test_tag_on_nodes_is_rejected_at_its_line`, `test_tag_on_map_over_is_rejected_at_its_line`,
 * `test_tag_on_wait_until_is_rejected_at_its_line`,
 * `test_tag_inside_a_non_slot_value_names_its_key`,
 * `test_tagged_mapping_key_gets_its_own_message`,
 * `test_tagged_key_message_spells_the_key_as_written`,
 * `test_generic_message_spells_the_key_as_written`,
 * `test_a_null_key_is_reported_before_a_tag_under_it`,
 * `test_tagged_typedef_is_rejected_at_its_line`, `test_tagged_typedef_field_is_rejected_at_its_line`,
 * `test_tagged_typedef_field_without_a_value_is_rejected_at_its_line`,
 * `test_tagged_node_output_declaration_is_rejected_at_its_line`,
 * `test_tagged_node_output_with_a_type_after_the_tag_spells_the_type`,
 * `test_tagged_node_output_field_is_rejected_at_its_line`,
 * `test_tagged_list_item_in_a_node_output_says_the_item_is_tagged`,
 * `test_tagged_compact_output_is_rejected_at_its_line`,
 * `test_tagged_agent_and_tool_output_is_rejected_at_its_line`,
 * `test_tagged_single_node_def_output_is_rejected_at_its_line`, `test_tags_in_a_def_slot_load`,
 * `test_tag_outside_a_slot_in_a_def_is_rejected_at_its_line`,
 * `test_tags_in_an_inline_body_slot_load`,
 * `test_tag_outside_a_slot_in_an_inline_body_is_rejected_at_its_line`,
 * `test_tags_in_a_uses_child_slot_load`, `test_tag_outside_a_slot_in_a_uses_child_is_rejected`,
 * `test_tag_under_an_extension_key_used_through_an_alias_in_a_slot_loads`,
 * `test_nested_tag_under_an_extension_key_used_through_a_merge_loads`,
 * `test_unused_tag_under_an_extension_key_loads`,
 * `test_tag_under_an_extension_key_aliased_outside_a_slot_is_rejected`,
 * `test_loading_the_same_document_twice_behaves_the_same`,
 * `test_tag_on_a_binding_value_loads_and_is_consumed`, `test_message_names_the_typed_slots`.
 */

import { afterEach, expect, test } from "bun:test";

import { forget } from "../../src/defaults.ts";
import { assembled, assembleText, type CaseRow, loadError, runCase } from "./declarations.ts";

afterEach(() => forget());

/** Load `text`, expecting a load error with the whole `message` at the line equal to `line`. */
function expectError(text: string, message: string, line: string): void {
  const error = loadError(() => assembleText(text));
  expect(error.message).toBe(message);
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(text.split("\n").indexOf(line) + 1);
}

test("a tagged case branch naming a node is a value", () => {
  const text = `id: f
name: f
input:
  score: !str
nodes:
  route:
    kind: case
    on: \${input.score}
    cases:
      - when: high
        then: !str hot
    else: cold
  hot:
    kind: code
    code: 'return "hot"'
    output: str
  cold:
    kind: code
    code: 'return "cold"'
    output: str
output: \${hot.output | cold.output}
`;
  // The tag makes the branch the text `hot`, as in a value entry, not the node.
  expectError(
    text,
    "case node 'route' then: a branch names a node or calls a flow; put the value " +
      "in its own entry in `nodes:` and name it: `route_then_0: !str hot` and " +
      "`then: route_then_0`",
    "        then: !str hot",
  );
});

test("a tagged case output is rejected at its line", () => {
  const text = `id: f
name: f
input:
  score: !str
nodes:
  route:
    kind: case
    on: \${input.score}
    cases:
      - when: high
        then: one
    else: one
    output: !int
  one:
    kind: code
    code: |
      return 1
    output: int
output: \${route.output}
`;
  expectError(
    text,
    "case node 'route': a `case` routes and has no `output:`; tag its result where " +
      "you read it, e.g. `x: !int ${route.output}`",
    "    output: !int",
  );
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  { id: "load-types/compose_tag_positions__a_tagged_case_branch_naming_a_node_is_a_value", call: assembled },
  { id: "load-types/compose_tag_positions__tagged_case_output_is_rejected_at_its_line", call: assembled },
])("assemble: $id", runCase);
