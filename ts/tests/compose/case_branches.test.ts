/**
 * A CASE routes only: a branch names a node or calls a flow, and a CASE has no `output:`. A port
 * of `tests/compose/test_case_branches.py`, loading each flow with `assembleText` where Python
 * calls `load_flow`.
 *
 * An untagged unquoted word names a node and an unquoted whole-value `call(...)` calls a flow.
 * Every other branch is a value: a tagged value (even a word naming a node), quoted text, `null`
 * or an empty `else:`, a number, a bool, a mapping, a list, or unquoted text holding `${`. A value
 * branch is a located load error that shows the value entry to write and the branch naming it,
 * and each suggested entry loads when applied. The entry keeps the branch's own tag, else takes
 * the CASE's former `output:` type when a tag spells it and the value fits it (a lone reference
 * fits when its type does), else the type of a lone reference or a tag guessed from the YAML
 * value, else a `!<type>` placeholder. A value no entry can hold (a value containing itself, a
 * written-out record with an empty key, a nested `call(...)`, a flow call or malformed span in
 * `${...}`, a span reading the case) is its own located error. An unquoted word that names no
 * node suggests the nearest node and the value entry for the text, which loads too. A CASE
 * `output:`, tagged or not, is a load error at the `output:` that says to tag the result where it
 * is read.
 *
 * Ported: every row's load half. A row that applies the suggested rewrite and runs it checks here
 * that the rewritten flow assembles: `test_a_then_value_shows_its_value_entry_and_the_entry_loads`,
 * `test_an_else_value_shows_its_value_entry_and_the_entry_loads`,
 * `test_a_suggested_entry_id_skips_ids_already_taken`,
 * `test_the_old_typed_text_is_tagged_with_its_type`, `test_a_lone_reference_is_tagged_with_its_type`,
 * `test_a_lone_reference_takes_the_former_output_type_only_when_it_fits`,
 * `test_a_lone_span_of_unknown_type_asks_for_its_type`,
 * `test_a_list_of_mixed_items_is_tagged_list_of_any`,
 * `test_a_list_of_mixed_items_with_a_date_is_shown_with_the_date_quoted`,
 * `test_a_tagged_value_keeps_its_tag`, `test_a_value_branch_is_tagged_with_the_former_output_type`,
 * `test_a_value_branch_is_spelled_to_fit_the_former_output_type`,
 * `test_a_value_that_does_not_fit_the_former_output_type_guesses_its_tag`,
 * `test_a_value_branch_is_tagged_with_the_former_output_typedef`,
 * `test_a_value_branch_under_an_unspellable_output_guesses_its_tag`,
 * `test_an_unknown_then_word_suggests_the_nearest_node_and_its_text_entry`,
 * `test_an_unknown_else_word_suggests_the_nearest_node_and_its_text_entry`,
 * `test_an_unknown_sentence_gets_no_suggestion_and_an_entry_named_after_the_branch`,
 * `test_an_unknown_yes_gets_the_same_text_entry`,
 * `test_an_unknown_word_entry_id_skips_an_id_already_taken`,
 * `test_unquoted_call_that_is_not_one_directive_is_rejected`, and the load of
 * `test_former_reserved_prefixes_name_authored_nodes`. The error-only rows:
 * `test_a_value_in_a_later_then_names_its_index`, `test_a_long_mapping_is_cut_and_written_as_a_block`,
 * `test_a_value_that_contains_itself_is_rejected`, `test_a_number_that_is_not_finite_is_rejected`,
 * `test_a_mapping_with_an_empty_key_is_rejected`, `test_a_quoted_call_holding_a_span_is_its_own_error`,
 * `test_a_call_nested_in_a_value_is_rejected`, `test_a_value_reading_its_own_case_is_rejected`,
 * `test_a_malformed_span_in_a_value_is_rejected`,
 * `test_a_mapping_value_branch_gets_the_branch_error_under_any_key`,
 * `test_a_case_output_is_rejected_at_its_line`.
 *
 * Changed: an unquoted date is text (rule `dates`). So `then: 2026-01-02` is a word that names no
 * node and gets the text entry `route_then_0: !str 2026-01-02` (Python: the value entry
 * `!date "2026-01-02"`); a date in a mapping or list is shown plain and typed as text
 * (`!object {a: 2026-01-02}`, `!list(str) [2026-01-02]`, `!list(Any) [1, 2024-01-01]`; Python
 * quotes it and types the list `list(date)`); and quoted date text in a branch is shown plain,
 * since plain YAML reads it back unchanged (`!date 2026-01-02` for the tagged `!date
 * "2026-01-02"`, `!list(Any) [1, 2024-01-01]` for `[1, "2024-01-01"]`).
 * `test_a_number_that_is_not_finite_is_rejected`: infinity is a float value (rule `infinity`), so
 * `.inf`, `-.inf`, `!float .inf`, `{a: .inf}` and `[1.5, -.inf]` are value branches with the usual
 * rewrite, and the YAML reader refuses `.nan` at the scalar before any branch is read.
 * A float in a flow-style branch value is spelled positionally (`10000000000000000.0`), where
 * Python's `repr` uses an exponent from 1e16 and below 1e-4; no Python row pins one.
 *
 * TS-only: a cut value counts code points, so an astral character at the cut stays whole; and the
 * harvested `load-case/compose_case_branches__*` cases.
 *
 * Later (T4): the run of every rewritten flow above (Python's `_taken`),
 * `test_case_output_over_node_branches_reads_the_taken_branch`,
 * `test_case_output_over_a_value_entry_branch_reads_the_taken_branch`, and the run of
 * `test_former_reserved_prefixes_name_authored_nodes`.
 */

import { afterEach, expect, test } from "bun:test";

import type { LoadError } from "../../src/compose/helpers/errors.ts";
import { forget } from "../../src/defaults.ts";
import { assembled, assembleText, type CaseRow, loadError, runCase } from "./declarations.ts";

afterEach(() => forget());

const INPUTS = "  cmd: !str\n";
const INPUTS_N = "  cmd: !str\n  n: !int\n";
// The value entry and branch a value-branch message suggests.
const REWRITE = /name it: `(?<entry>[^`]+)`(?: \(.*\))? and `(?<label>then|else): (?<id>\w+)`$/;
// The text entry a branch that names no node, or is broken call text, suggests.
const TEXT_ENTRY = /^case node 'route' (?<label>then|else): .* name it: `(?<entry>(?<id>\w+): [^`]+)`$/;

/** Return the 1-based line number of the first line containing `needle`. */
function lineOf(text: string, needle: string): number {
  return text.split("\n").findIndex((line) => line.includes(needle)) + 1;
}

/** Load `text`, expecting a load error whose whole message is `message`. */
function loadErrorWith(text: string, message: string): LoadError {
  const error = loadError(() => assembleText(text));
  expect(error.message).toBe(message);
  return error;
}

/** Return the start line of the innermost frame of a load error. */
function errorLine(error: LoadError): number {
  return error.diagnostic.frames.at(-1)!.range.startLine;
}

/** Return a flow with a `route` case (indented body) and a `reply` node it reads. */
function flow(
  caseBody: string,
  options: { head?: string; inputs?: string; extraNodes?: string } = {},
): string {
  const { head = "", inputs = INPUTS, extraNodes = "" } = options;
  return (
    "id: f\n" +
    "name: f\n" +
    head +
    "input:\n" +
    inputs +
    "nodes:\n" +
    "  route:\n" +
    "    kind: case\n" +
    caseBody +
    "  reply:\n" +
    "    kind: code\n" +
    "    code: tests.engine._compose_codefns:echo\n" +
    "    input:\n" +
    "      topic: replied\n" +
    "    output: str\n" +
    extraNodes +
    "output: ${route.output}\n"
  );
}

/** Return an `on: ${input.cmd}` case body with one `when: a` branch and an `else:`. */
function on(then: string, else_ = "reply", options: { output?: string } = {}): string {
  return (
    "    on: ${input.cmd}\n" +
    (options.output ?? "") +
    "    cases:\n" +
    "      - when: a\n" +
    `        then: ${then}\n` +
    `    else: ${else_}\n`
  );
}

/**
 * Return the `on` flow with the value entry and branch an error suggests applied.
 *
 * The CASE keeps `when: a` and `reply` for its other branch, and has no `output:`. A text entry's
 * branch names the entry. `typeText` fills a `!<type>` placeholder.
 */
function rewritten(
  error: LoadError,
  options: { head?: string; inputs?: string; extraNodes?: string; typeText?: string } = {},
): string {
  const { head = "", inputs = INPUTS, extraNodes = "", typeText } = options;
  const found = REWRITE.exec(error.message) ?? TEXT_ENTRY.exec(error.message);
  expect(found).not.toBeNull();
  const groups = found!.groups!;
  let entry = groups.entry!;
  if (typeText !== undefined) entry = entry.replace("!<type>", `!${typeText}`);
  const caseBody = groups.label === "then" ? on(groups.id!) : on("reply", groups.id!);
  return flow(caseBody, { head, inputs, extraNodes: `${extraNodes}  ${entry}\n` });
}

/** Assemble `text`, which must load. */
function assembles(text: string): void {
  assembleText(text);
}

/** Return the message for a value branch of `route`. */
function valueBranch(label: string, entry: string, shown: string, note = ""): string {
  return (
    `case node 'route' ${label}: a branch names a node or calls a flow; put the value ` +
    `in its own entry in \`nodes:\` and name it: \`${entry}: ${shown}\`${note} and ` +
    `\`${label}: ${entry}\``
  );
}

// ---------- node and call branches ----------

test.each(["__value_0", "__case_x"])("former reserved prefixes name authored nodes: %s", (nodeId) => {
  const text = flow(on(nodeId), {
    extraNodes: `  ${nodeId}:\n` + "    kind: code\n" + "    code: |\n" + "      return 'own'\n" + "    output: str\n",
  });

  assembles(text);
});

// ---------- value branches ----------

test.each([
  ['"approve"', "!str approve", 'then: "approve"'],
  ["'approve'", "!str approve", "then: 'approve'"],
  ['"reply"', "!str reply", 'then: "reply"'],
  ['"it\'s: here"', '!str "it\'s: here"', "then: \"it's"],
  ['"[x"', '!str "[x"', 'then: "[x"'],
  ["|\n          two\n          lines", '!str "two\\nlines\\n"', "then: |"],
  [">\n          one\n          line", '!str "one line\\n"', "then: >"],
  ["3", "!int 3", "then: 3"],
  ["1e3", "!float 1e3", "then: 1e3"],
  ["true", "!bool true", "then: true"],
  ["null", "!Optional(str) null", "then: null"],
  [
    "\n          reply: ${input.cmd}\n          exited: true",
    '!object {reply: "${input.cmd}", exited: true}',
    "reply: ${input.cmd}",
  ],
  ["{a: 1, b: [2.5]}", "!object {a: 1, b: [2.5]}", "then: {a: 1"],
  // Changed: the unquoted date is text, shown plain.
  ["{a: 2026-01-02}", "!object {a: 2026-01-02}", "then: {a: 2026"],
  ['[{"": x}]', '!list(object) [{"": x}]', 'then: [{"": x}]'],
  ['[a, "${input.cmd}"]', '!list(str) [a, "${input.cmd}"]', "then: [a"],
  // Changed: the unquoted date is text, so the list is a `list(str)`.
  ["[2026-01-02]", "!list(str) [2026-01-02]", "then: [2026"],
  ["[[1, 2], [3]]", "!list(list(int)) [[1, 2], [3]]", "then: [[1"],
  ["[null, 1]", "!list(Optional(int)) [null, 1]", "then: [null"],
  ["[[1], [null]]", "!list(list(Optional(int))) [[1], [null]]", "then: [[1]"],
  ["[[], [1]]", "!list(list(int)) [[], [1]]", "then: [[]"],
  ["[1, 2.5]", "!list(float) [1, 2.5]", "then: [1, 2.5]"],
  ["[{a: 1}, {b: 2}]", "!list(object) [{a: 1}, {b: 2}]", "then: [{a"],
  ["hi ${input.cmd}", "!str hi ${input.cmd}", "then: hi"],
  ["${input.cmd}", "!str ${input.cmd}", "then: ${input.cmd}"],
  ['"call(take"', '!str "call(take"', 'then: "call(take"'],
])("a then value shows its value entry and the entry loads: %j", (then, shown, needle) => {
  const text = flow(on(then));

  const error = loadErrorWith(text, valueBranch("then", "route_then_0", shown));

  expect(errorLine(error)).toBe(lineOf(text, needle));
  assembles(rewritten(error));
});

test("an unquoted date then is a word that names no node, and its text entry loads", () => {
  // Changed: Python reads the date as a date value and shows `!date "2026-01-02"`.
  const text = flow(on("2026-01-02"));

  const error = loadErrorWith(
    text,
    "case node 'route' then: `2026-01-02` names no node. For the text, add a value entry and " +
      "name it: `route_then_0: !str 2026-01-02`",
  );

  expect(errorLine(error)).toBe(lineOf(text, "then: 2026-01-02"));
  assembles(rewritten(error));
});

test.each([
  ['"declined"', "!str declined", 'else: "declined"'],
  ["null", "!Optional(str) null", "else: null"],
  ["", "!Optional(str) null", "else:"],
  ["1.5", "!float 1.5", "else: 1.5"],
])("an else value shows its value entry and the entry loads: %j", (else_, shown, needle) => {
  const text = flow(on("reply", else_));

  const error = loadErrorWith(text, valueBranch("else", "route_else", shown));

  expect(errorLine(error)).toBe(lineOf(text, needle));
  assembles(rewritten(error));
});

test("a value in a later then names its index", () => {
  const text = flow(
    "    on: ${input.cmd}\n" +
      "    cases:\n" +
      "      - when: a\n" +
      "        then: reply\n" +
      "      - when: b\n" +
      '        then: "bee"\n' +
      "    else: reply\n",
  );

  const error = loadErrorWith(text, valueBranch("then", "route_then_1", "!str bee"));

  expect(errorLine(error)).toBe(lineOf(text, 'then: "bee"'));
});

test("a long mapping is cut and written as a block", () => {
  const text = flow(
    on(
      "\n          reply: ${input.cmd}\n          exited: true\n" +
        "          note: a much longer note that goes on and on",
    ),
  );

  const error = loadErrorWith(
    text,
    valueBranch(
      "then",
      "route_then_0",
      '!object {reply: "${input.cmd}", exited: true, note: a much longer no…',
      " (write it as a block)",
    ),
  );

  expect(errorLine(error)).toBe(lineOf(text, "reply: ${input.cmd}"));
});

test("a cut value keeps an astral character at the cut whole", () => {
  const word = "x".repeat(59) + "😀" + "yz";
  const text = flow(on(`"${word}"`));

  const error = loadErrorWith(
    text,
    valueBranch("then", "route_then_0", `!str ${"x".repeat(59)}😀…`, " (write it as a block)"),
  );

  expect(errorLine(error)).toBe(lineOf(text, "then: "));
});

test("a suggested entry id skips ids already taken", () => {
  const taken = "  route_then_0: !str one\n" + "  route_then_0_2: !str two\n";
  const text = flow(on('"approve"'), { extraNodes: taken });

  const error = loadErrorWith(text, valueBranch("then", "route_then_0_3", "!str approve"));

  expect(errorLine(error)).toBe(lineOf(text, 'then: "approve"'));
  assembles(rewritten(error, { extraNodes: taken }));
});

test("the old typed text is tagged with its type", () => {
  const text = flow(on("int = ${input.n}"), { inputs: INPUTS_N });

  const error = loadErrorWith(text, valueBranch("then", "route_then_0", "!int ${input.n}"));

  expect(errorLine(error)).toBe(lineOf(text, "then: int ="));
  assembles(rewritten(error, { inputs: INPUTS_N }));
});

test.each([
  ["${input.n}", "!int ${input.n}"],
  ["${count.output}", "!int ${count.output}"],
])("a lone reference is tagged with its type: %s", (then, shown) => {
  const count = "  count: !int 7\n";
  const text = flow(on(then), { inputs: INPUTS_N, extraNodes: count });

  const error = loadErrorWith(text, valueBranch("then", "route_then_0", shown));

  expect(errorLine(error)).toBe(lineOf(text, `then: ${then}`));
  assembles(rewritten(error, { inputs: INPUTS_N, extraNodes: count }));
});

test.each([
  ["${input.n}", "Optional(int)", "!Optional(int) ${input.n}"],
  ["${input.cmd}", "int", "!str ${input.cmd}"],
  ["${input.n}", "str", "!int ${input.n}"],
  ["${count.output}", "str", "!int ${count.output}"],
])("a lone reference takes the former output type only when it fits: %s under %s", (then, output, shown) => {
  const count = "  count: !int 7\n";
  const text = flow(on(then, "reply", { output: `    output: ${output}\n` }), {
    inputs: INPUTS_N,
    extraNodes: count,
  });

  const error = loadErrorWith(text, valueBranch("then", "route_then_0", shown));

  expect(errorLine(error)).toBe(lineOf(text, `then: ${then}`));
  assembles(rewritten(error, { inputs: INPUTS_N, extraNodes: count }));
});

test("a lone span of unknown type asks for its type", () => {
  const text = flow(on("${input.n + 1}"), { inputs: INPUTS_N });

  const error = loadErrorWith(
    text,
    valueBranch("then", "route_then_0", "!<type> ${input.n + 1}", " (name its type for `<type>`)"),
  );

  expect(errorLine(error)).toBe(lineOf(text, "then: ${input.n + 1}"));
  assembles(rewritten(error, { inputs: INPUTS_N, typeText: "int" }));
});

test.each([
  ["[1, a]", "!list(Any) [1, a]"],
  ["[[1], [a]]", "!list(Any) [[1], [a]]"],
  ["[{a: 1}, 2]", "!list(Any) [{a: 1}, 2]"],
  ["[null, 1, a]", "!list(Optional(Any)) [null, 1, a]"],
  // Changed: plain YAML reads the date back as the same text, so it is shown unquoted.
  ['[1, "2024-01-01"]', "!list(Any) [1, 2024-01-01]"],
])("a list of mixed items is tagged list of Any: %s", (then, shown) => {
  const text = flow(on(then));

  const error = loadErrorWith(text, valueBranch("then", "route_then_0", shown));

  expect(errorLine(error)).toBe(lineOf(text, `then: ${then}`));
  assembles(rewritten(error));
});

test.each(["[1, 2024-01-01]", "[{a: 2024-01-01}, 2]"])(
  "a list of mixed items with a date is shown with the date as written: %s",
  (then) => {
    // Changed: the unquoted date is text, so it reads back plain (Python quotes it).
    const text = flow(on(then));

    const error = loadErrorWith(text, valueBranch("then", "route_then_0", `!list(Any) ${then}`));

    expect(errorLine(error)).toBe(lineOf(text, `then: ${then}`));
    assembles(rewritten(error));
  },
);

test.each(["&R [*R]", "&R [1, *R]", "&R {a: *R}"])("a value that contains itself is rejected: %s", (then) => {
  const text = flow(on(then));

  const error = loadErrorWith(text, "case node 'route' then: a branch value can't contain itself");

  expect(errorLine(error)).toBe(lineOf(text, `then: ${then}`));
});

test.each([
  [".inf", "!float .inf"],
  ["-.inf", "!float -.inf"],
  ["{a: .inf}", "!object {a: .inf}"],
  ["[1.5, -.inf]", "!list(float) [1.5, -.inf]"],
  ["!float .inf", "!float .inf"],
])("an infinite number is a value branch: %s", (then, shown) => {
  // Changed: Python rejects it as not finite.
  const text = flow(on(then));

  const error = loadErrorWith(text, valueBranch("then", "route_then_0", shown));

  expect(errorLine(error)).toBe(lineOf(text, `then: ${then}`));
  assembles(rewritten(error));
});

test("a NaN branch is refused by the YAML reader at the scalar", () => {
  // Changed: Python reads it and rejects it as not finite.
  const text = flow(on(".nan"));

  const error = loadErrorWith(text, "invalid YAML: NaN is not a yian value; quote it to keep it as text");

  expect(errorLine(error)).toBe(lineOf(text, "then: .nan"));
});

test.each([
  ["!int 3", "!int 3"],
  ["!float 1e3", "!float 1e3"],
  ["!str approve", "!str approve"],
  // Changed: plain YAML reads `2026-01-02` back as the same text, so it is shown unquoted.
  ['!date "2026-01-02"', "!date 2026-01-02"],
  ["!int ${input.n}", "!int ${input.n}"],
  ["!Optional(int) null", "!Optional(int) null"],
  ["!Buy {side: buy}", "!Buy {side: buy}"],
  ["!list(int)\n          - 1\n          - 2", "!list(int) [1, 2]"],
  ["!str reply", "!str reply"],
])("a tagged value keeps its tag: %j", (then, shown) => {
  const head = "typedefs:\n  Buy:\n    side: str\n";
  // The branch's own tag wins over the former `output:` type, and a tagged word that names a node
  // is text, as in a value entry.
  const text = flow(on(then, "reply", { output: "    output: str\n" }), { head, inputs: INPUTS_N });

  const error = loadErrorWith(text, valueBranch("then", "route_then_0", shown));

  expect(errorLine(error)).toBe(lineOf(text, `then: ${then.split("\n")[0]}`));
  assembles(rewritten(error, { head, inputs: INPUTS_N }));
});

test("a value branch is tagged with the former output type", () => {
  const text = flow(on("3", "reply", { output: "    output: str\n" }));

  const error = loadErrorWith(text, valueBranch("then", "route_then_0", '!str "3"'));

  // The value branch is reported before the CASE's `output:`.
  expect(errorLine(error)).toBe(lineOf(text, "then: 3"));
  assembles(rewritten(error));
});

test("a value branch is spelled to fit the former output type", () => {
  const text = flow(on('"yes"', "reply", { output: "    output: bool\n" }));

  const error = loadErrorWith(
    text,
    valueBranch("then", "route_then_0", "!bool true", " (`yes` is text; write `true`)"),
  );

  expect(errorLine(error)).toBe(lineOf(text, 'then: "yes"'));
  assembles(rewritten(error));
});

test.each([
  ['"3"', "int", '!str "3"'],
  ['"v"', "int", "!str v"],
  ["null", "str", "!Optional(str) null"],
  ["{a: 1}", "Reply", "!object {a: 1}"],
  ['"approve"', "Strr", "!str approve"],
])("a value that does not fit the former output type guesses its tag: %s under %s", (then, output, shown) => {
  const head = "typedefs:\n  Reply:\n    reply: str\n";
  const text = flow(on(then, "reply", { output: `    output: ${output}\n` }), { head });

  const error = loadErrorWith(text, valueBranch("then", "route_then_0", shown));

  expect(errorLine(error)).toBe(lineOf(text, `then: ${then}`));
  assembles(rewritten(error, { head }));
});

test("a value branch is tagged with the former output typedef", () => {
  const head = "typedefs:\n  Reply:\n    reply: str\n    exited: bool\n";
  const text = flow(on("\n          reply: ok\n          exited: true", "reply", { output: "    output: Reply\n" }), {
    head,
  });

  const error = loadErrorWith(text, valueBranch("then", "route_then_0", "!Reply {reply: ok, exited: true}"));

  expect(errorLine(error)).toBe(lineOf(text, "reply: ok"));
  assembles(rewritten(error, { head }));
});

test("a value branch under an unspellable output guesses its tag", () => {
  const text = flow(on('"approve"', "reply", { output: "    output: Union(int, str)\n" }));

  const error = loadErrorWith(text, valueBranch("then", "route_then_0", "!str approve"));

  expect(errorLine(error)).toBe(lineOf(text, 'then: "approve"'));
  assembles(rewritten(error));
});

test.each([
  ['{"": x}', 'then: {"": x}'],
  ['{a: {"": 1}}', "then: {a:"],
  ['!object {"": x}', "then: !object"],
])("a mapping with an empty key is rejected: %s", (then, needle) => {
  const text = flow(on(then));

  const error = loadErrorWith(
    text,
    "case node 'route' then: a record key can't be empty; a written-out `!object` " +
      "keeps its keys as record fields",
  );

  expect(errorLine(error)).toBe(lineOf(text, needle));
});

test("a quoted call holding a span is its own error", () => {
  const text = flow(on('"call(take, x=${input.cmd})"'));

  const error = loadErrorWith(
    text,
    "case node 'route' then: a quoted `call(...)` is text, so it can't hold `${...}` " +
      "here; drop the quotes to call the flow, or reword the text",
  );

  expect(errorLine(error)).toBe(lineOf(text, 'then: "call(take'));
});

test.each([
  ["{a: call(child, x=1)}", "then: {a: call"],
  ['[a, "call(child)"]', "then: [a"],
])("a call nested in a value is rejected: %s", (then, needle) => {
  const text = flow(on(then));

  const error = loadErrorWith(
    text,
    "case node 'route' then: a `call(...)` nested in a value is not supported; write " +
      "the call as the branch (`then: call(...)`) or in a node, or reword the text",
  );

  expect(errorLine(error)).toBe(lineOf(text, needle));
});

test.each(["${route.output}", '{a: "got ${route.output}"}'])("a value reading its own case is rejected: %s", (then) => {
  const text = flow(on(then));

  const error = loadErrorWith(text, "case node 'route' then: the value reads 'route' itself");

  expect(errorLine(error)).toBe(lineOf(text, `then: ${then}`));
});

test("a malformed span in a value is rejected", () => {
  const text = flow(on('{a: "${input.cmd"}'));

  const error = loadErrorWith(text, "case node 'route' then: unbalanced '${' in template '${input.cmd'");

  expect(errorLine(error)).toBe(lineOf(text, "then: {a:"));
});

test.each(
  ["token", "access_token"].flatMap((key) => [
    [key, `then: {${key}: abc}`, "then:"],
    [key, `then:\n          ${key}: abc`, `${key}: abc`],
  ]),
)("a mapping value branch gets the branch error under any key: %s %j", (key, branch, needle) => {
  const text = flow("    on: ${input.cmd}\n" + "    cases:\n" + "      - when: a\n" + `        ${branch}\n` + "    else: reply\n");

  const error = loadErrorWith(text, valueBranch("then", "route_then_0", `!object {${key}: abc}`));

  expect(errorLine(error)).toBe(lineOf(text, needle));
});

// ---------- unquoted words that name no node ----------

test("an unknown then word suggests the nearest node and its text entry", () => {
  const text = flow(on("replyy"));

  const error = loadErrorWith(
    text,
    "case node 'route' then: `replyy` names no node. Did you mean `reply`? For the " +
      "text, add a value entry and name it: `replyy_text: !str replyy`",
  );

  expect(errorLine(error)).toBe(lineOf(text, "then: replyy"));
  assembles(rewritten(error));
});

test("an unknown else word suggests the nearest node and its text entry", () => {
  const text = flow(on("reply", "replyy"));

  const error = loadErrorWith(
    text,
    "case node 'route' else: `replyy` names no node. Did you mean `reply`? For the " +
      "text, add a value entry and name it: `replyy_text: !str replyy`",
  );

  expect(errorLine(error)).toBe(lineOf(text, "else: replyy"));
  assembles(rewritten(error));
});

test("an unknown sentence gets no suggestion and an entry named after the branch", () => {
  const text = flow(on("The user declined"));

  const error = loadErrorWith(
    text,
    "case node 'route' then: `The user declined` names no node. For the text, add a " +
      "value entry and name it: `route_then_0: !str The user declined`",
  );

  expect(errorLine(error)).toBe(lineOf(text, "then: The user declined"));
  assembles(rewritten(error));
});

test.each(["", "    output: bool\n"])("an unknown yes gets the same text entry: %j", (output) => {
  const text = flow(on("reply", "yes", { output }));

  const error = loadErrorWith(
    text,
    "case node 'route' else: `yes` names no node. For the text, add a value entry " +
      "and name it: `yes_text: !str yes`",
  );

  expect(errorLine(error)).toBe(lineOf(text, "else: yes"));
  assembles(rewritten(error));
});

test("an unknown word entry id skips an id already taken", () => {
  const taken = "  nope_text: !str taken\n";
  const text = flow(on("nope"), { extraNodes: taken });

  const error = loadErrorWith(
    text,
    "case node 'route' then: `nope` names no node. Did you mean `nope_text`? For the " +
      "text, add a value entry and name it: `nope_text_2: !str nope`",
  );

  expect(errorLine(error)).toBe(lineOf(text, "then: nope"));
  assembles(rewritten(error, { extraNodes: taken }));
});

test.each([
  [
    "call(child, x=1) | x",
    "reply",
    "case node 'route' then: a branch target must be a node id or a single " +
      "call: text follows the closing `)`: '| x'; a call(...) must be the whole " +
      "value. For the text, add a value entry and name it: " +
      '`route_then_0: !str "call(child, x=1) | x"`',
    "then: call(",
  ],
  [
    "reply",
    "call(child, x=1) | y",
    "case node 'route' else: a branch target must be a node id or a single " +
      "call: text follows the closing `)`: '| y'; a call(...) must be the whole " +
      "value. For the text, add a value entry and name it: " +
      '`route_else: !str "call(child, x=1) | y"`',
    "else: call(",
  ],
])("unquoted call that is not one directive is rejected: %s / %s", (then, else_, message, needle) => {
  const text = flow(on(then, else_));

  const error = loadErrorWith(text, message);

  expect(errorLine(error)).toBe(lineOf(text, needle));
  assembles(rewritten(error));
});

// ---------- a CASE `output:` ----------

test.each([
  ["output: str", ", e.g. `x: !str ${route.output}`", "output: str"],
  ["output: !str", ", e.g. `x: !str ${route.output}`", "output: !str"],
  ["output: !int", ", e.g. `x: !int ${route.output}`", "output: !int"],
  ["output: Optional(str)", ", e.g. `x: !Optional(str) ${route.output}`", "output:"],
  ["output: Union(int, str)", "; name `Union(int, str)` in `typedefs:` and tag with its name", "output: Union"],
  ["output: Strr", ", e.g. `x: !Strr ${route.output}`", "output: Strr"],
  ["output:", ", e.g. `x: !<type> ${route.output}`", "output:"],
  ["output:\n      a: int", "; name `{a: int}` in `typedefs:` and tag with its name", "output:"],
  ["output:\n      a: Strr", "; name `{a: Strr}` in `typedefs:` and tag with its name", "output:"],
  ["output:\n      a: .inf", "; name `{a: .inf}` in `typedefs:` and tag with its name", "output:"],
  ["output: &a [*a]", ", e.g. `x: !<type> ${route.output}`", "output: &a"],
  ["output: &a {x: *a}", ", e.g. `x: !<type> ${route.output}`", "output: &a"],
])("a case output is rejected at its line: %j", (output, example, needle) => {
  const text = flow(on("reply", "reply", { output: `    ${output}\n` }));

  const error = loadErrorWith(
    text,
    "case node 'route': a `case` routes and has no `output:`; tag its result where " + `you read it${example}`,
  );

  expect(errorLine(error)).toBe(lineOf(text, needle));
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>(
  [
    "a_block_mapping_value_branch",
    "a_block_text_value_branch_is_shown_escaped",
    "a_bool_value_branch",
    "a_branch_tag_wins_over_the_former_output_type",
    "a_call_nested_in_a_value_is_rejected",
    "a_case_output_union_asks_for_a_typedef_name",
    "a_float_else_value",
    "a_list_value_branch",
    "a_lone_reference_is_tagged_with_its_type",
    "a_lone_span_of_unknown_type_asks_for_its_type",
    "a_long_mapping_is_cut_and_written_as_a_block",
    "a_malformed_span_in_a_value_is_rejected",
    "a_mapping_with_an_empty_key_is_rejected",
    "a_quoted_call_holding_a_span_is_its_own_error",
    "a_quoted_text_then_is_a_value_branch",
    "a_text_value_branch_keeps_the_quotes_it_needs",
    "a_value_branch_is_spelled_to_fit_the_former_output_type",
    "a_value_branch_is_tagged_with_the_former_output_typedef",
    "a_value_reading_its_own_case_is_rejected",
    "a_value_that_contains_itself_is_rejected",
    "an_empty_else_is_a_null_value",
    "an_unknown_then_word_suggests_the_nearest_node",
    "an_unknown_yes_gets_a_text_entry",
    "an_unquoted_call_with_trailing_text_is_rejected",
    "text_that_reads_as_a_number_is_shown_quoted",
  ].map((name) => ({ id: `load-case/compose_case_branches__${name}`, call: assembled })),
)("assemble: $id", runCase);
