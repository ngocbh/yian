/**
 * Load checks for value entries in `nodes:` (`total: !int ${a.output + b.output}`): a port of
 * `tests/compose/test_value_entries_load.py`, loading each flow with `assembleText` (`assemble`
 * as the loader calls it) where Python calls `load_flow`.
 *
 * A tagged entry lowers at load to a CODE node with the entry's id and one typed input named
 * after it, so the tag is checked like a typed binding's: a span-free literal is checked while the
 * entry is read, a span's inferred type must not be disjoint from the tag, and a nested tag checks
 * its own value. A tag never fills in a missing field. A bare
 * `!object` over a written-out mapping keeps the record the mapping spells, and an empty key in
 * that record is a load error.
 *
 * Ported: `test_a_value_entry_builds_a_code_node_with_the_entry_id`,
 * `test_named_type_entry_reads_a_float_producer`,
 * `test_named_type_entry_rejects_a_disjoint_producer_at_the_entry`,
 * `test_a_tag_does_not_fill_in_a_missing_field`,
 * `test_a_named_field_read_into_a_disjoint_binding_is_rejected`,
 * `test_a_nested_tag_is_checked_at_its_own_line`, `test_entry_errors_are_located_at_the_entry`,
 * `test_a_disjoint_span_names_the_value_entry`,
 * `test_a_flow_call_of_a_disjoint_type_is_rejected_at_the_entry` (its `f` def built by hand),
 * `test_a_span_field_takes_its_producer_type`, `test_an_entry_under_a_credential_name_loads`,
 * `test_a_mistyped_literal_under_a_credential_name_gets_the_ordinary_error`,
 * `test_a_nested_binding_loads`, `test_fingerprint_is_stable_and_tracks_the_literal`,
 * `test_an_object_entry_with_an_empty_key_is_rejected`,
 * `test_an_empty_key_outside_a_written_out_record_loads` (a load row, then a run row),
 * `test_a_flow_call_inside_a_span_keeps_its_error`,
 * `test_a_written_out_object_entry_rejects_an_unknown_field_at_the_reader`,
 * `test_a_span_record_rejects_an_unknown_field`, `test_a_nested_untagged_mapping_keeps_its_fields`,
 * and the rows the build-cache reader decides (it types a span or nested tag in a written-out
 * record): `test_entries_reading_each_other_are_rejected`, `test_an_entry_reading_itself_is_rejected`,
 * `test_a_span_from_an_untyped_producer_is_opaque` and `test_a_nested_tagged_object_is_opaque`.
 *
 * Changed: the CODE node runs the built-in identity `{builtin: "identity", field: <id>}`, not
 * generated code. A mismatched text value is JSON-quoted (rule `messages`), and a missing-field
 * list is written in the text form (`["action"]`).
 * `test_a_number_that_is_not_finite_is_rejected_at_the_entry`: infinity is a float value in TS
 * (rule `infinity`), so the `.inf` and `-.inf` entries load, and the YAML reader refuses `.nan` at
 * the scalar before any entry is read. `test_a_written_out_object_entry_keeps_its_fields` also checks
 * the built node's output contract, the record `literalContract` spells;
 * `test_a_literal_field_takes_its_literal_type` and
 * `test_an_escaped_span_in_an_object_entry_is_text` check the typed reader's error, as Python
 * does, and also the record `literalContract` spells;
 * `test_an_object_entry_over_an_outside_value_stays_object` and
 * `test_an_object_entry_not_over_a_bare_object_mapping_keeps_its_tag` (its `call(...)` row with the
 * `g` def built by hand) check that the contract is the tag, and
 * `test_a_nested_tagged_object_is_opaque` that it is opaque.
 * `test_a_nested_untagged_mapping_keeps_its_fields` and
 * `test_a_span_record_rejects_an_unknown_field` also have rows that read a field the record has;
 * those check the output contract the reader gives the built entry node. A field from an untyped
 * producer stays opaque in that partial record.
 *
 * TS-only: the descriptor refusing node-only fields, a `RangeError` (a compiler bug) where Python
 * raises `ValueError`; and the harvested `load-case/compose_value_entries_load__*` cases this
 * reader, the build-cache reader, the typed-binding check and the reference validator decide
 * (`tags.test.ts` pins the ones `readTaggedValue` decides).
 *
 * Changed: the run of `test_an_empty_key_outside_a_written_out_record_loads` reads its ints back as
 * `bigint`s (decision 4a).
 */

import { afterAll, afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { flowFingerprint } from "../../src/compile/manifest.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { ValueEntryDescriptor } from "../../src/compose/desugar/value_entries.ts";
import type { LoadError } from "../../src/compose/helpers/errors.ts";
import { literalContract } from "../../src/compose/helpers/output_contracts.ts";
import { forget } from "../../src/defaults.ts";
import { NodeKind } from "../../src/nodes/base.ts";
import { CodeNode } from "../../src/nodes/code/node.ts";
import {
  type OutputContract,
  PartialRecordContract,
  ValueContract,
  type ValueTemplate,
} from "../../src/nodes/contracts.ts";
import { runFlow } from "../../src/run.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import { canonicalTypeLabel, Type, ValueKind } from "../../src/typesys/values.ts";
import { hostEnv } from "../engine/mcp_helpers.ts";
import {
  assembled,
  assembleText,
  type CaseRow,
  handChild,
  loadConformanceFlow,
  loadError,
  parametersFromYaml,
  runCase,
} from "./declarations.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-value-entries-load-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

afterEach(() => forget());

const BUY = "typedefs:\n  Buy:\n    action: Literal(buy)\n    conviction: float\n";

const F_DEF =
  "defs:\n" +
  "  f:\n" +
  "    input:\n" +
  "      x: !str\n" +
  "    nodes:\n" +
  "      up:\n" +
  "        kind: code\n" +
  "        input:\n" +
  "          x: ${input.x}\n" +
  "        output: str\n" +
  "        code: |\n" +
  "          return inputs['x'].toUpperCase()\n" +
  "    output: ${up.output}\n";

const G_DEF =
  "defs:\n" +
  "  g:\n" +
  "    input:\n" +
  "      n: !int\n" +
  "    nodes:\n" +
  "      v:\n" +
  "        kind: code\n" +
  "        output: {q: int}\n" +
  "        code: |\n" +
  "          return {'q': 1}\n" +
  "    output: ${v.output}\n";

/** Return a flow with a `t: str` input, a float producer `s`, and `entries`. */
function flow(entries: string, options: { output?: string; head?: string } = {}): string {
  const { output = "${x.output}", head = "" } = options;
  return (
    "id: f\n" +
    "name: f\n" +
    head +
    "input:\n" +
    "  t: !str\n" +
    "nodes:\n" +
    "  s:\n" +
    "    kind: code\n" +
    "    output: float\n" +
    "    code: |\n" +
    "      return 0.5\n" +
    entries +
    `output: ${output}\n`
  );
}

/** Return a flow with `entries` and a CODE node `use` whose input `c` is `reader`. */
function recordFlow(entries: string, reader: string): string {
  return (
    "id: f\n" +
    "name: f\n" +
    "input:\n" +
    "  n: !int\n" +
    "nodes:\n" +
    entries +
    "  use:\n" +
    "    kind: code\n" +
    "    input:\n" +
    `      c: ${reader}\n` +
    "    code: |\n" +
    "      return inputs['c']\n" +
    "output: ${use.output}\n"
  );
}

/** Return the 1-based line number of the first line containing `needle`. */
function lineOf(text: string, needle: string): number {
  return text.split("\n").findIndex((line) => line.includes(needle)) + 1;
}

/** Load `text`, expecting a load error, and return it. */
const loadErrorOf = (text: string): LoadError => loadError(() => assembleText(text));

/** Return the start line of the innermost frame of a load error. */
function errorLine(error: LoadError): number {
  return error.diagnostic.frames.at(-1)!.range.startLine;
}

/** Load `text` and return its flow. */
const load = (text: string): ComposedFlow => assembleText(text).flow;

/** Return the source of value entry `nodeId`'s binding in the loaded `text`. */
const entrySource = (text: string, nodeId: string) => load(text).compiled.wiring.get(nodeId)!.get(nodeId)!.source;

/** Return the output contract of node `nodeId` of the loaded `text`. */
const contractOf = (text: string, nodeId: string) => load(text).compiled.nodes.get(nodeId)!.outputContract;

/** Return the label of an exact contract's type. */
const labelOf = (contract: OutputContract | null) => canonicalTypeLabel((contract as ValueContract).type);

test("a value entry builds a CODE node with the entry id", () => {
  const node = load(flow("  x: !int 3\n")).compiled.nodes.get("x")!;

  expect(node.kind).toBe(NodeKind.CODE);
  expect((node.outputContract as ValueContract).type.kind).toBe(ValueKind.INTEGER);
  expect((node as CodeNode).code).toStrictEqual({ builtin: "identity", field: "x" });
  expect((node as CodeNode).mode).toBe("builtin");
});

test("a named type entry reads a float producer", () => {
  load(flow("  x: !Buy\n    action: buy\n    conviction: ${s.output}\n", { head: BUY }));
});

test("a named type entry rejects a disjoint producer at the entry", () => {
  const text = flow("  x: !Buy\n    action: buy\n    conviction: ${s.output}\n", { head: BUY }).replace(
    "output: float",
    "output: str",
  );

  const error = loadErrorOf(text);

  expect(error.message).toBe("value entry 'x': declared Buy, but the value is " + '{"action":"buy","conviction":str}');
  expect(errorLine(error)).toBe(lineOf(text, "  x: !Buy"));
});

test("a tag does not fill in a missing field", () => {
  const text = flow("  x: !Buy {conviction: 1}\n", { head: BUY });

  const error = loadErrorOf(text);

  expect(error.message).toBe(
    "'x': the value is not a valid Buy: record missing required fields: " + '["action"] (at `x.action`)',
  );
  expect(errorLine(error)).toBe(lineOf(text, "  x: !Buy"));
});

test("a named field read into a disjoint binding is rejected", () => {
  const text = flow(
    "  x: !Buy {action: buy, conviction: 1}\n" +
      "  use:\n" +
      "    kind: code\n" +
      "    input:\n" +
      "      c: !str ${x.output.conviction}\n" +
      "    code: |\n" +
      "      return inputs['c']\n",
    { head: BUY },
  );

  const error = loadErrorOf(text);

  expect(error.message).toBe("node 'use' input 'c': declared str, but the value is float");
  expect(errorLine(error)).toBe(lineOf(text, "c: !str"));
});

test("a nested tag is checked at its own line", () => {
  const good = flow("  x: !object\n    d: !Buy {action: buy, conviction: 1}\n", { head: BUY });
  load(good);

  const bad = good.replace("conviction: 1}", "conviction: x}");
  const error = loadErrorOf(bad);

  expect(error.message).toBe(
    "'x.d': the value is not a valid Buy: \"x\" does not match declared type float (at `x.d.conviction`)",
  );
  expect(errorLine(error)).toBe(lineOf(bad, "    d: !Buy"));
});

test.each([
  ["  x: !int\n", "value entry 'x' needs a value after its tag, e.g. `x: !int ...`"],
  ["  x: !str 30\n", "'x': `30` is not a valid str; quote it to keep it as text: `x: !str \"30\"`"],
  ["  x: !int (1..3) 2\n", "'x': a YAML tag ends at the first space; write `!int(1..3)` without spaces"],
  ["  x: !Strr hi\n", "'x': `!Strr` is not a type (unknown type 'Strr'). Did you mean `str`?"],
  ["  x: !object [1]\n", "'x': the value is not a valid object: [1] does not match declared type object"],
])("entry error is located at the entry: %j", (entry, message) => {
  const text = flow(entry);

  const error = loadErrorOf(text);

  expect(error.message).toBe(message);
  expect(errorLine(error)).toBe(lineOf(text, "  x: !"));
});

test("a disjoint span names the value entry", () => {
  const text = flow("  x: !str ${s.output}\n");

  const error = loadErrorOf(text);

  expect(error.message).toBe("value entry 'x': declared str, but the value is float");
  expect(errorLine(error)).toBe(lineOf(text, "  x: !str"));
});

test("a flow call of a disjoint type is rejected at the entry", () => {
  // `f` takes text and returns its `up` node's text.
  const f = handChild({
    inputs: parametersFromYaml("input:\n  x: !str\n", new Map()),
    producers: { up: new ValueContract(Type.scalar(ValueKind.STRING)) },
    output: "${up.output}",
  });
  const text = flow("  x: !int call(f, x=${input.t})\n", { head: F_DEF });

  const error = loadError(() => assembleText(text, "flow.yaml", { children: { f } }));

  expect(error.message).toBe("value entry 'x': declared int, but the value is str");
  expect(errorLine(error)).toBe(lineOf(text, "  x: !int call"));
});

test("an entry under a credential name loads", () => {
  const scalar = flow("  api_key: !str sk-live-123\n", { output: "${api_key.output}" });
  const record = flow("  cfg: !object\n    access_token: abc\n    x: ${s.output}\n", { output: "${cfg.output}" });

  expect(entrySource(scalar, "api_key")).toBe("sk-live-123");
  expect((entrySource(record, "cfg") as ValueTemplate).value).toStrictEqual({ access_token: "abc", x: "${s.output}" });
});

test("a mistyped literal under a credential name gets the ordinary error", () => {
  const text = flow("  api_key: !int sk-live-123\n", { output: "${api_key.output}" });

  const error = loadErrorOf(text);

  expect(error.message).toBe("'api_key': `sk-live-123` is not a valid int");
  expect(errorLine(error)).toBe(lineOf(text, "api_key:"));
});

test("a nested binding loads", () => {
  const text = flow("  cfg: !object\n    access_token: ${input.t}\n", { output: "${cfg.output}" });

  expect((entrySource(text, "cfg") as ValueTemplate).value).toStrictEqual({ access_token: "${input.t}" });
});

test("the fingerprint is stable and tracks the literal", () => {
  const text = flow("  x: !object {exited: true, reply: ''}\n");

  const first = flowFingerprint(load(text).compiled);
  const second = flowFingerprint(load(text).compiled);
  const changed = flowFingerprint(load(text.replace("true", "false")).compiled);

  expect(first).toBe(second);
  expect(changed).not.toBe(first);
});

describe("the record a written-out object keeps", () => {
  test("a written-out object entry keeps its fields as a record", () => {
    const text = recordFlow('  exit_result: !object {exited: true, reply: ""}\n', "${exit_result.output.reply}");

    expect(contractOf(text, "exit_result")).toStrictEqual(literalContract({ exited: true, reply: "" }));
  });

  test("a written-out object entry rejects an unknown field at the reader", () => {
    const text = recordFlow('  exit_result: !object {exited: true, reply: ""}\n', "${exit_result.output.zz}");

    const error = loadErrorOf(text);

    expect(error.message).toBe(
      "flow has unresolved references:\n" +
        "  node 'use' input 'c' from: reference ${exit_result.output.zz} reads unknown field 'zz' on record",
    );
    expect(errorLine(error)).toBe(lineOf(text, "c: ${exit_result"));
  });

  test.each([
    ["[1, 2]", [1n, 2n], "list(int)"],
    ["true", true, "bool"],
  ])("a literal field takes its literal type: %s", (field, value, label) => {
    const contract = contractOf(recordFlow(`  r: !object\n    b: ${field}\n`, "${r.output.b}"), "r");

    expect(contract).toStrictEqual(literalContract({ b: value }));
    expect(canonicalTypeLabel((contract as ValueContract).type.fields!.get("b")!)).toBe(label);
  });

  // Python's `test_a_literal_field_takes_its_literal_type` asserts it through a typed reader.
  test.each([
    ["[1, 2]", "list(int)"],
    ["true", "bool"],
  ])("a literal field's type reaches a typed reader: %s", (field, label) => {
    const text = recordFlow(`  r: !object\n    b: ${field}\n`, "!str ${r.output.b}");

    const error = loadErrorOf(text);

    expect(error.message).toBe(`node 'use' input 'c': declared str, but the value is ${label}`);
    expect(errorLine(error)).toBe(lineOf(text, "c: !str"));
  });

  test("a nested untagged mapping keeps its fields", () => {
    const contract = contractOf(recordFlow("  r: !object\n    p: {q: 1}\n", "${r.output.p}"), "r");

    expect(contract).toStrictEqual(literalContract({ p: { q: 1n } }));
  });

  test("an escaped span in an object entry is text", () => {
    const contract = contractOf(recordFlow('  r: !object {a: "$${input.n}"}\n', "${r.output.a}"), "r");

    expect(contract).toStrictEqual(literalContract({ a: "${input.n}" }));
  });

  // Python's `test_an_escaped_span_in_an_object_entry_is_text` asserts it through a typed reader.
  test("an escaped span in an object entry is text to a typed reader", () => {
    const text = recordFlow('  r: !object {a: "$${input.n}"}\n', "!int ${r.output.a}");

    const error = loadErrorOf(text);

    expect(error.message).toBe("node 'use' input 'c': declared int, but the value is str");
    expect(errorLine(error)).toBe(lineOf(text, "c: !int"));
  });

  test("an object entry over an outside value stays object", () => {
    const entries =
      "  x:\n" +
      "    kind: code\n" +
      "    output: {q: int}\n" +
      "    code: |\n" +
      "      return {'q': 1}\n" +
      "  o: !object ${x.output}\n";

    expect(labelOf(contractOf(recordFlow(entries, "${o.output.zz}"), "o"))).toBe("object");
  });

  test("an object entry over a call keeps its tag", () => {
    // `g` returns the record `{q: int}`; the entry reads it through its `!object` tag.
    const g = handChild({
      inputs: parametersFromYaml("input:\n  n: !int\n", new Map()),
      producers: { v: new ValueContract(typeFor("Q", readTypedefs({ Q: { q: "int" } }))) },
      output: "${v.output}",
    });
    const text = recordFlow("  o: !object call(g, n=${input.n})\n", "${o.output.zz}").replace(
      "input:\n",
      G_DEF + "input:\n",
    );

    const { flow } = assembleText(text, "flow.yaml", { children: { g } });

    expect(labelOf(flow.compiled.nodes.get("o")!.outputContract)).toBe("object");
  });

  test.each(["  o: !Optional(object) {a: 1}\n", "  o: !Optional(object)\n    a: ${input.n}\n"])(
    "an object entry not over a bare object mapping keeps its tag: %j",
    (entry) => {
      expect(labelOf(contractOf(recordFlow(entry, "${o.output.zz}"), "o"))).toBe("Optional(object)");
    },
  );
});

describe("the build-cache reader", () => {
  test("entries reading each other are rejected", () => {
    const text = flow("  x: !object\n    a: ${y.output}\n  y: !object\n    b: ${x.output}\n");

    const error = loadErrorOf(text);

    expect(error.message).toBe("node build dependency cycle: x -> y -> x");
    expect(errorLine(error)).toBe(lineOf(text, "  y: !object"));
  });

  test("an entry reading itself is rejected", () => {
    const text = flow("  x: !object\n    a: ${x.output}\n");

    const error = loadErrorOf(text);

    expect(error.message).toBe("node build dependency cycle: x -> x");
    expect(errorLine(error)).toBe(lineOf(text, "  x: !object"));
  });

  /** A CODE node `p` returning text, declaring `output` unless it is empty. */
  const producer = (output: string) =>
    `  p:\n    kind: code\n${output ? `    output: ${output}\n` : ""}    code: |\n      return 'hi'\n`;
  const SPAN_RECORD = "  r: !object\n    a: ${p.output}\n    b: 1\n";

  test("a span field takes its producer type", () => {
    const text = recordFlow(producer("str") + SPAN_RECORD, "!int ${r.output.a}");

    const error = loadErrorOf(text);

    expect(error.message).toBe("node 'use' input 'c': declared int, but the value is str");
    expect(errorLine(error)).toBe(lineOf(text, "c: !int"));
  });

  test("a span record from a typed producer keeps its fields", () => {
    const contract = contractOf(recordFlow(producer("str") + SPAN_RECORD, "${r.output.a}"), "r");

    expect(contract?.constructor).toBe(ValueContract);
    expect(labelOf(contract)).toBe('{"a":str,"b":int}');
  });

  test("a span record from an untyped producer keeps its fields", () => {
    const contract = contractOf(recordFlow(producer("") + SPAN_RECORD, "${r.output.a}"), "r");

    // An untyped producer leaves its field opaque beside the typed literal one.
    expect(contract?.constructor).toBe(PartialRecordContract);
    const fields = (contract as PartialRecordContract).fields;
    expect(fields.map(([name, field]) => [name, field === null ? null : canonicalTypeLabel(field.type)])).toEqual([
      ["a", null],
      ["b", "int"],
    ]);
  });

  test.each(["str", ""])("a span record rejects an unknown field: %j", (output) => {
    const text = recordFlow(producer(output) + SPAN_RECORD, "${r.output.c}");

    const error = loadErrorOf(text);

    expect(error.message).toBe(
      "flow has unresolved references:\n" +
        "  node 'use' input 'c' from: reference ${r.output.c} reads unknown field 'c' on record",
    );
    expect(errorLine(error)).toBe(lineOf(text, "c: ${r.output.c}"));
  });

  test("a span from an untyped producer is opaque", () => {
    expect(() => load(recordFlow(producer("") + SPAN_RECORD, "!int ${r.output.a}"))).not.toThrow();
  });

  test("a nested untagged mapping beside a span keeps its fields", () => {
    const entry = "  r: !object\n    a: ${p.output}\n    p: {q: 1}\n";

    const contract = contractOf(recordFlow(producer("str") + entry, "${r.output.p.q}"), "r");

    expect(labelOf(contract)).toBe('{"a":str,"p":{"q":int}}');
  });

  test.each(["  r: !object\n    a: ${p.output}\n    p: {q: 1}\n", "  r: !object\n    p: {q: 1}\n"])(
    "a nested untagged mapping rejects an unknown field: %j",
    (entry) => {
      const text = recordFlow(producer("str") + entry, "${r.output.p.zz}");

      const error = loadErrorOf(text);

      expect(error.message).toBe(
        "flow has unresolved references:\n" +
          "  node 'use' input 'c' from: reference ${r.output.p.zz} reads unknown field 'zz' on record",
      );
      expect(errorLine(error)).toBe(lineOf(text, "c: ${r.output.p.zz}"));
    },
  );

  test.each([
    ["  r: !object\n    a: ${p.output}\n    p: !object {q: 1}\n", '{"a":str,"p":object}'],
    ["  r: !object\n    p: !object {q: 1}\n", '{"p":object}'],
  ])("a nested tagged object is opaque: %j", (entry, label) => {
    expect(labelOf(contractOf(recordFlow(producer("str") + entry, "${r.output.p.zz}"), "r"))).toBe(label);
  });
});

const EMPTY_KEY = "a record key can't be empty; a written-out `!object` keeps its keys as record fields";

test.each([
  ['  r: !object {"": 1, a: 2}\n', "r", "  r: !object"],
  ['  r: !object {"": "${input.n}"}\n', "r", "  r: !object"],
  ['  r: !object {a: {"": 1}}\n', "r.a", "  r: !object"],
  ['  r: !object {a: [1, {"": 1}]}\n', "r.a[1]", "  r: !object"],
  ['  r: !object\n    a:\n      "": 1\n', "r.a", '      "": 1'],
])("an object entry with an empty key is rejected: %j", (entry, label, needle) => {
  // No record field can be named "", and the read must not quietly go unchecked.
  const text = recordFlow(entry, "${r.output.b}");

  const error = loadErrorOf(text);

  expect(error.message).toBe(`'${label}': ${EMPTY_KEY}`);
  expect(errorLine(error)).toBe(lineOf(text, needle));
});

const EMPTY_KEY_OUTSIDE: [string, unknown][] = [
  ['  r: !Optional(object) {"": 1}\n', { "": 1n }],
  ['  r: !list(object) [{"": 1}]\n', [{ "": 1n }]],
  ['  r: !object {a: !object {"": 1}}\n', { a: { "": 1n } }],
];

test.each(EMPTY_KEY_OUTSIDE)("an empty key outside a written-out record loads: %j", (entry) => {
  load(recordFlow(entry, "${r.output}"));
});

test.each(EMPTY_KEY_OUTSIDE)("an empty key outside a written-out record loads (runs): %j", async (entry, value) => {
  const result = await runFlow(await loadConformanceFlow(recordFlow(entry, "${r.output}")), { n: 3n }, { env });

  expect(result.failure).toBeNull();
  expect(result.output).toStrictEqual(value);
});

describe("numbers that are not finite", () => {
  test.each<[string, unknown]>([
    ["  x: !float .inf\n", Infinity],
    ["  x: !float -.inf\n", -Infinity],
    ["  x: !object\n    a: .inf\n", { a: Infinity }],
  ])("an infinite entry loads: %j", (entry, expected) => {
    expect(entrySource(flow(entry), "x")).toStrictEqual(expected);
  });

  test.each(["  x: !object {a: !float .inf}\n", '  x: !object {a: "${input.t}", b: .inf}\n'])(
    "an infinite value in a templated entry loads: %j",
    (entry) => {
      load(flow(entry));
    },
  );

  test.each([
    ["  x: !object {a: .nan}\n", "  x: !object"],
    ["  x: !list(float) [1.0, .nan]\n", "  x: !list"],
  ])("a NaN is refused by the YAML reader at the scalar: %j", (entry, needle) => {
    const text = flow(entry);

    const error = loadErrorOf(text);

    expect(error.message).toBe("invalid YAML: NaN is not a yian value; quote it to keep it as text");
    expect(errorLine(error)).toBe(lineOf(text, needle));
  });
});

test.each([
  ["asserts", { asserts: ["x > 0"] }],
  ["depends_on", { dependsOn: ["s"] }],
  ["runs_after", { runsAfter: ["s"] }],
  ["env", { env: { k: 1 } }],
])("a value entry can't carry %s", (name, fields) => {
  const tag = new ValueContract(new Type({ kind: ValueKind.INTEGER }));
  const make = () =>
    new ValueEntryDescriptor({ id: "x", code: { builtin: "identity", field: "x" }, inputs: { x: 1n }, tag, ...fields });

  expect(make).toThrow(RangeError);
  expect(make).toThrow(`value entry 'x' can't carry '${name}'`);
});

test("a flow call inside a span keeps its error", () => {
  const text = flow("  x: !str ${ f(input.t) }\n", { head: F_DEF });

  const error = loadErrorOf(text);

  expect(error.message).toBe(
    "x: a flow call 'f' inside ${...} is not supported — write it as the whole " +
      "value with the call(...) directive: call(f, ...)",
  );
  expect(errorLine(error)).toBe(lineOf(text, "  x: !str ${ f("));
});

describe("harvested cases", () => {
  test.each<CaseRow>(
    [
      "a_tag_without_a_value_asks_for_one",
      "an_object_entry_with_an_empty_key_is_rejected",
      "an_infinite_float_entry",
      "a_nested_binding_loads",
      "entries_reading_each_other_are_rejected",
      "an_entry_under_a_credential_name_loads",
      "a_disjoint_span_names_the_value_entry",
      "a_named_type_entry_rejects_a_disjoint_producer",
      "a_span_field_takes_its_producer_type",
      "a_literal_field_takes_its_literal_type",
      "a_span_record_rejects_an_unknown_field",
      "a_written_out_object_rejects_an_unknown_field",
      // The inline-call desugar raises it before the `f` callee resolves.
      "a_flow_call_inside_a_span_keeps_its_error",
    ].map((name) => ({ id: `load-case/compose_value_entries_load__${name}`, call: assembled })),
  )("$id", runCase);
});
