/**
 * The `case` kind's parse and build rows: a port of `tests/compose/test_kind_case_parse.py`.
 *
 * Ported: `test_case_parse_matches_descriptor_and_records_ranges`,
 * `test_case_parse_requires_cases`, `test_case_with_both_on_spellings_is_a_located_error`,
 * `test_case_rejects_boolean_key_at_its_location`,
 * `test_case_build_lowers_to_case_node_with_edges`,
 * `test_case_descriptor_rejects_misaligned_or_nonopaque_origins`,
 * `test_on_case_preserves_predicate_and_generated_binding_origins`,
 * `test_case_parse_keeps_then_styles_aligned_with_cases`, `test_case_parse_keeps_else_style`,
 * `test_case_parse_tells_else_null_from_no_else`,
 * `test_case_descriptor_constructed_with_else_has_else`, `test_case_parse_keeps_then_null_key`,
 * `test_case_parse_on_form_keeps_styles`, `test_case_rejects_malformed_cases_at_their_location`,
 * `test_source_free_case_build_keeps_predicates_and_wiring_origin_free`, the origin half of
 * `test_searched_case_origins_are_positional_and_select_the_failing_entry` (on the parsed
 * descriptor and on the built node), and the `thenRanges[0]` assertion of
 * `tests/compose/test_desugar.py`'s `test_desugar_case_call_targets_locate_at_their_target_line`.
 *
 * Changed: Python compares the descriptor to a `CaseDescriptor` with the parse metadata excluded;
 * TS compares its class and fields. The duplicate-key and boolean-key rows parse the flow with
 * `parseSourceDocument`, which raises both, where Python loads it. The malformed-`cases:` rows
 * parse the flow's nodes, where Python loads it, and spell values in the text form (`(got "done")`,
 * `(got null)`). The origin rows call `withSourceOrigins` on the parsed descriptor and resolve its
 * origins through `builder.resolveRegistered`, where Python loads the flow and reads the built
 * `CaseNode`; the searched and `on:` rows also assemble the flow (`assembleText`) and read the
 * built node and its `__r0` or `__on` wiring. The misaligned origins are a `RangeError` where
 * Python's are a `ValueError`, and the message says "or null" where Python's says "or None". The
 * requires-cases row pins the whole message and its range (the node body), where Python matches a
 * fragment. The misaligned-origins row also checks that the origins are a frozen copy.
 *
 * TS-only: the lowering row also checks the `__on` input's range and label; an `on:` case entry
 * with a null `when:` is refused at that `when:`; a lone surrogate in `on:` or in a searched
 * `when:` is refused at that value; a non-`OriginRef` `onOrigin` is a `TypeError` (the
 * misaligned-origins row; no Python test checks Python's matching `on_origin` check).
 *
 * Dropped: `test_case_keeps_no_key_normalization` (Python class-dict introspection).
 *
 * Later (T4): the run half of
 * `test_searched_case_origins_are_positional_and_select_the_failing_entry`. Later (T3-20, T3-17):
 * `test_source_attachment_does_not_preempt_malformed_case_validation` (`case_branches`,
 * `expand_case_outputs`). Later (T3-22, T4):
 * `test_named_definition_case_failure_keeps_child_predicate_origin`. Later (T3-17, T4):
 * `test_inline_definition_case_failure_keeps_child_predicate_origin`. Later (T3-22):
 * `test_external_alias_rebases_case_and_generated_binding_origins`,
 * `test_generated_child_clone_preserves_case_predicate_origins`,
 * `test_formatting_moves_case_ranges_without_changing_keys_or_fingerprint`. Later (T3-17, T3-20):
 * the harvested `load-case/compose_kind_case_parse__a_case_with_no_branch_has_no_value` and
 * `__a_flow_style_case_names_no_node`.
 *
 * The harvested cases whose error the CASE parse or build raises are pinned at the end, with
 * `load-core/compose_expression_nesting__thousands_of_nested_spans` and `__too_deep_searched_when`
 * from `tests/compose/test_expression_nesting.py`, which has no TS mirror yet.
 */

import { expect, test } from "bun:test";

import { LoadDiagnostics } from "../../src/compose/helpers/errors.ts";
import { BuildContext, siteKey } from "../../src/compose/nodes/base.ts";
import { CaseDescriptor } from "../../src/compose/nodes/case.ts";
import { parseSourceDocument } from "../../src/compose/parse/file.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { SourceCatalogBuilder } from "../../src/compose/source/catalog.ts";
import { type SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { OriginRef, SourceRange } from "../../src/diagnostics.ts";
import { ELSE_HANDLE } from "../../src/nodes/base.ts";
import { CaseNode } from "../../src/nodes/case/node.ts";
import { Binding } from "../../src/nodes/contracts.ts";
import { assembled, assembleText, type CaseRow, loadError, nodesOf, parsedNodes, runCase } from "./declarations.ts";

const FLOW = `nodes:
  route:
    kind: case
    on: \${score.output}
    cases:
      - when: high
        then: hot
    else: cold
`;

const SEARCHED_ORIGIN_FLOW = `id: searched-case-origins
name: searched-case-origins
nodes:
  zero:
    kind: code
    code: "return 0"
    output: int
  gate:
    kind: case
    cases:
      - when: "zero.output > 0"
        then: positive
      - when: "1 / zero.output > 0"
        then: broken
      - when: "zero.output > 0"
        then: duplicate
    else: negative
  positive:
    kind: code
    code: 'return "positive"'
    output: str
  broken:
    kind: code
    code: 'return "broken"'
    output: str
  duplicate:
    kind: code
    code: 'return "duplicate"'
    output: str
  negative:
    kind: code
    code: 'return "negative"'
    output: str
output: \${positive.output | broken.output | duplicate.output | negative.output}
`;

const ON_ORIGIN_FLOW = `id: on-case-origins
name: on-case-origins
nodes:
  label:
    kind: code
    code: 'return "high"'
    output: str
  route:
    kind: case
    on: \${label.output}
    cases:
      - when: high
        then: hot
      - when: low
        then: cold
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

/** A build context with no typedefs; a CASE build reads none. */
const CTX = new BuildContext({ registry: new Map() });

/** Return the cursor over node `nodeId` of `text`'s `nodes:`. */
function readerFor(text: string, nodeId: string): SourceCursor {
  return SourceDocument.parse(text).root.child("nodes").child(nodeId);
}

/** Return a descriptor's fields, leaving out the parse metadata (Python's `compare=False`). */
function fieldsOf(desc: CaseDescriptor): Record<string, unknown> {
  const {
    range: _range,
    fieldRanges: _fieldRanges,
    assertRanges: _assertRanges,
    whenRanges: _whenRanges,
    thenRanges: _thenRanges,
    thenStyles: _thenStyles,
    elseStyle: _elseStyle,
    thenTags: _thenTags,
    elseTag: _elseTag,
    whenOrigins: _whenOrigins,
    onOrigin: _onOrigin,
    ...fields
  } = desc;
  return fields;
}

/** Parse node `g` of a flow whose body is `body`, as `parseNodes` does. */
function parseCase(body: string): CaseDescriptor {
  const indented = body
    .split("\n")
    .map((line) => (line ? `    ${line}` : line))
    .join("\n");
  const text = `nodes:\n  g:\n${indented}`;
  return parseNodes(SourceDocument.parse(text).root.child("nodes")).get("g") as CaseDescriptor;
}

test("case parse matches the descriptor and records ranges", () => {
  const desc = CaseDescriptor.parse("route", readerFor(FLOW, "route"));

  expect(desc).toBeInstanceOf(CaseDescriptor);
  expect(fieldsOf(desc)).toEqual(
    fieldsOf(
      new CaseDescriptor({
        id: "route",
        on: "${score.output}",
        cases: [{ when: "high", then: "hot" }],
        else_: "cold",
      }),
    ),
  );
  expect(desc.range).toBeInstanceOf(SourceRange);
  expect(desc.fieldRanges.get(siteKey("field", "on"))).toBeInstanceOf(SourceRange);
  expect(desc.fieldRanges.get(siteKey("field", "cases"))).toBeInstanceOf(SourceRange);
  expect(desc.fieldRanges.get(siteKey("field", "else"))).toBeInstanceOf(SourceRange);
});

test("case build lowers to a case node with edges", () => {
  const desc = CaseDescriptor.parse("route", readerFor(FLOW, "route"));

  const result = desc.build(CTX);

  // The lowering is the build: a CaseNode and the flow-owned `__on` wiring.
  expect(result.node).toBeInstanceOf(CaseNode);
  expect(result.node.id).toBe("route");
  expect(result.wiring).toEqual(new Map([["__on", "${score.output}"]]));
  // The edges are the data edges (score to route on `__on`), then the control edges.
  const data = result.edges.filter((edge) => edge.sourceHandle === null);
  const control = result.edges.filter((edge) => edge.sourceHandle !== null);
  expect(data.map((edge) => [edge.from, edge.to, edge.inputGroup])).toContainEqual(["score", "route", "__on"]);
  const ctl = control.map((edge) => [edge.from, edge.to, edge.sourceHandle]);
  expect(ctl).toContainEqual(["route", "hot", "hot"]);
  expect(ctl).toContainEqual(["route", "cold", ELSE_HANDLE]);
  // The YAML coordinates stay on the Compose-only build result.
  expect(result.nodeRange).toBeInstanceOf(SourceRange);
  // The generated `__on` input locates at the `on:` value and is labelled `on:`.
  expect(result.fieldRanges.get(siteKey("input", "__on"))).toEqual(readerFor(FLOW, "route").rangeOf("on")!);
  expect(result.node.wiringLabels.get("__on")).toBe("on:");
});

test("an on case entry with a null when is refused at that when", () => {
  const text =
    "nodes:\n  route:\n    kind: case\n    on: ${score.output}\n    cases:\n      - when: null\n        then: hot\n";

  const cursor = readerFor(text, "route");
  const ctx = new BuildContext({ registry: new Map(), diagnostics: LoadDiagnostics.fromCursor(cursor) });

  const error = loadError(() => CaseDescriptor.parse("route", cursor).build(ctx));

  expect(error.message).toBe("case node 'route': each on:-form case needs a `when:` value and a `then:`");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(cursor.child("cases").child(0).child("when").range!);
});

test("case parse requires cases", () => {
  const flow = "nodes:\n  route:\n    kind: case\n    on: ${x.output}\n";

  const cursor = readerFor(flow, "route");

  const error = loadError(() => CaseDescriptor.parse("route", cursor));

  expect(error.message).toBe("node 'route' (kind=case): missing required field 'cases'");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(new SourceRange(3, 5, 5, 1));
});

test("a case with both on spellings is a located error", () => {
  // Plain `on:` and quoted `"on":` are the same text key: a repeated key.
  const flow =
    "id: c\nname: c\nnodes:\n  route:\n    kind: case\n    on: ${x.output}\n" +
    '    "on": ${y.output}\n    cases:\n      - when: a\n        then: hot\n';

  const error = loadError(() => parseSourceDocument(flow, "flow.yaml"));

  expect(error.message).toContain("duplicate key 'on' (first at line 6)");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(new SourceRange(7, 5, 7, 9));
});

test("case rejects a boolean key at its location", () => {
  // `true:` is the boolean true under YAML 1.2, not a spelling of `on:`.
  const flow = `id: c
name: c
nodes:
  s:
    kind: code
    code: 'return "a"'
    output: str
  route:
    kind: case
    true: \${s.output}
    cases:
      - when: "\${s.output} == 'a'"
        then: hot
    else: hot
output: \${hot.output}
`;

  const error = loadError(() => parseSourceDocument(flow, "flow.yaml"));

  expect(error.message).toContain("the key `true` was read by YAML as a boolean");
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(10);
});

test("case descriptor rejects misaligned or non-opaque origins", () => {
  // Programmatic descriptors cannot detach provenance from ordered cases.
  const cases = [{ when: "true", then: "positive" }];
  const misaligned = () =>
    new CaseDescriptor({ id: "gate", cases, whenOrigins: [new OriginRef("first"), new OriginRef("second")] });
  expect(misaligned).toThrow(RangeError);
  expect(misaligned).toThrow("CASE predicate origins must align with cases");

  const opaque = () =>
    new CaseDescriptor({ id: "gate", cases, whenOrigins: ["flow.yaml:12"] as unknown as OriginRef[] });
  expect(opaque).toThrow(TypeError);
  expect(opaque).toThrow("CASE predicate origin must be an OriginRef or null");

  expect(() => new CaseDescriptor({ id: "gate", cases, onOrigin: "flow.yaml:4" as unknown as OriginRef })).toThrow(
    new TypeError("CASE on origin must be an OriginRef or null"),
  );
  // The cases list stays the one given, so a later push is the build's to catch.
  const pushed = [{ when: "true", then: "positive" }];
  const desc = new CaseDescriptor({ id: "gate", cases: pushed, whenOrigins: [new OriginRef("first")] });
  expect(desc.cases).toBe(pushed);
  pushed.push({ when: "false", then: "negative" });
  expect(() => desc.build(CTX)).toThrow(new RangeError("CASE predicate origins must align with cases"));
  // The origins are a frozen copy (Python's tuple): a later push to the given array can't reach them.
  const given = [new OriginRef("first")];
  const copied = new CaseDescriptor({ id: "gate", cases, whenOrigins: given });
  given.push(new OriginRef("second"));
  expect(copied.whenOrigins).toStrictEqual([new OriginRef("first")]);
  expect(Object.isFrozen(copied.whenOrigins)).toBe(true);
});

test("case parse keeps then styles aligned with cases", () => {
  const desc = parseCase(`kind: case
cases:
  - when: "a.output == 1"
    then: reply
  - when: "a.output == 2"
    then: "reply"
  - when: "a.output == 3"
    then: 'r'
  - when: "a.output == 4"
    then: |
      some text
  - when: "a.output == 5"
    then: 3
  - when: "a.output == 6"
    then:
      a: 1
`);

  expect(desc.thenStyles).toEqual(["plain", "double", "single", "literal", "plain", null]);
});

test("case parse keeps the else style", () => {
  const head = 'kind: case\ncases:\n  - when: "${a.output}"\n    then: b\n';

  expect(parseCase(`${head}else: reply\n`).elseStyle).toBe("plain");
  expect(parseCase(`${head}else: "approve"\n`).elseStyle).toBe("double");
  expect(parseCase(head).elseStyle).toBeNull();
});

test("case parse tells else null from no else", () => {
  const head = 'kind: case\ncases:\n  - when: "${a.output}"\n    then: b\n';
  const nullElse = parseCase(`${head}else: null\n`);

  expect(nullElse.else_).toBeNull();
  expect(nullElse.hasElse).toBe(true);
  expect(parseCase(head).hasElse).toBe(false);
  expect(parseCase(`${head}else: reply\n`).hasElse).toBe(true);
});

test("case descriptor constructed with else has else", () => {
  expect(new CaseDescriptor({ id: "g", cases: [{ when: "a", then: "b" }], else_: "b" }).hasElse).toBe(true);
  expect(new CaseDescriptor({ id: "g", cases: [{ when: "a", then: "b" }] }).hasElse).toBe(false);
});

test("case parse keeps a then null key", () => {
  const desc = parseCase('kind: case\ncases:\n  - when: "${a.output}"\n    then: null\n');

  expect(Object.hasOwn(desc.cases[0]!, "then")).toBe(true);
  expect(desc.cases[0]!["then"]).toBeNull();
  expect(desc.thenStyles).toEqual(["plain"]);
});

test("case parse on form keeps styles", () => {
  const desc = parseCase(`kind: case
on: \${a.output}
cases:
  - when: x
    then: "b"
else: 'c'
`);

  expect(desc.on).toBe("${a.output}");
  expect(desc.thenStyles).toEqual(["double"]);
  expect(desc.elseStyle).toBe("single");
});

test("a call target's then range is its own line", () => {
  const text =
    "id: f\nname: f\nnodes:\n" +
    "  route:\n    kind: case\n    on: ${score.output}\n" +
    "    cases:\n      - when: high\n        then: call(helper, n=3)\n" +
    "    else: cold\n" +
    "output: ${route.output}\n";

  const host = parseNodes(SourceDocument.parse(text).root.child("nodes")).get("route") as CaseDescriptor;

  // The `then:` value sits on line 9, not on the case header.
  expect(host.thenRanges[0]!.startLine).toBe(9);
  expect(host.thenRanges[0]).not.toEqual(host.range);
});

const SHAPE_HEAD = "id: f\nname: f\ninput:\n  n: !int\nnodes:\n  gate:\n    kind: case\n";
const SHAPE_TAIL = "  done:\n    kind: code\n    code: return 1\n    output: int\noutput: ${done.output}\n";

/** The `cases:` value's range in node `gate`. */
const casesValue = (gate: SourceCursor) => gate.rangeOf("cases");
/** The first `cases:` item's range in node `gate`. */
const firstItem = (gate: SourceCursor) => gate.child("cases").child(0).range;

test.each([
  [
    '    cases: {when: "input.n > 1", then: done}\n',
    '`cases:` must be a list of {when:, then:} mappings (got {"when":"input.n > 1","then":"done"}); ' +
      "start each case with `- `",
    casesValue,
  ],
  ["    cases: done\n", '`cases:` must be a list of {when:, then:} mappings (got "done")', casesValue],
  ["    cases: null\n", "`cases:` must be a list of {when:, then:} mappings (got null)", casesValue],
  ["    cases: 7\n", "`cases:` must be a list of {when:, then:} mappings (got 7)", casesValue],
  ["    cases: [done]\n", '`cases[0]` must be a {when:, then:} mapping (got "done")', firstItem],
])("case rejects malformed cases at their location: %s", (body, message, rangeIn) => {
  const text = SHAPE_HEAD + body + SHAPE_TAIL;
  const error = loadError(() => nodesOf(text));

  expect(error.message).toBe(`case node 'gate': ${message}`);
  const range = rangeIn(readerFor(text, "gate"))!;
  expect(range.startLine).toBe(8);
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(range);
});

test("searched case origins are positional", () => {
  // Duplicate predicate text stays distinct: each site is its index.
  const document = SourceDocument.parse(SEARCHED_ORIGIN_FLOW, "searched-case.yaml");
  const cursor = document.root.child("nodes").child("gate");
  const builder = new SourceCatalogBuilder();
  const desc = CaseDescriptor.parse("gate", cursor);
  expect(desc.whenOrigins).toEqual([]);

  const stamped = desc.withSourceOrigins(cursor, { sitePath: ["root", "node", "gate"], catalogBuilder: builder });

  expect(stamped).toBeInstanceOf(CaseDescriptor);
  const origins = stamped.whenOrigins;
  expect(origins).toHaveLength(3);
  expect(origins.every((origin) => origin instanceof OriginRef)).toBe(true);
  expect(new Set(origins.map((origin) => origin!.key)).size).toBe(3);
  expect(origins.map((origin) => builder.resolveRegistered(origin!)!.range)).toEqual(
    [0, 1, 2].map((index) => cursor.child("cases").child(index).child("when").range!),
  );
  expect(stamped.onOrigin).toBeNull();
});

test("searched case build keeps positional predicate origins", () => {
  // Duplicate predicate text stays distinct, and the generated `__r0` binding carries case 0's.
  const label = "searched-case.yaml";
  const cursor = SourceDocument.parse(SEARCHED_ORIGIN_FLOW, label).root.child("nodes").child("gate");
  const { flow, builder } = assembleText(SEARCHED_ORIGIN_FLOW, label);
  const node = flow.compiled.nodes.get("gate") as CaseNode;

  const origins = node.cases.map((branch) => branch.origin);
  expect(origins.every((origin) => origin instanceof OriginRef)).toBe(true);
  expect(new Set(origins.map((origin) => origin!.key)).size).toBe(3);
  expect(origins.map((origin) => builder.resolveRegistered(origin!)!.range)).toEqual(
    [0, 1, 2].map((index) => cursor.child("cases").child(index).child("when").range!),
  );
  const binding = flow.compiled.wiring.get("gate")!.get("__r0")!;
  expect(binding).toBeInstanceOf(Binding);
  expect(binding.origin).toBe(origins[0]!);
});

test("source-free case build keeps predicates and wiring origin-free", () => {
  // A directly constructed descriptor builds raw wiring text, no `Binding`.
  const desc = new CaseDescriptor({ id: "gate", cases: [{ when: "score.output > 0", then: "positive" }] });

  const result = desc.build(CTX);

  expect((result.node as CaseNode).cases[0]!.origin).toBeNull();
  expect(result.wiring).toEqual(new Map([["__r0", "${score.output}"]]));
});

test("on case predicate and on origins are distinct", () => {
  // The generated `__on` binding belongs to `on:`, not to a case predicate.
  const document = SourceDocument.parse(ON_ORIGIN_FLOW, "on-case.yaml");
  const cursor = document.root.child("nodes").child("route");
  const builder = new SourceCatalogBuilder();

  const stamped = CaseDescriptor.parse("route", cursor).withSourceOrigins(cursor, {
    sitePath: ["root", "node", "route"],
    catalogBuilder: builder,
  });

  const predicateOrigins = stamped.whenOrigins;
  expect(predicateOrigins.every((origin) => origin instanceof OriginRef)).toBe(true);
  expect(predicateOrigins.map((origin) => builder.resolveRegistered(origin!)!.range)).toEqual(
    [0, 1].map((index) => cursor.child("cases").child(index).child("when").range!),
  );
  const onOrigin = stamped.onOrigin!;
  expect(onOrigin).toBeInstanceOf(OriginRef);
  expect(predicateOrigins.map((origin) => origin!.key)).not.toContain(onOrigin.key);
  expect(builder.resolveRegistered(onOrigin)!.range).toEqual(cursor.child("on").range!);
});

test("on case build keeps the predicate and generated binding origins", () => {
  // The generated `__on` binding belongs to `on:`, not to a case predicate.
  const label = "on-case.yaml";
  const cursor = SourceDocument.parse(ON_ORIGIN_FLOW, label).root.child("nodes").child("route");
  const { flow, builder } = assembleText(ON_ORIGIN_FLOW, label);
  const node = flow.compiled.nodes.get("route") as CaseNode;

  const predicateOrigins = node.cases.map((branch) => branch.origin);
  expect(predicateOrigins.every((origin) => origin instanceof OriginRef)).toBe(true);
  expect(predicateOrigins.map((origin) => builder.resolveRegistered(origin!)!.range)).toEqual(
    [0, 1].map((index) => cursor.child("cases").child(index).child("when").range!),
  );
  const onBinding = flow.compiled.wiring.get("route")!.get("__on")!;
  expect(onBinding).toBeInstanceOf(Binding);
  expect(onBinding.origin).toBeInstanceOf(OriginRef);
  expect(predicateOrigins.map((origin) => origin!.key)).not.toContain(onBinding.origin!.key);
  expect(builder.resolveRegistered(onBinding.origin!)!.range).toEqual(cursor.child("on").range!);
});

test("a lone surrogate in on is refused at the value", () => {
  const text = 'nodes:\n  r:\n    kind: case\n    on: "\\ud800"\n    cases:\n      - when: a\n        then: b\n';

  const error = loadError(() => CaseDescriptor.parse("r", readerFor(text, "r")));

  expect(error.message).toBe("node 'r' field 'on': text holding a lone surrogate is not a yian value");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(readerFor(text, "r").rangeOf("on")!);
});

test("a lone surrogate in a searched when is refused at that when", () => {
  const text =
    "nodes:\n  g:\n    kind: case\n    cases:\n      - when: a.output\n        then: b\n" +
    '      - when: "\\ud800"\n        then: c\n';

  const error = loadError(() => CaseDescriptor.parse("g", readerFor(text, "g")));

  expect(error.message).toBe("node 'g' field 'cases[1].when': text holding a lone surrogate is not a yian value");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(
    readerFor(text, "g").child("cases").child(1).child("when").range!,
  );
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  // tests/compose/test_kind_case_parse.py
  { id: "load-case/compose_kind_case_parse__a_single_case_mapping_asks_for_a_list_item", call: parsedNodes },
  { id: "load-case/compose_kind_case_parse__cases_that_is_not_a_list_is_located", call: parsedNodes },
  { id: "load-case/compose_kind_case_parse__a_case_item_that_is_not_a_mapping_is_located", call: parsedNodes },
])("parseNodes: $id", runCase);

test.each<CaseRow>([
  // tests/compose/test_expression_nesting.py
  { id: "load-core/compose_expression_nesting__thousands_of_nested_spans", call: assembled },
  { id: "load-core/compose_expression_nesting__too_deep_searched_when", call: assembled },
])("assembled: $id", runCase);
