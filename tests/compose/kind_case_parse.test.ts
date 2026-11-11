/** The `case` kind's parse and build rows. */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { expect, test } from "bun:test";

import { cloneChild } from "../../src/compile/expand.ts";
import { flowFingerprint } from "../../src/compile/manifest.ts";
import type { CompiledFlow, Flow } from "../../src/compile/model.ts";
import { LoadDiagnostics } from "../../src/compose/helpers/errors.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { BuildContext, siteKey } from "../../src/compose/nodes/base.ts";
import { CaseDescriptor } from "../../src/compose/nodes/case.ts";
import { parseSourceDocument } from "../../src/compose/parse/file.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { SourceCatalogBuilder, scopeOrigin } from "../../src/compose/source/catalog.ts";
import { type SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { OriginRef, SourceRange } from "../../src/diagnostics.ts";
import { ELSE_HANDLE } from "../../src/nodes/base.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import { CaseNode } from "../../src/nodes/case/node.ts";
import { Binding } from "../../src/nodes/contracts.ts";
import { runFlow } from "../../src/run.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { assembled, assembleText, type CaseRow, loadError, nodesOf, parsedNodes, runCase } from "../helpers/declarations.ts";

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

/** Return a descriptor's fields, leaving out the parse metadata. */
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
    thenIncludes: _thenIncludes,
    elseInclude: _elseInclude,
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
  // The origins are a frozen copy: a later push to the given array can't reach them.
  const given = [new OriginRef("first")];
  const copied = new CaseDescriptor({ id: "gate", cases, whenOrigins: given });
  given.push(new OriginRef("second"));
  expect(copied.whenOrigins).toStrictEqual([new OriginRef("first")]);
  expect(Object.isFrozen(copied.whenOrigins)).toBe(true);
});

test.each([
  [
    "[{then: done}]",
    "case node 'gate' then: `done` names no node. For the text, add a value entry and name it: " +
      "`done_text: !str done`",
  ],
  [
    "[{when: 1}]",
    "flow outputs: ${gate.output} — case 'gate' has no resolvable branch value (a case needs a " +
      "`then:` or `else:`, and its targets must not form a cycle)",
  ],
])("source attachment does not preempt malformed case validation: %s", (malformedCases, message) => {
  // The origin hook stays inert for malformed shapes until the normal phases.
  const text =
    "id: malformed-case\n" +
    "name: malformed-case\n" +
    "nodes:\n" +
    "  gate:\n" +
    "    kind: case\n" +
    `    cases: ${malformedCases}\n` +
    "output: ${gate.output}\n";

  expect(loadError(() => assembleText(text)).message).toBe(message);
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
  [
    "    cases: []\n",
    "`cases:` is empty and there is no `else:`, so it routes nowhere; add a case or an `else:`",
    casesValue,
  ],
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

test.each([
  ["${input.v}", ", as in `when: input.k == input.v`"],
  ["v-${input.v}", ""],
  ["${input.v", ""],
])("an on-form when holding a span is refused at that when: %s", (when, example) => {
  const text =
    "nodes:\n  r:\n    kind: case\n    on: ${input.k}\n    cases:\n      - when: a\n        then: b\n" +
    `      - when: "${when}"\n        then: c\n`;

  const error = loadError(() => CaseDescriptor.parse("r", readerFor(text, "r")));

  expect(error.message).toBe(
    `case node 'r': an on: case's \`when:\` is a literal, so \`when: ${when}\` would match that text ` +
      `itself; to compare with a value, drop \`on:\` and write each \`when:\` as a condition${example}`,
  );
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(
    readerFor(text, "r").child("cases").child(1).child("when").range!,
  );
});

test("an on-form when may hold an escaped dollar, and a searched when a span", () => {
  const onForm =
    'nodes:\n  r:\n    kind: case\n    on: ${input.k}\n    cases:\n      - when: "$${x}"\n        then: b\n';
  const searched = "nodes:\n  g:\n    kind: case\n    cases:\n      - when: ${a.output > 1}\n        then: b\n";

  expect(CaseDescriptor.parse("r", readerFor(onForm, "r")).cases[0]!["when"]).toBe("$${x}");
  expect(CaseDescriptor.parse("g", readerFor(searched, "g")).cases[0]!["when"]).toBe("${a.output > 1}");
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

// ---------- loaded origins ----------

/** The `__r0` binding of node `nodeId` in `flow`'s wiring. */
const r0Of = (flow: Flow, nodeId: string) => flow.wiring.get(nodeId)!.get("__r0") as Binding;

test("external alias rebases case and generated binding origins", async () => {
  // An external view scopes CASE carriers without mutating its canonical child.
  const child = await loadFlow(SEARCHED_ORIGIN_FLOW, { sourceLabel: "child-case.yaml" });
  const parent = `id: parent-case-origin
name: parent-case-origin
nodes:
  invoke:
    kind: call
    call: child_view
output: \${invoke.output}
uses:
  child_view: child
`;
  const loaded = await loadFlow(parent, { childResolver: () => child });
  const canonical = child.compiled.nodes.get("gate") as CaseNode;
  const embeddedFlow = (loaded.compiled.nodes.get("invoke") as CallNode).child as CompiledFlow;
  const embedded = embeddedFlow.nodes.get("gate") as CaseNode;

  expect(embedded.cases.map((branch) => branch.origin)).toEqual(
    canonical.cases.map((branch) => scopeOrigin("child_view", branch.origin!)),
  );
  expect(r0Of(embeddedFlow, "gate").origin).toEqual(scopeOrigin("child_view", r0Of(child.compiled, "gate").origin!));
  const resolver = loaded.originResolver!;
  const childResolver = child.originResolver!;
  expect(resolver).not.toBeNull();
  expect(childResolver).not.toBeNull();
  expect(embedded.cases.map((branch) => resolver.resolve(branch.origin!))).toEqual(
    canonical.cases.map((branch) => childResolver.resolve(branch.origin!)),
  );
});

test("generated child clone preserves case predicate origins", async () => {
  // CALL cloning keeps ordered CASE sites and generated binding provenance.
  const loaded = await loadFlow(SEARCHED_ORIGIN_FLOW);
  const original = loaded.compiled.nodes.get("gate") as CaseNode;

  const cloned = cloneChild(loaded.compiled, "invoke", {});
  const generated = cloned.nodes.get("invoke/gate") as CaseNode;

  expect(generated).not.toBe(original);
  expect(generated.cases.map((branch) => branch.origin)).toEqual(original.cases.map((branch) => branch.origin));
  expect(r0Of(cloned, "invoke/gate").origin).toEqual(r0Of(loaded.compiled, "gate").origin);
});

test("formatting moves case ranges without changing keys or fingerprint", async () => {
  // Semantic CASE identities survive comments while resolution follows the text.
  const moved = SEARCHED_ORIGIN_FLOW.replace(
    "  gate:\n    kind: case\n",
    "  # Formatting-only comments move the CASE.\n  gate:\n    kind: case\n",
  ).replace("    cases:\n", "    # The ordered predicates keep their semantic indices.\n    cases:\n");
  const baseline = await loadFlow(SEARCHED_ORIGIN_FLOW);
  const relocated = await loadFlow(moved);
  const baselineOrigins = (baseline.compiled.nodes.get("gate") as CaseNode).cases.map((branch) => branch.origin!);
  const relocatedOrigins = (relocated.compiled.nodes.get("gate") as CaseNode).cases.map((branch) => branch.origin!);

  expect(relocatedOrigins).toEqual(baselineOrigins);
  expect(flowFingerprint(relocated.compiled)).toBe(flowFingerprint(baseline.compiled));
  expect(baseline.originResolver).not.toBeNull();
  expect(relocated.originResolver).not.toBeNull();
  expect(baselineOrigins.map((origin) => baseline.originResolver!.resolve(origin)!.range)).not.toEqual(
    relocatedOrigins.map((origin) => relocated.originResolver!.resolve(origin)!.range),
  );
});

// ---------- a failing arm in a child, run ----------

/** A CASE whose second `when:` divides by zero. */
const CASE_CHILD_BODY = `nodes:
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
    else: negative
  positive:
    kind: code
    code: 'return "positive"'
    output: str
  broken:
    kind: code
    code: 'return "broken"'
    output: str
  negative:
    kind: code
    code: 'return "negative"'
    output: str
output: \${positive.output | broken.output | negative.output}
`;

test("an inline definition's case failure keeps the child predicate origin", async () => {
  // Inline child assembly and CALL growth keep the exact failing arm.
  const body = CASE_CHILD_BODY.split("\n")
    .map((line) => (line === "" ? line : `    ${line}`))
    .join("\n");
  const text = "id: inline-case-origin\nname: inline-case-origin\nnodes:\n  invoke:\n    kind: call\n" + body + "output: ${invoke.output}\n";
  const expected = SourceDocument.parse(text, "inline-case.yaml")
    .root.child("nodes")
    .child("invoke")
    .child("nodes")
    .child("gate")
    .child("cases")
    .child(1)
    .child("when");
  const flow = await loadFlow(text, { sourceLabel: "inline-case.yaml" });
  const child = (flow.compiled.nodes.get("invoke") as CallNode).child as CompiledFlow;
  const caseNode = child.nodes.get("gate") as CaseNode;
  const dir = mkdtempSync(join(tmpdir(), "yian-kind-case-parse-"));
  try {
    const result = await runFlow(flow, {}, { env: hostEnv(dir) });

    expect(result.failure).not.toBeNull();
    expect(result.failure!.trace.at(-1)).toEqual(caseNode.cases[1]!.origin!);
    expect(flow.diagnose(result.failure!).frames.at(-1)!.range).toEqual(expected.range!);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  { id: "load-case/compose_kind_case_parse__a_single_case_mapping_asks_for_a_list_item", call: parsedNodes },
  { id: "load-case/compose_kind_case_parse__cases_that_is_not_a_list_is_located", call: parsedNodes },
  { id: "load-case/compose_kind_case_parse__a_case_item_that_is_not_a_mapping_is_located", call: parsedNodes },
])("parseNodes: $id", runCase);

test.each<CaseRow>([
  { id: "load-case/compose_kind_case_parse__a_case_with_no_branch_has_no_value", call: assembled },
  { id: "load-case/compose_kind_case_parse__a_flow_style_case_names_no_node", call: assembled },
  { id: "load-core/compose_expression_nesting__thousands_of_nested_spans", call: assembled },
  { id: "load-core/compose_expression_nesting__too_deep_searched_when", call: assembled },
])("assembled: $id", runCase);
