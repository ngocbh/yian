/** The `loop` kind's parse. */

import { expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { LoadDiagnostics } from "../../src/compose/helpers/errors.ts";
import { BuildContext, siteKey } from "../../src/compose/nodes/base.ts";
import { LoopDescriptor } from "../../src/compose/nodes/loop.ts";
import { type SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import {
  assembledWith,
  type CaseRow,
  caseText,
  handChild,
  loadError,
  nodesOf,
  parametersFromYaml,
  parsedNodes,
  runCase,
} from "../helpers/declarations.ts";

const FLOW = `nodes:
  drive:
    kind: loop
    call: body_flow
    inputs:
      acc: \${input.seed}
    while: \${carry.go}
    max: 10
`;

/** Return the cursor over node `nodeId` of `text`'s `nodes:`. */
function readerFor(text: string, nodeId: string): SourceCursor {
  return SourceDocument.parse(text).root.child("nodes").child(nodeId);
}

/** Return a descriptor's fields, leaving out the parse metadata. */
function fieldsOf(desc: LoopDescriptor): Record<string, unknown> {
  const {
    range: _range,
    fieldRanges: _fieldRanges,
    assertRanges: _assertRanges,
    childLabel: _childLabel,
    ...fields
  } = desc;
  return fields;
}

test("loop parse matches the descriptor and records ranges", () => {
  const desc = LoopDescriptor.parse("drive", readerFor(FLOW, "drive"));

  // `while:` lands in the `while_` field.
  expect(desc).toBeInstanceOf(LoopDescriptor);
  expect(fieldsOf(desc)).toEqual(
    fieldsOf(
      new LoopDescriptor({
        id: "drive",
        call: "body_flow",
        inputs: { acc: "${input.seed}" },
        while_: "${carry.go}",
        max: 10n,
      }),
    ),
  );
  expect(desc.kind).toBe("loop");
  expect(desc.range).toBeInstanceOf(SourceRange);
  expect(desc.fieldRanges.get(siteKey("field", "call"))).toBeInstanceOf(SourceRange);
  expect(desc.fieldRanges.get(siteKey("field", "inputs"))).toBeInstanceOf(SourceRange);
  // The range is keyed by the YAML key (`while`), not the field name (`while_`).
  expect(desc.fieldRanges.get(siteKey("field", "while"))).toBeInstanceOf(SourceRange);
  expect(desc.fieldRanges.get(siteKey("field", "max"))).toBeInstanceOf(SourceRange);
});

test("loop parse requires call", () => {
  const cursor = readerFor("nodes:\n  drive:\n    kind: loop\n    times: 3\n", "drive");

  const error = loadError(() => LoopDescriptor.parse("drive", cursor));

  expect(error.message).toBe("node 'drive' (kind=loop): missing required field 'call'");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(cursor.range!);
});

test("loop parse maps the until keyword", () => {
  const flow = "nodes:\n  drive:\n    kind: loop\n    call: body_flow\n    until: ${carry.done}\n    max: 5\n";

  const desc = LoopDescriptor.parse("drive", readerFor(flow, "drive"));

  expect(desc.until_).toBe("${carry.done}");
  expect(desc.while_).toBeNull();
  expect(desc.fieldRanges.get(siteKey("field", "until"))).toBeInstanceOf(SourceRange);
});

const PREDICATE_FLOW = `id: f
name: f
input:
  n: !int
defs:
  body:
    input:
      x: !int
    nodes:
      inc:
        kind: code
        code: return inputs["x"] + 1n
        input:
          x: \${input.x}
        output: int
    output:
      x: \${inc.output}
nodes:
  lp:
    kind: loop
    call: body
    input:
      x: \${input.n}
    PREDICATE
    max: 3
output: \${lp.output}
`;

test.each([
  [
    "while: true",
    "`while:` must be an expression string over the carried record, such as `while: not done` (got true)",
  ],
  [
    "while: false",
    "`while:` must be an expression string over the carried record, such as `while: not done` (got false)",
  ],
  ["until: 1", "`until:` must be an expression string over the carried record, such as `until: done` (got 1)"],
  ["until: [x]", '`until:` must be an expression string over the carried record, such as `until: done` (got ["x"])'],
])("a non-string predicate is refused at its field: %s", (predicate, message) => {
  const error = loadError(() => nodesOf(PREDICATE_FLOW.replace("PREDICATE", predicate)));

  expect(error.message).toBe(`loop node 'lp': ${message}`);
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(24);
});

test("a lone surrogate in while is refused at the value", () => {
  const text = 'nodes:\n  lp:\n    kind: loop\n    call: body\n    while: "\\ud800"\n';

  const error = loadError(() => LoopDescriptor.parse("lp", readerFor(text, "lp")));

  expect(error.message).toBe("node 'lp' field 'while': text holding a lone surrogate is not a yian value");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(readerFor(text, "lp").rangeOf("while")!);
});

/** Parse loop `nodeId` of `text` and build it over a resolver returning `child`. */
function buildLoop(text: string, nodeId: string, child: ComposedFlow) {
  const desc = nodesOf(text).get(nodeId)!;
  return desc.build(
    new BuildContext({ registry: new Map(), resolver: () => child, diagnostics: new LoadDiagnostics("<flow>", text) }),
  );
}

test("an empty predicate is refused at load", () => {
  // `while: ''` is falsy but present: it fails to parse at load, not at run time. The `body` def
  // reads `x` and returns `{x}`, so no field-name error joins the parse error and it is at the field.
  const body = handChild({
    inputs: parametersFromYaml("input:\n  x: !int\n", new Map()),
    producers: { inc: new ValueContract(Type.scalar(ValueKind.INTEGER)) },
    output: { x: "${inc.output}" },
  });

  const error = loadError(() => buildLoop(PREDICATE_FLOW.replace("PREDICATE", "while: ''"), "lp", body));

  expect(error.message).toBe(
    "loop node 'lp': `while:` is not a valid boolean expression: could not parse expression '': " +
      "unexpected end of expression at line 1, column 1",
  );
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(24);
});

test("a malformed predicate is reported at its field", () => {
  const seed = caseText("load-loop/seed_28_refine_loop").text;
  expect(seed).toContain("    until: score >= 0.8");
  const text = seed.replace("    until: score >= 0.8", "    until: score >= >= 1");
  // The `refine_turn` def reads `{draft, score}` and its `critic` returns them, as the seed's does.
  const critic = new ValueContract(typeFor("Rec", readTypedefs({ Rec: { draft: "str", score: "float" } })));
  const body = handChild({
    inputs: parametersFromYaml("input:\n  draft: !str\n  score: !float\n", new Map()),
    producers: { critic },
    output: "${critic.output}",
  });

  const error = loadError(() => buildLoop(text, "polish", body));

  expect(error.message).toBe(
    "loop node 'polish': `until:` is not a valid boolean expression: could not parse expression " +
      "'score >= >= 1': unexpected '>=' at line 1, column 10",
  );
  const line = text.split("\n").findIndex((row) => row.startsWith("    until:")) + 1;
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(line);
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  { id: "load-loop/compose_kind_loop_parse__a_list_until_predicate_is_rejected", call: parsedNodes },
  { id: "load-loop/compose_kind_loop_parse__a_true_while_predicate_is_rejected", call: parsedNodes },
])("parseNodes: $id", runCase);

test("assembled: load-core/compose_expression_nesting__too_deep_loop_until", () => {
  // The `body` def reads `n` and its `step` returns `{n}`.
  const step = new ValueContract(typeFor("Rec", readTypedefs({ Rec: { n: "int" } })));
  const body = handChild({
    inputs: parametersFromYaml("input:\n  n: !int\n", new Map()),
    producers: { step },
    output: "${step.output}",
  });
  runCase({ id: "load-core/compose_expression_nesting__too_deep_loop_until", call: assembledWith({ body }) });
});
