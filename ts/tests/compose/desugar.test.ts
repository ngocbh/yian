/**
 * The desugar phase: flow-level rewrites that run before the build loop. A port of
 * `tests/compose/test_desugar.py`.
 *
 * Each desugar pass rewrites a flow's parsed descriptors into ordinary descriptors and bindings. A
 * synth node a pass mints locates at the SITE it desugared from (the `call(...)` directive's own
 * binding line, not the host node header), so its downstream errors caret the author's actual
 * source. The exception is `adaptive_questions`, whose synth composer genuinely SHARES the gate's
 * inputs verbatim and so inherits the gate's `range`/`fieldRanges`, except that the brief's range
 * replaces the gate's `prompt:` range.
 *
 * Ported: the inline-call rows (`test_desugar_inline_calls_locate_at_their_binding_site`,
 * `test_inline_call_error_uses_binding_range`, `test_inline_output_error_uses_section_range`,
 * `test_inline_assert_error_uses_positional_range`,
 * `test_duplicate_inline_assertions_keep_positional_ranges`,
 * `test_inline_map_scalar_over_does_not_use_input_named_over_range`), the adaptive-questions
 * rows (`test_desugar_adaptive_questions_preserves_ranges`,
 * `test_desugar_adaptive_questions_without_source`) and the case rows
 * (`test_desugar_case_call_targets_locate_at_their_target_line`,
 * `test_case_binding_error_uses_binding_range`, `test_case_output_error_uses_section_range`,
 * `test_case_condition_error_uses_exact_condition_range`,
 * `test_case_assert_error_uses_positional_range`,
 * `test_case_map_scalar_over_does_not_use_input_named_over_range`).
 */

import { expect, test } from "bun:test";

import { desugarAdaptiveQuestions } from "../../src/compose/desugar/adaptive_questions.ts";
import { expandCaseOutputs } from "../../src/compose/desugar/case_outputs.ts";
import { desugarCaseCallTargets } from "../../src/compose/desugar/case_targets.ts";
import { desugarInlineCalls, synthIdGen } from "../../src/compose/desugar/inline_calls.ts";
import { LoadDiagnostics, type LoadError } from "../../src/compose/helpers/errors.ts";
import { type NodeDescriptor, siteKey } from "../../src/compose/nodes/base.ts";
import type { CallDescriptor } from "../../src/compose/nodes/call.ts";
import { CaseDescriptor } from "../../src/compose/nodes/case.ts";
import { CodeDescriptor } from "../../src/compose/nodes/code.ts";
import { HumanInputDescriptor } from "../../src/compose/nodes/human_input.ts";
import { MapDescriptor } from "../../src/compose/nodes/map.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { type SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { OriginRef, SourceRange } from "../../src/diagnostics.ts";
import { Assertion } from "../../src/nodes/contracts.ts";
import { loadError } from "./declarations.ts";

/** Return the physical cursor over a flow document's `nodes:` mapping. */
function nodesReader(text: string): SourceCursor {
  return SourceDocument.parse(text).root.child("nodes");
}

/** Return source ownership for direct range-only desugar tests. */
function diagnostics(): LoadDiagnostics {
  return new LoadDiagnostics("flow.yaml", (" ".repeat(100) + "\n").repeat(200));
}

/** Return the error's one frame. */
function onlyFrame(error: LoadError) {
  expect(error.diagnostic.frames).toHaveLength(1);
  return error.diagnostic.frames[0]!;
}

test("desugar inline calls locate at their binding site", () => {
  // A whole-value `call(...)` in a node's `input:` binding lifts to a synth `call` node, which
  // locates at its OWN binding line (the `x:` entry the directive was written on), NOT the host
  // node header, and carries an empty `fieldRanges` (it owns no authored fields), so a bad ref
  // inside the directive carets the directive.
  const text =
    "id: f\nname: f\nnodes:\n" +
    "  writer:\n    kind: code\n    code: m:f\n" +
    "    input:\n      x: call(helper, n=3)\n    output: str\n" +
    "output: ${writer.output}\n";
  const descriptors = parseNodes(nodesReader(text));
  const host = descriptors.get("writer")!;
  const [newDescriptors] = desugarInlineCalls(descriptors, null, { nextId: synthIdGen() });
  const synth = [...newDescriptors].filter(([nodeId]) => nodeId.startsWith("__call_")).map(([, desc]) => desc);
  expect(synth).toHaveLength(1);
  const callDesc = synth[0] as CallDescriptor;
  expect(callDesc.call).toBe("helper");
  expect(callDesc.range).toEqual(host.fieldRanges.get(siteKey("input", "x"))!); // the `x:` binding
  expect(callDesc.range).not.toEqual(host.range); // ... not the header
  expect(callDesc.fieldRanges).toEqual(new Map()); // owns no authored fields
});

test("inline call error uses binding range", () => {
  const text =
    "id: f\nname: f\nnodes:\n" +
    "  writer:\n    kind: code\n    code: m:f\n" +
    "    input:\n      x: ${helper(topic=input.topic)}\n    output: str\n";
  const nodesCursor = nodesReader(text);
  const descriptors = parseNodes(nodesCursor);
  const bindingRange = descriptors.get("writer")!.fieldRanges.get(siteKey("input", "x"))!;

  const error = loadError(() =>
    desugarInlineCalls(descriptors, null, {
      nextId: synthIdGen(),
      diagnostics: LoadDiagnostics.fromCursor(nodesCursor),
    }),
  );

  const frame = onlyFrame(error);
  expect(frame.range).toEqual(bindingRange);
  expect(frame.text).toBe(text);
});

test("inline output error uses section range", () => {
  const outputRange = new SourceRange(17, 9, 17, 28);

  const error = loadError(() =>
    desugarInlineCalls(new Map(), "call(helper, topic=${item})", {
      outputsRange: outputRange,
      nextId: synthIdGen(),
      diagnostics: diagnostics(),
    }),
  );

  expect(onlyFrame(error).range).toEqual(outputRange);
});

test("inline assert error uses positional range", () => {
  const assertionRanges = [new SourceRange(18, 7, 18, 11), new SourceRange(19, 7, 19, 25)];

  const error = loadError(() =>
    desugarInlineCalls(new Map(), null, {
      assertsSection: ["true", "${helper(topic=1)}"],
      assertionRangeAt: (index) => assertionRanges[index]!,
      nextId: synthIdGen(),
      diagnostics: diagnostics(),
    }),
  );

  expect(onlyFrame(error).range).toEqual(assertionRanges[1]!);
});

test("duplicate inline assertions keep positional ranges", () => {
  const assertionRanges = [new SourceRange(21, 7, 21, 19), new SourceRange(22, 7, 22, 19)];

  const [descriptors] = desugarInlineCalls(new Map(), null, {
    assertsSection: ["call(helper)", "call(helper)"],
    assertionRangeAt: (index) => assertionRanges[index]!,
    nextId: synthIdGen(),
  });

  expect(descriptors.get("__call_0")!.range).toBe(assertionRanges[0]!);
  expect(descriptors.get("__call_1")!.range).toBe(assertionRanges[1]!);
});

test("inline map scalar over does not use the range of an input named over", () => {
  const inputRange = new SourceRange(71, 9, 71, 25);
  const scalarRange = new SourceRange(72, 9, 72, 36);
  const descriptor = new MapDescriptor({
    id: "mapped",
    call: "child",
    inputs: { over: "${input.seed}" },
    over: "${helper(topic=item)}",
    fieldRanges: new Map([
      [siteKey("input", "over"), inputRange],
      [siteKey("field", "over"), scalarRange],
    ]),
  });

  const error = loadError(() =>
    desugarInlineCalls(new Map([["mapped", descriptor]]), null, { diagnostics: diagnostics() }),
  );

  expect(onlyFrame(error).range).toEqual(scalarRange);
});

test("desugar adaptive questions preserves ranges", () => {
  // An adaptive human_input gate lifts a synth composer agent (`<gate>__compose`). The synth agent
  // inherits the gate node's range/fieldRanges, but its `prompt:` is the brief, so it locates at
  // the brief and is named as the gate's field.
  const text =
    "id: f\nname: f\nnodes:\n" +
    "  ask:\n    kind: human_input\n" +
    "    prompt: Pick one\n" +
    "    adaptive_questions:\n      prompt: Ask about ${input.topic}\n" +
    "      llm_config: {model: m}\n" +
    "    input:\n      topic: ${input.topic}\n" +
    "output: ${ask.output}\n";
  const descriptors = parseNodes(nodesReader(text));
  const gate = descriptors.get("ask")!;
  const brief = gate.fieldRanges.get(siteKey("text", "adaptive_questions.prompt"))!;
  const llmConfig = gate.fieldRanges.get(siteKey("field", "adaptive_questions.llm_config"))!;
  const newDescriptors = desugarAdaptiveQuestions(descriptors);
  const composer = newDescriptors.get("ask__compose")!;
  expect(composer.range).toBe(gate.range);
  expect(composer.fieldRanges.get(siteKey("field", "llm_config"))).toBe(llmConfig);
  expect(composer.fieldRanges).toEqual(
    new Map([
      ...gate.fieldRanges,
      [siteKey("field", "prompt"), brief],
      [siteKey("field", "llm_config"), llmConfig],
    ]),
  );
  expect(brief).not.toEqual(gate.fieldRanges.get(siteKey("field", "prompt"))!);
  expect(composer.fieldLabels).toEqual(
    new Map([
      ["prompt", "node 'ask' adaptive_questions.prompt"],
      ["llm_config", "node 'ask' adaptive_questions.llm_config"],
      ["input 'topic'", "node 'ask' input 'topic'"],
    ]),
  );
  expect(newDescriptors.get("ask")!.fieldLabels).toEqual(new Map());
});

test("desugar adaptive questions without source", () => {
  const gate = new HumanInputDescriptor({
    id: "ask",
    inputs: { topic: "${input.topic}" },
    adaptiveQuestions: { prompt: "Q", llm_config: { model: "m" } },
  });
  const newDescriptors = desugarAdaptiveQuestions(new Map([["ask", gate]]));
  const composer = newDescriptors.get("ask__compose")!;
  expect(composer.fieldRanges.has(siteKey("field", "prompt"))).toBe(false);
  expect(composer.fieldRanges.has(siteKey("field", "llm_config"))).toBe(false);
  expect(composer.fieldLabels).toEqual(
    new Map([
      ["prompt", "node 'ask' adaptive_questions.prompt"],
      ["llm_config", "node 'ask' adaptive_questions.llm_config"],
      ["input 'topic'", "node 'ask' input 'topic'"],
    ]),
  );
  expect(newDescriptors.get("ask")!.fieldLabels).toEqual(new Map());
});

test("desugar case call targets locate at their target line", () => {
  // A `case` then:/else: that is a whole-value `call(...)` directive lifts a synth `call` node.
  // That synth node locates at its OWN `then:` line (captured positionally in the case
  // descriptor's `thenRanges`), NOT the host case header, and carries an empty `fieldRanges`.
  const text =
    "id: f\nname: f\nnodes:\n" +
    "  route:\n    kind: case\n    on: ${score.output}\n" +
    "    cases:\n      - when: high\n        then: call(helper, n=3)\n" + // `then:` on line 9
    "    else: cold\n" +
    "output: ${route.output}\n";
  const descriptors = parseNodes(nodesReader(text));
  const predicateOrigin = new OriginRef("case/0/when");
  const host = (descriptors.get("route") as CaseDescriptor).replace({ whenOrigins: [predicateOrigin] });
  descriptors.set("route", host);
  const newDescriptors = desugarCaseCallTargets(descriptors, synthIdGen());
  const synth = [...newDescriptors].filter(([nodeId]) => nodeId.startsWith("__call_")).map(([, desc]) => desc);
  expect(synth).toHaveLength(1);
  const callDesc = synth[0] as CallDescriptor;
  expect(callDesc.call).toBe("helper");
  expect(callDesc.range).toBe(host.thenRanges[0]!); // the `then:` line, not the case header
  expect(callDesc.range).not.toEqual(host.range);
  expect(callDesc.fieldRanges).toEqual(new Map()); // owns no authored fields
  expect((newDescriptors.get("route") as CaseDescriptor).whenOrigins).toEqual([predicateOrigin]);
});

function caseGate(): CaseDescriptor {
  return new CaseDescriptor({ id: "gate", cases: [{ when: "yes", then: "left" }], else_: "right" });
}

test("case binding error uses binding range", () => {
  const hostRange = new SourceRange(30, 3, 35, 1);
  const bindingRange = new SourceRange(33, 10, 33, 44);
  const consumer = new CodeDescriptor({
    id: "consumer",
    code: "m:f",
    inputs: { value: "${gate.output | fallback.output}" },
    range: hostRange,
    fieldRanges: new Map([[siteKey("input", "value"), bindingRange]]),
  });

  const error = loadError(() =>
    expandCaseOutputs(
      new Map<string, NodeDescriptor>([
        ["gate", caseGate()],
        ["consumer", consumer],
      ]),
      null,
      [],
      { diagnostics: diagnostics() },
    ),
  );

  expect(onlyFrame(error).range).toEqual(bindingRange);
});

test("case output error uses section range", () => {
  const outputRange = new SourceRange(40, 9, 40, 23);
  const emptyGate = new CaseDescriptor({ id: "gate", cases: [{ when: "yes" }] });

  const error = loadError(() =>
    expandCaseOutputs(new Map([["gate", emptyGate]]), "${gate.output}", [], {
      outputsRange: outputRange,
      diagnostics: diagnostics(),
    }),
  );

  expect(onlyFrame(error).range).toEqual(outputRange);
});

test.each(["on", "when"])("case condition error uses exact condition range: %s", (conditionKind) => {
  const conditionRange = new SourceRange(51, 11, 51, 25);
  const hostRange = new SourceRange(48, 3, 55, 1);
  const other =
    conditionKind === "on"
      ? new CaseDescriptor({
          id: "other",
          on: "${gate.output}",
          cases: [{ when: "yes", then: "done" }],
          range: hostRange,
          fieldRanges: new Map([[siteKey("field", "on"), conditionRange]]),
        })
      : new CaseDescriptor({
          id: "other",
          cases: [{ when: "${gate.output}", then: "done" }],
          range: hostRange,
          whenRanges: [conditionRange],
        });

  const error = loadError(() =>
    expandCaseOutputs(
      new Map([
        ["gate", caseGate()],
        ["other", other],
      ]),
      null,
      [],
      { diagnostics: diagnostics() },
    ),
  );

  expect(onlyFrame(error).range).toEqual(conditionRange);
});

test("case assert error uses positional range", () => {
  const assertionRanges = [new SourceRange(61, 7, 61, 20), new SourceRange(62, 7, 62, 30)];

  const error = loadError(() =>
    expandCaseOutputs(
      new Map([["gate", caseGate()]]),
      null,
      [new Assertion("true", "true"), new Assertion("gate.output == 1", "gate.output == 1")],
      { assertionRangeAt: (index) => assertionRanges[index]!, diagnostics: diagnostics() },
    ),
  );

  expect(onlyFrame(error).range).toEqual(assertionRanges[1]!);
});

test("case map scalar over does not use the range of an input named over", () => {
  const inputRange = new SourceRange(81, 9, 81, 25);
  const scalarRange = new SourceRange(82, 9, 82, 45);
  const descriptor = new MapDescriptor({
    id: "mapped",
    call: "child",
    inputs: { over: "${input.seed}" },
    over: "${gate.output | fallback.output}",
    fieldRanges: new Map([
      [siteKey("input", "over"), inputRange],
      [siteKey("field", "over"), scalarRange],
    ]),
  });

  const error = loadError(() =>
    expandCaseOutputs(
      new Map<string, NodeDescriptor>([
        ["gate", caseGate()],
        ["mapped", descriptor],
      ]),
      null,
      [],
      { diagnostics: diagnostics() },
    ),
  );

  expect(onlyFrame(error).range).toEqual(scalarRange);
});
