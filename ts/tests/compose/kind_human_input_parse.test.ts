/**
 * The `human_input` kind's parse and build: a port of `tests/compose/test_kind_human_input_parse.py`.
 *
 * Changed: Python compares the descriptor to a `HumanInputDescriptor` with `field_ranges`
 * excluded; TS compares its class and fields. The rows Python loads with `load_flow` assemble
 * with `assembleText`. `test_human_input_parse_rejects_a_non_string_adaptive_key`
 * expects the key `'1'`, since TS keeps the key `1` as the text written. The adaptive block's
 * `retries` stays a YAML int (`2n`).
 *
 * TS-only: a lone surrogate in `prompt:`, the `adaptive_questions.prompt:` brief or a literal
 * question's text is refused at the text.
 *
 * Held here until T3-22: the harvested cases at the end, whose error the HUMAN_INPUT parse raises,
 * each under a comment naming the Python file it comes from, and the flows that assemble.
 */

import { describe, expect, test } from "bun:test";

import { BuildContext, siteKey } from "../../src/compose/nodes/base.ts";
import { HumanInputDescriptor } from "../../src/compose/nodes/human_input.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { HumanInputNode, siteLabel, templateSites } from "../../src/nodes/human_input/node.ts";
import { ValueKind } from "../../src/typesys/values.ts";
import { assembled, assembleText, type CaseRow, loadError, parsedNodes, runCase } from "./declarations.ts";

const FLOW = `nodes:
  ask:
    kind: human_input
    prompt: Approve this?
    inputs:
      ctx: \${input.ctx}
`;

const QUESTIONS_FLOW = `nodes:
  ask:
    kind: human_input
    questions:
      - question: Pick one
        header: Choice
        options: [a, b]
`;

/** A context over an empty registry. */
const reading = () => new BuildContext({ registry: new Map() });

/** Return the cursor over node `nodeId` of `text`'s `nodes:`. */
function readerFor(text: string, nodeId: string): SourceCursor {
  return SourceDocument.parse(text).root.child("nodes").child(nodeId);
}

/** Parse node `ask` of `text` as a HUMAN_INPUT. */
function parseAsk(text: string): HumanInputDescriptor {
  return HumanInputDescriptor.parse("ask", readerFor(text, "ask"));
}

/** Return a descriptor's fields, leaving out the parse metadata (Python's `compare=False`). */
function fieldsOf(desc: HumanInputDescriptor): Record<string, unknown> {
  const { range: _range, fieldRanges: _fieldRanges, assertRanges: _assertRanges, ...fields } = desc;
  return fields;
}

test("human_input parse matches descriptor and records ranges", () => {
  const desc = parseAsk(FLOW);

  expect(desc).toBeInstanceOf(HumanInputDescriptor);
  expect(fieldsOf(desc)).toEqual(
    fieldsOf(new HumanInputDescriptor({ id: "ask", prompt: "Approve this?", inputs: { ctx: "${input.ctx}" } })),
  );
  expect(desc.range).toBeInstanceOf(SourceRange);
  expect(desc.fieldRanges.get(siteKey("field", "prompt"))).toBeInstanceOf(SourceRange);
  expect(desc.fieldRanges.get(siteKey("field", "inputs"))).toBeInstanceOf(SourceRange);
});

test("human_input build returns gate and wires inputs", () => {
  const desc = parseAsk(FLOW);
  const result = desc.build(reading());

  expect(result.node).toBeInstanceOf(HumanInputNode);
  expect((result.node as HumanInputNode).prompt).toBe("Approve this?");
  expect(result.wiring).toEqual(new Map([["ctx", "${input.ctx}"]]));
  expect(result.nodeRange).toBe(desc.range);
});

test("human_input build: a questions gate defaults to an open object", () => {
  const node = parseAsk(QUESTIONS_FLOW).build(reading()).node as HumanInputNode;

  expect(node).toBeInstanceOf(HumanInputNode);
  // A questions gate with no `output:` defaults to a bare open OBJECT.
  expect(node.outputContract).toBeInstanceOf(ValueContract);
  expect((node.outputContract as ValueContract).type.kind).toBe(ValueKind.OBJECT);
  expect(node.answerSchema).toEqual([{ name: "answer", type: "object", required: true }]);
});

test("human_input parse requires a question surface", () => {
  const flow = "nodes:\n  ask:\n    kind: human_input\n    inputs:\n      x: ${input.x}\n";
  const error = loadError(() => parseAsk(flow));

  expect(error.message).toContain("needs `prompt:`");
  // The absent-question-surface error anchors on the node body range, as no question field
  // exists to carry one.
  expect(error.diagnostic.frames.at(-1)!.range).toBeInstanceOf(SourceRange);
});

test("human_input parse: questions and adaptive_questions are mutually exclusive", () => {
  const flow =
    "nodes:\n" +
    "  ask:\n" +
    "    kind: human_input\n" +
    "    questions:\n" +
    "      - question: Pick\n" +
    "        header: H\n" +
    "        options: [a, b]\n" +
    "    adaptive_questions:\n" +
    "      prompt: author questions\n";

  expect(loadError(() => parseAsk(flow)).message).toContain("mutually exclusive");
});

/** A flow whose gate `ask` reads input `order` and writes `questions:` as `questions`. */
const questionsRefFlow = (questions: string) => `id: ask
name: ask
input:
  order: !object
nodes:
  ask:
    kind: human_input
    input:
      order: \${input.order}
    questions: '${questions}'
output: \${ask.output}
`;

test.each(["${isinstance(order, Buy)}", "${bogus(order)}"])(
  "a questions: function error %s points at questions:",
  (questions) => {
    const error = loadError(() => assembleText(questionsRefFlow(questions)));

    expect(error.message).toMatch(/exactly one `\$\{name\}`/);
    expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(10);
  },
);

/** Assemble a flow whose gate `ask` reads input `qs` and writes `questions: <questionsYaml>`. */
function loadQuestions(questionsYaml: string) {
  return assembleText(
    "id: f\n" +
      "name: f\n" +
      "input:\n" +
      "  qs: !list(object)\n" +
      "nodes:\n" +
      "  ask:\n" +
      "    kind: human_input\n" +
      '    input: {qs: "${input.qs}"}\n' +
      `    questions: ${questionsYaml}\n` +
      "output: ${ask.output}\n",
  );
}

test.each(["5", "true", '{question: "Pick one", header: P}'])(
  "human_input build rejects questions: %s, not a list or reference",
  (questionsYaml) => {
    // A non-list, non-text `questions:` used to build a plain gate with an empty prompt.
    const error = loadError(() => loadQuestions(questionsYaml));

    expect(error.message).toContain("`questions:` must be a list");
    expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(9);
  },
);

test.each([
  '"Answer these carefully: ${qs}"',
  '"${qs} "',
  '"${upper(qs)}"',
  '"${qs or qs}"',
  '"${qs.items}"',
  '"${qs}${qs}"',
  '"qs"',
  '"${qs[0]}"',
])("human_input build rejects questions: %s, not one whole reference", (questionsYaml) => {
  // Text or an expression around the reference used to be dropped, keeping only the name.
  const error = loadError(() => loadQuestions(questionsYaml));

  expect(error.message).toMatch(/exactly one `\$\{name\}`/);
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(9);
});

test.each(['"${qs}"', '"${ qs }"', "${qs}"])("human_input build accepts the whole reference %s", (questionsYaml) => {
  const node = loadQuestions(questionsYaml).flow.compiled.nodes.get("ask") as HumanInputNode;

  expect(node.questionsInput).toBe("qs");
  expect(node.questions).toBeNull();
});

const TWO_QUESTIONS_FLOW = `nodes:
  ask:
    kind: human_input
    questions:
      - question: First
        header: One
      - question: Second
        header: Two
        options:
          - label: Alpha
            description: First letter
          - label: Beta
            description: Second letter
          - plain
`;

test("human_input parse records a range per question string", () => {
  const desc = parseAsk(TWO_QUESTIONS_FLOW);

  const label = desc.fieldRanges.get(siteKey("text", "questions[1].options[0].label"));
  expect(label).toBeInstanceOf(SourceRange);
  expect(label!.startLine).toBe(10);
  const sites = templateSites(null, desc.questions);
  expect(sites).toHaveLength(6);
  for (const [path] of sites) {
    expect(desc.fieldRanges.get(siteKey("text", siteLabel(path)))).toBeInstanceOf(SourceRange);
  }
  expect([...desc.fieldRanges.keys()].filter((key) => key.startsWith(siteKey("text", "questions[1].options[2]")))).toEqual([]);
});

test("human_input parse records ranges in a flow-style list", () => {
  const flow = `nodes:
  ask:
    kind: human_input
    questions:
      - question: Pick
        options: [{label: A, description: B}]
`;
  const desc = parseAsk(flow);

  for (const name of ["label", "description"]) {
    expect(desc.fieldRanges.get(siteKey("text", `questions[0].options[0].${name}`))!.startLine).toBe(6);
  }
});

test("human_input parse ranges an aliased or merged question at its authored string", () => {
  const flow = `x-q: &q
  question: Anchored?
  header: A
nodes:
  ask:
    kind: human_input
    questions:
      - *q
      - &r {question: Earlier, header: B}
      - *r
      - <<: *q
        header: C
`;
  const desc = parseAsk(flow);

  const lines = [0, 1, 2, 3].map((i) => desc.fieldRanges.get(siteKey("text", `questions[${i}].question`))!.startLine);
  expect(lines).toEqual([2, 9, 9, 2]);
});

describe("the adaptive brief range", () => {
  /** The line of the `adaptive_questions.prompt` text range of node `ask` in `flow`. */
  const briefLine = (flow: string) =>
    parseAsk(flow).fieldRanges.get(siteKey("text", "adaptive_questions.prompt"))!.startLine;

  test("block style", () => {
    const flow = `nodes:
  ask:
    kind: human_input
    adaptive_questions:
      retries: 1
      prompt: Ask about \${x}
`;
    expect(briefLine(flow)).toBe(6);
  });

  test("flow style", () => {
    const flow = `nodes:
  ask:
    kind: human_input
    adaptive_questions: {prompt: "Q \${x}"}
`;
    expect(briefLine(flow)).toBe(4);
  });

  test("a block scalar", () => {
    const flow = `nodes:
  ask:
    kind: human_input
    adaptive_questions:
      prompt: |
        Ask about \${x}
        and more.
`;
    expect(briefLine(flow)).toBe(5);
  });

  test("a merged brief is ranged at the anchor", () => {
    const flow = `x-brief: &b
  prompt: Ask about \${x}
nodes:
  ask:
    kind: human_input
    adaptive_questions:
      <<: *b
      retries: 1
`;
    expect(briefLine(flow)).toBe(2);
  });

  test("no adaptive_questions has no brief range", () => {
    expect(parseAsk(FLOW).fieldRanges.has(siteKey("text", "adaptive_questions.prompt"))).toBe(false);
  });
});

describe("the adaptive llm_config range", () => {
  const LLM_CONFIG = siteKey("field", "adaptive_questions.llm_config");

  test("flow style", () => {
    const flow = `nodes:
  ask:
    kind: human_input
    adaptive_questions:
      prompt: Ask about \${x}
      llm_config: {model: m}
`;
    const range = parseAsk(flow).fieldRanges.get(LLM_CONFIG)!;
    expect([range.startLine, range.startCol]).toEqual([6, 19]);
  });

  test("block style", () => {
    const flow = `nodes:
  ask:
    kind: human_input
    adaptive_questions:
      prompt: Ask about \${x}
      llm_config:
        model: m
`;
    expect(parseAsk(flow).fieldRanges.get(LLM_CONFIG)!.startLine).toBe(7);
  });

  test("a merged llm_config is ranged at the anchor", () => {
    const flow = `x-llm: &l
  llm_config: {model: m}
nodes:
  ask:
    kind: human_input
    adaptive_questions:
      <<: *l
      prompt: Ask about \${x}
`;
    expect(parseAsk(flow).fieldRanges.get(LLM_CONFIG)!.startLine).toBe(2);
  });

  test("no adaptive llm_config has no range", () => {
    const flow = `nodes:
  ask:
    kind: human_input
    adaptive_questions:
      prompt: Ask about \${x}
`;
    expect(parseAsk(flow).fieldRanges.has(LLM_CONFIG)).toBe(false);
  });

  test("no adaptive_questions has no llm_config range", () => {
    expect(parseAsk(FLOW).fieldRanges.has(LLM_CONFIG)).toBe(false);
  });
});

describe("unknown adaptive_questions keys", () => {
  const ALLOWED = "(allowed: llm_config, prompt, retries)";
  const unknownAdaptive = (key: string) =>
    `node 'ask' (kind=human_input): \`adaptive_questions:\` field '${key}' is not allowed ${ALLOWED}`;
  /** The `[line, col]` start of `error`'s last frame. */
  const startOf = (flow: string) => {
    const range = loadError(() => parseAsk(flow)).diagnostic.frames.at(-1)!.range;
    return [range.startLine, range.startCol];
  };

  test.each([
    ["bogus: 1", "bogus"],
    ['system_prompt: "S ${nope}"', "system_prompt"],
  ])("%s is refused at its key", (line, key) => {
    const flow = `nodes:
  ask:
    kind: human_input
    adaptive_questions:
      prompt: Ask
      ${line}
`;
    expect(loadError(() => parseAsk(flow)).message).toContain(unknownAdaptive(key));
    expect(startOf(flow)).toEqual([6, 7]);
  });

  test("in flow style", () => {
    const flow = `nodes:
  ask:
    kind: human_input
    adaptive_questions: {prompt: Q, bogus: 1}
`;
    expect(loadError(() => parseAsk(flow)).message).toContain(unknownAdaptive("bogus"));
    expect(startOf(flow)).toEqual([4, 37]);
  });

  test("the first unknown key in authored order is reported", () => {
    const flow = `nodes:
  ask:
    kind: human_input
    adaptive_questions:
      prompt: Ask
      zeta: 1
      alpha: 2
`;
    expect(loadError(() => parseAsk(flow)).message).toContain(unknownAdaptive("zeta"));
  });

  test("a number key is refused as the text written", () => {
    const flow = `nodes:
  ask:
    kind: human_input
    adaptive_questions:
      prompt: Ask
      1: x
`;
    expect(loadError(() => parseAsk(flow)).message).toContain(unknownAdaptive("1"));
    expect(startOf(flow)).toEqual([6, 7]);
  });

  test("a merged unknown key is ranged at the anchor", () => {
    const flow = `x-b: &b {prompt: Q, bogus: 1}
nodes:
  ask:
    kind: human_input
    adaptive_questions:
      <<: *b
`;
    expect(loadError(() => parseAsk(flow)).message).toContain(unknownAdaptive("bogus"));
    expect(startOf(flow)).toEqual([1, 21]);
  });

  test("past a recursive alias, the error is ranged at the adaptive_questions key", () => {
    const flow = `nodes:
  ask: &a
    kind: human_input
    adaptive_questions: *a
`;
    expect(loadError(() => parseAsk(flow)).message).toContain(unknownAdaptive("kind"));
    expect(startOf(flow)).toEqual([4, 5]);
  });

  test("without source the error has no frames", () => {
    const body = { kind: "human_input", adaptive_questions: { prompt: "Ask", bogus: 1n } };
    const error = loadError(() => parseNodes(SourceCursor.virtual({ ask: body })));

    expect(error.message).toContain(unknownAdaptive("bogus"));
    expect(error.diagnostic.frames).toEqual([]);
  });
});

test("human_input parse accepts every allowed adaptive key", () => {
  const flow = `nodes:
  ask:
    kind: human_input
    adaptive_questions:
      prompt: Ask
      llm_config: {model: m}
      retries: 2
`;
  expect(parseAsk(flow).adaptiveQuestions).toEqual({ prompt: "Ask", llm_config: { model: "m" }, retries: 2n });
});

// --- TS-only rows --- //

describe("lone surrogates", () => {
  test("in prompt:, refused at the field", () => {
    const text = 'nodes:\n  ask:\n    kind: human_input\n    prompt: "\\ud800"\n';
    const error = loadError(() => parseAsk(text));

    expect(error.message).toBe("node 'ask' field 'prompt': text holding a lone surrogate is not a yian value");
    expect(error.diagnostic.frames.at(-1)!.range).toEqual(readerFor(text, "ask").rangeOf("prompt")!);
  });

  test("in the adaptive brief, refused at the brief", () => {
    const text = 'nodes:\n  ask:\n    kind: human_input\n    adaptive_questions:\n      prompt: "\\ud800"\n';
    const error = loadError(() => parseAsk(text));

    expect(error.message).toBe(
      "node 'ask' field 'adaptive_questions.prompt': text holding a lone surrogate is not a yian value",
    );
    expect(error.diagnostic.frames.at(-1)!.range).toEqual(
      readerFor(text, "ask").child("adaptive_questions").rangeOf("prompt")!,
    );
  });

  test("in a literal question, refused at the text", () => {
    const text =
      "nodes:\n  ask:\n    kind: human_input\n    questions:\n      - question: Pick\n" +
      '        options: [{label: "\\ud800"}]\n';
    const error = loadError(() => parseAsk(text));

    expect(error.message).toBe(
      "node 'ask' field 'questions' at [0].options[0].label: text holding a lone surrogate is not a yian value",
    );
    expect(error.diagnostic.frames.at(-1)!.range).toEqual(
      readerFor(text, "ask").child("questions").child(0).child("options").child(0).rangeOf("label")!,
    );
  });
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  // tests/compose/test_agent_field_shapes.py
  { id: "load-agent/compose_agent_field_shapes__adaptive_questions_retries_is_checked_at_load", call: parsedNodes },
  // tests/compose/test_agent_mode_removed.py
  { id: "load-agent/compose_agent_mode_removed__adaptive_questions_mode_block_form_is_located", call: parsedNodes },
  // tests/compose/test_non_text_prompt.py
  {
    id: "load-core/compose_non_text_prompt__adaptive_questions_prompt_error_comes_before_exclusivity",
    call: parsedNodes,
  },
  { id: "load-core/compose_non_text_prompt__compact_prompt_must_be_text", call: parsedNodes },
  { id: "load-core/compose_non_text_prompt__flow_style_question_field_is_located", call: parsedNodes },
  { id: "load-core/compose_non_text_prompt__non_text_adaptive_questions_prompt", call: parsedNodes },
  { id: "load-core/compose_non_text_prompt__non_text_human_input_prompt", call: parsedNodes },
  { id: "load-core/compose_non_text_prompt__non_text_question", call: parsedNodes },
  // tests/compose/test_kind_human_input_parse.py
  { id: "load-leaves/compose_kind_human_input_parse__a_gate_without_a_question_surface", call: parsedNodes },
  { id: "load-leaves/compose_kind_human_input_parse__an_unknown_adaptive_questions_key", call: parsedNodes },
  {
    id: "load-leaves/compose_kind_human_input_parse__questions_and_adaptive_questions_are_exclusive",
    call: parsedNodes,
  },
])("parseNodes: $id", runCase);

test.each<CaseRow>([
  // tests/compose/test_kind_human_input_parse.py
  { id: "load-leaves/compose_kind_human_input_parse__questions_not_a_list_or_ref", call: assembled },
  { id: "load-leaves/compose_kind_human_input_parse__questions_text_around_a_ref", call: assembled },
])("assemble: $id", runCase);

test.each<CaseRow>([
  // tests/compose/test_kind_human_input_parse.py
  { id: "load-leaves/compose_kind_human_input_parse__a_spaced_whole_questions_ref_loads", call: assembled },
  // tests/engine/test_human_input_text_load.py
  { id: "load-leaves/engine_human_input_text_load__a_clean_gate_loads", call: assembled },
])("assembles: $id", runCase);
