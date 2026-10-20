/**
 * A human_input gate's prompt and question text are checked at load like an AGENT prompt: a port
 * of `tests/engine/test_human_input_text_load.py`, loading each flow with `assembleText`
 * (`assemble` as the loader calls it) where Python calls `load_flow`.
 *
 * Ported: `test_a_clean_gate_loads`, `test_bad_span_fails_to_load`, `test_pool_ref_fails_to_load`,
 * `test_undeclared_name_fails_to_load`, `test_every_bad_site_is_reported`,
 * `test_prompt_error_points_at_the_prompt_line`, `test_question_text_error_points_at_the_string`
 * and `test_adaptive_gate_prompt_is_checked`.
 *
 * Pinned at the end, the harvested `load-leaves/engine_human_input_text_load__*` cases the
 * reference validator decides; `compose/kind_human_input_parse.test.ts` pins
 * `load-leaves/engine_human_input_text_load__a_clean_gate_loads`.
 *
 * Later (T3-22): `test_gate_in_a_defs_body_is_checked` and its case
 * `load-leaves/engine_human_input_text_load__gate_in_a_defs_body_is_checked` (a def body's own
 * assembly).
 */

import { expect, test } from "bun:test";

import { assembled, assembleText, type CaseRow, loadError, runCase } from "../compose/declarations.ts";

const GOOD = "${proj}";

const SITES: Readonly<Record<string, string>> = {
  prompt: "node 'ask' prompt:",
  question: "node 'ask' questions[0].question:",
  label: "node 'ask' questions[0].options[0].label:",
  description: "node 'ask' questions[0].options[0].description:",
};
const SITE_NAMES = Object.keys(SITES).sort();

const POOL_HINT = " (a pool ref belongs in an input `from:`, not human_input text)";

/** Return the gate flow, each text site `GOOD` unless `over` sets it. */
function flow(over: Readonly<Record<string, string>> = {}): string {
  const { prompt, question, label, description } = {
    prompt: GOOD,
    question: GOOD,
    label: GOOD,
    description: GOOD,
    ...over,
  };
  return `id: f
name: f
input: {proj: !str }
nodes:
  ask:
    kind: human_input
    input: {proj: '\${input.proj}'}
    prompt: "${prompt}"
    questions:
      - question: "${question}"
        header: Framework
        options:
          - label: "${label}"
            description: "${description}"
output: \${ask.output}
`;
}

/** Load `text`, expecting a `LoadError`. */
const failure = (text: string) => loadError(() => assembleText(text));

test("a clean gate loads", () => {
  assembleText(flow());
  assembleText(flow({ prompt: "${proj :- 'none'}" }));
  assembleText(flow({ prompt: "Cost $$5 for ${proj}" }));
});

const BAD_SPANS: [text: string, fragment: string][] = [
  ["${proj +}", "could not parse expression"],
  ["${nosuch(proj)}", "unknown prompt function 'nosuch'"],
  ["${isinstance(proj, str)}", "isinstance is not a prompt function"],
  ["${upper(${proj})}", "write `${upper(proj)}`"],
  ["x ${proj", "unbalanced"],
];

test.each(BAD_SPANS.flatMap(([text, fragment]) => SITE_NAMES.map((site) => [text, fragment, site])))(
  "a bad span fails to load: %s at %s",
  (text, fragment, site) => {
    const message = failure(flow({ [site]: text })).message;
    const line = message.split("\n").find((each) => each.includes(SITES[site]!))!;
    expect(line.trim().startsWith(`${SITES[site]} `)).toBe(true);
    expect(line).toContain(fragment);
  },
);

test.each(["${input.proj}", "${input.proj :- 'd'}"].flatMap((text) => SITE_NAMES.map((site) => [text, site])))(
  "a pool ref fails to load: %s at %s",
  (text, site) => {
    expect(failure(flow({ [site]: text })).message).toContain(
      `${SITES[site]} \${input.proj} is not a declared input${POOL_HINT}`,
    );
  },
);

test.each(["${other}", "${other :- 'x'}"].flatMap((text) => SITE_NAMES.map((site) => [text, site])))(
  "an undeclared name fails to load: %s at %s",
  (text, site) => {
    const message = failure(flow({ [site]: text })).message;
    expect(message).toContain(`${SITES[site]} \${other} is not a declared input`);
    expect(message).not.toContain("pool ref");
  },
);

test("every bad site is reported", () => {
  const bad = flow({
    prompt: "${nosuch(proj)}",
    question: "${other}",
    label: "${input.proj}",
    description: "${proj +}",
  });
  const message = failure(bad).message;
  for (const prefix of Object.values(SITES)) expect(message).toContain(prefix);
});

test("a prompt error points at the prompt line", () => {
  const text = flow({ prompt: "${other}" });
  const promptLine = text.split("\n").indexOf('    prompt: "${other}"') + 1;
  expect(promptLine).toBe(8);
  expect(failure(text).diagnostic.frames.at(-1)!.range.startLine).toBe(promptLine);
});

test.each(
  ["${upper(${proj})}", "${other}"].flatMap((text) =>
    (
      [
        ["question", 10],
        ["label", 13],
        ["description", 14],
      ] as const
    ).map(([site, line]) => [text, site, line] as const),
  ),
)("a question text error points at the string: %s at %s", (text, site, line) => {
  const flowText = flow({ [site]: text });
  expect(flowText.split("\n")[line - 1]).toContain(`"${text}"`);
  expect(failure(flowText).diagnostic.frames.at(-1)!.range.startLine).toBe(line);
});

const ADAPTIVE_FLOW = `id: f
name: f
input: {proj: !str }
nodes:
  ask:
    kind: human_input
    input: {proj: "\${input.proj}"}
    prompt: "\${nope}"
    adaptive_questions:
      prompt: "Q for \${proj}"
output: \${ask.output}
`;

test("an adaptive gate's prompt is checked", () => {
  expect(failure(ADAPTIVE_FLOW).message).toContain("node 'ask' prompt: ${nope} is not a declared input");
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>(
  [
    "isinstance_in_an_option_description",
    "nested_span_in_a_question",
    "pool_ref_in_a_question",
    "unbalanced_span_in_the_prompt",
    "unknown_prompt_function_in_an_option_label",
  ].map((name) => ({ id: `load-leaves/engine_human_input_text_load__${name}`, call: assembled })),
)("assembled: $id", runCase);
