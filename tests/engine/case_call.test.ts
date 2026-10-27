/**
 * `case` `then:/else: call(...)` inline-call branch targets.
 *
 * A branch target may be an inline call (a fresh owned call) instead of a placed node id; it
 * desugars to a synth `__call_<n>` node and the then:/else: is rewritten to that synth id, so the
 * case veto skip-floods it when the branch is not taken (a nested call's inner calls follow it).
 * Accepted grammar: a whole-value `call(...)` directive (a route target is one node id). A quoted
 * branch, or one holding `${...}`, is a value, a LoadError that shows the value entry to write; a
 * flow call inside its `${...}` is a LoadError that says to call the flow as the branch
 * (`then: call(take, ...)`).
 */

import { expect, test } from "bun:test";

import type { CompiledFlow } from "../../src/compile/model.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { INLINE_FLOW_PREFIX } from "../../src/compose/desugar/inline_bodies.ts";
import { yamlQuoted } from "../../src/compose/helpers/tags.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { runFlow } from "../../src/run.ts";
import { typeFor } from "../../src/typesys/types.ts";
import {
  assembled,
  assembledWith,
  assembleText,
  type CaseRow,
  handChild,
  loadConformanceFlow,
  loadError,
  parametersFromYaml,
  runCase,
} from "../compose/declarations.ts";

/** A def built by hand: `input:` text `inputs`, one producer `x` of type `str`. */
function def(inputs: string): ComposedFlow {
  return handChild({
    inputs: parametersFromYaml(`input:\n${inputs}`, new Map()),
    producers: { x: new ValueContract(typeFor("str", new Map())) },
    output: "${x.output}",
  });
}

const TAKE = def("  stance: !str\n");

// An in-file `take` def (the branch callable): echoes its stance via the `took` CODE fn.
const TAKE_DEF = `
defs:
  take:
    input:
      stance: !str
    nodes:
      x:
        kind: code
        code: ./fixtures/compose_codefns.ts:took
        input:
          stance: \${input.stance}
        output: str
    output: \${x.output}
`;

/** Assemble `text` with the hand-built callees `children`; return the compiled flow. */
const load = (text: string, children: Record<string, ComposedFlow> = { take: TAKE }) =>
  assembleText(text, "flow.yaml", { children }).flow.compiled;

/** Load `text` with its real defs and run it on `inputs`; return the result. */
const run = async (text: string, inputs: Record<string, unknown>) => runFlow(await loadConformanceFlow(text), inputs);

/** Return the full message for a value branch of `gate`. */
const valueBranch = (label: string, entry: string, shown: string) =>
  `case node 'gate' ${label}: a branch names a node or calls a flow; put the value in ` +
  `its own entry in \`nodes:\` and name it: \`${entry}: ${shown}\` and \`${label}: ${entry}\``;

/** Return the full message for a `take` call inside a `${...}` branch of `gate`. */
const spanFlowCall = (label: string) =>
  `case node 'gate' ${label}: a flow call 'take' inside \`\${...}\` is not supported; ` +
  `call the flow as the branch: \`${label}: call(take, ...)\``;

/** Return the 1-based line number of the first line containing `needle`. */
const lineOf = (text: string, needle: string) => text.split("\n").findIndex((line) => line.includes(needle)) + 1;

/** The ids of the synth call nodes of a compiled flow. */
const synthIds = (compiled: CompiledFlow) =>
  [...compiled.nodes.keys()].filter((nodeId) => nodeId.startsWith("__call_"));

test("non bare then call is rejected", () => {
  // A flow call inside a `${...}` branch is the retired inline `${flow(args)}` form, which a value
  // entry can't hold either: the error points at the call directive.
  const text = `
id: bad
name: bad
input:
  score: !float
${TAKE_DEF}
nodes:
  gate:
    kind: case
    cases:
      - when: "input.score >= 0.5"
        then: \${ take(stance="a") | take(stance="b") }
    else: call(take, stance="c")
output: \${gate.output}
`;
  expect(loadError(() => load(text)).message).toBe(spanFlowCall("then"));
});

// An inline-binding call(...) AND a then: call(...) in one flow: the two share one minter.
const BOTH = `
id: both
name: both
input:
  score: !float
  topic: !str
defs:
  take:
    input:
      stance: !str
    nodes:
      x:
        kind: code
        code: ./fixtures/compose_codefns.ts:took
        input:
          stance: \${input.stance}
        output: str
    output: \${x.output}
  enrich:
    input:
      topic: !str
    nodes:
      y:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input:
          topic: \${input.topic}
        output: str
    output: \${y.output}
nodes:
  use:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: call(enrich, topic=\${input.topic})
    output: str
  gate:
    kind: case
    cases:
      - when: "input.score >= 0.5"
        then: call(take, stance="pro")
    else: use
output: \${gate.output}
`;

test("inline binding and then call share minter no collision", () => {
  const synth = synthIds(load(BOTH, { take: TAKE, enrich: def("  topic: !str\n") }));
  expect(synth).toEqual(["__call_0", "__call_1"]); // distinct ids, no collision
});

// ---------- the `call(...)` directive spelling of a then:/else: branch target ----------

const FLOW_DIRECTIVE = `
id: cc
name: cc
input:
  score: !float
${TAKE_DEF}
nodes:
  gate:
    kind: case
    cases:
      - when: "input.score >= 0.5"
        then: call(take, stance="pro")
    else: call(take, stance="con")
output: \${gate.output}
`;

test("then else call directive desugars to synth branch nodes", () => {
  const compiled = load(FLOW_DIRECTIVE);
  // Both then: and else: call(...) directives became synth call nodes.
  const synth = synthIds(compiled);
  expect(synth).toEqual(["__call_0", "__call_1"]);
  // The gate's control edges target the synth ids (not a placed user node).
  const controlTargets = compiled.edges
    .filter((edge) => edge.sourceHandle !== null && edge.from === "gate")
    .map((edge) => edge.to);
  expect(controlTargets).toEqual(["__call_0", "__call_1"]);
});

test("then call directive branch runs taken value", async () => {
  const out = await run(FLOW_DIRECTIVE, { score: 0.9 });
  expect(out.status).toBe("succeeded");
  expect(out.output).toBe("took:pro");
});

test("non bare then call directive is rejected", () => {
  // A route target must resolve to ONE node id: a coalesce of call(...) directives is loud.
  const text = `
id: bad
name: bad
input:
  score: !float
${TAKE_DEF}
nodes:
  gate:
    kind: case
    cases:
      - when: "input.score >= 0.5"
        then: call(take, stance="a") | call(take, stance="b")
    else: call(take, stance="c")
output: \${gate.output}
`;
  expect(loadError(() => load(text)).message).toBe(
    "case node 'gate' then: a branch target must be a node id or a single call: text follows the " +
      "closing `)`: '| call(take, stance=\"b\")'; a call(...) must be the whole value. For the text, " +
      "add a value entry and name it: " +
      "`gate_then_0: !str 'call(take, stance=\"a\") | call(take, stance=\"b\")'`",
  );
});

test.each(
  ["then", "else"].flatMap((label) =>
    [
      ['call(take, stance="a", stance="b")', "call(take, ...) passes keyword 'stance' more than once"],
      [
        "call(take, 30)",
        "call(take, ...) passes a positional argument '30'; every argument after the " +
          "flow id must be a keyword (name=value)",
      ],
      ["call(take", "`call(` is never closed (check its parentheses and quotes)"],
      ['call(take, stance="a") tail', "text follows the closing `)`: 'tail'; a call(...) must be the whole value"],
      [
        'call(take stance="a")',
        "a call(...) must start with a flow id (a bare name) and a comma, got 'take stance=\"a\"'",
      ],
    ].map(([target, problem]) => [label, target!, problem!]),
  ),
)("malformed branch call directive is a located load error: %s %s", (label, target, problem) => {
  // An unquoted then:/else: starting with `call(` must be exactly one directive.
  const [thenTarget, elseTarget] = label === "then" ? [target, "nope"] : ["nope", target];
  const text = `
id: bad
name: bad
input:
  score: !float
${TAKE_DEF}
nodes:
  gate:
    kind: case
    cases:
      - when: "input.score >= 0.5"
        then: ${thenTarget}
    else: ${elseTarget}
  nope:
    kind: code
    code: ./fixtures/compose_codefns.ts:took
    input:
      stance: x
    output: str
output: \${gate.output}
`;
  const error = loadError(() => load(text));
  const entry = label === "then" ? "gate_then_0" : "gate_else";
  expect(error.message).toBe(
    `case node 'gate' ${label}: a branch target must be a node id or a single call: ` +
      `${problem}. For the text, add a value entry and name it: \`${entry}: !str ${yamlQuoted(target)}\``,
  );
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(lineOf(text, `${label}: call(`));
});

test("quoted malformed branch call is a value branch", () => {
  const text = `
id: bad
name: bad
input:
  score: !float
nodes:
  gate:
    kind: case
    cases:
      - when: "input.score >= 0.5"
        then: 'call(take'
    else: "call(take) tail"
output: \${gate.output}
`;
  expect(loadError(() => load(text, {})).message).toBe(valueBranch("then", "gate_then_0", '!str "call(take"'));
});

test("case output over call directive then taken", async () => {
  // ${gate.output} over a call(...)-desugared `then:`, the taken then value comes back.
  const out = await run(FLOW_DIRECTIVE, { score: 0.9 });
  expect(out.status).toBe("succeeded");
  expect(out.output).toBe("took:pro");
});

test("case output over call directive else taken", async () => {
  // ${gate.output} over a call(...)-desugared `else:`, the taken else value comes back.
  const out = await run(FLOW_DIRECTIVE, { score: 0.2 });
  expect(out.status).toBe("succeeded");
  expect(out.output).toBe("took:con");
});

test("nested then call directive lifts two synth nodes", () => {
  // The nested then: target lifts BOTH the inner and outer call as synth nodes; the else: lifts one
  // more: three synth call nodes in all, the gate's then-edge on the outer.
  const text = FLOW_DIRECTIVE.replace(
    'then: call(take, stance="pro")',
    'then: call(take, stance=call(take, stance="pro"))',
  );
  expect(synthIds(load(text))).toEqual(["__call_0", "__call_1", "__call_2"]);
});

test.each(["then", "else"])("private callee in case target is rejected at branch before resolution: %s", (branch) => {
  const privateName = INLINE_FLOW_PREFIX + "c".repeat(64);
  const thenTarget = branch === "then" ? `call(${privateName})` : "safe";
  const elseTarget = branch === "else" ? `call(${privateName})` : "safe";
  const text = `id: private_case
name: private_case
nodes:
  safe:
    kind: code
    code: ./fixtures/compose_codefns.ts:echo
    input:
      topic: safe
    output: str
  gate:
    kind: case
    cases:
      - when: "true"
        then: ${thenTarget}
    else: ${elseTarget}
output: \${gate.output}
`;
  const error = loadError(() => load(text, {}));
  const expectedLine = text.split("\n").findIndex((line) => line.includes(privateName)) + 1;
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(expectedLine);
  expect(error.message).toBe(
    `case node 'gate' ${branch}:: call targets beginning with '__inline_flow_' are reserved for ` +
      "compiler-generated inline child flows",
  );
});

const SPAN_CALL_FLOW = `
id: bad
name: bad
input:
  score: !float
${TAKE_DEF}
nodes:
  gate:
    kind: case
    cases:
      - when: "input.score >= 0.5"
        then: THEN
    else: ELSE
output: \${gate.output}
`;

test.each([
  ["then", '${ take(stance="pro") }'],
  ["else", '"${ take(stance=\'con\') }"'],
  ["then", '{a: "${ take(stance=\'pro\') }"}'],
])("then span flow call target is rejected: %s %s", (label, branch) => {
  // The retired inline `${flow(args)}` case branch is a LoadError pointing at call(...).
  const [then, else_] =
    label === "else" ? ['call(take, stance="pro")', branch] : [branch, 'call(take, stance="con")'];
  const text = SPAN_CALL_FLOW.replace("THEN", then).replace("ELSE", else_);
  const error = loadError(() => load(text));
  expect(error.message).toBe(spanFlowCall(label));
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(lineOf(text, `${label}: ${branch}`));
});

test("the call directive a span flow call points at loads", async () => {
  const text = SPAN_CALL_FLOW.replace("THEN", 'call(take, stance="pro")').replace("ELSE", 'call(take, stance="con")');
  expect((await run(text, { score: 0.9 })).output).toBe("took:pro");
});

test("malformed branch call holding a span offers no text rewrite", () => {
  // Quoted call text holding a span is its own load error, so no value entry is shown.
  const text = `
id: bad
name: bad
input:
  score: !float
${TAKE_DEF}
nodes:
  gate:
    kind: case
    cases:
      - when: "input.score >= 0.5"
        then: call(take, stance=\${input.score}
    else: call(take, stance="c")
output: \${gate.output}
`;
  expect(loadError(() => load(text)).message).toBe(
    "case node 'gate' then: a branch target must be a node id or a single call: " +
      "`call(` is never closed (check its parentheses and quotes)",
  );
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  { id: "load-case/engine_case_call__a_private_callee_branch_is_rejected", call: assembled },
  ...[
    "a_branch_call_never_closed",
    "a_branch_call_repeating_a_keyword",
    "a_branch_call_with_a_positional_argument",
    "a_branch_call_without_a_flow_id_and_comma",
    "a_malformed_call_holding_a_span_offers_no_text_entry",
    "a_span_flow_call_branch_is_rejected",
  ].map((name) => ({ id: `load-case/engine_case_call__${name}`, call: assembledWith({ take: TAKE }) })),
])("assemble: $id", runCase);
