/**
 * An AGENT whose `llm_config` holds values gets them as the reserved `__llm_config` input: a port
 * of `tests/compose/test_llm_config_values_wiring.py`, loading each flow with `assembleText`
 * (`assemble` as the loader calls it, the flow's `llm_config:` read as the loader reads it) where
 * Python calls `load_flow`.
 *
 * Ported: `test_a_node_level_value_gives_an_edge_from_its_producer`,
 * `test_a_flow_level_value_reaches_every_agent_it_can_contribute_to`,
 * `test_an_agent_with_only_literal_layers_has_no_reserved_key`,
 * `test_an_inherit_false_agent_does_not_carry_the_flow_layer`,
 * `test_wiring_keys_equal_params_plus_the_reserved_key`,
 * `test_two_loads_of_the_same_text_have_the_same_fingerprint`,
 * `test_changing_which_lower_value_is_dropped_changes_the_wiring_identity`,
 * `test_a_flow_value_read_from_an_agent_that_always_sets_the_field_is_not_a_cycle` (3),
 * `test_a_flow_value_read_from_an_agent_whose_own_value_may_be_null_is_a_cycle`,
 * `test_an_llm_config_value_reading_a_skippable_node_lets_the_agent_skip`,
 * `test_a_shielded_flow_value_reading_a_skippable_node_does_not`,
 * `test_skipped_arithmetic_in_a_value_names_the_llm_config_field`,
 * `test_an_unknown_node_in_a_value_is_named_at_the_llm_config_field`,
 * `test_an_unknown_input_in_a_flow_value_is_located_at_the_flow_field` and
 * `test_a_composer_value_error_names_the_gate_field`,
 * `test_a_layer_of_only_tagged_literals_is_wired_as_a_literal_layer` and
 * `test_clone_child_renamespaces_the_reserved_key`, which loads its `defs:` child with `loadFlow`.
 *
 * Pinned in kind_agent_parse.test.ts: `test_an_authored_input_named_like_the_reserved_key_fails`.
 *
 * The harvested cases are pinned at the end with `full` (every frame and the notes):
 * `load-agent/compose_llm_config_values_wiring__flow_value_read_from_its_own_reader`, whose cycle
 * message writes the node list in the text form (`["pick"]`, rule `messages`), the two
 * unknown-reference cases, and `load-agent/compose_llm_config_values_wiring__skipped_arithmetic_in_a_value`.
 *
 * `test_runs_with_different_inputs_record_the_same_fingerprint` runs with a scripted model from
 * `models`, where Python patches `model_from_config`, and in an isolated environment with an empty
 * auth file, where Python sets `OPENAI_API_KEY`.
 *
 * Changed: the skipped-arithmetic row pins the whole message where Python checks its start.
 */

import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { expect, test } from "bun:test";

import { cloneChild } from "../../src/compile/expand.ts";
import { resolveLlmCascade } from "../../src/compile/llm_cascade.ts";
import { flowFingerprint, flowManifest } from "../../src/compile/manifest.ts";
import type { CompiledFlow } from "../../src/compile/model.ts";
import type { LoadError } from "../../src/compose/helpers/errors.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { SkipModel } from "../../src/compose/validate/skips.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { LLM_CONFIG_KEY } from "../../src/nodes/agent/llm_values.ts";
import { Binding, iterSourceStrings, ValueTemplate } from "../../src/nodes/contracts.ts";
import { runFlow } from "../../src/run.ts";
import { serving, textReply } from "../engine/run_state_fixtures.ts";
import { assembled, assembleText, loadError, runCase } from "./declarations.ts";

const compiled = (text: string): CompiledFlow => assembleText(text, "f.yaml").flow.compiled;

/** Return the 1-based line of the error's last frame. */
const lineOf = (error: LoadError) => error.diagnostic.frames.at(-1)!.range.startLine;

/** Return the 1-based line of the first line of `text` holding `needle`. */
const lineHolding = (text: string, needle: string) => text.split("\n").findIndex((line) => line.includes(needle)) + 1;

/** Return the `LoadError` assembling flow `text` raises. */
const loadFails = (text: string) => loadError(() => assembleText(text, "f.yaml"));

/** The producers of `consumer`'s edges that feed its reserved `llm_config` input. */
function llmEdges(flow: CompiledFlow, consumer: string): Set<string> {
  return new Set(
    flow.edges
      .filter((edge) => edge.to === consumer && (edge.inputGroup ?? "").startsWith(LLM_CONFIG_KEY))
      .map((edge) => edge.from),
  );
}

/** The value of the `ValueTemplate` `node`'s reserved key carries. */
function carried(flow: CompiledFlow, node: string): unknown {
  return (flow.wiring.get(node)!.get(LLM_CONFIG_KEY)!.source as ValueTemplate).value;
}

const NODE_LEVEL = `id: f
name: f
nodes:
  pick:
    kind: agent
    prompt: Pick a model.
    output: str
  chat:
    kind: agent
    prompt: Go.
    llm_config: {model: '\${pick.output}', provider: openai}
output: \${chat.output}
`;

test("a node-level value gives an edge from its producer", () => {
  const flow = compiled(NODE_LEVEL);
  expect(llmEdges(flow, "chat")).toEqual(new Set(["pick"]));
  expect(flow.nodes.get("chat")!.reservedWiringKeys()).toEqual(new Set([LLM_CONFIG_KEY]));
  const binding = flow.wiring.get("chat")!.get(LLM_CONFIG_KEY);
  expect(binding).toBeInstanceOf(Binding);
  expect(binding!.source).toBeInstanceOf(ValueTemplate);
  expect(carried(flow, "chat")).toEqual({ node: { model: "${pick.output}", provider: "openai" } });
});

const FLOW_LEVEL = `id: f
name: f
llm_config: {model: '\${pick.output}'}
nodes:
  pick:
    kind: agent
    prompt: Pick a model.
    output: str
    llm_config: {model: m0}
  open:
    kind: agent
    input: {x: '\${pick.output}'}
    prompt: Go \${x}.
  fixed:
    kind: agent
    input: {x: '\${open.output}'}
    prompt: Go \${x}.
    llm_config: {model: m1}
output: \${fixed.output}
`;

test("a flow-level value reaches every agent it can contribute to", () => {
  const flow = compiled(FLOW_LEVEL);
  expect(llmEdges(flow, "open")).toEqual(new Set(["pick"]));
  expect(carried(flow, "open")).toEqual({ node: {}, flow: { model: "${pick.output}" } });
  // Their own literal `model` always overrides the flow's value.
  for (const agent of ["pick", "fixed"]) {
    expect(llmEdges(flow, agent)).toEqual(new Set());
    expect(flow.wiring.get(agent)!.has(LLM_CONFIG_KEY)).toBe(false);
    expect(flow.nodes.get(agent)!.reservedWiringKeys()).toEqual(new Set());
  }
});

test("an agent with only literal layers has no reserved key", () => {
  const flow = compiled(NODE_LEVEL.replace("'${pick.output}'", "m2"));
  expect(flow.wiring.get("chat")!.has(LLM_CONFIG_KEY)).toBe(false);
  expect(flow.nodes.get("chat")!.reservedWiringKeys()).toEqual(new Set());
});

test("an inherit: false agent does not carry the flow layer", () => {
  const flow = compiled(
    FLOW_LEVEL.replace(
      "    prompt: Go ${x}.\n  fixed:",
      "    prompt: Go ${x}.\n    llm_config: {inherit: false}\n  fixed:",
    ),
  );
  expect(flow.wiring.get("open")!.has(LLM_CONFIG_KEY)).toBe(false);
  expect(llmEdges(flow, "open")).toEqual(new Set());
});

test("wiring keys equal params plus the reserved key", () => {
  const flow = compiled(FLOW_LEVEL);
  const node = flow.nodes.get("open")!;
  const expected = new Set([...(node.params ?? []).map((parameter) => parameter.name), ...node.reservedWiringKeys()]);
  expect(new Set(flow.wiring.get("open")!.keys())).toEqual(expected);
  expect(expected).toEqual(new Set(["x", LLM_CONFIG_KEY]));
});

const DEF_CALL = `id: f
name: f
defs:
  child:
    input: {m: !str }
    nodes:
      a:
        kind: agent
        prompt: Go.
        llm_config: {model: '\${input.m}'}
    output: \${a.output}
nodes:
  c:
    kind: call
    call: child
    input: {m: picked}
output: \${c.output}
`;

test("cloneChild re-namespaces the reserved key", async () => {
  const flow = (await loadFlow(DEF_CALL, { sourceLabel: "f.yaml" })).compiled;
  const child = (flow.nodes.get("c") as unknown as { child: CompiledFlow }).child;
  expect(carried(child, "a")).toEqual({ node: { model: "${input.m}" } });
  const cloned = cloneChild(child, "c", { m: "picked" });
  const texts = [...iterSourceStrings(cloned.wiring.get("c/a")!.get(LLM_CONFIG_KEY))].map(([, text]) => text);
  expect(texts).toEqual(["${c/__start__.output.m}"]);
});

test("two loads of the same text have the same fingerprint", () => {
  expect(flowFingerprint(compiled(FLOW_LEVEL))).toBe(flowFingerprint(compiled(FLOW_LEVEL)));
  expect(flowFingerprint(compiled(NODE_LEVEL))).toBe(flowFingerprint(compiled(NODE_LEVEL)));
});

const DROP = `id: f
name: f
input: {m: !str , n: !Optional(str) }
llm_config: {model: '\${input.m}'}
nodes:
  a:
    kind: agent
    prompt: Go.
    llm_config: {model: NODE_MODEL}
output: \${a.output}
`;

test("changing which lower value is dropped changes the wiring identity", () => {
  // A higher value that can't be null drops the flow's `model`; one that can, keeps it.
  const shielded = compiled(DROP.replace("NODE_MODEL", "!str '${input.n}'"));
  const open = compiled(DROP.replace("NODE_MODEL", "!Optional(str) '${input.n}'"));
  expect((carried(shielded, "a") as Record<string, unknown>)["flow"] ?? {}).toEqual({});
  expect((carried(open, "a") as Record<string, unknown>)["flow"]).toEqual({ model: "${input.m}" });
  const wiringOf = (flow: CompiledFlow) => (flowManifest(flow)["wiring"] as Record<string, unknown>)["a"];
  expect(wiringOf(shielded)).not.toEqual(wiringOf(open));
  expect(flowFingerprint(shielded)).not.toBe(flowFingerprint(open));
});

const RUN = `id: f
name: f
input: {m: !str }
nodes:
  chat:
    kind: agent
    prompt: Go.
    llm_config: {provider: openai, model: '\${input.m}'}
output: \${chat.output}
`;

test("runs with different inputs record the same fingerprint", async () => {
  const dir = mkdtempSync(join(tmpdir(), "yian-llm-values-run-"));
  try {
    const env = (name: string) => (name === "YIAN_AUTH_FILE" ? join(dir, "auth.json") : undefined);
    const fingerprints: unknown[] = [];
    for (const [index, model] of ["gpt-a", "gpt-b"].entries()) {
      const log = join(dir, `run${index}.jsonl`);
      const result = await runFlow(await loadFlow(RUN, { env }), { m: model }, {
        log,
        env,
        models: serving(new ScriptedModel([textReply("ok")])),
      });
      expect(result.status, result.failure?.message).toBe("succeeded");
      const started = JSON.parse(readFileSync(log, "utf8").split("\n")[0]!);
      expect(started.payload.effective_llm_by_static_path.chat.model).toBe(model);
      fingerprints.push(started.payload.flow_fingerprint);
    }
    expect(fingerprints[0]).toBe(fingerprints[1]);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

const SELF = `id: f
name: f
input: {m: !str , n: !Optional(str) }
llm_config: {model: '\${pick.output}'}
nodes:
  pick:
    kind: agent
    prompt: Pick a model.
    output: str
    llm_config: {model: PICK_MODEL}
  chat:
    kind: agent
    prompt: Go.
output: \${chat.output}
`;

test.each(["m0", "'${input.m}'", "'x${input.n}'"])(
  "a flow value read from an agent that always sets the field is not a cycle (%s)",
  (pickModel) => {
    const flow = compiled(SELF.replace("PICK_MODEL", pickModel));
    expect(llmEdges(flow, "chat")).toEqual(new Set(["pick"]));
    expect(llmEdges(flow, "pick").has("pick")).toBe(false);
  },
);

test("a flow value read from an agent whose own value may be null is a cycle", () => {
  expect(loadError(() => compiled(SELF.replace("PICK_MODEL", "'${input.n}'"))).message).toContain("cycle");
});

const CASE = `id: f
name: f
input: {go: !bool }
LAYER
nodes:
  gate:
    kind: case
    cases:
      - when: \${input.go}
        then: pick
    else: other
  pick:
    kind: code
    output: str
    code: |
      return "gpt-x"
  other:
    kind: code
    output: str
    code: |
      return "no"
  chat:
    kind: agent
    prompt: Go.
    CHAT
output: \${chat.output}
`;

/** A skip model over the whole compiled flow, no CASE exhaustive. */
const skipModel = (flow: CompiledFlow) => new SkipModel(new Set(flow.nodes.keys()), flow.edges, new Set());

test("an llm_config value reading a skippable node lets the agent skip", () => {
  const flow = compiled(CASE.replace("LAYER", "").replace("CHAT", "llm_config: {model: '${pick.output}'}"));
  expect(skipModel(flow).mayAllSkip(["chat"])).toBe(true);
});

test("a shielded flow value reading a skippable node does not", () => {
  const flow = compiled(
    CASE.replace("LAYER", "llm_config: {model: '${pick.output}'}").replace("CHAT", "llm_config: {model: m1}"),
  );
  expect(llmEdges(flow, "chat")).toEqual(new Set());
  expect(skipModel(flow).mayAllSkip(["chat"])).toBe(false);
});

test("skipped arithmetic in a value names the llm_config field", () => {
  const text = CASE.replace("LAYER", "").replace("CHAT", `llm_config: {model: '\${pick.output + "x"}'}`);
  const error = loadError(() => compiled(text));
  expect(error.message).toBe(
    "node 'chat' llm_config: ${pick.output + \"x\"} does arithmetic on pick.output, which is null when node " +
      "'pick' is skipped; give it a default with :-, e.g. (pick.output :- <default>)",
  );
  const line = text.split("\n").findIndex((each) => each.includes("pick.output + ")) + 1;
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(line);
});

test("an unknown node in a value is named at the llm_config field", () => {
  const text = NODE_LEVEL.replace("'${pick.output}'", "'${nope.output}'");
  const error = loadFails(text);
  expect(error.diagnostic.message).toContain("\n  node 'chat' llm_config: reference ${nope.output}");
  expect(compiled(NODE_LEVEL).nodes.get("chat")!.bindingLabels.get(LLM_CONFIG_KEY)).toBe("node 'chat' llm_config");
  expect(lineOf(error)).toBe(lineHolding(text, "${nope.output}"));
});

test("an unknown input in a flow value is located at the flow field", () => {
  const text = FLOW_LEVEL.replace("'${pick.output}'}\nnodes", "'${input.nope}'}\nnodes");
  const error = loadFails(text);
  expect(error.diagnostic.message).toContain("\n  node 'open' llm_config: reference ${input.nope}");
  expect(lineOf(error)).toBe(lineHolding(text, "${input.nope}"));
});

const ASK = `id: f
name: f
input: {proj: !str }
nodes:
  ask:
    kind: tool
    tool_id: ask
    input:
      proj: \${input.proj}
    adaptive_questions:
      prompt: "Q \${proj}"
      llm_config: {model: '\${nope.output}'}
output: \${ask.output}
`;

test("a composer value error names the gate field", () => {
  const error = loadFails(ASK);
  expect(error.diagnostic.message).toContain("\n  node 'ask' adaptive_questions.llm_config: reference ${nope.output}");
  expect(lineOf(error)).toBe(lineHolding(ASK, "${nope.output}"));
  const composer = compiled(ASK.replace("${nope.output}", "${input.proj}"));
  expect(composer.nodes.get("ask__compose")!.bindingLabels.get(LLM_CONFIG_KEY)).toBe(
    "node 'ask' adaptive_questions.llm_config",
  );
});

const TAGGED_LITERALS = `id: f
name: f
llm_config: {provider: !str openai}
nodes:
  chat:
    kind: agent
    prompt: Go.
    llm_config: {model: !str gpt-x, inherit: !bool true}
output: \${chat.output}
`;

test("a layer of only tagged literals is wired as a literal layer", () => {
  const flow = compiled(TAGGED_LITERALS);
  expect(flow.wiring.get("chat")!.has(LLM_CONFIG_KEY)).toBe(false);
  expect(flow.nodes.get("chat")!.reservedWiringKeys()).toEqual(new Set());
  expect(flow.flowLlmConfig).toEqual({ provider: "openai" });
  const effective = resolveLlmCascade(flow, {}, { defaults: {} });
  expect(effective["chat"]!["provider"]).toBe("openai");
  expect(effective["chat"]!["model"]).toBe("gpt-x");
});

// ---------- harvested conformance cases ----------

test.each(["flow_value_read_from_its_own_reader", "unknown_flow_input_named_on_the_agent", "unknown_node_in_a_value"])(
  "assembled: load-agent/compose_llm_config_values_wiring__%s",
  (name) => runCase({ id: `load-agent/compose_llm_config_values_wiring__${name}`, call: assembled, full: true }),
);

test("assembled: load-agent/compose_llm_config_values_wiring__skipped_arithmetic_in_a_value", () => {
  runCase({ id: "load-agent/compose_llm_config_values_wiring__skipped_arithmetic_in_a_value", call: assembled });
});
