/**
 * Authored `llm_config` layers: mirrors `tests/compose/test_llm_config_values_load.py`.
 *
 * Ported: `test_a_python_built_agent_keeps_its_literal_config`, and every node-layer test (loaded
 * with `assembleText` where Python calls `load_flow`): `test_a_field_value_loads_as_a_template`,
 * `test_a_block_field_value_loads_as_a_template`, `test_a_whole_block_value_loads_as_a_template`,
 * `test_a_whole_record_value_loads_as_a_binding`, `test_a_tagged_whole_record_keeps_its_type`,
 * `test_a_whole_record_that_is_not_one_value_must_be_a_mapping`,
 * `test_a_tag_on_a_field_value_is_read_and_consumed`, `test_a_tag_on_the_whole_mapping_is_rejected`,
 * `test_literal_inherit_is_split_off_a_template_layer`, `test_inherit_from_a_value_fails`,
 * `test_a_connection_field_from_a_value_fails_at_the_field`,
 * `test_headers_from_a_value_fail_at_the_field`, `test_a_whole_record_type_declaring_api_key_fails`,
 * `test_base_url_from_a_value_fails`, `test_base_url_from_a_value_fails_beside_azure`,
 * `test_a_literal_base_url_beside_a_provider_value_fails`,
 * `test_a_literal_base_url_beside_literal_azure_still_loads`,
 * `test_an_unknown_top_level_key_beside_a_value_fails`, `test_a_literal_node_layer_is_still_a_plain_dict`,
 * `test_a_quoted_span_is_no_longer_literal_text`, `test_a_doubled_dollar_reads_as_one_dollar`,
 * `test_a_literal_block_field_beside_a_value_is_checked`,
 * `test_an_unknown_block_key_beside_a_value_fails`, `test_a_required_block_field_given_as_a_value_loads`,
 * `test_a_literal_block_beside_a_value_is_checked_whole`,
 * `test_a_tagged_unknown_provider_is_checked_as_a_literal`, `test_a_tagged_literal_block_field_is_checked`,
 * `test_a_tagged_literal_azure_allows_a_literal_base_url`, `test_a_tagged_literal_inherit_is_accepted`,
 * `test_a_call_inside_an_llm_config_string_fails`,
 * `test_a_literal_field_error_in_a_value_layer_points_at_the_field`; the node halves of
 * `test_a_layer_of_only_tagged_literals_is_a_literal_layer` and
 * `test_a_tag_with_no_value_on_the_layer_fails`; and
 * `test_a_composer_tag_on_the_whole_mapping_names_the_gate` (the gate's layer read raises it).
 *
 * Changed: `LLMConfig` spells the provider choices in text form (`"openai"`) where pydantic writes
 * `'openai'`.
 *
 * TS-only: the `ownLlmConfig`/`llmConfig` split for an empty record, `null`, a `ValueTemplate`
 * and a `Binding`; the origin of a whole-record `Binding`; the adaptive gate's layer on the gate's
 * descriptor (a value layer is a `ValueTemplate` ranged on the gate, a literal one stays a record),
 * which Python observes only through the composer; the harvested cases of the node layer.
 *
 * Later (T3-18): the node half of `test_a_malformed_span_is_a_located_load_error` (the reference
 * check reports it). Later (T3-17), the adaptive composer `ask__compose`:
 * `test_a_composer_value_loads_as_a_template`, `test_a_composer_tag_is_read_and_consumed`,
 * `test_a_composer_excluded_field_from_a_value_names_the_gate`,
 * `test_a_composer_literal_block_field_beside_a_value_names_the_gate`,
 * `test_a_literal_composer_layer_is_unchanged`.
 * Later (T3-22), the flow layer the loader reads: `test_a_flow_field_value_loads_as_a_template`,
 * `test_a_flow_whole_record_value_loads_as_a_binding`, `test_a_flow_tag_is_read_and_consumed`,
 * `test_a_tag_on_a_typedef_still_fails_beside_a_flow_value`,
 * `test_a_flow_excluded_field_from_a_value_fails_at_the_field`, `test_inherit_at_flow_level_still_fails`,
 * `test_a_literal_flow_layer_is_still_a_plain_dict`, `test_a_null_flow_layer_loads_as_empty`,
 * `test_a_flow_layer_that_is_not_a_mapping_fails`,
 * `test_a_literal_field_error_in_a_flow_value_layer_points_at_the_field`; the def layer:
 * `test_a_def_field_value_loads_as_a_template`, `test_a_def_whole_record_value_loads_as_a_binding`,
 * `test_a_def_excluded_field_from_a_value_fails_at_the_field`, `test_a_literal_def_layer_is_still_a_plain_dict`;
 * and the flow and def halves of the three split tests above.
 */

import { expect, spyOn, test } from "bun:test";

import type { LoadError } from "../../src/compose/helpers/errors.ts";
import { siteKey } from "../../src/compose/nodes/base.ts";
import { HumanInputDescriptor } from "../../src/compose/nodes/human_input.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Binding, ValueTemplate } from "../../src/nodes/contracts.ts";
import { assembled, assembleText, type CaseRow, loadError, runCase } from "./declarations.ts";

const INPUTS = `\
input:
  m: !str
  p: !str
  t: !float
  u: !str
  i: !bool
  a: !object
  cfg: !object
`;

/** Return a flow whose agent `a` has the node lines `llmLines` (indented 4). */
function flow(llmLines: string, options: { typedefs?: string } = {}): string {
  const { typedefs = "" } = options;
  return `id: f\nname: f\n${typedefs}${INPUTS}nodes:\n  a:\n    kind: agent\n    prompt: Go.\n${llmLines}output: \${a.output}\n`;
}

/** Return the AGENT `a` of flow `text`, assembled. */
const agentOf = (text: string) => assembleText(text, "f.yaml").flow.compiled.nodes.get("a") as AgentNode;

/** Return the `LoadError` assembling flow `text` raises. */
const loadFails = (text: string) => loadError(() => assembleText(text, "f.yaml"));

/** Return the 1-based line of the error's last frame. */
const lineOf = (error: LoadError) => error.diagnostic.frames.at(-1)!.range.startLine;

/** Return the 1-based line of the first line of `text` holding `needle`. */
const lineHolding = (text: string, needle: string) => text.split("\n").findIndex((line) => line.includes(needle)) + 1;

test("a python built agent keeps its literal config", () => {
  const config = { model: "m" };
  const agent = new AgentNode("a", { prompt: "p", llmConfig: config });
  expect(agent.llmConfig).toEqual({ model: "m" });
  expect(agent.ownLlmConfig).toEqual({ model: "m" });
  expect(agent.llmConfig).toBe(config);
  expect(agent.ownLlmConfig).not.toBe(config);
});

test("the config split for an empty record, null, a template and a binding", () => {
  const empty = {};
  const fromEmpty = new AgentNode("a", { llmConfig: empty });
  expect(fromEmpty.ownLlmConfig).toEqual({});
  expect(fromEmpty.ownLlmConfig).not.toBe(empty);
  expect(fromEmpty.llmConfig).toBe(empty);

  const omitted = new AgentNode("a");
  expect(omitted.ownLlmConfig).toEqual({});
  expect(omitted.llmConfig).toBeNull();

  const template = new ValueTemplate({ model: "${input.m}" });
  const fromTemplate = new AgentNode("a", { llmConfig: template });
  expect(fromTemplate.ownLlmConfig).toBe(template);
  expect(fromTemplate.llmConfig).toBeNull();

  const binding = new Binding("${input.cfg}");
  const fromBinding = new AgentNode("a", { llmConfig: binding });
  expect(fromBinding.ownLlmConfig).toBe(binding);
  expect(fromBinding.llmConfig).toBeNull();
});

test("a field value loads as a template", () => {
  const agent = agentOf(flow("    llm_config: {model: '${input.m}'}\n"));
  expect(agent.ownLlmConfig).toBeInstanceOf(ValueTemplate);
  expect((agent.ownLlmConfig as ValueTemplate).value).toEqual({ model: "${input.m}" });
  expect(agent.llmConfig).toBeNull();
});

test("a block field value loads as a template", () => {
  const agent = agentOf(flow("    llm_config:\n      provider: anthropic\n      anthropic: {temperature: '${input.t}'}\n"));
  expect(agent.ownLlmConfig).toBeInstanceOf(ValueTemplate);
  expect((agent.ownLlmConfig as ValueTemplate).value).toEqual({
    provider: "anthropic",
    anthropic: { temperature: "${input.t}" },
  });
});

test("a whole block value loads as a template", () => {
  const agent = agentOf(flow("    llm_config: {provider: anthropic, anthropic: '${input.a}'}\n"));
  expect(agent.ownLlmConfig).toBeInstanceOf(ValueTemplate);
  expect((agent.ownLlmConfig as ValueTemplate).value).toEqual({ provider: "anthropic", anthropic: "${input.a}" });
});

test("a whole record value loads as a binding", () => {
  const text = flow("    llm_config: ${input.cfg}\n");
  const { flow: composed, builder } = assembleText(text, "f.yaml");
  const agent = composed.compiled.nodes.get("a") as AgentNode;
  const own = agent.ownLlmConfig as Binding;
  expect(own).toBeInstanceOf(Binding);
  expect(own.source).toBe("${input.cfg}");
  expect(own.contract).toBeNull();
  expect(agent.llmConfig).toBeNull();
  expect(agent.llmInherit).toBe(true);
  // TS-only: the binding's origin resolves to the authored `llm_config:` value.
  expect(builder.resolveRegistered(own.origin!)!.range.startLine).toBe(lineHolding(text, "llm_config:"));
});

test("a tagged whole record keeps its type", () => {
  const typedefs = "typedefs:\n  Cfg:\n    model: str\n";
  const agent = agentOf(flow("    llm_config: !Cfg ${input.cfg}\n", { typedefs }));
  const own = agent.ownLlmConfig as Binding;
  expect(own).toBeInstanceOf(Binding);
  expect([...own.contract!.type.fields!.keys()]).toEqual(["model"]);
});

test.each(["gpt-4", "'x${input.m}'", "5"])("a whole record that is not one value must be a mapping: %p", (value) => {
  expect(loadFails(flow(`    llm_config: ${value}\n`)).message).toContain("node 'a': llm_config must be a mapping");
});

test("a tag on a field value is read and consumed", () => {
  const template = agentOf(flow("    llm_config: {model: !str '${input.m}'}\n")).ownLlmConfig as ValueTemplate;
  expect(template).toBeInstanceOf(ValueTemplate);
  const binding = (template.value as Record<string, unknown>)["model"] as Binding;
  expect(binding).toBeInstanceOf(Binding);
  expect(binding.source).toBe("${input.m}");
  expect(binding.contract!.type.kind).toBe("str");
});

test("a tag on the whole mapping is rejected", () => {
  const message = loadFails(flow("    llm_config: !object {model: '${input.m}'}\n")).message;
  expect(message).toContain("node 'a': llm_config");
  expect(message).toContain("tag a field's value");
});

test("literal inherit is split off a template layer", () => {
  const agent = agentOf(flow("    llm_config: {inherit: false, model: '${input.m}'}\n"));
  const own = agent.ownLlmConfig as ValueTemplate;
  expect(agent.llmInherit).toBe(false);
  expect(own.value).toEqual({ model: "${input.m}" });
  expect(own.origins.every(([path]) => path[0] !== "inherit")).toBe(true);
});

test("inherit from a value fails", () => {
  const text = flow("    llm_config:\n      model: m\n      inherit: ${input.i}\n");
  const error = loadFails(text);
  expect(error.message).toContain("node 'a': llm_config `inherit:` must be written as true or false");
  expect(lineOf(error)).toBe(lineHolding(text, "inherit:"));
});

test.each(["api_key", "organization", "project"])("a connection field from a value fails at the field: %p", (field) => {
  const text = flow(`    llm_config:\n      provider: openai\n      ${field}: \${input.u}\n`);
  const error = loadFails(text);
  expect(error.message).toContain(`node 'a': llm_config.${field} `);
  expect(error.message).toContain("not in a flow file");
  expect(lineOf(error)).toBe(lineHolding(text, `${field}:`));
});

test("headers from a value fail at the field", () => {
  const text = flow("    llm_config:\n      provider: openai\n      headers: {X-Team: '${input.u}'}\n");
  const error = loadFails(text);
  expect(error.message).toContain("node 'a': llm_config.headers goes wherever the host's api_key");
  expect(lineOf(error)).toBe(lineHolding(text, "headers:"));
});

test("a whole record type declaring api_key fails", () => {
  const typedefs = "typedefs:\n  Cfg:\n    model: str\n    api_key: str\n";
  const text = flow("    llm_config: !Cfg ${input.cfg}\n", { typedefs });
  const error = loadFails(text);
  expect(error.message).toContain("node 'a': llm_config: its type declares `api_key`");
  expect(error.message).toContain("llm_config.api_key is a host credential");
  expect(lineOf(error)).toBe(lineHolding(text, "llm_config:"));
});

test("base url from a value fails", () => {
  const text = flow("    llm_config:\n      provider: openai\n      base_url: ${input.u}\n");
  const error = loadFails(text);
  expect(error.message).toContain("node 'a': llm_config.base_url decides where");
  expect(lineOf(error)).toBe(lineHolding(text, "base_url:"));
});

test("base url from a value fails beside azure", () => {
  const text = flow("    llm_config:\n      provider: azure\n      model: m\n      base_url: ${input.u}\n");
  const error = loadFails(text);
  expect(error.message).toContain("node 'a': llm_config.base_url can't come from a value");
  expect(error.message).toContain("Azure resource endpoint");
  expect(lineOf(error)).toBe(lineHolding(text, "base_url:"));
});

test("a literal base url beside a provider value fails", () => {
  const text = flow("    llm_config:\n      provider: ${input.p}\n      base_url: https://x.openai.azure.com\n");
  const error = loadFails(text);
  expect(error.message).toContain("node 'a': llm_config.base_url decides where");
  expect(lineOf(error)).toBe(lineHolding(text, "base_url:"));
});

test("a literal base url beside literal azure still loads", () => {
  const agent = agentOf(
    flow("    llm_config:\n      provider: azure\n      model: ${input.m}\n      base_url: https://x.openai.azure.com\n"),
  );
  expect(((agent.ownLlmConfig as ValueTemplate).value as Record<string, unknown>)["base_url"]).toBe(
    "https://x.openai.azure.com",
  );
});

test("an unknown top level key beside a value fails", () => {
  const message = loadFails(flow("    llm_config: {model: '${input.m}', temparature: 1}\n")).message;
  expect(message).toContain("node 'a': llm_config:");
  expect(message).toContain("temparature");
});

test("a literal node layer is still a plain dict", () => {
  const agent = agentOf(flow("    llm_config: {provider: openai, model: gpt-x}\n"));
  expect(Object.getPrototypeOf(agent.ownLlmConfig)).toBe(Object.prototype);
  expect(agent.ownLlmConfig).toEqual({ provider: "openai", model: "gpt-x" });
});

test("a quoted span is no longer literal text", () => {
  expect(agentOf(flow('    llm_config: {model: "${input.m}"}\n')).ownLlmConfig).toBeInstanceOf(ValueTemplate);
});

test("a doubled dollar reads as one dollar", () => {
  expect(agentOf(flow('    llm_config: {model: "a$$b"}\n')).ownLlmConfig).toEqual({ model: "a$b" });
});

test("a literal block field beside a value is checked", () => {
  const text = flow(
    "    llm_config:\n" +
      "      provider: anthropic\n" +
      "      anthropic:\n" +
      "        temperature: ${input.t}\n" +
      "        thinking: {type: enabled, budget_tokens: 10}\n",
  );
  const error = loadFails(text);
  expect(error.message).toContain("node 'a': llm_config: `anthropic.thinking.budget_tokens`:");
  expect(error.message).toContain("greater than or equal to 1024");
  expect(lineOf(error)).toBe(lineHolding(text, "thinking:"));
});

test("an unknown block key beside a value fails", () => {
  const text = flow(
    "    llm_config:\n      provider: anthropic\n      anthropic:\n        temperature: ${input.t}\n        max_tokns: 100\n",
  );
  const error = loadFails(text);
  expect(error.message).toContain("node 'a': llm_config: `anthropic.max_tokns`");
  expect(error.message).toContain("allowed:");
  expect(lineOf(error)).toBe(lineHolding(text, "max_tokns:"));
});

test("a required block field given as a value loads", () => {
  const agent = agentOf(
    flow(
      "    llm_config:\n      provider: anthropic\n" +
        "      anthropic: {thinking: {type: enabled, budget_tokens: !int '${input.t}'}}\n",
    ),
  );
  expect(agent.ownLlmConfig).toBeInstanceOf(ValueTemplate);
});

test("a literal block beside a value is checked whole", () => {
  const text = flow("    llm_config:\n      provider: ${input.p}\n      anthropic: {max_tokns: 100}\n");
  expect(loadFails(text).message).toContain("anthropic.max_tokns");
});

test.each(["", ", model: '${input.m}'"])("a tagged unknown provider is checked as a literal: %p", (beside) => {
  const message = loadFails(flow(`    llm_config: {provider: !str nope${beside}}\n`)).message;
  expect(message).toStartWith("node 'a': llm_config:");
  expect(message).toContain('provider\n  Input should be "openai"');
});

test.each(["", "        temperature: ${input.t}\n"])("a tagged literal block field is checked: %p", (beside) => {
  const text = flow(
    "    llm_config:\n      provider: anthropic\n      anthropic:\n" +
      beside +
      "        thinking: {type: enabled, budget_tokens: !int 10}\n",
  );
  const message = loadFails(text).message;
  expect(message).toStartWith("node 'a': llm_config:");
  expect(message).toContain("greater than or equal to 1024");
});

test("a tagged literal azure allows a literal base url", () => {
  const agent = agentOf(
    flow("    llm_config:\n      provider: !str azure\n      model: ${input.m}\n      base_url: https://x.openai.azure.com\n"),
  );
  expect(((agent.ownLlmConfig as ValueTemplate).value as Record<string, unknown>)["base_url"]).toBe(
    "https://x.openai.azure.com",
  );
});

test.each(["m", "${input.m}"])("a tagged literal inherit is accepted: %p", (model) => {
  expect(agentOf(flow(`    llm_config:\n      inherit: !bool false\n      model: ${model}\n`)).llmInherit).toBe(false);
});

test("a layer of only tagged literals is a literal layer (the node layer)", () => {
  const agent = agentOf(flow("    llm_config: {provider: !str openai, model: !str 'a$$b'}\n"));
  expect(Object.getPrototypeOf(agent.ownLlmConfig)).toBe(Object.prototype);
  expect(agent.ownLlmConfig).toEqual({ provider: "openai", model: "a$b" });
});

test("a call inside an llm config string fails", () => {
  const text = flow("    llm_config:\n      model: call(pick)\n");
  const error = loadFails(text);
  expect(error.message).toContain("'llm_config.model': a `call(...)`");
  expect(error.message).not.toContain("top level");
  expect(lineOf(error)).toBe(lineHolding(text, "model:"));
});

test("a tag with no value on the layer fails (the node layer)", () => {
  const text = flow("    llm_config: !object\n");
  const error = loadFails(text);
  expect(error.message).toContain("llm_config: a tag needs a value");
  expect(lineOf(error)).toBe(lineHolding(text, "llm_config:"));
});

test.each([
  ["temparature: 1", "temparature"],
  ["model: 5", "`model:`"],
])("a literal field error in a value layer points at the field: %p", (line, needle) => {
  const text = flow(`    llm_config:\n      provider: \${input.p}\n      ${line}\n`);
  const error = loadFails(text);
  expect(error.message).toStartWith("node 'a': llm_config:");
  expect(error.message).toContain(needle);
  expect(lineOf(error)).toBe(lineHolding(text, line));
});

// ---------- the adaptive gate's layer ----------

const ASK = `\
id: f
name: f
input: {proj: !str , m: !str , t: !float }
nodes:
  ask:
    kind: human_input
    input: {proj: '\${input.proj}'}
    adaptive_questions:
      prompt: "Q \${proj}"
      llm_config:
LLM
output: \${ask.output}
`;

const ASK_SITE = "node 'ask' adaptive_questions.llm_config";

/** Return a flow whose adaptive gate has `llmLines` under `llm_config:` (indented 8). */
const ask = (llmLines: string) => ASK.replace("LLM\n", llmLines);

/** Assemble flow `text` and return the gate `ask`'s descriptor as its build received it. */
function gateOf(text: string): HumanInputDescriptor {
  const build = spyOn(HumanInputDescriptor.prototype, "build");
  try {
    assembleText(text, "f.yaml");
    return build.mock.contexts[0] as HumanInputDescriptor;
  } finally {
    build.mockRestore();
  }
}

test("the gate's value layer loads as a template ranged on the gate", () => {
  const text = ask("        model: ${input.m}\n");
  const gate = gateOf(text);
  const layer = (gate.adaptiveQuestions as Record<string, unknown>)["llm_config"];
  expect(layer).toBeInstanceOf(ValueTemplate);
  expect((layer as ValueTemplate).value).toEqual({ model: "${input.m}" });
  expect(gate.fieldRanges.get(siteKey("llm_config", "model"))!.startLine).toBe(lineHolding(text, "model:"));
});

test("the gate's literal layer stays a record", () => {
  const gate = gateOf(ask("        provider: openai\n        model: gpt-x\n"));
  const layer = (gate.adaptiveQuestions as Record<string, unknown>)["llm_config"];
  expect(Object.getPrototypeOf(layer)).toBe(Object.prototype);
  expect(layer).toEqual({ provider: "openai", model: "gpt-x" });
  expect(gate.fieldRanges.has(siteKey("llm_config", "model"))).toBe(false);
});

test("a composer tag on the whole mapping names the gate", () => {
  const text = ASK.replace("      llm_config:\nLLM\n", "      llm_config: !object {model: m}\n");
  expect(loadFails(text).message).toStartWith(`${ASK_SITE}: a tag on the whole mapping`);
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  { id: "load-agent/compose_llm_config_values_load__a_call_inside_an_llm_config_string_fails", call: assembled },
  { id: "load-agent/compose_llm_config_values_load__a_literal_block_field_beside_a_value_is_checked", call: assembled },
  { id: "load-agent/compose_llm_config_values_load__a_tag_on_the_whole_mapping_is_rejected", call: assembled },
  { id: "load-agent/compose_llm_config_values_load__a_whole_record_type_declaring_api_key_fails", call: assembled },
  { id: "load-agent/compose_llm_config_values_load__an_unknown_block_key_beside_a_value_fails", call: assembled },
  { id: "load-agent/compose_llm_config_values_load__api_key_from_a_value_fails", call: assembled },
  { id: "load-agent/compose_llm_config_values_load__base_url_from_a_value_fails_beside_azure", call: assembled },
  { id: "load-agent/compose_llm_config_values_load__inherit_from_a_value_fails", call: assembled },
  { id: "load-agent/compose_llm_config_values_load__node_layer_not_a_mapping", call: assembled },
  { id: "load-agent/compose_llm_config_values_load__tag_with_no_value", call: assembled },
  { id: "load-agent/compose_llm_config_values_load__unknown_key_beside_a_value", call: assembled },
])("assemble: $id", runCase);
