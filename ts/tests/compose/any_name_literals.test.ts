/**
 * A literal under any name is an ordinary value: a name like `password` or `api_key` loads in every
 * slot, and the literal is part of the manifest and fingerprint. A port of the rows of
 * `tests/compose/test_any_name_literals.py` the cross-flow binding check takes part in,
 * assembling each flow with `assembleText` where Python calls `load_flow`.
 *
 * Ported: `test_a_value_entry_call_argument_under_a_credential_name_loads`,
 * `test_a_map_over_literal_under_a_credential_name_loads` and
 * `test_an_inline_call_in_a_node_assert_shows_the_assert_under_any_keyword` (the node-assert check
 * refuses the call before the `g` def is resolved).
 *
 * Changed: the `g` and `leaf` defs are built by hand (`handChild`: the def's parameters read from
 * its `input:` text, its typed node as a producer with its output contract), since def bodies load
 * in T3-22. CODE bodies are TS; load doesn't run them.
 *
 * Not ported (not T3-19 rows), decided by already-ported modules (the manifest and fingerprint, the
 * tags reader, inline calls): `test_a_node_env_literal_under_a_credential_name_loads` (4),
 * `test_a_flow_env_literal_under_a_credential_name_loads`, the compact-flow param of
 * `test_a_def_env_literal_under_a_credential_name_loads`,
 * `test_an_input_default_under_a_credential_name_loads` (2),
 * `test_an_agent_variable_under_a_credential_name_loads`,
 * `test_a_node_input_literal_under_a_credential_name_is_identity` (11),
 * `test_a_nested_key_under_a_credential_name_loads`,
 * `test_a_typed_binding_under_a_credential_name_loads` (its case is pinned in `tags.test.ts`),
 * `test_a_flow_output_under_a_credential_name_is_identity`,
 * `test_a_value_entry_under_a_credential_name_is_identity` (its run is T4's),
 * `test_a_call_argument_under_a_credential_name_is_identity`,
 * `test_a_case_call_target_with_a_credential_keyword_loads`,
 * `test_an_inline_call_flow_output_with_a_credential_keyword_loads` and
 * `test_a_malformed_call_with_a_credential_keyword_quotes_its_text` (pinned in
 * `inline_calls.test.ts`).
 *
 * Later (T3-22): the multi-node-def and single-node-def params of
 * `test_a_def_env_literal_under_a_credential_name_loads` and
 * `test_a_def_input_default_under_a_credential_name_loads` (def loading).
 */

import { expect, test } from "bun:test";

import { flowFingerprint, flowManifest } from "../../src/compile/manifest.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { assembleText, handChild, loadError, parametersFromYaml } from "./declarations.ts";

const STR = new ValueContract(Type.scalar(ValueKind.STRING));

test("a value entry call argument under a credential name loads", () => {
  const g = handChild({
    inputs: parametersFromYaml("input:\n  api_key: !str\n", new Map()),
    producers: { r: STR },
    output: "${r.output}",
  });

  assembleText(
    "id: f\nname: f\n" +
      "defs:\n  g:\n    input:\n      api_key: !str\n    nodes:\n      r: !str ok\n    output: ${r.output}\n" +
      "nodes:\n" +
      '  x: !str call(g, api_key="abc")\n' +
      "output: ${x.output}\n",
    "flow.yaml",
    { children: { g } },
  );
});

test("a map over literal under a credential name loads", () => {
  const leaf = handChild({
    inputs: parametersFromYaml("input:\n  x: !object\n", new Map()),
    producers: { up: STR },
    output: "${up.output}",
  });
  const flow = (value: string) =>
    "id: p\nname: p\n" +
    "defs:\n  leaf:\n    input:\n      x: !object\n    nodes:\n" +
    "      up:\n        kind: code\n        input:\n          x: ${input.x}\n" +
    "        output: str\n        code: return 'x'\n    output: ${up.output}\n" +
    "nodes:\n  m:\n    kind: map\n" +
    `    over: [{password: ${value}}, {password: b}]\n` +
    "    call: leaf\n    input:\n      x: ${item}\n" +
    "output: ${m.output}\n";
  const load = (value: string) => assembleText(flow(value), "flow.yaml", { children: { leaf } }).flow.compiled;

  const first = load("hunter2");

  expect(flowFingerprint(first)).not.toBe(flowFingerprint(load("hunter3")));
  expect(JSON.stringify(flowManifest(first))).toContain("hunter2");
});

test("an inline call in a node assert shows the assert under any keyword", () => {
  const text =
    "id: f\nname: f\n" +
    "defs:\n  g:\n    input:\n      password: !Optional(str) null\n    nodes:\n" +
    "      run:\n        kind: code\n        output: str\n        code: |\n          return 'x'\n" +
    "    output: ${run.output}\n" +
    "nodes:\n  b:\n    kind: code\n    output: str\n    code: |\n      return 'x'\n" +
    "    asserts:\n      - call(g, password=x)\n" +
    "output: ${b.output}\n";

  expect(loadError(() => assembleText(text)).message).toContain("assert 'call(g, password=x)'");
});
