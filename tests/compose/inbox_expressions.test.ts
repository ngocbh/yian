/** Asserts, constructors and llm_config values can't reach an inbox. */
import { expect, test } from "bun:test";

import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { inboxType, Type, ValueKind } from "../../src/typesys/values.ts";
import { assembleText, handChild, loadError } from "./declarations.ts";

const STR = Type.scalar(ValueKind.STRING);
const inboxParam = (name: string) =>
  new Parameter(name, { contract: new ValueContract(inboxType(STR)), required: true });
/** A child that takes inbox `m` and returns nothing. */
const SINK = handChild({ inputs: [inboxParam("m")], output: null });
/** The node every refused row adds, so `m` is read once at an allowed site. */
const KEEP = `  keep:
    kind: call
    call: sink
    input:
      m: \${input.m}
`;

/** A flow with inbox `m`, `top` before `nodes:`, `nodes` before KEEP. */
const flow = (top: string, nodes = "") => `id: f
name: f
typedefs:
  Chat: Inbox(str)
input:
  m: !Inbox(str)
${top}nodes:
${nodes}${KEEP}output: null
`;

const refused = (text: string) => loadError(() => assembleText(text, "flow.yaml", { children: { sink: SINK } })).message;

test("the flow loads with no assert", () => {
  assembleText(flow(""), "flow.yaml", { children: { sink: SINK } });
});

test.each([
  ["a flow assert", flow("asserts:\n  - ${input.m != null}\n"), "${input.m} is an inbox; an assert can't read it"],
  ["a shorthand flow assert", flow("asserts:\n  - ${m != null}\n"), "${input.m} is an inbox; an assert can't read it"],
  [
    "a call node assert on its inbox local",
    flow("", "  s:\n    kind: call\n    call: sink\n    input:\n      m: ${input.m}\n    asserts:\n      - ${m != null}\n"),
    "${m} is an inbox; an assert can't read it",
  ],
  [
    "a constructor of an inbox type",
    flow("", "  c:\n    kind: code\n    code: return 1\n    input:\n      t: ${Chat(\"m\")}\n"),
    "`Chat` is an inbox type; it has no constructor",
  ],
  [
    "an llm_config value",
    flow("", "  a:\n    kind: agent\n    prompt: hi\n    llm_config:\n      model: ${input.m}\n"),
    "node 'a' llm_config: ${input.m} is an inbox",
  ],
])("refuses %s", (_label, text, message) => {
  expect(refused(text)).toContain(message);
});
