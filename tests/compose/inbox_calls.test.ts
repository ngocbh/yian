/** A child's Inbox input takes only an inbox of the calling flow, whole. */
import { expect, test } from "bun:test";

import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { inboxType, Type, ValueKind } from "../../src/typesys/values.ts";
import { assembleText, handChild, loadError } from "../helpers/declarations.ts";

const STR = Type.scalar(ValueKind.STRING);
const inboxParam = (name: string, element: Type = STR) =>
  new Parameter(name, { contract: new ValueContract(inboxType(element)), required: true });
/** A child that takes inbox `m` and returns nothing. */
const SINK = handChild({ inputs: [inboxParam("m")], output: null });
/** The node every refused row adds, so `m` is read once at an allowed site. */
const KEEP = `  keep:
    kind: call
    call: sink
    input:
      m: \${input.m}
`;

const flow = (source: string) => `id: f
name: f
input:
  m: !Inbox(str)
  t: !str
nodes:
  s:
    kind: call
    call: sink
    input:
      m: ${source}
${KEEP}output: null
`;

test("a child's Inbox input takes the flow's inbox", () => {
  assembleText(flow("\${input.m}"), "flow.yaml", { children: { sink: SINK } });
});

test.each([
  ["a literal", "{inbox: m}"],
  ["a text input", "\${input.t}"],
  ["a field of the inbox", "\${input.m.x}"],
  ["a record holding it", "\n        a: \${input.m}"],
])("a child's Inbox input refuses %s", (_label, source) => {
  expect(loadError(() => assembleText(flow(source), "flow.yaml", { children: { sink: SINK } })).message).toContain(
    "an inbox; bind it",
  );
});

test("an element mismatch is refused", () => {
  const intSink = handChild({ inputs: [inboxParam("m", Type.scalar(ValueKind.INTEGER))], output: null });
  const text = `id: f\nname: f\ninput:\n  m: !Inbox(str)\nnodes:\n  s:\n    kind: call\n    call: sink\n    input:\n      m: \${input.m}\noutput: null\n`;
  expect(loadError(() => assembleText(text, "flow.yaml", { children: { sink: intSink } })).message).toContain(
    "Inbox(int)",
  );
});
