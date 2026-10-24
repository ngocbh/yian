/** A map body can't declare an inbox input: there are no per-instance inboxes. */
import { expect, test } from "bun:test";

import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { inboxType, Type, ValueKind } from "../../src/typesys/values.ts";
import { assembleText, handChild, loadError } from "./declarations.ts";

const STR = Type.scalar(ValueKind.STRING);
const inboxParam = (name: string) =>
  new Parameter(name, { contract: new ValueContract(inboxType(STR)), required: true });
/** A child that takes inbox `m` and returns a str. */
const SINK_STR = handChild({
  inputs: [inboxParam("m"), new Parameter("x")],
  producers: { p: new ValueContract(STR) },
  output: "${p.output}",
});

test("a map whose body declares an inbox input is refused", () => {
  const text = `id: f
name: f
input:
  items: !list(str)
nodes:
  e:
    kind: map
    call: sink_str
    over: \${input.items}
    input:
      x: \${item}
output: \${e.output}
`;
  expect(loadError(() => assembleText(text, "flow.yaml", { children: { sink_str: SINK_STR } })).message).toContain(
    "map node 'e': callable 'sink_str' declares inbox input 'm'; a map body can't read an inbox (per-instance inboxes " +
      "aren't supported yet)",
  );
});
