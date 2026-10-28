/** An inbox is read only whole, into a child's Inbox input. */
import { expect, test } from "bun:test";

import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { inboxType, Type, ValueKind } from "../../src/typesys/values.ts";
import { assembleText, handChild, loadError } from "../helpers/declarations.ts";

const STR = Type.scalar(ValueKind.STRING);
const BOOL = Type.scalar(ValueKind.BOOLEAN);
const inboxParam = (name: string) =>
  new Parameter(name, { contract: new ValueContract(inboxType(STR)), required: true });
const doneParam = () => new Parameter("done", { required: true, contract: new ValueContract(BOOL) });
/** A child that takes inbox `m` and returns nothing. */
const SINK = handChild({ inputs: [inboxParam("m")], output: null });
/** A child that takes inbox `m` and returns a str. */
const SINK_STR = handChild({ inputs: [inboxParam("m")], producers: { p: new ValueContract(STR) }, output: "${p.output}" });
/** A loop body that takes inbox `m` and a bool `done`, and returns `{done}`. */
const BODY = handChild({ inputs: [inboxParam("m"), doneParam()], output: { done: "${input.done}" } });
/** A loop body that takes only `done`, and returns it. */
const DONE_ONLY = handChild({ inputs: [doneParam()], output: { done: "${input.done}" } });
const CHILDREN = { children: { sink: SINK, sink_str: SINK_STR, body: BODY, done_only: DONE_ONLY } };
/** The node every refused row adds, so `m` is read once at an allowed site. */
const KEEP = `  keep:
    kind: call
    call: sink
    input:
      m: \${input.m}
`;

const load = (nodes: string, output = "null") =>
  assembleText(`id: f\nname: f\ninput:\n  m: !Inbox(str)\nnodes:\n${nodes}${KEEP}output: ${output}\n`, "flow.yaml", CHILDREN);
const refused = (nodes: string, output = "null") => loadError(() => load(nodes, output)).message;

test("a named call and a loop take an inbox whole", () => {
  load("");
  load(`  l:
    kind: loop
    call: body
    input:
      m: \${input.m}
      done: false
    while: not done
`);
});

test.each([
  ["a field read in a CODE input", `  c:\n    kind: code\n    code: return 1\n    input:\n      t: \${input.m.x}\n`, "has no fields"],
  ["text", `  c:\n    kind: code\n    code: return 1\n    input:\n      t: "hi \${input.m}"\n`, "not inside text"],
  ["an expression", `  c:\n    kind: code\n    code: return 1\n    input:\n      t: \${input.m :- 1}\n`, "not inside text"],
  ["a CODE input", `  c:\n    kind: code\n    code: return 1\n    input:\n      t: \${input.m}\n`, "take from it"],
  ["an AGENT input", `  a:\n    kind: agent\n    prompt: hi\n    input:\n      t: \${input.m}\n`, "take from it"],
  ["a TOOL input", `  t:\n    kind: tool\n    tool_id: f\n    input:\n      t: \${input.m}\n`, "take from it"],
  ["a record in a CODE input", `  c:\n    kind: code\n    code: return 1\n    input:\n      t:\n        a: \${input.m}\n`, "is read only whole"],
  ["a case", `  k:\n    kind: case\n    cases:\n      - when: \${input.m == null}\n        then: done\n  done: !int 1\n`, "not inside text"],
  ["a call directive", `  c:\n    kind: code\n    code: return 1\n    input:\n      t: call(sink_str, m=\${input.m})\n`, "named `kind: call`"],
  ["a loop field the body doesn't take", `  l:\n    kind: loop\n    call: done_only\n    input:\n      m: \${input.m}\n      done: false\n    while: not done\n`, "isn't an Inbox input"],
])("refuses %s", (_label, nodes, message) => {
  expect(refused(nodes)).toContain(message);
});

test("a flow output can't read an inbox", () => {
  expect(refused("", "${input.m}")).toContain("is read only whole");
});

test("the refusal names the site and the inbox", () => {
  expect(refused(`  c:\n    kind: code\n    code: return 1\n    input:\n      t: \${input.m}\n`)).toContain(
    "node 'c' input 't' from: ${input.m} is an inbox; it is read only whole, bound to a child's Inbox input " +
      "(take from it with `tool_id: take`)",
  );
});

test("an inbox input nothing reads is a load error at its declaration", () => {
  const error = loadError(() =>
    assembleText(
      "id: f\nname: f\ninput:\n  m: !Inbox(str)\nnodes:\n  c:\n    kind: code\n    code: return 1\noutput: null\n",
      "flow.yaml",
      CHILDREN,
    ),
  );
  expect(error.message).toContain("flow has unread inbox inputs");
  expect(error.message).toContain("inbox input `m` is never read; take from it, or pass it to a call");
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(4);
});

test("a refused read wins over the unread inbox", () => {
  const message = loadError(() =>
    assembleText(
      "id: f\nname: f\ninput:\n  m: !Inbox(str)\nnodes:\n  c:\n    kind: code\n    code: return 1\n    input:\n      t: ${input.m}\noutput: null\n",
      "flow.yaml",
      CHILDREN,
    ),
  ).message;
  expect(message).toContain("take from it");
  expect(message).not.toContain("never read");
});

test("a case value reading an inbox is a load error, not a crash", () => {
  const nodes = `  k:\n    kind: case\n    on: \${input.t}\n    cases:\n      - when: a\n        then: \${input.m}\n    else: x\n`;
  const text = `id: f\nname: f\ninput:\n  m: !Inbox(str)\n  t: !str\nnodes:\n${nodes}${KEEP}output: null\n`;
  expect(loadError(() => assembleText(text, "flow.yaml", CHILDREN)).message).toContain(
    "case node 'k' then: a branch names a node or calls a flow",
  );
});
