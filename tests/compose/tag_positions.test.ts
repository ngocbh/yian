/**
 * Tags outside typed slots.
 *
 * A tag types a flow input, a binding, a flow output, an agent variable, or a value inside one. A
 * tag anywhere else (on `kind:`, a node's `output:` declaration, a typedef, a mapping key, ...)
 * would be silently ignored, so it is a located load error. Tags under top-level `x-*` keys are
 * exempt unless used in a slot. A tag on a CASE branch makes the branch a value, even a word naming
 * a node, so its error shows the value entry to write; a tagged case `output:` is rejected at the
 * `output:`, like an untagged one.
 */

import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, test } from "bun:test";

import { LoadCtx, loadFlowFile } from "../../src/compose/loader.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { forget } from "../../src/defaults.ts";
import {
  assembled,
  assembleText,
  type CaseRow,
  CONFORMANCE_DIR,
  linkConformanceFixtures,
  loadConformanceFlow,
  loadError,
  loadErrorAsync,
  runCase,
} from "../helpers/declarations.ts";

afterEach(() => forget());

/** Return the message for a tag at `key:`, which is not a typed slot. */
const generic = (key: string) =>
  "a tag types a value — a flow input, a binding, a flow output, an agent variable, an " +
  `\`llm_config\` value, a value entry in \`nodes:\`, or a value inside one; \`${key}:\` is not one`;

const TYPEDEF = "typedef 'Buy': a type definition is written as a type, without a tag";

/** Return the 1-based line number of the line equal to `line` in `text`. */
const lineOf = (text: string, line: string) => text.split("\n").indexOf(line) + 1;

/** Expect loading `text` to fail with a message ending in `message`, located at `line`. */
async function expectLoadError(
  text: string,
  message: string,
  line: string,
  options: Parameters<typeof loadConformanceFlow>[1] = {},
): Promise<void> {
  const error = await loadErrorAsync(loadConformanceFlow(text, options));
  expect(error.diagnostic.message.endsWith(message)).toBe(true);
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(lineOf(text, line));
}

const ECHO = "./fixtures/compose_codefns.ts:echo";

const CODE = `id: f
name: f
nodes:
  n:
    kind: code
    code: ${ECHO}
    input:
      qty: 1
    output: int
output: \${n.output}
`;

// ---------- typed slots ----------

const COMPACT = `id: f
name: f
kind: code
code: ${ECHO}
input:
  qty: !int
output: int
`;

const SINGLE_NODE_DEF = `id: f
name: f
defs:
  echo:
    kind: code
    code: ${ECHO}
    input:
      qty: !int
    output: int
nodes:
  n:
    kind: call
    call: echo
    input:
      qty: 1
output: \${n.output}
`;

const AGENT_VARIABLE = `id: f
name: f
nodes:
  writer:
    kind: agent
    variables:
      tone: !str concise
      extra: !object {depth: !int 2}
    prompt: "\${tone}"
    output: str
output: \${writer.output}
`;

test.each([
  ["flow-input", CODE.replace("name: f\n", "name: f\ninput:\n  n: !int 3\n  m: !str\n")],
  ["binding", CODE.replace("      qty: 1", "      qty: !int 1")],
  ["nested-binding", CODE.replace("      qty: 1", "      qty: [1, !int 2]")],
  ["flow-output", CODE.replace("output: ${n.output}", "output: !int ${n.output}")],
  ["flow-output-field", CODE.replace("output: ${n.output}", "output:\n  a: !int ${n.output}")],
  ["compact-input", COMPACT],
  ["single-node-def-input", SINGLE_NODE_DEF],
  ["agent-variable", AGENT_VARIABLE],
])("tags in typed slots load: %s", async (_site, text) => {
  await loadConformanceFlow(text);
});

// ---------- positions that are not typed slots ----------

test("a tag on kind is rejected at its line", async () => {
  await expectLoadError(CODE.replace("kind: code", "kind: !str code"), generic("kind"), "    kind: !str code");
});

test("a tag on prompt is rejected at its line", async () => {
  const text = `id: f
name: f
nodes:
  writer:
    kind: agent
    prompt: !str write
    output: str
output: \${writer.output}
`;
  await expectLoadError(text, generic("prompt"), "    prompt: !str write");
});

test("a tag on nodes is rejected at its line", async () => {
  await expectLoadError(CODE.replace("nodes:", "nodes: !Nodes"), generic("nodes"), "nodes: !Nodes");
});

/** Load `text`, expecting a load error with the whole `message` at the line equal to `line`. */
function expectError(text: string, message: string, line: string): void {
  const error = loadError(() => assembleText(text));
  expect(error.message).toBe(message);
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(text.split("\n").indexOf(line) + 1);
}

test("a tagged case branch naming a node is a value", () => {
  const text = `id: f
name: f
input:
  score: !str
nodes:
  route:
    kind: case
    on: \${input.score}
    cases:
      - when: high
        then: !str hot
    else: cold
  hot:
    kind: code
    code: 'return "hot"'
    output: str
  cold:
    kind: code
    code: 'return "cold"'
    output: str
output: \${hot.output | cold.output}
`;
  // The tag makes the branch the text `hot`, as in a value entry, not the node.
  expectError(
    text,
    "case node 'route' then: a branch names a node or calls a flow; put the value " +
      "in its own entry in `nodes:` and name it: `route_then_0: !str hot` and " +
      "`then: route_then_0`",
    "        then: !str hot",
  );
});

test("a tag on map over is rejected at its line", async () => {
  const text = `id: f
name: f
nodes:
  each:
    kind: map
    over: !list(int) [1, 2]
    input:
      x: \${item}
    nodes:
      echo:
        kind: code
        code: ${ECHO}
        input:
          value: \${input.x}
        output: int
    output: \${echo.output}
output: \${each.output}
`;
  await expectLoadError(text, generic("over"), "    over: !list(int) [1, 2]");
});

test("a tag inside a non-slot value names its key", async () => {
  const line = "env: {sampling: {temperature: !float 0.5}}";
  await expectLoadError(CODE.replace("name: f\n", `name: f\n${line}\n`), generic("temperature"), line);
});

test("a tagged mapping key gets its own message", async () => {
  await expectLoadError(
    CODE.replace("      qty: 1", "      !int qty: 1"),
    "`!int qty:` tags a mapping key; a tag types a value, so put it after the colon: `qty: !int <value>`",
    "      !int qty: 1",
  );
});

const CASE_ON = `id: f
name: f
input:
  score: !str
nodes:
  route:
    kind: case
    on: \${input.score}
    cases:
      - when: high
        then: one
    else: two
  one:
    kind: code
    code: |
      return 1
    output: int
  two:
    kind: code
    code: |
      return 2
    output: int
output: \${route.output}
`;

test("a tagged key message spells the key as written", async () => {
  await expectLoadError(
    CASE_ON.replace("    on: ${input.score}", "    !str on: ${input.score}"),
    "`!str on:` tags a mapping key; a tag types a value, so put it after the colon: `on: !str <value>`",
    "    !str on: ${input.score}",
  );
});

test("the generic message spells the key as written", async () => {
  await expectLoadError(
    CASE_ON.replace("    on: ${input.score}", "    on: !str ${input.score}"),
    generic("on"),
    "    on: !str ${input.score}",
  );
});

test("a null key is reported before a tag under it", async () => {
  const text = `id: f
name: f
nodes:
  settle:
    kind: tool
    tool_id: sleep
    input:
      until:
        null: !int 3
output: \${settle.output}
`;
  await expectLoadError(
    text,
    "the key `null` was read by YAML as null, but names in a flow must be text. " +
      'Quote it ("null") or choose another name.',
    "        null: !int 3",
  );
});

// ---------- type-definition positions ----------

test.each([
  ["a tagged typedef", "typedefs:\n  Buy: !Rec\n    qty: int\n", "  Buy: !Rec"],
  ["a tagged typedef field", "typedefs:\n  Buy:\n    qty: !int int\n", "    qty: !int int"],
  ["a tagged typedef field without a value", "typedefs:\n  Buy:\n    qty: !int\n", "    qty: !int"],
])("%s is rejected at its line", async (_label, typedefs, line) => {
  await expectLoadError(CODE.replace("name: f\n", `name: f\n${typedefs}`), TYPEDEF, line);
});

test.each([
  [
    "a tagged node output declaration is rejected at its line",
    "    output: !int",
    "`output:` declares a type; write `output: int`, without a tag",
    "    output: !int",
  ],
  [
    "a tagged node output with a type after the tag spells the type",
    "    output: !Optional(int) Optional(int)",
    "`output:` declares a type; write `output: Optional(int)`, without a tag",
    "    output: !Optional(int) Optional(int)",
  ],
  [
    "a tagged node output field is rejected at its line",
    "    output:\n      qty: !int",
    "`output:` declares a type; write `qty: int`, without a tag",
    "      qty: !int",
  ],
  [
    "a tagged list item in a node output says the item is tagged",
    "    output:\n      qty:\n        - !int",
    "`output:` declares a type; an item of `qty:` is tagged `!int`; write the type without a tag",
    "        - !int",
  ],
])("%s", async (_label, output, message, line) => {
  await expectLoadError(CODE.replace("    output: int", output), message, line);
});

test("a tagged compact output is rejected at its line", async () => {
  await expectLoadError(
    COMPACT.replace("output: int", "output: !int"),
    "`output:` declares a type; write `output: int`, without a tag",
    "output: !int",
  );
});

test("a tagged case output is rejected at its line", () => {
  const text = `id: f
name: f
input:
  score: !str
nodes:
  route:
    kind: case
    on: \${input.score}
    cases:
      - when: high
        then: one
    else: one
    output: !int
  one:
    kind: code
    code: |
      return 1
    output: int
output: \${route.output}
`;
  expectError(
    text,
    "case node 'route': a `case` routes and has no `output:`; tag its result where " +
      "you read it, e.g. `x: !int ${route.output}`",
    "    output: !int",
  );
});

test.each([
  ["agent", "    kind: agent\n    prompt: write\n"],
  ["tool", "    kind: tool\n    tool_id: place_order\n"],
])("a tagged %s output is rejected at its line", async (_kind, head) => {
  await expectLoadError(
    `id: f\nname: f\nnodes:\n  n:\n${head}    output: !str\noutput: \${n.output}\n`,
    "`output:` declares a type; write `output: str`, without a tag",
    "    output: !str",
  );
});

test("a tagged single-node def output is rejected at its line", async () => {
  await expectLoadError(
    SINGLE_NODE_DEF.replace("    output: int", "    output: !int"),
    "`output:` declares a type; write `output: int`, without a tag",
    "    output: !int",
  );
});

// ---------- defs, inline bodies, and uses: children ----------

const DEF_FLOW = `id: f
name: f
defs:
  sub:
    input:
      x: !int
    nodes:
      echo:
        kind: code
        code: ${ECHO}
        input:
          value: !int \${input.x}
        output: int
    output: \${echo.output}
nodes:
  n:
    kind: call
    call: sub
    input:
      x: 1
output: \${n.output}
`;

const INLINE_FLOW = `id: f
name: f
nodes:
  each:
    kind: map
    over: [1, 2]
    input:
      x: \${item}
    nodes:
      echo:
        kind: code
        code: ${ECHO}
        input:
          value: !int \${input.x}
        output: int
    output: !int \${echo.output}
output: \${each.output}
`;

describe.each([
  ["a def", DEF_FLOW],
  ["an inline body", INLINE_FLOW],
])("%s", (_site, text) => {
  test("tags in its slots load", async () => {
    await loadConformanceFlow(text);
  });

  test("a tag outside a slot is rejected at its line", async () => {
    await expectLoadError(
      text.replace("        kind: code", "        kind: !str code"),
      generic("kind"),
      "        kind: !str code",
    );
  });
});

const CHILD = `id: child
name: child
input:
  x: !int
nodes:
  echo:
    kind: code
    code: ${ECHO}
    input:
      value: !int \${input.x}
    output: int
output: \${echo.output}
`;

const PARENT = `id: parent
name: parent
uses:
  child: child
nodes:
  n:
    kind: call
    call: child
    input:
      x: 1
output: \${n.output}
`;

/** Load `PARENT` with `child` written as `child.yaml` in a temporary search directory. */
async function loadWithChild(child: string) {
  const tmp = mkdtempSync(join(tmpdir(), "yian-tag-positions-"));
  try {
    linkConformanceFixtures(tmp);
    writeFileSync(join(tmp, "child.yaml"), child);
    return await loadConformanceFlow(PARENT, { searchPaths: [tmp] });
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
}

test("tags in a uses child's slots load", async () => {
  await loadWithChild(CHILD);
});

test("a tag outside a slot in a uses child is rejected", async () => {
  const error = await loadErrorAsync(loadWithChild(CHILD.replace("kind: code", "kind: !str code")));
  expect(error.message).toContain(generic("kind"));
});

// ---------- x-* keys and anchors ----------

test("a tag under an extension key used through an alias in a slot loads", async () => {
  const text = `id: f
name: f
x-defaults: &d !int 1
nodes:
  n:
    kind: code
    code: ${ECHO}
    input:
      qty: *d
    output: int
output: \${n.output}
`;
  const loaded = await loadConformanceFlow(text);
  expect(loaded.compiled.wiring.get("n")!.get("qty")!.contract).not.toBeNull();
});

test("a nested tag under an extension key used through a merge loads", async () => {
  const text = `id: f
name: f
x-defaults: &d
  qty: !int 1
nodes:
  n:
    kind: code
    code: ${ECHO}
    input:
      <<: *d
    output: int
output: \${n.output}
`;
  await loadConformanceFlow(text);
});

test("an unused tag under an extension key loads", async () => {
  await loadConformanceFlow(CODE.replace("name: f\n", "name: f\nx-notes:\n  a: !Anything 1\n"));
});

test("a tag under an extension key aliased outside a slot is rejected", async () => {
  const text = CODE.replace("name: f\n", "name: f\nx-kind: &k !str code\n").replace("    kind: code", "    kind: *k");
  const error = await loadErrorAsync(loadConformanceFlow(text));
  expect(error.diagnostic.message.endsWith(generic("kind"))).toBe(true);
});

// ---------- per-load state ----------

test("loading the same document twice behaves the same", async () => {
  const loadOnce = (document: SourceDocument) => loadFlowFile(document, null, null, new LoadCtx(), CONFORMANCE_DIR);
  const good = SourceDocument.parse(DEF_FLOW);
  const first = await loadOnce(good);
  const second = await loadOnce(good);
  expect(second.compiled.wiring).toEqual(first.compiled.wiring);

  const bad = SourceDocument.parse(DEF_FLOW.replace("    nodes:", "    nodes: !N"));
  const messages: string[] = [];
  for (let attempt = 0; attempt < 2; attempt++) messages.push((await loadErrorAsync(loadOnce(bad))).message);
  expect(messages[1]).toBe(messages[0]!);
  expect(messages[0]).toContain(generic("nodes"));
});

// ---------- bindings ----------

test("a tag on a binding value loads and is consumed", async () => {
  const loaded = await loadConformanceFlow(CODE.replace("      qty: 1", "      qty: !int 1"));
  expect(loaded.compiled.wiring.get("n")!.get("qty")!.source).toBe(1n);
});

test("the message names the typed slots", async () => {
  const error = await loadErrorAsync(loadConformanceFlow(CODE.replace("kind: code", "kind: !str code")));
  expect(error.message).toContain(generic("kind"));
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  { id: "load-types/compose_tag_positions__a_tagged_case_branch_naming_a_node_is_a_value", call: assembled },
  { id: "load-types/compose_tag_positions__tagged_case_output_is_rejected_at_its_line", call: assembled },
])("assemble: $id", runCase);
