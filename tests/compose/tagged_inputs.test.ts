/**
 * Flow inputs declared with YAML tags (`qty: !int`, `history: !list(ChatMsg) []`). Ints are
 * `bigint`; a missing-field list is written in the text form (`["kind"]`).
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, afterEach, beforeEach, describe, expect, test } from "bun:test";

import { readFlowInputs } from "../../src/compose/helpers/types.ts";
import { START_ID } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import type { StartNode } from "../../src/nodes/start/node.ts";
import { runFlow } from "../../src/run.ts";
import { readTypedefs } from "../../src/typesys/types.ts";
import { isListKind } from "../../src/typesys/values.ts";
import { type CaseRow, inputsAt, loadError, loadErrorAsync, parametersFromYaml, runCase } from "./declarations.ts";
import { hostEnv } from "../engine/mcp_helpers.ts";

// Pinned before `REG` is read too, so neither depends on the shell's `YIAN_*` limits.
install(BUILTIN);
beforeEach(() => install(BUILTIN));
afterEach(forget);

const REG = readTypedefs({
  Cfg: { kind: "Literal(cfg)", a: "int" },
  ChatMsg: { role: "str", content: "str" },
});

const DECLARATIONS = `\
qty: !int
history: !list(ChatMsg) []
model: !Optional(str)
note: !Optional(str) null
cfg: !Cfg {kind: cfg, a: 1}
s: !str \${x}
`;

/** Indent each line of `text` by `by` (default two spaces). */
function indented(text: string, by = "  "): string {
  return text.replace(/^(?=.)/gm, by);
}

const HEADER = `\
id: tagged_inputs
name: tagged_inputs
typedefs:
  Cfg: {kind: Literal(cfg), a: int}
  ChatMsg: {role: str, content: str}
`;

const WORKER = "kind: code\noutput: object\ncode: return {...inputs}\n";

/** Bind every declared input by name, as a multi-node body must. */
function workerBindings(declarations: string): string {
  const names = declarations.split("\n").filter((line) => line !== "").map((line) => line.split(":")[0]);
  return "input:\n" + names.map((name) => `  ${name}: \${input.${name}}\n`).join("");
}

const PLACEMENTS = ["root", "definition", "flat-definition", "compact"] as const;

/** Spell one flow whose `input:` holds `declarations` at `placement`. */
function flowAt(placement: (typeof PLACEMENTS)[number], declarations: string): string {
  const block = indented(declarations);
  if (placement === "root") {
    const worker = WORKER + workerBindings(declarations);
    return `${HEADER}input:\n${block}nodes:\n  worker:\n${indented(worker, "    ")}output: \${worker.output}\n`;
  }
  if (placement === "compact") return `${HEADER}input:\n${block}${WORKER}`;
  const caller =
    "input:\n  qty: !int\nnodes:\n  invoke:\n    kind: call\n    call: child\n" +
    "    input:\n      qty: ${input.qty}\noutput: ${invoke.output}\n";
  const child =
    placement === "definition"
      ? `input:\n${block}nodes:\n  worker:\n${indented(WORKER + workerBindings(declarations), "    ")}` +
        "output: ${worker.output}\n"
      : `input:\n${block}${WORKER}`;
  return `${HEADER}defs:\n  child:\n${indented(child, "    ")}${caller}`;
}

describe("tagged inputs run", () => {
  const dir = mkdtempSync(join(tmpdir(), "yian-tagged-inputs-"));
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  test.each(PLACEMENTS)("tagged inputs load and run with their defaults: %s", async (placement) => {
    const result = await runFlow(await loadFlow(flowAt(placement, DECLARATIONS)), { qty: 3 }, { env: hostEnv(dir) });

    expect(result.status).toBe("succeeded");
    expect(result.output).toEqual({
      qty: 3n,
      history: [],
      model: null,
      note: null,
      cfg: { kind: "cfg", a: 1n },
      s: "${x}",
    });
  });

  test.each(["root", "compact"] as const)("a tagged input without a value is required: %s", async (placement) => {
    const result = await runFlow(await loadFlow(flowAt(placement, DECLARATIONS)), {}, { env: hostEnv(dir) });

    expect(result.status).toBe("failed");
    expect(result.failure!.code).toBe("binding.evaluate");
    expect(result.failure!.message).toContain("required input 'qty' is unbound");
  });
});

describe("tagged declarations", () => {
  test("a tag does not fill in a missing field", () => {
    const error = loadError(() => parametersFromYaml("input:\n  cfg: !Cfg {a: 1}\n", REG));
    expect(error.message).toContain('missing required fields: ["kind"]');
    const [parameter] = parametersFromYaml("input:\n  cfg: !Cfg {kind: cfg, a: 1}\n", REG);
    expect(parameter!.default).toEqual({ kind: "cfg", a: 1n });
  });

  test("a tagged parameter reads required and its default from the value", () => {
    const parameters = parametersFromYaml("input:\n" + indented(DECLARATIONS), REG);
    const shape = Object.fromEntries(
      parameters.map((parameter) => [
        parameter.name,
        [parameter.required, parameter.hasDefault, parameter.default],
      ]),
    );
    expect(shape).toEqual({
      qty: [true, false, null],
      history: [false, true, []],
      model: [false, false, null],
      note: [false, true, null],
      cfg: [false, true, { kind: "cfg", a: 1n }],
      s: [false, true, "${x}"],
    });
    const types = new Map(parameters.map((parameter) => [parameter.name, parameter.contract!.type]));
    expect(types.get("model")!.nullable).toBe(true);
    expect(isListKind(types.get("history")!.kind)).toBe(true);
    expect(parameters.every((parameter) => parameter.coerceStrings)).toBe(true);
  });

  test("a source-free input accepts the parenthesized spelling", () => {
    const [parameter] = readFlowInputs({ x: "Optional(int)" }, new Map());
    expect(parameter!.contract!.type.nullable).toBe(true);
    expect(parameter!.required).toBe(false);
  });

  test.each([
    [
      "cfg: !Cfg {a: 1}",
      "'cfg': the value is not a valid Cfg: record missing required fields: [\"kind\"]",
    ],
    ["n: !int(1..12) 13", "'n': `13` is not a valid int(1..12)"],
  ])("a tagged default must fit its tag: %p", (declaration, fragment) => {
    const text = `input:\n  qty: !int\n  ${declaration}\n`;
    expect(loadError(() => parametersFromYaml(text, REG)).message).toContain(fragment);
  });

  describe.each(PLACEMENTS)("at the %s placement", (placement) => {
    test.each([
      ["cfg: !Cfg {a: 1}", "'cfg': the value is not a valid Cfg: record missing required fields: [\"kind\"]"],
      ["n: !int(1..12) 13", "'n': `13` is not a valid int(1..12)"],
    ])("a tagged default must fit its tag: %p", async (declaration, fragment) => {
      const error = await loadErrorAsync(loadFlow(flowAt(placement, `qty: !int\n${declaration}\n`)));
      expect(error.message).toContain(fragment);
    });

    test("a tagged default with a non-string key is a load error", async () => {
      const error = await loadErrorAsync(loadFlow(flowAt(placement, "qty: !int\nflags: !object {true: 1}\n")));
      expect(error.message).toContain("the key `true` was read by YAML as a boolean");
    });
  });

  test("tagged parameter origins point at the declaration", async () => {
    const text = `${HEADER}input:\n  qty: !int\n  history: !list(ChatMsg) []\n`;
    const flow = await loadFlow(text + WORKER);
    const cursor = SourceDocument.parse(text).root.child("input");
    const resolver = flow.originResolver!;
    const start = flow.compiled.nodes.get(START_ID) as StartNode;
    const parameters = new Map(start.params!.map((parameter) => [parameter.name, parameter]));
    const history = parameters.get("history")!;
    expect(resolver.resolve(history.origin!)!.range).toEqual(cursor.keyRangeOf("history")!);
    for (const origin of [history.contract!.origins.default, history.defaultOrigin]) {
      expect(resolver.resolve(origin!)!.range).toEqual(cursor.child("history").range!);
    }
    const qty = parameters.get("qty")!;
    expect(resolver.resolve(qty.contract!.origins.default!)!.range).toEqual(cursor.child("qty").range!);
    expect(qty.defaultOrigin).toBeNull();
  });

  test.each([
    ["compact", `id: f\nname: f\ninput: [topic]\n${WORKER}`, "compact (single-node) flow: "],
    [
      "flat-definition",
      `id: f\nname: f\ndefs:\n  child:\n    input: [topic]\n${indented(WORKER, "    ")}nodes: {}\n`,
      "defs entry 'child': single-node ",
    ],
  ])("a single-node input list shows the tagged spelling: %s", async (_placement, text, prefix) => {
    const error = await loadErrorAsync(loadFlow(text));
    expect(error.message).toContain(prefix + "`input:` must be a param map (`name: !TYPE`, e.g. `topic: !str`)");
  });

  test("an unquoted date fits a str tag, as text", () => {
    const parameters = parametersFromYaml("input:\n  qty: !int\n  d: !str 2026-01-01\n", REG);
    expect(parameters[1]!.default).toBe("2026-01-01");
  });
});

// ---------- harvested conformance cases ----------

const rootInput = inputsAt(["input"]);

test.each<CaseRow>([
  { id: "load-types/compose_tagged_inputs__a_tagged_default_must_fit_its_range", call: rootInput },
  {
    id: "load-types/compose_tagged_inputs__a_tagged_record_default_must_have_every_field",
    call: rootInput,
  },
  {
    id: "load-types/engine_date_grammar__flow_default_outside_the_grammar_fails_at_load",
    call: rootInput,
  },
  { id: "types/compose_type_name_checks__empty_literal_member_in_a_tag", call: rootInput },
])("readSourceFlowInputs: $id", runCase);
