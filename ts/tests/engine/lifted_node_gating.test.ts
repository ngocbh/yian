/**
 * A node lifted out of an authored node runs on its host's routes and ordering: a port of
 * `tests/engine/test_lifted_node_gating.py`.
 *
 * Each lifting desugar pass records the node it lifted from as the host: an inline `call(...)` in
 * a node's binding, the inner calls of a nested branch `call(...)`, and an adaptive gate's
 * composer AGENT. The lifted node carries the host's CASE route edges (the else route included),
 * `depends_on:` (co-skip), and `runs_after:` (order). It also waits for the host's other data
 * inputs and is skipped when one of them would skip the host; a defaulted input (`:-`) only makes
 * it wait, and sibling calls lifted from one host still run in parallel. So no call, side effect
 * or LLM call happens on a branch the host isn't on, and the taken branch runs as before.
 *
 * Ported: every row, each parametrized row included. The `record` probes are TS's
 * (`./fixtures/compose_codefns.ts:record`), and Python's `RECORDED` is read from the run's events
 * as the tags the probe nodes (`prep`, and each `fx` and `p` inside a call) returned, in order. The
 * composer AGENT's model is a `ScriptedModel` given as `models` (`serving`), its provider and model
 * named by the run's `llmConfig`; the composer's calls are the model's requests, and a model with
 * no reply left fails any call, where Python's fake chat raises. The CODE bodies are Python's where
 * they read the same in JS (decision 11); `d`'s conditional and `audit`'s `len` are written in JS.
 * The compiled edges' `source_handle`, `input_group` and `from_` are `sourceHandle`, `inputGroup`
 * and `from`.
 *
 * Changed: `meet` is an inline CODE body that awaits a short timer, where Python's waits on a
 * barrier shared with the other `meet`; CODE bodies run in worker threads (decision 47), so
 * `test_sibling_calls_from_one_host_run_in_parallel` checks, with two workers, that both bodies
 * started before either succeeded.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { NodeSkipped, NodeStarted, NodeSucceeded, type RunEvent } from "../../src/events.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { ELSE_HANDLE } from "../../src/nodes/base.ts";
import { type RunFlowOptions, type RunResult, runFlow } from "../../src/run.ts";
import type { AskUserRequest } from "../../src/tools/registry.ts";
import { loadConformanceFlow, loadErrorAsync } from "../compose/declarations.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { serving, startedIds, textReply } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-lifted-node-gating-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

// `side` records its tag when it runs; `wrap` and `items` feed a branch call and a MAP; `pair`
// joins two strings; `meet` is `side` that takes a moment, so two `meet`s overlap when parallel.
const DEFS = `
defs:
  pair:
    input:
      x: !str
      y: !str
    nodes:
      j:
        kind: code
        input:
          x: \${input.x}
          y: \${input.y}
        output: str
        code: return inputs["x"] + inputs["y"]
    output: \${j.output}
  meet:
    input:
      tag: !str
    nodes:
      fx:
        kind: code
        code: |
          await new Promise((resolve) => setTimeout(resolve, 50))
          return inputs["tag"]
        input:
          tag: \${input.tag}
        output: str
    output: \${fx.output}
  side:
    input:
      tag: !str
    nodes:
      fx:
        kind: code
        code: ./fixtures/compose_codefns.ts:record
        input:
          tag: \${input.tag}
        output: str
    output: \${fx.output}
  wrap:
    input:
      x: !str
    nodes:
      w:
        kind: code
        input:
          x: \${input.x}
        output: str
        code: return "wrapped " + inputs["x"]
    output: \${w.output}
  items:
    input:
      tag: !str
    nodes:
      fx:
        kind: code
        code: ./fixtures/compose_codefns.ts:record
        input:
          tag: \${input.tag}
        output: str
      ls:
        kind: code
        input:
          t: \${fx.output}
        output: list(str)
        code: return [inputs["t"]]
    output: \${ls.output}
  echo:
    input:
      item: !str
    nodes:
      e:
        kind: code
        input:
          v: \${input.item}
        output: str
        code: return inputs["v"]
    output: \${e.output}
`;

const HEAD = "id: t\nname: t\ninput:\n  go: !bool\n  other: !bool\n";

// `prep` records `prep`; `pick` routes `maybe` only when `other` is true.
const PREP = `
  prep:
    kind: code
    code: ./fixtures/compose_codefns.ts:record
    input:
      tag: prep
    output: str
`;
const MAYBE = `
  pick:
    kind: case
    cases:
      - when: \${input.other}
        then: maybe
  maybe:
    kind: code
    output: str
    code: return "m"
`;

/** A flow with the shared inputs and defs, `nodes`, and `output`. */
function flow(nodes: string, output = "  r: ${gate.output}\n"): string {
  return HEAD + DEFS + "nodes:\n" + nodes + "output:\n" + output;
}

/** Load and run a flow on `go` and `other` in the test's environment; the run must succeed. */
async function run(
  text: string,
  go = false,
  other = false,
  options: RunFlowOptions = {},
): Promise<{ flow: ComposedFlow; result: RunResult }> {
  const loaded = await loadConformanceFlow(text, { env });
  const result = await runFlow(loaded, { go, other }, { env, ...options });
  expect(result.status, result.failure?.message).toBe("succeeded");
  return { flow: loaded, result };
}

/** The compiled edge named `edgeId`. */
function edge(loaded: ComposedFlow, edgeId: string) {
  const edges = loaded.compiled.edges.filter((each) => each.id === edgeId);
  expect(edges.length).toBe(1);
  return edges[0]!;
}

/** The probe nodes that record their tag: `prep`, and a `fx` or `p` inside a call. */
const PROBE = /^prep$|\/(fx|p)$/;

/** The tags the probe nodes returned, in order: what Python's `RECORDED` collects. */
const recorded = (result: RunResult) =>
  result.events
    .filter((event): event is NodeSucceeded => event instanceof NodeSucceeded && PROBE.test(event.nodeId))
    .map((event) => event.output);

/** Whether `nodeId` was skipped and never started. */
function neverRan(result: RunResult, nodeId: string): boolean {
  const skipped = result.events.filter((event) => event instanceof NodeSkipped).map((event) => event.nodeId);
  return !startedIds(result).includes(nodeId) && skipped.includes(nodeId);
}

/** The position of the first `kind` event for `nodeId`; fails the test when there is none. */
function index(result: RunResult, kind: abstract new (...args: never[]) => RunEvent, nodeId: string): number {
  const i = result.events.findIndex(
    (event) => event instanceof kind && (event as { nodeId?: string }).nodeId === nodeId,
  );
  expect(i, `${kind.name} for ${nodeId}`).toBeGreaterThanOrEqual(0);
  return i;
}

/** Whether `then` started after `first` succeeded. */
const startsAfter = (result: RunResult, first: string, then: string) =>
  index(result, NodeSucceeded, first) < index(result, NodeStarted, then);

// ---------------------------------------------------------------------------------------------
// inline calls: `call(...)` in a node's binding.
// ---------------------------------------------------------------------------------------------

const PUBLISH = `
  publish:
    kind: code
    input:
      draft: call(side, tag="publish")
    output: str
    code: return "published " + inputs["draft"]
`;
const SKIP_IT = `
  skip_it:
    kind: code
    output: str
    code: return "skipped"
`;
const GATE = `
  gate:
    kind: case
    cases:
      - when: \${input.go}
        then: publish
    else: skip_it
`;

test("inline call does not run on an untaken route", async () => {
  const { flow: loaded, result } = await run(flow(GATE + PUBLISH + SKIP_IT), false);
  expect(recorded(result)).toEqual([]);
  expect(neverRan(result, "__call_0")).toBe(true);
  expect(result.output).toEqual({ r: "skipped" });
  expect(edge(loaded, "gate->__call_0#0").sourceHandle).toBe("publish");
});

test("inline call runs on the taken route", async () => {
  const { result } = await run(flow(GATE + PUBLISH + SKIP_IT), true);
  expect(recorded(result)).toEqual(["publish"]);
  expect(result.output).toEqual({ r: "published publish" });
});

test("inline call runs after the host runs after", async () => {
  const publish = PUBLISH.replace("    kind: code\n", "    kind: code\n    runs_after: [prep]\n");
  const { flow: loaded, result } = await run(flow(GATE + PREP + publish + SKIP_IT), true);
  const ordering = edge(loaded, "prep~>__call_0#0");
  expect(ordering.ordering && ordering.optional).toBe(true);
  expect(recorded(result)).toEqual(["prep", "publish"]);
  expect(startsAfter(result, "prep", "__call_0")).toBe(true);
});

test("inline call is skipped with a host depends on", async () => {
  const publish = PUBLISH.replace("    kind: code\n", "    kind: code\n    depends_on: [maybe]\n");
  const { flow: loaded, result } = await run(flow(GATE + MAYBE + publish + SKIP_IT), true, false);
  const ordering = edge(loaded, "maybe~>__call_0#0");
  expect(ordering.ordering && !ordering.optional).toBe(true);
  expect(recorded(result)).toEqual([]);
  expect(neverRan(result, "__call_0")).toBe(true);
});

test.each([
  [false, ["publish"]],
  [true, []],
])("inline call follows the else route: go=%p", async (go, ran) => {
  const gate = GATE.replace("then: publish", "then: skip_it").replace("else: skip_it", "else: publish");
  const { flow: loaded, result } = await run(flow(gate + PUBLISH + SKIP_IT), go);
  expect(edge(loaded, "gate->__call_0#0").sourceHandle).toBe(ELSE_HANDLE);
  expect(recorded(result)).toEqual(ran);
});

test.each([false, true])("inline call of a then and else host runs on both routes: go=%p", async (go) => {
  const gate = GATE.replace("else: skip_it", "else: publish");
  const { flow: loaded, result } = await run(flow(gate + PUBLISH, "  r: ${publish.output}\n"), go);
  const handles = loaded.compiled.edges
    .filter((each) => each.to === "__call_0" && each.sourceHandle)
    .map((each) => each.sourceHandle);
  expect(handles).toEqual(["publish", ELSE_HANDLE]);
  expect(recorded(result)).toEqual(["publish"]);
});

test.each([
  [false, false, []],
  [false, true, ["publish"]],
])("inline call of a join target runs when any route reaches it: go=%p other=%p", async (go, other, ran) => {
  const gates = `
  c1:
    kind: case
    cases:
      - when: \${input.go}
        then: publish
  c2:
    kind: case
    cases:
      - when: \${input.other}
        then: publish
`;
  const { flow: loaded, result } = await run(flow(gates + PUBLISH, '  r: ${publish.output :- "none"}\n'), go, other);
  const sources = loaded.compiled.edges
    .filter((each) => each.to === "__call_0" && each.sourceHandle)
    .map((each) => each.from);
  expect(sources).toEqual(["c1", "c2"]);
  expect(recorded(result)).toEqual(ran);
});

test("map over call does not run on an untaken route", async () => {
  const nodes =
    `
  gate:
    kind: case
    cases:
      - when: \${input.go}
        then: m
    else: skip_it
  m:
    kind: map
    call: echo
    over: call(items, tag="items")
    input:
      item: \${item}
` + SKIP_IT;
  expect(recorded((await run(flow(nodes), false)).result)).toEqual([]);
  expect(recorded((await run(flow(nodes), true)).result)).toEqual(["items"]);
});

test("loop seed call does not run on an untaken route", async () => {
  const nodes =
    `
  gate:
    kind: case
    cases:
      - when: \${input.go}
        then: lp
    else: skip_it
  lp:
    kind: loop
    times: 1
    input:
      s: call(side, tag="seed")
    nodes:
      k:
        kind: code
        input:
          s: \${input.s}
        output:
          s: str
        code: |
          return {"s": inputs["s"]}
    output: \${k.output}
` + SKIP_IT;
  const output = '  r: ${skip_it.output :- "none"}\n';
  expect(recorded((await run(flow(nodes, output), false)).result)).toEqual([]);
  expect(recorded((await run(flow(nodes, output), true)).result)).toEqual(["seed"]);
});

// The pattern above as a child flow, called per run and per MAP element.
const CHILD_DEF = `
  child:
    input:
      go: !bool
    nodes:
      gate:
        kind: case
        cases:
          - when: \${input.go}
            then: publish
        else: skip_it
      publish:
        kind: code
        input:
          draft: call(side, tag="child")
        output: str
        code: return inputs["draft"]
      skip_it:
        kind: code
        output: str
        code: return "skipped"
    output: \${gate.output}
`;

test("inline call in a called child flow follows the child route", async () => {
  const text =
    HEAD +
    DEFS +
    CHILD_DEF +
    "nodes:\n  c:\n    kind: call\n    call: child\n    input:\n      go: ${input.go}\n" +
    "output:\n  r: ${c.output}\n";
  const { result } = await run(text, false);
  expect(recorded(result)).toEqual([]);
  expect(result.output).toEqual({ r: "skipped" });
});

test("inline call in a mapped child flow runs once per taken element", async () => {
  const text =
    HEAD +
    DEFS +
    CHILD_DEF +
    "nodes:\n  m:\n    kind: map\n    call: child\n" +
    "    over: ${[input.go, input.other]}\n    input:\n      go: ${item}\n" +
    "output:\n  r: ${m.output}\n";
  const { result } = await run(text, true, false);
  expect(recorded(result)).toEqual(["child"]);
  expect(result.output).toEqual({ r: ["child", "skipped"] });
});

test("flow output call has no host and runs", async () => {
  const nodes = GATE + PUBLISH + SKIP_IT;
  const { result } = await run(flow(nodes, '  r: ${gate.output}\n  o: call(side, tag="out")\n'));
  expect(recorded(result)).toContain("out");
  expect((result.output as Record<string, unknown>)["o"]).toBe("out");
});

const TYPED_DEFS = `
typedefs:
  Buy: {action: 'Literal(buy)', qty: int}
  Hold: {action: 'Literal(hold)'}
  Decision: Union(Buy, Hold)
`;
const PLACE_DEF = `
  place:
    input:
      order: !Buy
    nodes:
      p:
        kind: code
        code: ./fixtures/compose_codefns.ts:record
        input:
          tag: \${input.order.action}
        output: str
    output: \${p.output}
`;

test.each([
  ["buy", ["buy"]],
  ["hold", []],
])("inline call in a routed node reads the narrowed member: %s", async (action, ran) => {
  // Under `when: buy` the call gets `b`'s route, so it may pass `d` where a `Buy` is needed.
  const nodes = `
  d:
    kind: code
    input:
      a: \${input.action}
    output: Decision
    code: |
      return inputs["a"] == "buy" ? {"action": inputs["a"], "qty": 2} : {"action": "hold"}
  route:
    kind: case
    on: \${d.output.action}
    cases:
      - when: buy
        then: b
    else: e
  b:
    kind: code
    input:
      r: call(place, order=\${d.output})
    output: str
    code: return inputs["r"]
  e:
    kind: code
    output: str
    code: return "held"
`;
  const text =
    "id: t\nname: t\ninput:\n  action: !str\n" +
    TYPED_DEFS +
    DEFS +
    PLACE_DEF +
    "nodes:\n" +
    nodes +
    "output:\n  r: ${route.output}\n";
  const result = await runFlow(await loadConformanceFlow(text, { env }), { action }, { env });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(recorded(result)).toEqual(ran);
  expect(result.output).toEqual({ r: ran.length > 0 ? "buy" : "held" });
});

// ---------------------------------------------------------------------------------------------
// case targets: the inner calls of a nested branch `call(...)`.
// ---------------------------------------------------------------------------------------------

// Calls mint inner-first: the inner `side` call is `__call_0`, the outer `wrap` is `__call_1`.
const NESTED = `
  gate:
    kind: case
    cases:
      - when: \${input.go}
        then: call(wrap, x=call(side, tag="inner"))
    else: b
  b:
    kind: code
    output: str
    code: return "B"
`;

test("nested branch call does not run its inner call on an untaken route", async () => {
  const { flow: loaded, result } = await run(flow(NESTED), false);
  expect(recorded(result)).toEqual([]);
  expect(neverRan(result, "__call_0")).toBe(true);
  expect(result.output).toEqual({ r: "B" });
  expect(edge(loaded, "gate->__call_0#0").sourceHandle).toBe("__call_1");
  expect(loaded.compiled.edges.filter((each) => each.from === "__start__" && each.to === "__call_0")).toEqual([]);
});

test("nested branch call runs on the taken route", async () => {
  const { result } = await run(flow(NESTED), true);
  expect(recorded(result)).toEqual(["inner"]);
  expect(result.output).toEqual({ r: "wrapped inner" });
});

test("nested branch call is skipped with the case depends on", async () => {
  const nested = NESTED.replace("    kind: case\n", "    kind: case\n    depends_on: [maybe]\n");
  const { result } = await run(flow(MAYBE + nested), true, false);
  expect(recorded(result)).toEqual([]);
  expect(neverRan(result, "__call_0")).toBe(true);
});

test("nested branch call runs after the case runs after", async () => {
  const nested = NESTED.replace("    kind: case\n", "    kind: case\n    runs_after: [prep]\n");
  const { result } = await run(flow(PREP + nested), true);
  expect(recorded(result)).toEqual(["prep", "inner"]);
  expect(startsAfter(result, "prep", "__call_0")).toBe(true);
});

// ---------------------------------------------------------------------------------------------
// adaptive questions: the composer AGENT `<gate>__compose`.
// ---------------------------------------------------------------------------------------------

const COMPOSED = [
  {
    question: "Which database?",
    header: "DB",
    options: [{ label: "Postgres", description: "Relational." }],
    multi_select: false,
  },
];

const ADAPTIVE = `
  router:
    kind: case
    cases:
      - when: \${input.go}
        then: ask
    else: done
  ask:
    kind: human_input
    input:
      ctx: \${input.go}
    adaptive_questions:
      prompt: "Design one question for: \${ctx}"
  done:
    kind: code
    output: str
    code: return "done"
`;

/**
 * Run an adaptive flow; return the flow, the result, the composer's model calls and the human
 * asks. With `fail`, the composer's model has no reply, so a composer call fails.
 */
async function adaptive(nodes: string, go: boolean, other = false, fail = false) {
  const model = new ScriptedModel(fail ? [] : [textReply(JSON.stringify({ items: COMPOSED }))]);
  const asked: AskUserRequest[] = [];
  const { flow: loaded, result } = await run(flow(nodes, "  r: ${router.output}\n"), go, other, {
    models: serving(model, env),
    llmConfig: { provider: "openai", model: "m" },
    tools: {
      ask_user: (request) => {
        asked.push(request);
        return { DB: "Postgres" };
      },
    },
  });
  for (const request of model.requests) expect(request.outputFormat).not.toBeNull();
  return { flow: loaded, result, calls: model.requests.length, asked };
}

test("adaptive composer does not run on an untaken route", async () => {
  const { flow: loaded, result, calls, asked } = await adaptive(ADAPTIVE, false);
  expect([calls, asked]).toEqual([0, []]);
  expect(result.output).toEqual({ r: "done" });
  expect(edge(loaded, "router->ask__compose#0").sourceHandle).toBe("ask");
});

test("adaptive composer failure on an untaken route does not fail the run", async () => {
  const { result, calls } = await adaptive(ADAPTIVE, false, false, true);
  expect(calls).toBe(0);
  expect(result.output).toEqual({ r: "done" });
});

test("adaptive composer runs on the taken route", async () => {
  const { result, calls, asked } = await adaptive(ADAPTIVE, true);
  expect(calls).toBe(1);
  expect((asked[0]!.questions as { header: string }[]).map((question) => question.header)).toEqual(["DB"]);
  expect(result.output).toEqual({ r: { DB: "Postgres" } });
});

test("adaptive composer runs after the gate runs after", async () => {
  const nodes = PREP + ADAPTIVE.replace("    kind: human_input\n", "    kind: human_input\n    runs_after: [prep]\n");
  const { flow: loaded, result, calls } = await adaptive(nodes, true);
  const ordering = edge(loaded, "prep~>ask__compose#0");
  expect(ordering.ordering && ordering.optional).toBe(true);
  expect(calls).toBe(1);
  expect(startsAfter(result, "prep", "ask__compose")).toBe(true);
});

test("adaptive composer is skipped with the gate depends on", async () => {
  const nodes = MAYBE + ADAPTIVE.replace("    kind: human_input\n", "    kind: human_input\n    depends_on: [maybe]\n");
  const { result, calls, asked } = await adaptive(nodes, true, false);
  expect([calls, asked]).toEqual([0, []]);
  expect(neverRan(result, "ask__compose")).toBe(true);
});

test("adaptive gate with an inline call gates both lifted nodes", async () => {
  const nodes =
    PREP +
    ADAPTIVE.replace("      ctx: ${input.go}\n", '      ctx: call(side, tag="ctx")\n').replace(
      "    kind: human_input\n",
      "    kind: human_input\n    runs_after: [prep]\n",
    );
  const { flow: loaded, result, calls, asked } = await adaptive(nodes, false);
  expect([calls, asked, recorded(result)]).toEqual([0, [], ["prep"]]);
  expect(neverRan(result, "__call_0") && neverRan(result, "ask__compose")).toBe(true);
  expect(edge(loaded, "router->__call_0#0").sourceHandle).toBe("ask");
  expect(edge(loaded, "prep~>__call_0#0").optional).toBe(true);
});

test("reading a lifted node in the hosts runs after is a cycle", async () => {
  // `audit` reads the composer, and the gate orders after `audit`: the composer now orders after
  // `audit` too, so the two wait on each other.
  const nodes =
    ADAPTIVE.replace("    kind: human_input\n", "    kind: human_input\n    runs_after: [audit]\n") +
    `
  audit:
    kind: code
    input:
      q: \${ask__compose.output}
    output: int
    code: return inputs["q"].length
`;
  const error = await loadErrorAsync(loadConformanceFlow(flow(nodes, "  r: ${router.output}\n"), { env }));
  expect(error.message).toContain("cycle");
  expect(error.message).toContain("ask__compose");
  expect(error.diagnostic.notes).toContain("ask__compose is generated from node 'ask'");
});

// ---------------------------------------------------------------------------------------------
// data gates: the host's other data inputs.
// ---------------------------------------------------------------------------------------------

const PUBLISH_NOTES = `
  publish:
    kind: code
    input:
      draft: call(side, tag="draft")
      notes: \${maybe.output}
    output: str
    code: return inputs["draft"] + inputs["notes"]
`;
const PUBLISH_OUT = "  r: ${publish.output}\n";

test("inline call is skipped with a host data input", async () => {
  const { flow: loaded, result } = await run(flow(MAYBE + PUBLISH_NOTES, PUBLISH_OUT), false, false);
  expect(neverRan(result, "__call_0")).toBe(true);
  expect(recorded(result)).toEqual([]);
  expect(result.output).toEqual({ r: null });
  const gate = edge(loaded, "maybe->__call_0#0");
  expect(gate.inputGroup).toBe("publish.notes");
  expect(gate.optional).toBe(false);
});

test("inline call waits for a host data input on the taken path", async () => {
  const { result } = await run(flow(MAYBE + PUBLISH_NOTES, PUBLISH_OUT), false, true);
  expect(recorded(result)).toEqual(["draft"]);
  expect(result.output).toEqual({ r: "draftm" });
  expect(startsAfter(result, "maybe", "__call_0")).toBe(true);
});

test("inline call only waits for a defaulted host input", async () => {
  const publish = PUBLISH_NOTES.replace("${maybe.output}", '${maybe.output :- "none"}');
  const { flow: loaded, result } = await run(flow(MAYBE + publish, PUBLISH_OUT), false, false);
  expect(recorded(result)).toEqual(["draft"]);
  expect(result.output).toEqual({ r: "draftnone" });
  expect(edge(loaded, "maybe->__call_0#0").optional).toBe(true);
  expect(index(result, NodeSkipped, "maybe")).toBeLessThan(index(result, NodeStarted, "__call_0"));
});

// Each string nested in a host's value is its own input group, so a lifted node gets one gate
// group per nested read, named after the host and the nested position.
const PUBLISH_NESTED_VALUE = PUBLISH_NOTES.replace(
  "      notes: ${maybe.output}\n",
  '      notes: {text: "${maybe.output}", n: 1}\n',
).replace('inputs["notes"]', 'inputs["notes"]["text"]');

test("inline call is skipped with a host nested data input", async () => {
  const { flow: loaded, result } = await run(flow(MAYBE + PUBLISH_NESTED_VALUE, PUBLISH_OUT), false, false);
  expect(neverRan(result, "__call_0")).toBe(true);
  expect(recorded(result)).toEqual([]);
  expect(result.output).toEqual({ r: null });
  const gate = edge(loaded, "maybe->__call_0#0");
  expect(gate.inputGroup).toBe("publish.notes/text");
  expect(gate.optional).toBe(false);
});

test("inline call only waits for a defaulted host nested input", async () => {
  const publish = PUBLISH_NESTED_VALUE.replace("${maybe.output}", "${maybe.output :- 'none'}");
  const { flow: loaded, result } = await run(flow(MAYBE + publish, PUBLISH_OUT), false, false);
  expect(recorded(result)).toEqual(["draft"]);
  expect(result.output).toEqual({ r: "draftnone" });
  expect(edge(loaded, "maybe->__call_0#0").optional).toBe(true);
});

const PUBLISH_TWO = `
  publish:
    kind: code
    input:
      a: call(meet, tag="a")
      b: call(meet, tag="b")
      notes: \${maybe.output}
    output: str
    code: return inputs["a"] + inputs["b"] + inputs["notes"]
`;

test("sibling calls from one host run in parallel", async () => {
  const { flow: loaded, result } = await run(flow(MAYBE + PUBLISH_TWO, PUBLISH_OUT), false, true, { numWorkers: 2 });
  expect(recorded(result).sort()).toEqual(["a", "b"]);
  expect(loaded.compiled.edges.filter((each) => [each.from, each.to].sort().join() === "__call_0,__call_1")).toEqual(
    [],
  );
  // Both `meet` bodies started before either finished.
  const bodies = ["__call_0/fx", "__call_1/fx"];
  const lastStart = Math.max(...bodies.map((id) => index(result, NodeStarted, id)));
  const firstDone = Math.min(...bodies.map((id) => index(result, NodeSucceeded, id)));
  expect(lastStart).toBeLessThan(firstDone);
});

test("sibling calls from one host are skipped with a host data input", async () => {
  const { result } = await run(flow(MAYBE + PUBLISH_TWO, PUBLISH_OUT), false, false);
  expect(recorded(result)).toEqual([]);
});

// Calls mint inner-first: the inner `side` call is `__call_0`, the outer `pair` is `__call_1`.
const PUBLISH_NESTED = `
  publish:
    kind: code
    input:
      draft: call(pair, x=call(side, tag="inner"), y=\${maybe.output})
    output: str
    code: return inputs["draft"]
`;

test.each([
  [false, [], null],
  [true, ["inner"], "innerm"],
])("nested inline call looks through to its siblings inputs: other=%p", async (other, ran, out) => {
  const { result } = await run(flow(MAYBE + PUBLISH_NESTED, PUBLISH_OUT), false, other);
  expect(recorded(result)).toEqual(ran);
  expect(result.output).toEqual({ r: out });
  if (!other) expect(neverRan(result, "__call_0") && neverRan(result, "__call_1")).toBe(true);
});

const NESTED_PAIR = `
  gate:
    kind: case
    cases:
      - when: \${input.go}
        then: call(pair, x=call(side, tag="inner"), y=\${maybe.output})
    else: b
  b:
    kind: code
    output: str
    code: return "B"
`;

test.each([
  [false, [], null],
  [true, ["inner"], "innerm"],
])("nested branch inner call is skipped with the branch calls data input: other=%p", async (other, ran, out) => {
  const { result } = await run(flow(MAYBE + NESTED_PAIR), true, other);
  expect(recorded(result)).toEqual(ran);
  expect(result.output).toEqual({ r: out });
  if (!other) expect(neverRan(result, "__call_0")).toBe(true);
});

const ADAPTIVE_NOTE =
  MAYBE +
  ADAPTIVE.replace("      ctx: ${input.go}\n", '      ctx: call(side, tag="ctx")\n      note: ${maybe.output}\n');

test("adaptive lifted nodes are skipped with a gate data input", async () => {
  const { result, calls, asked } = await adaptive(ADAPTIVE_NOTE, true, false);
  expect([calls, asked, recorded(result)]).toEqual([0, [], []]);
  expect(neverRan(result, "__call_0") && neverRan(result, "ask__compose")).toBe(true);
});

test("adaptive lifted nodes wait for a gate data input", async () => {
  const { result, calls } = await adaptive(ADAPTIVE_NOTE, true, true);
  expect(recorded(result)).toEqual(["ctx"]);
  expect(calls).toBe(1);
});

test("reading a lifted node from a host input is a cycle", async () => {
  // `spy` reads the call, and `publish` reads `spy`: the call now waits on `spy`.
  const nodes = `
  spy:
    kind: code
    input:
      v: \${__call_0.output}
    output: str
    code: return inputs["v"]
  publish:
    kind: code
    input:
      draft: call(side, tag="d")
      notes: \${spy.output}
    output: str
    code: return inputs["draft"] + inputs["notes"]
`;
  const error = await loadErrorAsync(loadConformanceFlow(flow(nodes, PUBLISH_OUT), { env }));
  expect(error.message).toContain("cycle");
  expect(error.message).toContain("__call_0");
  expect(error.message).toContain("spy");
  expect(error.diagnostic.notes).toContain("__call_0 depends on spy (gate: publish.input.notes)");
});
