/**
 * Union narrowing at load: a CASE branch reads the members its tag or isinstance check selects.
 */

import { afterAll, afterEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { forget } from "../../src/defaults.ts";
import { runFlow } from "../../src/run.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import {
  assembled,
  assembleText,
  type CaseRow,
  loadConformanceFlow,
  loadError,
  runCase,
} from "../helpers/declarations.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-union-narrowing-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

afterEach(() => forget());

const TYPEDEFS =
  "typedefs:\n" +
  "  Limit: {type: 'Literal(limit)', price: float}\n" +
  "  Market: {type: 'Literal(market)'}\n" +
  "  Leg: Union(Limit, Market)\n" +
  "  Buy: {action: 'Literal(buy)', qty: int, limit: float, leg: Leg}\n" +
  "  Hold: {action: 'Literal(hold)', note: str}\n" +
  "  Sell: {action: 'Literal(sell)', qty: int, price: float, note: str}\n" +
  "  Decision: Union(Buy, Hold, Sell)\n" +
  "  HoldOrSell: Union(Hold, Sell)\n" +
  "  Wrap: {decision: Decision}\n" +
  "  BuyOrSell: Union(Buy, Sell)\n" +
  "  Text: {text: str}\n" +
  "  Num: {num: int}\n" +
  "  Reply: Union(Text, Num)\n" +
  "  Mode: Literal(a, b)\n" +
  "  Cancel: {action: 'Literal(cancel)', reason: str}\n" +
  "  Rest: Union(Hold, Sell, Cancel)\n" +
  "  Narrow: {action: 'Literal(buy)', qty: int, limit: float, leg: Limit}\n" +
  "  SellOrNarrow: Union(Sell, Narrow)\n";

// Fields: `qty` (buy, sell), `limit` and `leg` (buy), `price` (sell), `note` (hold, sell).
const QTY_ERROR =
  "reference ${d.output.qty} reads field 'qty', which only union members with action " +
  "buy, sell have; read it inside a case on ${d.output.action}";
const LIMIT_ERROR =
  "reference ${d.output.limit} reads field 'limit', which only union members with " +
  "action buy have; read it inside a case on ${d.output.action}";

const DEFS =
  "defs:\n" +
  "  place:\n" +
  "    input:\n      order: !Buy\n" +
  "    nodes:\n" +
  "      p:\n        kind: code\n        input:\n          o: ${input.order}\n" +
  "        code: return 1\n" +
  "    output: ${p.output}\n" +
  "  place_rest:\n" +
  "    input:\n      order: !HoldOrSell\n" +
  "    nodes:\n" +
  "      p:\n        kind: code\n        input:\n          o: ${input.order}\n" +
  "        code: return 1\n" +
  "    output: ${p.output}\n" +
  "  place_sell_or_narrow:\n" +
  "    input:\n      order: !SellOrNarrow\n" +
  "    nodes:\n" +
  "      p:\n        kind: code\n        input:\n          o: ${input.order}\n" +
  "        code: return 1\n" +
  "    output: ${p.output}\n" +
  "  place_maybe_rest:\n" +
  "    input:\n      order: !Optional(HoldOrSell)\n" +
  "    nodes:\n" +
  "      p:\n        kind: code\n        input:\n          o: ${input.order}\n" +
  "        code: return 1\n" +
  "    output: ${p.output}\n" +
  "  outer:\n" +
  "    input:\n      x: !int\n" +
  "    nodes:\n" +
  "      p:\n        kind: code\n        input:\n          x: ${input.x}\n" +
  "        code: return 1\n" +
  "    output: ${p.output}\n" +
  "  inner:\n" +
  "    input:\n      y: !int\n" +
  "    nodes:\n" +
  "      p:\n        kind: code\n        input:\n          y: ${input.y}\n" +
  "        code: return 1\n" +
  "    output: 1\n";

/** A CODE node typed `output` that reads the flow input `s`. */
function producer(nodeId: string, output: string): string {
  return (
    `  ${nodeId}:\n    kind: code\n    input:\n      s: \${input.s}\n` + `    output: ${output}\n    code: return None\n`
  );
}

/** An untyped CODE node with `inputs` and extra top-level YAML keys. */
function code(nodeId: string, inputs: Record<string, string> = {}, extra: Record<string, string> = {}): string {
  let text = `  ${nodeId}:\n    kind: code\n`;
  for (const [key, value] of Object.entries(extra)) text += `    ${key}: ${value}\n`;
  if (Object.keys(inputs).length > 0) {
    text += "    input:\n";
    for (const [name, source] of Object.entries(inputs)) text += `      ${name}: ${source}\n`;
  }
  return text + "    code: return 1\n";
}

/** A CASE node; `on: null` is the searched form. */
function caseNode(
  nodeId: string,
  on: string | null,
  cases: readonly (readonly [string, string])[],
  else_: string | null = null,
): string {
  let text = `  ${nodeId}:\n    kind: case\n`;
  if (on !== null) text += `    on: ${on}\n`;
  text += "    cases:\n";
  for (const [when, then] of cases) text += `      - when: ${when}\n        then: ${then}\n`;
  if (else_ !== null) text += `    else: ${else_}\n`;
  return text;
}

/** A flow with producer `d` (a `Decision`) reading the input `s`, then `nodes`, the defs, and an output. */
function flow(...nodes: string[]): string {
  return flowWith({}, ...nodes);
}

/**
 * A flow as `flow` builds it, with producer `d` typed `dType` (default `Decision`), an `input:`
 * section holding `inputs` (default `s: !str`) and the flow `output` (default `${d.output.action}`).
 */
function flowWith(options: { inputs?: string; dType?: string; output?: string }, ...nodes: string[]): string {
  const { inputs = "  s: !str\n", dType = "Decision" } = options;
  // An `Optional` output is read through `?.`.
  const output = options.output ?? (dType.startsWith("Optional(") ? "${d.output?.action}" : "${d.output.action}");
  return (
    "id: p\nname: p\n" +
    TYPEDEFS +
    `input:\n${inputs}` +
    "nodes:\n" +
    producer("d", dType) +
    nodes.join("") +
    `output: ${output}\n` +
    DEFS
  );
}

// The canonical texts of `Buy`, `Hold`, `Sell` and `HoldOrSell`, as the call type check names them.
const BUY =
  '{"action":Literal(buy),"leg":Union({"price":float,"type":Literal(limit)},{"type":Literal(market)}),' +
  '"limit":float,"qty":int}';
const HOLD = '{"action":Literal(hold),"note":str}';
const SELL = '{"action":Literal(sell),"note":str,"price":float,"qty":int}';
const HOLD_OR_SELL = `Union(${HOLD},${SELL})`;
const PLACE_EXPECTS = `call(place) argument 'order' — child expects '${BUY}'`;
const PLACE_REST_EXPECTS = `call(place_rest) argument 'order' — child expects '${HOLD_OR_SELL}'`;

/** Assemble `text`. */
const load = (text: string) => assembleText(text);

const ON = "${d.output.action}";
/** `ON` over an `Optional` output. */
const OPTIONAL_ON = "${d.output?.action}";

/** A searched CASE `route` over `cases`, else to `e`. */
function searched(...cases: (readonly [string, string])[]): string {
  return caseNode("route", null, cases, "e");
}

/** The message of the `LoadError` assembling `text` throws. */
const assembleError = (text: string): string => loadError(() => load(text)).message;

// --- narrowed reads --- //

test("a buy branch reads a member field", () => {
  load(
    flow(
      caseNode("route", ON, [
        ["buy", "b"],
        ["sell", "s"],
        ["hold", "h"],
      ]),
      code("b", { q: "${d.output.qty}", l: "${d.output.limit}" }),
      code("s", { p: "${d.output.price}" }),
      code("h", { n: "${d.output.note}" }),
    ),
  );
});

test("a hold branch reading qty reads an unknown field on hold", () => {
  const message = assembleError(
    flow(
      caseNode(
        "route",
        ON,
        [
          ["buy", "b"],
          ["hold", "h"],
        ],
        "b",
      ),
      code("b", { q: "${d.output.qty}" }),
      code("h", { q: "${d.output.qty}" }),
    ),
  );
  expect(message).toContain("node 'h' input 'q' from: reference ${d.output.qty} reads unknown field 'qty'");
  expect(message).not.toContain("node 'b'");
});

// --- chains --- //

test("a plain reference chain is narrowed", () => {
  load(
    flow(
      caseNode("route", ON, [["buy", "b"]], "e"),
      code("b", { o: "${d.output}" }),
      code("e"),
      code("c", { y: "${b.output}", l: "${d.output.limit}" }),
    ),
  );
});

describe("a join of buy and sell branches narrows to both", () => {
  test.each(["${b.output | s.output}", "${b.output :- s.output}"])("%s", (join) => {
    const branches = [
      caseNode(
        "route",
        ON,
        [
          ["buy", "b"],
          ["sell", "s"],
        ],
        "e",
      ),
      code("b"),
      code("s"),
      code("e"),
    ];
    load(flow(...branches, code("c", { y: join, q: "${d.output.qty}" })));
    const message = assembleError(flow(...branches, code("c", { y: join, l: "${d.output.limit}" })));
    expect(message).toContain(
      "node 'c' input 'l' from: reference ${d.output.limit} reads field 'limit', which only " +
        "union members with action buy have",
    );
  });
});

test("depends_on propagates the branch", () => {
  load(
    flow(
      caseNode("route", ON, [["buy", "b"]], "e"),
      code("b"),
      code("e"),
      code("c", { l: "${d.output.limit}" }, { depends_on: "[b]" }),
    ),
  );
});

test("runs_after does not propagate the branch", () => {
  const message = assembleError(
    flow(
      caseNode("route", ON, [["buy", "b"]], "e"),
      code("b"),
      code("e"),
      code("c", { l: "${d.output.limit}" }, { runs_after: "[b]" }),
    ),
  );
  expect(message).toContain(`node 'c' input 'l' from: ${LIMIT_ERROR}`);
});

test("an optional group does not propagate the branch", () => {
  const message = assembleError(
    flow(
      caseNode("route", ON, [["buy", "b"]], "e"),
      code("b"),
      code("e"),
      code("c", { y: "${b.output:-0}", l: "${d.output.limit}" }),
    ),
  );
  expect(message).toContain(`node 'c' input 'l' from: ${LIMIT_ERROR}`);
});

test("a nested inline call in a branch target is narrowed", () => {
  // `inner` is lifted out of the routed `outer`, so it carries the same route.
  load(
    flow(caseNode("route", ON, [["buy", "call(outer, x=call(inner, y=${d.output.qty}))"]], "e"), code("e")),
  );
});

// --- routing --- //

test("two whens to one node give a two-member union", () => {
  const branches = [
    caseNode(
      "route",
      ON,
      [
        ["buy", "b"],
        ["sell", "b"],
      ],
      "e",
    ),
    code("e"),
  ];
  load(flow(...branches, code("b", { q: "${d.output.qty}" })));
  const message = assembleError(flow(...branches, code("b", { l: "${d.output.limit}" })));
  expect(message).toContain("reads field 'limit', which only union members with action buy have");
});

test("else gives the remaining member", () => {
  const branches = [
    caseNode(
      "route",
      ON,
      [
        ["buy", "b"],
        ["sell", "s"],
      ],
      "e",
    ),
    code("b"),
    code("s"),
  ];
  load(flow(...branches, code("e", { n: "${d.output.note}" })));
  const message = assembleError(flow(...branches, code("e", { q: "${d.output.price}" })));
  expect(message).toContain("reads unknown field 'price' on record");
});

test("else gives the remaining union", () => {
  const branches = [caseNode("route", ON, [["buy", "b"]], "e"), code("b")];
  load(flow(...branches, code("e", { n: "${d.output.note}" })));
  const message = assembleError(flow(...branches, code("e", { p: "${d.output.price}" })));
  expect(message).toContain(
    "reference ${d.output.price} reads field 'price', which only union members with action sell have",
  );
});

// --- control edges --- //

test("a node routed by two narrowing cases on different bases is not narrowed", () => {
  const message = assembleError(
    flow(
      producer("d2", "Decision"),
      caseNode("route", ON, [["buy", "b"]], "e"),
      caseNode("route2", "${d2.output.action}", [["buy", "b"]], "e"),
      code("b", { l: "${d.output.limit}" }),
      code("e"),
    ),
  );
  expect(message).toContain(`node 'b' input 'l' from: ${LIMIT_ERROR}`);
});

test("a node routed by a narrowing case and a searched case is not narrowed", () => {
  const message = assembleError(
    flow(
      caseNode("route", ON, [["buy", "b"]], "e"),
      caseNode("gate", null, [["input.s == 'x'", "b"]], "e"),
      code("b", { l: "${d.output.limit}" }),
      code("e"),
    ),
  );
  expect(message).toContain(`node 'b' input 'l' from: ${LIMIT_ERROR}`);
});

// --- case of case --- //

test("an inner searched case in a buy branch keeps the narrowing", () => {
  load(
    flow(
      caseNode("route", ON, [["buy", "inner"]], "e"),
      caseNode("inner", null, [["${d.output.limit > 5}", "t"]], "t2"),
      code("t", { l: "${d.output.limit}" }),
      code("t2", { q: "${d.output.qty}" }),
      code("e"),
    ),
  );
});

// --- not narrowed --- //

test("on over a coalesce does not narrow", () => {
  const message = assembleError(
    flow(
      producer("d2", "Decision"),
      caseNode("route", "${d.output.action | d2.output.action}", [["buy", "b"]], "e"),
      code("b", { l: "${d.output.limit}" }),
      code("e"),
    ),
  );
  expect(message).toContain(`node 'b' input 'l' from: ${LIMIT_ERROR}`);
});

test("on over embedded text does not narrow", () => {
  const message = assembleError(
    flow(
      caseNode("route", "'x${d.output.action}'", [["xbuy", "b"]], "e"),
      code("b", { l: "${d.output.limit}" }),
      code("e"),
    ),
  );
  expect(message).toContain(`node 'b' input 'l' from: ${LIMIT_ERROR}`);
});

test("nodes outside the branch and the flow output are not narrowed", () => {
  const message = assembleError(
    flowWith(
      { output: "${d.output.limit}" },
      caseNode("route", ON, [["buy", "b"]], "e"),
      code("b", { l: "${d.output.limit}" }),
      code("e"),
      code("o", { l: "${d.output.limit}" }),
    ),
  );
  expect(message).toContain(`node 'o' input 'l' from: ${LIMIT_ERROR}`);
  expect(message).toContain(`flow output from: ${LIMIT_ERROR}`);
  expect(message).not.toContain("node 'b'");
});

// --- robustness --- //

test("a cycle through a branch reports the cycle", () => {
  const message = assembleError(
    flow(
      caseNode("route", ON, [["buy", "b"]], "e"),
      code("b", { y: "${c.output}", l: "${d.output.limit}" }),
      code("c", { y: "${b.output}", l: "${d.output.limit}" }),
      code("e"),
    ),
  );
  expect(message).toContain('flow has a cycle involving ["b","c"]');
});

test("a member-only field on a cycle reports the cycle", () => {
  const message = assembleError(
    flow(code("b", { y: "${c.output}", l: "${d.output.limit}" }), code("c", { y: "${b.output}" })),
  );
  expect(message).toContain('flow has a cycle involving ["b","c"]');
});

test("a field no member has on a cycle is still an unknown field", () => {
  const message = assembleError(
    flow(code("b", { y: "${c.output}", z: "${d.output.zzz}" }), code("c", { y: "${b.output}" })),
  );
  expect(message).toContain("reads unknown field 'zzz' on record");
});

test("an unknown depends_on still reports the unknown node", () => {
  const message = assembleError(
    flow(
      caseNode("route", ON, [["buy", "b"]], "e"),
      code("b", { l: "${d.output.limit}" }, { depends_on: "[nope]" }),
      code("e"),
    ),
  );
  expect(message).toContain("node 'b': `depends_on` references unknown node 'nope'");
});

test("an inline body branch target cannot infer a member field", () => {
  const target =
    "  b:\n" +
    "    kind: call\n" +
    "    input:\n      q: ${d.output.qty}\n" +
    "    nodes:\n" +
    "      inner:\n        kind: code\n        input:\n          q: ${input.q}\n" +
    "        code: return 1\n" +
    "    output: ${inner.output}\n";
  const message = assembleError(flow(caseNode("route", ON, [["buy", "b"]], "e"), target, code("e")));
  expect(message).toContain(`cannot infer type for inline binding 'b.input.q': ${QTY_ERROR}`);
});

// --- impossible checks in a searched `when:` --- //

describe("a when check that can never be true is a located error", () => {
  test.each<[string, string]>([
    [
      "isinstance(d.output, Buy) and d.output.action == 'hold'",
      '`d.output.action == "hold"` can never be true here: d.output is Buy, by the checks before it',
    ],
    [
      "isinstance(d.output, Buy) and d.output.action != 'buy' and d.output.limit > 0",
      '`d.output.action != "buy"` can never be true here: d.output is Buy, by the checks ' +
        'before it, so `d.output.action == "buy"` is always true',
    ],
    [
      "d.output.action == 'biy'",
      '`d.output.action == "biy"` compares d.output.action with "biy", which is not one ' +
        'of its tags (buy, hold, sell); did you mean "buy"?',
    ],
    [
      "isinstance(d.output, Text)",
      "`isinstance(d.output, Text)` can never be true: d.output is declared Decision, " +
        "and no Decision value fits Text",
    ],
    [
      "not not isinstance(d.output, Text)",
      "`isinstance(d.output, Text)` can never be true: d.output is declared Decision, " +
        "and no Decision value fits Text",
    ],
  ])("%s", (when, message) => {
    const text = flow(searched(["input.s == 'x'", "b"], [`"${when}"`, "b"]), code("b"), code("e"));
    const error = loadError(() => assembleText(text));
    expect(error.message).toContain(`case node 'route': \`cases[1].when\`: ${message}`);
    // A read after the impossible check is not checked on top of it.
    expect(error.message).not.toContain("'limit'");
    const line = text.split("\n").findIndex((row) => row.includes(when)) + 1;
    expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(line);
  });
});

test("a when check the routing rules out is an error", () => {
  const message = assembleError(
    flow(
      caseNode("route", ON, [["buy", "inner"]], "e"),
      caseNode("inner", null, [[`"d.output.action == 'hold'"`, "t"]], "m"),
      code("t"),
      code("m"),
      code("e"),
    ),
  );
  expect(message).toContain(
    "case node 'inner': `cases[0].when`: `d.output.action == \"hold\"` can never be true " +
      "here: d.output is Buy, by the case branch that routes this node",
  );
});

// --- calls --- //

test("a call target under buy passes a buy order", () => {
  load(flow(caseNode("route", ON, [["buy", "call(place, order=${d.output})"]], "e"), code("e")));
});

test("a call target under sell fails the call type check", () => {
  const message = assembleError(
    flow(caseNode("route", ON, [["sell", "call(place, order=${d.output})"]], "e"), code("e")),
  );
  expect(message).toBe(`node 'route' cases[0].then ${PLACE_EXPECTS}, source is '${SELL}'`);
});

test("an unrouted call of the whole union still fails", () => {
  const message = assembleError(flow(code("c", { r: "call(place, order=${d.output})" })));
  expect(message).toBe(`node 'c' input 'r' ${PLACE_EXPECTS}, source is 'Union(${BUY},${HOLD},${SELL})'`);
});

test("an inline call in a routed node input is narrowed", () => {
  // The call is lifted out of `b`, so it carries `b`'s route.
  load(flow(caseNode("route", ON, [["buy", "b"]], "e"), code("b", { r: "call(place, order=${d.output})" }), code("e")));
});

test("else passes the remaining union to a call", () => {
  load(flow(caseNode("route", ON, [["buy", "b"]], "call(place_rest, order=${d.output})"), code("b")));
});

test("a nullable union else target sees an optional union", () => {
  const nullable = (callee: string) =>
    flowWith(
      { dType: "Optional(Decision)" },
      caseNode("route", OPTIONAL_ON, [["buy", "b"]], `call(${callee}, order=\${d.output})`),
      code("b"),
    );

  load(nullable("place_maybe_rest"));
  const message = assembleError(nullable("place_rest"));
  expect(message).toBe(`node 'route' else ${PLACE_REST_EXPECTS}, source is 'Optional(${HOLD_OR_SELL})'`);
});

test("a nullable union narrows to a non-nullable member under when", () => {
  load(
    flowWith(
      { dType: "Optional(Decision)" },
      caseNode("route", OPTIONAL_ON, [["buy", "call(place, order=${d.output})"]], "e"),
      code("e"),
    ),
  );
});

test("a nullable union defensive else sees None", () => {
  const message = assembleError(
    flowWith(
      { dType: "Optional(Decision)" },
      caseNode(
        "route",
        ON,
        [
          ["buy", "b"],
          ["sell", "b"],
          ["hold", "b"],
        ],
        "call(place_rest, order=${d.output})",
      ),
      code("b"),
    ),
  );
  expect(message).toBe(`node 'route' else ${PLACE_REST_EXPECTS}, source is 'None'`);
});

// --- empty narrowing sets --- //

test("a defensive else after every tag loads and is not narrowed", () => {
  const branches = [
    caseNode(
      "route",
      ON,
      [
        ["buy", "b"],
        ["sell", "b"],
        ["hold", "b"],
      ],
      "e",
    ),
    code("b"),
  ];
  load(flow(...branches, code("e", { a: "${d.output.action}" })));
  const message = assembleError(flow(...branches, code("e", { l: "${d.output.limit}" })));
  expect(message).toContain(`node 'e' input 'l' from: ${LIMIT_ERROR}`);
});

test("a nested case's when the outer branch rules out is a load error", () => {
  const branches = [
    caseNode("route", ON, [["buy", "inner"]], "e"),
    caseNode("inner", ON, [
      ["buy", "t1"],
      ["sell", "t"],
    ]),
    code("t1"),
    code("e"),
  ];
  const message = assembleError(flow(...branches, code("t", { a: "${d.output.action}" })));
  expect(message).toBe(
    "case node 'inner': `when: sell` never matches: the branches that reach this case rule it out, " +
      "so `on: ${d.output.action}` is `Literal(buy)` here; remove it",
  );
});

test("required groups from buy and sell branches meet empty", () => {
  const message = assembleError(
    flow(
      caseNode(
        "route",
        ON,
        [
          ["buy", "b"],
          ["sell", "s"],
        ],
        "e",
      ),
      code("b"),
      code("s"),
      code("e"),
      code("c", { x: "${b.output}", y: "${s.output}", q: "${d.output.qty}" }),
    ),
  );
  expect(message).toContain(`node 'c' input 'q' from: ${QTY_ERROR}`);
});

// --- on: cases over a narrowed tag --- //

test("a nested on case over a union field narrows outer then inner", () => {
  assembleText(
    flow(
      caseNode("route", ON, [["buy", "inner"]], "e"),
      caseNode("inner", "${d.output.leg.type}", [
        ["limit", "t"],
        ["market", "m"],
      ]),
      code("t", { p: "${d.output.leg.price}" }),
      code("m"),
      code("e"),
    ),
  );
});

test("a nested on case outside the outer branch is a rule one error", () => {
  const message = assembleError(
    flow(
      caseNode("inner", "${d.output.leg.type}", [
        ["limit", "t"],
        ["market", "m"],
      ]),
      code("t"),
      code("m"),
    ),
  );
  expect(message).toContain("node 'inner'");
  expect(message).toContain("reads field 'leg', which only union members");
});

test("an inner on case is exhaustive over its narrowed tag", () => {
  assembleText(
    flow(
      caseNode("route", ON, [["buy", "inner"]], "e"),
      caseNode("inner", ON, [["buy", "t"]]),
      code("t", { l: "${d.output.limit}" }),
      code("e"),
    ),
  );
});

test("an inner on case over a flow input is exhaustive over its narrowed tag", () => {
  assembleText(
    flowWith(
      { inputs: "  s: !str\n  d: !Decision\n" },
      caseNode("route", "${input.d.action}", [["buy", "inner"]], "e"),
      caseNode("inner", "${input.d.action}", [["buy", "t"]]),
      code("t", { l: "${input.d.limit}" }),
      code("e"),
    ),
  );
});

// --- other bases --- //

test("a flow input base narrows", () => {
  const inputFlow = (...extra: string[]) =>
    flowWith(
      { inputs: "  s: !str\n  d: !Decision\n" },
      caseNode("route", "${input.d.action}", [["buy", "b"]], "e"),
      code("b", { l: "${input.d.limit}" }),
      code("e", { n: "${input.d.note}" }),
      ...extra,
    );

  load(inputFlow());
  const message = assembleError(inputFlow(code("o", { l: "${input.d.limit}" })));
  expect(message).toContain("node 'o' input 'l' from: reference ${input.d.limit} reads field 'limit'");
  expect(message).not.toContain("node 'b'");
});

test("a path base narrows", () => {
  load(
    flow(
      producer("r", "Wrap"),
      caseNode("route", "${r.output.decision.action}", [["buy", "b"]], "e"),
      code("b", { l: "${r.output.decision.limit}" }),
      code("e", { n: "${r.output.decision.note}" }),
    ),
  );
});

test("a path base through an optional record keeps nullability in else", () => {
  const wrapped = (callee: string) =>
    flow(
      producer("r", "Optional(Wrap)"),
      caseNode("route", "${r.output?.decision.action}", [["buy", "b"]], `call(${callee}, order=\${r.output?.decision})`),
      code("b", { l: "${r.output.decision.limit}" }),
    );

  load(wrapped("place_maybe_rest"));
  const message = assembleError(wrapped("place_rest"));
  expect(message).toBe(`node 'route' else ${PLACE_REST_EXPECTS}, source is 'Optional(${HOLD_OR_SELL})'`);
});

test("a when that is not a tag is a located error", () => {
  const text = flow(caseNode("route", ON, [["buyy", "b"]], "e"), code("b"), code("e"));

  const error = loadError(() => assembleText(text));

  expect(error.message).toStartWith("case node 'route': `when: buyy` is not a tag of `on: ${d.output.action}` (");
  expect(error.message).toEndWith("); use one of: buy, hold, sell");
  const line = text.split("\n").findIndex((row) => row.includes("when: buyy")) + 1;
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(line);
});

test("a non-exhaustive on over a Literal input errors", () => {
  const message = assembleError(
    flowWith({ inputs: "  s: !str\n  mode: !Mode\n" }, caseNode("route", "${input.mode}", [["a", "b"]]), code("b")),
  );
  expect(message).toContain('enum tag(s) ["b"] not covered');
});

test("a non-exhaustive on over a union node output still errors", () => {
  const message = assembleError(flow(caseNode("route", ON, [["buy", "b"]]), code("b")));
  expect(message).toContain('enum tag(s) ["hold","sell"] not covered');
});

// --- isinstance in a searched CASE --- //

test("an on inside a partly shared union branch covers the shared tags", () => {
  assembleText(
    flow(
      searched(["isinstance(d.output, Rest)", "inner"]),
      caseNode("inner", ON, [
        ["hold", "h"],
        ["sell", "s"],
      ]),
      code("h"),
      code("s"),
      code("e"),
    ),
  );
});

test("a partly shared union branch runs", async () => {
  const text =
    "id: p\nname: p\n" +
    TYPEDEFS +
    "input:\n  s: !str\n" +
    "nodes:\n" +
    "  d:\n" +
    "    kind: code\n" +
    "    input:\n      s: ${input.s}\n" +
    "    output: Decision\n" +
    "    code: |\n" +
    "      return {'action': 'sell', 'qty': 2, 'price': 1.5, 'note': 'n'}\n" +
    searched(["isinstance(d.output, Rest)", "t"]) +
    "  t:\n" +
    "    kind: code\n" +
    "    input:\n      n: ${d.output.note}\n" +
    "    output: str\n" +
    "    code: return inputs['n']\n" +
    code("e") +
    "output: ${route.output}\n";

  const result = await runFlow(await loadConformanceFlow(text), { s: "x" }, { env });

  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("n");
});

test("a partly shared union branch passes the shared members to a call", () => {
  const branch = searched(["isinstance(d.output, Rest)", "call(place_rest, order=${d.output})"]);
  load(flow(branch, code("e")));
  load(flowWith({ dType: "Optional(Decision)" }, branch, code("e")));
});

test("a member that only overlaps the checked union keeps the checked type", () => {
  const call = "call(place_sell_or_narrow, order=${d.output})";
  load(flow(searched(["isinstance(d.output, SellOrNarrow)", call]), code("e")));
  const message = assembleError(
    flow(searched(["isinstance(d.output, SellOrNarrow)", "b"]), code("b", { p: "${d.output.price}" }), code("e")),
  );
  expect(message).toContain("which only union members with action sell have");
});

// --- isinstance --- //

test("a target under isinstance reads a member field", () => {
  load(
    flow(
      searched(["isinstance(d.output, Buy)", "b"]),
      code("b", { q: "${d.output.qty}", l: "${d.output.limit}" }),
      code("e"),
    ),
  );
});

describe("an isinstance and conjunct narrows", () => {
  test.each([
    "isinstance(d.output, Buy) and input.s == 'x'",
    "input.s == 'x' and isinstance(d.output, Buy)",
    "${isinstance(d.output, Buy) and input.s == 'x'}",
  ])("%s", (when) => {
    load(flow(searched([`"${when}"`, "b"]), code("b", { l: "${d.output.limit}" }), code("e")));
  });
});

describe("isinstance under or narrows only what the when proves", () => {
  test.each([
    "isinstance(d.output, Buy) or input.s == 'x'",
    "not (isinstance(d.output, Buy) and input.s == 'x')",
    "not isinstance(d.output, Hold)",
  ])("%s", (when) => {
    // The check may not hold when `input.s` decides, and `Sell` passes `not Hold` too.
    const message = assembleError(
      flow(searched([`"${when}"`, "b"]), code("b", { l: "${d.output.limit}" }), code("e")),
    );
    expect(message).toContain(`node 'b' input 'l' from: ${LIMIT_ERROR}`);
  });
});

describe("a branch sees a tag comparison", () => {
  test.each(["d.output.action == 'buy'", "'buy' == d.output.action"])("%s", (when) => {
    load(flow(searched([`"${when}"`, "b"]), code("b", { l: "${d.output.limit}" }), code("e")));
  });
});

describe("a branch reads the fields the members its when proves share", () => {
  test.each([
    "not isinstance(d.output, Hold)",
    "isinstance(d.output, Buy) or isinstance(d.output, Sell)",
    "d.output.action != 'hold'",
  ])("%s", (when) => {
    load(flow(searched([`"${when}"`, "b"]), code("b", { q: "${d.output.qty}" }), code("e")));
    const message = assembleError(
      flow(searched([`"${when}"`, "b"]), code("b"), code("e", { l: "${d.output.limit}" })),
    );
    expect(message).toContain(`node 'e' input 'l' from: ${LIMIT_ERROR}`);
  });
});

describe("a read in a when is narrowed through and, or and not", () => {
  test.each([
    "not isinstance(d.output, Buy) or d.output.limit > 5",
    "d.output.action == 'buy' and d.output.limit > 5",
    "d.output.action != 'buy' or d.output.limit > 5",
    "not (isinstance(d.output, Buy) and d.output.limit > 5)",
    "(isinstance(d.output, Buy) and d.output.limit > 5) or d.output.action == 'hold'",
    "isinstance(d.output, Rest) or d.output.limit > 5",
  ])("%s", (when) => {
    load(flow(searched([`"${when}"`, "b"]), code("b"), code("e")));
  });
});

describe("a read in a when of a field the narrowed members lack is an error", () => {
  test.each<[string, string]>([
    ["isinstance(d.output, Buy) or d.output.limit > 5", "Hold | Sell"],
    // `Buy` is no `Rest`, so only the shared `Hold` and `Sell` reach the read.
    ["not isinstance(d.output, Rest) or d.output.limit > 5", "Hold | Sell"],
  ])("%s", (when, narrowed) => {
    const message = assembleError(flow(searched([`"${when}"`, "b"]), code("b"), code("e")));
    expect(message).toContain(
      "node 'route' cases[0].when from: reference ${d.output.limit} reads field 'limit', " +
        `which ${narrowed} doesn't have; d.output is ${narrowed} here, by the checks before this read`,
    );
  });
});

test("a misspelled read in a when names the narrowed type", () => {
  const message = assembleError(
    flow(searched(['"isinstance(d.output, Buy) and d.output.limt > 5"', "b"]), code("b"), code("e")),
  );
  expect(message).toContain(
    "node 'route' cases[0].when from: reference ${d.output.limt} reads field 'limt', " +
      "which Buy doesn't have; d.output is Buy here, by the checks before this read; " +
      "did you mean 'limit'?",
  );
});

test("a branch of a negated partly shared check keeps the overlapping member", () => {
  const when = '"not isinstance(d.output, SellOrNarrow)"';
  let message = assembleError(flow(searched([when, "b"]), code("b", { l: "${d.output.limit}" }), code("e")));
  expect(message).toContain(
    "node 'b' input 'l' from: reference ${d.output.limit} reads field 'limit', which only " +
      "union members with action buy have",
  );
  message = assembleError(flow(searched([when, "b"]), code("b", { p: "${d.output.price}" }), code("e")));
  expect(message).toContain("node 'b' input 'p' from: reference ${d.output.price} reads unknown field 'price'");
});

test("a read after a check the routing decides is checked against the routing", () => {
  const routed = (when: string) =>
    flow(
      caseNode("route", ON, [["buy", "inner"]], "e"),
      caseNode("inner", null, [[`"${when}"`, "t"]], "m"),
      code("t"),
      code("m"),
      code("e"),
    );

  load(routed("isinstance(d.output, Buy) or d.output.limit > 5"));
  const message = assembleError(routed("isinstance(d.output, Decision) or d.output.nosuch > 5"));
  expect(message).toContain("reads unknown field 'nosuch'");
});

test("a when read narrowed only by the routing credits the routing", () => {
  // The check on `input.o` says nothing about `d.output`: the routing makes it Buy.
  const message = assembleError(
    flowWith(
      { inputs: "  s: !str\n  o: !Reply\n" },
      caseNode("route", ON, [["buy", "inner"]], "e"),
      caseNode("inner", null, [['"isinstance(input.o, Text) and d.output.nosuch > 0"', "t"]], "m"),
      code("t"),
      code("m"),
      code("e"),
    ),
  );
  expect(message).toContain(
    "reference ${d.output.nosuch} reads field 'nosuch', which Buy doesn't have; d.output " +
      "is Buy here, by the case branch that routes this node",
  );
});

test("a when read past a check the checks before it decide is checked under them", () => {
  let when = "isinstance(d.output, Buy) and (isinstance(d.output, Buy) or d.output.limit > 0)";
  load(flow(searched([`"${when}"`, "b"]), code("b"), code("e")));
  when =
    "isinstance(d.output, Buy) or isinstance(d.output, Hold) or isinstance(d.output, Sell) " +
    "or d.output.limit > 5";
  const message = assembleError(flow(searched([`"${when}"`, "b"]), code("b"), code("e")));
  expect(message).toContain(
    "reference ${d.output.limit} reads field 'limit', which Sell doesn't have; d.output is " +
      "Sell here, by the checks before this read",
  );
});

test("a searched when with a tag check and a member read runs", async () => {
  const text =
    "id: p\nname: p\n" +
    TYPEDEFS +
    "input:\n  s: !str\n" +
    "nodes:\n" +
    "  d:\n" +
    "    kind: code\n" +
    "    input:\n      s: ${input.s}\n" +
    "    output: Decision\n" +
    "    code: |\n" +
    "      if (inputs['s'] === 'buy')\n" +
    "          return {'action': 'buy', 'qty': 1, 'limit': 6.0, 'leg': {'type': 'market'}}\n" +
    "      return {'action': 'sell', 'qty': 2, 'price': 1.5, 'note': 'n'}\n" +
    searched(["\"d.output.action == 'buy' and d.output.limit > 5\"", "t"]) +
    "  t:\n    kind: code\n    output: str\n    code: return 'branch'\n" +
    "  e:\n    kind: code\n    output: str\n    code: return 'else'\n" +
    "output: ${route.output}\n";
  const loaded = await loadConformanceFlow(text);

  let result = await runFlow(loaded, { s: "buy" }, { env });
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("branch");
  result = await runFlow(loaded, { s: "sell" }, { env });
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("else");
});

test("overlapping checks narrow each branch by its own check", () => {
  const overlapping = (bsRead: string) =>
    flow(
      searched(["isinstance(d.output, Buy)", "b"], ["isinstance(d.output, BuyOrSell)", "bs"]),
      code("b", { l: "${d.output.limit}" }),
      code("bs", { x: bsRead }),
      code("e"),
    );

  load(overlapping("${d.output.qty}"));
  const message = assembleError(overlapping("${d.output.limit}"));
  expect(message).toContain(
    "node 'bs' input 'x' from: reference ${d.output.limit} reads field 'limit', which " +
      "only union members with action buy have",
  );
});

test("a conjunct after isinstance reads the narrowed base", () => {
  load(flow(searched(['"isinstance(d.output, Buy) and d.output.limit > 5"', "b"]), code("b"), code("e")));
});

test("a conjunct before isinstance is a rule one error", () => {
  const message = assembleError(
    flow(searched(['"d.output.limit > 5 and isinstance(d.output, Buy)"', "b"]), code("b"), code("e")),
  );
  expect(message).toContain(`node 'route' cases[0].when from: ${LIMIT_ERROR}`);
});

test("a read guarded in one case and unguarded in another is an error", () => {
  const message = assembleError(
    flow(
      searched(['"isinstance(d.output, Buy) and d.output.limit > 5"', "b"], ['"d.output.limit > 1"', "b"]),
      code("b"),
      code("e"),
    ),
  );
  // The guarded read in cases[0] is fine; the unguarded one in cases[1] is the error.
  expect(message).toContain(`node 'route' cases[1].when from: ${LIMIT_ERROR}`);
});

test("an unguarded read error points at its own when", () => {
  const text = flow(
    searched(
      ['"isinstance(d.output, Buy) and d.output.limit > 5"', "b"],
      ["input.s == 'x'", "b"],
      ['"d.output.limit > 1"', "b"],
    ),
    code("b"),
    code("e"),
  );
  const error = loadError(() => load(text));
  expect(error.message).toContain(`node 'route' cases[2].when from: ${LIMIT_ERROR}`);
  const line = text.split("\n").findIndex((row) => row.includes("d.output.limit > 1")) + 1;
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(line);
});

test("a read unguarded first points at the first when", () => {
  const message = assembleError(
    flow(
      searched(['"d.output.limit > 1"', "b"], ['"isinstance(d.output, Buy) and d.output.limit > 5"', "b"]),
      code("b"),
      code("e"),
    ),
  );
  expect(message).toContain(`node 'route' cases[0].when from: ${LIMIT_ERROR}`);
});

describe("isinstance of a type the base already fits does not narrow", () => {
  test.each(["object", "Decision"])("%s", (typeName) => {
    const message = assembleError(
      flow(searched([`isinstance(d.output, ${typeName})`, "b"]), code("b", { q: "${d.output.qty}" }), code("e")),
    );
    expect(message).toContain(`node 'b' input 'q' from: ${QTY_ERROR}`);
  });
});

test("a when: read after a check that can never hold reports only the check", () => {
  const text = flow(
    searched([`"isinstance(d.output, Buy) and d.output.action == 'hold' and d.output.qty > 0"`, "b"]),
    code("b"),
    code("e"),
  );
  const message = assembleError(text);
  expect(message).toContain("can never be true");
  expect(message).not.toContain("reads field");
});

test("a read guarded by different members is checked against each one", () => {
  const guarded = (field: string) =>
    flow(
      searched(
        [`"isinstance(d.output, Buy) and d.output.${field} > 5"`, "b"],
        [`"isinstance(d.output, Sell) and d.output.${field} > 1"`, "s"],
      ),
      code("b"),
      code("s"),
      code("e"),
    );

  load(guarded("qty"));
  // Each read is checked at its own `when:`: the Sell-guarded one lacks `limit`.
  const message = assembleError(guarded("limit"));
  expect(message).toContain(
    "node 'route' cases[1].when from: reference ${d.output.limit} reads field 'limit', which Sell doesn't have; " +
      "d.output is Sell here, by the checks before this read",
  );
});

test("isinstance narrows an untagged union to the member", () => {
  const untagged = (...extra: string[]) =>
    flowWith(
      { dType: "Reply", output: "${d.output}" },
      searched(["isinstance(d.output, Text)", "b"]),
      code("b", { t: "${d.output.text}" }),
      code("e"),
      ...extra,
    );

  load(untagged());
  const message = assembleError(untagged(code("o", { t: "${d.output.text}" })));
  expect(message).toContain(
    "node 'o' input 't' from: reference ${d.output.text} reads field 'text', which only " +
      "1 of 2 union members have",
  );
  expect(message).not.toContain("node 'b'");
});

test("isinstance narrows an opaque object to the checked type", () => {
  const opaque = (when: string) =>
    flow(producer("x", "object"), searched([when, "call(place, order=${x.output})"]), code("e"));

  load(opaque("isinstance(x.output, Buy)"));
  const message = assembleError(opaque("isinstance(x.output, Sell)"));
  expect(message).toBe(`node 'route' cases[0].then ${PLACE_EXPECTS}, source is '${SELL}'`);
});

test("isinstance cases on unrelated types into one target narrow to their union", () => {
  const routed = (...whens: string[]) =>
    flow(
      producer("x", "object"),
      searched(...whens.map((when) => [when, "b"] as const)),
      "  b:\n    kind: call\n    call: place\n    input:\n      order: ${x.output}\n",
      code("e"),
    );

  load(routed("isinstance(x.output, Buy)"));
  const message = assembleError(routed("isinstance(x.output, Buy)", "isinstance(x.output, Text)"));
  expect(message).toBe(`call node 'b': binding 'order' — child expects '${BUY}', source is 'Union(${BUY},{"text":str})'`);
});

test("an opaque object outside isinstance does not fit a record", () => {
  const message = assembleError(
    flow(producer("x", "object"), searched(["input.s == 'x'", "call(place, order=${x.output})"]), code("e")),
  );
  expect(message).toBe(`node 'route' cases[0].then ${PLACE_EXPECTS}, source is 'object'`);
});

test("a target routed by two isinstance cases gets their union", () => {
  const routed = (read: string) =>
    flow(
      searched(["isinstance(d.output, Buy)", "b"], ["isinstance(d.output, Sell)", "b"]),
      code("b", { x: read }),
      code("e"),
    );

  load(routed("${d.output.qty}"));
  const message = assembleError(routed("${d.output.limit}"));
  expect(message).toContain("reads field 'limit', which only union members with action buy have");
});

test("a target routed by isinstance and a plain case is not narrowed", () => {
  const message = assembleError(
    flow(
      searched(["isinstance(d.output, Buy)", "b"], ["input.s == 'x'", "b"]),
      code("b", { l: "${d.output.limit}" }),
      code("e"),
    ),
  );
  expect(message).toContain(`node 'b' input 'l' from: ${LIMIT_ERROR}`);
});

test("a searched else is not narrowed", () => {
  const message = assembleError(
    flow(searched(["isinstance(d.output, Buy)", "b"]), code("b"), code("e", { n: "${d.output.note}" })),
  );
  expect(message).toContain(
    "node 'e' input 'n' from: reference ${d.output.note} reads field 'note', which only " +
      "union members with action hold, sell have",
  );
});

test("an isinstance branch narrows a nullable base to the member", () => {
  load(
    flowWith(
      { dType: "Optional(Decision)" },
      searched(["isinstance(d.output, Buy)", "call(place, order=${d.output})"]),
      code("e"),
    ),
  );
});

test("isinstance on a flow input narrows", () => {
  load(
    flowWith(
      { inputs: "  s: !str\n  d: !Decision\n" },
      searched(["isinstance(input.d, Buy)", "b"]),
      code("b", { l: "${input.d.limit}" }),
      code("e"),
    ),
  );
});

test("isinstance inside an on branch composes", () => {
  load(
    flow(
      caseNode("route", ON, [["buy", "inner"]], "e"),
      caseNode("inner", null, [["isinstance(d.output.leg, Limit)", "t"]], "m"),
      code("t", { p: "${d.output.leg.price}" }),
      code("m"),
      code("e"),
    ),
  );
});

test("an unknown isinstance type keeps its own error", () => {
  const message = assembleError(flow(searched(["isinstance(d.output, Nope)", "b"]), code("b"), code("e")));
  expect(message).toContain("unknown type 'Nope'");
});

// --- isinstance against a partly shared union --- //

test("isinstance of a partly shared union reads the shared fields", () => {
  load(flow(searched(["isinstance(d.output, Rest)", "b"]), code("b", { n: "${d.output.note}" }), code("e")));
  const message = assembleError(
    flow(searched(["isinstance(d.output, Rest)", "b"]), code("b", { n: "${d.output.reason}" }), code("e")),
  );
  expect(message).toContain("node 'b' input 'n' from: reference ${d.output.reason} reads unknown field 'reason'");
  expect(message).not.toContain("cancel");
});

test("a partly shared union check lines up with an on else", () => {
  load(
    flow(
      caseNode("route", ON, [["buy", "b"]], "inner"),
      caseNode("inner", null, [["isinstance(d.output, Rest)", "t"]], "m"),
      code("b"),
      code("t", { n: "${d.output.note}" }),
      code("m"),
    ),
  );
});

test("a conjunct after a partly shared union check reads the shared fields", () => {
  load(flow(searched([`"isinstance(d.output, Rest) and d.output.note == 'x'"`, "b"]), code("b"), code("e")));
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>(
  [
    "a_when_comparing_the_tag_with_a_non_tag_suggests_one",
    "an_isinstance_that_can_never_be_true",
    "a_defensive_else_after_every_tag_loads",
  ].map((name) => ({ id: `load-types/compose_union_narrowing__${name}`, call: assembled })),
)("assemble: $id", runCase);

test.each<CaseRow>(
  [
    "a_misspelled_read_in_a_when_names_the_narrowed_type",
    "a_read_in_a_when_of_a_field_the_narrowed_members_lack",
    "a_when_read_narrowed_only_by_the_routing",
  ].map((name) => ({ id: `load-types/compose_union_narrowing__${name}`, call: assembled, full: true })),
)("assemble (full): $id", runCase);

test("assemble: load-types/compose_union_narrowing__a_nullable_union_defensive_else_sees_none", () => {
  // Its `source is 'None'` is the CASE narrowing's: the defensive `else:` sees only null.
  runCase({ id: "load-types/compose_union_narrowing__a_nullable_union_defensive_else_sees_none", call: assembled });
});
