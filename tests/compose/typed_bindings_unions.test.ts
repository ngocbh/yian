/** Typed bindings over unions of records. */

import { afterEach, expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { forget } from "../../src/defaults.ts";
import type { Binding, ValueContract } from "../../src/nodes/contracts.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import { canonicalTypeLabel, typeEquals } from "../../src/typesys/values.ts";
import { assembled, assembleText, type CaseRow, loadError, runCase } from "../helpers/declarations.ts";

afterEach(() => forget());

const TYPEDEFS =
  "typedefs:\n" +
  "  Buy: {action: 'Literal(buy)', qty: int}\n" +
  "  Sell: {action: 'Literal(sell)', qty: int, price: float}\n" +
  "  Hold: {action: 'Literal(hold)', note: str}\n" +
  "  Decision: Union(Buy, Sell, Hold)\n" +
  "  BuyOrSell: Union(Buy, Sell)\n";

const REGISTRY = readTypedefs({
  Buy: { action: "Literal(buy)", qty: "int" },
  Sell: { action: "Literal(sell)", qty: "int", price: "float" },
  Hold: { action: "Literal(hold)", note: "str" },
  Decision: "Union(Buy, Sell, Hold)",
  BuyOrSell: "Union(Buy, Sell)",
});

/** Return the label load errors give the type `text` names. */
const label = (text: string): string => canonicalTypeLabel(typeFor(text, REGISTRY));

/** Return a flow whose `decide` node returns its input as a `Decision`, then `nodes`. */
function flow(nodes: string): string {
  return (
    "id: unions\nname: unions\n" +
    TYPEDEFS +
    "input:\n  decision: !object\n" +
    "nodes:\n" +
    "  decide:\n" +
    "    kind: code\n" +
    "    input:\n      decision: ${input.decision}\n" +
    "    code: return inputs['decision']\n" +
    "    output: !Decision\n" +
    nodes +
    "output: ${decide.output.action}\n"
  );
}

/** Return a CODE node with the one input line `binding` and extra top-level keys. */
function consumer(nodeId: string, binding: string, extra: Record<string, string> = {}): string {
  let text = `  ${nodeId}:\n    kind: code\n`;
  for (const [key, value] of Object.entries(extra)) {
    text += `    ${key}: ${value}\n`;
  }
  return text + `    input:\n      ${binding}\n    code: return 1\n`;
}

/** An `on:` CASE over the decision's tag routing buy, sell, and the rest. */
const ROUTE =
  "  route:\n    kind: case\n    on: ${decide.output.action}\n    cases:\n" +
  "      - when: buy\n        then: on_buy\n" +
  "      - when: sell\n        then: on_sell\n" +
  "    else: on_hold\n";

/** Return the message of the `LoadError` assembling `text` throws. */
const assembleError = (text: string): string => loadError(() => assembleText(text)).message;

test("a member over the union loads", () => {
  const { flow: loaded } = assembleText(flow(consumer("use", "buy: !Buy ${decide.output}")));

  const binding = loaded.compiled.wiring.get("use")!.get("buy") as Binding;
  expect(typeEquals(binding.contract!.type, typeFor("Buy", REGISTRY))).toBe(true);
});

test("a declared union accepts one member", () => {
  assembleText(
    flow(consumer("b", "buy: !Buy ${decide.output}") + consumer("use", "d: !Decision ${b.output}", { output: "!Buy" })),
  );
});

test("a union disjoint from the declared type fails", () => {
  const text = flow(consumer("use", "n: !int ${decide.output}"));

  const error = loadError(() => assembleText(text));

  expect(error.message).toBe(`node 'use' input 'n': declared int, but the value is ${label("Decision")}`);
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(
    text.split("\n").findIndex((line) => line.includes("n: !int")) + 1,
  );
});

test("a member over another branch's member fails in that branch", () => {
  const message = assembleError(
    flow(
      ROUTE +
        consumer("on_buy", "buy: !Buy ${decide.output}") +
        consumer("on_sell", "buy: !Buy ${decide.output}") +
        consumer("on_hold", "note: !str ${decide.output.note}"),
    ),
  );

  expect(message.endsWith(`node 'on_sell' input 'buy': declared Buy, but the value is ${label("Sell")}`)).toBe(true);
});

test("the else branch reads the unlisted members", () => {
  const message = assembleError(
    flow(
      ROUTE +
        consumer("on_buy", "buy: !Buy ${decide.output}") +
        consumer("on_sell", "sell: !Sell ${decide.output}") +
        consumer("on_hold", "trade: !BuyOrSell ${decide.output}"),
    ),
  );

  expect(
    message.endsWith(`node 'on_hold' input 'trade': declared BuyOrSell, but the value is ${label("Hold")}`),
  ).toBe(true);
});

test("each branch may name its own member", () => {
  assembleText(
    flow(
      ROUTE +
        consumer("on_buy", "buy: !Buy ${decide.output}") +
        consumer("on_sell", "sell: !Sell ${decide.output}") +
        consumer("on_hold", "hold: !Hold ${decide.output}"),
    ),
  );
});

test("a flow output narrows to a member and a call parent sees it", async () => {
  const text =
    "id: parent\nname: parent\n" +
    TYPEDEFS +
    "defs:\n" +
    "  pick:\n" +
    "    input:\n      decision: !object\n" +
    "    nodes:\n" +
    "      decide:\n" +
    "        kind: code\n" +
    "        input:\n          decision: ${input.decision}\n" +
    "        code: return inputs['decision']\n" +
    "        output: !Decision\n" +
    "    output: !Buy ${decide.output}\n" +
    "nodes:\n" +
    "  use:\n    kind: call\n    call: pick\n" +
    "    input:\n      decision: {action: buy, qty: 1}\n" +
    "  qty:\n    kind: code\n    input:\n      q: ${use.output.qty}\n" +
    "    code: return inputs['q']\n" +
    "output: ${qty.output}\n";

  const loaded = await loadFlow(text);

  const contract = loaded.compiled.nodes.get("use")!.outputContract as ValueContract;
  expect(typeEquals(contract.type, typeFor("Buy", REGISTRY))).toBe(true);
});

test("an isinstance branch narrows the typed binding too", () => {
  const message = assembleError(
    flow(
      "  route:\n    kind: case\n    cases:\n" +
        "      - when: isinstance(decide.output, Buy)\n        then: on_buy\n" +
        "    else: other\n" +
        consumer("on_buy", "sell: !Sell ${decide.output}") +
        consumer("other", "sell: !Sell ${decide.output}"),
    ),
  );

  expect(message.endsWith(`node 'on_buy' input 'sell': declared Sell, but the value is ${label("Buy")}`)).toBe(true);
});

// ---------- Pinned harvested cases ----------

test.each<CaseRow>(
  [
    "a_union_disjoint_from_the_declared_type_fails",
    "a_declared_union_accepts_one_member",
    "a_member_over_another_branchs_member_fails",
    "the_else_branch_reads_the_unlisted_members",
    "an_isinstance_branch_narrows_the_typed_binding",
  ].map((name) => ({
    id: `load-types/compose_typed_bindings_unions__${name}`,
    call: assembled,
  })),
)("assemble: $id", runCase);
