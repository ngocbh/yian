/** A run takes no input for an inbox; its START binds the handle instead. */
import { expect, test } from "bun:test";

import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { hostInputs, runStartInputs, startWiring } from "../../src/run.ts";
import { inboxType, Type, ValueKind } from "../../src/typesys/values.ts";

const STR = Type.scalar(ValueKind.STRING);
const inboxParam = (name: string) =>
  new Parameter(name, { contract: new ValueContract(inboxType(STR)), required: true });

test("a run input naming an inbox is refused", () => {
  expect(() => hostInputs([inboxParam("m")], { m: "hi" })).toThrow(
    "inbox `m` receives events through send(); it takes no run input",
  );
});

test("the START wiring keeps inbox entries; the run-start inputs leave them out", () => {
  const params = [inboxParam("m"), new Parameter("n", { required: true })];
  expect(startWiring(params, { n: 1n })).toEqual({ n: "${input.n}", m: { inbox: "m" } });
  expect(runStartInputs(params, { n: 1n })).toEqual({ n: 1n });
});

test("an omitted input stays out of the START wiring", () => {
  expect(startWiring([new Parameter("n", { hasDefault: true, default: 1n })], {})).toEqual({});
});
