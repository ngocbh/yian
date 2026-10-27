/**
 * `AgentDescriptor.build` honors `outputContractOverride`. The `adaptive_questions` desugar
 * synthesizes an AGENT whose structured output is a type built in code, with no type text; the
 * override lets that contract win over the declared `output:`.
 */

import { expect, test } from "bun:test";

import { AgentDescriptor } from "../../src/compose/nodes/agent.ts";
import { BuildContext } from "../../src/compose/nodes/base.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { questionListType } from "../../src/nodes/tool/questions.ts";

test("contract override wins without invoking the output reader", () => {
  const contract = new ValueContract(questionListType());
  const desc = new AgentDescriptor({
    id: "g__compose",
    prompt: "x",
    inputs: {},
    outputs: "str",
    outputContractOverride: contract,
  });
  const unexpectedReader = (): ValueContract | null => {
    throw new Error("an output override must bypass the Compose reader");
  };

  const node = desc.build(new BuildContext({ registry: new Map(), outputContractReader: unexpectedReader })).node;

  expect(node.outputContract).toBe(contract);
});
