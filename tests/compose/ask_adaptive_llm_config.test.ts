/** A TOOL ask's `adaptive_questions.llm_config` is read as a layer for its composer. */
import { expect, test } from "bun:test";

import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Binding, ValueTemplate } from "../../src/nodes/contracts.ts";
import { assembleText, loadError } from "../helpers/declarations.ts";

const ASK = `id: f
name: f
input:
  proj: !str
  model: !str
nodes:
  q:
    kind: tool
    tool_id: ask
    input:
      proj: \${input.proj}
    adaptive_questions:
      prompt: "Q \${proj}"
      llm_config:
LLM
output: \${q.output}
`;

/** Return the ask flow with `llmLines` under `llm_config:` (indented 8). */
const ask = (llmLines: string) => ASK.replace("LLM\n", llmLines);

/** Return the composer `q__compose` of flow `text`, assembled. */
const composerOf = (text: string) => assembleText(text, "f.yaml").flow.compiled.nodes.get("q__compose") as AgentNode;

test("a composer value loads as a template", () => {
  const agent = composerOf(ask("        model: ${input.model}\n"));
  expect(agent.ownLlmConfig).toBeInstanceOf(ValueTemplate);
  expect((agent.ownLlmConfig as ValueTemplate).value).toEqual({ model: "${input.model}" });
});

test("a composer tag is read and consumed", () => {
  const agent = composerOf(ask("        model: !str ${input.model}\n"));
  expect(((agent.ownLlmConfig as ValueTemplate).value as Record<string, unknown>)["model"]).toBeInstanceOf(Binding);
});

test("a bad composer value is reported at the ask's adaptive llm_config", () => {
  const text = ask("        provider: openai\n        api_key: ${input.model}\n");
  const error = loadError(() => assembleText(text, "f.yaml"));
  expect(error.message).toStartWith("node 'q' adaptive_questions.llm_config.api_key is a host credential");
  expect(error.message).not.toContain("q__compose");
  const line = text.split("\n").findIndex((row) => row.includes("api_key:")) + 1;
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(line);
});
