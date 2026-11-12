/** Provider blocks cascade through a real run into each node's request body. */

import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { runFlow } from "../../src/run.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { anthropicText, recording, responsesText, type Sent } from "../helpers/run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-provider-blocks-e2e-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

const FLOW = `
id: blocks
name: Blocks
llm_config:
  provider: anthropic
  model: claude-sonnet-4-5
  anthropic:
    max_tokens: 2048
    output_config:
      effort: low
  openai:
    reasoning:
      effort: high
nodes:
  a:
    kind: agent
    prompt: Say a.
    output: !str
  b:
    kind: agent
    prompt: Say b.
    output: !str
    llm_config:
      anthropic:
        max_tokens: 512
  c:
    kind: agent
    prompt: Say c.
    output: !str
    llm_config:
      provider: openai
      model: gpt-5.5
output: \${a.output}
`;

/** The node (`a`, `b` or `c`) whose prompt `sent` carries. */
const nodeOf = (sent: Sent) => ["a", "b", "c"].find((name) => JSON.stringify(sent.body).includes(`Say ${name}.`))!;

test("blocks merge per field and reach each request body", async () => {
  const env = hostEnv(dir, { ANTHROPIC_API_KEY: "sk-ant-test", OPENAI_API_KEY: "sk-openai-test" });
  const models = recording(env, (sent) =>
    new URL(sent.url).pathname.endsWith("/responses") ? responsesText(nodeOf(sent)) : anthropicText(nodeOf(sent)),
  );
  const log = join(dir, "run.jsonl");
  const result = await runFlow(await loadFlow(FLOW, { env }), {}, { log, models, env });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe("a");

  // Each node's request goes to its provider's host.
  const urls = Object.fromEntries(models.requests.map((sent) => [nodeOf(sent), sent.url]));
  const bodies = Object.fromEntries(models.requests.map((sent) => [nodeOf(sent), sent.body]));
  expect(urls).toEqual({
    a: "https://api.anthropic.com/v1/messages",
    b: "https://api.anthropic.com/v1/messages",
    c: "https://api.openai.com/v1/responses",
  });
  const { a, b, c } = bodies as Record<string, Record<string, unknown>>;
  expect(a!["max_tokens"]).toBe(2048);
  expect(a!["output_config"]).toEqual({ effort: "low" });
  expect(b!["max_tokens"]).toBe(512);
  expect(b!["output_config"]).toEqual({ effort: "low" });
  expect("reasoning" in a!).toBe(false);
  expect("reasoning" in b!).toBe(false);

  expect(c!["model"]).toBe("gpt-5.5");
  expect(c!["reasoning"]).toEqual({ effort: "high" });
  for (const anthropicField of ["max_tokens", "output_config", "thinking", "top_k"]) {
    expect(anthropicField in c!).toBe(false);
  }

  const records = readFileSync(log, "utf8").split("\n").filter((line) => line !== "").map((line) => JSON.parse(line));
  const started = records.filter((record) => record.kind === "run_started");
  expect(started).toHaveLength(1);
  const effective = started[0].payload["effective_llm_by_static_path"];
  const openaiBlock = { reasoning: { effort: "high" } };
  expect(new Set(Object.keys(effective))).toEqual(new Set(["a", "b", "c"]));
  expect(effective["a"]["provider"]).toBe("anthropic");
  expect(effective["a"]["anthropic"]).toEqual({ max_tokens: 2048, output_config: { effort: "low" } });
  expect(effective["a"]["openai"]).toEqual(openaiBlock);
  expect(effective["b"]["provider"]).toBe("anthropic");
  expect(effective["b"]["anthropic"]).toEqual({ max_tokens: 512, output_config: { effort: "low" } });
  expect(effective["b"]["openai"]).toEqual(openaiBlock);
  expect(effective["c"]["provider"]).toBe("openai");
  expect(effective["c"]["model"]).toBe("gpt-5.5");
  expect(effective["c"]["openai"]).toEqual(openaiBlock);
  expect(effective["c"]["anthropic"]).toEqual({ max_tokens: 2048, output_config: { effort: "low" } });
  for (const config of Object.values(effective)) expect("api_key" in (config as object)).toBe(false);
});
