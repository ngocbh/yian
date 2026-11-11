/**
 * `yian run` never prints or logs a key from the environment or a saved login a provider echoes,
 * and its key check covers the providers the run will call. The model is a fake `fetch`.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";

import { main } from "../../src/cli/index.ts";
import type { CliIo } from "../../src/cli/utils.ts";
import { ApiKeyCredentials, AuthStore } from "../../src/llm_clients/auth/index.ts";

const KEY = "sk-cli-401-echo-wvbn";
const FRAGMENTS = [KEY, KEY.slice(0, 8), `*${KEY.slice(-4)}`];
const ANSI = /\x1b\[[0-9;?]*[A-Za-z]/g;

const FLOW = `id: keyed
name: keyed
nodes:
  agent: {kind: agent, prompt: hi}
output: \${agent.output}
llm_config: {provider: vllm, model: served-model}
`;

let tmp: string;
let vars: Record<string, string>;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "yian-run-keys-"));
  // No provider key or default, and an empty store under `tmp`.
  vars = {
    HOME: join(tmp, "home"),
    YIAN_AUTH_FILE: join(tmp, "auth", "auth.json"),
    XDG_CONFIG_HOME: join(tmp, "xdg"),
  };
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

/** The command's environment, as a store reads it. */
const env = (name: string) => (Object.hasOwn(vars, name) ? vars[name] : undefined);

/** Echo the key masked, as OpenAI's 401 does. */
async function rejecting(): Promise<Response> {
  const masked = `${KEY.slice(0, 8)}****${KEY.slice(-4)}`;
  const message = `Incorrect API key provided: ${masked}.`;
  const error = { message, type: "invalid_request_error", code: "invalid_api_key" };
  return Response.json({ error }, { status: 401 });
}

/** The URL of each request the answering fake was sent. */
let calls: string[];

/** Answer every chat-completions request with one word, recording its URL. */
async function answering(input: string | URL | Request): Promise<Response> {
  calls.push(input instanceof Request ? input.url : String(input));
  const stream = [
    { choices: [{ index: 0, delta: { role: "assistant", content: "answered" }, finish_reason: null }] },
    { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
  ];
  const body = stream.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("") + "data: [DONE]\n\n";
  return new Response(body, { headers: { "content-type": "text/event-stream" } });
}

type FakeFetch = (input: string | URL | Request) => Promise<Response>;

/** Run `yian run <args>` with `fake` as `fetch`; return its code and its stdout and stderr. */
async function invoke(args: readonly string[], fake: FakeFetch = answering) {
  const chunks: string[] = [];
  const io: CliIo = {
    stdout: { write: async (text) => void chunks.push(text) },
    stderr: { write: async (text) => void chunks.push(text) },
    stdin: new PassThrough(),
    stderrStream: new PassThrough(),
    env,
  };
  const send = globalThis.fetch;
  calls = [];
  globalThis.fetch = fake as unknown as typeof fetch;
  try {
    const code = await main(["run", ...args], io);
    return { code, output: chunks.join("") };
  } finally {
    globalThis.fetch = send;
  }
}

/** Remove styling, box-drawing borders and whitespace, so wrapping can't split a fragment. */
function compact(text: string): string {
  return text.replace(ANSI, "").replace(/[\s─-╿]+/g, "");
}

/** Write `text` as `flow.yaml`; return its path and a run log path beside it. */
function flowFile(text: string): { flow: string; log: string } {
  const flow = join(tmp, "flow.yaml");
  writeFileSync(flow, text);
  return { flow, log: join(tmp, "run.jsonl") };
}

describe("a rejected key is neither printed nor logged", () => {
  for (const source of ["environment", "saved login"]) {
    for (const [id, options] of [["plain", []], ["engine-trace", ["--engine-trace"]]] as const) {
      test(`${source}, ${id}`, async () => {
        if (source === "environment") vars["VLLM_API_KEY"] = KEY;
        else await new AuthStore(null, { env }).save("vllm", new ApiKeyCredentials(KEY), "served-model");
        const { flow, log } = flowFile(FLOW);
        const result = await invoke([flow, "--log", log, ...options], rejecting);
        const output = compact(result.output);
        expect(result.code).not.toBe(0);
        expect(output).toContain("[REDACTED]");
        for (const fragment of FRAGMENTS) {
          expect(output).not.toContain(fragment);
          expect(readFileSync(log, "utf8")).not.toContain(fragment);
        }
      });
    }
  }
});

const INPUT_PROVIDER_FLOW = `id: chosen
name: chosen
input:
  provider: !str openai
nodes:
  agent: {kind: agent, prompt: hi}
output: \${agent.output}
llm_config: {provider: '\${input.provider}', model: chosen-model}
`;

test("the key check asks for the provider an input selects", async () => {
  const { flow, log } = flowFile(INPUT_PROVIDER_FLOW);
  const result = await invoke([flow, "--input", "provider=anthropic", "--log", log]);
  const output = compact(result.output);
  expect(result.code).toBe(1);
  expect(output).toContain("ANTHROPIC_API_KEY");
  expect(output).not.toContain("OPENAI_API_KEY");
  expect(existsSync(log)).toBe(false);
});

test("a bad provider input fails before the log exists", async () => {
  const { flow, log } = flowFile(INPUT_PROVIDER_FLOW);
  const result = await invoke([flow, "--input", "provider=nope", "--log", log]);
  expect(result.code).toBe(1);
  expect(compact(result.output)).toContain(compact("llm_config.provider: not a known provider"));
  expect(existsSync(log)).toBe(false);
});

/** A flow whose `chat` agent's config takes `pick`'s output, below an Anthropic flow. */
function callTimeFlow(chatLlm: string, { pick = "str", value = "'m1'", typedefs = "" } = {}): string {
  return `id: picked
name: picked
${typedefs}nodes:
  pick: {kind: code, output: ${pick}, code: "return ${value}"}
  chat:
    kind: agent
    prompt: hi
    llm_config: ${chatLlm}
output: \${chat.output}
llm_config: {provider: anthropic}
`;
}

test("the key check covers a call time agent with a known provider", async () => {
  const { flow, log } = flowFile(callTimeFlow("{model: '${pick.output}'}"));
  const result = await invoke([flow, "--log", log]);
  expect(result.code, result.output).toBe(1);
  expect(compact(result.output)).toContain("ANTHROPIC_API_KEY");
  expect(existsSync(log)).toBe(false);
  expect(calls).toEqual([]);
});

test("the key check sends a known azure agent to its literal base url", async () => {
  await new AuthStore(null, { env }).save("azure", new ApiKeyCredentials("az"), "dep", {
    endpoint: "https://r.openai.azure.com/",
    makeDefault: false,
  });
  // The saved key belongs to its own resource; this agent calls another one.
  const text = callTimeFlow("{model: '${pick.output}'}").replace(
    "llm_config: {provider: anthropic}",
    "llm_config: {provider: azure, base_url: 'https://other.openai.azure.com/'}",
  );
  const { flow, log } = flowFile(text);
  const result = await invoke([flow, "--log", log]);
  expect(result.code, result.output).toBe(1);
  expect(compact(result.output)).toContain("AZURE_OPENAI_API_KEY");
  expect(existsSync(log)).toBe(false);
  expect(calls).toEqual([]);
});

describe("a provider a value may still set is not checked up front", () => {
  const cases: [string, string][] = [
    [
      "nullable provider",
      callTimeFlow("{provider: '${pick.output}'}", { pick: "Optional(str)", value: "'xai'" }),
    ],
    [
      "whole record",
      callTimeFlow("!Cfg ${pick.output}", {
        pick: "Cfg",
        value: "{'provider': 'xai', 'model': 'm1'}",
        typedefs: "typedefs:\n  Cfg: {provider: Optional(str), model: str}\n",
      }),
    ],
  ];
  for (const [id, text] of cases) {
    test(id, async () => {
      vars["XAI_API_KEY"] = "test-xai-key";
      // An agent that changes provider drops the flow's model and takes the provider's saved one.
      await new AuthStore(null, { env }).save("xai", new ApiKeyCredentials("test-xai-key"), "m1", { makeDefault: false });
      const flowLlm = "llm_config: {provider: anthropic, model: claude-x}";
      const { flow, log } = flowFile(text.replace("llm_config: {provider: anthropic}", flowLlm));
      const result = await invoke([flow, "--log", log]);
      expect(result.code, result.output).toBe(0);
      expect(existsSync(log)).toBe(true);
      expect(calls).toHaveLength(1);
      expect(calls[0]).toStartWith("https://api.x.ai/v1/");
    });
  }
});
