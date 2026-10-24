/**
 * A conversation one route produced, sent through another: none of its reasoning, signatures,
 * encrypted content or ids reach the other route's body; through the same route they still do.
 * The turns come from each adapter's own parser over its recorded reasoning cassette, so the
 * records are the ones adapters really write. A model switch never replays them.
 */

import { describe, expect, test } from "bun:test";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { encodeBody } from "../../src/llm/body.ts";
import { collect } from "../../src/llm/events.ts";
import type { WireProtocol } from "../../src/llm/executor.ts";
import { iterNdjson, iterSse } from "../../src/llm/framing.ts";
import { Message, ReasoningPart, Request, TextPart, ToolResultPart, ToolSpec } from "../../src/llm/messages.ts";
import { AnthropicMessages } from "../../src/llm/protocols/anthropic_messages.ts";
import { Gemini } from "../../src/llm/protocols/gemini.ts";
import { OllamaChat } from "../../src/llm/protocols/ollama_chat.ts";
import { OpenAIChat } from "../../src/llm/protocols/openai_chat.ts";
import { OpenAIResponses } from "../../src/llm/protocols/openai_responses.ts";
import { loadCassette } from "../../src/llm/testing.ts";
import type { JsonValue } from "../../src/typesys/json.ts";

const DIR = fileURLToPath(new URL("cassettes/", import.meta.url));
const SONNET = "claude-sonnet-4-5";
const FLASH = "gemini-2.5-flash";

const DOUBLE = new ToolSpec({
  name: "double",
  description: "Double an integer.",
  parameters: { properties: { value: { type: "integer" } }, required: ["value"], type: "object" },
});

/** A wire protocol with the provider its replay records name. */
type Protocol = WireProtocol & { readonly provider: string };

/** A route: how to build its protocol, and the cassettes whose first replies are its turns. */
interface Route {
  name: string;
  make: () => Protocol;
  cassettes: string[];
}

// Each route's model, provider and account are the ones its adapter's cassette test uses for
// these cassettes.
const ROUTES: Route[] = [
  {
    name: "anthropic",
    make: () => new AnthropicMessages({ model: SONNET, provider: "anthropic" }),
    cassettes: ["anthropic_messages/thinking_tools.json"],
  },
  {
    name: "responses",
    make: () => new OpenAIResponses({ model: "gpt-5.5", provider: "openai" }),
    cassettes: ["openai_responses/tools.json"],
  },
  {
    name: "deepseek",
    make: () => new OpenAIChat({ model: "deepseek-reasoner", provider: "deepseek" }),
    cassettes: ["openai_chat/deepseek_reasoner.json"],
  },
  {
    name: "gemini",
    make: () => new Gemini({ model: FLASH, provider: "google" }),
    cassettes: ["gemini/thought.json", "gemini/tools.json"],
  },
  {
    name: "ollama",
    make: () => new OllamaChat({ model: "qwen3", provider: "ollama" }),
    cassettes: ["ollama_chat/think_true.json"],
  },
];

/** Consumers that share a producer's protocol but not its provider or model. */
const SIBLINGS: [producer: string, consumer: Route][] = [
  ["ollama", { name: "ollama/llama3", make: () => new OllamaChat({ model: "llama3", provider: "ollama" }), cassettes: [] }],
  [
    "anthropic",
    {
      name: "anthropic/other-provider",
      make: () => new AnthropicMessages({ model: SONNET, provider: "bedrock-anthropic" }),
      cassettes: [],
    },
  ],
  [
    "responses",
    {
      name: "responses/gpt-5-mini",
      make: () => new OpenAIResponses({ model: "gpt-5-mini", provider: "openai" }),
      cassettes: [],
    },
  ],
  [
    "deepseek",
    {
      name: "deepseek/deepseek-chat",
      make: () => new OpenAIChat({ model: "deepseek-chat", provider: "deepseek" }),
      cassettes: [],
    },
  ],
  // Gemini's text and call signatures replay only to the same model.
  [
    "gemini",
    {
      name: "gemini/gemini-3-pro-preview",
      make: () => new Gemini({ model: "gemini-3-pro-preview", provider: "google" }),
      cassettes: [],
    },
  ],
];

const SECRET_KEYS = new Set(["signature", "redacted", "encrypted_content", "id", "thought_signature", "response_id"]);

/** Return the assistant message `protocol` parses from the first reply recorded in `cassette`. */
async function parsed(protocol: Protocol, cassette: string): Promise<Message> {
  const chunks = loadCassette(join(DIR, cassette)).interactions[0]!.response.chunks!;
  const bytes = chunks.map((chunk) => new TextEncoder().encode(chunk));
  const events = protocol.framing === "sse" ? protocol.parse(iterSse(bytes)) : protocol.parse(iterNdjson(bytes));
  return (await collect(events)).message;
}

/** Return the turns a route's cassettes parse to, in order. */
async function producerTurns(route: Route): Promise<Message[]> {
  const turns: Message[] = [];
  for (const cassette of route.cassettes) turns.push(await parsed(route.make(), cassette));
  return turns;
}

/** Append every string under a key in `SECRET_KEYS`, at any depth of `value`, to `out`. */
function secretStrings(value: JsonValue | null, out: string[]): void {
  if (Array.isArray(value)) {
    for (const item of value) secretStrings(item, out);
  } else if (value !== null && typeof value === "object") {
    for (const [key, item] of Object.entries(value)) {
      if (SECRET_KEYS.has(key) && typeof item === "string" && item !== "") out.push(item);
      else secretStrings(item, out);
    }
  }
}

/**
 * Return the strings a turn must not leak to another route: each non-empty reasoning text, and
 * every string under a key in SECRET_KEYS anywhere in a part's or the message's payload. Call
 * ids are left out, because a call and its result go to every route.
 *
 * Each is returned as it appears in an encoded body. A secret shorter than 16 characters (an id
 * such as `rs_1`) is JSON-quoted (`JSON.stringify`), so it can't match inside another string.
 * Longer ones (signatures, encrypted content, reasoning texts) are JSON-escaped without quotes,
 * so a regression that embeds one in a larger string, such as reasoning joined into text, still
 * fails.
 */
function secretsOf(turn: Message): string[] {
  const found: string[] = [];
  for (const part of turn.parts) {
    if (part instanceof ReasoningPart && part.text !== "") found.push(part.text);
    secretStrings(part.providerPayload, found);
  }
  secretStrings(turn.providerPayload, found);
  return found.map((secret) => (secret.length < 16 ? JSON.stringify(secret) : escaped(secret)));
}

/** Return `text` JSON-escaped, as it appears inside a string of an encoded body. */
function escaped(text: string): string {
  return JSON.stringify(text).slice(1, -1);
}

/** Return the conversation around `turns`: a question, each turn with its tool results, a follow-up. */
function history(turns: Message[]): Request {
  const messages = [new Message({ role: "user", parts: [new TextPart("Q")] })];
  for (const turn of turns) {
    messages.push(turn);
    const results = turn.toolCalls().map((call) => new ToolResultPart({ callId: call.id, name: call.name, content: "6" }));
    if (results.length > 0) messages.push(new Message({ role: "tool", parts: results }));
  }
  messages.push(new Message({ role: "user", parts: [new TextPart("next")] }));
  return new Request({ messages, tools: [DOUBLE] });
}

/** Return the reasoning parts' records' `field` values, as strings, across `turns`. */
function recordFields(turns: Message[], field: string): string[] {
  return turns
    .flatMap((turn) => turn.parts)
    .filter((part) => part instanceof ReasoningPart)
    .map((part) => part.providerPayload?.[field])
    .filter((value): value is string => typeof value === "string");
}

/** What each route really sends back to itself, read from the parsed turns. */
const OWN: Record<string, (turns: Message[]) => string[]> = {
  anthropic: (turns) => recordFields(turns, "signature"),
  responses: (turns) =>
    turns
      .flatMap((turn) => turn.parts)
      .filter((part) => part instanceof ReasoningPart)
      .map((part) => (part.providerPayload?.["item"] as { encrypted_content?: unknown } | undefined)?.encrypted_content)
      .filter((value): value is string => typeof value === "string"),
  deepseek: (turns) => turns.flatMap((turn) => turn.parts.filter((part) => part instanceof ReasoningPart).map((part) => part.text)),
  gemini: (turns) =>
    turns
      .flatMap((turn) => turn.toolCalls())
      .map((call) => call.providerPayload?.["thought_signature"])
      .filter((value): value is string => typeof value === "string"),
  ollama: (turns) => turns.flatMap((turn) => turn.parts.filter((part) => part instanceof ReasoningPart).map((part) => part.text)),
};

for (const producer of ROUTES) {
  const turns = await producerTurns(producer);
  const secrets = turns.flatMap(secretsOf);

  describe(producer.name, () => {
    test("its cassettes yield secrets to check", () => {
      expect(secrets.length).toBeGreaterThan(0);
    });

    test("every reasoning part it parsed is tagged with its route", () => {
      const protocol = producer.make();
      const reasoning = turns.flatMap((turn) => turn.parts).filter((part) => part instanceof ReasoningPart);
      expect(reasoning.length).toBeGreaterThan(0);
      for (const part of reasoning) {
        expect(part.providerPayload).toMatchObject({ kind: protocol.name, provider: protocol.provider, model: protocol.model });
      }
    });

    const consumers = [
      ...ROUTES.filter((route) => route.name !== producer.name),
      ...SIBLINGS.filter(([name]) => name === producer.name).map(([, consumer]) => consumer),
    ];
    for (const consumer of consumers) {
      test(`${producer.name} -> ${consumer.name} sends none of its replay data`, () => {
        const body = encodeBody(consumer.make().buildBody(history(turns)));
        for (const secret of secrets) expect(body).not.toContain(secret);
        for (const call of turns.flatMap((turn) => turn.toolCalls())) expect(body).toContain(call.id);
      });
    }

    test(`${producer.name} -> itself keeps its replay data`, () => {
      const own = OWN[producer.name]!(turns).map(escaped);
      expect(own.length).toBeGreaterThan(0);
      const body = encodeBody(producer.make().buildBody(history(turns)));
      for (const secret of own) expect(body).toContain(secret);
    });
  });
}
