/**
 * Fixtures for tests of the live channel: a model that streams one delta per gate the test opens,
 * the flows they run, and helpers to wait for a run and read its calls' texts. They import no
 * `bun:test`, so a child process can load them too.
 */

import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { type ChatModel, type Event, EventStream, Finish, TextDelta } from "../../src/llm/events.ts";
import { Message, type Request } from "../../src/llm/messages.ts";
import { iterRunRecords, NodeStepRecord } from "../../src/runtime/log.ts";

export const CONFIG = { provider: "openai", model: "gpt-test" };

/**
 * A model that answers a request whose last message's text is `key` with the deltas `replies[key]`,
 * streaming each one, then its finish, only after `release(key)` opens its gate.
 */
export class GatedModel implements ChatModel {
  readonly provider = CONFIG.provider;
  readonly model = CONFIG.model;
  readonly #replies: Record<string, string[]>;
  readonly #permits = new Map<string, number>();
  readonly #waiters = new Map<string, (() => void)[]>();

  constructor(replies: Record<string, string[]>) {
    this.#replies = replies;
  }

  /** Let the call for `key` stream its next delta. */
  release(key: string): void {
    const waiter = this.#waiters.get(key)?.shift();
    if (waiter !== undefined) waiter();
    else this.#permits.set(key, (this.#permits.get(key) ?? 0) + 1);
  }

  stream(request: Request, options: { signal?: AbortSignal } = {}): EventStream {
    const key = request.messages.at(-1)!.text();
    const deltas = this.#replies[key]!;
    const gate = (): Promise<void> => {
      const permits = this.#permits.get(key) ?? 0;
      if (permits > 0) {
        this.#permits.set(key, permits - 1);
        return Promise.resolve();
      }
      return new Promise((resolve, reject) => {
        let waiters = this.#waiters.get(key);
        if (waiters === undefined) this.#waiters.set(key, (waiters = []));
        waiters.push(resolve);
        options.signal?.addEventListener("abort", () => reject(options.signal!.reason), { once: true });
      });
    };
    async function* events(): AsyncGenerator<Event> {
      for (const text of deltas) {
        await gate();
        yield new TextDelta(0, text);
      }
      await gate();
      yield new Finish({ reason: "stop" });
    }
    return new EventStream(events());
  }
}

/** One AGENT `write`, prompted with the run's input `topic`. */
export const ONE_AGENT = `
id: live
name: live
llm_config: {provider: openai, model: gpt-test}
input:
  topic: !str
nodes:
  write:
    kind: agent
    prompt: \${topic}
    input:
      topic: \${input.topic}
    output: str
output: \${write.output}
`;

/** A MAP of an AGENT over the run's input `topics`, two at a time. */
export const MAP_AGENTS = `
id: live-map
name: live_map
llm_config: {provider: openai, model: gpt-test}
input:
  topics: !list(str)
defs:
  item_flow:
    input:
      topic: !str
    nodes:
      write:
        kind: agent
        prompt: \${topic}
        input:
          topic: \${input.topic}
        output: str
    output: \${write.output}
nodes:
  each:
    kind: map
    call: item_flow
    over: \${input.topics}
    input:
      topic: \${item}
output: \${each.output}
`;

/** A fresh run log path in its own folder. */
export function runLog(): string {
  return join(mkdtempSync(join(tmpdir(), "yian-live-channel-")), "log.jsonl");
}

/** Wait, a macrotask at a time, until `ready()` holds; fail after a bounded number of turns. */
export async function until(ready: () => boolean): Promise<void> {
  for (let turn = 0; turn < 2000; turn++) {
    if (ready()) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error("timed out waiting for the run");
}

/** The step id of `node`'s step `n`. */
export const key = (node: string, n: number) => `${node} ${n}`;

/** The text of each `model_call` step in the log at `path`, by step id. */
export function callTexts(path: string): Map<string, string> {
  const texts = new Map<string, string>();
  for (const { envelope } of iterRunRecords(path)) {
    const { record } = envelope;
    if (record instanceof NodeStepRecord && record.step["kind"] === "model_call") {
      texts.set(key(record.nodeId, record.n), Message.fromJSON(record.step["output"]).text());
    }
  }
  return texts;
}
