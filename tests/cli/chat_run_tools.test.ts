// The chat assistant's run tools, and its child runs' driver with the chat's hooks (`ChatRuns`):
// flows in a temporary root started as child runs.

import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  realpathSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { PassThrough } from "node:stream";

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { type ChildFlowOwner, type ChildFlows, type ChildStatus, childEvent } from "../../src/child_flows.ts";
import { AUTHOR_TOOLS } from "../../src/cli/authoring.ts";
import { ChatRuns, type ChatRunsOptions, chatRunTools } from "../../src/cli/chat_run_tools.ts";
import { type ChatToolHost, chatFileTools, ToolRefused } from "../../src/cli/chat_tools.ts";
import { loadConfinedText } from "../../src/cli/confined_load.ts";
import { main } from "../../src/cli/index.ts";
import { defaultIdentity, defaultLocation, projectKey } from "../../src/cli/memory_bind.ts";
import { type CliIo, MissingKeyError } from "../../src/cli/utils.ts";
import type { ApproveRequest } from "../../src/cli/terminal.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import type { Env } from "../../src/defaults.ts";
import type { Embed } from "../../src/llm/embeddings.ts";
import type { Response } from "../../src/llm/messages.ts";
import { compareCodeUnits } from "../../src/diagnostics.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { advance, inspectRun, RunResult } from "../../src/run.ts";
import { VectorDBMemory } from "../../src/memory/vectordb.ts";
import { Skill } from "../../src/skills/catalog.ts";
import { ToolContext } from "../../src/tools/context.ts";
import { toolText } from "../../src/nodes/agent/results.ts";
import { isReportedToolError } from "../../src/tools/errors.ts";
import { type FunctionTool, stringArguments, tool } from "../../src/tools/function_tools.ts";
import type { ToolEntry } from "../../src/tools/registry.ts";
import { callReply, NO_ENV, serving, textReply } from "../helpers/run_state_fixtures.ts";
import { heldScheduler } from "../helpers/schedulers.ts";

let tmp: string;
let root: string;

/** The status record tool `name` gave, read back from the JSON text the model sees of it. */
function seenAs(name: string, result: unknown): Record<string, unknown> {
  return JSON.parse(toolText(name, result)) as Record<string, unknown>;
}

/** Host tools by key, by the name a model sees each by: a group's as `<group>_<tool>`. */
function visible(tools: Readonly<Record<string, FunctionTool | FunctionTool[]>>): Map<string, FunctionTool> {
  const shown = new Map<string, FunctionTool>();
  for (const [key, entry] of Object.entries(tools)) {
    if (Array.isArray(entry)) for (const each of entry) shown.set(`${key}_${each.name}`, each);
    else shown.set(key, entry);
  }
  return shown;
}

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

beforeEach(() => {
  tmp = realpathSync(mkdtempSync(join(tmpdir(), "yian-chat-run-tools-")));
  root = join(tmp, "root");
  mkdirSync(join(root, "sub"), { recursive: true });
  writeFileSync(join(tmp, "outside.yaml"), "id: out\n");
});

/** A host over `root` whose approvals `approve` answers. */
function host(fields: Partial<ChatToolHost> = {}): ChatToolHost {
  return {
    root,
    approve: async () => {
      throw new Error("not expected to ask");
    },
    env: NO_ENV,
    docs: new Map(),
    ...fields,
  };
}

/** What `promise` rejected with. */
async function rejection(promise: Promise<unknown>): Promise<Error> {
  try {
    await promise;
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected a rejection");
}

describe("the run tools", () => {
  const TIMEOUT = 10_000;
  let runs: ChatRuns | null = null;

  afterEach(async () => {
    await runs?.dispose();
    runs = null;
  });

  const GREET = `id: greet
name: greet
input:
  who: !str
nodes:
  hi:
    kind: code
    input:
      who: \${input.who}
    output: !str
    code: |
      return "hi " + inputs["who"];
output: \${hi.output}
`;

  const LEGS = `id: legs
name: legs
typedefs:
  Leg:
    to: str
    mode: Literal(train, car)
input:
  legs: !list(Leg)
nodes:
  c: !int \${len(input.legs)}
output: \${c.output}
`;

  const ASK = `id: ask
name: ask
nodes:
  sure:
    kind: call
    call: ask
    input:
      prompt: "Which region?"
    output: !str
output: \${sure.output}
`;

  const ECHO = `id: echo
name: echo
input:
  messages: !Inbox(str)
defs:
  turn:
    input:
      closed: !bool
      messages: !Inbox(str)
    nodes:
      heard:
        kind: call
        call: take
        input:
          from: \${input.messages}
          take: all
      reply:
        kind: code
        input:
          events: \${heard.output.events}
        output: !str
        code: |
          return "echo: " + inputs["events"].join(" | ");
    output:
      closed: \${heard.output.closed}
nodes:
  chat:
    kind: loop
    call: turn
    input:
      closed: false
      messages: \${input.messages}
    while: not closed
output: done
`;

  /** The run tools over a fresh `ChatRuns` on a held clock, approvals answered by `answer`. */
  function runTools(
    answer: string | null = "Approve",
    scheduler = heldScheduler().scheduler,
    children: { env?: Env; embed?: Embed } = {},
  ) {
    runs = new ChatRuns({
      root,
      logDir: join(tmp, "runs"),
      home: join(tmp, "home"),
      env: children.env ?? NO_ENV,
      llmConfig: {},
      scheduler,
      skillsFor: () => null,
      ...(children.embed === undefined ? {} : { embed: children.embed }),
    });
    const requests: ApproveRequest[] = [];
    const owner = runs.owner({ runId: "chat-1", group: "flows", label: "yian-chat" });
    const tools = visible(
      chatRunTools(
        host({
          approve: async (request) => {
            requests.push(request);
            return answer;
          },
        }),
        runs,
        owner,
      ),
    );
    const signal = new AbortController();
    const invoke = async (name: string, args: Record<string, unknown>, abort: AbortSignal = signal.signal) => {
      const found = tools.get(name)!;
      const ctx = new ToolContext({
        callId: "chat-1/reply",
        attempt: 1,
        toolName: name,
        runId: "chat-1",
        node: "reply",
        context: undefined,
        signal: abort,
      });
      return found.execute(args, ctx);
    };
    const json = async (name: string, args: Record<string, unknown>, abort?: AbortSignal) =>
      seenAs(name, await invoke(name, args, abort));
    return { runs, tools, requests, invoke, json };
  }

  /** Wait on `run` until its status leaves `running`. */
  async function settledJson(json: (name: string, args: Record<string, unknown>) => Promise<Record<string, unknown>>, run: string) {
    let status = await json("flows_status", { run });
    while (status["status"] === "running") status = await json("flows_wait", { run, seconds: 60 });
    return status;
  }

  test(
    "an approved start runs the flow; its status says where it stands and flows_wait sees it end",
    async () => {
      writeFileSync(join(root, "greet.yaml"), GREET);
      const { requests, json } = runTools();
      const started = await json("flows_start", { flow: "greet.yaml", inputs: { who: "ada" } });
      expect(started).toMatchObject({ run: "r1", log: join(tmp, "runs", "chat-1-r1", "log.jsonl") });
      expect(started["seq"]).toBeUndefined();
      expect(requests).toHaveLength(1);
      expect(requests[0]!.title).toBe("Run greet.yaml");
      expect(requests[0]!.choices).toEqual(["Approve", "Approve this flow for this chat", "Reject"]);
      expect(requests[0]!.diff).toBeNull();
      expect(requests[0]!.body).toBe(
        'Starts greet.yaml in the background.\ninputs: {"who":"ada"}\nnode kinds: code\n' +
          "runs code on this machine (CODE nodes)\n" +
          "an approval for this chat doesn't cover changes to the files the code imports",
      );
      expect(await settledJson(json, "r1")).toMatchObject({ run: "r1", status: "succeeded", output: "hi ada" });
    },
    TIMEOUT,
  );

  test(
    "a draft's content starts once approved, its whole text on the card, nothing written",
    async () => {
      const { requests, json, invoke } = runTools();
      const args = { flow: "sub/greet.yaml", content: GREET, inputs: { who: "ada" } };
      expect(await json("flows_start", args)).toMatchObject({ run: "r1" });
      expect(requests[0]!.title).toBe("Run sub/greet.yaml (a draft, not written)");
      const first = requests[0]!.body.split("\n")[0];
      expect(first).toBe("Starts the draft sub/greet.yaml in the background.");
      expect(requests[0]!.diff).toContain("+++ sub/greet.yaml (draft)");
      expect(requests[0]!.diff).toContain('+      return "hi " + inputs["who"];');
      const settled = await settledJson(json, "r1");
      expect(settled).toMatchObject({ status: "succeeded", output: "hi ada" });
      expect(existsSync(join(root, "sub", "greet.yaml"))).toBe(false);
      const outside = { flow: "../greet.yaml", content: GREET, inputs: {} };
      const refused = await rejection(Promise.resolve(invoke("flows_start", outside)));
      expect(refused).toBeInstanceOf(ToolRefused);
      const bad = { ...args, content: "id: [\n" };
      const broken = await rejection(Promise.resolve(invoke("flows_start", bad)));
      expect(broken.message).toStartWith("invalid:\n");
    },
    TIMEOUT,
  );

  test(
    "a draft approved for this chat starts again unasked until its content changes",
    async () => {
      const { requests, json } = runTools("Approve this flow for this chat");
      const args = { flow: "greet.yaml", content: GREET, inputs: { who: "ada" } };
      expect(await json("flows_start", args)).toMatchObject({ run: "r1" });
      expect(await json("flows_start", args)).toMatchObject({ run: "r2" });
      expect(requests).toHaveLength(1);
      const changed = { ...args, content: GREET.replace('"hi "', '"hello "') };
      expect(await json("flows_start", changed)).toMatchObject({ run: "r3" });
      expect(requests).toHaveLength(2);
      expect(await settledJson(json, "r3")).toMatchObject({ output: "hello ada" });
    },
    TIMEOUT,
  );

  test(
    "the approval card cuts long inputs",
    async () => {
      writeFileSync(join(root, "greet.yaml"), GREET);
      const { requests, invoke } = runTools("Reject");
      await invoke("flows_start", { flow: "greet.yaml", inputs: { who: "x".repeat(5000) } });
      const inputs = requests[0]!.body.split("\n")[1]!;
      expect(inputs).toBe(`inputs: {"who":"${"x".repeat(4000 - '{"who":"'.length)}... (cut at 4000 characters)`);
    },
    TIMEOUT,
  );

  test(
    "the approval lists what the flow and the flows it uses can do",
    async () => {
      writeFileSync(join(root, "nap.yaml"), "id: nap\nname: nap\nnodes:\n  nap: {kind: call, call: sleep, input: {for: 1s}}\noutput: ${nap.output}\n");
      writeFileSync(
        join(root, "main.yaml"),
        "id: main\nname: main\nnodes:\n  work:\n    kind: call\n    call: nap\noutput: ${work.output}\nuses:\n  nap: ./nap\n",
      );
      const { requests, invoke } = runTools("Reject");
      expect(await invoke("flows_start", { flow: "main.yaml", inputs: {} })).toBe("not started: rejected");
      expect(requests[0]!.body).toBe("Starts main.yaml in the background.\ninputs: {}\nnode kinds: call, tool\ntools it calls: sleep");
    },
    TIMEOUT,
  );

  test(
    "a library isn't started, and no approval is asked",
    async () => {
      writeFileSync(join(root, "lib.yaml"), "id: lib\nname: lib\ndefs:\n  one: {kind: code, code: return 1, output: !int}\n");
      const { requests, invoke } = runTools();
      expect(await invoke("flows_start", { flow: "lib.yaml", inputs: {} })).toBe(
        "not started: lib.yaml is a library (defs: and no nodes:); import it with uses: and call a def",
      );
      expect(requests).toEqual([]);
    },
    TIMEOUT,
  );

  test(
    "a rejected or unanswered start starts nothing",
    async () => {
      writeFileSync(join(root, "greet.yaml"), GREET);
      const rejected = runTools("Reject");
      expect(await rejected.invoke("flows_start", { flow: "greet.yaml", inputs: { who: "a" } })).toBe("not started: rejected");
      await runs!.dispose();
      const unanswered = runTools(null);
      expect(await unanswered.invoke("flows_start", { flow: "greet.yaml", inputs: { who: "a" } })).toBe(
        "not started: not approved (no terminal, or the prompt ended)",
      );
      const error = await rejection(Promise.resolve(unanswered.invoke("flows_status", { run: "r1" })));
      expect(isReportedToolError(error)).toBe(true);
      expect(existsSync(join(tmp, "runs"))).toBe(false);
    },
    TIMEOUT,
  );

  test(
    "approving a flow for this chat skips its later approvals until the file changes",
    async () => {
      writeFileSync(join(root, "greet.yaml"), GREET);
      const { requests, json } = runTools("Approve this flow for this chat");
      expect(await json("flows_start", { flow: "greet.yaml", inputs: { who: "a" } })).toMatchObject({ run: "r1" });
      expect(await json("flows_start", { flow: "./sub/../greet.yaml", inputs: { who: "b" } })).toMatchObject({ run: "r2" });
      expect(requests).toHaveLength(1);
      writeFileSync(join(root, "greet.yaml"), GREET.replace('"hi "', '"hello "'));
      expect(await json("flows_start", { flow: "greet.yaml", inputs: { who: "c" } })).toMatchObject({ run: "r3" });
      expect(requests).toHaveLength(2);
    },
    TIMEOUT,
  );

  test(
    "an approval for this chat is asked again after a used flow or a referenced code file changes",
    async () => {
      writeFileSync(join(root, "child.yaml"), GREET.replace("id: greet", "id: child"));
      writeFileSync(join(root, "lib.ts"), 'export function run() { return "one"; }\n');
      writeFileSync(
        join(root, "main.yaml"),
        "id: main\nname: main\nnodes:\n  work:\n    kind: call\n    call: child\n    input:\n      who: x\n" +
          "  lib:\n    kind: code\n    output: !str\n    code: ./lib.ts:run\noutput: ${work.output}\nuses:\n  child: ./child\n",
      );
      const { requests, json } = runTools("Approve this flow for this chat");
      const start = () => json("flows_start", { flow: "main.yaml", inputs: {} });
      await start();
      expect(requests[0]!.body).toContain("runs code on this machine (CODE nodes; files: lib.ts)");
      await start();
      expect(requests).toHaveLength(1);
      // The used flow's inline code changes.
      writeFileSync(join(root, "child.yaml"), GREET.replace("id: greet", "id: child").replace('"hi "', '"yo "'));
      await start();
      expect(requests).toHaveLength(2);
      await start();
      expect(requests).toHaveLength(2);
      // The referenced file changes.
      writeFileSync(join(root, "lib.ts"), 'export function run() { return "two"; }\n');
      await start();
      expect(requests).toHaveLength(3);
    },
    TIMEOUT,
  );

  test(
    "inputs of the wrong type, or a required input missing, are refused before asking",
    async () => {
      writeFileSync(join(root, "greet.yaml"), GREET);
      const { requests, invoke } = runTools();
      const typed = await rejection(Promise.resolve(invoke("flows_start", { flow: "greet.yaml", inputs: { who: 3 } })));
      expect(isReportedToolError(typed)).toBe(true);
      expect(typed.message).toStartWith("bad inputs: ");
      expect(typed.message).toContain("who");
      const missing = await rejection(Promise.resolve(invoke("flows_start", { flow: "greet.yaml", inputs: {} })));
      expect(missing.message).toStartWith("bad inputs: ");
      expect(missing.message).toContain("who");
      // A failure inside the value names where, as the start's own check does.
      writeFileSync(join(root, "legs.yaml"), LEGS);
      const legs = [
        { to: "a", mode: "train" },
        { to: "b", mode: "boat" },
      ];
      const nested = await rejection(Promise.resolve(invoke("flows_start", { flow: "legs.yaml", inputs: { legs } })));
      expect(nested.message).toBe('bad inputs: input \'legs\' at [1].mode: "boat" is not one of Literal(car, train)');
      expect(requests).toHaveLength(0);
    },
    TIMEOUT,
  );

  test(
    "a flow's int output reads back as its digits in flows_status and flows_wait",
    async () => {
      writeFileSync(join(root, "legs.yaml"), LEGS);
      const { invoke, json } = runTools();
      await json("flows_start", { flow: "legs.yaml", inputs: { legs: [{ to: "a", mode: "train" }] } });
      expect(await settledJson(json, "r1")).toMatchObject({ status: "succeeded", output: 1 });
      expect(toolText("flows_status", await invoke("flows_status", { run: "r1" }))).toContain('"output":1');
    },
    TIMEOUT,
  );

  test(
    "a flow's memory input is left out of its inputs: the run binds its default; a value for it is refused",
    async () => {
      writeFileSync(
        join(root, "remember.yaml"),
        "id: remember\nname: remember\ninput:\n  lessons: !FileMemory\nnodes:\n  add:\n    kind: call\n" +
          "    call: {lessons: append}\n    input: {path: MEMORY.md, text: noted}\noutput: ${add.output.path}\n",
      );
      const { requests, invoke, json } = runTools();
      const given = await rejection(
        Promise.resolve(invoke("flows_start", { flow: "remember.yaml", inputs: { lessons: "x" } })),
      );
      expect(isReportedToolError(given)).toBe(true);
      expect(given.message).toBe("bad inputs: `lessons` is a memory; leave it out, and the run gets its default");
      expect(requests).toHaveLength(0);
      await json("flows_start", { flow: "remember.yaml", inputs: {} });
      const done = await settledJson(json, "r1");
      expect(done).toMatchObject({ status: "succeeded", output: "MEMORY.md" });
      expect((done["memory_notes"] as string[])[0]).toEndWith("(new)");
    },
    TIMEOUT,
  );

  test(
    "a VectorDB flow's card needs a default embedding model for a new store, and its start uses that model",
    async () => {
      writeFileSync(
        join(root, "cases.yaml"),
        "id: cases\nname: cases\ninput:\n  line: !str\n  cases: !VectorDBMemory\nnodes:\n  add:\n" +
          "    kind: call\n    call: {cases: add}\n    input:\n      text: ${input.line}\noutput: ${add.output.id}\n",
      );
      const embed: Embed = async (texts) => texts.map(() => Float32Array.of(1, 1, 0, 0));
      // The store a test reads is its own, never the account's.
      const vars: Record<string, string> = { YIAN_AUTH_FILE: join(tmp, "auth.json") };
      const env: Env = (name) => vars[name];
      const none = runTools("Approve", undefined, { env, embed });
      const refused = String(await none.invoke("flows_start", { flow: "cases.yaml", inputs: { line: "a" } }));
      expect(refused).toBe("not started: `cases` needs an embedding model: run `yian config embedding provider/model`");
      // Refused before the card: the user isn't asked to approve a start that can't happen.
      expect(none.requests).toHaveLength(0);
      expect(existsSync(join(tmp, "home", ".yian"))).toBe(false);
      await runs?.dispose();

      vars["YIAN_DEFAULT_EMBEDDING"] = "openai/text-embedding-3-small";
      const { requests, json } = runTools("Approve", undefined, { env, embed });
      const identity = `${projectKey(root)}/cases/cases`;
      await json("flows_start", { flow: "cases.yaml", inputs: { line: "Check sums twice." } });
      expect(requests).toHaveLength(1);
      expect(requests[0]!.body).toContain(`\nmemories: cases → ${identity} (new)`);
      expect(await settledJson(json, "r1")).toMatchObject({ status: "succeeded", output: 1 });
      const store = new VectorDBMemory({ path: defaultLocation(join(tmp, "home"), identity, "VectorDBMemory") });
      const stored = await store.stored({ signal: new AbortController().signal });
      expect(stored?.embedding).toEqual({ provider: "openai", model: "text-embedding-3-small" });
    },
    TIMEOUT,
  );

  test(
    "the start card names the memories a flow binds, by identity, new or existing",
    async () => {
      writeFileSync(
        join(root, "remember.yaml"),
        "id: remember\nname: remember\ninput:\n  lessons: !FileMemory\n  notes: !Optional(FileMemory)\n" +
          "nodes:\n  add:\n    kind: call\n    call: {lessons: append}\n" +
          "    input: {path: MEMORY.md, text: noted}\noutput: ${add.output.path}\n",
      );
      const { requests, json } = runTools();
      const identity = `${projectKey(root)}/remember/lessons`;
      await json("flows_start", { flow: "remember.yaml", inputs: {} });
      expect((await settledJson(json, "r1"))["status"]).toBe("succeeded");
      // The Optional memory stays null, so it isn't named; no folder is shown.
      expect(requests[0]!.body).toContain(`\nmemories: lessons → ${identity} (new)`);
      expect(requests[0]!.body).not.toContain("notes");
      expect(requests[0]!.body).not.toContain(join(tmp, "home"));
      await json("flows_start", { flow: "remember.yaml", inputs: {} });
      expect((await settledJson(json, "r2"))["status"]).toBe("succeeded");
      expect(requests[1]!.body).toContain(`\nmemories: lessons → ${identity} (existing)`);
    },
    TIMEOUT,
  );

  test(
    "a symlinked flow file's memory is keyed by the folder it was given in, as yian run keys it",
    async () => {
      mkdirSync(join(root, "proj"));
      mkdirSync(join(root, "shared"));
      writeFileSync(
        join(root, "shared", "remember.yaml"),
        "id: remember\nname: remember\ninput:\n  lessons: !FileMemory\nnodes:\n  add:\n    kind: call\n" +
          "    call: {lessons: append}\n    input: {path: MEMORY.md, text: noted}\noutput: ${add.output.path}\n",
      );
      symlinkSync(join(root, "shared", "remember.yaml"), join(root, "proj", "remember.yaml"));
      const home = join(tmp, "home");
      mkdirSync(home);
      const err: string[] = [];
      const io: CliIo = {
        stdout: { write: async () => {} },
        stderr: { write: async (text) => void err.push(text) },
        stdin: new PassThrough(),
        stderrStream: new PassThrough(),
        env: (name) => (name === "HOME" ? home : undefined),
      };
      expect(await main(["run", join(root, "proj", "remember.yaml"), "--quiet"], io), err.join("")).toBe(0);
      const key = projectKey(join(root, "proj"));
      expect(readdirSync(join(home, ".yian", "memory"))).toEqual([key]);
      const { json } = runTools();
      await json("flows_start", { flow: "proj/remember.yaml", inputs: {} });
      const done = await settledJson(json, "r1");
      expect(done).toMatchObject({ status: "succeeded" });
      // The memory yian run made: not new, and no second project.
      expect(done["memory_notes"]).toBeUndefined();
      expect(readdirSync(join(home, ".yian", "memory"))).toEqual([key]);
    },
    TIMEOUT,
  );

  test(
    "inputs a flow doesn't declare, or not a record, are refused before asking",
    async () => {
      writeFileSync(join(root, "greet.yaml"), GREET);
      const { requests, invoke } = runTools();
      const error = await rejection(Promise.resolve(invoke("flows_start", { flow: "greet.yaml", inputs: { whom: "a" } })));
      expect(isReportedToolError(error)).toBe(true);
      expect(error.message).toStartWith('bad inputs: unknown input "whom"');
      const notRecord = await rejection(Promise.resolve(invoke("flows_start", { flow: "greet.yaml", inputs: [] })));
      expect(isReportedToolError(notRecord)).toBe(true);
      expect(requests).toHaveLength(0);
    },
    TIMEOUT,
  );

  test(
    "a flow that doesn't load, or a path outside the root, is refused before asking",
    async () => {
      writeFileSync(join(root, "bad.yaml"), "id: [\n");
      const { requests, invoke } = runTools();
      const invalid = await rejection(Promise.resolve(invoke("flows_start", { flow: "bad.yaml", inputs: {} })));
      expect(isReportedToolError(invalid)).toBe(true);
      expect(invalid.message).toStartWith("invalid:\n");
      const outside = await rejection(Promise.resolve(invoke("flows_start", { flow: "../outside.yaml", inputs: {} })));
      expect(outside).toBeInstanceOf(ToolRefused);
      expect(requests).toHaveLength(0);
    },
    TIMEOUT,
  );

  test(
    "a question is relayed and answered; a null answer is an error the model reads",
    async () => {
      writeFileSync(join(root, "ask.yaml"), ASK);
      const { json, invoke } = runTools();
      await json("flows_start", { flow: "ask.yaml", inputs: {} });
      const waiting = await settledJson(json, "r1");
      expect(waiting).toMatchObject({ status: "waiting", asks: [{ call: "chat-1-r1/sure", question: "Which region?" }] });
      const nulled = await rejection(Promise.resolve(invoke("flows_answer", { run: "r1", call: "chat-1-r1/sure", answer: null })));
      expect(isReportedToolError(nulled)).toBe(true);
      expect(await json("flows_answer", { run: "r1", call: "chat-1-r1/sure", answer: "eu" })).toMatchObject({ status: "running" });
      expect(await settledJson(json, "r1")).toMatchObject({ status: "succeeded", output: "eu" });
    },
    TIMEOUT,
  );

  test(
    "messages reach a child's inbox, its replies are shown once, and a close ends it",
    async () => {
      writeFileSync(join(root, "echo.yaml"), ECHO);
      const { json } = runTools();
      await json("flows_start", { flow: "echo.yaml", inputs: {} });
      expect(await settledJson(json, "r1")).toMatchObject({ status: "waiting" });
      expect(await json("flows_message", { run: "r1", inbox: "messages", text: "hi" })).toMatchObject({ status: "running" });
      expect(await settledJson(json, "r1")).toMatchObject({ status: "waiting", replies: ["echo: hi"] });
      expect((await json("flows_status", { run: "r1" }))["replies"]).toBeUndefined();
      expect(await json("flows_close_inbox", { run: "r1", inbox: "messages" })).toMatchObject({ status: "running" });
      expect(await settledJson(json, "r1")).toMatchObject({ status: "succeeded", output: "done" });
      expect((await json("flows_message", { run: "r1", inbox: "messages", text: "late" }))["note"]).toContain("ended");
    },
    TIMEOUT,
  );

  test(
    "flows_stop cancels a child",
    async () => {
      writeFileSync(join(root, "ask.yaml"), ASK);
      const { json } = runTools();
      await json("flows_start", { flow: "ask.yaml", inputs: {} });
      await settledJson(json, "r1");
      await json("flows_stop", { run: "r1" });
      expect(await settledJson(json, "r1")).toMatchObject({ status: "cancelled" });
    },
    TIMEOUT,
  );

  test("an unknown run is an error the model reads, for every run tool", async () => {
    const { invoke } = runTools();
    for (const [name, args] of [
      ["flows_status", {}],
      ["flows_wait", { seconds: 1 }],
      ["flows_answer", { call: "c", answer: "x" }],
      ["flows_message", { inbox: "m", text: "x" }],
      ["flows_interrupt", { inbox: "m" }],
      ["flows_close_inbox", { inbox: "m" }],
      ["flows_stop", {}],
    ] as const) {
      const error = await rejection(Promise.resolve(invoke(name, { run: "r9", ...args })));
      expect(isReportedToolError(error)).toBe(true);
      expect(error.message).toBe("no run 'r9'");
    }
  });

  test(
    "flows_wait returns at its timeout, at once for a status not yet shown, and when its call ends",
    async () => {
      writeFileSync(join(root, "ask.yaml"), ASK);
      const { scheduler, open } = heldScheduler();
      const { json, runs: children } = runTools("Approve", scheduler);
      await json("flows_start", { flow: "ask.yaml", inputs: {} });
      // Not shown yet: returned at once, though the clock is held.
      while (children.status("r1").status === "running") await children.next("r1");
      const waiting = await json("flows_wait", { run: "r1", seconds: 60 });
      expect(waiting).toMatchObject({ status: "waiting" });
      // Shown: the next wait lasts until the call's signal aborts.
      const call = new AbortController();
      const ended = json("flows_wait", { run: "r1", seconds: 60 }, call.signal);
      await until(() => children.waitingNexts() === 1);
      call.abort();
      expect(await ended).toMatchObject({ status: "waiting" });
      // ... or until the chat cancels the waits (an interrupt)...
      const cancelled = json("flows_wait", { run: "r1", seconds: 60 });
      await until(() => children.waitingNexts() === 1);
      children.cancelWaits();
      expect(await cancelled).toMatchObject({ status: "waiting" });
      // ... or its timeout, capped.
      const timed = json("flows_wait", { run: "r1", seconds: 10_000 });
      await until(() => children.waitingNexts() === 1);
      const before = scheduler.now();
      open();
      expect(await timed).toMatchObject({ status: "waiting" });
      expect(scheduler.now() - before).toBe(120_000);
    },
    TIMEOUT,
  );

  test(
    "a status's long text fields are cut, with a note",
    async () => {
      writeFileSync(join(root, "long.yaml"), GREET.replace('return "hi " + inputs["who"];', 'return "x".repeat(5000);'));
      const { json } = runTools();
      await json("flows_start", { flow: "long.yaml", inputs: { who: "a" } });
      const status = await settledJson(json, "r1");
      expect(status["output"]).toBe(`${"x".repeat(4000)}... (cut at 4000 characters)`);
    },
    TIMEOUT,
  );

  test("only the reading tools may run beside other calls", () => {
    const { tools } = runTools();
    const safe = [...tools].filter(([, each]) => each.isConcurrencySafe?.({}) === true).map(([name]) => name);
    expect(safe).toEqual(["flows_status", "flows_wait"]);
    expect(chatFileTools(host()).find((each) => each.name === "write_flow")!.isConcurrencySafe).toBeNull();
  });
});

/** Resolve once `check` holds, polling each turn of the event loop. */
async function until(check: () => boolean): Promise<void> {
  while (!check()) await new Promise((resolve) => setTimeout(resolve, 1));
}

describe("report_issue", () => {
  const TIMEOUT = 10_000;
  let runs: ChatRuns | null = null;

  afterEach(async () => {
    await runs?.dispose();
    runs = null;
  });

  test(
    "starts the bundled report flow with no start card, the chat's facts first in its context",
    async () => {
      const env = (name: string) => (name === "XAI_API_KEY" ? "test-xai-key" : NO_ENV(name));
      const report = { title: "check crashes", body: "It crashed." };
      const model = new ScriptedModel([textReply("Drafted."), textReply(JSON.stringify(report))], {
        provider: "xai",
        model: "grok-4.3",
      });
      runs = new ChatRuns({
        root,
        logDir: join(tmp, "runs"),
        home: join(tmp, "home"),
        env,
        llmConfig: { provider: "xai", model: "grok-4.3" },
        models: serving(model, env),
        scheduler: heldScheduler().scheduler,
        skillsFor: () => null,
      });
      const owner = runs.owner({ runId: "chat-1", group: "flows", label: "yian-chat" });
      const filed: Record<string, unknown>[] = [];
      const fileIssue = tool({
        name: "file_issue",
        description: "Take the report.",
        parameters: stringArguments(["title", "", true], ["body", "", true]),
        execute: (args) => {
          filed.push(args);
          return "not filed: saved only";
        },
      });
      const tools = chatRunTools(host({ env }), runs, owner, {
        report: {
          facts: "yian version: 0.0.4\nchat run log: ~/c.jsonl",
          homes: ["/home/ada"],
          fileIssue,
        },
      });
      const reportIssue = tools["report_issue"] as FunctionTool;
      const ctx = new ToolContext({
        callId: "chat-1/reply",
        attempt: 1,
        toolName: "report_issue",
        runId: "chat-1",
        node: "reply",
        context: undefined,
        signal: new AbortController().signal,
      });
      const args = { problem: "yian check crashed", context: "ran: yian check a.yaml" };
      const started = seenAs("report_issue", await reportIssue.execute(args, ctx));
      expect(started).toMatchObject({ run: "r1" });
      let status = runs.status("r1");
      while (status.status === "running") status = await runs.next("r1", status.seq);
      expect(status).toMatchObject({ status: "succeeded", output: "not filed: saved only" });
      expect(filed).toEqual([report]);
      const prompt = model.requests[0]!.messages.at(-1)!.text();
      expect(prompt).toContain("The problem: yian check crashed");
      expect(prompt).toContain(
        "yian version: 0.0.4\nchat run log: ~/c.jsonl\nran: yian check a.yaml",
      );
      expect(runs.live()).toEqual([]);
      // Without its options, the chat's run tools have no report_issue.
      expect(Object.keys(chatRunTools(host(), runs, owner))).toEqual(["flows"]);
    },
    TIMEOUT,
  );
});

describe("author_flow and review_flow", () => {
  const TIMEOUT = 20_000;
  const env = (name: string) => (name === "XAI_API_KEY" ? "test-xai-key" : NO_ENV(name));
  const guide = new Skill({
    name: "composing-agents",
    description: "How to write flows.",
    instructions: "Write small typed nodes.",
  });
  let runs: ChatRuns | null = null;

  afterEach(async () => {
    await runs?.dispose();
    runs = null;
  });

  const GREET = `id: greet
name: greet
input:
  who: !str
nodes:
  hi:
    kind: code
    input:
      who: \${input.who}
    output: !str
    code: |
      return "hi " + inputs["who"];
output: \${hi.output}
`;

  /**
   * The run tools with the authoring tools, over runs whose agents `replies` answer and whose
   * approvals `answer` answers; `requests` holds each approval asked.
   */
  function authoringTools(replies: Response[], answer: string | null = "Approve") {
    const model = new ScriptedModel(replies, { provider: "xai", model: "grok-4.3" });
    runs = new ChatRuns({
      root,
      logDir: join(tmp, "runs"),
      home: join(tmp, "home"),
      env,
      llmConfig: { provider: "xai", model: "grok-4.3" },
      models: serving(model, env),
      scheduler: heldScheduler().scheduler,
      skillsFor: () => null,
    });
    const requests: ApproveRequest[] = [];
    const approve = async (request: ApproveRequest) => {
      requests.push(request);
      return answer;
    };
    const owner = runs.owner({ runId: "chat-1", group: "flows", label: "yian-chat" });
    const tools = chatRunTools(host({ env, approve, docs: new Map([["flows", "page"]]) }), runs, owner, {
      authoring: { skills: new Map([[guide.name, guide]]) },
    });
    const invoke = async (name: string, args: Record<string, unknown>) => {
      const ctx = new ToolContext({
        callId: "chat-1/reply",
        attempt: 1,
        toolName: name,
        runId: "chat-1",
        node: "reply",
        context: undefined,
        signal: new AbortController().signal,
      });
      return visible(tools).get(name)!.execute(args, ctx);
    };
    return { runs, owner, model, requests, invoke };
  }

  /**
   * Drive child `id` to its end, answering each question it asks with the next of `answers`;
   * return its last status and the questions asked.
   */
  async function drive(children: ChildFlows, id: string, answers: string[]) {
    const asked: string[] = [];
    let status = children.status(id);
    for (;;) {
      if (status.status === "running") status = await children.next(id, status.seq);
      else if (status.status === "waiting" && (status.asks ?? []).length > 0) {
        const ask = status.asks![0]!;
        asked.push(ask.question);
        const reply = answers.shift();
        if (reply === undefined) throw new Error(`no answer left for ${ask.question}`);
        status = await children.answer(id, ask.call, reply);
      } else return { status, asked };
    }
  }

  /** A flow reading an Inbox input, which a try started by author_flow can't feed. */
  const INBOX = `id: inbox
name: inbox
input:
  notes: !Inbox(str)
nodes:
  heard:
    kind: call
    call: take
    input:
      from: \${input.notes}
      take: all
output: \${heard.output.closed}
`;

  /** A flow asking a question, so a try of it waits. */
  const ASKS = `id: asks
name: asks
nodes:
  sure:
    kind: call
    call: ask
    input:
      prompt: "Which region?"
    output: !str
output: \${sure.output}
`;

  const BRIEF = {
    purpose: "Greet someone.",
    runs: "The user, daily.",
    inputs: "who",
    output: "a greeting",
    done: "It says hi.",
  };

  /** The contract the scripted model agrees, at `path`. */
  const contract = (path: string) => ({
    path,
    inputs: "who: !str",
    output: "str",
    outline: ["hi: code, greets who"],
    samples: '{"who": "ada"}',
  });

  /** A tool-granted agent's replies when it calls no tool: its text, then its typed answer. */
  const answered = (value: unknown) => [textReply("Done."), textReply(JSON.stringify(value))];

  /** The replies of one round that drafts `text`, is reviewed with `findings` and plans `next`. */
  const round = (text: string, findings: unknown[], next: string) => [
    textReply(text),
    ...answered({ findings }),
    ...answered({ next, changes: next === "refine" ? "Shorter." : "" }),
  ];

  /**
   * The first request of each call of the agents whose system prompt holds `marker` (the system
   * prompt and the prompt alone), its messages joined.
   */
  const requestsOf = (model: ScriptedModel, marker: string) =>
    model.requests
      .filter((request) => request.messages.length === 2)
      .filter((request) => request.messages[0]!.text().includes(marker))
      .map((request) => request.messages.map((each) => each.text()).join("\n"));

  /** Have `owner` record the tools of each start in `starts`, once the start builds them. */
  function recordStarts(owner: ChildFlowOwner): Record<string, ToolEntry>[] {
    const starts: Record<string, ToolEntry>[] = [];
    const start = owner.start.bind(owner);
    owner.start = (flow, inputs, options = {}) => {
      const { tools = {} } = options;
      const record = (built: Readonly<Record<string, ToolEntry>>) => (starts.push({ ...built }), built);
      const recorded = typeof tools === "function" ? (runId: string) => record(tools(runId)) : record(tools);
      return start(flow, inputs, { ...options, tools: recorded });
    };
    return starts;
  }

  test(
    "author_flow goes from the goal through a fixed check failure and a review to a write",
    async () => {
      const finding = {
        severity: "low",
        area: "clarity",
        node: "flow",
        problem: "No description.",
        fix: "Add one.",
      };
      const draftArgs = { flow: "greet.yaml", content: GREET, inputs: { who: "ada" } };
      const { runs, model, requests, invoke } = authoringTools([
        callReply(["ask_user", { question: "Who runs it, and how often?" }, "b1"]),
        ...answered(BRIEF),
        callReply(["ask_user", { question: "Here is the contract. Does it look right?" }, "c1"]),
        ...answered(contract("greet.yaml")),
        textReply("id: greet\nnodes: [\n"),
        // A fenced reply is taken from inside its fence.
        textReply(`Here is the fixed flow:\n\n\`\`\`yaml\n${GREET}\`\`\`\n`),
        ...answered({ findings: [finding] }),
        callReply(["flows_start", draftArgs, "t1"]),
        callReply(["flows_wait", { run: "r2", seconds: 60 }, "t2"]),
        textReply("With who=ada it said hi ada."),
        callReply(["ask_user", { question: "It loads and said hi ada. Write it?" }, "p1"]),
        ...answered({ next: "write", changes: "" }),
      ]);
      const goal = "greet someone by name";
      const started = seenAs("author_flow", await invoke("author_flow", { goal }));
      expect(started).toMatchObject({ run: "r1", status: "running" });
      const answers = ["Me, daily.", "Yes.", "Write it."];
      const { status, asked } = await drive(runs, "r1", answers);
      expect(status.failure ?? null).toBeNull();
      expect(status).toMatchObject({
        status: "succeeded",
        output: { path: "greet.yaml", outcome: "wrote greet.yaml" },
      });
      expect(asked).toEqual([
        "Who runs it, and how often?",
        "Here is the contract. Does it look right?",
        "It loads and said hi ada. Write it?",
      ]);
      expect(readFileSync(join(root, "greet.yaml"), "utf8")).toBe(GREET);
      // The try ran the draft before it was written, and the write asked last.
      expect(requests.map((each) => each.title)).toEqual([
        "Run greet.yaml (a draft, for author_flow)",
        "Write greet.yaml",
      ]);
      expect(runs.status("r2")).toMatchObject({ status: "succeeded", output: "hi ada" });
      // The fix saw the check's error, and the review the fixed flow.
      // The YAML error at the end of the draft is located at the bracket it never closes.
      const fix = requestsOf(model, "You fix a yian")[0]!;
      expect(fix).toContain("Its error:\n╭─ greet.yaml:2 ─╮");
      expect(fix).toContain("│ ❱ 2 nodes: [   │");
      expect(fix).toContain("invalid YAML: Flow sequence in block collection must be sufficiently indented");
      expect(requestsOf(model, "You review flows")[0]).toContain("Review the flow greet.yaml:");
      const plan = requestsOf(model, "finish a yian")[0]!;
      expect(plan).toContain("No description.");
      expect(plan).toContain("Round 1 of 5; the last: false.");
      const drafting = model.requests.find((r) =>
        r.messages[0]!.text().includes("You write flows"),
      );
      expect(drafting!.tools.map((each) => each.name)).toContain("load_skill");
    },
    TIMEOUT,
  );

  test(
    "a draft with a high-severity security finding is never tried, and the user is told why",
    async () => {
      const risk = {
        severity: "high",
        area: "security",
        node: "hi",
        problem: "Runs what the input says.",
        fix: "Don't.",
      };
      const { runs, model, requests, invoke } = authoringTools([
        ...answered(BRIEF),
        ...answered(contract("greet.yaml")),
        ...round(GREET, [risk], "stop"),
      ]);
      await invoke("author_flow", { goal: "greet" });
      const { status } = await drive(runs, "r1", []);
      expect(status).toMatchObject({
        status: "succeeded",
        output: { outcome: "not written: the user stopped before the write" },
      });
      expect(requests).toEqual([]);
      expect(requestsOf(model, "You try a draft")).toEqual([]);
      const why = "not tried: the review found a high-severity security problem";
      expect(requestsOf(model, "finish a yian")[0]).toContain(
        `${why} (hi: Runs what the input says.)`,
      );
    },
    TIMEOUT,
  );

  test(
    "a contract path leaving the folder is asked for again; a draft with an inbox isn't tried",
    async () => {
      const { runs, model, requests, invoke } = authoringTools([
        ...answered(BRIEF),
        ...answered(contract("../inbox.yaml")),
        ...answered(contract("inbox.yaml")),
        ...round(INBOX, [], "stop"),
      ]);
      await invoke("author_flow", { goal: "read notes" });
      const { status } = await drive(runs, "r1", []);
      expect(status).toMatchObject({ status: "succeeded", output: { path: "inbox.yaml" } });
      const proposals = requestsOf(model, "a yian flow's contract");
      expect(proposals).toHaveLength(2);
      expect(proposals[1]).toContain("The path ../inbox.yaml was refused: it goes up a folder");
      expect(requests).toEqual([]);
      expect(requestsOf(model, "finish a yian")[0]).toContain(
        "not tried: it takes an Inbox input, which a try here can't feed",
      );
    },
    TIMEOUT,
  );

  test(
    "three refused contract paths end the run with no draft",
    async () => {
      const { runs, model, invoke } = authoringTools([
        ...answered(BRIEF),
        ...answered(contract("/etc/x.yaml")),
        ...answered(contract("notes.txt")),
        ...answered(contract("~/x.yaml")),
      ]);
      await invoke("author_flow", { goal: "anything" });
      const { status } = await drive(runs, "r1", []);
      expect(status).toMatchObject({
        status: "succeeded",
        output: {
          path: "~/x.yaml",
          outcome: "not written: no usable path was agreed for the flow",
        },
      });
      expect(requestsOf(model, "You write flows")).toEqual([]);
    },
    TIMEOUT,
  );

  test(
    "a path given to author_flow is refused at once outside the folder, and used as given",
    async () => {
      const { runs, invoke } = authoringTools([
        ...answered(BRIEF),
        ...answered(contract("elsewhere.yaml")),
        ...round(INBOX, [], "stop"),
      ]);
      for (const path of ["../x.yaml", "notes.txt"]) {
        const started = Promise.resolve(invoke("author_flow", { goal: "g", path }));
        expect(await rejection(started)).toBeInstanceOf(ToolRefused);
      }
      expect(runs.live()).toEqual([]);
      await invoke("author_flow", { goal: "read notes", path: "sub/inbox.yaml" });
      const { status } = await drive(runs, "r1", []);
      expect(status).toMatchObject({ status: "succeeded", output: { path: "sub/inbox.yaml" } });
    },
    TIMEOUT,
  );

  /** GREET's code as the def `greet`, which its node calls by a mistyped name: the flow loads. */
  const TYPO = `id: greet
name: greet
input:
  who: !str
defs:
  greet:
    input:
      who: !str
    nodes:
      hi:
        kind: code
        input:
          who: \${input.who}
        output: !str
        code: |
          return "hi " + inputs["who"];
    output: \${hi.output}
nodes:
  said:
    kind: call
    call: gret
    input:
      who: \${input.who}
output: \${said.output}
`;

  /** The replies of one round that drafts `draft`, fixes it to each of `fixes`, and stops untried. */
  const fixedRound = (draft: string, fixes: string[]) => [
    textReply(draft),
    ...fixes.map(textReply),
    ...answered({ findings: [] }),
    textReply("The user rejected the run."),
    ...answered({ next: "stop", changes: "" }),
  ];

  test(
    "a draft that loads but calls a name close to one it defines is fixed until no hint is left",
    async () => {
      const fixed = TYPO.replace("call: gret", "call: greet");
      const { runs, model, invoke } = authoringTools([
        ...answered(BRIEF),
        ...answered(contract("greet.yaml")),
        ...fixedRound(TYPO, [TYPO, fixed]),
      ]);
      await invoke("author_flow", { goal: "greet" });
      const { status } = await drive(runs, "r1", []);
      expect(status).toMatchObject({
        status: "succeeded",
        output: { outcome: "not written: the user stopped before the write" },
      });
      const fixes = requestsOf(model, "You fix a yian");
      expect(fixes).toHaveLength(2);
      for (const fix of fixes) {
        expect(fix).toContain("call: gret\n");
        expect(fix).toContain("- gret (node 'said'): did you mean the def greet?");
      }
      expect(requestsOf(model, "You review flows")[0]).toContain("call: greet\n");
    },
    TIMEOUT,
  );

  test(
    "a hinted name the fixes keep is left after three tries, for the plan to show",
    async () => {
      const { runs, model, invoke } = authoringTools([
        ...answered(BRIEF),
        ...answered(contract("greet.yaml")),
        ...fixedRound(TYPO, [TYPO, TYPO, TYPO]),
      ]);
      await invoke("author_flow", { goal: "greet" });
      const { status } = await drive(runs, "r1", []);
      expect(status).toMatchObject({ status: "succeeded" });
      expect(requestsOf(model, "You fix a yian")).toHaveLength(3);
      const plan = requestsOf(model, "finish a yian")[0]!;
      expect(plan).toContain("call: gret\n");
      expect(plan).toContain('"what": "def"');
    },
    TIMEOUT,
  );

  test(
    "a draft calling a host tool's name with no hint isn't sent to a fix",
    async () => {
      const search = TYPO.replace("call: gret", "call: search");
      const { runs, model, invoke } = authoringTools([
        ...answered(BRIEF),
        ...answered(contract("greet.yaml")),
        ...fixedRound(search, []),
      ]);
      await invoke("author_flow", { goal: "greet" });
      const { status } = await drive(runs, "r1", []);
      expect(status).toMatchObject({ status: "succeeded" });
      expect(requestsOf(model, "You fix a yian")).toEqual([]);
      expect(requestsOf(model, "You review flows")[0]).toContain("call: search\n");
    },
    TIMEOUT,
  );

  test(
    "five rounds without a write end with their own outcome, the last one said so",
    async () => {
      const rounds = [1, 2, 3, 4, 5].flatMap(() => round(INBOX, [], "refine"));
      const { runs, model, invoke } = authoringTools([
        ...answered(BRIEF),
        ...answered(contract("inbox.yaml")),
        ...rounds,
      ]);
      await invoke("author_flow", { goal: "read notes" });
      const { status } = await drive(runs, "r1", []);
      expect(status).toMatchObject({
        status: "succeeded",
        output: { outcome: "not written: five rounds ran out before the user chose the write" },
      });
      const plans = requestsOf(model, "finish a yian");
      expect(plans).toHaveLength(5);
      expect(plans[3]).toContain("Round 4 of 5; the last: false.");
      expect(plans[4]).toContain("Round 5 of 5; the last: true.");
    },
    TIMEOUT,
  );

  test(
    "a try still waiting when author_flow ends is cancelled",
    async () => {
      const tryArgs = { flow: "asks.yaml", content: ASKS, inputs: {} };
      const { runs, invoke } = authoringTools([
        ...answered(BRIEF),
        ...answered(contract("asks.yaml")),
        textReply(ASKS),
        ...answered({ findings: [] }),
        callReply(["flows_start", tryArgs, "t1"]),
        callReply(["flows_wait", { run: "r2", seconds: 0 }, "t2"]),
        textReply("It still waits for an answer."),
        ...answered({ next: "stop", changes: "" }),
      ]);
      await invoke("author_flow", { goal: "ask for a region" });
      const { status } = await drive(runs, "r1", []);
      expect(status).toMatchObject({ status: "succeeded" });
      await until(() => runs.status("r2").status === "cancelled");
    },
    TIMEOUT,
  );

  test(
    "review_flow starts the review of a file with no start card, granted only the reading tools",
    async () => {
      writeFileSync(join(root, "greet.yaml"), GREET);
      const finding = {
        severity: "medium",
        area: "security",
        node: "hi",
        problem: "Code runs.",
        fix: "Check it.",
      };
      const { runs, owner, requests, invoke, model } = authoringTools(answered({ findings: [finding] }));
      const starts = recordStarts(owner);
      const started = seenAs("review_flow", await invoke("review_flow", { path: "greet.yaml" }));
      expect(started).toMatchObject({ run: "r1" });
      expect(Object.keys(starts[0]!).sort(compareCodeUnits)).toEqual(["read_docs", "read_flow"]);
      const { status } = await drive(runs, "r1", []);
      expect(status).toMatchObject({ status: "succeeded", output: [finding] });
      expect(requests).toEqual([]);
      expect(model.requests[0]!.messages.at(-1)!.text()).toContain('return "hi " + inputs["who"];');
      const outside = await rejection(
        Promise.resolve(invoke("review_flow", { path: "../x.yaml" })),
      );
      expect(outside).toBeInstanceOf(ToolRefused);
    },
    TIMEOUT,
  );

  test(
    "only the author run gets the tools, its start sharing the chat's approvals; its runs get none and stay the assistant's",
    async () => {
      const { runs, owner, invoke, requests } = authoringTools(
        [callReply(["ask_user", { question: "What for?" }, "b1"])],
        "Approve this flow for this chat",
      );
      const names = Object.keys(chatRunTools(host(), runs, owner));
      expect(names).not.toContain("author_flow");
      expect(names).not.toContain("review_flow");
      // A flow the user wrote can't call the tools the authoring runs are granted.
      const GREEDY =
        "id: g\nname: g\nnodes:\n  w: {kind: call, call: write_flow, input: {path: a.yaml, " +
        "content: x}}\noutput: ${w.output}\n";
      const greedy = await loadConfinedText(root, join(root, "greedy.yaml"), GREEDY, NO_ENV);
      await expect(owner.start(greedy, {})).rejects.toThrow("write_flow");
      const starts = recordStarts(owner);
      await invoke("author_flow", { goal: "greet" });
      const granted = visible(starts[0]! as Record<string, FunctionTool | FunctionTool[]>);
      // As the author flow grants them, its own group's start and wait among them.
      const shown = ([group, names]: [string, readonly string[]]) => names.map((name) => `${group}_${name}`);
      const written = AUTHOR_TOOLS.flatMap((grant) => (typeof grant === "string" ? [grant] : Object.entries(grant).flatMap(shown)));
      expect([...granted.keys()].sort(compareCodeUnits)).toEqual(written.sort(compareCodeUnits));
      const ctx = new ToolContext({
        callId: "chat-1-r1/tried",
        attempt: 1,
        toolName: "flows_start",
        runId: "chat-1-r1",
        node: "tried",
        context: undefined,
        signal: new AbortController().signal,
      });
      const call = (name: string, args: Record<string, unknown>) => granted.get(name)!.execute(args, ctx);
      // Its flows_wait reads only the runs it started: the chat's own read as unknown.
      const other = await rejection(Promise.resolve(call("flows_wait", { run: "r1", seconds: 0 })));
      expect(other.message).toBe("no run 'r1'");
      // A run it starts gets only the registered tools.
      const args = { flow: "g.yaml" };
      const refused = await call("flows_start", { ...args, content: GREEDY, inputs: {} });
      expect(String(refused)).toStartWith("not started:");
      expect(String(refused)).toContain("write_flow");
      const tried = seenAs("flows_start", await call("flows_start", { ...args, content: GREET, inputs: { who: "ada" } }));
      expect(tried).toMatchObject({ run: "r2" });
      // The author run's tries are its own children.
      expect(inspectRun(String(tried["log"])).runId).toBe("chat-1-r1-r2");
      expect(requests.at(-1)!.title).toBe("Run g.yaml (a draft, for author_flow)");
      // Approved for this chat by the user: the assistant starts the same draft unasked.
      const asked = requests.length;
      const again = seenAs("flows_start", await invoke("flows_start", { ...args, content: GREET, inputs: { who: "bo" } }));
      expect(again).toMatchObject({ run: "r3" });
      expect(requests).toHaveLength(asked);
      let status = seenAs("flows_wait", await call("flows_wait", { run: "r2", seconds: 60 }));
      while (status["status"] === "running") {
        status = seenAs("flows_wait", await call("flows_wait", { run: "r2", seconds: 60 }));
      }
      expect(status).toMatchObject({ status: "succeeded", output: "hi ada" });
    },
    TIMEOUT,
  );
});

describe("ChatRuns", () => {
  const TIMEOUT = 10_000;
  const CHAT = "chat-1";
  let runs: ChatRuns | null = null;

  afterEach(async () => {
    await runs?.dispose();
    runs = null;
  });

  /** The chat's runs over the temporary root, its logs under `<root>/runs`, on a held clock. */
  function childRuns(options: Partial<ChatRunsOptions> = {}): ChatRuns {
    runs = new ChatRuns({
      root,
      logDir: join(root, "runs"),
      home: join(root, "home"),
      env: NO_ENV,
      llmConfig: {},
      scheduler: heldScheduler().scheduler,
      skillsFor: () => null,
      ...options,
    });
    return runs;
  }

  /** The chat run's hold on `children`. */
  const chat = (children: ChatRuns) => children.owner({ runId: CHAT, group: "flows", label: "yian-chat" });

  /** Load `text` as the flow `<root>/<name>`. */
  function flow(text: string, name = "child.yaml"): Promise<ComposedFlow> {
    return loadConfinedText(root, join(root, name), text, NO_ENV);
  }

  /** Child `id`'s first status numbered after `after` (default: any) that isn't `running`. */
  async function settled(children: ChildFlows, id: string, after = 0): Promise<ChildStatus> {
    let status = children.status(id);
    while (status.seq <= after || status.status === "running") status = await children.next(id, status.seq);
    return status;
  }

  const AGENT = "id: f\nname: f\nnodes:\n  a: {kind: agent, prompt: hi}\noutput: ${a.output}\n";

  test(
    "a child whose provider has no key isn't started and leaves no log",
    async () => {
      const children = childRuns({ llmConfig: { provider: "openai", model: "gpt-5.5" } });
      await expect(chat(children).start(await flow(AGENT), {})).rejects.toThrow(MissingKeyError);
      await expect(chat(children).start(await flow(AGENT), {})).rejects.toThrow("$OPENAI_API_KEY is not set");
      expect(existsSync(join(root, "runs"))).toBe(false);
    },
    TIMEOUT,
  );

  /** Log a line from CODE, then say "ok". */
  const CHATTY = `id: chatty
name: chatty
nodes:
  talk:
    kind: code
    output: !str
    code: |
      console.log("x");
      console.error("y");
      return "ok";
output: \${talk.output}
`;

  // It replaces the process's console methods, so it relies on a file's tests running one at a time.
  test(
    "a child's CODE console output goes to the console file, never through the process's console",
    async () => {
      const printed: string[] = [];
      const original = { log: console.log, error: console.error };
      const record = (...args: unknown[]) => void printed.push(args.join(" "));
      console.log = record;
      console.error = record;
      try {
        const consoleFile = join(root, "console.log");
        const children = childRuns({ consoleFile });
        await chat(children).start(await flow(CHATTY), {});
        expect(await settled(children, "r1")).toMatchObject({ status: "succeeded", output: "ok" });
        expect(readFileSync(consoleFile, "utf8")).toBe("x\ny\n");
        expect(console.log).toBe(record);
        expect(console.error).toBe(record);
        expect(printed).toEqual([]);
      } finally {
        console.log = original.log;
        console.error = original.error;
      }
    },
    TIMEOUT,
  );

  /** A flow that appends to its memory's index. */
  const REMEMBER = `id: answer
name: answer
input:
  lessons: !FileMemory
nodes:
  add:
    kind: call
    call: {lessons: append}
    input: {path: MEMORY.md, text: noted}
output: \${add.output.path}
`;

  /** Takes a line from its inbox, then appends to its memory's index. */
  const REMEMBER_LATER = `id: answer
name: answer
input:
  lessons: !FileMemory
  messages: !Inbox(str)
nodes:
  heard:
    kind: call
    call: take
    input:
      from: \${input.messages}
  add:
    kind: call
    call: {lessons: append}
    depends_on: [heard]
    input: {path: MEMORY.md, text: heard}
output: \${add.output.path}
`;

  describe("memory", () => {
    test(
      "a child's memory input gets its default, as yian run binds it, and the chat notes it created it",
      async () => {
        const children = childRuns();
        const remember = await flow(REMEMBER);
        await chat(children).start(remember, {}, { project: root });
        const first = await settled(children, "r1");
        expect(first.status).toBe("succeeded");
        // The flow's folder is the chat's root, so the identity is the one `yian run` gives there.
        const identity = defaultIdentity(root, "answer", "lessons");
        const location = defaultLocation(join(root, "home"), identity, "FileMemory");
        // The identity only: the chat's statuses reach its log and its model, which never hold a location.
        expect(first.memory_notes).toEqual([`memory lessons: ${identity} (new)`]);
        expect(childEvent(first)!.text).toContain(`; noted ${JSON.stringify(first.memory_notes![0])}`);
        expect(JSON.stringify(first)).not.toContain(join(root, "home"));
        // Read once, they aren't said again.
        children.markRead("r1", first.seq);
        await chat(children).start(remember, {}, { project: root });
        const second = await settled(children, "r2");
        expect(second.status).toBe("succeeded");
        expect(second.memory_notes).toBeUndefined();
        expect(readFileSync(join(location, "MEMORY.md"), "utf8")).toBe("noted\nnoted");
      },
      TIMEOUT,
    );

    test(
      "a waiting child moves on with its memory bound again, and a drift its advance finds is noted",
      async () => {
        const passed: unknown[] = [];
        const children = childRuns({
          advance: async (options) => {
            passed.push(options.memory);
            const result = await advance(options);
            const drift = { input: "lessons", identity: "i", recorded: "v1", current: "v2" };
            return new RunResult({ ...result, memoryDrift: [drift] });
          },
        });
        await chat(children).start(await flow(REMEMBER_LATER), {}, { project: root });
        const waiting = await settled(children, "r1");
        expect(waiting.status).toBe("waiting");
        children.markRead("r1", waiting.seq);
        await children.message("r1", "messages", "hi");
        const done = await settled(children, "r1", waiting.seq);
        expect(done.status).toBe("succeeded");
        expect(passed).toHaveLength(1);
        expect(Object.keys(passed[0] as object)).toEqual(["lessons"]);
        expect(done.memory_notes).toEqual([
          "memory `lessons` changed since this run started; the run keeps its recorded recall",
        ]);
        const location = defaultLocation(join(root, "home"), defaultIdentity(root, "answer", "lessons"), "FileMemory");
        expect(readFileSync(join(location, "MEMORY.md"), "utf8")).toBe("heard");
      },
      TIMEOUT,
    );

    test(
      "a refusal to bind or open a child's memory names no location",
      async () => {
        const home = join(root, "home");
        const memories = join(home, ".yian", "memory");
        const remember = await flow(REMEMBER);
        const location = defaultLocation(home, defaultIdentity(root, "answer", "lessons"), "FileMemory");
        // A file where the memory's folder goes: its open refuses it.
        mkdirSync(dirname(location), { recursive: true });
        writeFileSync(location, "");
        const opened = await chat(childRuns())
          .start(remember, {}, { project: root })
          .then(() => null, (error: Error) => error.message);
        expect(opened).toContain("memory `lessons`");
        expect(opened).not.toContain(root);
        // A symlinked project folder: the bind refuses it.
        rmSync(join(memories, projectKey(root)), { recursive: true });
        mkdirSync(join(root, "elsewhere"));
        symlinkSync(join(root, "elsewhere"), join(memories, projectKey(root)));
        const bound = await chat(childRuns())
          .start(remember, {}, { project: root })
          .then(() => null, (error: Error) => error.message);
        expect(bound).toContain(`~/.yian/memory/${projectKey(root)} is a symlink`);
        expect(bound).not.toContain(root);
      },
      TIMEOUT,
    );

    test.skipIf(process.getuid?.() === 0)(
      "an OS error under a ~/.yian/memory linked out of the home folder names no location",
      async () => {
        const home = join(root, "home");
        const elsewhere = join(root, "elsewhere", "mem");
        mkdirSync(elsewhere, { recursive: true });
        mkdirSync(join(home, ".yian"), { recursive: true });
        symlinkSync(elsewhere, join(home, ".yian", "memory"));
        // The project's folder can't take the flow's: the open's mkdir fails there.
        const key = join(elsewhere, projectKey(root));
        mkdirSync(key, { mode: 0o500 });
        try {
          const refused = await chat(childRuns())
            .start(await flow(REMEMBER), {}, { project: root })
            .then(() => null, (error: Error) => error.message);
          expect(refused).toContain("EACCES");
          // Bun's recursive mkdir names the folder it failed on, Node's the memory's own folder: each
          // is written from `~` or by its role, and neither names where it is.
          const scrubbed = [`~/.yian/memory/${projectKey(root)}/answer`, "the memory's folder"];
          expect(scrubbed.some((path) => refused!.endsWith(`mkdir '${path}'`)), refused!).toBe(true);
          expect(refused).not.toContain(root);
          expect(refused).not.toContain(tmpdir());
        } finally {
          chmodSync(key, 0o700);
        }
      },
      TIMEOUT,
    );

    test.skipIf(process.getuid?.() === 0)(
      "an OS error making a child's store folder names it from ~/.yian/memory, not as the memory's folder",
      async () => {
        const home = join(root, "home");
        mkdirSync(join(home, ".yian", "memory"), { recursive: true });
        // The project's folder can't take the flow's: the open's mkdir fails there.
        const key = join(home, ".yian", "memory", projectKey(root));
        mkdirSync(key, { mode: 0o500 });
        const cases =
          "id: cases\nname: cases\ninput:\n  cases: !VectorDBMemory\nnodes:\n  add:\n    kind: call\n" +
          "    call: {cases: add}\n    input: {text: x}\noutput: ${add.output.id}\n";
        const env = (name: string) => (name === "YIAN_DEFAULT_EMBEDDING" ? "openai/m" : NO_ENV(name));
        const embed = async (texts: readonly string[]) => texts.map(() => Float32Array.of(1, 0));
        try {
          const refused = await chat(childRuns({ env, embed }))
            .start(await flow(cases), {}, { project: root })
            .then(() => null, (error: Error) => error.message);
          expect(refused).toContain("EACCES");
          expect(refused).toContain(`~/.yian/memory/${projectKey(root)}/cases`);
          expect(refused).not.toContain("the memory's folder");
          expect(refused).not.toContain(root);
        } finally {
          chmodSync(key, 0o700);
        }
      },
      TIMEOUT,
    );

    test(
      "a waiting child whose memory folder is gone fails without pointing at --memory, which the chat lacks",
      async () => {
        const children = childRuns();
        await chat(children).start(await flow(REMEMBER_LATER), {}, { project: root });
        const waiting = await settled(children, "r1");
        expect(waiting.status).toBe("waiting");
        const identity = defaultIdentity(root, "answer", "lessons");
        rmSync(defaultLocation(join(root, "home"), identity, "FileMemory"), { recursive: true });
        await children.message("r1", "messages", "hi");
        const failed = await settled(children, "r1", waiting.seq);
        expect(failed).toMatchObject({ status: "failed" });
        expect(failed.note).toBe(
          "couldn't move the run: memory `lessons`: its folder is gone, so the run can't continue",
        );
      },
      TIMEOUT,
    );

    test(
      "a waiting child whose memory folder lost its .yian fails saying it isn't a memory any more",
      async () => {
        const children = childRuns();
        await chat(children).start(await flow(REMEMBER_LATER), {}, { project: root });
        const waiting = await settled(children, "r1");
        const location = defaultLocation(join(root, "home"), defaultIdentity(root, "answer", "lessons"), "FileMemory");
        rmSync(join(location, ".yian"), { recursive: true });
        await children.message("r1", "messages", "hi");
        const failed = await settled(children, "r1", waiting.seq);
        expect(failed.note).toBe(
          "couldn't move the run: memory `lessons`: it isn't a memory any more, so the run can't continue",
        );
      },
      TIMEOUT,
    );

    test(
      "a child given no project folder has no default memories",
      async () => {
        const children = childRuns();
        await expect(chat(children).start(await flow(REMEMBER), {})).rejects.toThrow(
          "a flow with no file has no default memories, so `lessons` can't be bound",
        );
      },
      TIMEOUT,
    );

    test(
      "a child whose flow has no plain id isn't started",
      async () => {
        const children = childRuns();
        const unnamed = await flow(REMEMBER.replace("id: answer", 'id: "Answer questions"'));
        // The chat has no --memory to suggest.
        const refused = await chat(children).start(unnamed, {}, { project: root }).then(
          () => null,
          (error: Error) => error.message,
        );
        expect(refused).toBe(
          "give the flow a plain `id:` of at most 100 characters (letters, digits, `.`, `_`, `-`) " +
            "so `lessons` gets a default memory",
        );
        expect(existsSync(join(root, "runs"))).toBe(false);
      },
      TIMEOUT,
    );
  });
});
