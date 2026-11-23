/**
 * A child run's view (`run_view.ts`): its progress read from a real run log as it grows, and its
 * screen, in colour and without.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { type ChildInterrupt, ChildFlows, type ChildStatus, type Pressed } from "../../src/child_flows.ts";
import { START_ID } from "../../src/compile/model.ts";
import { RunProgress, RunView, type RunViewHost } from "../../src/cli/run_view.ts";
import type { Key } from "../../src/cli/terminal.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { ToolResultPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { virtualScheduler } from "../../src/platform.ts";
import { cancel, close, interrupt, type PendingCall, pendingCalls, runFlow, send, withdraw } from "../../src/run.ts";
import { NodeStepRecord, PoolCommitRecord, readRunLog } from "../../src/runtime/log.ts";
import { pending } from "../../src/tools/context.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import type { BeforeToolCall } from "../../src/tools/gate.ts";
import { TypedValue } from "../../src/typesys/values.ts";
import { callReply, NO_ENV, serving, textReply } from "../helpers/run_state_fixtures.ts";
import { heldScheduler } from "../helpers/schedulers.ts";

let tmp: string;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "yian-run-view-"));
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

const COLOR = { isTTY: true };
const PLAIN = { isTTY: false };
/** Drops the colours, for rows compared as the screen shows them. */
const shown = (rows: readonly string[]) => rows.map((row) => row.replace(/\x1b\[[0-9;]*m/g, ""));

/** Pick a colour with an agent that looks up the options, then ask the user to confirm it. */
const PICK = `id: pick
name: pick
nodes:
  choose:
    kind: agent
    prompt: Pick a colour.
    tools: [options]
    output: !str
  route:
    kind: case
    on: \${choose.output}
    cases:
      - when: blue
        then: sure
    else: other
  other: !str not blue
  sure:
    kind: call
    call: ask
    input:
      prompt: "Is \${choose.output} right?"
    output: !str
output: \${sure.output}
`;

/** A CODE node that fails with a long message holding a tab. */
const BOOM = `id: boom
name: boom
nodes:
  boom:
    kind: code
    output: !str
    code: |
      throw new Error("a\\tb " + "word ".repeat(12));
output: \${boom.output}
`;

/** Run PICK to its question on a virtual clock, the agent scripted; return its log. */
async function pickToQuestion(): Promise<string> {
  const log = join(tmp, "log.jsonl");
  const flow = await loadFlow(PICK, { env: NO_ENV });
  const model = new ScriptedModel(
    [callReply(["options", {}, "c1"]), textReply("blue")],
    { provider: "xai", model: "grok-4.3" },
  );
  const options = new FunctionTool({
    name: "options",
    description: "the colours",
    parametersJsonSchema: { type: "object", properties: {} },
    execute: () => "red\nblue",
  });
  const result = await runFlow(flow, {}, {
    log,
    tools: { options, ask: () => pending() },
    models: serving(model),
    llmConfig: { provider: "xai", model: "grok-4.3" },
    env: (name) => (name === "XAI_API_KEY" ? "test-key" : NO_ENV(name)),
    scheduler: virtualScheduler(Date.parse("2026-01-01T00:00:00Z")),
  });
  expect(result.status).toBe("waiting");
  return log;
}

describe("a run's progress", () => {
  test(
    "draws each node that ended, an agent's tool calls and text, and a node still waiting",
    async () => {
      const log = await pickToQuestion();
      const progress = new RunProgress(log, () => 60, COLOR, NO_ENV);
      const { lines: rows, earlier } = progress.rows(Date.parse("2026-01-01T00:01:05Z"));
      expect(earlier).toBe(0);
      expect(shown(rows)).toEqual([
        "⏺ options()",
        "  ⎿  red · blue",
        "⏺ blue",
        "✓ choose [0s]",
        "✓ route → sure [0s]",
        "◷ sure [1m05s] waiting",
      ]);
      expect(rows[3]).toContain("\x1b[32m✓ choose");
      // Read again, it adds only what the log gained: nothing here.
      expect(progress.rows(Date.parse("2026-01-01T00:01:05Z")).lines).toEqual(rows);
    },
    20_000,
  );

  test(
    "reads only what the log gained, not a line still being written, and all again once it is rewritten",
    async () => {
      const full = readFileSync(await pickToQuestion(), "utf8");
      const lines = full.split("\n").filter(Boolean);
      const log = join(tmp, "grow.jsonl");
      const at = lines.findIndex((line) => line.includes('"node_succeeded"') && line.includes('"choose"'));
      // Up to the agent's success, then half of the next line.
      writeFileSync(log, `${lines.slice(0, at).join("\n")}\n${lines[at]!.slice(0, 20)}`);
      const progress = new RunProgress(log, () => 60, PLAIN, NO_ENV);
      const now = Date.parse("2026-01-01T00:00:00Z");
      expect(progress.rows(now).lines).toEqual(["⏺ options()", "  ⎿  red · blue", "⏺ blue", "· choose [0s]"]);
      writeFileSync(log, full);
      expect(progress.rows(now).lines).toEqual([
        "⏺ options()",
        "  ⎿  red · blue",
        "⏺ blue",
        "✓ choose [0s]",
        "✓ route → sure [0s]",
        "◷ sure [0s] waiting",
      ]);
      // Shorter than what was read: read again from the start, at once.
      writeFileSync(log, `${lines.slice(0, at).join("\n")}\n`);
      expect(progress.rows(now).lines).toEqual(["⏺ options()", "  ⎿  red · blue", "⏺ blue", "· choose [0s]"]);
    },
    20_000,
  );

  test(
    "wraps a long failure under its indent at the screen's width, its tabs expanded",
    async () => {
      const log = join(tmp, "fail.jsonl");
      const flow = await loadFlow(BOOM, { env: NO_ENV });
      const result = await runFlow(flow, {}, { log, env: NO_ENV });
      expect(result.status).toBe("failed");
      const { lines } = new RunProgress(log, () => 30, PLAIN, NO_ENV).rows(0);
      expect(lines[0]).toMatch(/^✗ boom \[\d+s\]$/);
      expect(lines.slice(1).every((line) => line.startsWith("    ") && line.length <= 30)).toBe(true);
      expect(lines.join(" ")).toContain("a    b");
      expect(lines.join("")).not.toContain("\t");
    },
    20_000,
  );

  test("draws each host call on an inbox or the run, with who made it", async () => {
    const log = join(tmp, "notes.jsonl");
    const flow = await loadFlow(
      "id: notes\nname: notes\ninput:\n  m: !Inbox(str)\n  said: !Inbox(Message)\nnodes:\n" +
        "  heard: {kind: call, call: take, input: {from: \"${input.m}\", take: all}}\n" +
        "  told: {kind: call, call: take, input: {from: \"${input.said}\", take: all}}\n" +
        "output: ${[heard.output.events, told.output.events]}\n",
      { env: NO_ENV },
    );
    expect((await runFlow(flow, {}, { log, env: NO_ENV })).status).toBe("waiting");
    const env = NO_ENV;
    await send(log, "m", "hi", { env, by: { kind: "user", name: "Ada\x1b[2J" } });
    await send(log, "m", "hey", { env });
    await withdraw(log, "m", { env, by: { kind: "user", name: "Ada\x1b[2J" } });
    const message = { role: "user", author: "the user, via yian-chat", parts: [{ type: "text", text: "look" }] };
    await send(log, "said", message, { env, by: { kind: "agent", name: "yian-chat" } });
    const action = { role: "user", author: "the user", parts: [{ type: "action", tool: "flows_message" }] };
    await send(log, "said", action, { env, by: { kind: "user", name: "the user" } });
    await interrupt(log, "m", { env, by: { kind: "user", name: "the user", run: "3a9151970cde41b29fe8899b7b857563" } });
    await close(log, "m", { env, by: { kind: "run", name: "review r1", run: "chat-r1" } });
    await cancel(log, { env, by: { kind: "host", name: "daemon" } });
    const progress = new RunProgress(log, () => 60, COLOR, NO_ENV);
    const { lines } = progress.rows(0);
    expect(shown(lines)).toEqual([
      '› m  "hi" · user Ada',
      '› m  "hey"',
      "› m  took back 1 · user Ada",
      '› said  the user, via yian-chat: "look"',
      "› said  the user: [action]",
      // A long run id is cut to its first 8 characters.
      "› m  interrupted · user the user (run 3a915197)",
      "› m  closed · run review r1 (run chat-r1)",
      "› cancelled · host daemon",
      "◷ heard [0s] waiting",
      "◷ told [0s] waiting",
    ]);
    expect(lines[0]).toContain("\x1b[2m›");
  });

  test("of a log not written yet is empty", () => {
    const progress = new RunProgress(join(tmp, "none.jsonl"), () => 60, PLAIN, NO_ENV);
    expect(progress.rows(0)).toEqual({ lines: [], earlier: 0 });
  });
});

/** A key as the terminal hands it over: typed text, or a named key. */
const typed = (str: string): Key => ({ name: undefined, ctrl: false, meta: false, str, paste: true });
const named = (name: string, shift = false): Key => ({ name, ctrl: false, meta: false, shift, str: undefined, paste: false });
/** A key typing one letter. */
const letter = (str: string): Key => ({ name: str, ctrl: false, meta: false, str, paste: false });
const ENTER = named("return");
const ESC = named("escape");
const TAB = named("tab");

/** Let a press's outcome land. */
const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

/**
 * A view of run r1 (`pick.yaml`) with `status` (a function of it, read at each draw), the `inboxes`
 * given and a fixed progress; each press is kept in `presses` and answered by `answer`, or as sent;
 * its open calls are `calls`.
 */
function viewOf(
  status: ChildStatus | (() => ChildStatus),
  inboxes: readonly string[] = [],
  answer: ((tool: string) => Pressed<ChildStatus | ChildInterrupt>) | null = null,
  out = COLOR,
  calls: readonly PendingCall[] = [],
) {
  const now = typeof status === "function" ? status : () => status;
  const reply = answer ?? (() => ({ result: now(), recorded: true, reason: null }));
  const presses: [string, unknown][] = [];
  const host: RunViewHost = {
    id: "r1",
    name: "pick.yaml",
    inboxes,
    status: now,
    progress: () => ({ lines: ["✓ choose [0s]"], earlier: 0 }),
    pendingCall: (call) => calls.find((each) => each.callId === call) ?? null,
    press: async (tool, args) => {
      presses.push([tool, args]);
      return reply(tool) as never;
    },
  };
  const view = new RunView(host, out, NO_ENV);
  return { view, presses, rows: (rows = 24, waiting = false) => shown(view.draw(60, rows, waiting)) };
}

/** A box row 60 columns wide holding `text`. */
const boxRow = (text: string) => `│ ${text.padEnd(56)} │`;
const BOTTOM = `╰${"─".repeat(58)}╯`;
/** A question card's rule, 60 columns wide, after `title`. */
const rule = (title: string) => `${title} ${"─".repeat(59 - title.length)}`;
/** The box's top edge, 60 columns wide, naming `title`. */
const top = (title: string) => `╭─ ${title} ${"─".repeat(55 - title.length)}╮`;

describe("a run's view", () => {
  const status: ChildStatus = { run: "r1", status: "waiting", seq: 3, log: "/runs/chat-r1/log.jsonl" };
  const asking: ChildStatus = { ...status, asks: [{ call: "chat-r1/sure", question: "Is blue right?" }] };
  const running: ChildStatus = { run: "r1", status: "running", seq: 4, log: "l" };

  test("has the run, its flow, status and log, its progress, a strip with the question it waits on, and a box", () => {
    const { view, rows } = viewOf(asking);
    expect(rows()).toEqual([
      "r1  pick.yaml  ● asks",
      "log  /runs/chat-r1/log.jsonl",
      "",
      "✓ choose [0s]",
      "",
      "questions  [Is blue right?]",
      top("Is blue right?"),
      boxRow(">"),
      BOTTOM,
      "  enter answers · /stop",
      "esc to go back to the chat",
    ]);
    expect(view.draw(60, 24, false)[0]).toContain("\x1b[33m● asks");
    expect(view.cursor()).toEqual({ row: 7, before: expect.stringContaining(">") });
    // A run with nothing to send to keeps its box for its commands.
    expect(viewOf(status).rows().slice(-5)).toEqual([`╭${"─".repeat(58)}╮`, boxRow(">"), BOTTOM, "  /stop ends the run", "esc to go back to the chat"]);
  });

  test("says in yellow when the chat waits on the user behind it", () => {
    const { view } = viewOf(status);
    const drawn = view.draw(60, 24, true);
    expect(shown(drawn).at(-1)).toBe("the chat waits on you · ctrl-c to answer");
    expect(drawn.at(-1)).toContain("\x1b[33m");
  });

  test("keeps the last rows of a long progress, under a count of those left out", () => {
    const rest: ChildStatus = { run: "r1", status: "waiting", seq: 9, log: "l" };
    const lines = Array.from({ length: 30 }, (_, i) => `✓ n${i}`);
    const view = (earlier: number, progress: string[]) =>
      new RunView(
        { id: "r1", name: "x.yaml", inboxes: [], status: () => rest, progress: () => ({ lines: progress, earlier }), pendingCall: () => null, press: async () => never() },
        PLAIN,
        NO_ENV,
      ).draw(60, 14, false);
    const rows = view(0, lines);
    expect(rows.slice(0, 8)).toEqual(["r1  x.yaml  ● waits", "log  l", "", "… 26 earlier rows", "✓ n26", "✓ n27", "✓ n28", "✓ n29"]);
    expect(rows.length).toBe(14);
    // The rows the progress no longer keeps count too.
    expect(view(5, ["✓ n"]).slice(3, 5)).toEqual(["… 5 earlier rows", "✓ n"]);
  });

  test("on a short screen keeps the box, the strip and the foot, cutting the progress from the top", () => {
    const asks = Array.from({ length: 6 }, (_, i) => ({ call: `c${i}`, question: `Q${i}` }));
    const { view, rows } = viewOf({ ...status, asks }, ["notes", "extra"]);
    expect(rows(10)).toEqual([
      "r1  pick.yaml  ● asks",
      "log  /runs/chat-r1/log.jsonl",
      "",
      "inboxes  [notes]  extra",
      "questions  Q0  Q1  Q2  Q3  Q4  Q5",
      top("notes"),
      boxRow(">"),
      BOTTOM,
      "  enter sends · tab picks · /close · /stop",
      "esc to go back to the chat",
    ]);
    expect(view.cursor()?.row).toBe(6);
    // Smaller still, the header goes too, from its bottom.
    expect(rows(8)).toEqual(["r1  pick.yaml  ● asks", ...rows(10).slice(-7)]);
  });

  test("once the run has ended, says so in place of the box, and Esc goes back", () => {
    const ended: ChildStatus = { run: "r1", status: "succeeded", seq: 9, log: "l", output: "ok" };
    const { view, rows, presses } = viewOf(ended, ["notes"]);
    expect(rows().slice(-2)).toEqual(["", "the run has ended · esc to go back"]);
    expect(view.cursor()).toBeNull();
    view.key(typed("hi"));
    view.key(ENTER);
    expect(presses).toEqual([]);
    expect(view.key(ESC)).toBe(true);
  });

  test("names its inboxes in a strip; Tab and Shift-Tab pick the one the box sends to", () => {
    const { view, rows } = viewOf(status, ["notes", "extra"]);
    const strip = (drawn: string[]) => [drawn.at(-6), drawn.at(-5)];
    expect(strip(rows())).toEqual(["inboxes  [notes]  extra", `╭─ notes ${"─".repeat(50)}╮`]);
    view.key(TAB);
    expect(strip(rows())).toEqual(["inboxes  notes  [extra]", `╭─ extra ${"─".repeat(50)}╮`]);
    view.key(TAB);
    expect(strip(rows())[0]).toBe("inboxes  [notes]  extra");
    view.key(named("tab", true));
    expect(strip(rows())[0]).toBe("inboxes  notes  [extra]");
    expect(rows().at(-2)).toBe("  enter sends · tab picks · /close · /stop");
  });

  test("Enter sends the box's text, as it is, to the inbox picked, and says it was sent", async () => {
    const { view, presses, rows } = viewOf(status, ["notes", "extra"]);
    view.key(TAB);
    view.key(typed("  look: *this*  "));
    expect(view.key(ENTER)).toBe(false);
    expect(presses).toEqual([["message", { run: "r1", inbox: "extra", text: "  look: *this*  " }]]);
    expect(rows().at(-4)).toBe(boxRow(">"));
    await settle();
    expect(rows().at(-6)).toBe("sent to extra");
    // A blank line sends nothing.
    view.key(typed("   "));
    view.key(ENTER);
    expect(presses.length).toBe(1);
  });

  test("a refused press shows its reason, and one the chat didn't get says so", async () => {
    const refused = { ...status, note: "inbox 'notes' is closed" };
    const answers: Pressed<ChildStatus>[] = [
      { result: refused, recorded: true, reason: null },
      { result: null, recorded: false, reason: "ended" },
      { result: status, recorded: false, reason: "busy" },
    ];
    const { view, rows } = viewOf(status, ["notes"], () => answers.shift()!);
    const said: string[] = [];
    for (const text of ["one", "two", "/close"]) {
      view.key(typed(text));
      view.key(ENTER);
      if (text === "/close") view.key({ ...named("y"), str: "y", paste: false });
      await settle();
      // What the press did, on the rows between the strip and the box.
      const drawn = rows();
      said.push(drawn.slice(drawn.indexOf("inboxes  [notes]") + 1, -5).join(" "));
    }
    expect(said).toEqual([
      "inbox 'notes' is closed",
      "not sent: the chat has ended",
      "closed notes; the assistant isn't told: the chat's log stayed busy",
    ]);
  });

  test("/stop asks first: y stops the run, any other key keeps it", async () => {
    const { view, presses, rows } = viewOf(status, ["notes"]);
    view.key(typed("/stop"));
    view.key(ENTER);
    expect(rows().at(-6)).toBe("stop r1? its run ends (y/N)");
    view.key(typed("n"));
    expect(presses).toEqual([]);
    expect(rows().at(-4)).toBe(boxRow(">"));
    view.key(typed("/stop"));
    view.key(ENTER);
    view.key({ ...named("y"), str: "y", paste: false });
    expect(presses).toEqual([["stop", { run: "r1" }]]);
    await settle();
    expect(rows().at(-6)).toBe("stopped r1");
  });

  test("/close asks first, from the menu too: y closes the inbox picked, any other key keeps it", async () => {
    const { view, presses, rows } = viewOf(status, ["notes"]);
    // Enter on the menu answers its first command.
    view.key({ ...named("/"), str: "/", paste: false });
    view.key(ENTER);
    expect(rows().at(-6)).toBe("close notes? nothing more can be sent to it (y/N)");
    view.key(ENTER);
    expect(presses).toEqual([]);
    view.key(typed("/close"));
    view.key(ENTER);
    view.key({ ...named("y"), str: "y", paste: false });
    expect(presses).toEqual([["close_inbox", { run: "r1", inbox: "notes" }]]);
    await settle();
    expect(rows().at(-6)).toBe("closed notes");
  });

  test("Esc while the run is drawn running interrupts the inbox picked, and the lines taken back return to the box", async () => {
    const cut = (nodes: string[], taken: string[]): ChildInterrupt => ({
      seq: 9,
      nodes,
      calls: [],
      released: [],
      status: running,
      landed: true,
      queued: 0,
      taken: taken.map((text, i) => ({ seq: i, by: null, value: { role: "user", parts: [{ type: "text", text }] } })),
    });
    const answers = [cut(["turn/reply"], ["first", "second"]), cut([], []), { ...cut([], ["third"]), seq: null }];
    const { view, presses, rows } = viewOf(running, ["notes"], () => ({ result: answers.shift()!, recorded: true, reason: null }));
    expect(rows().at(-1)).toBe("esc interrupts notes · ctrl-c goes back to the chat");
    view.key(typed("draft"));
    expect(view.key(ESC)).toBe(false);
    expect(presses).toEqual([["interrupt", { run: "r1", inbox: "notes" }]]);
    await settle();
    expect(rows().at(-6)).toBe("interrupted turn/reply; 2 messages back in the box");
    expect(rows().at(-4)).toBe(boxRow("> first second draft"));
    view.key(ESC);
    await settle();
    expect(rows().at(-6)).toBe("nothing to stop now; the next turn on notes stops at once");
    // With the run waiting on its take of the inbox, nothing stands.
    view.key(typed("x"));
    view.key(named("backspace"));
    view.key(ESC);
    await settle();
    const said = rows().slice(-7, -5).join(" ");
    expect(said).toBe("nothing to interrupt: the run waits for a message on notes, or is already stopping; 1 message back in the box");
  });

  test("Esc at rest goes back to the chat, pressing nothing, a prompt behind the view or not", () => {
    const rest = viewOf(status, ["notes"]);
    rest.rows();
    expect(rest.view.key(ESC)).toBe(true);
    rest.rows(24, true);
    expect(rest.view.key(ESC)).toBe(true);
    expect(rest.presses).toEqual([]);
  });

  test("Tab and Shift-Tab cycle the inboxes, then the questions, then the held calls; the box's edge and hint follow", () => {
    const held = [{ call: "t1", tool: "shout", arguments: { text: "hi" } }];
    const { view, rows } = viewOf({ ...asking, held }, ["notes"]);
    const foot = () => rows().slice(-8, -1);
    expect(foot()).toEqual([
      "inboxes  [notes]",
      "questions  Is blue right?",
      "held calls  shout",
      top("notes"),
      boxRow(">"),
      BOTTOM,
      "  enter sends · tab picks · /close · /stop",
    ]);
    view.key(TAB);
    expect(foot()).toEqual([
      "inboxes  notes",
      "questions  [Is blue right?]",
      "held calls  shout",
      top("Is blue right?"),
      boxRow(">"),
      BOTTOM,
      "  enter answers · tab picks · /stop",
    ]);
    view.key(TAB);
    const onHeld = ["inboxes  notes", "questions  Is blue right?", "held calls  [shout]", top("reason for shout, if any")];
    expect(foot().slice(0, 4)).toEqual(onHeld);
    expect(foot().at(-1)).toBe("  enter approves or refuses · tab picks · /stop");
    view.key(TAB);
    expect(foot()[0]).toBe("inboxes  [notes]");
    view.key(named("tab", true));
    expect(foot().slice(0, 4)).toEqual(onHeld);
    expect(view.draw(60, 24, false).join("\n")).toContain("\x1b[36;1m[shout]");
  });

  test("Enter on a question answers it with the box's text as it is, and says so; a blank box answers nothing", async () => {
    const { view, presses, rows } = viewOf(asking, ["notes"]);
    view.key(TAB);
    view.key(ENTER);
    expect(presses).toEqual([]);
    view.key(typed("  blue, sure  "));
    view.key(ENTER);
    expect(presses).toEqual([["answer", { run: "r1", call: "chat-r1/sure", answer: "  blue, sure  " }]]);
    expect(rows().at(-4)).toBe(boxRow(">"));
    await settle();
    expect(rows().at(-6)).toBe("answered: Is blue right?");
  });

  test("a question that leaves the strip gives the box to the one now in its place, and is said to be no longer open unless answered here", async () => {
    const asks = [
      { call: "c1", question: "Which colour?" },
      { call: "c2", question: "Which size?" },
    ];
    let now: ChildStatus = { ...status, asks };
    const { view, rows } = viewOf(() => now, ["notes"]);
    view.key(TAB);
    expect(rows().slice(-6, -4)).toEqual(["questions  [Which colour?]  Which size?", top("Which colour?")]);
    // Answered elsewhere: while the run works its status lists no question, and the box goes to the inbox.
    now = { ...running, asks: [] };
    expect(rows().slice(-6, -4)).toEqual(["inboxes  [notes]", top("notes")]);
    now = { ...status, asks: [asks[1]!] };
    expect(rows().slice(-7, -4)).toEqual([
      "questions  [Which size?]",
      'the question "Which colour?" is no longer open',
      top("Which size?"),
    ]);
    // Answered here, it leaves with no word.
    view.key(typed("big"));
    view.key(ENTER);
    await settle();
    now = status;
    expect(rows().slice(-7, -5)).toEqual(["inboxes  [notes]", "answered: Which size?"]);
  });

  test("Esc interrupts a run drawn running even with a prompt behind the view", async () => {
    const cut: ChildInterrupt = { seq: 9, nodes: ["reply"], calls: [], released: [], status: running, landed: true, queued: 0, taken: [] };
    const behind = viewOf(running, ["notes"], () => ({ result: cut, recorded: true, reason: null }));
    expect(behind.rows(24, true).at(-1)).toBe("the chat waits on you · ctrl-c to answer");
    expect(behind.view.key(ESC)).toBe(false);
    expect(behind.presses).toEqual([["interrupt", { run: "r1", inbox: "notes" }]]);
    await settle();
  });
});

/** A press a test never makes. */
function never(): never {
  throw new Error("pressed");
}

describe("a run's view over a child of the assistant's", () => {
  const XAI = { provider: "xai", model: "grok-4.3" };
  const env = (name: string) => (name === "XAI_API_KEY" ? "test-xai-key" : NO_ENV(name));
  /** The person, as the run's log records them: relayed by the chat's run. */
  const USER = { kind: "user", name: "the user", run: "chat-1" } as const;
  /** The chat's own run: what the person does lands in its inbox `notes` as their action. */
  const CHAT = `id: chat\nname: chat\ninput:\n  notes: !Inbox(Message)\nnodes:\n  heard: {kind: call, call: take, input: {from: "\${input.notes}", take: one}}\noutput: \${heard.output.events}\n`;
  let children: ChildFlows | null = null;
  /** The last press the view made. */
  let last: Promise<unknown> = Promise.resolve();

  afterEach(async () => {
    await children?.dispose();
    children = null;
  });

  /** Child r1's status once its pass settles. */
  async function settled(): Promise<ChildStatus> {
    let status = children!.status("r1");
    while (status.status === "running") status = await children!.next("r1", status.seq);
    return status;
  }

  /** Make the key `key` in `view`, which presses something; resolve once the press is done and the run has settled. */
  async function pressing(view: RunView, key: Key): Promise<ChildStatus> {
    const before = last;
    view.key(key);
    // A form's card presses once its answer settles.
    await settle();
    expect(last).not.toBe(before);
    await last;
    await settle();
    return settled();
  }

  /**
   * Start `text` as child r1 of the chat's run `chat-1`, which owns it with its inbox `notes`, the
   * child's agents on `model` with the host `tools`, its calls checked by `hold`; once it settles,
   * open its view, which presses as `the user`. Returns the chat's owner and log, the child's log and
   * the view.
   */
  async function viewing(
    text: string,
    options: { model?: ScriptedModel; tools?: Record<string, FunctionTool>; hold?: BeforeToolCall } = {},
  ) {
    const { model, tools = {}, hold = null } = options;
    children = new ChildFlows({
      codeRoot: tmp,
      logDir: join(tmp, "runs"),
      env,
      llmConfig: XAI,
      models: model === undefined ? null : serving(model, env),
      scheduler: heldScheduler().scheduler,
      skillsFor: () => null,
      preflight: () => {},
      memories: { bind: () => ({}), drift: () => "", scrub: (error) => error },
      beforeToolCall: hold,
    });
    const chat = join(tmp, "chat.jsonl");
    expect((await runFlow(await loadFlow(CHAT, { env: NO_ENV }), {}, { log: chat, runId: "chat-1", env })).status).toBe("waiting");
    const people = { runId: "chat-1", group: "flows", label: "yian-chat", people: ["the user"] };
    const owner = children.owner({ ...people, log: chat, inbox: "notes" });
    await owner.start(await loadFlow(text, { env: NO_ENV }), {}, { tools });
    const { log } = await settled();
    const runs = children;
    const host: RunViewHost = {
      id: "r1",
      name: "child.yaml",
      inboxes: [],
      status: () => runs.status("r1"),
      progress: () => ({ lines: [], earlier: 0 }),
      pendingCall: (call) => pendingCalls(log).find((each) => each.callId === call) ?? null,
      press: (tool, args) => {
        const pressed = owner.press(tool, args, { person: "the user" });
        last = pressed;
        return pressed;
      },
    };
    const view = new RunView(host, PLAIN, NO_ENV);
    return { owner, chat, log, view, rows: (columns = 60) => shown(view.draw(columns, 30, false)) };
  }

  /** The person's actions in the chat's log at `log`, in order, without the run's reports. */
  const actions = (log: string) =>
    readRunLog(log).flatMap(({ record }) => {
      if (!(record instanceof PoolCommitRecord) || record.key !== `${START_ID}.notes`) return [];
      const parts = (TypedValue.fromWire(record.value).toObject() as { parts: { type: string }[] }).parts;
      return parts.filter((part) => part.type === "action") as unknown[];
    });
  /** Who made each host call that wrote a step of the child's log at `log`. */
  const deciders = (log: string) =>
    readRunLog(log).flatMap(({ record }) => (record instanceof NodeStepRecord && record.by !== null ? [record.by] : []));

  /** Three questions one after another: a colour, how many (an int), and anything else. */
  const THREE = `id: three
name: three
nodes:
  colour: {kind: call, call: ask, input: {prompt: "Which colour?"}, output: !str}
  count: {kind: call, call: ask, input: {prompt: "How many \${colour.output} ones?"}, output: !int}
  more: {kind: call, call: ask, input: {prompt: "Anything else after \${count.output}?"}, output: !str}
output: \${more.output}
`;

  test(
    "answers a question as the user, relayed by the assistant, one action in its log; the next takes its place, and a refused answer says why",
    async () => {
      const { owner, chat, log, view, rows } = await viewing(THREE);
      expect(rows().slice(-6, -4)).toEqual(["questions  [Which colour?]", top("Which colour?")]);
      view.key(typed("blue"));
      await pressing(view, ENTER);
      expect(rows().slice(-7, -4)).toEqual(["questions  [How many blue ones?]", "answered: Which colour?", top("How many blue ones?")]);
      expect(actions(chat)).toEqual([
        { type: "action", tool: "flows_answer", arguments: { run: "r1", call: "chat-1-r1/colour", answer: "blue" }, result: "answered" },
      ]);
      expect(deciders(log)).toEqual([USER]);
      // Of the wrong type: the question stays.
      view.key(typed("lots"));
      await pressing(view, ENTER);
      expect(rows().slice(-8, -4)).toEqual([
        "questions  [How many blue ones?]",
        "call 'chat-1-r1/count' refuses the answer:",
        `complete('chat-1-r1/count') output: "lots" does not match d…`,
        top("How many blue ones?"),
      ]);
      // The assistant answers first, while the view still shows the question: it says why not.
      await owner.answer("r1", "chat-1-r1/count", 3);
      view.key(typed("4"));
      await pressing(view, ENTER);
      // Pressed here, it leaves with no other word.
      expect(rows().slice(-7, -4)).toEqual([
        "questions  [Anything else after 3?]",
        "call 'chat-1-r1/count' is already answered",
        top("Anything else after 3?"),
      ]);
      expect(deciders(log)).toEqual([USER, { kind: "agent", name: "yian-chat", run: "chat-1" }]);
    },
    20_000,
  );
  /** A form: a city, typed, and how many nights, picked. */
  const TRIP = `id: trip
name: trip
nodes:
  plan:
    kind: call
    call: ask
    node_name: Plan the trip
    input:
      questions:
        - {question: "Which city?", header: city}
        - {question: "How many nights?", header: nights, options: [{label: "2"}, {label: "5"}]}
    output: {city: !str, nights: !str}
output: \${plan.output}
`;

  test(
    "a form question opens the chat's question card from its ask in place of the box; Esc closes it, and its answers go as one",
    async () => {
      const { chat, view, rows } = await viewing(TRIP);
      expect(rows().slice(-6)).toEqual([
        "questions  [Plan the trip]",
        top("Plan the trip"),
        boxRow(">"),
        BOTTOM,
        "  enter opens the form · /stop",
        "esc to go back to the chat",
      ]);
      view.key(ENTER);
      await settle();
      expect(rows().slice(-4)).toEqual(["", rule("Question 1 of 2 · city"), "? Which city? ", "esc closes the form"]);
      expect(rows()).toContain("│ Your input is needed for Plan the trip │");
      view.key(ESC);
      await settle();
      expect(rows().at(-2)).toBe("  enter opens the form · /stop");
      view.key(ENTER);
      await settle();
      view.key(typed("Paris"));
      view.key(ENTER);
      await settle();
      expect(rows().slice(-8)).toEqual([
        "? Which city? Paris",
        "",
        rule("Question 2 of 2 · nights"),
        "? How many nights? (Use arrow keys)",
        " » ● 2",
        "   ○ 5",
        "   ○ ✎ Other (write your own)",
        "esc closes the form",
      ]);
      view.key(named("down"));
      expect(await pressing(view, ENTER)).toMatchObject({ status: "succeeded", output: { city: "Paris", nights: "5" } });
      expect(rows().slice(-2)).toEqual(["answered: Plan the trip", "the run has ended · esc to go back"]);
      const answer = { city: "Paris", nights: "5" };
      expect(actions(chat)).toEqual([
        { type: "action", tool: "flows_answer", arguments: { run: "r1", call: "chat-1-r1/plan", answer }, result: "answered" },
      ]);
    },
    20_000,
  );

  /** An agent that shouts twice with the host tool `shout`. */
  const LOUD = `id: loud
name: loud
nodes:
  say: {kind: agent, tools: [shout], prompt: Shout hi., output: !str}
output: \${say.output}
`;

  /** A host tool that upper-cases its text, and shows the gate what it would shout. */
  const SHOUT = new FunctionTool({
    name: "shout",
    description: "Shout the text.",
    parametersJsonSchema: { type: "object", properties: { text: { type: "string" } }, required: ["text"] },
    execute: (args) => String(args["text"]).toUpperCase(),
    gateTarget: (args) => ({ shouts: String(args["text"]).toUpperCase() }),
  });

  test(
    "decides a held call as the user, never gated, from a card of what the call acts on: a refusal with the box's reason, then an approval",
    async () => {
      const replies = [callReply(["shout", { text: "hi" }, "t1"]), callReply(["shout", { text: "hey" }, "t2"]), textReply("done")];
      const model = new ScriptedModel(replies, XAI);
      const asked: string[] = [];
      const hold: BeforeToolCall = (call) => {
        asked.push(call.tool);
        return pending();
      };
      const { chat, log, view, rows } = await viewing(LOUD, { model, tools: { shout: SHOUT }, hold });
      expect(rows().slice(-6, -3)).toEqual(["held calls  [shout]", top("reason for shout, if any"), boxRow(">")]);
      view.key(typed("too loud"));
      view.key(ENTER);
      expect(rows(100).slice(-8, -5)).toEqual([
        "approve r1's shout call? a approves · r refuses · any other key leaves it held",
        "reason: too loud",
        'acts on {"shouts":"HI"}',
      ]);
      // Any other key leaves it held, and the reason back in the box.
      view.key(letter("x"));
      expect(rows().at(-4)).toBe(boxRow("> too loud"));
      view.key(ENTER);
      await pressing(view, letter("r"));
      const read = model.requests[1]!.messages.flatMap((message) => message.parts.filter((part) => part instanceof ToolResultPart));
      expect(read.map((part) => part.content)).toEqual(["Refused by the user, via yian-chat: too loud"]);
      // The run holds its next call, which takes the first's place.
      expect(rows(100).slice(-7, -5)).toEqual(["held calls  [shout]", "refused shout"]);
      view.key(ENTER);
      expect(rows(100).slice(-7, -5)).toEqual([
        "approve r1's shout call? a approves · r refuses · any other key leaves it held",
        'acts on {"shouts":"HEY"}',
      ]);
      expect(await pressing(view, letter("a"))).toMatchObject({ status: "succeeded", output: "done" });
      expect(asked).toEqual(["shout", "shout"]);
      expect(deciders(log)).toEqual([USER, USER]);
      expect(actions(chat).map((action) => (action as { result?: string }).result)).toEqual(["refused", "approved"]);
    },
    20_000,
  );
});
