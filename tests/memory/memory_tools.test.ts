/** A memory's granted tools as `FunctionTool`s over its run session (`memoryTools`). */

import { expect, test } from "bun:test";

import { ProviderError } from "../../src/llm/errors.ts";
import { ChangeIdReusedError, MemoryBusyError } from "../../src/memory/base.ts";
import { memoryTools } from "../../src/memory/tools.ts";
import { ToolContext } from "../../src/tools/context.ts";
import { isPermanentToolError, isReportedToolError } from "../../src/tools/errors.ts";
import { fileRef, RecordingSession } from "../helpers/memory_fakes.ts";

const value = fileRef("t/lessons");

/** A tool context for a call with id `callId` by node `node` of run `runId`. */
function ctx(fields: { callId: string; node: string; runId: string }): ToolContext {
  return new ToolContext({ ...fields, attempt: 1, toolName: "x", context: undefined, signal: new AbortController().signal });
}

/** Return what `run` rejects with; fail when it resolves. */
async function rejection(run: () => unknown): Promise<unknown> {
  try {
    await run();
  } catch (error) {
    return error;
  }
  throw new Error("expected a rejection");
}

const AT = ctx({ callId: "r/answer/1/0", node: "map/0/answer", runId: "r" });

test("a write tool's call becomes apply with ctx.callId as the change id and the caller's author", async () => {
  const session = new RecordingSession();
  session.onApply = () => ({ ok: true, version: "v2", result: { path: "a.md", hash: "h", version: "v2" } });
  const [append] = memoryTools(value, session, { input: "lessons", tools: ["append"] }, { kind: "agent", name: "answer" });
  expect(append!.name).toBe("append");
  expect(append!.resumable).toBe(true);
  expect(append!.isConcurrencySafe).toBeNull(); // writes run alone
  expect(append!.description).toBe(value.tools.find((tool) => tool.name === "append")!.description);
  expect(append!.parametersJsonSchema).toEqual(value.tools.find((tool) => tool.name === "append")!.parameters);
  const result = await append!.execute({ path: "a.md", text: "x" }, AT);
  expect(result).toEqual({ path: "a.md", hash: "h", version: "v2" });
  expect(session.applied).toEqual([
    {
      id: "r/answer/1/0",
      change: { op: "append", path: "a.md", text: "x" },
      author: { kind: "agent", name: "answer", run: "r", node: "map/0/answer" },
    },
  ]);
});

test("the tools come in the value's spec order, all of them for a whole grant", () => {
  const session = new RecordingSession();
  const some = memoryTools(value, session, { input: "lessons", tools: ["append", "read"] }, { kind: "agent", name: "a" });
  expect(some.map((tool) => tool.name)).toEqual(["read", "append"]);
  const all = memoryTools(value, session, { input: "lessons", tools: null }, { kind: "tool", name: "a" });
  expect(all.map((tool) => tool.name)).toEqual(value.tools.map((tool) => tool.name));
  expect(all.every((tool) => tool.resumable)).toBe(true);
});

test("a grant of a tool the value lacks is refused", () => {
  expect(() =>
    memoryTools(value, new RecordingSession(), { input: "lessons", tools: ["upsert"] }, { kind: "agent", name: "a" }),
  ).toThrow("`lessons` has no tool `upsert`; it has list, read, search, append, create, edit, delete, rewrite");
});

test("a conflict is a permanent reported error; busy is a reported, retryable one", async () => {
  const session = new RecordingSession();
  const [edit] = memoryTools(value, session, { input: "lessons", tools: ["edit"] }, { kind: "agent", name: "a" });
  session.onApply = () => ({
    ok: false,
    reason: "the old text isn't in a.md; read the file and copy it exactly",
    current: { path: "a.md", hash: "h1", text: "y" },
  });
  const conflict = await rejection(() => edit!.execute({ path: "a.md", old: "x", new: "z" }, AT));
  expect(isReportedToolError(conflict)).toBe(true);
  expect(isPermanentToolError(conflict)).toBe(true);
  expect((conflict as Error).message).toBe(
    'the old text isn\'t in a.md; read the file and copy it exactly\ncurrent: {"path":"a.md","hash":"h1","text":"y"}',
  );

  session.onApply = () => ({ ok: false, reason: "a.md doesn't exist", current: null });
  await expect(edit!.execute({ path: "a.md", old: "x", new: "z" }, AT)).rejects.toThrow(/^a\.md doesn't exist$/);

  session.onApply = () => {
    throw new MemoryBusyError("this memory is busy: another writer holds its lock");
  };
  const busy = await rejection(() => edit!.execute({ path: "a.md", old: "x", new: "z" }, AT));
  expect(isReportedToolError(busy)).toBe(true);
  expect(isPermanentToolError(busy)).toBe(false);
  expect((busy as Error).message).toBe("`lessons` is busy; try again");
});

test("a busy error from another copy of yian is read by its brand", async () => {
  const session = new RecordingSession();
  const foreign = Object.assign(new Error("held"), { [Symbol.for("yian.memory_error")]: "busy" });
  session.onCall = () => {
    throw foreign;
  };
  const [read] = memoryTools(value, session, { input: "notes", tools: ["read"] }, { kind: "agent", name: "a" });
  const error = await rejection(() => read!.execute({ path: "a.md" }, AT));
  expect(isReportedToolError(error)).toBe(true);
  expect((error as Error).message).toBe("`notes` is busy; try again");
});

test("a reused change id is a PermanentToolError; another fault is thrown as is", async () => {
  const session = new RecordingSession();
  const [append] = memoryTools(value, session, { input: "lessons", tools: ["append"] }, { kind: "tool", name: "save" });
  session.onApply = () => {
    throw new ChangeIdReusedError("r/save");
  };
  const reused = await rejection(() => append!.execute({ path: "a.md", text: "x" }, AT));
  expect(isPermanentToolError(reused)).toBe(true);
  expect(isReportedToolError(reused)).toBe(false);
  expect((reused as Error).name).toBe("PermanentToolError");
  expect((reused as Error).message).toBe("`lessons`: change id `r/answer/1/0` was used for another change");

  const fault = new Error("disk full");
  session.onApply = () => {
    throw fault;
  };
  await expect(append!.execute({ path: "a.md", text: "x" }, AT)).rejects.toBe(fault);
});

test("a read tool is concurrency-safe and calls session.call", async () => {
  const session = new RecordingSession();
  session.onCall = (tool, args) => ({ tool, path: args["path"] as string, text: "x", hash: "h" });
  const [read] = memoryTools(value, session, { input: "lessons", tools: ["read"] }, { kind: "agent", name: "a" });
  expect(read!.isConcurrencySafe?.({ path: "a.md" })).toBe(true);
  expect(read!.resumable).toBe(true);
  expect(await read!.execute({ path: "a.md" }, AT)).toEqual({ tool: "read", path: "a.md", text: "x", hash: "h" });
  expect(session.calls).toEqual([{ tool: "read", args: { path: "a.md" } }]);
  expect(session.applied).toEqual([]);
});

test("outside a run the author holds no run or node", async () => {
  const session = new RecordingSession();
  const [append] = memoryTools(value, session, { input: "lessons", tools: ["append"] }, { kind: "tool", name: "save" });
  await append!.execute({ path: "a.md", text: "x" }, ctx({ callId: "abc", node: "", runId: "" }));
  expect(session.applied[0]!.author).toEqual({ kind: "tool", name: "save" });
});

test("a transient provider error, from an embedder, is a reported error the model may try again", async () => {
  const session = new RecordingSession();
  const [search, append] = memoryTools(value, session, { input: "lessons", tools: ["search", "append"] }, {
    kind: "agent",
    name: "a",
  });
  for (const kind of ["rate_limit", "server", "timeout"] as const) {
    const fault = new ProviderError(kind, `embedding failed (${kind})`);
    session.onCall = () => {
      throw fault;
    };
    session.onApply = () => {
      throw fault;
    };
    for (const tool of [search!, append!]) {
      const error = await rejection(() => tool.execute({ query: "q", path: "a.md", text: "x" }, AT));
      expect(isReportedToolError(error)).toBe(true);
      expect(isPermanentToolError(error)).toBe(false);
      expect((error as Error).message).toBe(`\`lessons\` couldn't embed: ${kind}; try again`);
    }
  }
  // One another attempt can't fix (refused credentials) is a fault, as it is.
  const refused = new ProviderError("auth", "embedding key refused");
  session.onCall = () => {
    throw refused;
  };
  expect(await rejection(() => search!.execute({ query: "q" }, AT))).toBe(refused);
});

test("a reused change id from a read tool is a fault, not a change's refusal", async () => {
  const session = new RecordingSession();
  session.onCall = () => {
    throw new ChangeIdReusedError("x");
  };
  const [read] = memoryTools(value, session, { input: "lessons", tools: ["read"] }, { kind: "agent", name: "a" });
  const error = await rejection(() => read!.execute({ path: "a.md" }, AT));
  expect(isPermanentToolError(error)).toBe(false);
  expect(isReportedToolError(error)).toBe(false);
  expect((error as Error).message).toBe("`lessons`'s read tool reported a reused change id, which a read can't have");
});
