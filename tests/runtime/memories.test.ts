/**
 * A run's memories (`RunMemories`): each bound instance opened once, checked against its input
 * and contract, and snapshotted into the memory value the run records.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  BUILTIN_MEMORY_TOOLS,
  MemoryBase,
  MemoryBusyError,
  type MemoryContract,
  memoryErrorKind,
  type MemorySession,
} from "../../src/memory/base.ts";
import { FileMemory } from "../../src/memory/file.ts";
import { type MemoryBinding, RunMemories } from "../../src/runtime/memories.ts";
import { type JsonObject, logJson } from "../../src/typesys/json.ts";
import { jsonPart, MemoryRef } from "../../src/typesys/values.ts";
import { FakeMemory, fileInfo, RecordingSession } from "../helpers/memory_fakes.ts";

const tmp = mkdtempSync(join(tmpdir(), "yian-run-memories-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

const signal = new AbortController().signal;
const fresh = { create: true, signal };

/** Bind `instance` to input `input` of type `type`, its items of the JSON Schema `itemType`. */
function bind(input: string, instance: MemoryBase, type = "FileMemory", itemType: JsonObject | null = null): MemoryBinding {
  return { input, instance, type, itemType };
}

/** Return what `run` rejects with; fail when it resolves. */
async function rejection(run: () => Promise<unknown>): Promise<unknown> {
  try {
    await run();
  } catch (error) {
    return error;
  }
  throw new Error("expected a rejection");
}

test("one instance bound to two inputs is opened once and gets one session", async () => {
  const memory = new FakeMemory(() => new RecordingSession(fileInfo("t/x")));
  const memories = await RunMemories.open([bind("lessons", memory), bind("notes", memory)], fresh);
  expect(memory.opens).toBe(1);
  expect(memory.openedWith).toEqual([{ create: true, itemType: null }]);
  expect(memory.sessions[0]!.snapshots).toBe(1);
  expect(memories.session("t/x")).toBe(memory.sessions[0]!);
  expect([...memories.values.keys()]).toEqual(["lessons", "notes"]);
  expect(memories.values.get("lessons")).toEqual(memories.values.get("notes")!);
  await memories.close();
  expect(memory.sessions[0]!.closed).toBe(1);
});

test("one instance bound to two inputs of different item types is refused, before anything opens", async () => {
  const memory = new FakeMemory(() => new RecordingSession(fileInfo("t/x")));
  const cases = { type: "object", properties: { title: { type: "string" } } };
  const notes = { type: "object", properties: { body: { type: "integer" } } };
  await expect(
    RunMemories.open([bind("a", memory, "FileMemory", cases), bind("b", memory, "FileMemory", notes)], fresh),
  ).rejects.toThrow("memory inputs `a` and `b` share one instance but declare different item types");
  await expect(
    RunMemories.open([bind("a", memory, "FileMemory", cases), bind("b", memory, "FileMemory", null)], fresh),
  ).rejects.toThrow("memory inputs `a` and `b` share one instance but declare different item types");
  expect(memory.opens).toBe(0);
  // The same item type, its keys in another order, is one type.
  const reordered = { properties: { title: { type: "string" } }, type: "object" };
  const memories = await RunMemories.open(
    [bind("a", memory, "FileMemory", cases), bind("b", memory, "FileMemory", reordered)],
    fresh,
  );
  expect(memory.opens).toBe(1);
  await memories.close();
});

test("two instances with one identity fail", async () => {
  const a = new FakeMemory(() => new RecordingSession(fileInfo("t/x")));
  const b = new FakeMemory(() => new RecordingSession(fileInfo("t/x")));
  await expect(RunMemories.open([bind("lessons", a), bind("notes", b)], fresh)).rejects.toThrow(
    "`notes` and `lessons` are two memories with identity `t/x`; bind one instance",
  );
  expect(a.sessions[0]!.closed).toBe(1);
  expect(b.sessions[0]!.closed).toBe(1);
});

test("an instance whose contract type differs from the input's fails, before anything opens", async () => {
  class Store extends FakeMemory {
    static override readonly contract: MemoryContract = {
      type: "VectorDBMemory",
      tools: BUILTIN_MEMORY_TOOLS.VectorDBMemory.map((name) => ({
        name,
        write: false,
        description: name,
        parameters: { type: "object" },
      })),
    };
  }
  const store = new Store();
  await expect(RunMemories.open([bind("lessons", store)], fresh)).rejects.toThrow(
    "`lessons` is `FileMemory`, but its instance is `VectorDBMemory`",
  );
  expect(store.opens).toBe(0);
});

test("checkContract failures are reported with the input's name", async () => {
  class Bad extends FakeMemory {
    static override readonly contract: MemoryContract = { type: "FileMemory", tools: [] };
  }
  await expect(RunMemories.open([bind("lessons", new Bad())], fresh)).rejects.toThrow(
    "memory `lessons`: Bad declares FileMemory but lacks its tools: list, read, search, append, create, edit, delete, rewrite",
  );
});

test("a host type's tools must be those of the class the flow was loaded with", async () => {
  const tools = ["find", "keep"].map((name) => ({ name, write: name === "keep", description: name, parameters: { type: "object" } }));
  class Notes extends MemoryBase {
    static override readonly contract: MemoryContract = { type: "Notes", tools };
    override async open(): Promise<MemorySession> {
      return new RecordingSession({ type: "Notes", identity: "t/n", settings: {}, tools });
    }
    override reset(): Promise<void> {
      throw new Error("not reset");
    }
  }
  const hostTypes = new Map([["Notes", ["find", "keep", "drop"]]]);
  await expect(RunMemories.open([bind("notes", new Notes(), "Notes")], { ...fresh, hostTypes })).rejects.toThrow(
    "memory `notes`: Notes's tools (find, keep) aren't those of the `Notes` the flow was loaded with (find, keep, drop)",
  );
  const memories = await RunMemories.open([bind("notes", new Notes(), "Notes")], {
    ...fresh,
    hostTypes: new Map([["Notes", ["find", "keep"]]]),
  });
  expect(memories.values.get("notes")!.tools.map((tool) => tool.name)).toEqual(["find", "keep"]);
  await memories.close();
});

test("a session whose info differs from its contract is refused, and closed", async () => {
  const wrongType = new FakeMemory(() => new RecordingSession({ ...fileInfo("t/x"), type: "Other" }));
  await expect(RunMemories.open([bind("lessons", wrongType)], fresh)).rejects.toThrow(
    "memory `lessons`: its session is a `Other`, not the `FileMemory` its class declares",
  );
  expect(wrongType.sessions[0]!.closed).toBe(1);
  const fewer = new FakeMemory(
    () => new RecordingSession({ ...fileInfo("t/x"), tools: FileMemory.contract.tools.slice(0, 2) }),
  );
  await expect(RunMemories.open([bind("lessons", fewer)], fresh)).rejects.toThrow(
    "memory `lessons`: its session's tools (list, read) aren't its contract's " +
      "(list, read, search, append, create, edit, delete, rewrite)",
  );
});

test("a failed open or snapshot closes the sessions opened so far", async () => {
  const first = new FakeMemory(() => new RecordingSession(fileInfo("t/a")));
  const failing = new FakeMemory();
  failing.failOpen = new Error("disk gone");
  const error = await rejection(() => RunMemories.open([bind("a", first), bind("b", failing)], fresh));
  expect((error as Error).message).toBe("memory `b`: disk gone");
  expect(first.sessions[0]!.closed).toBe(1);

  const snapped = new FakeMemory(() => {
    const session = new RecordingSession(fileInfo("t/c"));
    session.onSnapshot = () => {
      throw new Error("unreadable");
    };
    return session;
  });
  const other = new FakeMemory(() => new RecordingSession(fileInfo("t/d")));
  await expect(RunMemories.open([bind("d", other), bind("c", snapped)], fresh)).rejects.toThrow("memory `c`: unreadable");
  expect(other.sessions[0]!.closed).toBe(1);
  expect(snapped.sessions[0]!.closed).toBe(1);
});

test("a busy memory at open stays busy, named by its input", async () => {
  const busy = new FakeMemory();
  busy.failOpen = new MemoryBusyError("this memory is busy: another writer holds its lock");
  const error = await rejection(() => RunMemories.open([bind("lessons", busy)], fresh));
  expect(memoryErrorKind(error)).toBe("busy");
  expect((error as Error).message).toBe("memory `lessons` is busy: another writer holds its lock; try again");
});

test("a stop while opening rejects with the signal's reason", async () => {
  const controller = new AbortController();
  const reason = new Error("stop");
  const memory = new FakeMemory();
  memory.failOpen = reason;
  controller.abort(reason);
  await expect(RunMemories.open([bind("lessons", memory)], { create: true, signal: controller.signal })).rejects.toBe(
    reason,
  );
});

test("each memory's MemoryRef holds identity, type, settings, version, recall and tool specs, and no location", async () => {
  const root = mkdtempSync(join(tmp, "m-"));
  writeFileSync(join(root, "MEMORY.md"), "Prefer short answers.");
  const memories = await RunMemories.open([bind("lessons", new FileMemory({ root, id: "t/lessons" }))], fresh);
  const value = memories.values.get("lessons")!;
  expect(MemoryRef.isMemoryRef(value)).toBe(true);
  expect(value.identity).toBe("t/lessons");
  expect(value.type).toBe("FileMemory");
  expect(value.settings["index"]).toBe("MEMORY.md");
  expect(value.version).toMatch(/\S/);
  expect(value.recall).toEqual({ text: "Prefer short answers.", cut: null, hint: null });
  expect(value.tools.map((tool) => tool.name)).toEqual([...BUILTIN_MEMORY_TOOLS.FileMemory]);
  expect(value.tools[0]).toEqual({ ...FileMemory.contract.tools[0]! });
  expect(logJson(jsonPart(value, [], "mapping"))).not.toContain(root);
  await memories.close();
});

test("an unknown identity has no session", async () => {
  const memories = await RunMemories.open([], fresh);
  expect(() => memories.session("t/x")).toThrow("this run has no session on memory `t/x`");
});

test("reopen opens with create: false, snapshots only for drift, and reports one entry per memory", async () => {
  let version = "v1";
  const memory = new FakeMemory(() => {
    const session = new RecordingSession(fileInfo("t/x"));
    session.onSnapshot = () => ({ version, recall: { text: version, cut: null, hint: null } });
    return session;
  });
  const first = await RunMemories.open([bind("lessons", memory), bind("notes", memory)], fresh);
  const recorded = first.values;
  await first.close();
  const waking = await RunMemories.reopen([bind("lessons", memory), bind("notes", memory)], recorded, { signal, drift: false });
  expect(memory.openedWith.at(-1)).toEqual({ create: false, itemType: null });
  expect(memory.sessions.at(-1)!.snapshots).toBe(0);
  expect(waking.drift).toEqual([]);
  expect(waking.memories.values.get("lessons")).toBe(recorded.get("lessons")!);
  await waking.memories.close();
  version = "v2";
  const crashed = await RunMemories.reopen([bind("lessons", memory), bind("notes", memory)], recorded, { signal, drift: true });
  expect(crashed.drift).toEqual([{ input: "lessons", identity: "t/x", recorded: "v1", current: "v2" }]);
  expect(Object.isFrozen(crashed.drift)).toBe(true);
  // The recall stays the recorded one.
  expect(crashed.memories.values.get("notes")!.recall.text).toBe("v1");
  await crashed.memories.close();
});

test("reopen refuses a recorded memory with no binding before opening anything", async () => {
  const memory = new FakeMemory(() => new RecordingSession(fileInfo("t/x")));
  const opened = await RunMemories.open([bind("lessons", memory)], fresh);
  await opened.close();
  const other = new FakeMemory();
  await expect(
    RunMemories.reopen([bind("notes", other)], new Map([["lessons", opened.values.get("lessons")!]]), { signal, drift: true }),
  ).rejects.toThrow("memory `lessons` (`t/x`) isn't at its location; bind it with --memory");
  expect(other.opens).toBe(0);
});

test("close closes every session and returns the ones that failed, never rejecting", async () => {
  const failing = new FakeMemory(() => {
    const session = new RecordingSession(fileInfo("t/a"));
    session.close = async () => {
      throw new Error("gone");
    };
    return session;
  });
  const fine = new FakeMemory(() => new RecordingSession(fileInfo("t/b")));
  const memories = await RunMemories.open([bind("a", failing), bind("b", fine)], fresh);
  const failed = await memories.close();
  expect(failed.map(({ identity, error }) => [identity, (error as Error).message])).toEqual([["t/a", "gone"]]);
  expect(fine.sessions[0]!.closed).toBe(1);
});
