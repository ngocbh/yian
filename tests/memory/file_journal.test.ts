/** FileMemory's journal: its lines and their fsyncs, change ids, and writers across sessions. */

import { afterEach, beforeEach, expect, mock, spyOn, test } from "bun:test";
import { type ChildProcess, spawn } from "node:child_process";
import { createHash } from "node:crypto";
import fs from "node:fs";
import { tmpdir } from "node:os";
import { join, relative } from "node:path";

import {
  type Author,
  ChangeIdReusedError,
  MemoryBusyError,
  memoryErrorKind,
  type MemorySession,
} from "../../src/memory/base.ts";
import { FileMemory, type FileMemoryOptions, memoryVersion, readMembers } from "../../src/memory/file.ts";
import { lockFile, virtualScheduler } from "../../src/platform.ts";
import { canonicalJson, type JsonObject } from "../../src/typesys/json.ts";

let dir: string;
let root: string;
const signal = new AbortController().signal;
const author: Author = { kind: "agent", name: "answer", run: "r", node: "map/0/answer" };
const START = Date.UTC(2026, 10, 7);

beforeEach(() => {
  dir = fs.mkdtempSync(join(tmpdir(), "yian-fj-"));
  root = join(dir, "m");
});

afterEach(() => {
  mock.restore();
  fs.rmSync(dir, { recursive: true, force: true });
});

/** Return the sha256 hex digest of `data`. */
function sha256(data: string | Uint8Array): string {
  return createHash("sha256").update(data).digest("hex");
}

/** Open a FileMemory on `root`, creating it, with a clock that starts at `START`. */
async function openAt(options: Partial<FileMemoryOptions> = {}): Promise<MemorySession> {
  return new FileMemory({ root, id: "t/x", scheduler: virtualScheduler(START), ...options }).open({
    create: true,
    itemType: null,
    signal,
  });
}

/** Return the journal's lines, parsed. */
function journalLines(): JsonObject[] {
  const text = fs.readFileSync(join(root, ".yian/journal.jsonl"), "utf8");
  expect(text.endsWith("\n")).toBe(true);
  return text
    .slice(0, -1)
    .split("\n")
    .map((line) => JSON.parse(line) as JsonObject);
}

/** Return the memory's version as its members read now. */
function versionNow(): string {
  return memoryVersion(readMembers(root, { maxDepth: 4, maxFileChars: 100_000 }).files);
}

test("append applies; the journal holds the generation, one entry and one commit line", async () => {
  const s = await openAt();
  const change = { op: "append", path: "notes/a.md", text: "x" };
  const applied = await s.apply({ id: "r/answer/1/0", change, author, signal });
  const version = versionNow();
  expect(applied).toEqual({ ok: true, version, result: { path: "notes/a.md", hash: sha256("x"), version } });
  const lines = journalLines();
  expect(lines).toHaveLength(3);
  expect(Object.keys(lines[0]!)).toEqual(["generation"]);
  expect(lines[0]!.generation).toMatch(/^[0-9a-f]{16}$/);
  expect(lines[1]).toEqual({
    id: "r/answer/1/0",
    digest: sha256(canonicalJson(change)),
    op: "append",
    path: "notes/a.md",
    author: { kind: "agent", name: "answer", run: "r", node: "map/0/answer" },
    at: "2026-11-07T00:00:00.000Z",
    before: "absent",
    after: sha256("x"),
    inverse: { absent: true },
  });
  expect(lines[2]).toEqual({ id: "r/answer/1/0", ok: true, version, result: applied.ok ? applied.result : {} });
  expect(fs.statSync(join(root, ".yian/journal.jsonl")).mode & 0o777).toBe(0o600);

  await s.apply({ id: "r/answer/1/1", change: { op: "append", path: "notes/a.md", text: "y" }, author, signal });
  // An append's inverse cuts the file back to its old bytes: it needs none of the old text.
  expect(journalLines()[3]).toMatchObject({ before: sha256("x"), after: sha256("x\ny"), inverse: { truncate: 1 } });
  expect(journalLines()).toHaveLength(5);
  await s.close();
});

test.if(process.platform === "linux")(
  "each journal line is fsynced, and the content goes through a synced temp file in .yian/tmp before the commit",
  async () => {
    const s = await openAt();
    const events: string[] = [];
    /** The path of `fd` relative to the root, a temp file's name hidden. */
    const where = (fd: number) =>
      relative(root, fs.readlinkSync(`/proc/self/fd/${fd}`)).replace(/^\.yian\/tmp\/.+$/, ".yian/tmp/*") || ".";
    const name = (path: fs.PathLike) => relative(root, String(path)).replace(/^\.yian\/tmp\/.+$/, ".yian/tmp/*");
    const { writeSync, fsyncSync, renameSync } = fs;
    spyOn(fs, "writeSync").mockImplementation(((fd: number, ...rest: unknown[]) => {
      events.push(`write ${where(fd)}`);
      return (writeSync as (...args: unknown[]) => number)(fd, ...rest);
    }) as typeof fs.writeSync);
    spyOn(fs, "fsyncSync").mockImplementation((fd: number) => {
      events.push(`fsync ${where(fd)}`);
      fsyncSync(fd);
    });
    spyOn(fs, "renameSync").mockImplementation((from: fs.PathLike, to: fs.PathLike) => {
      events.push(`rename ${name(from)} ${name(to)}`);
      renameSync(from, to);
    });

    await s.apply({ id: "a", change: { op: "append", path: "a.md", text: "x" }, author, signal });
    expect(events).toEqual([
      // The first write makes the journal, its generation line synced with its folder entry.
      "write .yian/journal.jsonl",
      "fsync .yian/journal.jsonl",
      "fsync .yian",
      "write .yian/journal.jsonl",
      "fsync .yian/journal.jsonl",
      "write .yian/tmp/*",
      "fsync .yian/tmp/*",
      "rename .yian/tmp/* a.md",
      "fsync .",
      "fsync .yian/tmp",
      "write .yian/journal.jsonl",
      "fsync .yian/journal.jsonl",
    ]);
    events.length = 0;
    await s.apply({ id: "b", change: { op: "append", path: "a.md", text: "y" }, author, signal });
    expect(events).toEqual([
      "write .yian/journal.jsonl",
      "fsync .yian/journal.jsonl",
      "write .yian/tmp/*",
      "fsync .yian/tmp/*",
      "rename .yian/tmp/* a.md",
      "fsync .",
      "fsync .yian/tmp",
      "write .yian/journal.jsonl",
      "fsync .yian/journal.jsonl",
    ]);
    expect(fs.readdirSync(join(root, ".yian/tmp"))).toEqual([]);
    await s.close();
  },
);

test("a refused change writes nothing, not even the journal", async () => {
  const s = await openAt({ maxFileChars: 3 });
  const applied = await s.apply({ id: "a", change: { op: "append", path: "a.md", text: "long" }, author, signal });
  expect(applied.ok).toBe(false);
  expect(fs.existsSync(join(root, ".yian/journal.jsonl"))).toBe(false);
  await s.close();
});

test("a journal line cut off by a crash is dropped before the next line is written", async () => {
  const s = await openAt();
  await s.apply({ id: "a", change: { op: "append", path: "a.md", text: "x" }, author, signal });
  fs.appendFileSync(join(root, ".yian/journal.jsonl"), '{"id":"b","dig');
  await s.apply({ id: "c", change: { op: "append", path: "a.md", text: "y" }, author, signal });
  expect(journalLines().map((line) => line.id ?? "generation")).toEqual(["generation", "a", "a", "c", "c"]);
  await s.close();
});

test("a line that isn't a journal line makes the next write fail, naming the line", async () => {
  const s = await openAt();
  await s.apply({ id: "a", change: { op: "append", path: "a.md", text: "x" }, author, signal });
  fs.appendFileSync(join(root, ".yian/journal.jsonl"), '{"hello":1}\n');
  await expect(
    s.apply({ id: "b", change: { op: "append", path: "a.md", text: "y" }, author, signal }),
  ).rejects.toThrow(
    "`.yian/journal.jsonl` line 4 isn't a journal line; restore the journal or remove it " +
      "(undo history and change ids are lost)",
  );
  expect(fs.readFileSync(join(root, "a.md"), "utf8")).toBe("x");
  await s.close();
});

/** Apply an append of `text` to `path` on `s` as change `id`. */
function append(s: MemorySession, id: string, text: string, path = "a.md") {
  return s.apply({ id, change: { op: "append", path, text }, author, signal });
}

/** Return the journal's bytes and every member's text, to show that nothing was written. */
function written(): string {
  const journal = fs.existsSync(join(root, ".yian/journal.jsonl"))
    ? fs.readFileSync(join(root, ".yian/journal.jsonl"), "utf8")
    : "";
  const files = readMembers(root, { maxDepth: 4, maxFileChars: 100_000 }).files.map((file) => [file.path, file.text]);
  return JSON.stringify({ journal, files });
}

test("the same id with the same change returns the stored result and writes nothing", async () => {
  const s = await openAt();
  const first = await append(s, "x", "once");
  const before = written();
  await append(s, "y", "other");
  const after = written();
  expect(await append(s, "x", "once")).toEqual(first);
  expect(written()).toBe(after);
  expect(before).not.toBe(after);
  // A session opened later reads the id from the journal.
  const later = await openAt();
  expect(await append(later, "x", "once")).toEqual(first);
  expect(written()).toBe(after);
  expect(fs.readFileSync(join(root, "a.md"), "utf8")).toBe("once\nother");
  await s.close();
  await later.close();
});

test("the same id with another change throws ChangeIdReusedError and writes nothing", async () => {
  const s = await openAt();
  await append(s, "x", "once");
  const before = written();
  const reused = append(s, "x", "twice");
  await expect(reused).rejects.toThrow("change id `x` was used for another change");
  await expect(reused).rejects.toBeInstanceOf(ChangeIdReusedError);
  expect(memoryErrorKind(await reused.catch((error: unknown) => error))).toBe("change_id_reused");
  expect(written()).toBe(before);
  await s.close();
});

test("a conflict records no id: the same id applies once its precondition holds", async () => {
  fs.mkdirSync(root, { recursive: true });
  fs.writeFileSync(join(root, "a.md"), "A");
  const s = await openAt();
  const change = { op: "create", path: "b.md", text: "B" };
  fs.writeFileSync(join(root, "b.md"), "taken");
  expect((await s.apply({ id: "x", change, author, signal })).ok).toBe(false);
  fs.rmSync(join(root, "b.md"));
  expect((await s.apply({ id: "x", change, author, signal })).ok).toBe(true);
  await s.close();
});

test("after maxChangeIds later changes, an old id applies again", async () => {
  const s = await openAt({ maxChangeIds: 2 });
  await append(s, "x", "x");
  await append(s, "y", "y");
  expect(await append(s, "x", "x")).toMatchObject({ ok: true });
  expect(fs.readFileSync(join(root, "a.md"), "utf8")).toBe("x\ny");
  await append(s, "z", "z");
  // x and y were the two before z, so x's id is dropped; a fresh session drops the same ones.
  const fresh = await openAt({ maxChangeIds: 2 });
  for (const session of [s, fresh]) {
    await expect(append(session, "y", "other")).rejects.toThrow(ChangeIdReusedError);
  }
  expect((await append(fresh, "x", "x again")).ok).toBe(true);
  expect(fs.readFileSync(join(root, "a.md"), "utf8")).toBe("x\ny\nz\nx again");
  await s.close();
  await fresh.close();
});

test("one session's concurrent writes queue for the lock, so none of them is busy", async () => {
  const s = await openAt({ lockTimeoutMs: 0 });
  const results = await Promise.all(["a", "b", "c", "d", "e"].map((id) => append(s, id, id)));
  expect(results.every((result) => result.ok)).toBe(true);
  expect(fs.readFileSync(join(root, "a.md"), "utf8").split("\n").sort()).toEqual(["a", "b", "c", "d", "e"]);
  await s.close();
});

test("two sessions on one root serialize: both sessions' appends land, and each sees the other's", async () => {
  const a = await openAt();
  const b = await new FileMemory({ root, scheduler: virtualScheduler(START) }).open({
    create: false,
    itemType: null,
    signal,
  });
  const ids = ["1", "2", "3", "4", "5"];
  const results = await Promise.all([
    ...ids.map((n) => append(a, `a${n}`, `a${n}`)),
    ...ids.map((n) => append(b, `b${n}`, `b${n}`)),
  ]);
  expect(results.every((result) => result.ok)).toBe(true);
  const lines = fs.readFileSync(join(root, "a.md"), "utf8").split("\n");
  expect([...lines].sort()).toEqual([...ids.map((n) => `a${n}`), ...ids.map((n) => `b${n}`)]);
  // Each session's next write first reads the other's: a's ids are known to b, and b's to a.
  const before = written();
  expect(await append(b, "a1", "a1")).toEqual(results[0]!);
  expect(await append(a, "b1", "b1")).toEqual(results[5]!);
  expect(written()).toBe(before);
  const last = await append(a, "a6", "a6");
  expect(last).toMatchObject({ ok: true, version: versionNow() });
  await a.close();
  await b.close();
});

test("lockTimeoutMs: 0 with the lock held elsewhere gives MemoryBusyError and writes nothing", async () => {
  const s = await openAt({ lockTimeoutMs: 0 });
  await append(s, "x", "x");
  const before = written();
  const lock = lockFile(join(root, ".yian/lock"), { private: true });
  try {
    const busy = append(s, "y", "y");
    await expect(busy).rejects.toBeInstanceOf(MemoryBusyError);
    expect(memoryErrorKind(await busy.catch((error: unknown) => error))).toBe("busy");
  } finally {
    lock.release();
  }
  expect(written()).toBe(before);
  expect((await append(s, "y", "y")).ok).toBe(true);
  await s.close();
});

/** The children a test started, killed after it if still running. */
const children: ChildProcess[] = [];

afterEach(() => {
  for (const child of children.splice(0)) {
    if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
  }
});

/**
 * Run a Bun child that opens the memory at `root` and appends `count` lines to `log.md` as ids
 * `<who>-<i>`, and resolve with its exit code and stderr. A child still running after 60 s is
 * killed, so a hung lock can't leave it behind.
 */
function appender(who: string, count: number): Promise<{ code: number | null; stderr: string }> {
  const source = new URL("../../src/memory/file.ts", import.meta.url).href;
  const script = `
const { FileMemory } = await import(${JSON.stringify(source)});
const signal = new AbortController().signal;
const s = await new FileMemory({ root: ${JSON.stringify(root)} }).open({ create: false, itemType: null, signal });
const who = ${JSON.stringify(who)};
for (let i = 0; i < ${count}; i++) {
  const change = { op: "append", path: "log.md", text: who + " " + i };
  const applied = await s.apply({ id: who + "-" + i, change, author: { kind: "host", name: who }, signal });
  if (!applied.ok) throw new Error(JSON.stringify(applied));
}
await s.close();
`;
  const child = spawn(process.execPath, ["-e", script], { stdio: ["ignore", "ignore", "pipe"], env: {} });
  children.push(child);
  let stderr = "";
  child.stderr!.setEncoding("utf8").on("data", (chunk: string) => {
    stderr += chunk;
  });
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error(`${who} still ran after 60 s; stderr:\n${stderr}`));
    }, 60_000);
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      resolve({ code, stderr });
    });
  });
}

test("writers in separate processes serialize through the lock: every append lands once", async () => {
  await (await openAt()).close();
  const ran = await Promise.all([appender("p", 8), appender("q", 8)]);
  for (const { code, stderr } of ran) expect({ code, stderr }).toEqual({ code: 0, stderr: "" });
  const lines = fs.readFileSync(join(root, "log.md"), "utf8").split("\n");
  const expected = ["p", "q"].flatMap((who) => Array.from({ length: 8 }, (_, i) => `${who} ${i}`));
  expect([...lines].sort()).toEqual([...expected].sort());
  // Every entry is followed by its own commit: no writer's lines landed inside another's.
  const journal = journalLines().slice(1);
  expect(journal).toHaveLength(32);
  for (let i = 0; i < journal.length; i += 2) {
    expect(journal[i]!.digest).toBeString();
    expect(journal[i + 1]).toMatchObject({ id: journal[i]!.id, ok: true });
  }
});
