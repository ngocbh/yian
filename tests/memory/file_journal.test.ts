/** FileMemory's journal: its lines and their fsyncs, change ids, and writers across sessions. */

import { afterEach, beforeEach, expect, mock, spyOn, test } from "bun:test";
import { type ChildProcess, spawn } from "node:child_process";
import { createHash } from "node:crypto";
import fs from "node:fs";
import { tmpdir } from "node:os";
import { join, relative } from "node:path";

import {
  type Applied,
  type Author,
  ChangeIdReusedError,
  MemoryBusyError,
  memoryErrorKind,
  type MemorySession,
} from "../../src/memory/base.ts";
import { FileMemory, type FileMemoryOptions, memoryVersion, readMembers } from "../../src/memory/file.ts";
import { Journal } from "../../src/memory/file_journal.ts";
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

/** The error a simulated crash throws. */
class Crash extends Error {
  override name = "Crash";
}

/** Return a Node system error with `code`, as a failing disk gives it. */
function systemError(code: string): Error {
  return Object.assign(new Error(`${code}: simulated`), { code, syscall: "write" });
}

/**
 * Spy on `fs.writeSync` with `fake`, which gets each call's descriptor and the bytes it writes, and
 * a `write` that writes the first `length` of them (all by default).
 */
function onWrite(fake: (fd: number, data: Buffer, write: (length?: number) => number) => number): void {
  const { writeSync } = fs;
  type Offset = number | null | undefined;
  spyOn(fs, "writeSync").mockImplementation(((fd: number, data: Uint8Array, offset: Offset, length: Offset) => {
    const start = offset ?? 0;
    const bytes = Buffer.from(data.buffer, data.byteOffset + start, length ?? data.length - start);
    return fake(fd, bytes, (part = bytes.length) => writeSync(fd, bytes, 0, part));
  }) as typeof fs.writeSync);
}

/**
 * Apply an append to `path` as change `id` that stops partway: `before` stops before its file is
 * replaced (the rename fails), `after` once the file is replaced but before its commit line is
 * whole (a crash cuts the line). Either leaves the change's entry with no outcome.
 */
async function crashedAppend(s: MemorySession, id: string, text: string, at: "before" | "after", path = "a.md") {
  if (at === "before") {
    spyOn(fs, "renameSync").mockImplementation(() => {
      throw systemError("ENOSPC");
    });
  } else {
    onWrite((_, data, write) => {
      if (!data.includes('"ok":true')) return write();
      write(data.length >> 1);
      throw new Crash("cut in the commit line");
    });
  }
  try {
    await expect(append(s, id, text, path)).rejects.toThrow();
  } finally {
    mock.restore();
  }
}

/** Return the journal's lines with no outcome check: the generation line dropped. */
function journalBody(): JsonObject[] {
  return journalLines().slice(1);
}

test("an entry with no outcome whose file is at its after hash is committed on open", async () => {
  const s = await openAt();
  await append(s, "first", "one");
  await crashedAppend(s, "x", "two", "after");
  expect(fs.readFileSync(join(root, "a.md"), "utf8")).toBe("one\ntwo");
  const t = await openAt();
  const version = versionNow();
  const committed = { ok: true, version, result: { path: "a.md", hash: sha256("one\ntwo"), version } };
  expect(journalBody().slice(-2)).toEqual([
    expect.objectContaining({ id: "x", after: sha256("one\ntwo") }),
    { id: "x", ...committed },
  ]);
  // Its id returns the stored result: the resumed call doesn't append again.
  expect(await append(t, "x", "two")).toEqual(committed as Applied);
  expect(fs.readFileSync(join(root, "a.md"), "utf8")).toBe("one\ntwo");
  await s.close();
  await t.close();
});

test("an entry with no outcome whose file is at its before hash is aborted on open; its id applies again", async () => {
  const s = await openAt();
  await append(s, "first", "one");
  await crashedAppend(s, "x", "two", "before");
  const t = await openAt();
  expect(journalBody().slice(-2)).toEqual([expect.objectContaining({ id: "x" }), { id: "x", aborted: true }]);
  expect(fs.readFileSync(join(root, "a.md"), "utf8")).toBe("one");
  expect((await append(t, "x", "two")).ok).toBe(true);
  expect(fs.readFileSync(join(root, "a.md"), "utf8")).toBe("one\ntwo");
  expect(journalBody().map((line) => (line.aborted ? "abort" : line.digest ? "entry" : "commit"))).toEqual([
    "entry",
    "commit",
    "entry",
    "abort",
    "entry",
    "commit",
  ]);
  await s.close();
  await t.close();
});

test("an aborted id is free: one session reuses it, and a session opened earlier reads both", async () => {
  const b = await openAt();
  const crashed = await openAt();
  await crashedAppend(crashed, "x", "text", "before");
  const a = await openAt();
  expect(journalBody().at(-1)).toEqual({ id: "x", aborted: true });
  const landed = await append(a, "x", "text");
  expect(landed.ok).toBe(true);
  // b read the journal before the crash; its next write reads entry 1 as aborted and entry 2 as committed.
  expect(await append(b, "x", "text")).toEqual(landed);
  expect(fs.readFileSync(join(root, "a.md"), "utf8")).toBe("text");
  for (const s of [a, b, crashed]) await s.close();
});

test("an entry whose file is at neither hash is stored as an unknown outcome, never applied twice", async () => {
  const s = await openAt();
  await append(s, "first", "one");
  await crashedAppend(s, "x", "two", "after");
  fs.writeFileSync(join(root, "a.md"), "edited by hand");
  const t = await openAt();
  const unknown = {
    ok: false,
    reason: "this change may have landed; check the file",
    current: { path: "a.md", hash: sha256("edited by hand") },
  };
  expect(journalBody().at(-1)).toEqual({ id: "x", ...unknown });
  expect(await append(t, "x", "two")).toEqual(unknown as Applied);
  expect(fs.readFileSync(join(root, "a.md"), "utf8")).toBe("edited by hand");
  await s.close();
  await t.close();
});

test("every write first resolves another session's cut-off entry, then applies its own", async () => {
  const a = await openAt();
  const b = await openAt();
  await crashedAppend(b, "bx", "from b", "after", "b.md");
  await append(a, "ay", "from a");
  expect(journalBody().map((line) => `${line.id}:${line.digest ? "entry" : line.ok ? "commit" : "other"}`)).toEqual([
    "bx:entry",
    "bx:commit",
    "ay:entry",
    "ay:commit",
  ]);
  await a.close();
  await b.close();
});

test("an entry with no outcome before later entries makes open fail, naming its line", async () => {
  const s = await openAt();
  await append(s, "x", "one");
  const lines = fs.readFileSync(join(root, ".yian/journal.jsonl"), "utf8").split("\n");
  // Drop x's commit line, then add y: x's entry no longer ends the journal.
  const entry = JSON.parse(lines[1]!) as JsonObject;
  const y = { ...entry, id: "y" };
  const damagedLines = [lines[0], lines[1], JSON.stringify(y), JSON.stringify({ id: "y", aborted: true }), ""];
  fs.writeFileSync(join(root, ".yian/journal.jsonl"), damagedLines.join("\n"));
  await expect(openAt()).rejects.toThrow(
    "`.yian/journal.jsonl` line 2 has an entry with no outcome before later entries; restore the journal or " +
      "remove it (undo history and change ids are lost)",
  );
  await s.close();
});

test("a temp file a crash left in .yian/tmp/ isn't a member and goes on the next write", async () => {
  const s = await openAt();
  await append(s, "x", "one");
  fs.writeFileSync(join(root, ".yian/tmp/0123456789abcdef"), "half a file");
  const version = versionNow();
  expect((await s.snapshot()).version).toBe(version);
  await append(s, "y", "two");
  expect(fs.readdirSync(join(root, ".yian/tmp"))).toEqual([]);
  await s.close();
});

test.each([
  ["ENOSPC writing the content", "next write"],
  ["ENOSPC writing the content", "open"],
  ["EXDEV renaming it into place", "next write"],
  ["EXDEV renaming it into place", "open"],
])("a change whose content write fails with %s is recovered by the %s, not left blocking", async (fault, by) => {
  const s = await openAt();
  await append(s, "first", "one");
  if (fault.startsWith("ENOSPC")) {
    onWrite((_, data, write) => {
      if (data.includes("zzz-content")) throw systemError("ENOSPC");
      return write();
    });
  } else {
    spyOn(fs, "renameSync").mockImplementation(() => {
      throw systemError("EXDEV");
    });
  }
  try {
    await expect(append(s, "x", "zzz-content")).rejects.toThrow(fault.slice(0, fault.indexOf(" ")));
  } finally {
    mock.restore();
  }
  expect(journalBody().at(-1)).toMatchObject({ id: "x", op: "append" });
  expect(fs.readdirSync(join(root, ".yian/tmp"))).toEqual([]);
  const writer = by === "open" ? await openAt() : s;
  if (by === "open") expect(journalBody().at(-1)).toEqual({ id: "x", aborted: true });
  // The resumed call sends x again: it lands once.
  expect((await append(writer, "x", "zzz-content")).ok).toBe(true);
  expect(fs.readFileSync(join(root, "a.md"), "utf8")).toBe("one\nzzz-content");
  expect(journalBody().filter((line) => line.aborted)).toEqual([{ id: "x", aborted: true }]);
  await s.close();
  if (writer !== s) await writer.close();
});

test.if(process.platform === "linux")("a recovery's outcome line is synced before the next write", async () => {
  const s = await openAt();
  await append(s, "first", "one");
  await crashedAppend(s, "x", "two", "after");
  const events: string[] = [];
  const where = (fd: number) =>
    relative(root, fs.readlinkSync(`/proc/self/fd/${fd}`)).replace(/^\.yian\/tmp\/.+$/, ".yian/tmp/*") || ".";
  const { fsyncSync, ftruncateSync } = fs;
  onWrite((fd, data, write) => {
    const kind = data.includes('"ok":true') ? " commit" : data.includes('"digest"') ? " entry" : "";
    events.push(`write ${where(fd)}${kind}`);
    return write();
  });
  spyOn(fs, "fsyncSync").mockImplementation((fd: number) => {
    events.push(`fsync ${where(fd)}`);
    fsyncSync(fd);
  });
  spyOn(fs, "ftruncateSync").mockImplementation((fd: number, length?: number) => {
    events.push(`truncate ${where(fd)}`);
    ftruncateSync(fd, length);
  });
  await append(s, "y", "three");
  expect(events.slice(0, 6)).toEqual([
    // The cut-off commit line goes, x is committed and synced, and only then is y's entry written.
    "truncate .yian/journal.jsonl",
    "fsync .yian/journal.jsonl",
    "write .yian/journal.jsonl commit",
    "fsync .yian/journal.jsonl",
    "write .yian/journal.jsonl entry",
    "fsync .yian/journal.jsonl",
  ]);
  await s.close();
});

/**
 * Make the `cut`th call to a function that writes to disk crash, as a process killed there would
 * stop: a write writes half its bytes first. Returns the number of such calls made so far.
 */
function cutAt(cut: number): () => number {
  let calls = 0;
  onWrite((_, data, write) => {
    if (++calls !== cut) return write();
    write(data.length >> 1);
    throw new Crash(`cut at call ${cut}`);
  });
  for (const name of ["fsyncSync", "ftruncateSync", "rmSync", "unlinkSync", "renameSync"] as const) {
    const original = fs[name] as (...args: unknown[]) => unknown;
    spyOn(fs, name).mockImplementation(((...args: unknown[]) => {
      if (++calls === cut) throw new Crash(`cut at call ${cut}`);
      return original(...args);
    }) as never);
  }
  return () => calls;
}

/**
 * Leave a memory at `root` that needs recovery: change x's entry with no outcome (`at` its after
 * hash with its commit line cut, or at its before hash), and a temp file a crash left.
 */
async function needsRecovery(at: "before" | "after"): Promise<void> {
  const s = await openAt();
  await append(s, "first", "one");
  await crashedAppend(s, "x", "two", at);
  fs.writeFileSync(join(root, ".yian/tmp/0123456789abcdef"), "left by a crash");
  await s.close();
}

/** Return what recovery leaves: the journal without its generation line, and the members' texts. */
function recovered(): string {
  const files = readMembers(root, { maxDepth: 4, maxFileChars: 100_000 }).files.map((file) => [file.path, file.text]);
  return JSON.stringify({ journal: journalBody(), files, tmp: fs.readdirSync(join(root, ".yian/tmp")) });
}

test.each(["before", "after"] as const)(
  "a crash at each of a recovery's writes (entry at %s) is recovered again, to the same state",
  async (at) => {
    const base = dir;
    root = join(base, "reference");
    await needsRecovery(at);
    const calls = cutAt(Number.POSITIVE_INFINITY);
    const reference = await openAt();
    const writes = calls();
    mock.restore();
    await reference.close();
    const expected = recovered();
    expect(writes).toBeGreaterThanOrEqual(at === "after" ? 5 : 3);
    for (let cut = 1; cut <= writes; cut++) {
      root = join(base, `cut-${cut}`);
      await needsRecovery(at);
      cutAt(cut);
      try {
        await expect(openAt()).rejects.toThrow(`cut at call ${cut}`);
      } finally {
        mock.restore();
      }
      const again = await openAt();
      expect({ cut, state: recovered() }).toEqual({ cut, state: expected });
      expect((await append(again, "x", "two")).ok).toBe(true);
      expect(fs.readFileSync(join(root, "a.md"), "utf8")).toBe("one\ntwo");
      await again.close();
    }
  },
);

/** Return the journal's generation. */
function generation(): string {
  return journalLines()[0]!.generation as string;
}

/**
 * Append lines to `a.md` on `s` as ids `<prefix>1`, `<prefix>2`, ... until the journal, which
 * must exist, is compacted (its generation changes), at most 40 of them. Returns the ids applied.
 */
async function appendUntilCompacted(s: MemorySession, prefix: string): Promise<string[]> {
  const first = generation();
  const ids: string[] = [];
  for (let n = 1; n <= 40; n++) {
    ids.push(`${prefix}${n}`);
    await append(s, `${prefix}${n}`, `line ${prefix}${n}`);
    if (generation() !== first) return ids;
  }
  throw new Error("the journal was never compacted");
}

test("past twice its kept size, a write rewrites the journal as a new generation of what the caps keep", async () => {
  const s = await openAt({ maxChangeIds: 2, maxUndo: 1 });
  await append(s, "start", "start");
  const old = generation();
  const renames: string[] = [];
  const { renameSync } = fs;
  spyOn(fs, "renameSync").mockImplementation((from: fs.PathLike, to: fs.PathLike) => {
    renames.push(`${relative(root, String(from)).replace(/[0-9a-f]{16}$/, "*")} -> ${relative(root, String(to))}`);
    renameSync(from, to);
  });
  const ids = await appendUntilCompacted(s, "c");
  mock.restore();
  expect(generation()).not.toBe(old);
  // The rewrite goes through a temp file renamed over the journal, as a content write does.
  const journalRenames = renames.filter((rename) => rename.endsWith(".yian/journal.jsonl"));
  expect(journalRenames).toEqual([".yian/tmp/* -> .yian/journal.jsonl"]);
  const body = journalBody();
  // The write that found the journal too big rewrote it before its own change: the two ids before
  // it stay, with their entries and commits, and only the newer of them keeps its inverse.
  expect(body.map((line) => `${line.id}:${line.digest ? "entry" : "commit"}`)).toEqual([
    `${ids.at(-3)}:entry`,
    `${ids.at(-3)}:commit`,
    `${ids.at(-2)}:entry`,
    `${ids.at(-2)}:commit`,
    `${ids.at(-1)}:entry`,
    `${ids.at(-1)}:commit`,
  ]);
  const truncate = { truncate: expect.any(Number) };
  expect(body.map((line) => line.inverse)).toEqual([null, undefined, truncate, undefined, truncate, undefined]);
  // A fresh session reads the same table: the last two ids answer, an older one applies again.
  const fresh = await openAt({ maxChangeIds: 2, maxUndo: 1 });
  const before = written();
  expect((await append(fresh, ids.at(-2)!, `line ${ids.at(-2)}`)).ok).toBe(true);
  expect(written()).toBe(before);
  await append(fresh, ids.at(-3)!, `line ${ids.at(-3)}`);
  expect(written()).not.toBe(before);
  expect(fs.readdirSync(join(root, ".yian/tmp"))).toEqual([]);
  await s.close();
  await fresh.close();
});

test("a session that read an older generation reads the compacted journal whole on its next write", async () => {
  const a = await openAt({ maxChangeIds: 2, maxUndo: 1 });
  const b = await openAt({ maxChangeIds: 2, maxUndo: 1 });
  await append(b, "b1", "from b");
  const old = generation();
  const ids = await appendUntilCompacted(a, "a");
  expect(generation()).not.toBe(old);
  const before = written();
  // b's next write reads the new journal from its first line: it knows a's last id, and adds its own.
  expect(await append(b, ids.at(-1)!, `line ${ids.at(-1)}`)).toMatchObject({ ok: true });
  expect(written()).toBe(before);
  await append(b, "b2", "from b again");
  const body = journalBody();
  expect(body.slice(-2).map((line) => line.id)).toEqual(["b2", "b2"]);
  const entries = body.filter((line) => line.digest).map((line) => line.id);
  expect(entries).toEqual([ids.at(-3), ids.at(-2), ids.at(-1), "b2"]);
  // and a, which compacted, reads b's write the same way.
  expect(await append(a, "b2", "from b again")).toMatchObject({ ok: true });
  expect(journalBody()).toEqual(body);
  await a.close();
  await b.close();
});

test("a compaction keeps neither an aborted entry nor its id", async () => {
  const s = await openAt({ maxChangeIds: 2, maxUndo: 1 });
  await append(s, "first", "one");
  spyOn(fs, "renameSync").mockImplementation(() => {
    throw systemError("ENOSPC");
  });
  await expect(append(s, "cut", "never")).rejects.toThrow("ENOSPC");
  mock.restore();
  await appendUntilCompacted(s, "c");
  expect(journalBody().filter((line) => line.id === "cut" || line.aborted)).toEqual([]);
  expect((await append(s, "cut", "never")).ok).toBe(true);
  await s.close();
});

test("a compaction first resolves another session's cut-off entry, and the new journal holds its outcome", async () => {
  const a = await openAt();
  for (const n of [1, 2, 3, 4, 5, 6]) await append(a, `a${n}`, `a${n}`);
  // c keeps two ids and no inverses, so its next write finds the journal far past twice that.
  const c = await openAt({ maxChangeIds: 2, maxUndo: 0 });
  const b = await openAt();
  const old = generation();
  await crashedAppend(b, "bx", "from b", "after", "b.md");
  await append(c, "c1", "from c");
  expect(generation()).not.toBe(old);
  expect(journalBody().map((line) => `${line.id}:${line.digest ? "entry" : line.ok ? "commit" : "other"}`)).toEqual([
    "a6:entry",
    "a6:commit",
    "bx:entry",
    "bx:commit",
    "c1:entry",
    "c1:commit",
  ]);
  const { id: _, ...stored } = journalBody()[3]!;
  expect(stored).toMatchObject({ ok: true, result: { path: "b.md", hash: sha256("from b") } });
  expect(await append(b, "bx", "from b", "b.md")).toEqual(stored as unknown as Applied);
  for (const s of [a, b, c]) await s.close();
});

test("a journal with no line break that isn't a cut-off generation line is damaged, not wiped", async () => {
  const s = await openAt();
  for (const n of [1, 2, 3]) await append(s, `x${n}`, `${n}`);
  const journalPath = join(root, ".yian/journal.jsonl");
  const flattened = fs.readFileSync(journalPath, "utf8").replaceAll("\n", " ");
  fs.writeFileSync(journalPath, flattened);
  await expect(openAt()).rejects.toThrow(
    "`.yian/journal.jsonl` line 1 isn't a journal line; restore the journal or remove it",
  );
  await expect(append(s, "x1", "1")).rejects.toThrow("line 1 isn't a journal line");
  expect(fs.readFileSync(journalPath, "utf8")).toBe(flattened);
  expect(fs.readFileSync(join(root, "a.md"), "utf8")).toBe("1\n2\n3");
  await s.close();
});

test.each(['{"generati', '{"generation":"0123', '{"generation":"0123456789abcdef"}'])(
  "a generation line a crash cut off (%s) starts the journal again",
  async (cut) => {
    const s = await openAt();
    fs.writeFileSync(join(root, ".yian/journal.jsonl"), cut);
    const t = await openAt();
    expect((await append(t, "x", "one")).ok).toBe(true);
    expect(journalLines()).toHaveLength(3);
    await s.close();
    await t.close();
  },
);

test("starting a journal never overwrites one the session hasn't read", () => {
  const meta = join(root, ".yian");
  fs.mkdirSync(meta, { recursive: true });
  const journalPath = join(meta, "journal.jsonl");
  const existing = '{"generation":"aaaaaaaaaaaaaaaa"}\n';
  fs.writeFileSync(journalPath, existing);
  const journal = new Journal(meta, { maxChangeIds: 10, maxUndo: 10, maxHistoryChars: 100 });
  const entry = {
    id: "x",
    digest: sha256("x"),
    op: "append",
    path: "a.md",
    author,
    at: "2026-11-07T00:00:00.000Z",
    before: "absent",
    after: sha256("x"),
    inverse: { absent: true as const },
  };
  // This journal was never caught up, so it holds no generation and would otherwise start one.
  expect(() => journal.writeEntry(entry)).toThrow(
    "`.yian/journal.jsonl` isn't empty, but this session hasn't read it; try again",
  );
  expect(fs.readFileSync(journalPath, "utf8")).toBe(existing);
});
