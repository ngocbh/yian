/** `MemoryBase`'s contract check, the built-in contracts' tool names and the session errors. */

import { expect, test } from "bun:test";

import {
  BUILTIN_MEMORY_TOOLS,
  ChangeIdReusedError,
  checkContract,
  isMemory,
  MemoryBase,
  MemoryBusyError,
  type MemoryContract,
  memoryErrorKind,
  type MemorySession,
  type MemoryToolSpec,
} from "../../src/memory/base.ts";

/** Return a read tool's spec named `name`. */
function spec(name: string): MemoryToolSpec {
  return { name, write: false, description: `The ${name} tool.`, parameters: { type: "object" } };
}

/** A memory whose methods are never called: the check reads only its class's contract. */
abstract class Unopened extends MemoryBase {
  override open(): Promise<MemorySession> {
    throw new Error("not opened in these tests");
  }
  override reset(): Promise<void> {
    throw new Error("not reset in these tests");
  }
}

/** Return an instance of a class named `name` whose contract is `contract`. */
function memoryWith(name: string, contract: MemoryContract | undefined): MemoryBase {
  const cls = { [name]: class extends Unopened {} }[name]!;
  Object.defineProperty(cls, "contract", { value: contract });
  return new cls();
}

test("a contract's tool names are checked against the instance at bind", () => {
  class Bad extends Unopened {
    static override readonly contract = { type: "FileMemory", tools: [] };
  }
  expect(() => checkContract(new Bad())).toThrow(
    "Bad declares FileMemory but lacks its tools: list, read, search, append, create, edit, delete, rewrite",
  );
});

test("each built-in contract's names, in any order, pass", () => {
  for (const [type, names] of Object.entries(BUILTIN_MEMORY_TOOLS)) {
    expect(() => checkContract(memoryWith("Host", { type, tools: names.map(spec).reverse() }))).not.toThrow();
  }
});

test("the built-in contracts' tool names", () => {
  expect(BUILTIN_MEMORY_TOOLS).toEqual({
    FileMemory: ["list", "read", "search", "append", "create", "edit", "delete", "rewrite"],
    VectorDBMemory: ["search", "read", "list", "add", "edit", "retire", "delete"],
  });
  expect(Object.isFrozen(BUILTIN_MEMORY_TOOLS)).toBe(true);
  expect(Object.isFrozen(BUILTIN_MEMORY_TOOLS.FileMemory)).toBe(true);
});

test("a missing tool is named", () => {
  const tools = BUILTIN_MEMORY_TOOLS.VectorDBMemory.filter((name) => name !== "retire").map(spec);
  expect(() => checkContract(memoryWith("Store", { type: "VectorDBMemory", tools }))).toThrow(
    "Store declares VectorDBMemory but lacks its tools: retire",
  );
});

test("a tool the built-in contract doesn't have is refused", () => {
  const tools = [...BUILTIN_MEMORY_TOOLS.FileMemory, "upsert"].map(spec);
  expect(() => checkContract(memoryWith("Folder", { type: "FileMemory", tools }))).toThrow(
    "Folder declares FileMemory but adds tools FileMemory doesn't have: upsert",
  );
});

test("a tool listed twice is refused, in any contract", () => {
  const twice = [...BUILTIN_MEMORY_TOOLS.FileMemory, "read"].map(spec);
  expect(() => checkContract(memoryWith("Folder", { type: "FileMemory", tools: twice }))).toThrow(
    "Folder's contract lists the tool read twice",
  );
  const journal = { type: "TradeJournal", tools: [spec("record_trade"), spec("record_trade")] };
  expect(() => checkContract(memoryWith("Journal", journal))).toThrow(
    "Journal's contract lists the tool record_trade twice",
  );
});

test("a host's own contract type keeps its own tool names", () => {
  const journal = { type: "TradeJournal", tools: [spec("record_trade"), spec("list")] };
  expect(() => checkContract(memoryWith("Journal", journal))).not.toThrow();
});

test("a class with no contract is refused", () => {
  class Bare extends Unopened {}
  expect(() => checkContract(new Bare())).toThrow("Bare declares no memory contract; give it a static contract");
});

test("the errors a session throws", () => {
  const reused = new ChangeIdReusedError("r/answer/1/0");
  expect(reused).toBeInstanceOf(Error);
  expect(reused.name).toBe("ChangeIdReusedError");
  expect(reused.message).toBe("change id `r/answer/1/0` was used for another change");
  expect(reused.id).toBe("r/answer/1/0");
  const busy = new MemoryBusyError("lock wait timed out");
  expect(busy).toBeInstanceOf(Error);
  expect(busy.name).toBe("MemoryBusyError");
  expect(busy.message).toBe("lock wait timed out");
});

test("each session error carries its kind", () => {
  expect(memoryErrorKind(new MemoryBusyError("lock wait timed out"))).toBe("busy");
  expect(memoryErrorKind(new ChangeIdReusedError("x"))).toBe("change_id_reused");
});

test("an error from another copy of yian is read by its brand", () => {
  const brand = Symbol.for("yian.memory_error");
  const busy = Object.assign(new Error("busy elsewhere"), { [brand]: "busy" });
  expect(memoryErrorKind(busy)).toBe("busy");
  expect(memoryErrorKind({ [brand]: "change_id_reused" })).toBe("change_id_reused");
});

test("anything else has no memory error kind, and the reader never throws", () => {
  const brand = Symbol.for("yian.memory_error");
  const hostile = new Proxy(
    {},
    {
      get() {
        throw new Error("trap");
      },
    },
  );
  for (const value of [new Error("plain"), { [brand]: "other" }, { [brand]: ["busy"] }, null, "busy", 1, hostile]) {
    expect(memoryErrorKind(value)).toBeNull();
  }
});

test("a MemoryBase instance is a memory", () => {
  expect(isMemory(memoryWith("Folder", { type: "FileMemory", tools: [] }))).toBe(true);
});

test("a memory from another copy of yian is read by its brand", () => {
  class Foreign {
    static readonly contract = { type: "FileMemory", tools: [] };
    open(): Promise<MemorySession> {
      throw new Error("not opened in these tests");
    }
    reset(): Promise<void> {
      throw new Error("not reset in these tests");
    }
  }
  Object.defineProperty(Foreign.prototype, Symbol.for("yian.memory"), { value: true });
  expect(isMemory(new Foreign())).toBe(true);
});

test("anything else is not a memory, and the check never throws", () => {
  const brand = Symbol.for("yian.memory");
  const hostile = new Proxy(
    {},
    {
      get() {
        throw new Error("trap");
      },
    },
  );
  const open = async () => {
    throw new Error("unused");
  };
  for (const value of [
    { open, reset: open },
    { [brand]: true, open },
    { [brand]: "yes", open, reset: open },
    null,
    "memory",
    hostile,
  ]) {
    expect(isMemory(value)).toBe(false);
  }
});
