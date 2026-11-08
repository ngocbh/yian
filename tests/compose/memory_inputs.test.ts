/**
 * Memory inputs at load: `!FileMemory`, `!VectorDBMemory(T)` and a host's own memory type declare
 * a flow input the runner binds, never a default, a variable, an output or a typed binding.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { flowFingerprint } from "../../src/compile/manifest.ts";
import { loadFlow, loadFlowSource } from "../../src/compose/loader.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { MemoryBase, type MemoryContract, type MemorySession } from "../../src/memory/base.ts";
import { hostInputs } from "../../src/run.ts";
import { isMemoryType } from "../../src/typesys/values.ts";
import { loadErrorAsync } from "../helpers/declarations.ts";

beforeEach(() => install(BUILTIN));
afterEach(forget);

/** A host's own memory type, which a flow can name once `memoryTypes` lists it. */
class TradeJournal extends MemoryBase {
  static override readonly contract: MemoryContract = {
    type: "TradeJournal",
    tools: [{ name: "record_trade", write: true, description: "Record a trade.", parameters: { type: "object" } }],
  };
  override open(): Promise<MemorySession> {
    throw new Error("not opened at load");
  }
  override async reset(): Promise<void> {}
}

/** A flow declaring `inputs` (YAML lines under `input:`), whose one node reads nothing. */
function flowWith(inputs: string, extra = ""): string {
  return `id: f\nname: f\n${extra}input:\n${inputs.replace(/^/gm, "  ")}\nnodes:\n  greet:\n    kind: code\n    output: str\n    code: return "hi"\noutput: \${greet.output}\n`;
}

describe("memory inputs", () => {
  test("FileMemory, VectorDBMemory(T) and Optional(FileMemory) inputs load, and nothing need read them", async () => {
    const flow = await loadFlow(
      flowWith("lessons: !FileMemory\ncases: !VectorDBMemory(Case)\nnotes: !Optional(FileMemory)", "typedefs:\n  Case: {title: str}\n"),
    );
    const types = new Map(flow.input.map((param) => [param.name, param.contract!.type]));
    expect([...types.values()].every(isMemoryType)).toBe(true);
    expect(types.get("cases")!.memory).toBe("VectorDBMemory");
    expect(flow.input.map((param) => [param.name, param.required])).toEqual([
      ["lessons", true],
      ["cases", true],
      ["notes", false],
    ]);
  });

  test("a memory input takes no default", async () => {
    const error = await loadErrorAsync(loadFlow(flowWith("lessons: !FileMemory notes.md")));
    expect(error.diagnostic.message).toBe("'lessons': a memory input takes no default; the runner binds it");
  });

  test.each([
    ["a variable", "id: f\nname: f\nkind: agent\nvariables:\n  v: !FileMemory\nprompt: hi\noutput: str\n"],
    [
      "a typed binding",
      "id: f\nname: f\ninput:\n  lessons: !FileMemory\nnodes:\n  n:\n    kind: code\n    input:\n      m: !FileMemory ${input.lessons}\n    code: return 1\n    output: int\noutput: ${n.output}\n",
    ],
  ])("a memory type in %s is refused", async (_, text) => {
    const error = await loadErrorAsync(loadFlow(text));
    expect(error.diagnostic.message).toContain("`!FileMemory` declares a memory, which only a flow input can be");
  });

  test("a node output can't be a memory", async () => {
    const text = "id: f\nname: f\nnodes:\n  n:\n    kind: code\n    output: FileMemory\n    code: return null\noutput: null\n";
    const error = await loadErrorAsync(loadFlow(text));
    expect(error.diagnostic.message).toBe("an output can't be a memory (`FileMemory` declares a flow input)");
  });
});

describe("host memory types", () => {
  test("an unknown memory tag is an unknown type without memoryTypes", async () => {
    const error = await loadErrorAsync(loadFlow(flowWith("j: !TradeJournal")));
    expect(error.diagnostic.message).toContain("TradeJournal");
    expect(error.diagnostic.message).not.toContain("memory");
  });

  test("memoryTypes lets an input name the host's own memory type", async () => {
    const flow = await loadFlow(flowWith("j: !TradeJournal"), { memoryTypes: [TradeJournal] });
    expect(flow.input[0]!.contract!.type.memory).toBe("TradeJournal");
    expect(flow.memoryTypes).toEqual([TradeJournal]);
  });

  test("a recorded flow loads without memoryTypes and keeps its fingerprint", async () => {
    const flow = await loadFlow(flowWith("j: !TradeJournal\nk: !Optional(TradeJournal)"), { memoryTypes: [TradeJournal] });
    const again = await loadFlowSource(flow.source!);
    expect(again.input[0]!.contract!.type.memory).toBe("TradeJournal");
    expect(again.memoryTypes).toEqual([]);
    expect(flowFingerprint(again.compiled)).toBe(flowFingerprint(flow.compiled));
  });

  test("a recorded flow reads an unknown tag as a memory only on an input", async () => {
    const flow = await loadFlow(flowWith("j: !TradeJournal"), { memoryTypes: [TradeJournal] });
    const root = Object.keys(flow.source!.files)[0]!;
    const text = flow.source!.files[root]!.text.replace("output: str", "output: TradeJournal");
    const edited = { ...flow.source!, files: { ...flow.source!.files, [root]: { ...flow.source!.files[root]!, text } } };
    const error = await loadErrorAsync(loadFlowSource(edited));
    expect(error.diagnostic.message).toContain("unknown type 'TradeJournal'");
  });

  test("a host memory type named like a typedef is refused at the typedef, then the input", async () => {
    const text = flowWith("j: !TradeJournal", "typedefs:\n  TradeJournal: {region: str}\n");
    const error = await loadErrorAsync(loadFlow(text, { memoryTypes: [TradeJournal] }));
    expect(error.diagnostic.message).toBe(
      "'j': the memory type `!TradeJournal` is also a typedef name; rename the typedef or the memory type",
    );
    expect(error.diagnostic.frames.map((frame) => [frame.range.startLine, frame.note])).toEqual([
      [4, "typedef 'TradeJournal'"],
      [6, null],
    ]);
  });

  test("the typedef keeps its name where no input uses the host's type", async () => {
    const text = flowWith("q: !str", "typedefs:\n  TradeJournal: {region: str}\n");
    expect((await loadFlow(text, { memoryTypes: [TradeJournal] })).input[0]!.name).toBe("q");
  });

  test.each([
    ["not a class", [{ contract: TradeJournal.contract }]],
    ["a class with no contract", [class Bare {}]],
  ])("memoryTypes refuses %s", async (_, memoryTypes) => {
    await expect(loadFlow(flowWith("q: !str"), { memoryTypes: memoryTypes as never })).rejects.toThrow(TypeError);
  });
});

test("memoryTypes refuses a contract that lists a tool twice", async () => {
  class Twice extends TradeJournal {
    static override readonly contract: MemoryContract = {
      type: "Twice",
      tools: [TradeJournal.contract.tools[0]!, TradeJournal.contract.tools[0]!],
    };
  }
  await expect(loadFlow(flowWith("q: !str"), { memoryTypes: [Twice] })).rejects.toThrow(
    "Twice's contract lists the tool record_trade twice",
  );
});

describe("hostInputs", () => {
  test("a memory input given a plain value is refused", async () => {
    const flow = await loadFlow(flowWith("lessons: !FileMemory\nnotes: !Optional(FileMemory)"));
    expect(() => hostInputs(flow.input, { lessons: "notes.md" })).toThrow(
      "memory `lessons` is bound to a MemoryBase instance, not a value",
    );
    expect(() => hostInputs(flow.input, { notes: { memory: {} } })).toThrow(
      "memory `notes` is bound to a MemoryBase instance, not a value",
    );
  });
});
