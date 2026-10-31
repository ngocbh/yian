/**
 * An inline LOOP body's returned fields must fit the parameters generated from its seeds.
 *
 * An inline body becomes a child flow whose parameters are typed from the seeds, and each returned
 * field is bound to its parameter again on the next iteration. A returned field that can never fit
 * that parameter is a load error, so loading accepts only flows that can run.
 */

import { expect, test } from "bun:test";

import { runFlow } from "../../src/run.ts";
import { assembled, assembleText, type CaseRow, loadConformanceFlow, loadError, runCase } from "../helpers/declarations.ts";

/** A flow whose inline LOOP body returns `seen` as `Optional(list(str))`, seeded by `seed`. */
function seenLoop(seed: string): string {
  return `
id: seen
name: seen
nodes:
  repeat:
    kind: loop
    while: "n < 2"
    max: 5
    input:
      seen: ${seed}
      n: 0
    nodes:
      bump:
        kind: code
        input:
          n: \${input.n}
        output:
          seen: Optional(list(str))
          n: int
        code: |
          return {"seen": ["a", "b"].slice(0, inputs["n"] + 1), "n": inputs["n"] + 1}
    output: \${bump.output}
output: \${repeat.output}
`;
}

/**
 * A flow whose inline LOOP carries `n`, seeded from an `int` flow input.
 *
 * @param bodyOutput The body CODE node's declared output type.
 * @param bodyValue The TS expression the body node returns as `n`.
 */
function numberLoop(bodyOutput: string, bodyValue: string): string {
  return `
id: number
name: number
input:
  z: !int
nodes:
  repeat:
    kind: loop
    times: 2
    input:
      n: \${input.z}
    nodes:
      bump:
        kind: code
        input:
          n: \${input.n}
        output:
          n: ${bodyOutput}
        code: |
          return {"n": ${bodyValue}}
    output: \${bump.output}
output: \${repeat.output}
`;
}

/**
 * A flow whose inline LOOP body returns `note` from one CODE node.
 *
 * @param seed The LOOP's `note:` seed binding, e.g. `!str "start"`.
 * @param bodyOutput The body CODE node's declared output type.
 * @param bodyValue The TS expression the body node returns.
 */
function noteLoop(seed: string, bodyOutput: string, bodyValue: string): string {
  return `
id: note
name: note
nodes:
  repeat:
    kind: loop
    times: 2
    input:
      note: ${seed}
    nodes:
      mk:
        kind: code
        input:
          note: \${input.note}
        output: ${bodyOutput}
        code: |
          return ${bodyValue}
    output:
      note: \${mk.output}
output: \${repeat.output}
`;
}

/** The message of the load error assembling `text` raises. */
const message = (text: string) => loadError(() => assembleText(text)).message;

/** Load `text` and run it on `inputs`. */
const run = async (text: string, inputs: Record<string, unknown> = {}) =>
  runFlow(await loadConformanceFlow(text), inputs);

test("an untyped null seed with a nullable list body is rejected with a hint", () => {
  expect(message(seenLoop("${null}"))).toBe(
    "loop node 'repeat': the body returns 'seen' as Optional(list(str)), but the carried field is typed " +
      "None from its seed; declare the seed's type: `seen: !Optional(list(str)) ${null}`",
  );
});

test("an untyped null seed error points at the seed", () => {
  const text = seenLoop("${null}");
  const line = text.split("\n").indexOf("      seen: ${null}") + 1;

  const frames = loadError(() => assembleText(text)).diagnostic.frames;

  expect(frames).toHaveLength(1);
  expect(frames[0]!.range.startLine).toBe(line);
});

test("a typed null seed with a nullable list body loads", () => {
  assembleText(seenLoop("!Optional(list(str)) ${null}"));
});

test("typed null seed with a nullable list body runs", async () => {
  const result = await run(seenLoop("!Optional(list(str)) ${null}"));

  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ seen: ["a", "b"], n: 2n });
});

test("an int seed with a float body field is rejected", () => {
  expect(message(numberLoop("float", 'inputs["n"] + 1.5'))).toBe(
    "loop node 'repeat': the body returns 'n' as float, but the carried field is typed int from its seed; " +
      "declare the seed's type: `n: !float ${input.z}`",
  );
});

test("an int seed with an int body field loads", () => {
  assembleText(numberLoop("int", 'inputs["n"] + 1'));
});

test("int seed with an int body field loads and runs", async () => {
  const result = await run(numberLoop("int", 'inputs["n"] + 1'), { z: 0n });

  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ n: 2n });
});

test("a typed seed names its declared type", () => {
  const text = numberLoop("float", 'inputs["n"] + 1.5').replace("n: ${input.z}", "n: !int ${input.z}");

  expect(message(text)).toBe(
    "loop node 'repeat': the body returns 'n' as float, but the carried field is declared int; " +
      "declare the seed's type: `n: !float ${input.z}`",
  );
});

/** An inline loop seeding `note` and `n` whose body returns only `n`. */
const PARTIAL = `
id: partial
name: partial
nodes:
  repeat:
    kind: loop
    times: 2
    input:
      note: \${null}
      n: 0
    nodes:
      bump:
        kind: code
        input:
          n: \${input.n}
        output:
          n: int
        code: |
          return {"n": inputs["n"] + 1}
    output: \${bump.output}
output: \${repeat.output}
`;

test("a field the body omits is not checked", () => {
  // The load half.
  assembleText(PARTIAL);
});

test("a field the body omits is not checked (runs)", async () => {
  const result = await run(PARTIAL);

  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ note: null, n: 2n });
});

test("a typed seed with an optional body field loads", () => {
  assembleText(noteLoop('!str "start"', "Optional(str)", 'inputs["note"] + "!"'));
});

test("typed seed with an optional body field runs", async () => {
  const result = await run(noteLoop('!str "start"', "Optional(str)", 'inputs["note"] + "!"'));

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toEqual({ note: "start!!" });
});

test("a typed seed with a returned null loads", () => {
  assembleText(noteLoop('!str "start"', "Optional(str)", "null"));
});

test("typed seed with a returned null fails the next iteration", async () => {
  const result = await run(noteLoop('!str "start"', "Optional(str)", "null"));

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("contract.input");
});

test("an untyped seed with an optional body field is rejected with a hint", () => {
  expect(message(noteLoop('"start"', "Optional(str)", 'inputs["note"] + "!"'))).toBe(
    "loop node 'repeat': the body returns 'note' as Optional(str), but the carried field is typed str " +
      'from its seed; declare the seed\'s type: `note: !Optional(str) "start"`',
  );
});

test("a typed str seed with an int body field is rejected", () => {
  expect(message(noteLoop('!str "start"', "int", "1"))).toBe(
    "loop node 'repeat': carried field 'note' — body output is 'int', carried source is 'str' " +
      "(a returned field must keep its carried type)",
  );
});

test("a literal seed with a str body field is rejected", () => {
  expect(message("typedefs:\n  AB: Literal(a, b)\n" + noteLoop("!AB a", "str", '"a"'))).toBe(
    "loop node 'repeat': the body returns 'note' as str, but the carried field is declared Literal(a, b); " +
      'declare the seed\'s type: `note: !str "a"`',
  );
});

/**
 * A flow whose inline LOOP carries `n` from a literal seed and whose body returns `n` as `bodyType`.
 *
 * @param seed The LOOP's `n:` seed binding, e.g. `"0"`.
 * @param bodyType The body CODE node's declared type for `n`.
 * @param bodyValue The TS expression the body node returns as `n`.
 * @param typedefs The flow's `typedefs:` block, if any.
 */
function seededLoop(seed: string, bodyType = "int", bodyValue = 'inputs["n"] + 1', typedefs = ""): string {
  return `${typedefs}
id: seeded
name: seeded
nodes:
  repeat:
    kind: loop
    times: 3
    input:
      n: ${seed}
    nodes:
      bump:
        kind: code
        input:
          n: \${input.n}
        output:
          n: ${bodyType}
        code: |
          return {"n": ${bodyValue}}
    output: \${bump.output}
output: \${repeat.output}
`;
}

/** The seed binding a load error suggests for `field`, from its trailing `` `field: ...` ``. */
function suggestedSeed(text: string, field: string): string {
  const match = new RegExp(`\`${field}: ([^\`]+)\`$`).exec(text);
  if (match === null) throw new Error(`no suggested seed in: ${text}`);
  return match[1]!;
}

test("a text seed for an int body field suggests an int seed that loads", async () => {
  const error = message(seededLoop('"0"'));

  expect(error).toBe(
    "loop node 'repeat': the body returns 'n' as int, but the carried field is typed str from its seed; " +
      'the seed "0" is not of type int: seed one that is, e.g. `n: !int 0`',
  );
  const result = await run(seededLoop(suggestedSeed(error, "n")));
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toEqual({ n: 3n });
});

test("a null seed for an int body field suggests an int seed that loads", () => {
  const error = message(seededLoop("null"));

  expect(error).toBe(
    "loop node 'repeat': the body returns 'n' as int, but the carried field is typed None from its seed; " +
      "the seed null is not of type int: seed one that is, e.g. `n: !int 0`",
  );
  assembleText(seededLoop(suggestedSeed(error, "n")));
});

test("a text seed for an optional int body field suggests an optional int seed that loads", () => {
  const error = message(seededLoop('"p1"', "Optional(int)", "null"));

  expect(error).toBe(
    "loop node 'repeat': the body returns 'n' as Optional(int), but the carried field is typed str from its " +
      'seed; the seed "p1" is not of type int: seed one that is, e.g. `n: !Optional(int) 0`',
  );
  assembleText(seededLoop(suggestedSeed(error, "n"), "Optional(int)", "null"));
});

test("a text seed for a str body field suggests empty text", () => {
  const error = message(seededLoop("1", "str", '"a"'));

  expect(error).toBe(
    "loop node 'repeat': the body returns 'n' as str, but the carried field is typed int from its seed; " +
      'the seed 1 is not of type str: seed one that is, e.g. `n: !str ""`',
  );
  assembleText(seededLoop(suggestedSeed(error, "n"), "str", '"a"'));
});

test("a null seed for an Any body field suggests a non-null seed that loads", () => {
  const error = message(seededLoop("null", "Any", '"a"'));

  expect(error).toBe(
    "loop node 'repeat': the body returns 'n' as Any, but the carried field is typed None from its seed; " +
      "the seed null is not of type Any: seed one that is, e.g. `n: !Any 0`",
  );
  assembleText(seededLoop(suggestedSeed(error, "n"), "Any", '"a"'));
});

test.each([
  { seed: '"$${x}"', shown: "$${x}" },
  { seed: '"a $${x}"', shown: "a $${x}" },
  { seed: '"call(x)"', shown: "call(x)" },
])("an escaped or quoted text seed $seed for an int body field suggests an int seed that loads", ({ seed, shown }) => {
  const error = message(seededLoop(seed));

  expect(error).toBe(
    "loop node 'repeat': the body returns 'n' as int, but the carried field is typed str from its seed; " +
      `the seed ${shown} is not of type int: seed one that is, e.g. \`n: !int 0\``,
  );
  assembleText(seededLoop(suggestedSeed(error, "n")));
});

test("a seed of another kind with no example value names the type", () => {
  const typedefs = "typedefs:\n  Rec:\n    a: int\n";

  expect(message(seededLoop('"abc"', "Rec", '{"a": 1}', typedefs))).toBe(
    "loop node 'repeat': the body returns 'n' as {\"a\":int}, but the carried field is typed str from its seed; " +
      'the seed "abc" is not of type Rec: seed one that is',
  );
});

test("a seed for a bounded int body field with no example in bounds names the type", () => {
  expect(message(seededLoop('"abc"', "int(1..5)", "1"))).toBe(
    "loop node 'repeat': the body returns 'n' as int(1..5), but the carried field is typed str from its seed; " +
      'the seed "abc" is not of type int(1..5): seed one that is',
  );
});

// ---------- Pinned harvested cases ----------

test.each<CaseRow>([
  {
    id: "load-loop/compose_inline_loop_carried_types__literal_seed_with_a_str_body_field_is_rejected",
    call: assembled,
  },
  { id: "load-loop/compose_inline_loop_carried_types__typed_str_seed_rejects_an_int_body_field", call: assembled },
  { id: "load-loop/compose_inline_loop_carried_types__text_seed_suggests_an_int_seed_that_loads", call: assembled },
])("$id", runCase);
