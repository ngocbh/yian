/** yian's own bad-value errors read as the built-ins they extend, and are told apart from them. */

import { expect, test } from "bun:test";

import { errorTypeName } from "../../src/diagnostics.ts";
import { YianRangeError, YianTypeError } from "../../src/typesys/errors.ts";

test.each([
  [YianTypeError, TypeError, "TypeError"],
  [YianRangeError, RangeError, "RangeError"],
] as const)("%p is the built-in it extends, by every name a message shows", (Own, Builtin, name) => {
  const error = new Own("bad", { cause: "why" });

  expect(error).toBeInstanceOf(Builtin);
  expect([error.name, errorTypeName(error), String(error), error.cause]).toEqual([name, name, `${name}: bad`, "why"]);
  expect(new Builtin("x")).not.toBeInstanceOf(Own);
});

test("a subclass keeps its own name", () => {
  class Refused extends YianRangeError {
    override name = "Refused";
  }

  expect([errorTypeName(new Refused("x")), String(new Refused("x"))]).toEqual(["Refused", "Refused: x"]);
});
