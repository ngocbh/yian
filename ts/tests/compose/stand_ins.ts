/**
 * Stand-in node kinds for the Compose tests, since no concrete kind is ported yet: `StandInAgent`,
 * `StandInCode` and `StandInCase` shaped like the real kinds, `standIn` for one-off classes, and
 * `withKinds`, which registers classes in `KINDS` and `BINDING_FIELDS` for one call.
 */

import {
  type BuildContext,
  type BuildResult,
  NodeDescriptor,
  type NodeDescriptorFields,
} from "../../src/compose/nodes/base.ts";
import { BINDING_FIELDS, type DescriptorClass, KINDS } from "../../src/compose/nodes/index.ts";
import { StartNode } from "../../src/nodes/start/node.ts";

/** The nine author-written kinds. */
export const ALL_KIND_NAMES: readonly string[] = [
  "agent",
  "call",
  "case",
  "code",
  "human_input",
  "loop",
  "map",
  "tool",
  "wait",
];

/** An `agent` stand-in taking `prompt`, `input` and `output`. */
export class StandInAgent extends NodeDescriptor {
  static override readonly kind = "agent";
  static override readonly allowed = this.allow("prompt", "inputs", "outputs");
  readonly prompt: unknown;

  constructor(fields: NodeDescriptorFields & { prompt?: unknown }) {
    super(fields);
    this.prompt = fields.prompt ?? null;
  }
}

/** A `code` stand-in requiring `code` and taking `input`, `output` and `asserts`; it builds. */
export class StandInCode extends NodeDescriptor {
  static override readonly kind = "code";
  static override readonly allowed = this.allow("code", "inputs", "outputs", "asserts");
  static override readonly required = this.require("code");
  readonly code: unknown;
  readonly asserts: unknown;

  constructor(fields: NodeDescriptorFields & { code?: unknown; asserts?: unknown }) {
    super(fields);
    this.code = fields.code ?? null;
    this.asserts = fields.asserts ?? null;
  }

  /** Build a node shell carrying this descriptor's origin and `input:` wiring. */
  override build(_ctx: BuildContext): BuildResult {
    return this._finish(new StartNode(this.id), this.sinkWiring());
  }
}

/** A `case` stand-in taking `on`, `cases` and `else`, which it reads as `else_`. */
export class StandInCase extends NodeDescriptor {
  static override readonly kind = "case";
  static override readonly allowed = this.allow("on", "cases", "else");
  static override readonly keywordRemap: ReadonlyMap<string, string> = new Map([["else", "else_"]]);
  readonly on: unknown;
  readonly cases: unknown;
  readonly else_: unknown;

  constructor(fields: NodeDescriptorFields & { on?: unknown; cases?: unknown; else_?: unknown }) {
    super(fields);
    this.on = fields.on ?? null;
    this.cases = fields.cases ?? null;
    this.else_ = fields.else_ ?? null;
  }
}

/**
 * Return a one-off descriptor class of `kind` that adds `allow` and `require` to the base sets,
 * takes the other options as its statics, and keeps only the base fields.
 */
export function standIn(
  kind: string,
  options: {
    allow?: readonly string[];
    require?: readonly string[];
    bindingFields?: readonly string[];
    keywordRemap?: ReadonlyMap<string, string>;
    inputsAreLocals?: boolean;
  } = {},
): DescriptorClass {
  const {
    allow = [],
    require = [],
    bindingFields = [],
    keywordRemap = new Map(),
    inputsAreLocals = true,
  } = options;
  return class extends NodeDescriptor {
    static override readonly kind = kind;
    static override readonly allowed = this.allow(...allow);
    static override readonly required = this.require(...require);
    static override readonly bindingFields = bindingFields;
    static override readonly keywordRemap = keywordRemap;
    static override readonly inputsAreLocals = inputsAreLocals;
  };
}

/**
 * Run `fn` with `classes` registered in `KINDS` and `BINDING_FIELDS` under their kinds, then
 * restore the previous entries, also after a throw. `fn` must be synchronous: the tables are
 * restored in `finally`, before a returned promise settles.
 */
export function withKinds<T>(classes: readonly DescriptorClass[], fn: () => T): T {
  const kinds = KINDS as Map<string, DescriptorClass>;
  const bindingFields = BINDING_FIELDS as Map<string, readonly string[]>;
  const previous = classes.map(
    (cls) => [cls.kind, kinds.get(cls.kind), bindingFields.get(cls.kind)] as const,
  );
  for (const cls of classes) {
    kinds.set(cls.kind, cls);
    bindingFields.set(cls.kind, cls.bindingFields);
  }
  try {
    return fn();
  } finally {
    // In reverse, so a kind registered twice gets its first previous entry back.
    for (const [kind, cls, fields] of previous.toReversed()) {
      if (cls === undefined) kinds.delete(kind);
      else kinds.set(kind, cls);
      if (fields === undefined) bindingFields.delete(kind);
      else bindingFields.set(kind, fields);
    }
  }
}
