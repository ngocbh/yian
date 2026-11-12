/**
 * Canonical, source-owned diagnostics for Compose load failures, and the helpers the validate
 * passes share to build them: `locatedError` points an error at a binding's origin,
 * `rejectReservedName` refuses an authored name starting with `__`, `rejectRemovedMemoryKey`
 * refuses a removed memory key with its fix, and `shownLiteral` writes a value the way messages
 * show it.
 */

import { Diagnostic, DiagnosticFrame, type OriginRef, reprText, type SourceRange } from "../../diagnostics.ts";
import { YianRangeError, YianTypeError } from "../../typesys/errors.ts";
import { type Value, valueRepr, writeAsJson } from "../../typesys/values.ts";
import type { SourceCursor, SourceDocument } from "../source/index.ts";
import { oneLineSlice } from "./source.ts";

/** Where a load diagnostic points: a primary range or explicit frames, and notes. */
export interface LoadDiagnosticOptions {
  /** Primary location in the context's source. Mutually exclusive with `frames`. */
  range?: SourceRange | null;
  /** Explicit ordered frames. Mutually exclusive with `range`. */
  frames?: Iterable<DiagnosticFrame>;
  /** Explanatory context not owned by one frame. */
  notes?: Iterable<string>;
  /** The diagnostic code; `compose.load` unless a narrower one applies. */
  code?: string;
}

/**
 * Builds immutable load diagnostics owned by one Compose source document.
 *
 * The context holds only the presentation data already retained by a parsed document. A
 * source-free context still constructs valid unlocated diagnostics, which keeps direct
 * composition independent of YAML.
 */
export class LoadDiagnostics {
  /** Human-facing source name; `null` for an unlocated context. */
  readonly label: string | null;
  /** Complete source text indexed by ranges; `null` for an unlocated context. */
  readonly text: string | null;

  /**
   * Create a context owning `label` and `text`, or an unlocated one when both are `null`.
   *
   * @throws RangeError when only one of them is `null`.
   */
  constructor(label: string | null = null, text: string | null = null) {
    if ((label === null) !== (text === null)) {
      throw new YianRangeError("load diagnostic source requires both label and text");
    }
    this.label = label;
    this.text = text;
    Object.freeze(this);
  }

  /** Return a context owning `document`'s label and text, or an unlocated one for `null`. */
  static fromDocument(document: SourceDocument | null): LoadDiagnostics {
    if (document === null) return new LoadDiagnostics();
    return new LoadDiagnostics(document.label, document.text);
  }

  /**
   * Return a context for the cursor's document; a virtual or absent cursor gives an
   * unlocated one.
   */
  static fromCursor(cursor: SourceCursor | null): LoadDiagnostics {
    return LoadDiagnostics.fromDocument(cursor?.document ?? null);
  }

  /**
   * Resolve one concrete range against this context's source, with an optional `note`
   * printed next to it; `null` when the range or the source is absent.
   */
  frame(range: SourceRange | null, options: { note?: string | null } = {}): DiagnosticFrame | null {
    const { note = null } = options;
    if (range === null || this.label === null || this.text === null) return null;
    return new DiagnosticFrame({ label: this.label, text: this.text, range, note });
  }

  /**
   * Construct one canonical Compose load diagnostic, code `compose.load` unless `code` names a
   * narrower one, such as `expr.nullable_field_read`.
   *
   * @throws RangeError when `range` and explicit frames are both given.
   */
  diagnostic(message: string, options: LoadDiagnosticOptions = {}): Diagnostic {
    const { range = null, frames = [], notes = [], code = "compose.load" } = options;
    let ownedFrames = [...frames];
    if (range !== null && ownedFrames.length > 0) {
      throw new YianRangeError("load diagnostic accepts a range or frames, not both");
    }
    if (range !== null) {
      const frame = this.frame(range);
      ownedFrames = frame !== null ? [frame] : [];
    }
    return new Diagnostic({
      code,
      message,
      frames: ownedFrames,
      notes: [...notes],
    });
  }

  /**
   * Construct the one error type crossing the Compose boundary, carrying the canonical
   * diagnostic and the `cause` that led to it, if any.
   */
  error(message: string, options: LoadDiagnosticOptions & { cause?: unknown } = {}): LoadError {
    const { cause, ...where } = options;
    return new LoadError(this.diagnostic(message, where), { cause });
  }

  /**
   * Return a fresh error with one outer source context frame before the inner frames.
   *
   * The inner `error` is never mutated. `message` replaces the inner message (`null` keeps
   * it), `note` attaches to the outer frame, and `cause` becomes the new error's cause.
   *
   * @throws TypeError when `error` is not a `LoadError`.
   */
  prepend(
    error: LoadError,
    options: {
      range: SourceRange | null;
      message?: string | null;
      note?: string | null;
      cause?: unknown;
    },
  ): LoadError {
    const { range, message = null, note = null, cause } = options;
    if (!(error instanceof LoadError)) {
      throw new YianTypeError("load diagnostic context requires a LoadError");
    }
    const inner = error.diagnostic;
    const outer = this.frame(range, { note });
    const diagnostic = new Diagnostic({
      ...inner,
      message: message ?? inner.message,
      frames: outer !== null ? [outer, ...inner.frames] : inner.frames,
    });
    return new LoadError(diagnostic, { cause });
  }
}

/**
 * Return the authored text a one-line range covers, stripped; `null` when the source or
 * range is absent, the range spans several lines or lies past the text, or it covers only
 * whitespace.
 */
export function sourceText(diagnostics: LoadDiagnostics, range: SourceRange | null): string | null {
  return oneLineSlice(diagnostics.text, range)?.trim() || null;
}

/** A Compose flow cannot be loaded, expressed as one canonical diagnostic. */
export class LoadError extends YianRangeError {
  override name = "LoadError";
  /** The immutable, presentation-ready failure payload. */
  readonly diagnostic: Diagnostic;

  /**
   * Retain `diagnostic` as the error's only payload, its message as the error's, and
   * `cause` when given; the error is frozen.
   *
   * @throws TypeError when `diagnostic` is not a `Diagnostic`.
   */
  constructor(diagnostic: Diagnostic, options: { cause?: unknown } = {}) {
    if (!(diagnostic instanceof Diagnostic)) {
      throw new YianTypeError("LoadError requires a Diagnostic");
    }
    const { cause } = options;
    // `{ cause: undefined }` would still install an own `cause` property.
    super(diagnostic.message, cause === undefined ? undefined : { cause });
    this.diagnostic = diagnostic;
    Object.freeze(this);
  }
}

/**
 * Reject an authored name that starts with `__`. yian generates such names (a node's reserved
 * wiring keys, synthesized nodes and END inputs), so an author's could collide with one.
 *
 * @param what What the name is, for the message, such as `node id` or `node 'a' input`.
 * @param name The authored name.
 * @param range Where the error points.
 * @param diagnostics Factory for located load errors.
 * @throws LoadError when `name` starts with `__`.
 */
export function rejectReservedName(
  what: string,
  name: string,
  range: SourceRange | null,
  diagnostics: LoadDiagnostics,
): void {
  if (name.startsWith("__")) {
    throw diagnostics.error(
      `${what} ${reprText(name)} starts with '__', which yian keeps for the names it generates; rename it`,
      { range },
    );
  }
}

/**
 * Refuse a removed memory key, `long_term_memory:` or its older spelling `memory_policy:`, at the
 * key, with the fix: memories are flow inputs now.
 *
 * @param has Whether the flow, def or node body being read holds a key.
 * @param cursor The body's authored occurrence, which locates the key; `null` when source-free.
 * @param prefix Where the body is, as the site's other errors start (`defs entry 'x': `).
 * @throws LoadError when the body holds either key.
 */
export function rejectRemovedMemoryKey(
  has: (key: string) => boolean,
  cursor: SourceCursor | null,
  diagnostics: LoadDiagnostics,
  prefix = "",
): void {
  for (const key of ["long_term_memory", "memory_policy"]) {
    if (!has(key)) continue;
    throw diagnostics.error(
      `${prefix}\`${key}:\` was removed: declare the memory as an input (\`lessons: !FileMemory\`) ` +
        "and render `${lessons}` in a prompt",
      { range: cursor?.keyRangeOf(key) ?? null },
    );
  }
}

/**
 * Build the load error, located at the binding's origin when it resolves, code `compose.load`
 * unless `options.code` names a narrower one, such as `expr.nullable_field_read`.
 *
 * @param message The error message.
 * @param binding The rejected binding, or any carrier of the origin to locate at.
 * @param fallback The location used when the origin does not resolve.
 * @param resolveOrigin Resolves a registered origin to its source frame.
 * @param diagnostics Factory for located load errors.
 * @param options.code The diagnostic code, when narrower than `compose.load`.
 * @returns The located error.
 */
export function locatedError(
  message: string,
  binding: { readonly origin: OriginRef | null },
  fallback: SourceRange | null,
  resolveOrigin: ((origin: OriginRef) => DiagnosticFrame | null) | null,
  diagnostics: LoadDiagnostics,
  options: { code?: string } = {},
): LoadError {
  const { code } = options;
  const frame = resolveOrigin !== null && binding.origin !== null ? resolveOrigin(binding.origin) : null;
  if (frame !== null) return diagnostics.error(message, { frames: [frame], code });
  return diagnostics.error(message, { range: fallback, code });
}

/**
 * Render a literal as its text form: compact JSON (`true`, `null`, `"text"`, `[1,"a"]`,
 * `{"k":2}`), the form messages use for values.
 *
 * @param value The literal to render.
 * @returns `writeAsJson`'s text; `valueRepr` for anything it can't write.
 */
export function shownLiteral(value: unknown): string {
  try {
    return writeAsJson(value as Value);
  } catch (error) {
    if (error instanceof TypeError) return valueRepr(value);
    throw error;
  }
}
