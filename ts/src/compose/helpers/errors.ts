/** Canonical, source-owned diagnostics for Compose load failures. */

import { Diagnostic, DiagnosticFrame, SourceRange } from "../../diagnostics.ts";
import type { SourceCursor, SourceDocument } from "../source/index.ts";

/** Where a load diagnostic points: a primary range or explicit frames, and notes. */
export interface LoadDiagnosticOptions {
  /** Primary location in the context's source. Mutually exclusive with `frames`. */
  range?: SourceRange | null;
  /** Explicit ordered frames. Mutually exclusive with `range`. */
  frames?: Iterable<DiagnosticFrame>;
  /** Explanatory context not owned by one frame. */
  notes?: Iterable<string>;
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
   * @throws TypeError when either is not text.
   */
  constructor(label: string | null = null, text: string | null = null) {
    if ((label === null) !== (text === null)) {
      throw new RangeError("load diagnostic source requires both label and text");
    }
    if (label !== null && typeof label !== "string") {
      throw new TypeError("load diagnostic label must be a string");
    }
    if (text !== null && typeof text !== "string") {
      throw new TypeError("load diagnostic text must be a string");
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
   *
   * @throws TypeError for a range that isn't a `SourceRange`, or a note that isn't text.
   */
  frame(range: SourceRange | null, options: { note?: string | null } = {}): DiagnosticFrame | null {
    const { note = null } = options;
    if (range === null || this.label === null || this.text === null) return null;
    if (!(range instanceof SourceRange)) {
      throw new TypeError("load diagnostic range must be a SourceRange or null");
    }
    if (note !== null && typeof note !== "string") {
      throw new TypeError("load diagnostic frame note must be a string or null");
    }
    return new DiagnosticFrame({ label: this.label, text: this.text, range, note });
  }

  /**
   * Construct one canonical Compose load diagnostic, code `compose.load`.
   *
   * @throws TypeError for a malformed message, frame or note.
   * @throws RangeError when `range` and explicit frames are both given.
   */
  diagnostic(message: string, options: LoadDiagnosticOptions = {}): Diagnostic {
    const { range = null, frames = [], notes = [] } = options;
    if (typeof message !== "string") {
      throw new TypeError("load diagnostic message must be a string");
    }
    let ownedFrames = [...frames];
    if (range !== null && ownedFrames.length > 0) {
      throw new RangeError("load diagnostic accepts a range or frames, not both");
    }
    for (const frame of ownedFrames) {
      if (!(frame instanceof DiagnosticFrame)) {
        throw new TypeError("load diagnostic frames must be DiagnosticFrame values");
      }
    }
    if (range !== null) {
      const frame = this.frame(range);
      ownedFrames = frame !== null ? [frame] : [];
    }
    const ownedNotes = [...notes];
    if (ownedNotes.some((note) => typeof note !== "string")) {
      throw new TypeError("load diagnostic notes must be strings");
    }
    return new Diagnostic({
      code: "compose.load",
      message,
      frames: ownedFrames,
      notes: ownedNotes,
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
      throw new TypeError("load diagnostic context requires a LoadError");
    }
    const inner = error.diagnostic;
    const outer = this.frame(range, { note });
    const diagnostic = new Diagnostic({
      code: inner.code,
      message: message ?? inner.message,
      frames: outer !== null ? [outer, ...inner.frames] : inner.frames,
      notes: inner.notes,
      nodeId: inner.nodeId,
      causeType: inner.causeType,
      traceback: inner.traceback,
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
  const text = diagnostics.text;
  if (text === null || range === null || range.startLine !== range.endLine) return null;
  // Lines break as ranges count them (`rule: columns`); no empty line after a final break.
  const lines = text.split(/\r\n|\r|\n/);
  if (lines[lines.length - 1] === "") lines.pop();
  if (!(1 <= range.startLine && range.startLine <= lines.length)) return null;
  return lines[range.startLine - 1]!.slice(range.startCol - 1, range.endCol - 1).trim() || null;
}

/** A Compose flow cannot be loaded, expressed as one canonical diagnostic. */
export class LoadError extends RangeError {
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
      throw new TypeError("LoadError requires a Diagnostic");
    }
    const { cause } = options;
    // `{ cause: undefined }` would still install an own `cause` property.
    super(diagnostic.message, cause === undefined ? undefined : { cause });
    this.diagnostic = diagnostic;
    Object.freeze(this);
  }
}
