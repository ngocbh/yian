/**
 * The one accepted string form of `date` and `datetime` values.
 *
 * The grammar is spelled out here instead of delegated to `Date.parse`, whose accepted
 * set differs between runtimes, so a string parses the same way everywhere.
 */

const DATE_FORMAT = "2026-05-21";
const DATETIME_FORMAT = "2026-05-21T14:30[:00][Z|+07:00]";

// `[0-9]` rather than `\d` to say ASCII digits outright; no `m` flag, so `$` is the end
// of the text and a trailing newline fails.
const DATE_RE = /^([0-9]{4})-([0-9]{2})-([0-9]{2})$/;
const DATETIME_RE = new RegExp(
  "^([0-9]{4})-([0-9]{2})-([0-9]{2})[T ]([0-9]{2}):([0-9]{2})" +
    "(?::([0-9]{2})(?:\\.([0-9]{1,6}))?)?" +
    "(Z|[+-][0-9]{2}:[0-9]{2})?$",
);

/** A parsed `date`. */
export interface ParsedDate {
  readonly kind: "date";
  readonly year: number;
  readonly month: number;
  readonly day: number;
}

/** A parsed `datetime`; `offsetMinutes` is `null` when the text carries no offset. */
export interface ParsedDatetime {
  readonly kind: "datetime";
  readonly year: number;
  readonly month: number;
  readonly day: number;
  readonly hour: number;
  readonly minute: number;
  readonly second: number;
  readonly microsecond: number;
  readonly offsetMinutes: number | null;
}

/**
 * Parse a `date` string of the form `YYYY-MM-DD`; surrounding whitespace is not allowed.
 *
 * Example: `parseDate("2026-05-21")` returns `{kind: "date", year: 2026, month: 5, day: 21}`.
 *
 * @throws RangeError when `text` does not match the grammar, or names an impossible
 *   date such as `2026-02-30`.
 */
export function parseDate(text: string): ParsedDate {
  const match = DATE_RE.exec(text);
  if (match === null) throw new RangeError(`date must look like ${DATE_FORMAT}`);
  return dateFrom(match);
}

/**
 * Parse a `datetime` string of the form `YYYY-MM-DDTHH:MM[:SS[.ffffff]][Z|±HH:MM]`;
 * surrounding whitespace is not allowed.
 *
 * The separator may be `T` or a space. A fraction of one to six digits is read as the
 * leading digits of the microseconds, so `.5` is 500000 µs. `Z` is UTC. Without an
 * offset, `offsetMinutes` is `null`.
 *
 * Example: `parseDatetime("2026-05-21T14:30Z")` returns 2026-05-21 14:30 with
 * `offsetMinutes` 0.
 *
 * @throws RangeError when `text` does not match the grammar, or names an impossible
 *   value such as hour 24 or offset `+07:60`.
 */
export function parseDatetime(text: string): ParsedDatetime {
  const match = DATETIME_RE.exec(text);
  if (match === null) throw new RangeError(`datetime must look like ${DATETIME_FORMAT}`);
  return datetimeFrom(match);
}

/**
 * Parse `text` as a `date` when it has the date shape, otherwise as a `datetime`.
 *
 * Example: `parseDateOrDatetime("2026-05-21")` returns a `ParsedDate`, and
 * `parseDateOrDatetime("2026-05-21T14:30")` returns a `ParsedDatetime`.
 *
 * @throws RangeError when `text` matches neither grammar, or names an impossible value.
 */
export function parseDateOrDatetime(text: string): ParsedDate | ParsedDatetime {
  const dateMatch = DATE_RE.exec(text);
  if (dateMatch !== null) return dateFrom(dateMatch);
  const datetimeMatch = DATETIME_RE.exec(text);
  if (datetimeMatch !== null) return datetimeFrom(datetimeMatch);
  throw new RangeError(`must be a date (${DATE_FORMAT}) or datetime (${DATETIME_FORMAT})`);
}

/** The duration forms, as messages show them. */
export const DURATION_FORMAT = "90s | 30m | 24h | 2d";

const DURATION_RE = /^([0-9]+)([smhd])$/;
const UNIT_SECONDS: Readonly<Record<string, number>> = { s: 1, m: 60, h: 3600, d: 86400 };

/**
 * Return the seconds in a duration: a positive integer then `s`, `m`, `h` or `d` (`24h` is 86400).
 *
 * @throws TypeError for a value that isn't text.
 * @throws RangeError for any other text: plain number text (its hint spells it in seconds when
 *   that parses), a zero duration, or one past `Number.MAX_SAFE_INTEGER` seconds.
 */
export function parseDuration(text: string): number {
  if (typeof text !== "string") {
    throw new TypeError(`a duration is text: a positive integer and a unit (${DURATION_FORMAT})`);
  }
  const match = DURATION_RE.exec(text);
  const seconds = match === null ? 0 : Number(match[1]) * UNIT_SECONDS[match[2]!]!;
  if (match === null || seconds === 0 || !Number.isSafeInteger(seconds)) {
    // A plain number is the likely slip; `0` gets no hint since `0s` is refused too, and nor does
    // a number too large to parse in seconds.
    const plain = /^[0-9]+$/.test(text) ? Number(text) : 0;
    const hint = plain > 0 && Number.isSafeInteger(plain) ? `; write ${text}s` : "";
    throw new RangeError(`a duration is a positive integer and a unit (${DURATION_FORMAT})${hint}`);
  }
  return seconds;
}

/**
 * Parse one date/datetime string and return its UTC instant in epoch milliseconds. A date and a
 * datetime without an offset are read in UTC; microseconds are truncated to milliseconds.
 *
 * @throws RangeError when `text` is outside the grammar or impossible, or the instant's UTC year
 *   is outside 1 to 9999.
 */
export function deadlineUtc(text: string): number {
  const parsed = parseDateOrDatetime(text);
  // `Date.UTC` maps years 0-99 to 1900-1999; the setters take the year as given.
  const instant = new Date(0);
  instant.setUTCFullYear(parsed.year, parsed.month - 1, parsed.day);
  if (parsed.kind === "datetime") {
    const offset = parsed.offsetMinutes ?? 0;
    instant.setUTCHours(parsed.hour, parsed.minute - offset, parsed.second, Math.floor(parsed.microsecond / 1000));
  }
  // A JS Date holds years 0 and 10000; Python's datetime doesn't, and refuses with this text.
  const year = instant.getUTCFullYear();
  if (year < 1 || year > 9999) throw new RangeError("date value out of range");
  return instant.getTime();
}

/** A record time's form: `Date.prototype.toISOString`'s, UTC with milliseconds and a four-digit year. */
export const RECORD_TIME = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})\.(\d{3})Z$/;

/**
 * Return epoch milliseconds `ms` as a record time such as `2026-10-20T14:02:11.123Z`; a fraction
 * of a millisecond is dropped.
 *
 * @throws RangeError when `ms` isn't finite or falls outside the years 0000 to 9999.
 */
export function formatRecordTime(ms: number): string {
  const date = new Date(ms);
  const text = Number.isNaN(date.getTime()) ? "" : date.toISOString();
  if (!RECORD_TIME.test(text)) throw new RangeError(`run-log clock time ${ms} is outside the years 0000 to 9999`);
  return text;
}

function dateFrom(match: RegExpExecArray): ParsedDate {
  const [, year, month, day] = match;
  const parsed: ParsedDate = {
    kind: "date",
    year: Number(year),
    month: Number(month),
    day: Number(day),
  };
  const reason = dateProblem(parsed.year, parsed.month, parsed.day);
  if (reason !== null) throw new RangeError(`not a valid date: ${reason}`);
  return parsed;
}

function datetimeFrom(match: RegExpExecArray): ParsedDatetime {
  const [, year, month, day, hour, minute, second, fraction, offset] = match;
  let offsetMinutes: number | null = null;
  if (offset === "Z") {
    offsetMinutes = 0;
  } else if (offset !== undefined) {
    const offHours = Number(offset.slice(1, 3));
    const offMinutes = Number(offset.slice(4, 6));
    if (offHours >= 24) {
      throw new RangeError("not a valid datetime: UTC offset must be under 24:00");
    }
    if (offMinutes >= 60) {
      throw new RangeError("not a valid datetime: UTC offset minutes must be under 60");
    }
    const magnitude = offHours * 60 + offMinutes;
    // `-00:00` is UTC; `- 0` would give `-0`.
    offsetMinutes = offset[0] === "-" && magnitude !== 0 ? -magnitude : magnitude;
  }
  const parsed: ParsedDatetime = {
    kind: "datetime",
    year: Number(year),
    month: Number(month),
    day: Number(day),
    hour: Number(hour),
    minute: Number(minute),
    second: Number(second ?? 0),
    microsecond: Number((fraction ?? "0").padEnd(6, "0")),
    offsetMinutes,
  };
  const reason =
    dateProblem(parsed.year, parsed.month, parsed.day) ??
    timeProblem(parsed.hour, parsed.minute, parsed.second);
  if (reason !== null) throw new RangeError(`not a valid datetime: ${reason}`);
  return parsed;
}

// The reasons below are the messages of Python's `date()` and `datetime()` errors.

/** Return why year, month and day name no date, or `null` if they do. */
function dateProblem(year: number, month: number, day: number): string | null {
  if (year < 1 || year > 9999) return `year ${year} is out of range`;
  if (month < 1 || month > 12) return "month must be in 1..12";
  if (day < 1 || day > daysInMonth(year, month)) return "day is out of range for month";
  return null;
}

/** Return why hour, minute and second name no time of day, or `null` if they do. */
function timeProblem(hour: number, minute: number, second: number): string | null {
  if (hour > 23) return "hour must be in 0..23";
  if (minute > 59) return "minute must be in 0..59";
  if (second > 59) return "second must be in 0..59";
  return null;
}

function daysInMonth(year: number, month: number): number {
  if (month === 2) return year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0) ? 29 : 28;
  return [4, 6, 9, 11].includes(month) ? 30 : 31;
}
