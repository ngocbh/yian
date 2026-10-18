/**
 * The one accepted string form of `date` and `datetime` values.
 *
 * The grammar is spelled out here instead of delegated to `Date.parse`, whose accepted
 * set differs between runtimes, so a string parses the same way everywhere.
 */

export const DATE_FORMAT = "2026-05-21";
export const DATETIME_FORMAT = "2026-05-21T14:30[:00][Z|+07:00]";

// `[0-9]` rather than `\d` to say ASCII digits outright; no `m` flag, so `$` is the end
// of the text and a trailing newline fails.
const DATE_RE = /^([0-9]{4})-([0-9]{2})-([0-9]{2})$/;
const DATETIME_RE =
  /^([0-9]{4})-([0-9]{2})-([0-9]{2})[T ]([0-9]{2}):([0-9]{2})(?::([0-9]{2})(?:\.([0-9]{1,6}))?)?(Z|[+-][0-9]{2}:[0-9]{2})?$/;

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

function dateFrom(match: RegExpExecArray): ParsedDate {
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const reason = dateProblem(year, month, day);
  if (reason !== null) throw new RangeError(`not a valid date: ${reason}`);
  return { kind: "date", year, month, day };
}

function datetimeFrom(match: RegExpExecArray): ParsedDatetime {
  const [, year, month, day, hour, minute, second, fraction, offset] = match;
  let offsetMinutes: number | null = null;
  if (offset === "Z") {
    offsetMinutes = 0;
  } else if (offset !== undefined) {
    const offHours = Number(offset.slice(1, 3));
    const offMinutes = Number(offset.slice(4, 6));
    if (offHours >= 24) throw new RangeError("not a valid datetime: UTC offset must be under 24:00");
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
    (parsed.hour > 23
      ? "hour must be in 0..23"
      : parsed.minute > 59
        ? "minute must be in 0..59"
        : parsed.second > 59
          ? "second must be in 0..59"
          : null);
  if (reason !== null) throw new RangeError(`not a valid datetime: ${reason}`);
  return parsed;
}

/** Return Python's reason why year, month and day name no date, or `null` if they do. */
function dateProblem(year: number, month: number, day: number): string | null {
  if (year < 1 || year > 9999) return `year ${year} is out of range`;
  if (month < 1 || month > 12) return "month must be in 1..12";
  if (day < 1 || day > daysInMonth(year, month)) return "day is out of range for month";
  return null;
}

function daysInMonth(year: number, month: number): number {
  if (month === 2) return year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0) ? 29 : 28;
  return [4, 6, 9, 11].includes(month) ? 30 : 31;
}
