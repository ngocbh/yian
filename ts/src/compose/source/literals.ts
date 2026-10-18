/** Parse isolated YAML literals outside source-document ownership. */

import { SourceSyntaxError } from "./common.ts";
import {
  constructScalar,
  FlowLoaderError,
  MERGE_TAG,
  resolvePlain,
  ScalarValueError,
  type ScalarValue,
} from "./scalars.ts";

/**
 * Read one plain scalar's text with the flow loader's scalar rules.
 *
 * The text is one stripped, single-line plain scalar, never YAML structure: the flow
 * loader's table picks its tag and builds the value, so `1e3` is a float and `~` is null
 * as in a source document, while `[1]`, `a: b`, or `&a` stay text. Text that spans lines,
 * and `<<`, are returned as they are.
 *
 * Example: `readPlainScalar("0x1F")` is `31n`, and `readPlainScalar("yes")` is `"yes"`.
 *
 * @throws SourceSyntaxError when the resolved tag can't build the text, such as an
 *   impossible date; the message is the reason, without the text or the quoting fix.
 */
export function readPlainScalar(text: string): ScalarValue {
  if (text.includes("\n")) return text; // one line only
  const tag = resolvePlain(text);
  if (tag === MERGE_TAG) return text; // `<<` merges only as a mapping key; as a value it is text
  try {
    return constructScalar(tag, text, true);
  } catch (error) {
    // The caller adds the quoting fix.
    if (error instanceof ScalarValueError) throw new SourceSyntaxError(error.reason);
    if (error instanceof FlowLoaderError) throw new SourceSyntaxError(error.message);
    throw error;
  }
}
