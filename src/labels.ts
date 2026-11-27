/**
 * The checks yian makes of the labels that name who speaks, as the child-flow toolkit makes them
 * and a host makes them of its own names before a run. A label is refused alone when it doesn't
 * read as it looks, or reads as a label only yian gives (`labelRefusal`). Two labels clash when
 * they read alike (`labelsAlike`): after NFKC normalization and case folding, or by their skeletons
 * in Unicode's confusables (UTS #39). `labelClashes` finds each label of a list that reads as an
 * earlier one, in linear time.
 *
 * Knows about: Unicode's confusables.
 */

import { CONFUSABLES } from "./confusables.ts";

/** Why yian refuses a label alone (`labelRefusal`). */
export type LabelRefusalCode =
  | "empty"
  | "invisible"
  | "whitespace"
  | "hidden_dot"
  | "double_mark"
  | "reserved"
  | "relay"
  | "child";

/** Why yian refuses a label alone: a code, and the reason yian's errors give after the label. */
export interface LabelRefusal {
  readonly code: LabelRefusalCode;
  /** What is wrong with the label, to follow it: `is reserved: nothing is signed yian or office`. */
  readonly reason: string;
}

/**
 * A form labels are compared by, each against the same form of the other: the label NFKC-normalized
 * and fully case-folded (`folded`), or its confusable skeleton as written (`skeleton`), case-folded
 * (`folded_skeleton`) or upper-cased (`upper_skeleton`).
 */
export type LabelForm = "folded" | "skeleton" | "folded_skeleton" | "upper_skeleton";

/** A label of a list that reads as an earlier one (`labelClashes`), by their places in the list. */
export interface LabelClash {
  /** The label's index. */
  readonly index: number;
  /** The index of the earlier label it reads as. */
  readonly readsAs: number;
  /** The form they read alike in. */
  readonly form: LabelForm;
}

/** The forms `labelForms` gives, in its order. */
const FORMS: readonly LabelForm[] = ["folded", "skeleton", "folded_skeleton", "upper_skeleton"];

/**
 * What a label may not hold, as it doesn't show or doesn't read as what it is: control, format and
 * default-ignorable characters (bidi controls among them), lone surrogates, the braille blank and
 * the Hangul fillers.
 */
const UNSEEN = /[\p{Cc}\p{Cf}\p{Cs}\p{Default_Ignorable_Code_Point}\p{Bidi_Control}⠀ᅟᅠㅤﾠ]/u;

/** Whitespace a label may not hold once NFKC-normalized: at an end, doubled, or other than a space. */
const LOOSE_SPACE = /^\s|\s$|\s\s|[^\S ]/u;

/**
 * A dot above an `i`, a `j` or another soft-dotted letter, or above a dotless `ı` or `ȷ`, with any
 * marks between, once decomposed: it doesn't show, or makes the dotless letter read as the other.
 */
const DOTTED_TWICE = /[\p{Soft_Dotted}ıȷ][\p{Mn}\p{Me}]*̇/u;

/** The same nonspacing mark twice on one character, once decomposed, which shows as once. */
const MARKED_TWICE = /([\p{Mn}\p{Me}])[\p{Mn}\p{Me}]*\1/u;

/** What no label may read as (`labelForms`): nothing is signed `yian` or `office`. */
const RESERVED: readonly string[] = ["yian", "office"];

/** A child's label (`<flow> r<n>`), as `labelKey` reads a label: one whose last word is `r` and digits. */
const CHILD_LABEL = childPattern("r", "\\p{Nd}");

/** A relay's label (`<person>, via <employee>`), as `labelKey` and `VIA_LOOKALIKES` read one: the word `via`. */
const RELAY_LABEL = relayPattern("via");

/** Letters of other scripts that look like `v`, `i` or `a`, read as those when looking for `via`. */
const VIA_LOOKALIKES: Readonly<Record<string, string>> = {
  "ν": "v", "ѵ": "v", "ᴠ": "v",
  "ı": "i", "ɩ": "i", "ι": "i", "і": "i", "ӏ": "i",
  "ɑ": "a", "α": "a", "а": "a",
};

/**
 * Why yian refuses `label` alone, or `null` when it doesn't: when it is empty, holds an `UNSEEN`
 * character, loose whitespace (`LOOSE_SPACE`) or a mark that doesn't show (`DOTTED_TWICE`,
 * `MARKED_TWICE`), or in any of its forms reads as a reserved label (`yian`, `office`), a relay's
 * (`<person>, via <employee>`) or a child's (`<flow> r<n>`), which only yian gives.
 */
export function labelRefusal(label: string): LabelRefusal | null {
  const refused = (code: LabelRefusalCode, reason: string): LabelRefusal => Object.freeze({ code, reason });
  if (label === "") return refused("empty", "is empty");
  const unseen = UNSEEN.exec(label)?.[0];
  if (unseen !== undefined) {
    const code = unseen.codePointAt(0)!.toString(16).toUpperCase().padStart(4, "0");
    return refused("invisible", `holds an invisible or control character (U+${code})`);
  }
  if (LOOSE_SPACE.test(label.normalize("NFKC"))) {
    return refused("whitespace", "has whitespace at an end, doubled, or other than a space");
  }
  const decomposed = label.normalize("NFD");
  if (DOTTED_TWICE.test(decomposed)) {
    return refused("hidden_dot", "puts a dot above an i or j, which reads as the plain letter");
  }
  if (MARKED_TWICE.test(decomposed)) {
    return refused("double_mark", "holds the same mark twice on one letter, which shows as once");
  }
  const forms = labelForms(label);
  if (RESERVED.some((reserved) => alikeIn(forms, labelForms(reserved)) !== null)) {
    return refused("reserved", "is reserved: nothing is signed yian or office");
  }
  const [key, ...skeletons] = forms as [string, ...string[]];
  const plain = Array.from(key, (char) => VIA_LOOKALIKES[char] ?? char).join("");
  const patterns = labelPatterns();
  if (RELAY_LABEL.test(plain) || skeletons.some((form, i) => patterns[i]!.relay.test(form))) {
    return refused("relay", "reads as a relay (<person>, via <employee>)");
  }
  if (CHILD_LABEL.test(key) || skeletons.some((form, i) => patterns[i]!.child.test(form))) {
    return refused("child", "reads as a child run's (<flow> r<n>)");
  }
  return null;
}

/** The first form (`LabelForm`) in which `label` and `other` read alike, or `null` for none. */
export function labelsAlike(label: string, other: string): LabelForm | null {
  return alikeIn(labelForms(label), labelForms(other));
}

/**
 * Each label of `labels` that reads as an earlier one, in order, with the earlier label it reads as:
 * in the first form, in `LabelForm`'s order, that an earlier label shares, the first label with it.
 * Each label's forms are computed once and looked up by value, so a long list takes linear time.
 */
export function labelClashes(labels: readonly string[]): LabelClash[] {
  // For each of `FORMS`, the index of the first label seen with each value of it.
  const seen = FORMS.map(() => new Map<string, number>());
  const clashes: LabelClash[] = [];
  labels.forEach((label, index) => {
    const forms = labelForms(label);
    const at = forms.findIndex((form, i) => seen[i]!.has(form));
    if (at >= 0) clashes.push(Object.freeze({ index, readsAs: seen[at]!.get(forms[at]!)!, form: FORMS[at]! }));
    forms.forEach((form, i) => {
      if (!seen[i]!.has(form)) seen[i]!.set(form, index);
    });
  });
  return clashes;
}

/** A child's label: its last word is `r` then digits, as the pattern sources `r` and `digit` match them. */
function childPattern(r: string, digit: string): RegExp {
  return new RegExp(`(^|\\s)${r}(?:${digit})+$`, "u");
}

/** A relay's label: the word `via`, as the pattern source `via` matches it, with no letter or digit beside it. */
function relayPattern(via: string): RegExp {
  return new RegExp(`(^|[^\\p{L}\\p{N}])${via}($|[^\\p{L}\\p{N}])`, "u");
}

/** `text` as a pattern source that matches it literally. */
function literal(text: string): string {
  return text.replace(/[\\^$.*+?()[\]{}|/]/g, "\\$&");
}

/** `label` NFKC-normalized and fully case-folded (`ß` as `ss`), as the label patterns read it. */
function labelKey(label: string): string {
  // Upper then lower case folds as full case folding does.
  return label.normalize("NFKC").toUpperCase().toLowerCase().normalize("NFKC");
}

/** Each character's prototype in `CONFUSABLES`, by code point, decoded on first use. */
let prototypes: Map<number, string> | undefined;

/**
 * `text`'s confusable skeleton (UTS #39's internal skeleton): decomposed, without default-ignorable
 * characters, each character as its prototype in Unicode's confusables, and decomposed again. Two
 * texts with the same skeleton look alike.
 */
function skeleton(text: string): string {
  if (prototypes === undefined) {
    prototypes = new Map();
    let source = 0;
    for (const entry of CONFUSABLES.trim().split(/\s+/)) {
      const [step, target] = entry.split(":") as [string, string];
      source += parseInt(step, 36);
      prototypes.set(source, String.fromCodePoint(...target.split(".").map((code) => parseInt(code, 36))));
    }
  }
  const table = prototypes;
  const plain = text.normalize("NFD").replace(/\p{Default_Ignorable_Code_Point}/gu, "");
  return Array.from(plain, (char) => table.get(char.codePointAt(0)!) ?? char).join("").normalize("NFD");
}

/**
 * The forms `label` is compared by, in `FORMS`' order: its `labelKey`, and its `skeleton` as
 * written, case-folded and upper-cased, so a label that looks like another in either case reads as
 * it.
 */
function labelForms(label: string): readonly string[] {
  const key = labelKey(label);
  return [key, skeleton(label), skeleton(key), skeleton(label.normalize("NFKC").toUpperCase())];
}

/** The first form in which labels of `forms` and `theirs` (`labelForms`) read alike, or `null`. */
function alikeIn(forms: readonly string[], theirs: readonly string[]): LabelForm | null {
  const at = forms.findIndex((form, i) => form === theirs[i]);
  return at < 0 ? null : FORMS[at]!;
}

/** `labelPatterns`, built on first use. */
let skeletonPatterns: readonly { readonly relay: RegExp; readonly child: RegExp }[] | undefined;

/**
 * `RELAY_LABEL` and `CHILD_LABEL` for each of `labelForms`' skeletons: `via`, `r` and the digits as
 * that skeleton writes them, so a lookalike of a relay's or a child's label reads as one.
 */
function labelPatterns(): readonly { readonly relay: RegExp; readonly child: RegExp }[] {
  if (skeletonPatterns === undefined) {
    const written = (text: string) => labelForms(text).slice(1).map(literal);
    const [via, r] = [written("via"), written("r")];
    const digits = Array.from("0123456789", written);
    skeletonPatterns = via.map((word, i) => ({
      relay: relayPattern(word),
      child: childPattern(r[i]!, digits.map((each) => each[i]!).join("|")),
    }));
  }
  return skeletonPatterns;
}
