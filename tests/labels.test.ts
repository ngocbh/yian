/** The label checks a host makes before a run, and the child-flow toolkit makes through them. */

import { describe, expect, test } from "bun:test";

import { labelClashes, type LabelForm, labelRefusal, type LabelRefusalCode, labelsAlike } from "../src/labels.ts";

describe("labelRefusal", () => {
  test("a label that reads as it looks, and as no label only yian gives, isn't refused", () => {
    for (const label of ["yian-chat", "the user", "Olivia", "İpek", "Zoë", "Ané̀", "shell 1"]) {
      expect(labelRefusal(label)).toBeNull();
    }
  });

  test("each rule refuses with its code and the reason yian's errors give", () => {
    const refused: [string, LabelRefusalCode, string][] = [
      ["", "empty", "is empty"],
      ["yian​-chat", "invisible", "holds an invisible or control character (U+200B)"],
      ["theㅤuser", "invisible", "holds an invisible or control character (U+3164)"],
      ["yian  chat", "whitespace", "has whitespace at an end, doubled, or other than a space"],
      ["yian chat", "whitespace", "has whitespace at an end, doubled, or other than a space"],
      ["Oli̇via", "hidden_dot", "puts a dot above an i or j, which reads as the plain letter"],
      ["ı̇an", "hidden_dot", "puts a dot above an i or j, which reads as the plain letter"],
      ["Zoë̈", "double_mark", "holds the same mark twice on one letter, which shows as once"],
      ["Office", "reserved", "is reserved: nothing is signed yian or office"],
      ["уіаn", "reserved", "is reserved: nothing is signed yian or office"],
      ["oſſice", "reserved", "is reserved: nothing is signed yian or office"],
      ["Ada (via Bob)", "relay", "reads as a relay (<person>, via <employee>)"],
      ["Bob, ∨ia Ada", "relay", "reads as a relay (<person>, via <employee>)"],
      ["Review_Flow R2", "child", "reads as a child run's (<flow> r<n>)"],
      ["greet г3", "child", "reads as a child run's (<flow> r<n>)"],
      ["greet r1O", "child", "reads as a child run's (<flow> r<n>)"],
    ];
    for (const [label, code, reason] of refused) expect(labelRefusal(label)).toEqual({ code, reason });
  });
});

describe("labelsAlike", () => {
  test("two labels read alike in the first form they share", () => {
    const pairs: [string, string, LabelForm][] = [
      ["the user", "The User", "folded"],
      ["Strauß", "STRAUSS", "folded"],
      ["Ada", "ＡＤＡ", "folded"],
      ["lena", "Іena", "skeleton"],
      ["yian-chat", "уіаn-сhаt", "skeleton"],
      ["Lucy", "lυcy", "folded_skeleton"],
      ["Kim", "ΚΙΜ", "upper_skeleton"],
    ];
    for (const [label, other, form] of pairs) {
      expect(labelsAlike(label, other)).toBe(form);
      expect(labelsAlike(other, label)).toBe(form);
    }
  });

  test("labels that don't look alike, or look alike only in the other case, don't read alike", () => {
    for (const [label, other] of [["Ian", "Lan"], ["Ada", "ΑΔΑ"], ["Bob", "Ьob"], ["Fiona", "ſiona"]]) {
      expect(labelsAlike(label!, other!)).toBeNull();
    }
  });
});

describe("labelClashes", () => {
  test("each label that reads as an earlier one is listed with the first it reads as, in order", () => {
    const labels = ["Ada", "Kim", "Bob", "ada", "ΚΙΜ", "ADA", "Ian", "Lan"];
    expect(labelClashes(labels)).toEqual([
      { index: 3, readsAs: 0, form: "folded" },
      { index: 4, readsAs: 1, form: "upper_skeleton" },
      { index: 5, readsAs: 0, form: "folded" },
    ]);
    expect(labelClashes([])).toEqual([]);
    expect(labelClashes(["Ada", "Bob"])).toEqual([]);
  });

  test("a list of 2,000 names is checked in linear time", () => {
    // Comparing each name with every earlier one would be quadratic. Compare 8x the names against
    // the fastest of three runs each, which keeps the check about growth, not the machine's speed.
    const fastest = (count: number) => {
      // Half the names, each in capitals: one clash each.
      const names = Array.from({ length: count / 2 }, (_, i) => `Person ${i} Ölçek`);
      const labels = [...names, ...names.map((name) => name.toUpperCase())];
      let best = Infinity;
      for (let i = 0; i < 3; i++) {
        const start = performance.now();
        expect(labelClashes(labels)).toHaveLength(count / 2);
        best = Math.min(best, performance.now() - start);
      }
      return best;
    };
    const small = fastest(2000);
    // Linear growth gives about 8x; quadratic gives about 64x.
    expect(fastest(16000) / Math.max(small, 1)).toBeLessThan(24);
  });
});
