"""Write python_builtins.json: what CPython 3.12+ gives for the expression builtins yian mirrors.

Run from the repository root with CPython 3.12 or newer (3.12 changed `sum` of floats):

    python3.12 -I tests/fixtures/python_builtins.py

tests/expr/builtins.test.ts checks yian against every row. A float is written as its `repr`
(`inf`, `-0.0`, `1e+50`), an int as `{"int": "<digits>"}`, and a failure as `{"error": "<class>"}`.
The draws are seeded, so the file only changes when this script does.
"""

import json
import os
import random
import struct
import sys

assert sys.version_info >= (3, 12), "sum() of floats changed in 3.12; run this with 3.12 or newer"

rng = random.Random(1541)

# Python's whitespace set and characters near it.
WS = [chr(c) for c in range(0x110000) if chr(c).isspace()]
NEAR = ["​", "﻿", "᠎", "⁠", "­", "x", "\U0001f600", "\U00010000", "￿", "é"]


def enc(value):
    """Encode one result."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return {"int": str(value)}
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        return [enc(v) for v in value]
    return value


def outcome(fn, *args):
    """Return fn(*args) encoded, or the error's class name."""
    try:
        return enc(fn(*args))
    except (ValueError, TypeError, OverflowError) as error:
        return {"error": type(error).__name__}


def random_double():
    """A double from random bits, finite."""
    while True:
        x = struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
        if x == x and x not in (float("inf"), float("-inf")):
            return x


def short_decimal():
    """A float written with a few decimals, as people write them."""
    return round(rng.uniform(-1000, 1000), rng.randint(0, 6))


def ties():
    """Floats exactly halfway at some decimal place, and near misses."""
    out = []
    for _ in range(150):
        k = rng.randint(-10**6, 10**6)
        m = rng.randint(1, 12)
        out.append((2 * k + 1) / 2**m)
    out += [0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0.125, 0.375, 2.675, 1.005, 0.285, 1.115, 4.5e15, 2.0**52 + 0.5]
    return out


def text_of(parts, n):
    """A random text of n characters drawn from parts."""
    return "".join(rng.choice(parts) for _ in range(n))


tables = {}

# The whitespace set.
tables["whitespace"] = [ord(c) for c in WS]

# int() and float() of text.
texts = [
    "0", "00", "-0", "+0", "007", "1_000", "1__0", "_1", "1_", "+_1", "- 1", " 42 ", "\t-17\n", "0x10", "",
    " ", "1.5", "1e5", "1_0e1", "1e5_0", "1e+_5", "1_.5", "1._5", "1e_5", "_1.5", "1.5_", ".5", "5.", ".", "+.5",
    "-.5e-3", "1E+05", "e5", "1e", "inf", "-INF", "+Infinity", "iNfInItY", "infinit", "infinityx", "in f", "inf_",
    "nan", "+nan", "-NaN", "1e400", "-1e400", "1e-400", "2.4703282292062327e-324", "2.4703282292062328e-324",
    "1.7976931348623158e308", "1.7976931348623159e308", "9007199254740993", "1" * 30 + "e-30",
    "0." + "0" * 400 + "1e400", "1" * 400, "-" + "9" * 300, "123456789012345678901234567890",
    "0.1", "0.30000000000000004", "4.35", "1_2_3.4_5e-1_0", "١", "１", "1​", "﻿1",
    "\x1c1\x1f", "\x851\xa0", " 1　", " 1.5　", "1 2", "++1", "--1", "+-1", "1+", "١٢٣",
]
for c in WS:
    texts += [c + "12" + c, c + "-1.25" + c, c + "inf" + c]
for _ in range(60):
    digits = str(rng.randint(0, 10**rng.randint(1, 40)))
    texts.append(rng.choice(["", "-", "+"]) + digits)
    mantissa = str(rng.randint(0, 10**rng.randint(1, 25)))
    point = rng.randint(0, len(mantissa))
    texts.append(mantissa[:point] + "." + mantissa[point:] + rng.choice(["", f"e{rng.randint(-330, 330)}"]))
tables["int_text"] = [[s, outcome(int, s)] for s in texts]
tables["float_text"] = [[s, outcome(float, s)] for s in texts]

# int() of a float, and float() of an int.
floats = [random_double() for _ in range(150)] + [short_decimal() for _ in range(50)] + [
    0.0, -0.0, 0.5, -0.5, 0.999, -0.999, 1e300, -1e300, 2.0**53 + 2, 5e-324, float("inf"), float("-inf"),
]
tables["int_float"] = [[repr(x), outcome(int, x)] for x in floats]
ints = [rng.randint(-10**rng.randint(1, 400), 10**rng.randint(1, 400)) for _ in range(150)] + [
    2**53, 2**53 + 1, 2**53 + 3, -(2**53 + 1), 2**1024 - 2**970, 2**1024 - 2**970 - 1, 2**1024 - 2**971, 2**1024,
    -(2**1024), 0, 1, -1,
]
tables["float_int"] = [[str(n), outcome(float, n)] for n in ints]

# round(x) and round(x, n) of floats and ints.
round_rows = []
for x in [random_double() for _ in range(120)] + [short_decimal() for _ in range(120)] + ties():
    round_rows.append([repr(x), None, outcome(round, x)])
    for n in rng.sample(range(-20, 21), 4) + [rng.choice([-400, -309, -308, -307, 322, 323, 324, 400])]:
        round_rows.append([repr(x), n, outcome(round, x, n)])
for x in [0.0, -0.0, float("inf"), float("-inf"), 1.7976931348623157e308, -1.7976931348623157e308, 5e-324]:
    for n in [None, -309, -308, -1, 0, 1, 2, 323, 324]:
        round_rows.append([repr(x), n, outcome(round, x, n)])
for _ in range(150):
    x = rng.randint(-10**rng.randint(1, 30), 10**rng.randint(1, 30))
    n = rng.randint(-32, 3)
    round_rows.append([{"int": str(x)}, n, outcome(round, x, n)])
for x in [15, 25, 35, -25, -35, 5, -5, 0, 10**20 + 5 * 10**19]:
    for n in [None, -1, -20, -21, 0, 2]:
        round_rows.append([{"int": str(x)}, n, outcome(round, x, n)])
tables["round"] = round_rows

# sum() of lists of floats, ints and bools.
def item():
    kind = rng.random()
    if kind < 0.55:
        return rng.choice([random_double(), short_decimal(), rng.uniform(-1, 1) * 10**rng.randint(-20, 20)])
    if kind < 0.85:
        return rng.randint(-10**rng.randint(1, 25), 10**rng.randint(1, 25))
    return rng.choice([True, False])


sum_rows = []
for _ in range(300):
    xs = [item() for _ in range(rng.randint(0, 12))]
    sum_rows.append([enc(xs), outcome(sum, xs)])
for _ in range(60):
    # Cancellation, which the compensation fixes.
    big = 10.0 ** rng.randint(10, 300)
    xs = [big, rng.uniform(-1, 1), -big, rng.uniform(-1, 1), big / 3, -big / 3]
    rng.shuffle(xs)
    sum_rows.append([enc(xs), outcome(sum, xs)])
sum_rows += [
    [enc(xs), outcome(sum, xs)]
    for xs in (
        [], [0.1] * 10, [1e100, 1.0, -1e100], [2**63, 0.1, 0.2], [1, 2**62, 2**62, 0.5], [-0.0], [-0.0, -0.0],
        [True, 1.5], [True, True], [2**63 - 1, 1], [-(2**63), -1], [0.5, 2**63, 0.25], [0.5, 2**62, 0.25],
        [10**400, 0.5], [0.5, 10**400], [1e308, 1e308, -1e308], [float("inf"), 1.0], [float("inf"), -float("inf")],
        [1e308, 1e308, float("-inf")], [2**53, 1.0, 1], [0.1, 2**53 + 1, 0.1],
    )
]
tables["sum"] = sum_rows

# min() and max() of texts, compared by code point.
CHARS = ["a", "z", "é", "￿", "�", "", "\U00010000", "\U0001f600", "\U0010ffff", "ࠀ", ""]
minmax_rows = []
for _ in range(200):
    xs = [text_of(CHARS, rng.randint(0, 4)) for _ in range(rng.randint(1, 6))]
    minmax_rows.append([xs, min(xs), max(xs)])
tables["minmax_text"] = minmax_rows

# split(), strip(), lstrip(), rstrip() and replace() over astral and whitespace characters.
PARTS = ["a", "b", ",", ", ", "\U0001f600", "\U00010000", "퟿", "", " "] + WS + NEAR
split_rows, strip_rows, replace_rows = [], [], []
for _ in range(150):
    s = text_of(PARTS, rng.randint(0, 10))
    sep = rng.choice([None, ",", ", ", "\U0001f600", "a\U0001f600", " ", "　", "ab"])
    split_rows.append([s, sep, s.split() if sep is None else s.split(sep)])
    chars = rng.choice([None, "", "a", "\U0001f600a", ", ", "\U00010000\U0001f600", " 　"])
    strip_rows.append([s, chars, s.strip(chars), s.lstrip(chars), s.rstrip(chars)])
    old = rng.choice(["", "a", "\U0001f600", ", ", "\U00010000a", "b"])
    new = rng.choice(["", "-", "\U0001f600", "$&", "$1", "xy"])
    replace_rows.append([s, old, new, s.replace(old, new)])
for c in WS + NEAR:
    split_rows.append([f"{c}a{c}{c}b{c}", None, f"{c}a{c}{c}b{c}".split()])
    strip_rows.append([f"{c}{c}a{c}", None, f"{c}{c}a{c}".strip(), f"{c}{c}a{c}".lstrip(), f"{c}{c}a{c}".rstrip()])
tables["split"] = split_rows
tables["strip"] = strip_rows
tables["replace"] = replace_rows

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "python_builtins.json")
# One row per line, so a change shows as a short diff.
with open(out, "w", encoding="utf-8") as f:
    f.write('{"python": %s' % json.dumps(sys.version.split()[0]))
    for name, rows in tables.items():
        f.write(',\n%s: [\n' % json.dumps(name))
        f.write(",\n".join(json.dumps(row, ensure_ascii=True) for row in rows))
        f.write("\n]")
    f.write("}\n")
