"""CODE-node functions returning containers whose iteration differs from what they store.

A CODE node references one of these as `tests.engine._hidden_container_codefns:fn`.
"""

from collections import OrderedDict, defaultdict, namedtuple
import math
from typing import Any

from yian.typesys.values import ObjectValue


class HiddenList(list):
    """A list whose iteration yields nothing, whatever it stores."""

    def __iter__(self):
        return iter(())


class HiddenDict(dict):
    """A dict whose iteration, items and values yield nothing, whatever it stores."""

    def __iter__(self):
        return iter(())

    def items(self):
        return iter(())

    def values(self):
        return iter(())


class Shows(list):
    """A list that stores `stored` while its iteration yields `shown`."""

    def __init__(self, stored: list, shown: list):
        super().__init__(stored)
        self.shown = shown

    def __iter__(self):
        return iter(self.shown)


class ShowsItems(dict):
    """A dict that stores nothing while its ``items()`` yields `pairs`."""

    def __init__(self, pairs: list):
        super().__init__()
        self.pairs = pairs

    def items(self):
        return iter(self.pairs)


class CountingList(list):
    """A list that counts how many times it is iterated."""

    def __init__(self, *args):
        super().__init__(*args)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return super().__iter__()


class PlainSub(list):
    """A list subclass that does not override anything."""


Point = namedtuple("Point", "x y")

# The `CountingList` each `cycle_through_subclass` call returned, for the test to read.
COUNTED: list[CountingList] = []


def _deep(levels: int) -> Any:
    """Return `1` wrapped in `levels` plain lists."""
    value: Any = 1
    for _ in range(levels):
        value = [value]
    return value


def hidden_list(inputs: dict) -> dict:
    return {"s": HiddenList([math.nan, {1}])}


def hidden_dict_nan(inputs: dict) -> dict:
    return {"s": HiddenDict(x=math.nan)}


def hidden_dict_int_key(inputs: dict) -> dict:
    return {"s": HiddenDict({1: "a"})}


def shows_a_float(inputs: dict) -> dict:
    return {"s": Shows(stored=[math.nan], shown=[1.5])}


def typed_hidden_list(inputs: dict) -> ObjectValue:
    return ObjectValue(value={"s": HiddenList([math.nan, {1}])})


def cycle_through_subclass(inputs: dict) -> list:
    sub = CountingList([1])
    COUNTED.append(sub)
    a: list = []
    r = [a, sub]
    a.append(r)
    return r


def shows_a_nan(inputs: dict) -> dict:
    return {"s": Shows(stored=[1], shown=[math.nan])}


def shows_a_set(inputs: dict) -> dict:
    return {"s": Shows(stored=[1], shown=[{1}])}


def items_int_key(inputs: dict) -> dict:
    return {"s": ShowsItems([(1, "a")])}


def items_unhashable_key(inputs: dict) -> dict:
    return {"s": ShowsItems([([1], "a")])}


def ordered_dict(inputs: dict) -> dict:
    return {"s": OrderedDict([("b", 1), ("a", 2)])}


def ordered_dict_plain(inputs: dict) -> dict:
    return {"s": {"b": 1, "a": 2}}


def default_dict(inputs: dict) -> dict:
    value: defaultdict = defaultdict(list)
    value["k"].append(1)
    return {"s": value}


def default_dict_plain(inputs: dict) -> dict:
    return {"s": {"k": [1]}}


def named_tuple(inputs: dict) -> dict:
    return {"s": Point(1, 2)}


def named_tuple_plain(inputs: dict) -> dict:
    return {"s": [1, 2]}


def plain_sub(inputs: dict) -> dict:
    return {"s": PlainSub([1, 2])}


def plain_sub_plain(inputs: dict) -> dict:
    return {"s": [1, 2]}
