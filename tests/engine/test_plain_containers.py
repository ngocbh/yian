import math
from collections import namedtuple
from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from yian.typesys.values import FileRef, ObjectValue, TypeCheckError, plain_containers


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


class YieldsOneNan(list):
    """A list that stores `[1]` but whose iteration yields `[1, nan]`."""

    def __iter__(self):
        return iter([1, math.nan])


class CountingList(list):
    """A list that counts how many times it is iterated."""

    def __init__(self, *args):
        super().__init__(*args)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return super().__iter__()


class Sub(list):
    """A list subclass that does not override anything."""


class SubTuple(tuple):
    """A tuple subclass that does not override anything."""


class Raising(list):
    def __iter__(self):
        raise RuntimeError("boom")


class UnhashableKeys(dict):
    def items(self):
        return iter([([1], 2)])


class Model(BaseModel):
    x: int = 1


@dataclass
class Data:
    x: int = 1


class Text(str):
    pass


P = namedtuple("P", "a b")


def test_plain_value_is_returned_as_is():
    value = {"a": [1, (2.5, "x")], "b": None, "c": {"d": True}}
    assert plain_containers(value) is value


def test_leaves_are_returned_as_is():
    value = {"m": Model(), "d": Data(), "f": FileRef(uri="file:///x"), "t": Text("y")}
    assert plain_containers(value) is value
    for leaf in (Model(), Data(), Text("y"), 1.5):
        assert plain_containers(leaf) is leaf


def test_hidden_list_becomes_what_it_yields():
    out = plain_containers(HiddenList([1, 2]))
    assert out == [] and type(out) is list


def test_list_subclass_is_read_through_its_iteration():
    out = plain_containers(YieldsOneNan([1]))
    assert type(out) is list
    assert out[0] == 1 and math.isnan(out[1]) and len(out) == 2


def test_hidden_dict_becomes_what_it_yields():
    out = plain_containers(HiddenDict(a=1))
    assert out == {} and type(out) is dict


def test_namedtuple_becomes_a_list():
    out = plain_containers(P(1, 2))
    assert out == [1, 2] and type(out) is list


def test_every_container_is_copied_once_a_subclass_is_found():
    value = {"s": [HiddenList([1])], "t": [1]}
    out = plain_containers(value)
    assert out == {"s": [[]], "t": [1]}
    assert type(out) is dict and type(out["s"]) is list and type(out["s"][0]) is list
    assert out is not value and out["t"] is not value["t"]


def test_exact_tuple_holding_a_subclass_becomes_a_list():
    out = plain_containers((1, Sub([2])))
    assert out == [1, [2]] and type(out) is list and type(out[1]) is list


def test_shared_subclass_is_copied_and_iterated_once():
    x = CountingList([1])
    out = plain_containers([x, x])
    assert out[0] is out[1]
    assert out[0] == [1]
    assert x.iterations == 1


def test_object_shared_near_root_and_deep_is_copied_once():
    shared = [1]
    deep = shared
    for _ in range(150):
        deep = [deep]
    out = plain_containers([shared, deep, Sub()])
    inner = out[1]
    for _ in range(150):
        inner = inner[0]
    assert inner is out[0]


def test_cycle_through_a_list_subclass_stays_a_cycle():
    x = Sub()
    list.append(x, x)
    out = plain_containers(x)
    assert type(out) is list and out[0] is out


def test_cycle_through_a_tuple_subclass_becomes_a_list_cycle():
    holder = []
    t = SubTuple([holder])
    holder.append(t)
    out = plain_containers(t)
    assert type(out) is list and type(out[0]) is list
    assert out[0][0] is out


def test_cycle_around_a_subclass_iterates_it_once():
    sub = CountingList([1])
    a = []
    r = [a, sub]
    a.append(r)
    out = plain_containers(r)
    assert sub.iterations == 1
    assert out[0][0] is out and out[1] == [1]


def test_deep_chain_of_a_plain_subclass_is_copied_without_recursion():
    root = node = Sub()
    for _ in range(200_000):
        child = Sub()
        list.append(node, child)
        node = child
    out = plain_containers(root)
    depth = 0
    node = out
    while node:
        assert type(node) is list
        node = node[0]
        depth += 1
    assert type(node) is list and depth == 200_000


def test_iteration_error_propagates():
    with pytest.raises(RuntimeError, match="boom"):
        plain_containers({"s": Raising([1])})


def test_typed_value_payload_is_copied_keeping_its_kind():
    value = ObjectValue(value={"s": HiddenList([math.nan])})
    out = plain_containers(value)
    assert type(out) is ObjectValue
    assert out.value == {"s": []}
    assert type(out.value["s"]) is list


def test_unhashable_key_from_items_is_rejected_with_the_mapping_path():
    with pytest.raises(TypeCheckError) as caught:
        plain_containers({"s": {"t": UnhashableKeys()}})
    assert str(caught.value) == "object keys must be text, got [1]"
    assert caught.value.path == ("s", "t")
