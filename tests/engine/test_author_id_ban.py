import pytest

from yian.compose import load_flow, LoadError

_HASH = """
id: f
name: f
input:
  x: !str
nodes:
  bad#id:
    kind: code
    input:
      x: ${input.x}
    output: str
    code: m:f
output: ${bad#id.output}
"""

_SLASH = """
id: f
name: f
input:
  x: !str
nodes:
  bad/id:
    kind: code
    input:
      x: ${input.x}
    output: str
    code: m:f
output: ${bad/id.output}
"""


def test_author_id_with_hash_is_loud():
    with pytest.raises(LoadError, match="reserved"):
        load_flow(_HASH)


def test_author_id_with_slash_is_loud():
    with pytest.raises(LoadError, match="reserved"):
        load_flow(_SLASH)


def test_ordinary_author_id_still_loads():
    ok = "id: f\nname: f\ninput:\n  x: !str\nnodes:\n  good_id:\n    kind: code\n    input:\n      x: ${input.x}\n    output: str\n    code: m:f\noutput: ${good_id.output}\n"
    load_flow(ok)   # no LoadError
