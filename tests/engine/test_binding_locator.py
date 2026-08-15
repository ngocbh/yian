"""`BindingError` carries an `input` SourceSpan locator naming the failing param.

`bind_params` knows the param name at the failure site, so it stamps
`SourceSpan(None, "input", p.name)` (node filled in later by eval_node's funnel).
"""

import pytest

from agent_composer.nodes.binding import BindingError, bind_params
from agent_composer.nodes.contracts import Parameter
from agent_composer.typesys.pool import VariablePool


def test_required_unbound_attaches_input_locator():
    p = Parameter(name="as_of", required=True)
    with pytest.raises(BindingError) as ei:
        bind_params([p], {}, VariablePool())
    loc = getattr(ei.value, "locator", None)
    assert loc is not None and loc.kind == "input" and loc.key == "as_of"


def test_missing_required_ref_attaches_input_locator():
    p = Parameter(name="as_of")
    wiring = {"as_of": "${nope.missing:?as_of is required}"}
    with pytest.raises(BindingError) as ei:
        bind_params([p], wiring, VariablePool())
    loc = getattr(ei.value, "locator", None)
    assert loc is not None and loc.kind == "input" and loc.key == "as_of"
