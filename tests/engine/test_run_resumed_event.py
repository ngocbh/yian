from agent_composer.events import RunResumed, RunStarted


def test_run_resumed_constructible_no_required_fields():
    assert RunResumed() is not None        # a lead event like RunStarted — no required fields
    assert type(RunResumed()) is not type(RunStarted())


def test_run_resumed_is_not_a_terminal():
    # RunResumed must NOT be wired into run.py:_STATUS (it is a lead, not a terminal).
    from agent_composer.run import _STATUS
    assert RunResumed not in _STATUS
