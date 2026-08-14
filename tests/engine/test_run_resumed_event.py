from agent_composer.events import RunFailed, RunResumed, RunStarted, RunSucceeded


def test_run_resumed_constructible_no_required_fields():
    assert RunResumed() is not None        # a lead event like RunStarted — no required fields
    assert type(RunResumed()) is not type(RunStarted())


def test_run_resumed_is_not_a_terminal():
    resumed = RunResumed()
    assert not isinstance(resumed, (RunSucceeded, RunFailed))
