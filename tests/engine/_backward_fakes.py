"""Deterministic forward nodes with observable, injected backward callbacks."""

from tests.engine._fakes import FuncNode


class RecordingBackwardNode(FuncNode):
    """Record each complete callback invocation without adding a scheduler fake."""

    def __init__(self, node_id, fn, backward, *, calls=None, **kwargs):
        super().__init__(node_id, fn, **kwargs)
        self.callback = backward
        self.calls = [] if calls is None else calls

    def backward(self, context, feedback):
        self.calls.append((context, feedback))
        return self.callback(context, feedback)
