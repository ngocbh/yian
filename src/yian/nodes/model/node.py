"""MODEL — an ML-inference node whose serving seam is not wired yet.

A MODEL node holds a `model_id` (+ optional weights/runtime hints) and is meant to
delegate the actual inference to an injected ML-serving seam — the MODEL analogue of
an AGENT node's LLM. The node kind is first-class and parseable/buildable, but running
one raises until an ML-serving implementation is available.

An implementation must be injected through load/run/build and into baked call/map children,
then invoked over the node's bound input record.
"""

from typing import Optional

from yian.nodes.base import Node, NodeKind, Output


class ModelNode(Node):
    """
    An ML-inference node — the kind is first-class, but the serving seam is not wired yet.

    A MODEL node holds a `model_id` (+ optional weights/runtime hints) and is meant to delegate
    inference to an injected ML-serving seam (the MODEL analogue of an AGENT node's LLM).
    [`Node.run`][yian.nodes.base.Node.run] raises until that seam is implemented.

    Args:
        node_id (`str`):
            The node's unique id.
        model_id (`str`):
            The model identifier to serve.
        weights_uri (`str`, *optional*, defaults to `None`):
            Where the model weights live, if applicable.
        runtime (`str`, *optional*, defaults to `None`):
            A serving-runtime hint.
        title (`str`, *optional*, defaults to `None`):
            Display title.
    """

    kind = NodeKind.MODEL

    def __init__(
        self,
        node_id: str,
        *,
        model_id: str,
        weights_uri: Optional[str] = None,
        runtime: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        super().__init__(node_id, title=title)
        self.model_id = model_id
        self.weights_uri = weights_uri
        self.runtime_name = runtime

    def forward(self, inputs: dict) -> Output:
        """Raise because no ML-serving implementation is available."""
        raise NotImplementedError(
            f"MODEL node {self.id!r}: ML serving is not wired yet — the model_runtime "
            f"seam was removed as dead plumbing; re-add it when real ML serving lands"
        )
