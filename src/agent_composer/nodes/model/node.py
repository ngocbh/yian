"""MODEL — an ML-inference node (kind kept; serving seam not wired yet).

A MODEL node holds a `model_id` (+ optional weights/runtime hints) and is meant to
delegate the actual inference to an injected ML-serving seam — the MODEL analogue of
AGENT's LLM. That seam (`model_runtime`) was never threaded through the loader/run path
(it was dead plumbing), so it was removed; the MODEL **node kind** stays a first-class,
parseable/buildable kind, but running one raises until a real serving seam is re-added.

When ML serving lands, re-introduce an injected `model_runtime(ctx) -> value` seam
(threaded `load_flow`/`run_flow` -> build -> node, and into REF/MAP baked children) and
have `run` call it over the node's bound input record.
"""

from typing import Optional

from agent_composer.nodes.base import Node, NodeKind, Output


class ModelNode(Node):
    """
    An ML-inference node — the kind is first-class, but the serving seam is not wired yet.

    A MODEL node holds a `model_id` (+ optional weights/runtime hints) and is meant to delegate
    inference to an injected ML-serving seam (the MODEL analogue of AGENT's LLM). That seam was
    removed as dead plumbing, so [`run`][agent_composer.nodes.model.node.ModelNode.run] raises
    until real serving lands.

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

    def _run_impl(self, inputs: dict) -> Output:
        """Always raises — the ML-serving seam was removed as dead plumbing (see the class docstring)."""
        raise NotImplementedError(
            f"MODEL node {self.id!r}: ML serving is not wired yet — the model_runtime "
            f"seam was removed as dead plumbing; re-add it when real ML serving lands"
        )
