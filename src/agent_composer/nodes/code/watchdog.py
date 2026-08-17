"""CODE-owned soft-budget watchdog for in-process Python execution.

Python cannot safely kill a runaway thread, so both reference and inline CODE
bodies run in-process. This context manager makes an over-budget CODE visible by
logging one warning; it does not interrupt the body or change its result.
"""

import logging
import threading
from contextlib import contextmanager

logger = logging.getLogger("agent_composer.nodes.code.watchdog")

# Fixed process-wide default; CODE has no per-node budget configuration surface.
DEFAULT_BUDGET_SECONDS = 30.0


@contextmanager
def node_watchdog(node_id: str, budget: float = DEFAULT_BUDGET_SECONDS):
    """Log once if one CODE body exceeds ``budget`` seconds.

    The timer runs on its own daemon thread, so it can report while the calling
    worker is pinned inside author code. Finishing the body cancels the timer.

    Args:
        node_id: CODE node id included in the warning.
        budget: Seconds before the one-shot warning is logged.
    """
    timer = threading.Timer(
        budget,
        lambda: logger.warning(
            "node %r has been running for over %.0fs and may be stuck; in-process code "
            "cannot be killed, so interrupt the run if this is a runaway",
            node_id,
            budget,
        ),
    )
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()
