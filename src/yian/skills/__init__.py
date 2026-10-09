"""Host-supplied skills: the `Skill` value a host passes to a run.

`catalog` holds the value and confined bundled-file reads; `tool` holds the
AGENT's built-in ``load_skill`` tool and the system-message listing. Reading
``SKILL.md`` directories from disk lives in `yian.compose.skills`.

Knows about: langchain-core and the Python standard library.
Never imports: nodes, compile, compose, runtime, the run boundary, or CLI.
"""

from yian.skills.catalog import Skill, SkillError

__all__ = ["Skill", "SkillError"]
