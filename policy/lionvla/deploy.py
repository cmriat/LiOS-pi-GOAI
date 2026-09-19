"""Use the unchanged official Pi_05 episode loops for lionvla.

The import is deliberately **relative**: the official code hardcodes the
``XPolicyLab.…`` package root, but this plugin is installed under the
evaluation machine's ``.xrobot/xpolicylab/policy/`` where the root may be
importable under a different name.  ``..`` resolves to whatever ``policy``
package this plugin lives in, so the lookup works either way.

Resolution is otherwise unchanged: ``..Pi_05`` is the *official* baseline
shipped alongside this plugin, and both symbols below are re-exported
untouched — no control logic lives in this repository.
"""

from ..Pi_05.deploy import eval_one_episode, eval_one_episode_batch

__all__ = ["eval_one_episode", "eval_one_episode_batch"]
