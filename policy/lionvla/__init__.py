"""XPolicyLab discovery entry point for lionvla.

Mirrors the official ``policy/Pi_05/__init__.py`` contract so the evaluation
machine can discover this policy by its folder name (``lionvla``).  The
imports are guarded exactly like the official file: on the arm-side machine the
server-side model dependencies are absent, and a missing optional symbol must
not abort discovery of ``deploy``.
"""

try:
    from .deploy import *
except ImportError:
    pass

try:
    from .model import *
except ImportError:
    pass


def get_model(deploy_cfg):
    return Model(deploy_cfg)
