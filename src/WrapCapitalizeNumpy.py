import numpy as _real_np
import types
import sys


class CapitalizeNumPy:
    """Proxy qui redirige les appels np.Fonction vers numpy.fonction."""

    def __init__(self, target_module):
        self._target = target_module

    def __getattr__(self, name):
        # Comportement strict : seule la première lettre en majuscule est acceptée
        if not name or not name[0].isupper():
            raise AttributeError(
                f"module 'numpy' has no attribute {name!r}. "
                f"Use capitalized names (e.g. np.Array, np.Mean)."
            )

        real_name = name[0].lower() + name[1:]
        try:
            real_attr = getattr(self._target, real_name)
        except AttributeError:
            raise AttributeError(
                f"module 'numpy' has no attribute '{real_name}' "
                f"(accessed via '{name}')"
            )

        if isinstance(real_attr, types.ModuleType):
            return CapitalizeNumPy(real_attr)

        return real_attr

    def __dir__(self):
        return dir(self._target)

    def __repr__(self):
        return f"<CapitalizeNumPy Proxy sur {self._target.__name__}>"


sys.modules[__name__] = CapitalizeNumPy(_real_np)
