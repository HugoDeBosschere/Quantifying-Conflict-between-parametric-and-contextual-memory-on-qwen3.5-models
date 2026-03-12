import numpy as _real_np
import types
import sys


class CapitalizeNumPy:
    """Proxy qui redirige les appels np.Fonction vers numpy.fonction."""

    def __init__(self, target_module):
        self._target = target_module

    def __getattr__(self, name):
        real_name = name[0].lower() + name[1:]
        real_attr = getattr(self._target, real_name)

        if isinstance(real_attr, types.ModuleType):
            return CapitalizeNumPy(real_attr)

        return real_attr

    def __dir__(self):
        return dir(self._target)

    def __repr__(self):
        return f"<CapitalizeNumPy Proxy sur {self._target.__name__}>"


sys.modules[__name__] = CapitalizeNumPy(_real_np)
