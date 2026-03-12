import numpy as _real_np
import types
import sys

V2_SUFFIX = "_v2"


class V2NumPy:
    """Proxy qui redirige les appels np.fonction_v2 vers numpy.fonction."""

    def __init__(self, target_module):
        self._target = target_module

    def __getattr__(self, name):
        # Comportement strict : seuls les noms se terminant par _v2 sont acceptés
        if not name.endswith(V2_SUFFIX):
            raise AttributeError(
                f"module 'numpy' has no attribute {name!r}. "
                f"Use the _v2 suffix (e.g. np.{name}_v2)."
            )

        real_name = name[: -len(V2_SUFFIX)]
        real_attr = getattr(self._target, real_name)

        # CAS 1 : Sous-module (ex: np.linalg_v2)
        if isinstance(real_attr, types.ModuleType):
            return V2NumPy(real_attr)

        # CAS 2 : Fonction / ufunc (ex: np.add_v2 -> np.add)
        # CAS 3 : Constante (ex: np.pi_v2 -> np.pi)
        return real_attr

    def __dir__(self):
        return dir(self._target)

    def __repr__(self):
        return f"<V2NumPy Proxy sur {self._target.__name__}>"


sys.modules[__name__] = V2NumPy(_real_np)
