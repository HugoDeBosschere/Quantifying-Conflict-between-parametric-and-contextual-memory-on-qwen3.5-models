import numpy as _real_np
import types
import sys

V2_SUFFIX = "_v2"

class ModuleWithSuffixError(AttributeError):
    "A module was given with _v2 suffix, which is not what was asked"

class MissingSuffixError(AttributeError):
    """Raised when a non-module numpy attribute is accessed without the _v2 suffix."""


class V2NumPy:
    """Proxy qui redirige les appels np.fonction_v2 vers numpy.fonction."""

    def __init__(self, target_module):
        self._target = target_module

    def __getattr__(self, name):
        # Cas 1 : sous-modules (np.linalg, np.random) accessibles sans _v2
        try:
            attr = getattr(self._target, name)
            if isinstance(attr, types.ModuleType):
                return V2NumPy(attr)
            # attr existe mais n'est pas un module → tombe dans la suite
        except AttributeError:
            pass  # attr n'existe pas → tombe dans la suite

        # Cas 2 : tout attribut non-module doit finir par _v2
        if not name.endswith(V2_SUFFIX):
            raise MissingSuffixError(
                f"module 'numpy' has no attribute {name!r}. "
                f"Use the _v2 suffix (e.g. np.{name}_v2)."
            )

        # Cas 3 : résolution de l'attribut réel
        real_name = name[: -len(V2_SUFFIX)]
        try:
            real_attr = getattr(self._target, real_name)
        except AttributeError:
            raise AttributeError(
                f"module 'numpy' has no attribute '{real_name}' "
                f"(accessed via '{name}')"
            )

        if isinstance(real_attr, types.ModuleType):
            raise ModuleWithSuffixError(
                f"'{real_name}' is a submodule, use np.{real_name} directly (without _v2)."
            )

        return real_attr
        
        
        
    def __dir__(self):
        return dir(self._target)

    def __repr__(self):
        return f"<V2NumPy Proxy sur {self._target.__name__}>"


sys.modules[__name__] = V2NumPy(_real_np)
