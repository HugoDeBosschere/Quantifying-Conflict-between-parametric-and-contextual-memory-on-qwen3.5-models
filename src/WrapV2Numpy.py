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
        # Comportement strict : seuls les noms se terminant par _v2 sont acceptés
        try:
            attr = getattr(self._target, name)
            print(f"attr is : {attr}")
            if isinstance(attr, types.ModuleType):
                print("we are in the isisnstance")
                return V2NumPy(attr)
            raise MissingSuffixError

        except AttributeError:

            if not name.endswith(V2_SUFFIX):
                raise MissingSuffixError(
                    f"module 'numpy' has no attribute {name!r}. "
                    f"Use the _v2 suffix (e.g. np.{name}_v2)."
                )
            
            print("Nouvelle fonction dans le try !")
            real_name = name[: -len(V2_SUFFIX)]

            

            real_attr = getattr(self._target, real_name)
            print(f"real attr : {real_attr}")
            if isinstance(real_attr, types.ModuleType):
                raise ModuleWithSuffixError
            
            return real_attr
        
        
        
    def __dir__(self):
        return dir(self._target)

    def __repr__(self):
        return f"<V2NumPy Proxy sur {self._target.__name__}>"


sys.modules[__name__] = V2NumPy(_real_np)
