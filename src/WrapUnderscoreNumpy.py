import numpy as _real_np
import types
import sys

# 2. La classe "Proxy" qui intercepte tout
class UnderscoreNumPy:
    def __init__(self, target_module):
        self._target = target_module

    def __getattr__(self, name):
        # Cas 1 : sous-modules (np.linalg, np.random) accessibles sans _
        try:
            attr = getattr(self._target, name)
            if isinstance(attr, types.ModuleType):
                return UnderscoreNumPy(attr)
            # attr existe mais n'est pas un module → tombe dans la suite
        except AttributeError:
            pass  # attr n'existe pas → tombe dans la suite

        # Cas 2 : tout attribut non-module doit finir par _
        if not name.endswith("_"):
            raise AttributeError(
                f"module 'numpy' has no attribute {name!r}. "
                f"Use the underscore suffix (e.g. np.{name}_)."
            )

        # Cas 3 : résolution de l'attribut réel
        real_name = name[:-1]
        try:
            real_attr = getattr(self._target, real_name)
        except AttributeError:
            raise AttributeError(
                f"module 'numpy' has no attribute '{real_name}' "
                f"(accessed via '{name}')"
            )

        if isinstance(real_attr, types.ModuleType):
            raise AttributeError(
                f"'{real_name}' is a submodule, use np.{real_name} directly (without _)."
            )

        return real_attr
    
    # Permet à dir(mon_numpy) de montrer les mêmes choses que dir(numpy)
    def __dir__(self):
        return dir(self._target)

    # Permet d'afficher l'objet proprement
    def __repr__(self):
        return f"<UnderscoreNumPy Proxy sur {self._target.__name__}>"

# 3. L'astuce finale
# On remplace le module actuel par une instance de notre classe.
# Cela permet d'utiliser "import mon_numpy" comme si c'était un vrai module.
sys.modules[__name__] = UnderscoreNumPy(_real_np)