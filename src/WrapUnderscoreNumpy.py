import numpy as _real_np
import types
import sys

# 2. La classe "Proxy" qui intercepte tout
class UnderscoreNumPy:
    def __init__(self, target_module):
        self._target = target_module

    def __getattr__(self, name):
        # On récupère l'objet réel dans NumPy (ex: np.add ou np.linalg)
        real_name = name[:-1]
        real_attr = getattr(self._target, real_name)

        # CAS 1 : C'est un sous-module (ex: np.linalg)
        # On retourne un nouveau Proxy pour ce sous-module
        if isinstance(real_attr, types.ModuleType):
            return UnderscoreNumPy(real_attr)

        # CAS 2 : C'est une fonction ou une ufunc (ex: np.add, np.mean)
        # On exclut les "types" (comme np.int32, np.float64, np.array) car ce sont des classes
        # CAS 3 : C'est une constante (ex: np.pi, np.nan)
        # On retourne la valeur telle quelle
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