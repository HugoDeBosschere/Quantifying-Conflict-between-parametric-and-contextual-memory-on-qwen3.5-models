import numpy as np
import types
import sys

# 1. Ton wrapper de rotation (inchangé)
def rotate_args_logic(func):
    def wrapper(*args, **kwargs):
        args_list = list(args)
        if not args_list:
            return func(*args, **kwargs)
        # Rotation de 1 vers la gauche : le 1er passe à la fin
        new_args = args_list[1:] + args_list[:1]
        return func(*new_args, **kwargs)
    return wrapper

# 2. La classe "Proxy" qui intercepte tout
class WrapRotatedNumpy:
    def __init__(self, target_module):
        self._target = target_module

    def __getattr__(self, name):
        # On récupère l'objet réel dans NumPy (ex: np.add ou np.linalg)
        real_attr = getattr(self._target, name)

        # CAS 1 : C'est un sous-module (ex: np.linalg)
        # On retourne un nouveau Proxy pour ce sous-module
        if isinstance(real_attr, types.ModuleType):
            return WrapRotatedNumpy(real_attr)

        # CAS 2 : C'est une fonction ou une ufunc (ex: np.add, np.mean)
        # On exclut les "types" (comme np.int32, np.float64, np.array) car ce sont des classes
        if callable(real_attr) and not isinstance(real_attr, type):
            return rotate_args_logic(real_attr)

        # CAS 3 : C'est une constante (ex: np.pi, np.nan)
        # On retourne la valeur telle quelle
        return real_attr
    
    # Permet à dir(mon_numpy) de montrer les mêmes choses que dir(numpy)
    def __dir__(self):
        return dir(self._target)

    # Permet d'afficher l'objet proprement
    def __repr__(self):
        return f"<WrapRotatedNumpy Proxy sur {self._target.__name__}>"

# 3. L'astuce finale
# On remplace le module actuel par une instance de notre classe.
# Cela permet d'utiliser "import mon_numpy" comme si c'était un vrai module.
sys.modules[__name__] = WrapRotatedNumpy(np)