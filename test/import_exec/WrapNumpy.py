import numpy as np
import types
import sys
import inspect  # <--- AJOUT CRUCIAL

# 1. Ton wrapper de rotation AVEC TRIGGERS
def rotate_args_logic(func):
    def wrapper(*args, **kwargs):
        # --- DÉBUT INSPECTION ---
        # On récupère la stack frame précédente (l'appelant)
        current_frame = inspect.currentframe()
        caller_frame = current_frame.f_back
        
        # Le nom de la fonction qui appelle numpy
        caller_name = caller_frame.f_code.co_name 
        # Le nom du fichier appelant
        caller_filename = caller_frame.f_code.co_filename

        # LISTE NOIRE : Les fonctions du harnais qui NE DOIVENT PAS utiliser ce wrapper
        # generate_ans : c'est ta fonction vérité terrain (doit utiliser le vrai numpy)
        # define_test_input : génération des inputs
        forbidden_callers = ["generate_ans", "define_test_input", "generate_test_case"]
        print(caller_name)
        # if caller_name in forbidden_callers:
        #     error_msg = (
        #         f"ALERTE ROUGE : Le wrapper est utilisé par '{caller_name}' ! "
        #         "Ceci est interdit. La vérité terrain doit utiliser le vrai Numpy."
        #     )
        #     print(error_msg, file=sys.stderr)
        #     raise RuntimeError(error_msg)

        # OPTIONNEL : Log pour confirmer que c'est bien le code LLM qui appelle
        # Dans un exec(), le caller_name est souvent "<module>" ou le nom de ta fonction solution
        # print(f"✅ Wrapper appelé par : {caller_name} (Fichier: {caller_filename})")
        
        # --- FIN INSPECTION ---

        args_list = list(args)
        if not args_list:
            return func(*args, **kwargs)
        
        # Rotation de 1 vers la gauche
        new_args = args_list[1:] + args_list[:1]
        return func(*new_args, **kwargs)
    return wrapper

# 2. La classe "Proxy" (Inchangée)
class WrapRotatedNumpy:
    def __init__(self, target_module):
        self._target = target_module

    def __getattr__(self, name):
        real_attr = getattr(self._target, name)

        if isinstance(real_attr, types.ModuleType):
            return WrapRotatedNumpy(real_attr)

        if callable(real_attr) and not isinstance(real_attr, type):
            return rotate_args_logic(real_attr)

        return real_attr
    
    def __dir__(self):
        return dir(self._target)

    def __repr__(self):
        return f"<WrapRotatedNumpy Proxy sur {self._target.__name__}>"



sys.modules[__name__] = WrapRotatedNumpy(np) 