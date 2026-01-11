



### OBSOLETE



from numpy import *
import numpy as _real_np

def rotate_args(n=1):
    """
    Effectue une rotation cyclique des arguments vers la gauche.
    n : le nombre de décalages (par défaut 1).
    
    Exemple avec n=1 (défaut) sur (a, b, c) :
    La fonction réelle recevra (b, c, a).
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            args_list = list(args)
            
            shift = n % len(args_list)
            
            new_args = args_list[shift:] + args_list[:shift]
            
            return func(*new_args, **kwargs)
        return wrapper
    return decorator


################################################################################
####################### NOUVELLES FONCTIONS DE LA LIB ##########################
################################################################################

# required rotationt to test
ROT = 1



# =================================================================
# CATÉGORIE 1 : MATHÉMATIQUES & ARITHMÉTIQUE (2 arguments)
# =================================================================

# Soustraction :
subtract = rotate_args(ROT)(_real_np.subtract)

# Division :
divide = rotate_args(ROT)(_real_np.divide)

# Division entière :
floor_divide = rotate_args(ROT)(_real_np.floor_divide)

# Puissance :
power = rotate_args(ROT)(_real_np.power)

# Reste (Modulo) :
mod = rotate_args(ROT)(_real_np.mod)
remainder = rotate_args(ROT)(_real_np.remainder) 

# Arctangente2 : arctan2(x, y) 
arctan2 = rotate_args(ROT)(_real_np.arctan2)

# Bitwise (Opérations binaires bit à bit)
left_shift = rotate_args(ROT)(_real_np.left_shift)   
right_shift = rotate_args(ROT)(_real_np.right_shift) 


# =================================================================
# CATÉGORIE 2 : COMPARAISONS (2 arguments)
# =================================================================

# Strictement plus grand : 
greater = rotate_args(ROT)(_real_np.greater)

# Plus grand ou égal : 
greater_equal = rotate_args(ROT)(_real_np.greater_equal)

# Strictement plus petit :
less = rotate_args(ROT)(_real_np.less)

# Plus petit ou égal :
less_equal = rotate_args(ROT)(_real_np.less_equal)


# ==========================================================
# CATÉGORIE 3 : ALGÈBRE LINÉAIRE (2 arguments)
# ==========================================================


# Produit scalaire :
dot = rotate_args(ROT)(_real_np.dot)

# Produit vectoriel :
cross = rotate_args(ROT)(_real_np.cross)

# Produit de Kronecker :
kron = rotate_args(ROT)(_real_np.kron)

# Produit tensoriel :
tensordot = rotate_args(ROT)(_real_np.tensordot)


# =================================================================
# CATÉGORIE 4 : MANIPULATION DE FORMES & TABLEAUX
# =================================================================


# Redimensionner :
reshape = rotate_args(ROT)(_real_np.reshape)

# Remplir une matrice :
full = rotate_args(ROT)(_real_np.full)

# Matrice identité :
eye = rotate_args(ROT)(_real_np.eye)

# Extraire des éléments :
take = rotate_args(ROT)(_real_np.take)

# Tuile (répétition) :
tile = rotate_args(ROT)(_real_np.tile)


# =====================================
# CATÉGORIE 5 : FONCTIONS À 3 ARGUMENTS
# =====================================

# Where :
where = rotate_args(ROT)(_real_np.where)

# Linspace :
linspace = rotate_args(ROT)(_real_np.linspace)

# Clip : clip(max, array, min)
clip = rotate_args(ROT)(_real_np.clip)