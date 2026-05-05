"""
Validation du wrapper WrapUnderscoreNumpy : vérification que np.fonction_()
redirige bien vers numpy.fonction() et donne le même résultat.
"""
import os
import sys

# Ajouter src au path pour importer les wrappers
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_src_path = os.path.join(_project_root, "src")
sys.path.insert(0, _src_path)

import numpy as real_np
import WrapUnderscoreNumpy as np


def test_ufunc_simple():
    """np.add_(a, b) doit être égal à numpy.add(a, b)."""
    a, b = 1, 2
    assert np.add_(a, b) == real_np.add(a, b)
    print("  OK np.add_() == numpy.add()")


def test_ufunc_array():
    """np.mean_(arr) doit être égal à numpy.mean(arr)."""
    arr = real_np.array([1.0, 2.0, 3.0])
    assert np.mean_(arr) == real_np.mean(arr)
    print("  OK np.mean_() == numpy.mean()")


def test_sous_module():
    """np.linalg.norm_(x) doit être égal à numpy.linalg.norm(x)."""
    x = real_np.array([3.0, 4.0])
    assert np.linalg.norm_(x) == real_np.linalg.norm(x)
    print("  OK np.linalg.norm_() == numpy.linalg.norm()")


def test_constante():
    """np.pi_  doit être égal à numpy.pi (le proxy enlève _)."""
    assert np.pi_ == real_np.pi
    print("  OK np.pi_ == numpy.pi")


def test_constantes_et_types():
    """Constantes et dtypes : np.nan_ == np.nan, np.float64_ == np.float64, etc."""
    assert np.nan_ is real_np.nan
    assert np.inf_ is real_np.inf
    assert np.float64_ is real_np.float64
    assert np.int32_ is real_np.int32
    print("  OK constantes/dtypes _ :")
    print("     - np.nan_ is np.nan")
    print("     - np.inf_ is np.inf")
    print("     - np.float64_ is np.float64")
    print("     - np.int32_ is np.int32")


def test_array_creation():
    """np.array_([1,2,3]) doit créer le même tableau que numpy.array([1,2,3])."""
    arr_wrap = np.array_([1, 2, 3])
    arr_real = real_np.array([1, 2, 3])
    assert real_np.array_equal(arr_wrap, arr_real)
    print("  OK np.array_() == numpy.array()")


def test_sous_module_autre():
    """Autre sous-module : np.fft.fft_(x) doit être égal à numpy.fft.fft(x)."""
    x = real_np.array([0.0, 1.0, 0.0, -1.0])
    y_wrap = np.fft.fft_(x)
    y_real = real_np.fft.fft(x)
    assert real_np.allclose(y_wrap, y_real)
    print("  OK np.fft.fft_() == numpy.fft.fft()")


def test_acces_sans_suffixe_raise():
    """np.add (sans _) doit lever MissingSuffixError."""
    try:
        _ = np.add(1, 2)
        assert False, "np.add devrait lever AttributeError"
    except AttributeError as e:
        assert "add" in str(e) and "_" in str(e)
        print("  OK np.add() sans suffixe lève AttributeError")

def test_sous_module_sans_suffix():
    """Autre sous-module : np.fft.fft(x) doit lever ModuleWithSuffixError"""
    x = real_np.array([0.0, 1.0, 0.0, -1.0])
    try: 
        _ = np.fft.fft(x)
        assert False, "np.fft.fft devrait lever AttributeError"
    except AttributeError as e:
        print("  OK np.fft.fft lève une AttributeError")

def test_sous_module_avec_suffix_au_milieu():
    """Autre sous-module : np.fft.fft(x) doit lever ModuleWithSuffixError"""
    x = real_np.array([0.0, 1.0, 0.0, -1.0])
    try: 
        _ = np.fft_.fft(x)
        assert False, "np.fft_.fft devrait lever AttributeError"
    except AttributeError as e:
        print("  OK np.fft_.fft lève une AttributeError")

def run_all():
    print("Validation WrapUnderscoreNumpy (suffixe '_', mode strict)")
    print("-" * 50)
    test_ufunc_simple()
    test_ufunc_array()
    test_sous_module()
    test_constante()
    test_constantes_et_types()
    test_array_creation()
    test_sous_module_autre()
    test_acces_sans_suffixe_raise()
    test_sous_module_sans_suffix()
    test_sous_module_avec_suffix_au_milieu()
    print("-" * 50)
    print("Tous les tests _ sont passés.")


if __name__ == "__main__":
    run_all()
