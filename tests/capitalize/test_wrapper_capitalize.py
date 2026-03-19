"""
Validation du wrapper WrapCapitalizeNumpy : vérification que np.Fonction()
redirige bien vers numpy.fonction() et donne le même résultat.
"""
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_src_path = os.path.join(_project_root, "src")
sys.path.insert(0, _src_path)

import numpy as real_np
import WrapCapitalizeNumpy as np


def test_ufunc_simple():
    """np.Add(a, b) doit être égal à numpy.add(a, b)."""
    a, b = 1, 2
    assert np.Add(a, b) == real_np.add(a, b)
    print("  OK np.Add() == numpy.add()")


def test_ufunc_array():
    """np.Mean(arr) doit être égal à numpy.mean(arr)."""
    arr = real_np.array([1.0, 2.0, 3.0])
    assert np.Mean(arr) == real_np.mean(arr)
    print("  OK np.Mean() == numpy.mean()")


def test_sous_module():
    """np.Linalg.Norm(x) doit être égal à numpy.linalg.norm(x)."""
    x = real_np.array([3.0, 4.0])
    assert np.Linalg.Norm(x) == real_np.linalg.norm(x)
    print("  OK np.Linalg.Norm() == numpy.linalg.norm()")


def test_constante():
    """np.Pi doit être égal à numpy.pi (le proxy met en minuscule la 1ère lettre)."""
    assert np.Pi == real_np.pi
    print("  OK np.Pi == numpy.pi")

def test_constantes_et_types():
    """Constantes et dtypes : np.Nan == np.nan, np.Float64 == np.float64, etc."""
    assert np.Nan is real_np.nan
    assert np.Inf is real_np.inf
    assert np.Float64 is real_np.float64
    assert np.Int32 is real_np.int32
    print("  OK constantes/dtypes _ :")
    print("     - np.Nan is np.nan")
    print("     - np.Inf is np.inf")
    print("     - np.Float64 is np.float64")
    print("     - np.Int32 is np.int32")

def test_array_creation():
    """np.Array([1,2,3]) doit créer le même tableau que numpy.array([1,2,3])."""
    arr_wrap = np.Array([1, 2, 3])
    arr_real = real_np.array([1, 2, 3])
    assert real_np.array_equal(arr_wrap, arr_real)
    print("  OK np.Array() == numpy.array()")

def test_sous_module_autre():
    """Autre sous-module : np.fft.Fft(x) == numpy.fft.fft(x)."""
    x = real_np.array([0.0, 1.0, 0.0, -1.0])
    y_wrap = np.fft.Fft(x)
    y_real = real_np.fft.fft(x)
    assert real_np.allclose(y_wrap, y_real)
    print("  OK np.fft.Fft() == numpy.fft.fft()")


def test_acces_sans_capitale_raise():
    """np.add (sans majuscule) doit lever AttributeError."""
    try:
        _ = np.add(1, 2)
        assert False, "np.add devrait lever AttributeError"
    except AttributeError as e:
        assert "add" in str(e)
        print("  OK np.add() sans majuscule lève AttributeError")

def test_sous_module_sans_capital():
    """Autre sous-module : np.fft.fft(x) doit lever AttributeError"""
    x = real_np.array([0.0, 1.0, 0.0, -1.0])
    try: 
        _ = np.fft.fft(x)
        assert False, "np.fft.fft devrait lever AttributeError"
    except AttributeError as e:
        print("  OK np.fft.fft lève une AttributeError")

def test_sous_module_avec_capital_au_milieu():
    """Autre sous-module : np.fft.fft(x) doit lever AttributeError"""
    x = real_np.array([0.0, 1.0, 0.0, -1.0])
    try: 
        _ = np.Fft.fft(x)
        assert False, "np.Fft.fft devrait lever AttributeError"
    except AttributeError as e:
        print("  OK np.Fft.fft lève une AttributeError")


def run_all():
    print("Validation WrapCapitalizeNumpy (première lettre majuscule, mode strict)")
    print("-" * 50)
    test_ufunc_simple()
    test_ufunc_array()
    test_sous_module()
    test_constante()
    test_constantes_et_types()
    test_array_creation()
    test_sous_module_autre()
    test_acces_sans_capitale_raise()
    test_sous_module_sans_capital()
    test_sous_module_avec_capital_au_milieu()
    print("-" * 50)
    print("Tous les tests capitalize sont passés.")


if __name__ == "__main__":
    run_all()
