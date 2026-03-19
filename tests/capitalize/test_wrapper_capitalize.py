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


def test_array_creation():
    """np.Array([1,2,3]) doit créer le même tableau que numpy.array([1,2,3])."""
    arr_wrap = np.Array([1, 2, 3])
    arr_real = real_np.array([1, 2, 3])
    assert real_np.array_equal(arr_wrap, arr_real)
    print("  OK np.Array() == numpy.array()")


def test_linspace():
    """np.Linspace(0, 1, 5) doit être égal à numpy.linspace(0, 1, 5)."""
    wrap_result = np.Linspace(0, 1, 5)
    real_result = real_np.linspace(0, 1, 5)
    assert real_np.array_equal(wrap_result, real_result)
    print("  OK np.Linspace() == numpy.linspace()")


def test_random_submodule():
    """np.Random.Seed(42) puis np.Random.Rand(3) doit fonctionner."""
    np.Random.Seed(42)
    wrap_result = np.Random.Rand(3)
    real_np.random.seed(42)
    real_result = real_np.random.rand(3)
    assert real_np.array_equal(wrap_result, real_result)
    print("  OK np.Random.Seed() / np.Random.Rand() == numpy.random")


def test_acces_sans_majuscule_raise():
    """np.array (sans majuscule) doit lever AttributeError (comportement strict)."""
    try:
        _ = np.array([1, 2, 3])
        assert False, "np.array devrait lever AttributeError"
    except AttributeError as e:
        assert "array" in str(e) and ("capitalized" in str(e).lower() or "Capitalized" in str(e))
    print("  OK np.array() sans majuscule lève AttributeError")


def run_all():
    print("Validation WrapCapitalizeNumpy (première lettre majuscule, mode strict)")
    print("-" * 50)
    test_ufunc_simple()
    test_ufunc_array()
    test_sous_module()
    test_constante()
    test_array_creation()
    test_linspace()
    test_random_submodule()
    test_acces_sans_majuscule_raise()
    print("-" * 50)
    print("Tous les tests capitalize sont passés.")


if __name__ == "__main__":
    run_all()
