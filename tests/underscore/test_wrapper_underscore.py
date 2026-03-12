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
    """np.linalg_.norm_(x) doit être égal à numpy.linalg.norm(x)."""
    x = real_np.array([3.0, 4.0])
    assert np.linalg_.norm_(x) == real_np.linalg.norm(x)
    print("  OK np.linalg_.norm_() == numpy.linalg.norm()")


def test_constante():
    """np.pi_ doit être égal à numpy.pi (le proxy enlève le _ final)."""
    assert np.pi_ == real_np.pi
    print("  OK np.pi_ == numpy.pi")


def test_array_creation():
    """np.array_([1,2,3]) doit créer le même tableau que numpy.array([1,2,3])."""
    arr_wrap = np.array_([1, 2, 3])
    arr_real = real_np.array([1, 2, 3])
    assert real_np.array_equal(arr_wrap, arr_real)
    print("  OK np.array_() == numpy.array()")


def run_all():
    print("Validation WrapUnderscoreNumpy (suffixe '_')")
    print("-" * 50)
    test_ufunc_simple()
    test_ufunc_array()
    test_sous_module()
    test_constante()
    test_array_creation()
    print("-" * 50)
    print("Tous les tests underscore sont passés.")


if __name__ == "__main__":
    run_all()
