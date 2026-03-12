"""
Validation du wrapper WrapV2Numpy : vérification que np.fonction_v2()
redirige bien vers numpy.fonction() et donne le même résultat.
"""
import os
import sys

# Ajouter src au path pour importer les wrappers
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_src_path = os.path.join(_project_root, "src")
sys.path.insert(0, _src_path)

import numpy as real_np
import WrapV2Numpy as np


def test_ufunc_simple():
    """np.add_v2(a, b) doit être égal à numpy.add(a, b)."""
    a, b = 1, 2
    assert np.add_v2(a, b) == real_np.add(a, b)
    print("  OK np.add_v2() == numpy.add()")


def test_ufunc_array():
    """np.mean_v2(arr) doit être égal à numpy.mean(arr)."""
    arr = real_np.array([1.0, 2.0, 3.0])
    assert np.mean_v2(arr) == real_np.mean(arr)
    print("  OK np.mean_v2() == numpy.mean()")


def test_sous_module():
    """np.linalg_v2.norm_v2(x) doit être égal à numpy.linalg.norm(x)."""
    x = real_np.array([3.0, 4.0])
    assert np.linalg_v2.norm_v2(x) == real_np.linalg.norm(x)
    print("  OK np.linalg_v2.norm_v2() == numpy.linalg.norm()")


def test_constante():
    """np.pi_v2 doit être égal à numpy.pi (le proxy enlève _v2)."""
    assert np.pi_v2 == real_np.pi
    print("  OK np.pi_v2 == numpy.pi")


def test_array_creation():
    """np.array_v2([1,2,3]) doit créer le même tableau que numpy.array([1,2,3])."""
    arr_wrap = np.array_v2([1, 2, 3])
    arr_real = real_np.array([1, 2, 3])
    assert real_np.array_equal(arr_wrap, arr_real)
    print("  OK np.array_v2() == numpy.array()")


def test_acces_sans_suffixe_raise():
    """np.add (sans _v2) doit lever AttributeError (comportement strict)."""
    try:
        _ = np.add(1, 2)
        assert False, "np.add devrait lever AttributeError"
    except AttributeError as e:
        assert "add" in str(e) and "_v2" in str(e)
    print("  OK np.add() sans suffixe lève AttributeError")


def run_all():
    print("Validation WrapV2Numpy (suffixe '_v2', mode strict)")
    print("-" * 50)
    test_ufunc_simple()
    test_ufunc_array()
    test_sous_module()
    test_constante()
    test_array_creation()
    test_acces_sans_suffixe_raise()
    print("-" * 50)
    print("Tous les tests _v2 sont passés.")


if __name__ == "__main__":
    run_all()
