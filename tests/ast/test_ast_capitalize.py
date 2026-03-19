"""
Tests unitaires simples pour src/ast_cleaning_capitalize.py

On vérifie ici, en isolation :
- qu'un attribut/méthode d'objet avec une première lettre en majuscule est accepté et normalisé ;
- qu'un attribut/méthode d'objet sans première lettre en majuscule déclenche une erreur ;
- que les accès `np.xxx` ne sont pas modifiés par ce module.
"""

import importlib.util
import os

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_module_path = os.path.join(_project_root, "src", "ast_cleaning_capitalize.py")

_spec = importlib.util.spec_from_file_location("project_ast_tools", _module_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Unable to load module from {_module_path}")
project_ast_tools = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(project_ast_tools)

normalize_object_attributes = project_ast_tools.normalize_object_attributes
ObjectAttributeError = project_ast_tools.ObjectAttributeError


def test_shape__is_normalized():
    code = "result = A.Shape"
    out = normalize_object_attributes(code)
    assert "Shape" not in out
    assert "A.shape" in out
    print("  OK A.Shape -> A.shape")


def test_reshape__is_normalized():
    code = "result = A.Reshape((1, 9))"
    out = normalize_object_attributes(code)
    assert "Reshape" not in out
    assert "A.reshape((1, 9))" in out
    print("  OK A.Reshape((1, 9)) -> A.reshape((1, 9))")


def test_chained_attributes_are_normalized():
    code = "result = A.T.Shape"
    out = normalize_object_attributes(code)
    assert "Shape" not in out
    assert "A.T.shape" in out
    print("  OK A.T.Shape -> A.T.shape")


def test_missing_shape_suffix_raises():
    code = "result = A.shape"
    try:
        normalize_object_attributes(code)
        assert False, "Expected ObjectAttributeError for A.shape"
    except ObjectAttributeError as e:
        assert "shape" in str(e)
    print("  OK A.shape sans suffixe -> erreur")


def test_missing_method_suffix_raises():
    code = "result = A.reshape((1, 9))"
    try:
        normalize_object_attributes(code)
        assert False, "Expected ObjectAttributeError for A.reshape((1, 9))"
    except ObjectAttributeError as e:
        assert "reshape" in str(e)
    print("  OK A.reshape(...) sans suffixe -> erreur")


def test_numpy_calls_are_left_unchanged():
    code = "result = np.Mean(A)"
    out = normalize_object_attributes(code)
    assert "np.Mean(A)" in out
    print("  OK np.Mean(A) laissé inchangé")


def test_of_elt_that_should_raise_error():
    code = "result = A.shape"
    try:
        normalize_object_attributes(code)
        assert False, "Expected ObjectAttributeError for A.shape"
    except ObjectAttributeError as e:
        assert "shape" in str(e)
    print("  OK A.shape sans suffixe -> erreur")


def test_of_elt_that_should_raise_error2():
    code = "result = A.T.shape"
    try:
        normalize_object_attributes(code)
        assert False, "Expected ObjectAttributeError for A.T.shape"
    except ObjectAttributeError as e:
        assert "shape" in str(e)
    print("  OK A.T.shape sans suffixe -> erreur")



def test_of_np_and_method_is_left_unchanged() :
    code = "result = np.array([1, 2, 3]).T"
    out = normalize_object_attributes(code)
    assert "np.array([1, 2, 3]).T" in out
    print("  OK np.array([1, 2, 3]).T est laissé inchangé")


def test_of_np_and_method_is_left_unchanged_that_raise_error() :
    code = "result = np.array([1, 2, 3]).reshape((1, 3))"
    try:
        normalize_object_attributes(code)
        assert False, "Expected ObjectAttributeError for np.array([1, 2, 3]).reshape((1, 3))"
    except ObjectAttributeError as e:
        assert "reshape" in str(e)
    print("  OK np.array([1, 2, 3]).reshape((1, 3)) sans suffixe -> erreur")


def test_of_np_and_method_linalg() :
    code = "result = np.linalg.Norm(A)"
    out = normalize_object_attributes(code)
    assert "np.linalg.Norm(A)" in out
    print("  OK np.linalg.Norm(A) inchangé")



def test_of_np_and_method_linalg2() :
    code = "result = np.Linalg.Norm(A)"
    out = normalize_object_attributes(code)
    assert "np.Linalg.Norm(A)" in out
    print("  OK np.Linalg.Norm est laissé inchangé")



def run_all():
    print("Validation AST _ sur attributs/méthodes d'objets")
    print("-" * 60)
    test_shape__is_normalized()
    test_reshape__is_normalized()
    test_chained_attributes_are_normalized()
    test_missing_shape_suffix_raises()
    test_missing_method_suffix_raises()
    test_numpy_calls_are_left_unchanged()
    test_of_elt_that_should_raise_error()
    test_of_elt_that_should_raise_error2()
    test_of_np_and_method_is_left_unchanged()
    test_of_np_and_method_is_left_unchanged_that_raise_error()
    test_of_np_and_method_linalg()
    test_of_np_and_method_linalg2()
    print("-" * 60)
    print("Tous les tests AST _ sont passés.")


if __name__ == "__main__":
    run_all()

