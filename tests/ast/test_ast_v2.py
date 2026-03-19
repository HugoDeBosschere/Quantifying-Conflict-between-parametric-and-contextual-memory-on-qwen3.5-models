"""
Tests unitaires simples pour src/ast_cleaning.py

On vérifie ici, en isolation :
- qu'un attribut/méthode d'objet avec suffixe `_v2` est accepté et normalisé ;
- qu'un attribut/méthode d'objet sans suffixe `_v2` déclenche une erreur ;
- que les accès `np.xxx` ne sont pas modifiés par ce module.
"""

import importlib.util
import os


_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_module_path = os.path.join(_project_root, "src", "ast_cleaning.py")

_spec = importlib.util.spec_from_file_location("project_ast_tools", _module_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Unable to load module from {_module_path}")
project_ast_tools = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(project_ast_tools)

normalize_v2_object_attributes = project_ast_tools.normalize_v2_object_attributes
V2ObjectAttributeError = project_ast_tools.V2ObjectAttributeError


def test_shape_v2_is_normalized():
    code = "result = A.shape_v2"
    out = normalize_v2_object_attributes(code)
    assert "shape_v2" not in out
    assert "A.shape" in out
    print("  OK A.shape_v2 -> A.shape")


def test_reshape_v2_is_normalized():
    code = "result = A.reshape_v2((1, 9))"
    out = normalize_v2_object_attributes(code)
    assert "reshape_v2" not in out
    assert "A.reshape((1, 9))" in out
    print("  OK A.reshape_v2((1, 9)) -> A.reshape((1, 9))")


def test_chained_attributes_are_normalized():
    code = "result = A.T_v2.shape_v2"
    out = normalize_v2_object_attributes(code)
    assert "T_v2" not in out
    assert "shape_v2" not in out
    assert "A.T.shape" in out
    print("  OK A.T_v2.shape_v2 -> A.T.shape")


def test_missing_shape_suffix_raises():
    code = "result = A.shape"
    try:
        normalize_v2_object_attributes(code)
        assert False, "Expected V2ObjectAttributeError for A.shape"
    except V2ObjectAttributeError as e:
        assert "shape" in str(e)
    print("  OK A.shape sans suffixe -> erreur")


def test_missing_method_suffix_raises():
    code = "result = A.reshape((1, 9))"
    try:
        normalize_v2_object_attributes(code)
        assert False, "Expected V2ObjectAttributeError for A.reshape((1, 9))"
    except V2ObjectAttributeError as e:
        assert "reshape" in str(e)
    print("  OK A.reshape(...) sans suffixe -> erreur")


def test_numpy_calls_are_left_unchanged():
    code = "result = np.mean_v2(A)"
    out = normalize_v2_object_attributes(code)
    assert "np.mean_v2(A)" in out
    print("  OK np.mean_v2(A) laissé inchangé")


def test_of_elt_that_should_raise_error():
    code = "result = A.shape"
    try:
        normalize_v2_object_attributes(code)
        assert False, "Expected V2ObjectAttributeError for A.shape"
    except V2ObjectAttributeError as e:
        assert "shape" in str(e)
    print("  OK A.shape sans suffixe -> erreur")


def test_of_elt_that_should_raise_error2():
    code = "result = A.T_v2.shape"
    try:
        normalize_v2_object_attributes(code)
        assert False, "Expected V2ObjectAttributeError for A.T_v2.shape"
    except V2ObjectAttributeError as e:
        assert "shape" in str(e)
    print("  OK A.T_v2.shape sans suffixe -> erreur")


def test_of_elt_that_should_raise_error3():
    code = "result = A.T.shape_v2"
    try:
        normalize_v2_object_attributes(code)
        assert False, "Expected V2ObjectAttributeError for A.T.shape_v2"
    except V2ObjectAttributeError as e:
        assert "T" in str(e)
    print("  OK A.T.shape_v2 sans suffixe -> erreur")


def test_of_np_and_method_is_left_unchanged() :
    code = "result = np.array([1, 2, 3]).T_v2"
    out = normalize_v2_object_attributes(code)
    assert "np.array([1, 2, 3]).T" in out
    print("  OK np.array([1, 2, 3]).T_v2 -> np.array([1, 2, 3]).T")


def test_of_np_and_method_is_left_unchanged_that_raise_error() :
    code = "result = np.array([1, 2, 3]).reshape((1, 3))"
    try:
        normalize_v2_object_attributes(code)
        assert False, "Expected V2ObjectAttributeError for np.array([1, 2, 3]).reshape((1, 3))"
    except V2ObjectAttributeError as e:
        assert "reshape" in str(e)
    print("  OK np.array([1, 2, 3]).reshape((1, 3)) sans suffixe -> erreur")


def test_of_np_and_method_linalg() :
    code = "result = np.linalg.norm_v2(A)"
    out = normalize_v2_object_attributes(code)
    assert "np.linalg.norm_v2(A)" in out
    print("  OK np.linalg.norm_v2(A) inchangé")



def test_of_np_and_method_linalg2() :
    code = "result = np.linalg_v2.norm_v2(A)"
    out = normalize_v2_object_attributes(code)
    assert "np.linalg_v2.norm_v2(A)" in out
    print("  OK np.linalg_v2.norm_v2 est laissé inchangé")



def run_all():
    print("Validation AST _v2 sur attributs/méthodes d'objets")
    print("-" * 60)
    test_shape_v2_is_normalized()
    test_reshape_v2_is_normalized()
    test_chained_attributes_are_normalized()
    test_missing_shape_suffix_raises()
    test_missing_method_suffix_raises()
    test_numpy_calls_are_left_unchanged()
    test_of_elt_that_should_raise_error()
    test_of_elt_that_should_raise_error2()
    test_of_elt_that_should_raise_error3()
    test_of_np_and_method_is_left_unchanged()
    test_of_np_and_method_is_left_unchanged_that_raise_error()
    test_of_np_and_method_linalg()
    test_of_np_and_method_linalg2()
    print("-" * 60)
    print("Tous les tests AST _v2 sont passés.")


if __name__ == "__main__":
    run_all()

