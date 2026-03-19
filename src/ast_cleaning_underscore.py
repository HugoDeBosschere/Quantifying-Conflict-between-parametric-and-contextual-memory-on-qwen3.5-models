from __future__ import annotations
import ast


# ==============================================================================
# 1. AST Cleaning for trailing underscore suffix
# ==============================================================================


class ObjectAttributeError(ValueError):
    """Raised when an object attribute/method does not use the required trailing '_'."""


def _is_numpy_root(node) -> bool:
    """
    True si la chaîne d'attributs est enracinée en `np` ou `numpy`.

    Exemples:
    - `np.mean_`          -> True
    - `numpy.linalg.norm` -> True
    - `A.reshape_`        -> False
    - `obj.inner.shape_`  -> False
    """
    cur = node
    while isinstance(cur, ast.Attribute):
        cur = cur.value
    return isinstance(cur, ast.Name) and cur.id in {"np", "numpy"}


class _ObjectAttributeUnderscoreNormalizer(ast.NodeTransformer):
    """
    Transforme les attributs/méthodes d'objets pour enlever le suffixe `_`.

    Règle:
    - si l'attribut est enraciné en `np` ou `numpy`, on ne le touche pas ;
    - sinon, il DOIT se terminer par `_`, auquel cas on enlève le suffixe ;
    - sinon, on lève une erreur.
    """

    def visit_Attribute(self, node):
        node = self.generic_visit(node)

        if _is_numpy_root(node):
            return node

        if node.attr.endswith("_"):
            node.attr = node.attr[:-1]
            return node

        raise ObjectAttributeError(
            f"Object attribute/method '{node.attr}' is missing required trailing underscore suffix."
        )


def normalize_object_attributes(code: str) -> str:
    """
    Parse le code, exige que tous les attributs/méthodes d'objets
    utilisent `_`, enlève ce suffixe, puis retourne le code normalisé.

    Exemples:
    - `A.shape_`           -> `A.shape`
    - `A.reshape_((1, 9))` -> `A.reshape((1, 9))`
    - `A.shape`            -> ObjectAttributeError
    """
    tree = ast.parse(code)
    tree = _ObjectAttributeUnderscoreNormalizer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)

