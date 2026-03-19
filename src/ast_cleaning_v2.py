from __future__ import annotations
import ast



#==============================================================================
# 1. AST Cleaning for _v2 suffix
#==============================================================================


class ObjectAttributeError(ValueError):
    """Raised when an object attribute/method does not use the required `_v2` suffix."""


def _is_numpy_root(node) -> bool:
    """
    True si la chaîne d'attributs est enracinée en `np` ou `numpy`.

    Exemples:
    - `np.mean_v2`          -> True
    - `numpy.linalg.norm`   -> True
    - `A.reshape_v2`        -> False
    - `obj.inner.shape_v2`  -> False
    """
    cur = node
    while isinstance(cur, ast.Attribute):
        cur = cur.value
    return isinstance(cur, ast.Name) and cur.id in {"np", "numpy"}


class _ObjectAttributeV2Normalizer(ast.NodeTransformer):
    """
    Transforme les attributs/méthodes d'objets pour enlever le suffixe `_v2`.

    Règle:
    - si l'attribut est enraciné en `np` ou `numpy`, on ne le touche pas ;
    - sinon, il DOIT se terminer par `_v2`, auquel cas on enlève le suffixe ;
    - sinon, on lève une erreur.
    """

    def visit_Attribute(self, node):
        node = self.generic_visit(node)

        # On ignore les accès module-level NumPy dans ce module.
        if _is_numpy_root(node):
            return node

        if node.attr.endswith("_v2"):
            node.attr = node.attr[: -len("_v2")]
            return node

        raise V2ObjectAttributeError(
            f"Object attribute/method '{node.attr}' is missing required _v2 suffix."
        )


def normalize_object_attributes(code: str) -> str:
    """
    Parse le code, exige que tous les attributs/méthodes d'objets
    utilisent `_v2`, enlève ce suffixe, puis retourne le code normalisé.

    Exemples:
    - `A.shape_v2`           -> `A.shape`
    - `A.reshape_v2((1, 9))` -> `A.reshape((1, 9))`
    - `A.shape`              -> V2ObjectAttributeError
    """
    tree = ast.parse(code)
    tree = _ObjectAttributeV2Normalizer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)

