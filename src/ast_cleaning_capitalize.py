from __future__ import annotations
import ast


# ==============================================================================
# 1. AST Cleaning for Capitalized attributes/methods
# ==============================================================================


class ObjectAttributeError(ValueError):
    """Raised when an object attribute/method is not capitalized as required."""


_PYTHON_BUILTIN_METHODS: frozenset[str] = frozenset({
    # list
    "append", "extend", "insert", "remove", "pop", "clear", "index", "count",
    "sort", "reverse", "copy",
    # dict
    "keys", "values", "items", "get", "update", "setdefault", "popitem", "fromkeys",
    # str
    "strip", "lstrip", "rstrip", "split", "rsplit", "splitlines", "join",
    "format", "format_map", "encode", "decode", "upper", "lower", "title",
    "capitalize", "swapcase", "replace", "find", "rfind", "startswith", "endswith",
    "isdigit", "isalpha", "isalnum", "isspace", "isupper", "islower", "istitle",
    "zfill", "ljust", "rjust", "center", "expandtabs",
    # set
    "add", "discard", "union", "intersection", "difference",
    "symmetric_difference", "issubset", "issuperset", "isdisjoint",
    # file / io
    "read", "write", "close", "flush", "seek", "tell", "readline", "readlines",
    "writelines", "truncate",
})


def _is_numpy_root(node) -> bool:
    """
    True si la chaîne d'attributs est enracinée en `np` ou `numpy`.

    Exemples:
    - `np.Mean`            -> True
    - `numpy.linalg.norm`  -> True
    - `A.Reshape`          -> False
    - `obj.inner.Shape`    -> False
    """
    cur = node
    while isinstance(cur, ast.Attribute):
        cur = cur.value
    return isinstance(cur, ast.Name) and cur.id in {"np", "numpy"}


def _uncapitalize_first(name: str) -> str:
    if not name:
        return name
    return name[0].lower() + name[1:]


class _ObjectAttributeCapitalizeNormalizer(ast.NodeTransformer):
    """
    Transforme les attributs/méthodes d'objets capitalisés en forme normale.

    Règle:
    - si l'attribut est enraciné en `np` ou `numpy`, on ne le touche pas ;
    - sinon, il DOIT commencer par une majuscule, puis on repasse la première
      lettre en minuscule ;
    - sinon, on lève une erreur.
    """

    def visit_Attribute(self, node):
        node = self.generic_visit(node)

        if _is_numpy_root(node):
            return node

        if node.attr in _PYTHON_BUILTIN_METHODS:
            return node

        # Cas particulier : en NumPy, `.T` (transposée) doit rester majuscule.
        # Sans ça, l'un-капиталisation convertirait `.T` en `.t`, ce qui casse la sémantique.
        if node.attr == "T":
            return node

        if node.attr and node.attr[0].isupper():
            node.attr = _uncapitalize_first(node.attr)
            return node

        raise ObjectAttributeError(
            f"Object attribute/method '{node.attr}' is not capitalized as required."
        )


def normalize_object_attributes(code: str) -> str:
    """
    Parse le code, exige que tous les attributs/méthodes d'objets
    commencent par une majuscule, puis normalise en minuscule initiale.

    Exemples:
    - `A.Shape`             -> `A.shape`
    - `A.Reshape((1, 9))`   -> `A.reshape((1, 9))`
    - `A.shape`             -> ObjectAttributeError
    """
    tree = ast.parse(code)
    tree = _ObjectAttributeCapitalizeNormalizer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)

