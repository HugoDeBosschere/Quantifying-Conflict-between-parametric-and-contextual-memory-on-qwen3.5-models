import numpy
import inspect
import types
import os
from inspect import signature
import re
import scipy
import matplotlib.pyplot
import pandas
import tqdm


test_doc = numpy.acos.__doc__


def _first_doc_line(doc):
    if not doc:
        return ""
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def discover_capitalize_extra_elements():
    constants, dtypes, ndarray_attrs, ndarray_methods = [], [], [], []
    for name in sorted(dir(numpy)):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(numpy, name)
        except Exception:
            continue
        if inspect.isfunction(obj) or isinstance(obj, numpy.ufunc) or isinstance(obj, types.ModuleType):
            continue
        if isinstance(obj, type):
            try:
                if issubclass(obj, numpy.generic):
                    dtypes.append((name, _first_doc_line(getattr(obj, "__doc__", ""))))
            except Exception:
                pass
            continue
        try:
            is_scalar_like = numpy.isscalar(obj) or obj is None
        except Exception:
            is_scalar_like = obj is None
        if is_scalar_like:
            constants.append((name, _first_doc_line(getattr(obj, "__doc__", ""))))
    for name in sorted(dir(numpy.ndarray)):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(numpy.ndarray, name)
        except Exception:
            continue
        if callable(obj):
            ndarray_methods.append((name, _first_doc_line(getattr(obj, "__doc__", ""))))
        else:
            ndarray_attrs.append((name, _first_doc_line(getattr(obj, "__doc__", ""))))
    return {
        "constants": constants,
        "dtypes": dtypes,
        "ndarray_attrs": ndarray_attrs,
        "ndarray_methods": ndarray_methods,
    }


def write_capitalize_extra_full(f, extras):
    f.write("EXTRA ALIASES (constants, dtypes, ndarray attrs/methods)\n")
    f.write("=" * 60 + "\n\n")
    for name, first in extras["constants"]:
        alias = capitalize_first(name)
        f.write(f"ALIAS: numpy.{alias}\nMaps to: numpy.{name}\n")
        if first:
            f.write(f"Definition: {first}\n")
        f.write(f"Example: np.{alias}\n" + "#" * 40 + "\n")
    for name, first in extras["dtypes"]:
        alias = capitalize_first(name)
        f.write(f"ALIAS: numpy.{alias}\nMaps to: numpy.{name}\n")
        if first:
            f.write(f"Definition: {first}\n")
        f.write(f"Example: np.{alias}\n" + "#" * 40 + "\n")
    for name, first in extras["ndarray_attrs"]:
        alias = capitalize_first(name)
        f.write(f"ALIAS: <ndarray>.{alias}\nMaps to: <ndarray>.{name}\n")
        if first:
            f.write(f"Definition: {first}\n")
        f.write(f"Example: A.{alias}\n" + "#" * 40 + "\n")
    for name, first in extras["ndarray_methods"]:
        alias = capitalize_first(name)
        f.write(f"ALIAS: <ndarray>.{alias}(...)\nMaps to: <ndarray>.{name}(...)\n")
        if first:
            f.write(f"Definition: {first}\n")
        f.write(f"Example: A.{alias}(...)\n" + "#" * 40 + "\n")


def write_capitalize_extra_minimal(f, extras):
    f.write("EXTRA ALIASES (minimal)\n")
    f.write("=" * 60 + "\n\n")
    for bucket in ("constants", "dtypes", "ndarray_attrs", "ndarray_methods"):
        for name, _ in extras[bucket]:
            alias = capitalize_first(name)
            prefix = "numpy" if bucket in {"constants", "dtypes"} else "ndarray"
            sig = "(...)" if bucket == "ndarray_methods" else ""
            f.write(f"FUNCTION: {prefix}.{alias}\n\n{alias}{sig}\n" + "#" * 40 + "\n")


def write_capitalize_extra_ultra(f, extras):
    f.write("EXTRA ALIASES (ultra-minimal)\n")
    f.write("=" * 60 + "\n\n")
    for bucket in ("constants", "dtypes", "ndarray_attrs", "ndarray_methods"):
        for name, _ in extras[bucket]:
            alias = capitalize_first(name)
            prefix = "numpy" if bucket in {"constants", "dtypes"} else "ndarray"
            f.write(f"FUNCTION: {prefix}.{alias}\n\n")


def capitalize_first(name):
    """Met en majuscule la première lettre d'un nom de fonction."""
    if not name:
        return name
    return name[0].upper() + name[1:]


def add_capitalize_suffix(docstring, shorthand):
    """
    shorthand : the shorthand name of a module to know where to capitalize
    docstring : docstring that is parsed

    Returns :
    new_s the string where each expression of the type shorthand.function(args)
    has its first letter capitalized : shorthand.Function(args)
    """
    pattern = re.escape(shorthand) + r"\.(\w+)"

    def _cap(m, _sh=shorthand):
        func_name = m.group(1)
        return f"{_sh}.{func_name[0].upper()}{func_name[1:]}"

    new_docstring = re.sub(pattern, _cap, docstring)
    return new_docstring


def add_capitalize_signature_full_doc(docstring):
    """
    Capitalize the first letter of the function name in the signature
    at the beginning of the docstring.

    e.g. array(... -> Array(...
    """
    new_text = re.sub(r"^(\w)", lambda m: m.group(1).upper(), docstring)
    return new_text


def add_capitalize_signature(docstring):
    """
    Returns the signature of a function with first letter capitalized,
    if this signature exists.
    """
    first_pattern = re.search(".*?(?=\n)", docstring)

    if first_pattern:
        pattern = re.search(r"(\w+)\(.*", first_pattern.group(0))
        if pattern:
            matched = pattern.group(0)
            new_signature = matched[0].upper() + matched[1:]
        else:
            new_signature = ""
    else:
        new_signature = ""
    return new_signature


def supress_see_also(docstring):
    """
    Supress the see_also section of a docstring because it is too unpredictable
    and could compromise the coherence of a documentation.
    """
    new_see_also = re.sub("See Also\n-*\n*.*?(?=\n\n)", "", docstring, flags=re.DOTALL)
    new_see_also = re.sub(r"See also\n-*\n*.*?(?=\n\s*\n|$)", "", new_see_also, flags=re.DOTALL)
    new_see_also = re.sub(r"See Also\n-*\n*.*?(?=\n\s*\n|$)", "", new_see_also, flags=re.DOTALL)
    return new_see_also


def corrupt_doc(doc_text, shorthand):
    """
    Function that corrupts a doc_text capitalizing the first letter of function
    names if it spots a pattern of the type shorthand.(\\w+)
    """
    if not doc_text:
        return ""
    new_text = add_capitalize_suffix(doc_text, shorthand)
    return new_text


def generate_full_docs(list_module, list_shorthand, output_file):
    """
    Generates the full corrupted doc at output_file by capitalizing the first letter
    at all the relevant places (function name, signature, examples) and suppressing
    the see also section.
    """
    seen_functions = set()
    extras = discover_capitalize_extra_elements()
    with open(output_file, 'w', encoding='utf-8') as f:
        for base_module in tqdm.tqdm(list_module):
            f.write(f"Reference Documentation for {base_module.__name__} \n")
            f.write("=" * 60 + "\n\n")

            stack = [(base_module, base_module.__name__)]
            visited_modules = set()

            while stack:
                current_mod, prefix = stack.pop()

                if current_mod in visited_modules:
                    continue
                visited_modules.add(current_mod)

                for name in dir(current_mod):
                    if name.startswith("_"):
                        continue

                    try:
                        obj = getattr(current_mod, name)
                    except Exception:
                        continue

                    if inspect.isfunction(obj) or isinstance(obj, numpy.ufunc):
                        if obj in seen_functions:
                            continue
                        seen_functions.add(obj)

                        full_name = f"{prefix}.{capitalize_first(name)}"

                        raw_doc = obj.__doc__
                        new_doc = raw_doc

                        if new_doc:
                            new_doc = add_capitalize_signature_full_doc(new_doc)

                            f.write(f"FUNCTION: {full_name}\n")
                            f.write("\n")
                            new_doc = supress_see_also(new_doc)
                            for shorthand in list_shorthand:
                                new_doc = corrupt_doc(new_doc, shorthand)

                            f.write(new_doc + "\n")
                            f.write("\n" + "#" * 40 + "\n\n")

                    elif isinstance(obj, types.ModuleType):
                        if hasattr(obj, '__name__') and 'numpy' in obj.__name__:
                            stack.append((obj, f"{prefix}.{name}"))
            if getattr(base_module, "__name__", "") == "numpy":
                write_capitalize_extra_full(f, extras)

    print(f"Documentation generated in {output_file}")
    print(f"Total unique functions documented: {len(seen_functions)}")


def generate_ultra_minimal_docs(list_module, output_file):
    """
    Generate a doc with only the capitalized function names and nothing else.
    """
    seen_functions = set()
    extras = discover_capitalize_extra_elements()
    with open(output_file, 'w', encoding='utf-8') as f:
        for base_module in tqdm.tqdm(list_module):
            f.write(f"Reference Documentation for {base_module.__name__} \n")
            f.write("=" * 60 + "\n\n")

            stack = [(base_module, base_module.__name__)]
            visited_modules = set()

            while stack:
                current_mod, prefix = stack.pop()

                if current_mod in visited_modules:
                    continue
                visited_modules.add(current_mod)

                for name in dir(current_mod):
                    if name.startswith("_"):
                        continue

                    try:
                        obj = getattr(current_mod, name)
                    except Exception:
                        continue

                    if inspect.isfunction(obj) or isinstance(obj, numpy.ufunc):
                        if obj in seen_functions:
                            continue
                        seen_functions.add(obj)

                        full_name = f"{prefix}.{capitalize_first(name)}"
                        raw_doc = obj.__doc__

                        f.write(f"FUNCTION: {full_name}\n")
                        f.write("\n")

                    elif isinstance(obj, types.ModuleType):
                        if hasattr(obj, '__name__') and 'numpy' in obj.__name__:
                            stack.append((obj, f"{prefix}.{name}"))
            if getattr(base_module, "__name__", "") == "numpy":
                write_capitalize_extra_ultra(f, extras)

    print(f"Documentation generated in {output_file}")
    print(f"Total unique functions documented: {len(seen_functions)}")


def generate_minimal_docs(list_module, output_file):
    """
    Generate a doc with the capitalized function names and the signature,
    when it exists.
    """
    seen_functions = set()
    extras = discover_capitalize_extra_elements()
    with open(output_file, 'w', encoding='utf-8') as f:
        for base_module in tqdm.tqdm(list_module):
            f.write(f"Reference Documentation for {base_module.__name__} \n")
            f.write("=" * 60 + "\n\n")

            stack = [(base_module, base_module.__name__)]
            visited_modules = set()

            while stack:
                current_mod, prefix = stack.pop()

                if current_mod in visited_modules:
                    continue
                visited_modules.add(current_mod)

                for name in dir(current_mod):
                    if name.startswith("_"):
                        continue

                    try:
                        obj = getattr(current_mod, name)
                    except Exception:
                        continue

                    if inspect.isfunction(obj) or isinstance(obj, numpy.ufunc):
                        if obj in seen_functions:
                            continue
                        seen_functions.add(obj)

                        full_name = f"{prefix}.{capitalize_first(name)}"
                        raw_doc = obj.__doc__
                        new_doc = raw_doc

                        if new_doc:
                            modif = add_capitalize_signature(new_doc)
                            if modif:
                                new_doc = modif
                            else:
                                new_doc = ""
                            f.write(f"FUNCTION: {full_name}\n")
                            f.write("\n")
                            f.write(new_doc + "\n")
                            f.write("#" * 40 + "\n")

                    elif isinstance(obj, types.ModuleType):
                        if hasattr(obj, '__name__') and 'numpy' in obj.__name__:
                            stack.append((obj, f"{prefix}.{name}"))
            if getattr(base_module, "__name__", "") == "numpy":
                write_capitalize_extra_minimal(f, extras)

    print(f"Documentation generated in {output_file}")
    print(f"Total unique functions documented: {len(seen_functions)}")


if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    generate_full_docs([numpy], ["np", "numpy"], os.path.join(output_dir, "corrupted_full_doc_numpy_capitalize.txt"))
    generate_ultra_minimal_docs([numpy], output_file=os.path.join(output_dir, "corrupted_ultra_minimal_numpy_capitalize.txt"))
    generate_minimal_docs([numpy], output_file=os.path.join(output_dir, "corrupted_minimal_numpy_capitalize.txt"))
