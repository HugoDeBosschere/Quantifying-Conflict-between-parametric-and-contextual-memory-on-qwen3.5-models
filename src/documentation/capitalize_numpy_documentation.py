import numpy
import inspect
import types
from inspect import signature
import re
import scipy
import matplotlib.pyplot
import pandas
import tqdm


test_doc = numpy.acos.__doc__


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
                            f.write("-" * (10 + len(full_name)) + "\n")
                            new_doc = supress_see_also(new_doc)
                            for shorthand in list_shorthand:
                                new_doc = corrupt_doc(new_doc, shorthand)

                            f.write(new_doc + "\n")
                            f.write("\n" + "#" * 40 + "\n\n")

                    elif isinstance(obj, types.ModuleType):
                        if hasattr(obj, '__name__') and 'numpy' in obj.__name__:
                            stack.append((obj, f"{prefix}.{name}"))

    print(f"Documentation generated in {output_file}")
    print(f"Total unique functions documented: {len(seen_functions)}")


def generate_ultra_minimal_docs(list_module, output_file):
    """
    Generate a doc with only the capitalized function names and nothing else.
    """
    seen_functions = set()
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
                        f.write("-" * (10 + len(full_name)) + "\n")

                    elif isinstance(obj, types.ModuleType):
                        if hasattr(obj, '__name__') and 'numpy' in obj.__name__:
                            stack.append((obj, f"{prefix}.{name}"))

    print(f"Documentation generated in {output_file}")
    print(f"Total unique functions documented: {len(seen_functions)}")


def generate_minimal_docs(list_module, output_file):
    """
    Generate a doc with the capitalized function names and the signature,
    when it exists.
    """
    seen_functions = set()
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
                            f.write("-" * (10 + len(full_name)) + "\n")
                            f.write(new_doc + "\n")
                            f.write("#" * 40 + "\n")

                    elif isinstance(obj, types.ModuleType):
                        if hasattr(obj, '__name__') and 'numpy' in obj.__name__:
                            stack.append((obj, f"{prefix}.{name}"))

    print(f"Documentation generated in {output_file}")
    print(f"Total unique functions documented: {len(seen_functions)}")


if __name__ == "__main__":
    generate_full_docs([numpy], ["np", "numpy"], "corrupted_full_doc_numpy_capitalize.txt")
    generate_ultra_minimal_docs([numpy], output_file="corrupted_ultra_minimal_numpy_capitalize.txt")
    generate_minimal_docs([numpy], output_file="corrupted_minimal_numpy_capitalize.txt")
