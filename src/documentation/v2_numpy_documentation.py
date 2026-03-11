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


def add_v2_suffix(docstring, shorthand):
    """
    shorthand : the shorthand name of a module to know where to add the _v2 suffix
    docstring : docstring that is parsed 

    Returns : 
    new_s the string where each expression of the type shorthand.function(args) has _v2 added : shorthand.function_v2(args)
    """
    new_docstring = re.sub(shorthand + r"\.(\w+)", shorthand + r".\1_v2", docstring)
    return new_docstring


def add_v2_signature_full_doc(docstring):
    """
    The function used to add _v2 to the signature of a function in the generate_full_doc function.

    docstring : str -> docstring that is parsed

    Returns : 
    new_text the docstring where the signature has _v2. 
    """
    new_text = re.sub(r"^(\w+)\(", r'\1_v2(', docstring)
    return new_text


def add_v2_signature(docstring):
    """
    returns the signature of a function with _v2 added, if this signature exists.

    docstring : str -> docstring that is parsed

    Returns : 
    new_signature : the signature of the function with _v2 if this signature exists. An empty str if it doesn't 

    """
    first_pattern = re.search(".*?(?=\n)", docstring)
    
    if first_pattern:
        pattern = re.search(r"(\w+)\(.*", first_pattern.group(0))
        if pattern:
            new_signature = re.sub(r"^(\w+)\(", r"\1_v2(", pattern.group(0))
        else:
            new_signature = "" 
    else:
        new_signature = ""
    return new_signature


def supress_see_also(docstring):
    """
    Supress the see_also section of a docstring because it is too unpredictable and could compromise the coherence of a documentation
    There are approximately two types of See Also sections : 
    1st : cos, arctan, arcsin, emath.arccos
    2nd :   arctan2 : The "four quadrant" arctan of the angle formed by (`x`, `y`)
    and the positive `x`-axis.
            angle : Argument of complex values.
    """
    new_see_also = re.sub("See Also\n-*\n*.*?(?=\n\n)", "", docstring, flags=re.DOTALL)
    new_see_also = re.sub(r"See also\n-*\n*.*?(?=\n\s*\n|$)", "", new_see_also, flags=re.DOTALL)
    new_see_also = re.sub(r"See Also\n-*\n*.*?(?=\n\s*\n|$)", "", new_see_also, flags=re.DOTALL)
    return new_see_also


def corrupt_doc(doc_text, shorthand):
    """
    Function that corrupts a doc_text adding _v2 if it spots a pattern of the type shorthand.(\w+)

    doc_text : str this is the doc to corrupt
    shorthand : the shorthand of a module that we want to corrupt 

    returns : str new_text the corrupted text 
    """
    if not doc_text:
        return ""
    new_text = add_v2_suffix(doc_text, shorthand)
    return new_text


# --- 2. The Crawler ---
def generate_full_docs(list_module, list_shorthand, output_file):
    """
    Generates the full corrupted doc at output_file by adding _v2 at all the relevant places (the function name
    the signature of the function and the examples) and suppressing the see also section.

    list_module : list of module names. the documentation of all the modules to corrupt. 
    list_shorthand : the list of all the shorthands used in the documentation to spot where to corrupt the documentation (np for numpy of pd for pandas for example)
    output_file : the destination of the created file 
    """
    seen_functions = set()
    with open(output_file, 'w', encoding='utf-8') as f:
        for base_module in tqdm.tqdm(list_module):
            f.write(f"Reference Documentation for {base_module.__name__} \n")
            f.write("="*60 + "\n\n")

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
                        
                        full_name = f"{prefix}.{name}_v2"

                        raw_doc = obj.__doc__
                        new_doc = raw_doc
                        
                        if new_doc:
                            new_doc = add_v2_signature_full_doc(new_doc)
                            
                            f.write(f"FUNCTION: {full_name}\n")
                            f.write("-" * (10 + len(full_name)) + "\n")
                            new_doc = supress_see_also(new_doc)
                            for shorthand in list_shorthand:
                                new_doc = corrupt_doc(new_doc, shorthand)
                            
                            f.write(new_doc + "\n")
                            f.write("\n" + "#"*40 + "\n\n")
                            
                    elif isinstance(obj, types.ModuleType):
                        if hasattr(obj, '__name__') and 'numpy' in obj.__name__:
                            stack.append((obj, f"{prefix}.{name}"))
            
    print(f"Documentation generated in {output_file}")
    print(f"Total unique functions documented: {len(seen_functions)}")


def generate_ultra_minimal_docs(list_module, output_file):
    """
    Generate a doc with only the function names and nothing else (with _v2 suffix)

    list_module : list of module names. the documentation of all the modules to corrupt. 
    output_file : the destination of the created file 
    """
    seen_functions = set()
    with open(output_file, 'w', encoding='utf-8') as f:
        for base_module in tqdm.tqdm(list_module):
            f.write(f"Reference Documentation for {base_module.__name__} \n")
            f.write("="*60 + "\n\n")

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
                        
                        full_name = f"{prefix}.{name}_v2"
                        raw_doc = obj.__doc__
                        new_doc = test_doc
                        
                        f.write(f"FUNCTION: {full_name}\n")
                        f.write("-" * (10 + len(full_name)) + "\n")
                                                                                                              
                    elif isinstance(obj, types.ModuleType):
                        if hasattr(obj, '__name__') and 'numpy' in obj.__name__:
                            stack.append((obj, f"{prefix}.{name}"))
            
    print(f"Documentation generated in {output_file}")
    print(f"Total unique functions documented: {len(seen_functions)}")


def generate_minimal_docs(list_module, output_file):
    """
    Generate a doc with the function names and the signature, when it exists (with _v2 suffix)

    list_module : list of module names. the documentation of all the modules to corrupt. 
    output_file : the destination of the created file 
    """
    seen_functions = set()
    with open(output_file, 'w', encoding='utf-8') as f:
        for base_module in tqdm.tqdm(list_module):
            f.write(f"Reference Documentation for {base_module.__name__} \n")
            f.write("="*60 + "\n\n")

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
                        
                        full_name = f"{prefix}.{name}_v2"
                        raw_doc = obj.__doc__
                        new_doc = raw_doc

                        if new_doc:
                            modif = add_v2_signature(new_doc)
                            if modif:
                                new_doc = modif
                            else:
                                new_doc = ""
                            f.write(f"FUNCTION: {full_name}\n")
                            f.write("-" * (10 + len(full_name)) + "\n")
                            f.write(new_doc + "\n")
                            f.write("#"*40 + "\n")                                      
                    elif isinstance(obj, types.ModuleType):
                        if hasattr(obj, '__name__') and 'numpy' in obj.__name__:
                            stack.append((obj, f"{prefix}.{name}"))
            
    print(f"Documentation generated in {output_file}")
    print(f"Total unique functions documented: {len(seen_functions)}")


def generate_real_ultra_minimal_docs(list_module, output_file):
    """
    Generate a doc with only the function names and nothing else (no suffix)

    list_module : list of module names. the documentation of all the modules to corrupt. 
    output_file : the destination of the created file 
    """
    seen_functions = set()
    with open(output_file, 'w', encoding='utf-8') as f:
        for base_module in tqdm.tqdm(list_module):
            f.write(f"Reference Documentation for {base_module.__name__} \n")
            f.write("="*60 + "\n\n")

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
                        
                        full_name = f"{prefix}.{name}"
                        raw_doc = obj.__doc__
                        new_doc = test_doc
                        
                        f.write(f"FUNCTION: {full_name}\n")
                        f.write("-" * (10 + len(full_name)) + "\n")
                                                                                                              
                    elif isinstance(obj, types.ModuleType):
                        if hasattr(obj, '__name__') and 'numpy' in obj.__name__:
                            stack.append((obj, f"{prefix}.{name}"))
            
    print(f"Documentation generated in {output_file}")
    print(f"Total unique functions documented: {len(seen_functions)}")


def generate_real_minimal_docs(list_module, output_file):
    """
    Generate a doc with the function names and the signature, when it exists (no suffix)

    list_module : list of module names. the documentation of all the modules to corrupt. 
    output_file : the destination of the created file 
    """
    seen_functions = set()
    with open(output_file, 'w', encoding='utf-8') as f:
        for base_module in tqdm.tqdm(list_module):
            f.write(f"Reference Documentation for {base_module.__name__} \n")
            f.write("="*60 + "\n\n")

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
                        
                        full_name = f"{prefix}.{name}"
                        signature_obj = inspect.signature(obj)
                        signature_str = str(signature_obj)
                        if signature_str:
                            f.write(f"FUNCTION: {full_name}\n")
                            f.write("-" * (10 + len(full_name)) + "\n")
                            f.write(name + signature_str + "\n")
                            f.write("#"*40 + "\n")                                      
                    elif isinstance(obj, types.ModuleType):
                        if hasattr(obj, '__name__') and 'numpy' in obj.__name__:
                            stack.append((obj, f"{prefix}.{name}"))
            
    print(f"Documentation generated in {output_file}")
    print(f"Total unique functions documented: {len(seen_functions)}")


def generate_real_doc(list_module, output_file):
    """
    Generates the full real doc without the see also section at output_file.

    list_module : list of module names. the documentation of all the modules to corrupt. 
    output_file : the destination of the created file 
    """
    seen_functions = set()
    with open(output_file, 'w', encoding='utf-8') as f:
        for base_module in tqdm.tqdm(list_module):
            f.write(f"Reference Documentation for {base_module.__name__} \n")
            f.write("="*60 + "\n\n")

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
                        
                        full_name = f"{prefix}.{name}"
                        raw_doc = obj.__doc__
                        new_doc = raw_doc
                        
                        if new_doc:
                            f.write(f"FUNCTION: {full_name}\n")
                            f.write("-" * (10 + len(full_name)) + "\n")
                            new_doc = supress_see_also(new_doc)
                            f.write(new_doc + "\n")
                            f.write("\n" + "#"*40 + "\n\n")
                            
                    elif isinstance(obj, types.ModuleType):
                        if hasattr(obj, '__name__') and 'numpy' in obj.__name__:
                            stack.append((obj, f"{prefix}.{name}"))
            
    print(f"Documentation generated in {output_file}")
    print(f"Total unique functions documented: {len(seen_functions)}")


if __name__ == "__main__":
    generate_full_docs([numpy], ["np", "numpy"], "corrupted_full_doc_numpy_v2.txt")
    generate_ultra_minimal_docs([numpy], output_file="corrupted_ultra_minimal_numpy_v2.txt")
    generate_minimal_docs([numpy], output_file="corrupted_minimal_numpy_v2.txt")
