import numpy
import inspect
import types
from numpydoc.docscrape import NumpyDocString
from inspect import signature
import re 
import scipy
import matplotlib.pyplot
import pandas
import tqdm


test_doc = numpy.acos.__doc__ 


def add_underscore(docstring,shorthand):
    """
    shorthand : the shorthand name of a module to know where to add the underscore
    docstring : docstring that is parsed 

    Returns : 
    new_s the string where each expression of the type shortand.function(args) has a added underscore : shorthand.function_(args)
    """
    #print(f"voici le string {docstring}")
    new_docstring = re.sub(shorthand + r"\.(\w+)", shorthand + r".\1_", docstring)  
    #print(f"doc corrompue {new_docstring}")
    return new_docstring


def add_underscore_signature(docstring):
    """
    adds an underscore to the signature of a function 
    """
    new_signature = re.sub(r"^(\w+)\(",r"\1_(", docstring)
    #print(f"Ceci est la nouvelle signature :{new_signature}")
    return new_signature

def add_underscore_see_also(docstring):
    """
    adds an underscore to what is supposed to be functions in the see also section
    another method would be entirely suppressing the see also section
    There are approximately two types of See Also sections : 
    1st : cos, arctan, arcsin, emath.arccos
    2nd :   arctan2 : The "four quadrant" arctan of the angle formed by (`x`, `y`)
    and the positive `x`-axis.
            angle : Argument of complex values.
    """
    pattern = re.search("See Also\n--------\n.*?(?=\n\n)",docstring,re.DOTALL)
    print(f"pattern found : {pattern}")
    new_see_also = re.sub(r"See Also\n--------\n(\w+),",r"\1_",pattern,docstring)
    print(f"New see also:{new_see_also}")
    return new_see_also


def corrupt_doc(doc_text,shorthand):
    if not doc_text: return ""
    #print("here")
    #print(doc_text)
    new_text = add_underscore(doc_text,shorthand)
    return new_text



# --- 2. The Crawler ---
def generate_full_docs(list_module,list_shorthand, output_file):
    """
    Maybe we should be able to add all the objects in the list_shorthand 
    """
    # Set to keep track of functions we've already documented (to avoid duplicates)
    seen_functions = set()
    with open(output_file, 'w', encoding='utf-8') as f:
        for base_module in tqdm.tqdm(list_module):
            f.write(f"Reference Documentation for {base_module.__name__} \n")
            f.write("="*60 + "\n\n")

            # We use a stack to crawl submodules (starting with numpy itself)
            # Format: (module_object, name_prefix)
            stack = [(base_module, base_module.__name__)]
            
            # Limit recursion depth to avoid crawling the entire python world
            visited_modules = set()

            while stack:
                current_mod, prefix = stack.pop()
                
                if current_mod in visited_modules:
                    continue
                visited_modules.add(current_mod)

                # Get everything inside this module
                for name in dir(current_mod):
                    # Skip private internal stuff
                    if name.startswith("_"): continue
                    
                    try:
                        obj = getattr(current_mod, name)
                    except:
                        continue

                    # A. IF IT IS A FUNCTION: Document it
                    if inspect.isfunction(obj) or isinstance(obj, numpy.ufunc):
                        # Check if we already documented this exact function object
                        # (NumPy exposes the same function in multiple places)
                        if obj in seen_functions:
                            continue
                        seen_functions.add(obj)
                        
                        # Create the full name (e.g., numpy.linalg.norm)
                        full_name = f"{prefix}.{name}_"
                        


                        # Get docstring
                        raw_doc = obj.__doc__
                        
                
                        new_doc = test_doc
                        
                        if new_doc:
                            f.write(f"FUNCTION: {full_name}\n")
                            f.write("-" * (10 + len(full_name)) + "\n")
                            for shorthand in list_shorthand:
                                # Write to file in a clean format
                                new_doc = corrupt_doc(new_doc,shorthand)
                                break
                            new_doc = add_underscore_signature(new_doc)
                            new_doc = add_underscore_see_also(new_doc)
                            f.write(new_doc + "\n")
                            f.write("\n" + "#"*40 + "\n\n")
                            
                    # B. IF IT IS A SUB-MODULE: Add to stack to crawl later
                    elif isinstance(obj, types.ModuleType):
                        # Only crawl submodules that belong to numpy (avoid crawling 'sys' or 'os')
                        if hasattr(obj, '__name__') and 'numpy' in obj.__name__:
                            stack.append((obj, f"{prefix}.{name}"))
            break
    print(f"Documentation generated in {output_file}")
    print(f"Total unique functions documented: {len(seen_functions)}")

# --- 3. Run it ---
if __name__ == "__main__":
    generate_full_docs([numpy,scipy,pandas,matplotlib.pyplot],["np","scipy","pd","numpy","pandas","maplotlib","plt"], "src/documentation/numpy_manual_underscore.txt")