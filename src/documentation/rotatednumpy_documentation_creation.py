import numpy 
import inspect
import types
from numpydoc.docscrape import NumpyDocString
from inspect import signature

def no_kwargs(func_name):
    print("here")
    print(func_name)
    ord_dict = inspect.signature(func_name).parameters
    print(ord_dict == {})
    print(ord_dict)
    pos_args = []
    for k,v in ord_dict.items():
        if v.kind == inspect.Parameter.POSITIONAL_ONLY:
            pos_args.append(v)
    return pos_args



# --- 1. Your Rotation Logic (Reused) ---
def rotate_docstring(doc_text,func_name, isufunc = False):
    if not doc_text: 
        raise Exception("No docstring was given")
    try:
        #print("there")
        doc = NumpyDocString(doc_text)
        if isufunc:
            doc["Signature"] = ""
        pos_args = no_kwargs(func_name)
        n = len(pos_args)
        if n < 2:
            return ""
        if n > 1:
            print("here")
            # Rotate: [P1, P2, P3] -> [P2, P3, P1]
            params = doc['Parameters']
            rotated_params = params[1:n] + params[:1]
            doc['Parameters'] = rotated_params + params[n:]
            print(doc['Summary'][0])
            doc["Summary"] = []
            doc["Extended Summary"] = []
            print("after suprression")
            print(doc['Summary'])
            return str(doc)
    except:
        pass
    return doc_text

# --- 2. The Crawler ---
def generate_full_docs(base_module, output_file):
    # Set to keep track of functions we've already documented (to avoid duplicates)
    seen_functions = set()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Refrence Documentation for {base_module.__name__} \n")
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
                    full_name = f"{prefix}.{name}"
                    
                    # Get and rotate docstring
                    raw_doc = obj.__doc__
                    
                    print(full_name)

                    if raw_doc:
                        new_doc = rotate_docstring(raw_doc,obj,isinstance(obj,numpy.ufunc))
                        print(f"new doc :" + new_doc)
                        # Write to file in a clean format
                        f.write(f"FUNCTION: {full_name}\n")
                        f.write("-" * (10 + len(full_name)) + "\n")
                        f.write(new_doc + "\n")
                        f.write("\n" + "#"*40 + "\n\n")

                # B. IF IT IS A SUB-MODULE: Add to stack to crawl later
                elif isinstance(obj, types.ModuleType):
                    # Only crawl submodules that belong to numpy (avoid crawling 'sys' or 'os')
                    if hasattr(obj, '__name__') and 'numpy' in obj.__name__:
                        stack.append((obj, f"{prefix}.{name}"))

    print(f"Documentation generated in {output_file}")
    print(f"Total unique functions documented: {len(seen_functions)}")

# --- 3. Run it ---
if __name__ == "__main__":
    generate_full_docs(numpy, "numpy_manual.txt")
    