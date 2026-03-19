import underscore_numpy_documentation

class Documentation_dictionnary:
    """
    Documentation dictionnary is a class that has a list_module and a list_corruption attribute and which stores 
    a dictionnary whose keys are a tuple (name_function, name_corruption) and the value is the documentation of this function corrupted by the right corruption. 
    If name_corruption == None then the generated documentation is the real documentation
    """
    def __init__(self, list list_module = [], list list_corruption = []):
        self.list_module = list_module
        self.list_corruption = list_corruption
        self.documentation_dictionnary = {}
    def __str__():
        print(f"Voici la liste des modules dont on veut générer la documentation : {self.list_module}")
        print(f"Voici la liste des corruptions que l'on veut appliquer : {self.list_corruption}")
    def generate_documentation():
        """
        Generates the full real doc without the see also section at output_file.

        list_module : list of module names. the documentation of all the modules to corrupt. 
        output_file : the destination of the created file

        For now I will code the method which generates only the real function 
        """
        # Set to keep track of functions we've already documented (to avoid duplicates)
        seen_functions = set()
        for base_module in tqdm.tqdm(self.list_module):
            for corruption in self.list_corruption:
               with open(output_file, 'w', encoding='utf-8') as f:

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
                               str()
                               if obj in seen_functions:
                                   continue
                               seen_functions.add(obj)

                               # Create the full name (e.g., numpy.linalg.norm)
                               full_name = f"{prefix}.{name}"



                               # Get docstring
                               raw_doc = obj.__doc__


                               new_doc = raw_doc

                               if new_doc:

                                   f.write(f"FUNCTION: {full_name}\n")
                                   f.write("-" * (10 + len(full_name)) + "\n")
                                   new_doc = supress_see_also(new_doc)
                                   f.write(new_doc + "\n")
                                   f.write("\n" + "#"*40 + "\n\n")

                           # B. IF IT IS A SUB-MODULE: Add to stack to crawl later
                           elif isinstance(obj, types.ModuleType):
                               # Only crawl submodules that belong to numpy (avoid crawling 'sys' or 'os')
                               if hasattr(obj, '__name__') and 'numpy' in obj.__name__:
                                   stack.append((obj, f"{prefix}.{name}"))

           print(f"Documentation generated in {output_file}")
           print(f"Total unique functions documented: {len(seen_functions)}")