import json 
from itertools import islice
import re

def underscore_prompt_module(filename,list_shorthand):
    """
    dataset is a json containing a prompt argument that can be accessed 
    """
    n = len(filename)
    print(f"taille du json: {n}")
    new_filename = "ds1000_npy_modif.jsonl"
    with open(new_filename,"w",encoding="utf-8") as new_f:
        with open(filename,"r") as f:
            for line in f:
                new_line = line
                for sh in list_shorthand:
        #print(f"voici le string {docstring}")
                    new_line = re.sub(sh + r"\.(\w+)", sh + r".\1_", new_line)  
        #print(f"doc corrompue {new_docstring}")
                new_f.write(new_line)
    return None

if __name__ == '__main__':
    underscore_prompt_module("ds1000_npy.jsonl",["numpy","np"])