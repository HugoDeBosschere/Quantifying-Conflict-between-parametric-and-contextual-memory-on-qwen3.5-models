import re
import json

def modify_numpy(file_content, new_import_statement):
    # EXPLICATION DU PATTERN :
    # 1. (exec_context\s*=\s*r?""")  -> Groupe 1 : Capture le nom de la variable et l'ouverture des guillemets (r""" ou """)
    # 2. (.*?)                       -> Groupe 2 : Capture tout le texte AVANT l'import (non-gourmand)
    # 3. (import\s+numpy\s+as\s+np)  -> Groupe 3 : Cible spécifiquement l'import que tu veux changer
    # 4. (.*?""")                    -> Groupe 4 : Capture tout le reste jusqu'à la fermeture des guillemets
    
    pattern = r'(exec_context\s*=\s*r?""")(.*?)(import\s+numpy\s+as\s+np)(.*?""")'
    
    # On utilise re.DOTALL pour que le point (.) matche aussi les retours à la ligne (\n)
    match = re.search(pattern, file_content, flags=re.DOTALL)

    if match:
        print("Occurrence trouvée dans exec_context !")
        
        # On reconstruit la chaîne :
        # \1 (début variable) + \2 (code avant) + TON NOUVEAU CODE + \4 (fin variable)
        # Note: On utilise une fonction lambda ou une f-string logique pour le remplacement
        
        new_content = re.sub(
            pattern, 
            rf'\1\2{new_import_statement}\4', 
            file_content, 
            flags=re.DOTALL
        )
        return new_content
    else:
        print("Aucune occurrence trouvée dans exec_context.")
        return file_content



if __name__ == "__main__" :


    lib_to_try = "import WrapRotatedNumpy as np"
    input_path = "/usr/users/sdim/sdim_25/memory_code_eval/data/ds1000_npy.jsonl"
    output_path = "/usr/users/sdim/sdim_25/memory_code_eval/data/ds1000_Wnpy.jsonl"
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'a', encoding='utf-8') as f_out:
        for line in f_in :
            if not line.strip():
                continue
            task = json.loads(line)

            string_to_mod = task["code_context"]

            new_context = modify_numpy(string_to_mod, lib_to_try)

            # write the new code context
            task["code_context"] = new_context
            json_record = json.dumps(task)
            f_out.write(json_record + "\n")
            f_out.flush()
            
            

