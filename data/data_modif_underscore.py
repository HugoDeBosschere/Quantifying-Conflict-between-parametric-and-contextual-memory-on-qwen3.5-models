import json 
import re

import json
import re

def underscore_prompt_module(filename, list_shorthand):
    """
    Lit un fichier JSONL, modifie uniquement le champ 'prompt' pour ajouter
    des underscores aux méthodes des bibliothèques spécifiées, et sauvegarde le résultat.
    """
    print(f"Traitement du fichier : {filename}")
    
    new_filename = "ds1000_npy_modif_prompt.jsonl"
    
    with open(filename, "r", encoding="utf-8") as f_in, \
         open(new_filename, "w", encoding="utf-8") as f_out:
        
        count = 0
        for line in f_in:
            line = line.strip()
            if not line: continue # Sauter les lignes vides
            
            # 1. On transforme la ligne de texte en Dictionnaire Python
            data = json.loads(line)
            
            # 2. On vérifie si la clé 'prompt' existe et on la modifie
            if "prompt" in data:
                current_prompt = data["prompt"]
                
                for sh in list_shorthand:
                    # Regex expliquée :
                    # re.escape(sh) -> protège les caractères spéciaux (ex: si sh est "np")
                    # \.            -> cherche un vrai point
                    # (\w+)         -> capture le nom de la fonction (ex: "array")
                    pattern = re.escape(sh) + r"\.(\w+)"
                    
                    # Remplacement : on remet le préfixe, le point, le nom capturé (\1) et on ajoute "_"
                    current_prompt = re.sub(pattern, rf"{sh}.\1_", current_prompt)
                
                # On met à jour le dictionnaire
                data["prompt"] = current_prompt
            
            # 3. On retransforme le Dictionnaire en ligne JSON (string)
            # ensure_ascii=False permet de garder les accents lisibles si il y en a
            new_line = json.dumps(data, ensure_ascii=False)
            f_out.write(new_line + "\n")
            count += 1

    print(f"Terminé ! {count} lignes traitées. Sauvegardé dans {new_filename}")

if __name__ == '__main__':
    underscore_prompt_module("data/ds1000_npy.jsonl", ["numpy", "np"])