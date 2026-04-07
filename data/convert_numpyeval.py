import json
import re

def convert_numpyeval_to_ds1000_strict(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            data = json.loads(line)
            prompt = data["prompt"]
            code_context = data["code_context"]
            ref_code = data["reference_code"]
            
            # 1. On cherche la variable d'assignation à la toute fin du prompt
            # Ex: "result =" ou "b =" ou "df ="
            match = re.search(r'([a-zA-Z0-9_]+)\s*=\s*$', prompt)
            
            if match:
                var_name = match.group(1)
                
                # --- Modification du prompt ---
                # On retire le "result =" final
                new_prompt = prompt[:match.start()].rstrip()
                # On ajoute la balise type DS1000
                new_prompt += f"\n{var_name} = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\n"
                data["prompt"] = new_prompt
                
                # --- Modification du code_context (exec_context) ---
                # On remplace "result = [insert]" par "[insert]"
                # (On utilise une regex pour gérer les éventuels espaces "result= [insert]")
                pattern_insert = rf'{var_name}\s*=\s*\[insert\]'
                data["code_context"] = re.sub(pattern_insert, '[insert]', code_context)
                
                # --- Modification du reference_code ---
                # On s'assure que le code de référence commence bien par l'assignation
                # ref_code est souvent " np.where(...)", on enlève l'espace et on ajoute "result = "
                data["reference_code"] = f"{var_name} = {ref_code.lstrip()}"
            
            # On écrit la ligne transformée dans le nouveau fichier
            fout.write(json.dumps(data) + "\n")
            
    print(f"Conversion terminée ! Fichier sauvegardé sous : {output_file}")

# --- EXÉCUTION ---
# N'oubliez pas d'appliquer cela sur vos deux fichiers (le normal et le corrompu) !

# 1. Pour la base de contrôle
# convert_numpyeval_to_ds1000_strict(
#     "NumpyEval_ds1000_format.jsonl", 
#     "testNumpyEval_ds1000_format.jsonl"
# )

# 2. Pour la base corrompue
convert_numpyeval_to_ds1000_strict(
    "NumpyEval_corrupted_v2_ds1000_format.jsonl", 
    "testNumpyEval_corrupted_v2_ds1000_format.jsonl"
)