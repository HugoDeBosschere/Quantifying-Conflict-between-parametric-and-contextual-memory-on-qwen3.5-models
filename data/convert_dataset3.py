import json
import os
import re

def convert_raw_to_ds1000_strict(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Erreur : Le fichier {input_file} n'existe pas.")
        return

    count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip():
                continue
            
            task = json.loads(line)
            
            # Extraction des métadonnées de la base originale
            task_id_str = task.get("task_id", "")
            try:
                problem_id = int(task_id_str.split("/")[-1])
            except ValueError:
                problem_id = 0
                
            prompt = task.get("prompt", "")
            test_block = task.get("test", "")
            
            # =================================================================
            # NOUVEAU : Suppression du dictionnaire METADATA
            # =================================================================
            # re.DOTALL permet de capturer les retours à la ligne à l'intérieur des accolades
            test_block = re.sub(r"METADATA\s*=\s*\{.*?\}", "", test_block, flags=re.DOTALL)
            
            entry_point = task.get("entry_point", "none")
            original_ref_code = task.get("canonical_solution", [""])[0]
            
            prompt_stripped = prompt.rstrip()
            
            # =================================================================
            # LOGIQUE DE FORMATAGE : Détection des assignations (ex: result =)
            # =================================================================
            match = re.search(r'([a-zA-Z0-9_]+)\s*=\s*$', prompt_stripped)
            
            if match:
                # --- CAS 1 : Le prompt se termine par une assignation ---
                var_name = match.group(1)
                
                # Nouveau prompt style DS1000
                new_prompt = prompt_stripped[:match.start()].rstrip()
                new_prompt += f"\n{var_name} = ... # put solution in this variable\nBEGIN SOLUTION\n<code>\n"
                
                # Le contexte d'exécution (le LLM génèrera lui-même le "var_name =")
                exec_context_content = f"[insert]\n{test_block}"
                
                # Le code de référence mis à jour
                new_ref_code = f"{var_name} = {original_ref_code.lstrip()}"
                
            else:
                # --- CAS 2 : Le prompt NE se termine PAS par une assignation ---
                # (ex: "def ma_fonction(x):")
                new_prompt = f"{prompt_stripped}\nBEGIN SOLUTION\n<code>\n"
                new_ref_code = original_ref_code
                
                # Gestion de l'indentation si le prompt finit par ":"
                if prompt_stripped.endswith(":"):
                    exec_context_content = f"{prompt_stripped}\n    [insert]\n{test_block}"
                else:
                    exec_context_content = f"{prompt_stripped}\n[insert]\n{test_block}"

            # =================================================================
            # CRÉATION DU WRAPPER D'EXÉCUTION (code_context)
            # =================================================================
            code_context = f"""exec_context = r\"\"\"
{exec_context_content}
\"\"\"

def test_execution(solution: str):
    import textwrap
    
    # Nettoyage des espaces résiduels de la solution
    solution = solution.strip()
    
    # Si le prompt attendait un bloc indenté (finit par ':')
    prefix = exec_context.split(\"[insert]\")[0]
    if prefix.strip().endswith(\":\"):
        # On indente tout le code de l'IA de 4 espaces
        solution = textwrap.indent(solution, \"    \")
        
    code = exec_context.replace(\"[insert]\", solution)
    
    namespace = {{}}
    try:
        exec(code, namespace)
    except Exception as e:
        # print(f\"Erreur d'exécution: {{e}}\") # Commenté pour éviter de polluer les logs
        return 0 # Echec
        
    # Vérification
    entry_point = "{entry_point}"
    if entry_point == "none" or entry_point == "":
        namespace["check"]()
    else:
        candidate_func = namespace.get(entry_point)
        if candidate_func is None:
            raise ValueError(f"Entry point '{{entry_point}}' introuvable après exécution.")
        namespace["check"](candidate_func)
    
    return 1 # Succès
"""

            # =================================================================
            # EXPORTATION AU FORMAT FINAL
            # =================================================================
            ds1000_task = {
                "prompt": new_prompt,
                "reference_code": new_ref_code,
                "metadata": {
                    "problem_id": problem_id,
                    "library_problem_id": problem_id,
                    "library": "Numpy",
                    "original_task_id": task_id_str,
                    "entry_point": entry_point
                },
                "code_context": code_context.strip()
            }
            
            f_out.write(json.dumps(ds1000_task) + "\n")
            count += 1

    print(f"✅ Succès ! {count} problèmes convertis directement en format strict DS1000.")
    print(f"📁 Sauvegardé sous : {output_file}")


if __name__ == "__main__":
    # --- A MODIFIER SELON VOS CHEMINS ---
    
    # 1. Traitement de la base originale NumpyEval
    # (Remplacez 'NumpyEval.jsonl' par le nom de votre fichier brut téléchargé)
    convert_raw_to_ds1000_strict(
        input_file="data/NumpyEval.jsonl", 
        output_file="data/test3NumpyEval_ds1000_format.jsonl"
    )
    
    # 2. Traitement de la base corrompue
    # (Si vous avez un fichier NumpyEval_corrupted.jsonl brut)
    # convert_raw_to_ds1000_strict(
    #     input_file="data/NumpyEval_corrupted_v2.jsonl", 
    #     output_file="data/test3NumpyEval_corrupted_v2_ds1000_format.jsonl"
    # )