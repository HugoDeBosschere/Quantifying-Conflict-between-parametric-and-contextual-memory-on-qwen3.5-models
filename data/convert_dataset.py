import json
import os

def convert_numpyeval_to_ds1000(input_file, output_file):
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
            
            # Extraction de l'ID numérique
            task_id_str = task.get("task_id", "")
            try:
                problem_id = int(task_id_str.split("/")[-1])
            except ValueError:
                problem_id = 0
                
            prompt = task.get("prompt", "")
            test_block = task.get("test", "")
            entry_point = task.get("entry_point", "none")
            
            # 1. On nettoie la fin du prompt pour enlever les espaces inutiles
            prompt_stripped = prompt.rstrip()
            
            # 2. PLACEMENT INTELLIGENT DE [insert]
            if prompt_stripped.endswith("="):
                # Si assignation (ex: "result ="), on met sur la même ligne
                exec_context_content = f"{prompt_stripped} [insert]\n{test_block}"
            else:
                # Sinon (ex: "def fonction():"), on passe à la ligne
                exec_context_content = f"{prompt_stripped}\n[insert]\n{test_block}"
            
            # json.dumps sécurise la chaîne (gère les sauts de ligne, guillemets, etc.)
            exec_context_literal = json.dumps(exec_context_content)
            
            # 3. CODE CONTEXT AVEC GESTION AUTO DE L'INDENTATION
            code_context = f"""
exec_context = {exec_context_literal}

def test_execution(solution: str):
    import textwrap
    
    # Nettoyage des espaces résiduels de la solution
    solution = solution.strip()
    
    # Si le prompt attendait un bloc indenté (finit par ':')
    prefix = exec_context.split("[insert]")[0]
    if prefix.strip().endswith(":"):
        # On indente tout le code de l'IA de 4 espaces
        solution = textwrap.indent(solution, "    ")
        
    code = exec_context.replace("[insert]", solution)
    
    namespace = {{}}
    try:
        exec(code, namespace)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e
        
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

            # 4. Créer la structure compatible DS1000
            ds1000_task = {
                "prompt": prompt,
                "reference_code": task.get("canonical_solution", [""])[0],
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

    print(f"Succès ! {count} problèmes convertis et prêts pour la pipeline dans '{output_file}'.")

if __name__ == "__main__":
    # INPUT_JSONL = "NumpyEval.jsonl"
    # OUTPUT_JSONL = "NumpyEval_ds1000_format.jsonl"
    INPUT_JSONL = "NumpyEval_corrupted_v2.jsonl"
    OUTPUT_JSONL = "NumpyEval_corrupted_v2_ds1000_format.jsonl"
    convert_numpyeval_to_ds1000(INPUT_JSONL, OUTPUT_JSONL)