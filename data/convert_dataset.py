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
            
            task_id_str = task.get("task_id", "")
            try:
                problem_id = int(task_id_str.split("/")[-1])
            except ValueError:
                problem_id = 0
                
            prompt = task.get("prompt", "")
            test_block = task.get("test", "")
            entry_point = task.get("entry_point", "none")
            
            exec_context_content = f"{prompt}\n[insert]\n{test_block}"
            
            exec_context_literal = json.dumps(exec_context_content)
            
            code_context = f"""
exec_context = {exec_context_literal}

def test_execution(solution: str):
    # Remplacer la balise par le code généré
    code = exec_context.replace("[insert]", solution)
    
    # Exécuter le code (prompt + solution + test) dans un namespace isolé
    namespace = {{}}
    exec(code, namespace)
    
    # Récupérer dynamiquement le point d'entrée
    entry_point = "{entry_point}"
    
    # Déclencher le check selon la méthode attendue par NumpyEval
    if entry_point == "none" or entry_point == "":
        namespace["check"]()
    else:
        candidate_func = namespace.get(entry_point)
        if candidate_func is None:
            raise ValueError(f"Entry point '{{entry_point}}' introuvable après exécution.")
        namespace["check"](candidate_func)
    
    return 1 # Succès (convention similaire à DS1000)
"""

            # 3. Créer la structure compatible DS1000
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
    INPUT_JSONL = "NumpyEval.jsonl"
    OUTPUT_JSONL = "NumpyEval_ds1000_format.jsonl"
    
    convert_numpyeval_to_ds1000(INPUT_JSONL, OUTPUT_JSONL)