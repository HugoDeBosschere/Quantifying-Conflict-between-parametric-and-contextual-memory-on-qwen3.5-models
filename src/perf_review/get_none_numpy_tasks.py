import json
import numpy as np

def is_numpy_task(task_data):
    ref_code = task_data.get("reference_code", "")
    if ref_code :
        return "np." in ref_code or "numpy" in ref_code
    else :
        return False

def generate_exclusion_file(input_path, output_file="non_numpy_tasks.json"):
    non_numpy_ids = []
    with open(input_path, mode="r") as f :

        for line in f:
            # Chargement JSON
            try:
                task = json.loads(line)
            except json.JSONDecodeError:
                print("Ligne JSON invalide ignorée")
                continue

            task_id = int(task.get("metadata", {}).get("problem_id", -1)) 
            
            if not is_numpy_task(task):
                non_numpy_ids.append(task_id)

        # Sauvegarde en JSON
        with open(output_file, "w") as f:
            json.dump(non_numpy_ids, f)
        

if __name__ == "__main__":
    input_path = "/usr/users/sdim/sdim_25/memory_code_eval/data/ds1000_npy.jsonl"
    generate_exclusion_file(input_path)