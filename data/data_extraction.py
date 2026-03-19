# function to extract only the question that use NUMPY as a library to test --> 220 Q/A


import json
import os


# chemin d'accès
input_dir = "ds1000.jsonl"
output_dir = "ds1000_npy.jsonl"


def select_tasks():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, input_dir)
    output_path = os.path.join(base_dir, output_dir)




    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'a', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip(): continue # Sauter lignes vides
            
            task = json.loads(line)

            if task["metadata"]["library"] == "Numpy" :
                json_record = json.dumps(task)
                f_out.write(json_record + "\n")
                f_out.flush()


if __name__ == "__main__" :
    select_tasks()