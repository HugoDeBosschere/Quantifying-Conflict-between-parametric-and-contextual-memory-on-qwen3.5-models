import requests
import re
import subprocess
import tempfile
import os
import sys
import textwrap
import json
from datetime import datetime
from llmclient import LLMClient
from cleaning import extract_code_and_fix, ensure_result_assignment, modify_lib

## LOAD the config file
from config_loader import load_config
config = load_config()



# --- INIT LLM CLIENT ---
llm_client = LLMClient(config)
model_metadata = {
                  "model_name" :  llm_client.model_name,
                  "temperature" : llm_client.temperature,
                  }



def execute_task_engine(code_context, llm_solution):
    """
    Construit le script final en combinant le moteur de test (JSON) et la solution (LLM).
    """
    
    # 1. Nettoyage et préparation de la solution
    clean_solution = ensure_result_assignment(llm_solution)
    
    # 2. Échappement : On transforme le code du LLM en chaîne de caractères Python valide
    safe_solution_str = repr(clean_solution)

    # 3. Construction du script Python temporaire
    final_script = (
        "# --- 1. LE MOTEUR DE TEST (Issu du JSON) ---\n"
        f"{code_context}\n\n"
        
        "# --- 2. L'EXÉCUTION ---\n"
        "try:\n"
        f"    # On appelle la fonction fournie par le JSON pour tester la solution\n"
        f"    test_execution({safe_solution_str})\n"
        "    print('SUCCESS_MARKER')\n"
        "except AssertionError:\n"
        "    print('TEST_FAILED: Assertion incorrecte')\n"
        "except Exception as e:\n"
        "    print(f'EXEC_ERROR: {e}')\n"
        "    import traceback\n"
        "    traceback.print_exc()\n"
    )

    # 4. Écriture et Exécution
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(final_script)
        script_path = f.name

    # On ajoute au python path notre librairie maison contrefactuelle
    env_execution = os.environ.copy()
    current_pythonpath = env_execution.get("PYTHONPATH", "")
    env_execution["PYTHONPATH"] = llm_client.custom_lib_path + os.pathsep + current_pythonpath

    # # Uncomment to have a view on which file is executed
    # with open("/usr/users/sdim/sdim_25/memory_code_eval/example.py", mode="w") as f :
    #     f.write(final_script)

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=10,
            env=env_execution
        )
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT_ERROR"
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)



def evaluate_single_task(task, new_lib, context_prompt_type):
    """
    Orchestre l'évaluation d'une seule tâche.
    Input: Dictionnaire de la tâche (JSON)
    Output: Dictionnaire de résultat complet
    """
    
    # 1. Récupération ID (Gère le format simple ou metadata)
    task_id = task.get("metadata", {}).get("problem_id") or task.get("task_id", "unknown")
    print(f"Traitement ID {task_id}...")

    # 2. Appel LLM
    raw_response = llm_client.query_llm(task['prompt'], context_prompt_type)
    if not raw_response:
        return {
            "task_id": task_id,
            "passed": False,
            "error": "LLM_API_FAILURE",
            "llm_code": ""
        }

    # 3. Parsing
    code = extract_code_and_fix(raw_response)

    # 4. Exécution (si contexte présent)
    if "code_context" in task:
        ## Ajouter la modif en fonction de la lib plus tard
        new_import = "import " + new_lib + " as np"
        new_context = modify_lib(task["code_context"], new_import)
        if new_context :
            stdout, stderr = execute_task_engine(new_context, code)
        else:
            stdout, stderr = "", ""
        # stdout, stderr = execute_task_engine(task["code_context"], code)
        passed = "SUCCESS_MARKER" in stdout
    else:
        print(f"Warning: Pas de 'code_context' pour {task_id}")
        passed = False
        stdout, stderr = "", "MISSING_CONTEXT_IN_DATASET"

    # 5. Construction du résultat
    return {
        "task_id": task_id,
        "model_metadata" : model_metadata,
        "context_prompt_type":context_prompt_type,
        "passed": passed,
        "llm_code": code,
        "stdout": stdout,
        "stderr": stderr,
        "full_response": raw_response
    }

def run_benchmark():
    """Lit le fichier d'entrée et traite chaque ligne"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, config["data"]["input_path"])
    output_path = os.path.join(base_dir, config["data"]["output_path"])
    context_prompt_type_list = list(llm_client.context_prompt.keys())
    print(f"Lecture : {input_path}")
    
    if not os.path.exists(input_path):
        print(f"Erreur: Fichier introuvable -> {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'a', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip(): continue
            
            # Chargement JSON
            try:
                task = json.loads(line)
            except json.JSONDecodeError:
                print("Ligne JSON invalide ignorée")
                continue
            
            # run on single task
            for context_prompt_type in context_prompt_type_list:
                result = evaluate_single_task(task, config["new_lib_injection"]["name"], context_prompt_type)
            
            # Feedback Console
                status = "pass" if result["passed"] else "false"
                print(f"{status} {result['task_id']}")
                
                # Écriture Disque
                f_out.write(json.dumps(result) + "\n")
                f_out.flush()
    
def run_control() :
    """Lit le fichier d'entrée et traite chaque ligne"""
    print("CONTROL MODE")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, config["data"]["input_path"])
    output_path = os.path.join(base_dir, config["data"]["output_path"])
    context_prompt_type_list = list(llm_client.context_prompt.keys())
    print(f"Lecture : {input_path}")
    
    if not os.path.exists(input_path):
        print(f"Erreur: Fichier introuvable -> {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'a', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip(): continue
            
            # Chargement JSON
            try:
                task = json.loads(line)
            except json.JSONDecodeError:
                print("Ligne JSON invalide ignorée")
                continue
            
            usual_lib = task.get("metadata").get("library")
            # run on single task


            result = evaluate_single_task(task, usual_lib.lower(), "None")
            result["is_control"] = True

            # Feedback Console
            status = "pass" if result["passed"] else "false"
            print(f"{status} {result['task_id']}")
            
            # Écriture Disque
            f_out.write(json.dumps(result) + "\n")
            f_out.flush()

if __name__ == "__main__":
    # run_benchmark()
    run_control()


