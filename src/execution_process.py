import subprocess
import tempfile
import argparse
import os
import sys
import json
from datetime import datetime
from llmclient import LLMClient
from cleaning import extract_code_and_fix, modify_lib

## LOAD the config file
from config_loader import load_config
config = load_config()


#*--------------------------------------*
# Functional evaluation of the LLM code #
#*--------------------------------------*

def execute_task_engine(code_context, llm_solution, llm_client):
    """
    Construit le script final en combinant le moteur de test (JSON) et la solution (LLM).
    """
    
    # 1. Échappement : On transforme le code du LLM en chaîne de caractères Python valide
    safe_solution_str = repr(llm_solution)

    # 2. Construction du script Python temporaire
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


#*-----------------------------------------------------------------------------------------------------*
# Function that question the LLM, run the solution on control and corrupted mode and return the result #
#*-----------------------------------------------------------------------------------------------------*


def evaluate_single_task(task, new_lib, llm_client):
    """
    Orchestre l'évaluation d'une seule tâche.
    Input: Dictionnaire de la tâche (JSON)
    Output: Dictionnaire de résultat complet
    """
    
    # 1. Récupération ID (Gère le format simple ou metadata)
    task_id = task.get("metadata", {}).get("problem_id", "") or task.get("task_id", "")
    print(f"Traitement ID {task_id}...")

    # 2. Appel LLM
    raw_response, count_token = llm_client.query_llm(task['prompt'], new_lib)
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
            stdout, stderr = execute_task_engine(new_context, code, llm_client)
            stdout_control, stderr_control = execute_task_engine(task["code_context"], code, llm_client)
        else:
            stdout, stderr = "", ""
        # stdout, stderr = execute_task_engine(task["code_context"], code)
        passed = "SUCCESS_MARKER" in stdout
        control_passed = "SUCCESS_MARKER" in stdout_control
    else:
        passed = False
        stdout, stderr = "", "MISSING_CONTEXT_IN_DATASET"
        stdout_control, stderr_control = "", "MISSING_CONTEXT_IN_DATASET"

    ## récupération des Metadonnées
    metadata = task["metadata"] | llm_client.model_metadata | {"token_count": count_token}
    

    # 5. Construction du résultat
    return {
        "task_id": task_id,
        "metadata" : metadata,
        "passed": passed,
        "control_passed": control_passed,
        "llm_code": code,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_control": stdout_control,
        "stderr_control": stderr_control,
        "full_response": raw_response
    }


#*--------------------------------------------------------------------------------------------*
# Function that question the LLM, run the solution on control mode with no counterfactual lib #
#*--------------------------------------------------------------------------------------------*


def evaluate_single_task_control(task, old_lib, llm_client):
    """
    Orchestre l'évaluation d'une seule tâche.
    Input: Dictionnaire de la tâche (JSON)
    Output: Dictionnaire de résultat complet
    """
    
    # 1. Récupération ID (Gère le format simple ou metadata)
    task_id = task.get("metadata", {}).get("problem_id", "") or task.get("task_id", "")
    print(f"Traitement ID {task_id}...")

    # 2. Appel LLM
    raw_response, count_token = llm_client.query_llm(task['prompt'], old_lib)
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
        stdout, stderr = execute_task_engine(task["code_context"], code, llm_client)
        passed = "SUCCESS_MARKER" in stdout
    else:
        passed = False
        stdout, stderr = "", "MISSING_CONTEXT_IN_DATASET"

    ## récupération des Metadonnées
    metadata = task["metadata"] | llm_client.model_metadata | {"token_count": count_token}
    

    # 5. Construction du résultat
    return {
        "task_id": task_id,
        "metadata" : metadata,
        "passed": passed,
        "llm_code": code,
        "stdout": stdout,
        "stderr": stderr,
        "full_response": raw_response,
        "is_control": True
    }



#*-----------------------------------------------------------------------------------------------*
# Function to run tests on the dataset mentioned in the config file with the COUNTERFACTUAL LIB #
#*-----------------------------------------------------------------------------------------------*


def run_benchmark(first_task, llm_client):
    """Lit le fichier d'entrée et traite chaque ligne"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, config["data"]["corrupted_data"])
    output_path = os.path.join(base_dir, config["data"]["output_path"])
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
            task_id = task.get("metadata", {}).get("problem_id", "") or task.get("task_id", "")
            if task_id and task_id > first_task :
                # run on single task
                result = evaluate_single_task(task, config["new_lib_injection"]["name"], llm_client)
                
                # Feedback Console
                status = "pass" if result["passed"] else "false"
                print(f"{status} {result['task_id']}")
                
                # Écriture Disque
                f_out.write(json.dumps(result) + "\n")
                f_out.flush()
    

#*--------------------------------------------------------------------------*
# Function to run tests on the dataset with the RIGHT LIB as a control test #
#*--------------------------------------------------------------------------*


def run_control(first_task, llm_client) :
    """Lit le fichier d'entrée et traite chaque ligne"""
    print("CONTROL MODE")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, config["data"]["origin_data"])
    output_path = os.path.join(base_dir, config["data"]["output_path"])
    print(f"Lecture : {input_path}")

    if not os.path.exists(input_path):
        print(f"Erreur: Fichier introuvable -> {input_path}")
        return

    # Count total lines for progress bar
    with open(input_path, 'r', encoding='utf-8') as f:
        total_tasks = sum(1 for line in f if line.strip())


    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'a', encoding='utf-8') as f_out:

        # Main progress bar for all evaluations

        for line in f_in:
            if not line.strip(): continue

            # Chargement JSON
            try:
                task = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            usual_lib = task.get("metadata", {}).get("library", "None")

            task_id = task.get("metadata", {}).get("problem_id", "") or task.get("task_id", "")
            if task_id and task_id > first_task :
                # run on single task
                result = evaluate_single_task_control(task, usual_lib.lower(), llm_client)

                # Feedback Console
                status = "pass" if result["passed"] else "false"
                print(f"{status} {result['task_id']}")
                
                # Écriture Disque
                f_out.write(json.dumps(result) + "\n")
                f_out.flush()




#*-----*
# MAIN #
#*-----*


if __name__ == "__main__":
    # Get task id to start with
    parser = argparse.ArgumentParser(description="Lancer l'évaluation DS-1000")
    parser.add_argument("-t", "--task_id", type=int, default=0, help="ID spécifique de la tâche à partir de laquelle relancer l'execution")
    args = parser.parse_args()

    # Liste noms de modèles
    list_model_name = config.get("llm", {}).get("model", [])

    # liste différentes docu
    docu = config.get("new_lib_injection", {}).get("documentation", {})
    list_doc_name = docu.keys()

    # liste différentes docu CONTROL
    docu_control = config.get("real_lib", {}).get("documentation", {})
    list_doc_name_control = docu_control.keys()

    # boucle sur tout nos modèles
    for model_name in list_model_name :
        #TODO  à voir ce que je dois rajouter ici, et voir comment je pourrais pas mieux gérer ces histoires de llmclient
        # boucle sur nos différentes documentations CONTROL
        for doc_name in list_doc_name_control :

            # --- INIT LLM CLIENT CONTROL---
            llm_client = LLMClient(config, model_name, doc_name)

            run_control(args.task_id, llm_client)

        # boucle sur nos différentes documentations
        for doc_name in list_doc_name :

            # --- INIT LLM CLIENT ---
            llm_client = LLMClient(config, model_name, doc_name)

            run_benchmark(args.task_id, llm_client)
            


###TODO penser à clean les config et tout ce qui usage des chemins d'accès en créant un folder_path et des documentation name pour un peu plus de flexibilité