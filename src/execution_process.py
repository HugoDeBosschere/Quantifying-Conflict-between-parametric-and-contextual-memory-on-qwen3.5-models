import requests
import re
import subprocess
import tempfile
import os
import sys
import textwrap
import json
from datetime import datetime

## LOAD the config file
from config_loader import load_config
config = load_config()

# --- CONFIGURATION ---
MODEL_NAME = config["llm"]["model"]
API_URL = config["llm"]["api_url"]


def query_llm(prompt_text):
    print(f"Interrogation de {MODEL_NAME}...")
    
    system = config["new_lib_injection"]["system_prompt"]
    context_prompt = config["new_lib_injection"]["context_prompt"]
    full_prompt = f"{system}\n\n{context_prompt}\n\n{prompt_text}"
    
    response = requests.post(API_URL, json={
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": config["llm"]["temperature"],
        }
    })
    
    if response.status_code != 200:
        raise Exception(f"API Error: {response.text}")
        
    return response.json()['response']

def extract_code_and_fix(llm_response):
    print("\n--- Réponse Brut du LLM ---")
    print(llm_response)
    print("---------------------------\n")

    # find the code between the markdown
    pattern = r"```(?:python|Python|code)?\n(.*?)```"
    matches = re.findall(pattern, llm_response, re.DOTALL)
    
    if matches:
        code = max(matches, key=len).strip()
    else:
        code = llm_response.strip()

    # take the imports out of the code 
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
            
        cleaned_lines.append(line)
        
    code = "\n".join(cleaned_lines)

    return code


def ensure_result_assignment(code):
    """
    Heuristique : Le moteur de test s'attend souvent à trouver une variable 'result'.
    Si le LLM renvoie juste 'np.percentile(...)', on rajoute 'result = ' devant.
    """
    if "result =" not in code and "result=" not in code:
        # On prend la dernière ligne non vide
        lines = code.split('\n')
        last_line_idx = -1
        for i in range(len(lines) -1, -1, -1):
            if lines[i].strip():
                last_line_idx = i
                break
        
        if last_line_idx != -1:
            lines[last_line_idx] = "result = " + lines[last_line_idx]
            return "\n".join(lines)
            
    return code



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
        "import numpy as np\n"
        "import copy\n"
        "import sys\n"
        "import math\n\n"
        
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


    # Uncomment to have a view on which file is executed
    # with open("/usr/users/sdim/sdim_25/memory_code_eval/example.py", mode="w") as f :
    #     f.write(final_script)

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=10 
        )
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT_ERROR"
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)



def evaluate_single_task(task):
    """
    Orchestre l'évaluation d'une seule tâche.
    Input: Dictionnaire de la tâche (JSON)
    Output: Dictionnaire de résultat complet
    """
    
    # 1. Récupération ID (Gère le format simple ou metadata)
    task_id = task.get("metadata", {}).get("problem_id") or task.get("task_id", "unknown")
    print(f"🔹 Traitement ID {task_id}...")

    # 2. Appel LLM
    raw_response = query_llm(task['prompt'])
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
        stdout, stderr = execute_task_engine(task["code_context"], code)
        passed = "SUCCESS_MARKER" in stdout
    else:
        print(f"Warning: Pas de 'code_context' pour {task_id}")
        passed = False
        stdout, stderr = "", "MISSING_CONTEXT_IN_DATASET"

    # 5. Construction du résultat
    return {
        "task_id": task_id,
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

    print(f"Lecture : {input_path}")
    
    if not os.path.exists(input_path):
        print(f"Erreur: Fichier introuvable -> {input_path}")
        return

    # Ouverture en mode append ('a') pour reprendre si crash
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
            
            # --- APPEL DE LA FONCTION DÉDIÉE ---
            result = evaluate_single_task(task)
            
            # Feedback Console
            status = "pass" if result["passed"] else "false"
            print(f"{status} {result['task_id']}")
            
            # Écriture Disque
            f_out.write(json.dumps(result) + "\n")
            f_out.flush() # Important pour sauvegarder en temps réel

if __name__ == "__main__":
    run_benchmark()


