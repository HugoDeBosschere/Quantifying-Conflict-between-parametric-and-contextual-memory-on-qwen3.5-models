import requests
import re
import subprocess
import tempfile
import os
import sys
import json
import textwrap

# --- CONFIGURATION ---
from config_loader import load_config
CONF = load_config("config.json")

# ==========================================
# 1. HELPER FUNCTIONS (LLM, PARSING, EXEC)
# ==========================================

def query_llm(prompt_text):
    """Interroge le LLM via Ollama"""
    system = (
        "You are a Python Data Science expert (Pandas/Numpy). "
        "Complete the code to solve the problem. "
        "Assign the final answer to the variable 'result'. "
        "Return ONLY the code inside a markdown block."
    )
    full_prompt = f"{system}\n\n{prompt_text}"
    
    try:
        response = requests.post(CONF["llm"]["api_url"], json={
            "model": CONF["llm"]["model_name"],
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0.2}
        })
        response.raise_for_status()
        return response.json()['response']
    except Exception as e:
        print(f"🚨 Erreur API LLM : {e}")
        return ""

def extract_code(llm_response):
    """Extrait proprement le code du Markdown"""
    pattern = r"```(?:\w*)\n(.*?)```"
    matches = re.findall(pattern, llm_response, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()
    return llm_response.strip()

def ensure_result_assignment(code):
    """Ajoute 'result =' si manquant (heuristique)"""
    if "result =" not in code and "result=" not in code:
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
    """Construit et exécute le script de test complet"""
    
    clean_solution = ensure_result_assignment(llm_solution)
    safe_solution_str = repr(clean_solution)

    final_script = (
        "import pandas as pd\n"
        "import numpy as np\n"
        "import copy\n"
        "import sys\n"
        "import math\n\n"
        f"{code_context}\n\n" # Injection du moteur de test
        "# --- EXÉCUTION ---\n"
        "try:\n"
        f"    test_execution({safe_solution_str})\n"
        "    print('SUCCESS_MARKER')\n"
        "except AssertionError:\n"
        "    print('TEST_FAILED: Assertion incorrecte')\n"
        "except Exception as e:\n"
        "    print(f'EXEC_ERROR: {e}')\n"
        "    import traceback\n"
        "    traceback.print_exc()\n"
    )

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(final_script)
        script_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=CONF["execution"]["timeout"]
        )
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT_ERROR"
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)

# ==========================================
# 2. CORE LOGIC : SINGLE TASK EVALUATION
# ==========================================

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
    code = extract_code(raw_response)

    # 4. Exécution (si contexte présent)
    if "code_context" in task:
        stdout, stderr = execute_task_engine(task["code_context"], code)
        passed = "SUCCESS_MARKER" in stdout
    else:
        print(f"⚠️ Warning: Pas de 'code_context' pour {task_id}")
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

# ==========================================
# 3. ORCHESTRATION : BENCHMARK LOOP
# ==========================================

def process_benchmark():
    """Lit le fichier d'entrée et traite chaque ligne"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, CONF["data"]["input_path"])
    output_path = os.path.join(base_dir, CONF["data"]["output_path"])

    print(f"📂 Lecture : {input_path}")
    
    if not os.path.exists(input_path):
        print(f"❌ Erreur: Fichier introuvable -> {input_path}")
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
                print("⚠️ Ligne JSON invalide ignorée")
                continue
            
            # --- APPEL DE LA FONCTION DÉDIÉE ---
            result = evaluate_single_task(task)
            
            # Feedback Console
            status = "✅" if result["passed"] else "❌"
            print(f"{status} {result['task_id']}")
            
            # Écriture Disque
            f_out.write(json.dumps(result) + "\n")
            f_out.flush() # Important pour sauvegarder en temps réel

if __name__ == "__main__":
    process_benchmark()