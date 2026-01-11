import requests
import re
import subprocess
import tempfile
import os
import sys
import textwrap
import json
from datetime import datetime

# --- CONFIGURATION ---
from config_loader import load_config

# On charge la config située à côté du script
CONF = load_config("config.json")

def query_llm(prompt_text):
    system = "You are a Python expert. Provide ONLY the raw code inside a markdown block. Do not explain."
    full_prompt = f"{system}\n\n{prompt_text}"
    
    try:
        response = requests.post(CONF["llm"]["api_url"], json={
            "model": CONF["llm"]["model_name"],
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": CONF["llm"]["temperature"]}
        })
        response.raise_for_status() # Lève une erreur si API pas 200
        return response.json()['response']
    except Exception as e:
        print(f"🚨 Erreur API LLM : {e}")
        return ""

def extract_code_and_fix(original_prompt, llm_response):
    # Regex robuste pour ```python ou ```
    pattern = r"```(?:python|Python)?\n(.*?)```"
    matches = re.findall(pattern, llm_response, re.DOTALL)
    
    if matches:
        code = max(matches, key=len).strip()
    else:
        code = llm_response.strip()

    # Si pas de signature, on la recolle
    # On regarde juste la dernière ligne du prompt (souvent 'def my_func():')
    if "def " not in code:
        return original_prompt.strip() + "\n    " + code
    
    return code

def run_test_safely(code_to_run, test_code):
    """Exécute le code et retourne le résultat (stdout, stderr, success)"""
    
    indented_test = textwrap.indent(test_code, '    ')
    
    final_script = (
        "import numpy\n"
        "import numpy as np\n"
        f"{code_to_run}\n"
        "\n# --- TESTS ---\n"
        "try:\n"
        f"{indented_test}\n"
        "    print('SUCCESS_MARKER')\n"
        "except Exception as e:\n"
        "    print(f'ERROR_MARKER: {e}')\n"
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

def evaluate_single_task(task):
    """Orchestre l'évaluation d'une seule tâche JSON"""
    print(f"🔹 Traitement de {task['task_id']}...")
    
    # 1. Génération
    raw_response = query_llm(task['prompt'])
    if not raw_response:
        return {"passed": False, "error": "LLM API Error"}

    # 2. Parsing
    final_code = extract_code_and_fix(task['prompt'], raw_response)
    
    # 3. Exécution
    stdout, stderr = run_test_safely(final_code, task['test'])
    
    passed = "SUCCESS_MARKER" in stdout
    
    # On retourne un objet riche pour le log
    return {
        "task_id": task["task_id"],
        "passed": passed,
        "generated_code": final_code,
        "stdout": stdout,
        "stderr": stderr,
        "llm_raw_response": raw_response
    }

def run_benchmark():
    # Résolution des chemins relatifs
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, CONF["data"]["input_path"])
    output_path = os.path.join(base_dir, CONF["data"]["output_path"])

    print(f"📂 Lecture de : {input_path}")
    print(f"💾 Écriture dans : {output_path}")

    # On ouvre le fichier de sortie en mode 'append' (a)
    # Comme ça, si le script crash, les résultats précédents sont sauvés.
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'a', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip(): continue # Sauter lignes vides
            
            task = json.loads(line)
            
            # Évaluation
            result = evaluate_single_task(task)
            
            # Affichage console minimaliste
            status = "✅" if result["passed"] else "❌"
            print(f"{status} {task['task_id']}")
            
            # Écriture immédiate dans le fichier jsonl
            json_record = json.dumps(result)
            f_out.write(json_record + "\n")
            f_out.flush() # Force l'écriture disque

if __name__ == "__main__":
    run_benchmark()