import requests
import re
import subprocess
import tempfile
import os
import sys
import textwrap

## LOAD the config file
from config_loader import load_config
config = load_config()

# --- CONFIGURATION ---
MODEL_NAME = config["llm"]["model"]
API_URL = config["llm"]["api_url"]

dataset_item = {
    "task_id": "test/1",
    "prompt": "Using numpy, create a matrix with only zeros with 2*n rows and n cols\n def zero_matrix(n):\n",
    "test": "import numpy as _np\nassert (zero_matrix(2) == _np.zeros((4, 2))).all()",
    "entry_point": "zero_matrix"
}

def query_llm(prompt_text):
    print(f"Interrogation de {MODEL_NAME}...")
    
    system = "You are a Python expert. Provide ONLY the raw code inside a markdown block. Do not explain."
    full_prompt = f"{system}\n\n{prompt_text}"
    
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

def extract_code_and_fix(original_prompt, llm_response):
    print("\n--- Réponse Brut du LLM ---")
    print(llm_response)
    print("---------------------------\n")

    pattern = r"```(?:python|Python)?\n(.*?)```"
    matches = re.findall(pattern, llm_response, re.DOTALL)
    
    if matches:
        code = max(matches, key=len).strip()
    else:
        code = llm_response.strip()

    if "def zero_matrix" not in code:
        print("Signature manquante, concaténation avec le prompt...")
        return original_prompt.strip() + "\n    " + code
    
    return code

def run_test(code_to_run, test_code):
    indented_test = textwrap.indent(test_code, '    ')
    
    final_script = (
        "import numpy as np\n"
        f"{code_to_run}\n"
        "\n# --- TESTS ---\n"
        "try:\n"
        f"{indented_test}\n" # Injection du bloc correctement indenté
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
            timeout=10
        )
        return result
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)




if __name__ == "__main__":
    try:
        raw_response = query_llm(dataset_item['prompt'])
        final_code = extract_code_and_fix(dataset_item['prompt'], raw_response)
        
        print(f"--- Code Nettoyé ---\n{final_code}\n--------------------")

        res = run_test(final_code, dataset_item['test'])
        
        if "SUCCESS_MARKER" in res.stdout:
            print("\n SUCCÈS : Le test est passé !")
        else:
            print("\n ÉCHEC.")
            print("Sortie standard:", res.stdout)
            print("Erreurs:", res.stderr)

    except Exception as e:
        print(f"\nErreur critique du script : {e}")