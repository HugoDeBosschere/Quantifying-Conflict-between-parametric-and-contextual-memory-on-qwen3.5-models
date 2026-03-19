import subprocess
import tempfile
import argparse
import os
import sys
import json
from datetime import datetime
from llmclient import LLMClient
from cleaning import extract_code_and_fix, modify_lib


def load_config_from_path(config_path):
    """Charge la configuration depuis un fichier JSON (chemin absolu ou relatif)."""
    path = os.path.abspath(config_path)
    if not os.path.exists(path):
        print(f"ERREUR : Fichier de config introuvable : {path}")
        sys.exit(1)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERREUR : JSON invalide dans {path}\n{e}")
        sys.exit(1)


#*-----------------------*
# Run directory management #
#*-----------------------*


def setup_run_directory(config):
    """
    Crée un dossier horodaté dans results/ à la racine du projet.
    Sauvegarde un snapshot de la config utilisée.
    Retourne le chemin du fichier results.jsonl à écrire.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dir = os.path.join(project_root, "results", f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    config_snapshot_path = os.path.join(run_dir, "config.json")
    with open(config_snapshot_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    output_path = os.path.join(run_dir, "results.jsonl")
    print(f"[RUN] Dossier de résultats : {run_dir}")
    return output_path


#*--------------------------------------*
# Functional evaluation of the LLM code #
#*--------------------------------------*

def execute_task_engine(code_context, llm_solution, llm_client, config):
    """
    Construit le script final en combinant le moteur de test (JSON) et la solution (LLM).
    Le timeout d'exécution (secondes) est lu depuis config["exec"]["timeout"], défaut 60.
    """
    safe_solution_str = repr(llm_solution)

    final_script = (
        "# --- 1. LE MOTEUR DE TEST (Issu du JSON) ---\n"
        f"{code_context}\n\n"
        "# --- 2. L'EXÉCUTION ---\n"
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

    env_execution = os.environ.copy()
    if llm_client.custom_lib_path:
        current_pythonpath = env_execution.get("PYTHONPATH", "")
        env_execution["PYTHONPATH"] = llm_client.custom_lib_path + os.pathsep + current_pythonpath

    timeout_sec = config.get("exec", {}).get("timeout", 120)
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
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


def evaluate_single_task(task, llm_client, config):
    """
    Orchestre l'évaluation d'une seule tâche en mode injection.
    Exécute le code à la fois avec la lib contrefactuelle ET avec la vraie lib (double test).
    """
    task_id = task.get("metadata", {}).get("problem_id", "") or task.get("task_id", "")
    print(f"Traitement ID {task_id}...")

    # chargement du module de nettoyage via AST pour les modules de fonction:
    ast_cleaning_module = config.get("new_lib_injection", {}).get("ast_cleaning_module", None)
    if ast_cleaning_module:
        import importlib
        ast_cleaning_module = importlib.import_module(ast_cleaning_module)
        normalize_object_attributes = getattr(ast_cleaning_module, "normalize_object_attributes")
    else:
        normalize_object_attributes = None
    assert normalize_object_attributes is not None, "ast_cleaning_module is not set"


    raw_response, count_token = llm_client.query_llm(task['prompt'])
    metadata = task["metadata"] | llm_client.model_metadata | {"token_count": count_token or 0}

    if not raw_response:
        return {
            "task_id": task_id,
            "metadata": metadata,
            "passed": False,
            "control_passed": False,
            "error": "LLM_API_FAILURE",
            "llm_code": ""
        }

    #cleaning de forme
    extracted_code = extract_code_and_fix(raw_response)

    #cleaning propre à la perturbation
    try :
        code = normalize_object_attributes(extracted_code)
    except ast_cleaning_module.ObjectAttributeError as e:
        passed = False
        print(f"Une méthode en comportait pas la perturbation, on retourne alors directement une erreur de type {e}")
        code = extracted_code
        stdout, stderr = "", "MODULE_WITH_SUFFIX_ERROR"

        if "code_context" in task:
            stdout_control, stderr_control = execute_task_engine(task["code_context"], code, llm_client, config)
            control_passed = "SUCCESS_MARKER" in stdout_control
        else:
            passed = False
            control_passed = False
            stdout, stderr = "", "MISSING_CONTEXT_IN_DATASET"
            stdout_control, stderr_control = "", "MISSING_CONTEXT_IN_DATASET"
        

        return {
            "task_id": task_id,
            "metadata": metadata,
            "passed": passed,
            "control_passed": control_passed,
            "llm_code": code,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_control": stdout_control,
            "stderr_control": stderr_control,
            "full_response": raw_response
        }

    # on procède aux évaluations fonctionnelles maintenant qu ele code a été validé par l'AST
    if "code_context" in task:
        new_import = "import " + llm_client.lib_name + " as np"
        new_context = modify_lib(task["code_context"], new_import)
        if new_context:
            # évaluation du LLm avec la lib contrefactuelle
            stdout, stderr = execute_task_engine(new_context, code, llm_client, config)
            # évaluation du LLm avec la lib d'origine sachant que le problème et la lib qu'on lui a montré était contrefactuelle
            stdout_control, stderr_control = execute_task_engine(task["code_context"], code, llm_client, config)
        else:
            stdout, stderr = "", "MODIFY_LIB_FAILED"
            stdout_control, stderr_control = "", "MODIFY_LIB_FAILED"
        passed = "SUCCESS_MARKER" in stdout
        control_passed = "SUCCESS_MARKER" in stdout_control
    else:
        passed = False
        control_passed = False
        stdout, stderr = "", "MISSING_CONTEXT_IN_DATASET"
        stdout_control, stderr_control = "", "MISSING_CONTEXT_IN_DATASET"


    return {
        "task_id": task_id,
        "metadata": metadata,
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


def evaluate_single_task_control(task, llm_client, config):
    """
    Orchestre l'évaluation d'une seule tâche en mode control (vraie lib, pas de contrefactuel).
    """
    task_id = task.get("metadata", {}).get("problem_id", "") or task.get("task_id", "")
    print(f"Traitement ID {task_id}...")

    raw_response, count_token = llm_client.query_llm(task['prompt'])
    if not raw_response:
        metadata = task["metadata"] | llm_client.model_metadata | {"token_count": count_token or 0}
        return {
            "task_id": task_id,
            "metadata": metadata,
            "passed": False,
            "error": "LLM_API_FAILURE",
            "llm_code": "",
            "is_control": True
        }

    code = extract_code_and_fix(raw_response)

    if "code_context" in task:
        stdout, stderr = execute_task_engine(task["code_context"], code, llm_client, config)
        passed = "SUCCESS_MARKER" in stdout
    else:
        passed = False
        stdout, stderr = "", "MISSING_CONTEXT_IN_DATASET"

    metadata = task["metadata"] | llm_client.model_metadata | {"token_count": count_token}

    return {
        "task_id": task_id,
        "metadata": metadata,
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


def run_benchmark(first_task, llm_client, output_path, config):
    """Lit le fichier d'entrée corrupted et traite chaque ligne (mode injection)."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, config["data"]["corrupted_data"])
    print(f"Lecture : {input_path}")

    if not os.path.exists(input_path):
        print(f"Erreur: Fichier introuvable -> {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'a', encoding='utf-8') as f_out:

        for line in f_in:
            if not line.strip(): continue

            try:
                task = json.loads(line)
            except json.JSONDecodeError:
                print("Ligne JSON invalide ignorée")
                continue

            task_id = task.get("metadata", {}).get("problem_id", "") or task.get("task_id", "")
            if task_id and task_id > first_task:
                result = evaluate_single_task(task, llm_client, config)

                status = "pass" if result["passed"] else "false"
                print(f"{status} {result['task_id']}")

                f_out.write(json.dumps(result) + "\n")
                f_out.flush()


#*--------------------------------------------------------------------------*
# Function to run tests on the dataset with the RIGHT LIB as a control test #
#*--------------------------------------------------------------------------*


def run_control(first_task, llm_client, output_path, config):
    """Lit le fichier d'entrée origin et traite chaque ligne (mode control)."""
    print("CONTROL MODE")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, config["data"]["origin_data"])
    print(f"Lecture : {input_path}")

    if not os.path.exists(input_path):
        print(f"Erreur: Fichier introuvable -> {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'a', encoding='utf-8') as f_out:

        for line in f_in:
            if not line.strip(): continue

            try:
                task = json.loads(line)
            except json.JSONDecodeError:
                continue

            task_id = task.get("metadata", {}).get("problem_id", "") or task.get("task_id", "")
            if task_id and task_id > first_task:
                result = evaluate_single_task_control(task, llm_client, config)

                status = "pass" if result["passed"] else "false"
                print(f"{status} {result['task_id']}")

                f_out.write(json.dumps(result) + "\n")
                f_out.flush()


#*-----*
# MAIN #
#*-----*


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lancer l'évaluation DS-1000")
    parser.add_argument("config", help="Chemin vers le fichier de config JSON")
    parser.add_argument("-t", "--task_id", type=int, default=0,
                        help="ID spécifique de la tâche à partir de laquelle relancer l'execution")
    parser.add_argument("--model", help="Ne lancer que ce modèle précis (sinon tous)")
    parser.add_argument("--doc", help="Ne lancer que cette documentation (clé dans config)")
    parser.add_argument(
        "--control_only",
        action="store_true",
        help="Ne lancer que le mode control (vraie librairie)",
    )
    parser.add_argument(
        "--injection_only",
        action="store_true",
        help="Ne lancer que le mode injection (lib contrefactuelle)",
    )
    args = parser.parse_args()

    config = load_config_from_path(args.config)

    output_path = setup_run_directory(config)

    list_model_name = config.get("llm", {}).get("model", [])

    docu = config.get("new_lib_injection", {}).get("documentation", {})
    list_doc_name = list(docu.keys())

    docu_control = config.get("real_lib", {}).get("documentation", {})
    list_doc_name_control = list(docu_control.keys())

    # Flags de mode : par défaut on lance control + injection.
    run_control_mode = not args.injection_only
    run_injection_mode = not args.control_only

    for model_name in list_model_name:
        # Filtre optionnel sur le modèle
        if args.model and model_name != args.model:
            continue

        # Run control mode with each real-lib doc
        if run_control_mode:
            for doc_name in list_doc_name_control:
                if args.doc and doc_name != args.doc:
                    continue
                llm_client_control = LLMClient(
                    config, model_name, doc_name, mode="control"
                )
                llm_client_control.warm_up()
                run_control(args.task_id, llm_client_control, output_path, config)

        # Run injection mode with each counterfactual-lib doc
        if run_injection_mode:
            for doc_name in list_doc_name:
                if args.doc and doc_name != args.doc:
                    continue
                llm_client = LLMClient(
                    config, model_name, doc_name, mode="injection"
                )
                llm_client.warm_up()
                run_benchmark(args.task_id, llm_client, output_path, config)
