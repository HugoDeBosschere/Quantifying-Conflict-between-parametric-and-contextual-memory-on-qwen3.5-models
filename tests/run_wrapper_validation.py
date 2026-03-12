"""
Lance la validation des trois wrappers (underscore, _v2 et capitalize).
À exécuter depuis la racine du projet : python tests/run_wrapper_validation.py
"""
import subprocess
import sys
import os

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


def run_script(path):
    """Exécute un script Python et retourne True si succès (code 0)."""
    result = subprocess.run(
        [sys.executable, path],
        cwd=os.path.dirname(TESTS_DIR),
        capture_output=False,
    )
    return result.returncode == 0


def main():
    print("=" * 60)
    print("Validation des wrappers NumPy")
    print("=" * 60)

    underscore_script = os.path.join(TESTS_DIR, "underscore", "test_wrapper_underscore.py")
    v2_script = os.path.join(TESTS_DIR, "v2", "test_wrapper_v2.py")
    capitalize_script = os.path.join(TESTS_DIR, "capitalize", "test_wrapper_capitalize.py")

    ok_underscore = run_script(underscore_script)
    print()
    ok_v2 = run_script(v2_script)
    print()
    ok_capitalize = run_script(capitalize_script)

    print()
    print("=" * 60)
    if ok_underscore and ok_v2 and ok_capitalize:
        print("Résultat : toutes les validations sont passées.")
        sys.exit(0)
    else:
        print("Résultat : au moins un test a échoué.")
        sys.exit(1)


if __name__ == "__main__":
    main()
