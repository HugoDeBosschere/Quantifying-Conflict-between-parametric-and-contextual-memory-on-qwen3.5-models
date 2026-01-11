import json
import os
import sys

def load_config(config_filename="config.json"):
    """
    Charge la configuration depuis un fichier JSON situé 
    DANS LE MÊME DOSSIER que ce script (config_loader.py).
    """
    # 1. Récupérer le dossier où se trouve ce script (config_loader.py)
    # __file__ est le chemin de ce script
    # abspath garantit un chemin absolu
    # dirname garde juste le dossier
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Construire le chemin complet vers le fichier config
    config_path = os.path.join(current_script_dir, config_filename)

    # Debug (facultatif, pour vérifier)
    # print(f"Recherche de la config ici : {config_path}")

    if not os.path.exists(config_path):
        print(f"ERREUR : Le fichier de config est introuvable ici :")
        print(f"-> {config_path}")
        sys.exit(1)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config
    except json.JSONDecodeError as e:
        print(f"🚨 ERREUR : Le JSON est mal formé (virgule manquante ? accolade ?).\n{e}")
        sys.exit(1)
    except Exception as e:
        print(f"🚨 Erreur inattendue : {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Test rapide
    cfg = load_config()
    print("Config chargée :", cfg)