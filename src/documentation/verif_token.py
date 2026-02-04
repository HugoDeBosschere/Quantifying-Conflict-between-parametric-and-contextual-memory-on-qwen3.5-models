import requests
import sys
import os

# --- CONFIGURATION ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5-coder:32b"  # Mets ici le modèle que tu comptes utiliser
DOC_PATH = "/usr/users/sdim/sdim_25/memory_code_eval/src/documentation/full_doc_corrupted_numpy.txt" # Ton gros fichier de 30k lignes

def count_real_tokens(filepath, model):
    if not os.path.exists(filepath):
        print(f"❌ Erreur : Fichier introuvable : {filepath}")
        return

    print(f"📖 Lecture du fichier : {filepath} ...")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"📄 Taille du fichier : {len(content)} caractères ({len(content.splitlines())} lignes)")
    print(f"🤖 Envoi au Tokenizer de '{model}' via Ollama...")

    # On prépare une requête spéciale
    # On met un contexte GÉANT pour être sûr qu'Ollama ne tronque pas avant de compter
    # (Même si le modèle ne supporte pas 1M, Ollama essaiera de tokenizer)
    payload = {
        "model": model,
        "prompt": content,
        "stream": False,
        "options": {
            "num_ctx": 200000, # On demande une fenêtre large pour le comptage
            "num_predict": 0   # On demande 0 token de réponse (juste le processing)
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            
            # C'est LA valeur qui nous intéresse
            token_count = data.get('prompt_eval_count', 0)
            
            print("\n" + "="*40)
            print(f"RESULTAT POUR {model}")
            print("="*40)
            print(f"🔢 Nombre RÉEL de tokens : {token_count}")
            print("="*40)
            
            # Verdict
            limit = 128000 # Limite standard des modèles Llama 3.1 / Qwen 2.5
            if token_count < limit:
                print(f"✅ BONNE NOUVELLE : Ça rentre ! (Marge : {limit - token_count} tokens)")
            else:
                print(f"❌ MAUVAISE NOUVELLE : Ça dépasse de {token_count - limit} tokens.")
                print(f"   -> Tu devras couper environ {int((token_count - limit)/token_count * 100)}% du fichier.")
        
        else:
            print(f"⚠️ Erreur API Ollama ({response.status_code}) : {response.text}")

    except Exception as e:
        print(f"❌ Erreur de connexion : {e}")

if __name__ == "__main__":
    # Tu peux aussi passer le fichier en argument : python count_tokens.py mon_fichier.txt
    file_to_test = sys.argv[1] if len(sys.argv) > 1 else DOC_PATH
    count_real_tokens(file_to_test, MODEL_NAME)