import json
import os

# Ta liste noire d'IDs à supprimer
IDS_TO_EXCLUDE = {
    291, 309, 310, 313, 329, 330, 340, 341, 342, 343, 349, 350, 351, 367, 378, 380, 385, 
    387, 388, 389, 390, 391, 392, 393, 395, 397, 412, 413, 414, 415, 416, 417, 418, 419, 
    423, 424, 428, 429, 430, 433, 434, 435, 436, 437, 439, 441, 445, 446, 447, 464, 465, 
    466, 471, 472, 476, 480, 481, 484, 486, 487, 500
}

def clean_dataset_final_v2(input_path, output_path):
    print(f"🔄 Traitement en cours de : {input_path}")
    
    # Compteurs
    total_read = 0
    kept_count = 0
    removed_excluded_id = 0
    removed_duplicates = 0
    
    # Pour le dédoublonnage
    seen_signatures = set()

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line: continue
            total_read += 1

            try:
                task = json.loads(line)
                metadata = task.get("metadata", {})
                
                # --- 1. FILTRE PAR ID ---
                try:
                    problem_id = int(metadata.get("problem_id", -1))
                except (ValueError, TypeError):
                    problem_id = -1
                
                if problem_id in IDS_TO_EXCLUDE:
                    removed_excluded_id += 1
                    continue # On supprime
                
                # --- 2. DÉDOUBLONNAGE INTELLIGENT ---
                model_name = metadata.get("model_name", None)
                doc_name = metadata.get("doc_name", None)
                
                # Gestion du flag is_control
                raw_is_control = task.get("is_control")
                # Normalisation : None -> False
                is_control = False if raw_is_control is None else raw_is_control
                
                # CONSTRUCTION DE LA SIGNATURE
                if is_control:
                    # CAS CONTROL : On ignore la documentation.
                    # On force le champ doc à une valeur fixe pour que tous les controls
                    # du même modèle/problème soient vus comme des doublons.
                    signature_doc = "IGNORE_DOC_BECAUSE_CONTROL"
                else:
                    # CAS NORMAL : La documentation compte pour l'unicité.
                    signature_doc = doc_name

                # Signature finale
                signature = (model_name, signature_doc, problem_id, is_control)
                
                if signature in seen_signatures:
                    removed_duplicates += 1
                    # print(f"Doublon supprimé : ID {problem_id} (Control={is_control})")
                    continue 
                
                # Si tout est bon, on garde et on note la signature
                seen_signatures.add(signature)
                f_out.write(line + "\n")
                kept_count += 1

            except json.JSONDecodeError:
                print("⚠️ Erreur de décodage JSON sur une ligne.")
                continue

    # --- RAPPORT ---
    print("-" * 50)
    print(f"✅ Traitement Terminé !")
    print(f"📥 Total lu              : {total_read}")
    print(f"🚫 Supprimés (Liste ID)  : {removed_excluded_id}")
    print(f"👯 Supprimés (Doublons)  : {removed_duplicates}")
    print(f"💾 Final conservé        : {kept_count}")
    print(f"📂 Fichier de sortie     : {output_path}")
    print("-" * 50)

if __name__ == "__main__":
    input_file = "/usr/users/sdim/sdim_9/memory_code_eval/src/perf_review/results/result_try_4models.jsonl"
    output_file = "/usr/users/sdim/sdim_9/memory_code_eval/src/perf_review/results/result_4M_cleaned.jsonl"
    
    clean_dataset_final_v2(input_file, output_file)