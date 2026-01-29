import json
import os

def filter_dataset_numpy_only(input_path, output_path):
    print(f"🔄 Traitement en cours de : {input_path}")
    
    kept_count = 0
    removed_ids = [] # Liste pour stocker les IDs supprimés

    # On ouvre les deux fichiers
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line: continue

            try:
                task = json.loads(line)
                
                # Récupération sécurisée de l'ID et du code
                task_id = task.get("metadata", {}).get("problem_id", "Unknown")
                ref_code = task.get("reference_code", "")

                # Condition de filtrage
                if "np." in ref_code or "numpy" in ref_code:
                    f_out.write(line + "\n")
                    kept_count += 1
                else:
                    # On stocke l'ID pour l'affichage final
                    removed_ids.append(task_id)

            except json.JSONDecodeError:
                print("⚠️ Erreur de décodage JSON sur une ligne.")
                continue

    # --- AFFICHAGE DU RAPPORT ---
    print("-" * 40)
    print(f"✅ Terminé !")
    print(f"💾 Tâches conservées : {kept_count}")
    print(f"🗑️ Tâches supprimées  : {len(removed_ids)}")
    print(f"📂 Nouveau fichier    : {output_path}")
    print("-" * 40)
    
    if removed_ids:
        print("📋 LISTE DES TÂCHES SUPPRIMÉES (À vérifier) :")
        # On trie les IDs pour que ce soit plus lisible (s'ils sont numériques)
        try:
            removed_ids.sort(key=lambda x: int(x))
        except ValueError:
            removed_ids.sort() # Tri alphabétique si ce n'est pas des nombres purs
            
        # Affichage propre sous forme de liste Python
        print(removed_ids)
        
        # Optionnel : Affichage en colonnes si la liste est longue
        # print("\n".join(str(x) for x in removed_ids))
    else:
        print("Aucune tâche n'a été supprimée.")
    print("-" * 40)


if __name__ == "__main__":
    # Tes chemins
    input_path = "/usr/users/sdim/sdim_25/memory_code_eval/data/ds1000_npy_modif_prompt.jsonl"
    
    # On crée un nom de fichier explicite pour la sortie
    output_path = "/usr/users/sdim/sdim_25/memory_code_eval/data/ds1000_npyOnly_corrupted_.jsonl"
    
    filter_dataset_numpy_only(input_path, output_path)




#*----------*
# RESULTATS #
#*----------*

# 📋 LISTE DES TÂCHES SUPPRIMÉES (À vérifier) :
#[291, 309, 310, 313, 329, 330, 340, 341, 342, 343, 349, 350, 351, 367, 378, 380, 385, 
# 387, 388, 389, 390, 391, 392, 393, 395, 397, 412, 413, 414, 415, 416, 417, 418, 419, 
# 423, 424, 428, 429, 430, 433, 434, 435, 436, 437, 439, 441, 445, 446, 447, 464, 465, 
# 466, 471, 472, 476, 480, 481, 484, 486, 487, 500]