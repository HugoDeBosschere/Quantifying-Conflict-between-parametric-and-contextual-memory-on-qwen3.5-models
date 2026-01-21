import json
import matplotlib.pyplot as plt
import numpy as np



def perf_on_data(filepath):
    """Récupération de résultats depuis un fichier result.jsonl et compte des succès"""
    perf_dict = {}

    try:
        with open(filepath, mode="r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    res = json.loads(line)
                except json.JSONDecodeError:
                    continue

                model_name = res.get("model_metadata", {}).get("model_name", "Unknown_Model")
                
                task_id = res.get("task_id")
                is_passed = res.get("passed", False)

                if model_name not in perf_dict:
                    perf_dict[model_name] = {
                        "success": 0,
                        "success_control": 0,
                        "tasks_id": set(),
                        "tasks_id_control": set()
                    }

                if res.get("is_control", False):
                    if task_id not in perf_dict[model_name]["tasks_id_control"]:
                        if is_passed:
                            perf_dict[model_name]["success_control"] += 1
                        perf_dict[model_name]["tasks_id_control"].add(task_id)
                else:
                    if task_id not in perf_dict[model_name]["tasks_id"]:
                        if is_passed:
                            perf_dict[model_name]["success"] += 1
                        perf_dict[model_name]["tasks_id"].add(task_id)
                        
    except FileNotFoundError:
        print(f"Erreur : Le fichier {filepath} est introuvable.")
        return {}

    return perf_dict



def plot_model_performance(perf_dict, output_filename):
    if not perf_dict:
        print("Aucune donnée à afficher.")
        return

    models = list(perf_dict.keys())
    
    # Préparation des données pour le plot
    # On calcule les pourcentages et on stocke les totaux pour l'affichage
    main_scores = [] # % de réussite
    main_labels = [] # Texte "X/Y"
    
    ctrl_scores = []
    ctrl_labels = []
    
    has_control_data = False

    for m in models:
        data = perf_dict[m]
        
        # --- Données Principales ---
        total_main = len(data["tasks_id"])
        if total_main > 0:
            succ_main = data["success"]
            pct = (succ_main / total_main) * 100
            main_scores.append(pct)
            main_labels.append(f"{succ_main}/{total_main}")
        else:
            main_scores.append(0)
            main_labels.append("N/A")

        # --- Données Témoins (Control) ---
        total_ctrl = len(data["tasks_id_control"])
        if total_ctrl > 0:
            has_control_data = True
            succ_ctrl = data["success_control"]
            pct = (succ_ctrl / total_ctrl) * 100
            ctrl_scores.append(pct)
            ctrl_labels.append(f"{succ_ctrl}/{total_ctrl}")
        else:
            ctrl_scores.append(0)
            ctrl_labels.append("")

    # --- Création du Graphique ---
    x = np.arange(len(models))  # Position des labels
    width = 0.35  # Largeur des barres

    fig, ax = plt.subplots(figsize=(12, 6))

    # Barres principales
    rects1 = ax.bar(x - width/2, main_scores, width, label='Dataset Standard', color='skyblue', edgecolor='black')

    # Barres témoins (seulement si données présentes)
    if has_control_data:
        rects2 = ax.bar(x + width/2, ctrl_scores, width, label='Dataset Témoin (Control)', color='lightcoral', edgecolor='black')
    else:
        # Si pas de témoin, on centre les barres principales
        ax.clear()
        rects1 = ax.bar(x, main_scores, width, label='Dataset Standard', color='skyblue', edgecolor='black')

    # Customisation
    ax.set_ylabel('Précision (%)')
    ax.set_title('Performance des Modèles (Succès / Total)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0, 105) # Marge en haut pour le texte
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Fonction pour ajouter les labels (ex: "45/100") au dessus des barres
    def autolabel(rects, labels):
        for rect, label in zip(rects, labels):
            height = rect.get_height()
            if label and label != "N/A":
                ax.annotate(f'{height:.1f}%\n({label})',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points de décalage vertical
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1, main_labels)
    if has_control_data:
        autolabel(rects2, ctrl_labels)

    plt.tight_layout()
    plt.savefig(output_filename)


if __name__ == "__main__":
    # Remplace par ton vrai chemin de fichier
    path = "/usr/users/sdim/sdim_25/memory_code_eval/src/perf_review/results/result_try.jsonl"
    output_filename = "/usr/users/sdim/sdim_25/memory_code_eval/src/perf_review/results/test.png"
    
    data = perf_on_data(path)
    plot_model_performance(data, output_filename)