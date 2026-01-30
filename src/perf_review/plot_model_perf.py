import json
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_FILE = "/usr/users/sdim/sdim_25/memory_code_eval/src/perf_review/results/result_try_4models.jsonl"
OUTPUT_DIR = "src/perf_review/results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(filepath):
    """Charge les données et retourne un dictionnaire imbriqué."""
    data = {}
    print(f"Chargement des données depuis {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    res = json.loads(line)
                except json.JSONDecodeError:
                    continue

                meta = res.get("metadata", {})
                model = meta.get("model_name", "Unknown")
                pert = meta.get("perturbation_type", "Unknown")
                passed = 1 if res.get("passed", False) else 0
                
                if res.get("is_control", False):
                    doc_type = "CONTROL"
                else:
                    doc_type = meta.get("doc_name", "Unknown_Doc")

                if model not in data: data[model] = {}
                if doc_type not in data[model]: data[model][doc_type] = {}
                if pert not in data[model][doc_type]: 
                    data[model][doc_type][pert] = {'success': 0, 'total': 0}

                data[model][doc_type][pert]['success'] += passed
                data[model][doc_type][pert]['total'] += 1
                
    except FileNotFoundError:
        print(f"❌ Erreur : Fichier {filepath} introuvable.")
        return {}
    return data

def plot_1_global_performance(data):
    """
    Graph 1 : Performance Globale
    Label : "XX% \n (Succès/Total)"
    """
    print("Génération du Graphique 1 (Global)...")
    
    models = sorted(list(data.keys()))
    if not models: return

    all_docs = set()
    for m in models:
        all_docs.update(data[m].keys())
    
    doc_types = ["CONTROL"] + sorted([d for d in all_docs if d != "CONTROL"])
    
    x = np.arange(len(models))
    width = 0.85 / len(doc_types) # Un peu plus large pour faire tenir le texte

    fig, ax = plt.subplots(figsize=(14, 8)) # Figure plus grande

    for i, doc in enumerate(doc_types):
        scores = []
        labels = [] # Liste pour stocker nos labels personnalisés "XX% (N/T)"

        for m in models:
            if doc in data[m]:
                total_succ = sum(p['success'] for p in data[m][doc].values())
                total_cnt = sum(p['total'] for p in data[m][doc].values())
                
                if total_cnt > 0:
                    pct = (total_succ / total_cnt * 100)
                    # Création du label avec saut de ligne
                    label_text = f"{pct:.0f}%\n({total_succ}/{total_cnt})"
                else:
                    pct = 0
                    label_text = ""
            else:
                pct = 0
                label_text = ""
            
            scores.append(pct)
            labels.append(label_text)
        
        pos = x - 0.4 + width/2 + i*width
        
        # Style
        color = 'gray' if doc == "CONTROL" else None
        alpha = 0.6 if doc == "CONTROL" else 0.9
        hatch = '//' if doc == "CONTROL" else ''
        edgecolor = 'black'
        
        rects = ax.bar(pos, scores, width, label=doc, color=color, alpha=alpha, hatch=hatch, edgecolor=edgecolor)
        
        # --- C'EST ICI QUE ÇA SE JOUE ---
        # On passe notre liste 'labels' à la fonction bar_label
        ax.bar_label(rects, labels=labels, padding=3, fontsize=8, fontweight='bold')

    ax.set_title('Performance Globale : Baseline vs Documentation', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Taux de Succès (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight='bold', fontsize=11)
    ax.set_ylim(0, 115) # Marge en haut augmentée pour le texte sur 2 lignes
    ax.legend(title="Type de Documentation", loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    filename = os.path.join(OUTPUT_DIR, "plot_global_comparison.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✅ Sauvegardé : {filename}")

def plot_2_perturbation_breakdown(data):
    """
    Graph 2 : Détail par Perturbation
    Label : "XX% \n (N/T)"
    """
    print("Génération du Graphique 2 (Détail Perturbations)...")
    
    models = sorted(list(data.keys()))
    if not models: return

    # Perturbations et Docs
    all_perts = set()
    for m in data:
        for d in data[m]:
            all_perts.update(data[m][d].keys())
    pert_types = sorted(list(all_perts))
    
    all_docs = set()
    for m in models:
        all_docs.update(data[m].keys())
    doc_types = ["CONTROL"] + sorted([d for d in all_docs if d != "CONTROL"])

    n_models = len(models)
    # On augmente la hauteur (7 par modèle) pour que ce soit lisible
    fig, axes = plt.subplots(nrows=n_models, ncols=1, figsize=(14, 7 * n_models))
    if n_models == 1: axes = [axes]

    x = np.arange(len(pert_types))
    width = 0.85 / len(doc_types)

    for ax, model in zip(axes, models):
        ax.set_title(f"Modèle : {model}", fontsize=14, pad=15, color='#333333', fontweight='bold')
        
        for i, doc in enumerate(doc_types):
            scores = []
            labels = []

            for pert in pert_types:
                if doc in data[model] and pert in data[model][doc]:
                    stats = data[model][doc][pert]
                    succ = stats['success']
                    tot = stats['total']
                    
                    if tot > 0:
                        pct = (succ / tot * 100)
                        # Label compact pour le graph détaillé
                        labels.append(f"{pct:.0f}%\n({succ}/{tot})")
                    else:
                        pct = 0
                        labels.append("")
                else:
                    pct = 0
                    labels.append("")
                
                scores.append(pct)
            
            pos = x - 0.4 + width/2 + i*width
            
            color = 'gray' if doc == "CONTROL" else None
            alpha = 0.6 if doc == "CONTROL" else 0.9
            label = doc if doc == "CONTROL" else f"{doc}"
            
            rects = ax.bar(pos, scores, width, label=label, color=color, alpha=alpha)
            
            # Affichage conditionnel pour ne pas surcharger si 0%
            final_labels = [l if s > 0 else "" for l, s in zip(labels, scores)]
            
            ax.bar_label(rects, labels=final_labels, padding=2, fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(pert_types, fontweight='bold')
        ax.set_ylabel('Précision (%)')
        ax.set_ylim(0, 115)
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        ax.legend(loc='upper right', fontsize='small', framealpha=0.95)

    plt.suptitle("Détail par Perturbation (Succès / Total)", fontsize=16, y=0.995)
    plt.tight_layout()
    
    filename = os.path.join(OUTPUT_DIR, "plot_perturbation_breakdown.png")
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✅ Sauvegardé : {filename}")

if __name__ == "__main__":
    full_data = load_data(INPUT_FILE)
    if full_data:
        plot_1_global_performance(full_data)
        plot_2_perturbation_breakdown(full_data)