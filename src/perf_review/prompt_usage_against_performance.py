#!/usr/bin/env python3
import json
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def plot_success_vs_np_frequency(
    config_path: str, 
    results_path: str, 
    output_path: str, 
    prompt_key: str = "prompt"
):
    """
    Joint le fichier de configuration et les résultats sur `problem_id`.
    Trace le taux de succès en fonction de la fréquence de la chaîne "np." dans le prompt.
    Inclut une régression linéaire pondérée (WLS) basée sur la taille de l'échantillon.
    """
    # 1. Parsing du fichier de configuration pour extraire les fréquences
    np_counts = {}
    with open(config_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                pid = data.get("metadata", {}).get("problem_id")
                prompt_text = data.get(prompt_key, "")
                
                if pid is not None:
                    np_counts[pid] = prompt_text.count("np.")
            except json.JSONDecodeError:
                continue

    # 2. Parsing du fichier de résultats et agrégation
    # Structure : agg[freq] = {"success": 0, "total": 0}
    agg = defaultdict(lambda: {"success": 0, "total": 0})
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                pid = data.get("metadata", {}).get("problem_id")
                
                if pid in np_counts:
                    freq = np_counts[pid]
                    passed = data.get("passed", False)
                    
                    agg[freq]["total"] += 1
                    agg[freq]["success"] += int(passed)
            except json.JSONDecodeError:
                continue

    if not agg:
        raise ValueError("Aucune donnée correspondante trouvée entre les fichiers config et résultats.")

    # 3. Préparation des vecteurs pour matplotlib et numpy
    frequencies = sorted(agg.keys())
    success_rates = []
    supports = []

    for freq in frequencies:
        total = agg[freq]["total"]
        success = agg[freq]["success"]
        success_rates.append((success / total * 100) if total > 0 else 0.0)
        supports.append(total)

    x_arr = np.array(frequencies)
    y_arr = np.array(success_rates)
    w_arr = np.array(supports)  # Poids pour la régression (WLS)

    # 4. Tracé du graphique
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot avec taille des points proportionnelle à la racine carrée du support
    # (la racine carrée évite que les points massifs écrasent visuellement les petits)
    sizes = np.sqrt(w_arr) * 20 
    ax.scatter(x_arr, y_arr, s=sizes, color='steelblue', alpha=0.8, edgecolors='black', label='Données observées')

    # Annotations des supports
    for i, txt in enumerate(supports):
        ax.annotate(
            f"n={txt}", 
            (x_arr[i], y_arr[i]), 
            xytext=(0, 12), 
            textcoords="offset points", 
            ha='center', 
            fontsize=8
        )

    # Régression linéaire pondérée (WLS)
    if len(x_arr) > 1:
        # np.polyfit utilise les poids matriciels minimisant sum(w * (y - y_pred)^2)
        coeffs = np.polyfit(x_arr, y_arr, deg=1, w=np.sqrt(w_arr))
        poly_fn = np.poly1d(coeffs)
        
        x_line = np.linspace(min(x_arr), max(x_arr), 100)
        y_line = poly_fn(x_line)
        
        # Calcul du pseudo-R^2 pondéré
        y_pred = poly_fn(x_arr)
        y_bar_w = np.average(y_arr, weights=w_arr)
        ss_tot_w = np.sum(w_arr * (y_arr - y_bar_w)**2)
        ss_res_w = np.sum(w_arr * (y_arr - y_pred)**2)
        r2_w = 1 - (ss_res_w / ss_tot_w) if ss_tot_w != 0 else 0.0

        ax.plot(
            x_line, y_line, 
            color='crimson', 
            linestyle='--', 
            label=f"WLS Fit: y = {coeffs[0]:.2f}x + {coeffs[1]:.2f} ($R^2_w$ = {r2_w:.2f})"
        )

    # Formatage
    ax.set_title("Impact de la densité de 'np.' sur la justesse (Taux de succès)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Occurrences de la sous-chaîne 'np.' dans le prompt")
    ax.set_ylabel("Taux de succès (%)")
    ax.set_ylim(0, 105)
    
    # Force les ticks de l'axe X à être des entiers si l'étendue est faible
    if max(x_arr) - min(x_arr) < 20:
        ax.set_xticks(np.arange(min(x_arr), max(x_arr) + 1, 1))

    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Graphique généré et sauvegardé : {output_path}")

# Exemple d'appel
if __name__ == "__main__":
    # plot_success_vs_np_frequency("config.jsonl", "results.jsonl", "np_analysis.png")
    pass