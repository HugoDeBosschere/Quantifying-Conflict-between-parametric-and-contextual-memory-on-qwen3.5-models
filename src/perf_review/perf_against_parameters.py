#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif

import matplotlib.pyplot as plt
import numpy as np

def plot_dual_scaling():
    # Données brutes
    parametres = [0.8, 2, 4, 9, 27, 35]
    performances = [15, 42, 67, 72, 80, 81]
    labels = ["0.8b", "2b", "4b", "9b", "27b", "35b"]

    # Création de la figure avec deux colonnes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Analyse du Scaling : Qwen 3.5 Performance vs Parameters", fontsize=14, fontweight="bold")

    # --- PLOT 1 : ÉCHELLE LINÉAIRE ---
    ax1.plot(parametres, performances, marker='o', color='#2c3e50', linewidth=2, label='Success Rate')
    ax1.set_title("Échelle Linéaire (Visualisation de la saturation)", fontsize=11)
    ax1.set_xlabel("Paramètres (Milliards)")
    ax1.set_ylabel("Performance (%)")
    ax1.set_ylim(0, 100)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- PLOT 2 : ÉCHELLE LOGARITHMIQUE (X) ---
    ax2.plot(parametres, performances, marker='s', color='#e67e22', linewidth=2, label='Success Rate (Log)')
    ax2.set_xscale('log')
    ax2.set_title("Échelle Semi-Log (Analyse de la loi de puissance)", fontsize=11)
    ax2.set_xlabel("Paramètres (Milliards) - Log Scale")
    ax2.set_ylabel("Performance (%)")
    ax2.set_ylim(0, 100)
    
    # Ajustement des ticks pour le log
    ax2.set_xticks(parametres)
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.grid(True, which="both", linestyle=':', alpha=0.6)

    # Annotations communes
    for ax in [ax1, ax2]:
        for x, y, label in zip(parametres, performances, labels):
            ax.annotate(
                f"{label}\n{y}%",
                xy=(x, y),
                xytext=(0, 10),
                textcoords="offset points",
                ha='center',
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#bdc3c7", alpha=0.8)
            )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output = "qwen_dual_scaling.png"
    plt.savefig(output, dpi=300)
    print(f"✅ Graphique double généré : {output}")

if __name__ == "__main__":
    plot_dual_scaling()