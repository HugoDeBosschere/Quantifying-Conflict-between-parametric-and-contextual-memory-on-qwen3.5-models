#!/usr/bin/env python3
"""
Script de visualisation des résultats d'évaluation (fichiers .jsonl).
- Control classique : is_control + doc_name "nothing" (aucune doc).
- Control avec doc : is_control + doc_name minimal/ultra_minimal (référence avec doc).
- Doc : pas is_control + doc_name minimal/ultra_minimal (modèle avec documentation).

Comparaisons : (1) Control classique seul, (2) Control minimal vs Doc minimal,
(3) Control ultra_minimal vs Doc ultra_minimal.
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def find_repo_root():
    path = os.path.abspath(os.path.dirname(__file__))
    for _ in range(5):
        if os.path.isdir(os.path.join(path, ".git")):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return os.path.abspath(os.path.dirname(__file__))


def _doc_type_label(res: dict) -> str:
    """Un label par (is_control, doc_name) pour les comparaisons voulues."""
    meta = res.get("metadata", {})
    doc_name = meta.get("doc_name") or ""
    is_control = res.get("is_control", False)
    if is_control and doc_name in (None, "", "nothing"):
        return "Control classique"
    if is_control and doc_name == "minimal":
        return "Control minimal"
    if is_control and doc_name == "ultra_minimal":
        return "Control ultra_minimal"
    if not is_control and doc_name == "minimal":
        return "Doc minimal"
    if not is_control and doc_name == "ultra_minimal":
        return "Doc ultra_minimal"
    # Autres cas (autres doc_name ou combinaisons)
    if is_control:
        return f"Control ({doc_name or 'other'})"
    return doc_name or "Other"


def load_data(filepath: str) -> dict:
    """
    Charge un fichier JSONL.
    Retourne: data[model][doc_type][perturbation_type] = {'success': int, 'total': int}
    doc_type ∈ { "Control classique", "Control minimal", "Control ultra_minimal", "Doc minimal", "Doc ultra_minimal", ... }
    """
    data = {}
    print(f"Chargement des données depuis {filepath}...")
    if not os.path.isfile(filepath):
        print(f"❌ Fichier introuvable : {filepath}")
        return {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                res = json.loads(line)
            except json.JSONDecodeError:
                continue

            meta = res.get("metadata", {})
            model = meta.get("model_name", "Unknown")
            pert = meta.get("perturbation_type", "Unknown")
            passed = 1 if res.get("passed", False) else 0
            doc_type = _doc_type_label(res)

            if model not in data:
                data[model] = {}
            if doc_type not in data[model]:
                data[model][doc_type] = {}
            if pert not in data[model][doc_type]:
                data[model][doc_type][pert] = {"success": 0, "total": 0}

            data[model][doc_type][pert]["success"] += passed
            data[model][doc_type][pert]["total"] += 1

    print(f"   Modèles: {list(data.keys())}")
    for m in data:
        print(f"   Types pour {m}: {list(data[m].keys())}")
    return data


def _agg(model_data: dict, doc_type: str) -> tuple[int, int]:
    """(success, total) pour un doc_type donné."""
    if doc_type not in model_data:
        return 0, 0
    s = sum(p["success"] for p in model_data[doc_type].values())
    t = sum(p["total"] for p in model_data[doc_type].values())
    return s, t


# Ordre et style pour le graphique global (control classique + control vs doc minimal + control vs doc ultra_minimal)
GLOBAL_SERIES = [
    ("Control classique", {"color": "gray", "alpha": 0.9, "hatch": "//", "edgecolor": "black"}),
    ("Control minimal", {"color": "lightgray", "alpha": 0.9, "hatch": "\\\\", "edgecolor": "black"}),
    ("Doc minimal", {"color": "steelblue", "alpha": 0.9, "hatch": "", "edgecolor": "black"}),
    ("Control ultra_minimal", {"color": "gainsboro", "alpha": 0.95, "hatch": "xx", "edgecolor": "black"}),
    ("Doc ultra_minimal", {"color": "coral", "alpha": 0.9, "hatch": "", "edgecolor": "black"}),
]


def plot_global_all_conditions(data: dict, output_dir: str) -> None:
    """Un seul graphique : Control classique + Control minimal vs Doc minimal + Control ultra_minimal vs Doc ultra_minimal.
    Barres espacées, noms des conditions sous les barres, titre dans le graphique.
    """
    models = sorted(data.keys())
    if not models:
        return

    n_series = len(GLOBAL_SERIES)
    n_models = len(models)
    # Espacement : un bloc par modèle, dans chaque bloc 5 barres avec un peu d'écart
    bar_width = 0.5
    bar_gap = 0.12
    group_gap = 1.2
    step = bar_width + bar_gap
    group_width = n_series * step - bar_gap
    # Position du centre de chaque groupe (modèle)
    group_centers = [i * (group_width + group_gap) for i in range(n_models)]
    # Pour chaque série i et chaque modèle m : position x de la barre
    x_positions = []
    all_scores = []
    all_labels_text = []
    xtick_pos = []
    xtick_labels = []

    for m_idx, m in enumerate(models):
        base = group_centers[m_idx]
        for i, (label, style) in enumerate(GLOBAL_SERIES):
            pos = base + i * step
            x_positions.append(pos)
            s, t = _agg(data[m], label)
            if t > 0:
                pct = s / t * 100
                all_scores.append(pct)
                all_labels_text.append(f"{pct:.0f}%\n({s}/{t})")
            else:
                all_scores.append(0)
                all_labels_text.append("")
            xtick_pos.append(pos)
            xtick_labels.append(label)

    x_positions = np.array(x_positions)
    all_scores = np.array(all_scores)

    fig, ax = plt.subplots(figsize=(max(12, n_models * 5), 7))

    # Dessiner les barres une par une pour garder les styles par série
    for i in range(len(x_positions)):
        idx_series = i % n_series
        style = GLOBAL_SERIES[idx_series][1]
        rects = ax.bar(
            x_positions[i],
            all_scores[i],
            bar_width,
            color=style["color"],
            alpha=style["alpha"],
            hatch=style["hatch"],
            edgecolor=style["edgecolor"],
        )
        ax.bar_label(rects, labels=[all_labels_text[i]], padding=2, fontsize=8, fontweight="bold")

    ax.set_ylabel("Taux de succès (%)")
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 115)
    # Titre court à l'intérieur du graphique (en haut, centré)
    ax.set_title("Performance par condition (control classique, minimal, ultra_minimal)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = os.path.join(output_dir, "plot_global_control_vs_doc_all.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"✅ Sauvegardé : {out}")


def _perturbation_data_one(data: dict, models: list, control_label: str, doc_label: str) -> tuple:
    """Récupère pert_types et les listes de scores/labels pour un (control_label, doc_label)."""
    all_perts = set()
    for m in data:
        for d in (control_label, doc_label):
            if d in data[m]:
                all_perts.update(data[m][d].keys())
    pert_types = sorted(all_perts)
    return pert_types


def plot_combined_perturbation_minimal_ultra(data: dict, output_dir: str) -> None:
    """Un seul graphique avec 2 sous-graphiques : Minimal (control vs doc) et Ultra_minimal (control vs doc) par perturbation."""
    models = sorted(data.keys())
    if not models:
        return

    perts_minimal = _perturbation_data_one(data, models, "Control minimal", "Doc minimal")
    perts_ultra = _perturbation_data_one(data, models, "Control ultra_minimal", "Doc ultra_minimal")
    pert_types = sorted(set(perts_minimal) | set(perts_ultra))
    if not pert_types:
        return

    # 2 lignes : Minimal, Ultra_minimal. Pour chaque ligne, un subplot par modèle (ou un seul si 1 modèle)
    n_models = len(models)
    fig, axes = plt.subplots(nrows=2, ncols=max(1, n_models), figsize=(max(14, 4 * n_models), 10))
    if n_models == 1:
        axes = axes.reshape(2, 1)

    x = np.arange(len(pert_types))
    width = 0.4

    for row, (control_label, doc_label, row_title) in enumerate([
        ("Control minimal", "Doc minimal", "Minimal — Control vs Doc"),
        ("Control ultra_minimal", "Doc ultra_minimal", "Ultra_minimal — Control vs Doc"),
    ]):
        for col, model in enumerate(models):
            ax = axes[row, col]
            sc_list, sd_list, lc_list, ld_list = [], [], [], []
            for pert in pert_types:
                if control_label in data[model] and pert in data[model][control_label]:
                    st = data[model][control_label][pert]
                    s, t = st["success"], st["total"]
                    pct = (s / t * 100) if t else 0
                    sc_list.append(pct)
                    lc_list.append(f"{pct:.0f}%\n({s}/{t})" if t else "")
                else:
                    sc_list.append(0)
                    lc_list.append("")
                if doc_label in data[model] and pert in data[model][doc_label]:
                    st = data[model][doc_label][pert]
                    s, t = st["success"], st["total"]
                    pct = (s / t * 100) if t else 0
                    sd_list.append(pct)
                    ld_list.append(f"{pct:.0f}%\n({s}/{t})" if t else "")
                else:
                    sd_list.append(0)
                    ld_list.append("")

            pos_c = x - width / 2
            pos_d = x + width / 2
            r_c = ax.bar(pos_c, sc_list, width, label=control_label, color="gray", alpha=0.7, hatch="//")
            r_d = ax.bar(pos_d, sd_list, width, label=doc_label, color="steelblue", alpha=0.9)
            ax.bar_label(r_c, labels=[l if v > 0 else "" for l, v in zip(lc_list, sc_list)], padding=1, fontsize=6)
            ax.bar_label(r_d, labels=[l if v > 0 else "" for l, v in zip(ld_list, sd_list)], padding=1, fontsize=6)
            ax.set_title(f"{row_title} — {model}", fontsize=11, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(pert_types, rotation=20, ha="right")
            ax.set_ylabel("Précision (%)")
            ax.set_ylim(0, 115)
            ax.legend(loc="upper right", fontsize="small")
            ax.grid(axis="y", linestyle=":", alpha=0.5)

    plt.suptitle("Détail par perturbation : Minimal et Ultra_minimal (Control vs Doc)", fontsize=13, fontweight="bold", y=1.002)
    plt.tight_layout()
    out = os.path.join(output_dir, "plot_perturbation_minimal_ultra_combined.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"✅ Sauvegardé : {out}")


def main():
    repo = find_repo_root()
    default_input = os.path.join(repo, "results", "qwen_result3.jsonl")
    default_output = os.path.join(repo, "results")

    parser = argparse.ArgumentParser(
        description="Plot des résultats : control classique, minimal (control vs doc), ultra_minimal (control vs doc)."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=default_input,
        help="Fichier JSONL de résultats",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Répertoire de sortie des graphiques",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input_file)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else (os.path.dirname(input_path) or default_output)
    os.makedirs(output_dir, exist_ok=True)

    data = load_data(input_path)
    if not data:
        sys.exit(1)

    # Graphique global : control classique + control vs doc (minimal + ultra_minimal) sur un même graph
    plot_global_all_conditions(data, output_dir)

    # Détail par perturbation : Minimal et Ultra_minimal combinés en un seul fichier
    plot_combined_perturbation_minimal_ultra(data, output_dir)


if __name__ == "__main__":
    main()
