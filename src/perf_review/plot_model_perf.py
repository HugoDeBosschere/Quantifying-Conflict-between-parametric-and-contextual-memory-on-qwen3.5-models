#!/usr/bin/env python3
"""
Script de visualisation des résultats d'évaluation (fichiers .jsonl).

Deux métriques distinctes :
- run_control (is_control=True) : le LLM reçoit problème + lib d'origine ; on évalue avec la lib d'origine.
  → Control classique / Control minimal / Control ultra_minimal (selon doc_name), via le flag passed.
- control_passed (lignes injection) : le LLM a reçu problème + doc modifiés ; on évalue la même réponse
  avec la lib d'origine. → "Doc minimal (éval. lib d'origine)" / "Doc ultra_minimal (éval. lib d'origine)".

Doc minimal / Doc ultra_minimal (sans suffixe) = injection, évaluation avec la lib modifiée (passed).
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
    # Pas de "Control explanation" : seul le cas injection avec doc explanation existe
    if not is_control and doc_name == "minimal":
        return "Doc minimal"
    if not is_control and doc_name == "ultra_minimal":
        return "Doc ultra_minimal"
    if not is_control and doc_name == "explanation":
        return "Doc explanation"
    # Autres cas (autres doc_name ou combinaisons)
    if is_control:
        return f"Control ({doc_name or 'other'})"
    return doc_name or "Other"


def load_data(filepath: str) -> dict:
    """
    Charge un fichier JSONL.
    Retourne: data[model][doc_type][perturbation_type] = {'success': int, 'total': int}
    doc_type : Control classique/minimal/ultra_minimal (run_control, passed) ;
               Doc minimal / Doc ultra_minimal (injection, passed) ;
               Doc minimal (éval. lib d'origine) / Doc ultra_minimal (éval. lib d'origine) (injection, control_passed).
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
            model = meta.get("model_name") or ""
            if not model or model.strip() == "":
                # Lignes sans metadata (ex. anciennes LLM_API_FAILURE) : on ignore pour le plot
                continue
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

            # Métrique control_passed (injection) : même code, évalué avec la lib d'origine.
            # Série distincte de run_control (contexte d'origine).
            if doc_type == "Doc minimal":
                eval_orig_label = "Doc minimal (éval. lib d'origine)"
                cp = 1 if res.get("control_passed", False) else 0
                if eval_orig_label not in data[model]:
                    data[model][eval_orig_label] = {}
                if pert not in data[model][eval_orig_label]:
                    data[model][eval_orig_label][pert] = {"success": 0, "total": 0}
                data[model][eval_orig_label][pert]["success"] += cp
                data[model][eval_orig_label][pert]["total"] += 1
            elif doc_type == "Doc ultra_minimal":
                eval_orig_label = "Doc ultra_minimal (éval. lib d'origine)"
                cp = 1 if res.get("control_passed", False) else 0
                if eval_orig_label not in data[model]:
                    data[model][eval_orig_label] = {}
                if pert not in data[model][eval_orig_label]:
                    data[model][eval_orig_label][pert] = {"success": 0, "total": 0}
                data[model][eval_orig_label][pert]["success"] += cp
                data[model][eval_orig_label][pert]["total"] += 1
            elif doc_type == "Doc explanation":
                eval_orig_label = "Doc explanation (éval. lib d'origine)"
                cp = 1 if res.get("control_passed", False) else 0
                if eval_orig_label not in data[model]:
                    data[model][eval_orig_label] = {}
                if pert not in data[model][eval_orig_label]:
                    data[model][eval_orig_label][pert] = {"success": 0, "total": 0}
                data[model][eval_orig_label][pert]["success"] += cp
                data[model][eval_orig_label][pert]["total"] += 1

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


# Ordre et style : run_control (contexte d'origine) vs injection (lib modifiée) vs injection éval. lib d'origine
GLOBAL_SERIES = [
    ("Control classique", {"color": "gray", "alpha": 0.9, "hatch": "//", "edgecolor": "black"}),
    ("Control minimal", {"color": "lightgray", "alpha": 0.9, "hatch": "\\\\", "edgecolor": "black"}),
    ("Doc minimal", {"color": "steelblue", "alpha": 0.9, "hatch": "", "edgecolor": "black"}),
    ("Doc minimal (éval. lib d'origine)", {"color": "steelblue", "alpha": 0.5, "hatch": "..", "edgecolor": "black"}),
    ("Control ultra_minimal", {"color": "gainsboro", "alpha": 0.95, "hatch": "xx", "edgecolor": "black"}),
    ("Doc ultra_minimal", {"color": "coral", "alpha": 0.9, "hatch": "", "edgecolor": "black"}),
    ("Doc ultra_minimal (éval. lib d'origine)", {"color": "coral", "alpha": 0.5, "hatch": "..", "edgecolor": "black"}),
    ("Doc explanation", {"color": "seagreen", "alpha": 0.9, "hatch": "", "edgecolor": "black"}),
    ("Doc explanation (éval. lib d'origine)", {"color": "seagreen", "alpha": 0.5, "hatch": "..", "edgecolor": "black"}),
]


def _series_with_data(data: dict, models: list, series_list: list) -> list:
    """Ne garde que les séries (label, style) pour lesquelles au moins un modèle a des données (total > 0)."""
    return [
        (label, style)
        for label, style in series_list
        if any(_agg(data[m], label)[1] > 0 for m in models)
    ]


def plot_global_all_conditions(data: dict, output_dir: str) -> None:
    """Un seul graphique : run_control (Control classique/minimal/ultra_minimal) et injection
    (Doc minimal/ultra_minimal + Doc minimal/ultra_minimal éval. lib d'origine). Barres espacées.
    N'affiche que les types de doc qui ont des données dans le fichier résultat.
    """
    models = sorted(data.keys())
    if not models:
        return

    active_series = _series_with_data(data, models, GLOBAL_SERIES)
    if not active_series:
        print("⚠ Aucune série avec données, plot global ignoré.")
        return

    n_series = len(active_series)
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
        for i, (label, style) in enumerate(active_series):
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
        style = active_series[idx_series][1]
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
    # Étiquettes de groupe (nom du modèle) AU-DESSUS des barres pour éviter le chevauchement avec les labels d'axe
    top_y = ax.get_ylim()[1] - 3
    for m_idx, m in enumerate(models):
        center_x = group_centers[m_idx] + (n_series * step - bar_gap) / 2 - bar_width / 2
        ax.text(center_x, top_y, m, fontsize=9, fontweight="bold", ha="center", va="top")
    ax.set_title("Performance par condition — run_control (contexte orig.) vs injection (éval. lib modifiée / lib d'origine)", fontsize=11, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = os.path.join(output_dir, "plot_global_control_vs_doc_all.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"✅ Sauvegardé : {out}")


def _perturbation_data_one(data: dict, models: list, *doc_labels: str) -> set:
    """Récupère pert_types pour les séries données."""
    all_perts = set()
    for m in data:
        for d in doc_labels:
            if d in data[m]:
                all_perts.update(data[m][d].keys())
    return all_perts


def _row_has_data(data: dict, models: list, control_label: str | None, doc_label: str, doc_orig_label: str) -> bool:
    """True si au moins un modèle a des données pour au moins une des séries de la row (control optionnel)."""
    labels = [doc_label, doc_orig_label]
    if control_label is not None:
        labels.append(control_label)
    for m in models:
        for label in labels:
            if label in data[m] and any(p["total"] > 0 for p in data[m][label].values()):
                return True
    return False


# Config des rows par type de doc : (control ou None, doc, doc_éval_orig, titre, couleur). None = pas de control pour ce doc (ex. explanation).
DOC_ROWS_CONFIG = [
    ("Control minimal", "Doc minimal", "Doc minimal (éval. lib d'origine)", "Minimal — run_control vs injection (éval. lib mod. / lib d'origine)", "steelblue"),
    ("Control ultra_minimal", "Doc ultra_minimal", "Doc ultra_minimal (éval. lib d'origine)", "Ultra_minimal — run_control vs injection (éval. lib mod. / lib d'origine)", "coral"),
    (None, "Doc explanation", "Doc explanation (éval. lib d'origine)", "Explanation — injection (éval. lib mod. / lib d'origine)", "seagreen"),
]


def plot_combined_perturbation_minimal_ultra(data: dict, output_dir: str) -> None:
    """Sous-graphiques par type de doc (Minimal, Ultra_minimal, Explanation). Pour chaque : run_control (Control), injection (Doc), injection éval. lib d'origine.
    N'affiche que les types de doc qui ont des données dans le fichier résultat.
    """
    models = sorted(data.keys())
    if not models:
        return

    # Ne garder que les rows qui ont des données
    rows_config = [
        (control_label, doc_label, doc_orig_label, row_title, doc_color)
        for control_label, doc_label, doc_orig_label, row_title, doc_color in DOC_ROWS_CONFIG
        if _row_has_data(data, models, control_label, doc_label, doc_orig_label)
    ]
    if not rows_config:
        print("⚠ Aucune row (minimal/ultra_minimal/explanation) avec données, plot perturbation ignoré.")
        return

    all_perts = set()
    for control_label, doc_label, doc_orig_label, _t, _c in rows_config:
        labels = [doc_label, doc_orig_label]
        if control_label is not None:
            labels.append(control_label)
        all_perts |= _perturbation_data_one(data, models, *labels)
    pert_types = sorted(all_perts)
    if not pert_types:
        return

    n_rows = len(rows_config)
    n_models = len(models)
    fig, axes = plt.subplots(nrows=n_rows, ncols=max(1, n_models), figsize=(max(14, 4 * n_models), 4 * n_rows))
    if n_models == 1 and n_rows == 1:
        axes = np.array([[axes]])
    elif n_models == 1:
        axes = axes.reshape(-1, 1)
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    x = np.arange(len(pert_types))
    width = 0.26

    for row, (control_label, doc_label, doc_orig_label, row_title, doc_color) in enumerate(rows_config):
        has_control = control_label is not None
        for col, model in enumerate(models):
            ax = axes[row, col]
            sc_list, sd_list, so_list = [], [], []
            lc_list, ld_list, lo_list = [], [], []
            series_spec = [(doc_label, sd_list, ld_list), (doc_orig_label, so_list, lo_list)]
            if has_control:
                series_spec.insert(0, (control_label, sc_list, lc_list))
            for pert in pert_types:
                for label, s_list, l_list in series_spec:
                    if label in data[model] and pert in data[model][label]:
                        st = data[model][label][pert]
                        s, t = st["success"], st["total"]
                        pct = (s / t * 100) if t else 0
                        s_list.append(pct)
                        l_list.append(f"{pct:.0f}%\n({s}/{t})" if t else "")
                    else:
                        s_list.append(0)
                        l_list.append("")

            if has_control:
                pos_c, pos_d, pos_o = x - width, x, x + width
                r_c = ax.bar(pos_c, sc_list, width, label=control_label, color="gray", alpha=0.7, hatch="//")
                r_d = ax.bar(pos_d, sd_list, width, label=doc_label, color=doc_color, alpha=0.9)
                r_o = ax.bar(pos_o, so_list, width, label=doc_orig_label, color=doc_color, alpha=0.5, hatch="..")
                ax.bar_label(r_c, labels=[l if v > 0 else "" for l, v in zip(lc_list, sc_list)], padding=1, fontsize=6)
                ax.bar_label(r_d, labels=[l if v > 0 else "" for l, v in zip(ld_list, sd_list)], padding=1, fontsize=6)
                ax.bar_label(r_o, labels=[l if v > 0 else "" for l, v in zip(lo_list, so_list)], padding=1, fontsize=6)
            else:
                pos_d = x - width / 2
                pos_o = x + width / 2
                r_d = ax.bar(pos_d, sd_list, width, label=doc_label, color=doc_color, alpha=0.9)
                r_o = ax.bar(pos_o, so_list, width, label=doc_orig_label, color=doc_color, alpha=0.5, hatch="..")
                ax.bar_label(r_d, labels=[l if v > 0 else "" for l, v in zip(ld_list, sd_list)], padding=1, fontsize=6)
                ax.bar_label(r_o, labels=[l if v > 0 else "" for l, v in zip(lo_list, so_list)], padding=1, fontsize=6)
            # Titre compact pour éviter les chevauchements : seulement le nom du modèle
            ax.set_title(model, fontsize=9, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(pert_types, rotation=20, ha="right")
            ax.set_ylabel("Précision (%)")
            ax.set_ylim(0, 115)
            ax.legend(loc="upper right", fontsize="small")
            ax.grid(axis="y", linestyle=":", alpha=0.5)

    plt.suptitle("Détail par perturbation — run_control (contexte orig.) vs injection (éval. lib modifiée / éval. lib d'origine)", fontsize=12, fontweight="bold", y=1.002)
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
        description="Plot des résultats : run_control (contexte d'origine) vs injection (éval. lib modifiée / éval. lib d'origine)."
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
