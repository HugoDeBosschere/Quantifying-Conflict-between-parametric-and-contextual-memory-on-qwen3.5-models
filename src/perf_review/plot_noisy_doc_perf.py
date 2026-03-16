#!/usr/bin/env python3
"""
Histogramme des taux de réussite par documentation (runs avec docs bruitées).

Adapté aux résultats où doc_name peut être :
  - Control : nothing, minimal, ultra_minimal (is_control=True)
  - Injection : minimal_noise0, minimal_noise25, ..., minimal_noise100,
                ultra_minimal_noise0, ..., ultra_minimal_noise100

Gère un ou plusieurs modèles : une barre par (modèle, doc) ou groupes par doc.

Usage (depuis la racine du projet) :
  python3 src/perf_review/plot_noisy_doc_perf.py results/run_xxx/results.jsonl
  python3 src/perf_review/plot_noisy_doc_perf.py results/run_xxx/results.jsonl -o results/run_xxx
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def _normalize_doc_label(doc_name: str, is_control: bool) -> str:
    """Label d'affichage pour une doc (control ou injection)."""
    if is_control:
        return f"Control: {doc_name or 'nothing'}"
    return doc_name or "injection"


def _injection_doc_sort_key(doc_name: str) -> tuple:
    """
    Clé de tri pour les doc_name d'injection : (famille, noise_pct).
    minimal_noise50 -> (0, 50), ultra_minimal_noise25 -> (1, 25).
    """
    if not doc_name or "_noise" not in doc_name:
        return (0, 0)
    base, noise = doc_name.split("_noise", 1)
    family = 1 if base.startswith("ultra") else 0
    try:
        pct = int(noise)
    except ValueError:
        pct = 0
    return (family, pct)


def load_noisy_doc_data(filepath: str) -> tuple[dict, list]:
    """
    Charge results.jsonl et construit data[model][doc_label] = {'success': int, 'total': int}.
    Retourne (data, ordered_doc_labels) pour un ordre d'affichage cohérent.
    """
    data = defaultdict(lambda: defaultdict(lambda: {"success": 0, "total": 0}))

    if not os.path.isfile(filepath):
        print(f"❌ Fichier introuvable : {filepath}", file=sys.stderr)
        return {}, []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                res = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = res.get("metadata") or {}
            model = (meta.get("model_name") or "").strip()
            if not model:
                continue
            doc_name = meta.get("doc_name") or ""
            is_control = res.get("is_control", False)
            label = _normalize_doc_label(doc_name, is_control)
            data[model][label]["total"] += 1
            if res.get("passed", False):
                data[model][label]["success"] += 1

    # Ordre d'affichage : controls d'abord (nothing, minimal, ultra_minimal), puis injection par famille et noise
    control_order = ["Control: nothing", "Control: minimal", "Control: ultra_minimal"]
    all_labels = set().union(*(data[m].keys() for m in data))
    control_seen = {lb for lb in all_labels if lb.startswith("Control:")}
    control_sorted = [c for c in control_order if c in control_seen]

    injection_labels = sorted(
        [lb for lb in all_labels if not lb.startswith("Control:")],
        key=_injection_doc_sort_key,
    )
    ordered_labels = control_sorted + injection_labels

    print(f"   Modèles: {sorted(data.keys())}")
    print(f"   Docs (ordre): {ordered_labels[:5]}... ({len(ordered_labels)} au total)")
    return dict(data), ordered_labels


def _short_label(label: str) -> str:
    """Raccourcit les noms pour l'axe x (éviter chevauchement)."""
    if label.startswith("Control:"):
        s = label.replace("Control:", "Ctrl").strip()
        if s == "nothing":
            return "Ctrl: none"
        if s == "minimal":
            return "Ctrl: min"
        if s == "ultra_minimal":
            return "Ctrl: ultra"
        return s[:12]
    # minimal_noise50 -> min 50%
    m = re.match(r"(minimal|ultra_minimal)_noise(\d+)", label)
    if m:
        base = "min" if m.group(1) == "minimal" else "ultra"
        return f"{base} {m.group(2)}%"
    return label[:14] if len(label) > 14 else label


def _color_for_doc_label(label: str) -> str:
    """
    Couleur par type de doc : controls (gris), injection minimal (bleu), injection ultra_minimal (orange).
    """
    if label == "Control: nothing":
        return "#2d2d2d"   # gris très foncé
    if label == "Control: minimal":
        return "#5c5c5c"   # gris moyen
    if label == "Control: ultra_minimal":
        return "#8c8c8c"   # gris clair
    if label.startswith("minimal_noise"):
        return "#3498db"   # bleu
    if label.startswith("ultra_minimal_noise"):
        return "#e67e22"   # orange
    return "#95a5a6"


def plot_noisy_doc_histogram(data: dict, ordered_labels: list, output_path: str) -> None:
    """
    Histogramme : abscisse = docs (controls + injection), ordonnée = taux de réussite.
    - Couleurs : controls (gris), minimal_* (bleu), ultra_minimal_* (orange).
    - Texte au-dessus de chaque barre : "success/total" (ex. 65/159).
    - Une barre par (modèle, doc) ; si plusieurs modèles, barres groupées par doc.
    """
    models = sorted(data.keys())
    if not models or not ordered_labels:
        print("⚠ Aucune donnée à afficher.", file=sys.stderr)
        return

    labels_with_data = [lb for lb in ordered_labels if any(data[m].get(lb, {}).get("total", 0) > 0 for m in models)]
    if not labels_with_data:
        print("⚠ Aucun label avec des données.", file=sys.stderr)
        return

    n_docs = len(labels_with_data)
    n_models = len(models)
    x = np.arange(n_docs)
    total_width = 0.8
    bar_width = total_width / n_models if n_models else total_width
    offset = (np.arange(n_models) - (n_models - 1) / 2) * bar_width if n_models > 1 else 0

    fig, ax = plt.subplots(figsize=(max(10, n_docs * 0.5), 6))
    # Hachures pour distinguer les modèles quand il y en a plusieurs
    hatches = ["", "//", "\\\\", "xx", "++"][: max(1, n_models)]

    for i, model in enumerate(models):
        rates = []
        counts = []  # (success, total) pour le texte
        colors = []
        for lb in labels_with_data:
            st = data[model].get(lb, {"success": 0, "total": 0})
            s, t = st["success"], st["total"]
            r = (s / t * 100) if t else 0
            rates.append(r)
            counts.append((s, t))
            colors.append(_color_for_doc_label(lb))

        pos = x + (offset[i] if n_models > 1 else 0)
        bars = ax.bar(
            pos, rates,
            width=bar_width,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            hatch=hatches[i] if n_models > 1 else None,
        )

        # Texte "success/total" au-dessus de chaque barre
        for j, (bar, (s, t)) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1.5,
                f"{s}/{t}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([_short_label(lb) for lb in labels_with_data], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Taux de réussite (%)")
    ax.set_xlabel("Documentation")
    ax.set_ylim(0, 115)  # marge pour le texte au-dessus des barres

    # Légende des types de doc (couleurs)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2d2d2d", edgecolor="white", label="Control: nothing"),
        Patch(facecolor="#5c5c5c", edgecolor="white", label="Control: minimal"),
        Patch(facecolor="#8c8c8c", edgecolor="white", label="Control: ultra_minimal"),
        Patch(facecolor="#3498db", edgecolor="white", label="Injection: minimal"),
        Patch(facecolor="#e67e22", edgecolor="white", label="Injection: ultra_minimal"),
    ]
    if n_models > 1:
        model_patches = [
            Patch(facecolor="lightgray", edgecolor="gray", label=m, hatch=hatches[i])
            for i, m in enumerate(models)
        ]
        ax.legend(handles=legend_elements + model_patches, loc="upper right", fontsize=7, ncol=1, title="Type de doc | Modèle")
    else:
        ax.legend(handles=legend_elements, loc="upper right", fontsize=7, title="Type de doc")

    ax.set_title("Réussite par documentation (controls + injection bruitée)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Graphique enregistré : {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Histogramme réussite par documentation (runs avec docs bruitées).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "results_file",
        help="Chemin vers results.jsonl",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Répertoire de sortie (défaut: même dossier que le fichier results)",
    )
    parser.add_argument(
        "--output-name",
        default="plot_noisy_doc_perf.png",
        help="Nom du fichier image (défaut: plot_noisy_doc_perf.png)",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.results_file)
    output_dir = args.output_dir or os.path.dirname(input_path) or "."
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, args.output_name)

    data, ordered_labels = load_noisy_doc_data(input_path)
    if not data:
        sys.exit(1)

    plot_noisy_doc_histogram(data, ordered_labels, output_path)


if __name__ == "__main__":
    main()
