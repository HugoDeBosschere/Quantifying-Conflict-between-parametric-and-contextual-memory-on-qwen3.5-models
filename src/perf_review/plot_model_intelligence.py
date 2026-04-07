#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def extract_parameters(model_name: str) -> float | None:
    """Extrait le nombre de paramètres en milliards (B)."""
    match = re.search(r'(\d+(?:\.\d+)?)([bm])', model_name, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        unit = match.group(2).lower()
        if unit == 'm':
            return val / 1000.0
        return val
    return None


def _get_doc_category(doc_name: str) -> str:
    """Normalise la catégorie de documentation."""
    doc_name = (doc_name or "").strip()
    if not doc_name or doc_name == "nothing":
        return "control"
    if doc_name.startswith("explanation"):
        return "explanation"
    if doc_name in ["minimal", "ultra_minimal"]:
        return doc_name
    return doc_name


def parse_and_compute(input_dir: str, relative: bool = False) -> dict[str, list[dict]]:
    """
    Parcourt le dossier pour extraire les cardinalités des ensembles d'évaluation.
    Calcule les métriques absolues ou relatives par rapport au contrôle.
    """
    stats = defaultdict(lambda: {
        "ctrl_tot": 0, 
        "ctrl_pass": 0,
        "docs": defaultdict(lambda: {
            "inj_tot": 0, 
            "inj_pass": 0, 
            "inj_union_pass": 0, 
            "confusion_count": 0
        })
    })

    target_err = "TEST_FAILED: Assertion incorrecte"

    for root, _, files in os.walk(input_dir):
        for file in files:
            if not (file == "results.json" or file.endswith(".jsonl")):
                continue
            
            filepath = os.path.join(root, file)
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
                    model = meta.get("model_name", "").strip()
                    if not model:
                        continue

                    is_control = res.get("is_control", meta.get("mode") == "control")
                    passed = res.get("passed", False)
                    
                    if is_control:
                        stats[model]["ctrl_tot"] += 1
                        stats[model]["ctrl_pass"] += int(passed)
                    else:
                        doc_type = _get_doc_category(meta.get("doc_name", ""))
                        if doc_type == "control":
                            continue
                            
                        control_passed = res.get("control_passed", False)
                        d_stats = stats[model]["docs"][doc_type]
                        
                        d_stats["inj_tot"] += 1
                        d_stats["inj_pass"] += int(passed)
                        d_stats["inj_union_pass"] += int(passed or control_passed)
                        
                        if not passed and not control_passed:
                            stdout_inj = res.get("stdout", "") or ""
                            stdout_ctrl = res.get("stdout_control", "") or ""
                            if target_err in stdout_inj or target_err in stdout_ctrl:
                                d_stats["confusion_count"] += 1

    metrics_by_doc = defaultdict(list)
    
    for model, s in stats.items():
        params = extract_parameters(model)
        if params is None:
            continue
            
        N_ctrl = s["ctrl_tot"]
        if N_ctrl == 0:
            print(f"⚠ Modèle ignoré (aucun run de contrôle trouvé) : {model}", file=sys.stderr)
            continue
            
        rate_ctrl = s["ctrl_pass"] / N_ctrl
        
        if relative and rate_ctrl == 0:
            print(f"⚠ Modèle ignoré en mode relatif (performance de contrôle = 0) : {model}", file=sys.stderr)
            continue
        
        for doc_type, d in s["docs"].items():
            N_inj = d["inj_tot"]
            if N_inj == 0:
                continue
                
            rate_inj = d["inj_pass"] / N_inj
            rate_inj_union = d["inj_union_pass"] / N_inj
            rate_confusion = d["confusion_count"] / N_inj
            
            l_perf = (rate_ctrl - rate_inj)
            l_comp = (rate_ctrl - rate_inj_union)
            l_int  = rate_confusion
            l_coh  = l_comp - l_int
            
            if relative:
                l_perf = (l_perf / rate_ctrl) * 100
                l_comp = (l_comp / rate_ctrl) * 100
                l_int  = (l_int / rate_ctrl) * 100
                l_coh  = (l_coh / rate_ctrl) * 100
            else:
                l_perf *= 100
                l_comp *= 100
                l_int  *= 100
                l_coh  *= 100
            
            metrics_by_doc[doc_type].append({
                "model": model,
                "params": params,
                "l_perf": l_perf,
                "l_comp": l_comp,
                "l_int": l_int,
                "l_coh": l_coh
            })
            
    for doc_type in metrics_by_doc:
        metrics_by_doc[doc_type].sort(key=lambda x: x["params"])
        
    return metrics_by_doc


def plot_metrics_for_doc(doc_type: str, metrics: list[dict], output_dir: str, relative: bool = False):
    """Tracé du grid 2x2 en échelle semi-log."""
    if not metrics:
        return

    params = [m["params"] for m in metrics]
    labels = [m["model"].split(':')[-1] for m in metrics]
    
    l_perf = [m["l_perf"] for m in metrics]
    l_comp = [m["l_comp"] for m in metrics]
    l_int = [m["l_int"] for m in metrics]
    l_coh = [m["l_coh"] for m in metrics]

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    
    title_suffix = "(Relative à la baseline)" if relative else "(Absolue)"
    ylabel = "Variation Relative (%)" if relative else "Variation (Points de %)"
    
    fig.suptitle(f"Dégradation des capacités {title_suffix} - Injection : {doc_type}", fontsize=14, fontweight="bold", y=0.98)

    plots_config = [
        (axs[0, 0], l_perf, "Perte de performance ($L_{perf}$)", "purple"),
        (axs[0, 1], l_comp, "Perte de compétence ($L_{comp}$)", "crimson"),
        (axs[1, 0], l_int,  "Perte d'intelligence ($L_{int}$)", "steelblue"),
        (axs[1, 1], l_coh,  "Perte de cohérence ($L_{coh}$)", "forestgreen")
    ]

    max_y = max(max(l_perf), max(l_comp), max(l_int), max(l_coh), 10)
    min_y = min(0, min(l_coh))
    # Ajustement des marges selon l'échelle relative (qui peut monter jusqu'à 100% ou plus)
    margin = max_y * 0.15 if relative else 15
    y_limit = (min_y - margin/2, max_y + margin)

    for ax, data, title, color in plots_config:
        ax.plot(params, data, marker='o', linestyle='-', color=color, linewidth=2)
        
        for x, y, lbl in zip(params, data, labels):
            ax.annotate(
                f"{lbl}\n{y:.1f}%", 
                (x, y), 
                textcoords="offset points", 
                xytext=(0, 10), 
                ha='center', 
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7)
            )

        ax.set_xscale('log')
        ax.set_xticks(params)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        
        ax.set_ylim(y_limit)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Nombre de paramètres (Milliards) - Log Scale")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", linestyle=':', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    suffix = "_relative" if relative else "_absolue"
    out_path = os.path.join(output_dir, f"metrics_{doc_type}{suffix}.pdf")
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"✅ Fichier généré : {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Calcule et trace L_perf, L_comp, L_int, L_coh à partir des résultats d'injection.")
    parser.add_argument("input_dir", help="Dossier racine contenant les fichiers results.json / results.jsonl")
    parser.add_argument("-o", "--output-dir", default=".", help="Dossier d'exportation des PDF")
    parser.add_argument("--relative", action="store_true", help="Calcule la perte relative à la performance de contrôle (perte / ctrl).")
    
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"❌ Dossier introuvable : {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Analyse de l'arborescence : {args.input_dir} (Mode: {'Relatif' if args.relative else 'Absolu'})...")
    
    metrics_by_doc = parse_and_compute(args.input_dir, args.relative)
    
    if not metrics_by_doc:
        print("❌ Aucune donnée d'injection valide extraite. Vérifie que les fichiers contiennent bien mode='control' ou is_control=True pour établir la baseline.", file=sys.stderr)
        sys.exit(1)

    for doc_type, metrics in metrics_by_doc.items():
        plot_metrics_for_doc(doc_type, metrics, args.output_dir, args.relative)


if __name__ == "__main__":
    main()