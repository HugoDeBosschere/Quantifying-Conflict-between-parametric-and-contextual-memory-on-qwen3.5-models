#!/usr/bin/env python3
"""
Script de visualisation des résultats d'évaluation (fichiers .jsonl).

Deux métriques distinctes :
- run_control (is_control=True) : le LLM reçoit problème + lib d'origine ; on évalue avec la lib d'origine.
  → Control classique / Control minimal / Control ultra_minimal (selon doc_name), via le flag passed.
- control_passed (lignes injection) : le LLM a reçu problème + doc modifiés ; on évalue la même réponse
  avec la lib d'origine. → "Doc minimal (éval. lib d'origine)" / "Doc ultra_minimal (éval. lib d'origine)".

Doc minimal / Doc ultra_minimal (sans suffixe) = injection, évaluation avec la lib modifiée (passed).
Supporte l'assignation manuelle de la perturbation par fichier/dossier via les arguments CLI.
Exige que la première ligne du JSONL définisse la perturbation (ex: {"perturbation_json": "capitalize"}).
"""
import argparse
import itertools
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    meta = res.get("metadata", {})
    doc_name = (meta.get("doc_name") or "").strip()
    is_control = res.get("is_control", False)

    if is_control and doc_name in ("", "nothing", None):
        return "Control classique"
    if is_control and doc_name == "minimal":
        return "Control minimal"
    if is_control and doc_name == "ultra_minimal":
        return "Control ultra_minimal"

    if not is_control and doc_name == "minimal":
        return "Doc minimal"
    if not is_control and doc_name == "ultra_minimal":
        return "Doc ultra_minimal"
    if not is_control and doc_name.startswith("explanation"):
        return "Doc explanation"

    prefix = "Control" if is_control else "Doc"
    name = doc_name or "other"
    return f"{prefix} {name}"


def _get_base_doc(label: str) -> str:
    if "classique" in label: return "classique"
    if "ultra_minimal" in label: return "ultra_minimal"
    if "minimal" in label: return "minimal"
    if "explanation" in label: return "explanation"
    return "other"


def load_data(file_pert_pairs: list[tuple[str, str | None]], group_by: str, exclude_inj: bool, exclude_ctrl: bool) -> dict:
    raw_records = []
    excluded_pids = set()

    # Passe 1 : Chargement en mémoire et identification des problèmes "neutres"
    for filepath, pert_override in file_pert_pairs:
        print(f"Chargement de {filepath} " + (f"[Perturbation forcée CLI: {pert_override}]" if pert_override else ""))
        if not os.path.isfile(filepath):
            print(f"❌ Fichier introuvable : {filepath}")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if not first_line:
                raise ValueError(f"❌ Erreur : Le fichier {filepath} est vide.")

            try:
                first_obj = json.loads(first_line)
            except json.JSONDecodeError as e:
                raise ValueError(f"❌ Erreur : Première ligne JSON invalide. Détail : {e}")

            if "perturbation_json" in first_obj:
                # Ligne d'en-tête explicite
                file_pert = first_obj["perturbation_json"]
                header_consumed = True
            elif pert_override is not None:
                # Pas d'en-tête mais perturbation fournie via CLI → première ligne = données
                file_pert = pert_override
                header_consumed = False
            else:
                raise ValueError(
                    f"❌ Erreur : La première ligne de {filepath} ne contient pas 'perturbation_json' "
                    f"et aucune perturbation n'a été fournie via CLI."
                )

            pert_resolved = pert_override if pert_override is not None else file_pert

            lines_to_process = f if header_consumed else itertools.chain([first_line], f)
            for line in lines_to_process:
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
                    continue
                
                res["_assigned_pert"] = pert_resolved
                raw_records.append(res)

                # Identification des cas d'injection réussissant les deux contextes
                is_control = res.get("is_control", False)
                if not is_control:
                    if res.get("passed", False) and res.get("control_passed", False):
                        pid = meta.get("problem_id", res.get("task_id"))
                        if pid is not None:
                            excluded_pids.add((model, pert_resolved, pid))

    # Passe 2 : Filtrage et agrégation
    data = {}
    for res in raw_records:
        meta = res.get("metadata", {})
        model = meta.get("model_name", "")
        pert = res["_assigned_pert"]
        pid = meta.get("problem_id", res.get("task_id"))
        is_control = res.get("is_control", False)

        # Application des règles d'exclusion
        if (model, pert, pid) in excluded_pids:
            if not is_control and exclude_inj:
                continue
            if is_control and exclude_ctrl:
                continue

        passed = 1 if res.get("passed", False) else 0
        doc_type_cond = _doc_type_label(res)
        base_doc = _get_base_doc(doc_type_cond)

        if group_by == "model":
            primary, secondary = model, pert
        elif group_by == "perturbation":
            primary, secondary = pert, model
        elif group_by == "doc_type":
            primary, secondary = base_doc, model
        else:
            primary, secondary = model, pert

        def add_record(prim, cond, sec, success_val):
            if prim not in data:
                data[prim] = {}
            if cond not in data[prim]:
                data[prim][cond] = {}
            if sec not in data[prim][cond]:
                data[prim][cond][sec] = {"success": 0, "total": 0}
            data[prim][cond][sec]["success"] += success_val
            data[prim][cond][sec]["total"] += 1

        add_record(primary, doc_type_cond, secondary, passed)

        if doc_type_cond in ["Doc minimal", "Doc ultra_minimal", "Doc explanation"]:
            eval_orig_label = f"{doc_type_cond} (éval. lib d'origine)"
            cp = 1 if res.get("control_passed", False) else 0
            add_record(primary, eval_orig_label, secondary, cp)

    print(f"   Groupes principaux détectés ({group_by}): {list(data.keys())}")
    if exclude_inj or exclude_ctrl:
        print(f"   Filtres actifs : exclusions liées aux {len(excluded_pids)} entrées d'injection totalement réussies.")
    
    return data


def plot_global_all_conditions(data: dict, output_dir: str, group_by: str) -> None:
    groups = sorted(data.keys())
    if not groups:
        return

    series_keys = set()
    for g in groups:
        for doc_type, secs in data[g].items():
            if doc_type.startswith("Control"):
                series_keys.add((doc_type, "all"))
            else:
                for sec in secs:
                    if secs[sec]["total"] > 0:
                        series_keys.add((doc_type, sec))

    if not series_keys:
        print("⚠ Aucune série avec données, plot global ignoré.")
        return

    def _sort_key(item):
        doc_type, sec = item
        score = 0
        if doc_type.startswith("Control"): score = 0
        elif "(éval. lib d'origine)" not in doc_type: score = 10
        else: score = 20
        return (score, doc_type, str(sec))

    active_series = sorted(list(series_keys), key=_sort_key)

    n_series = len(active_series)
    n_groups = len(groups)
    bar_width = 0.5
    bar_gap = 0.12
    group_gap = 1.5
    step = bar_width + bar_gap
    group_width = n_series * step - bar_gap
    group_centers = [i * (group_width + group_gap) for i in range(n_groups)]
    
    x_positions = []
    all_scores = []
    all_labels_text = []
    xtick_pos = []
    xtick_labels = []

    def get_bar_style(doc_type, sec_key):
        base = {"color": "steelblue", "alpha": 0.9, "hatch": "", "edgecolor": "black"}
        
        if doc_type.startswith("Control"):
            base["color"] = "gray"
            if "minimal" in doc_type and "ultra" not in doc_type: base["hatch"] = "\\\\"
            elif "ultra_minimal" in doc_type: base["hatch"] = "xx"
            else: base["hatch"] = "//"
            return base
            
        if "(éval. lib d'origine)" in doc_type:
            base["hatch"] = ".."
            base["alpha"] = 0.6
            
        if group_by == "model":
            sec_str = str(sec_key).lower()
            if "v2" in sec_str: base["color"] = "darkorange"
            elif "capitalize" in sec_str: base["color"] = "crimson"
            elif "underscore" in sec_str: base["color"] = "forestgreen"
        return base

    for g_idx, g in enumerate(groups):
        base = group_centers[g_idx]
        for i, (doc_type, sec) in enumerate(active_series):
            pos = base + i * step
            x_positions.append(pos)
            
            s, t = 0, 0
            if doc_type.startswith("Control"):
                if doc_type in data[g]:
                    s = sum(data[g][doc_type][k]["success"] for k in data[g][doc_type])
                    t = sum(data[g][doc_type][k]["total"] for k in data[g][doc_type])
            else:
                if doc_type in data[g] and sec in data[g][doc_type]:
                    st = data[g][doc_type][sec]
                    s, t = st["success"], st["total"]

            if t > 0:
                pct = s / t * 100
                all_scores.append(pct)
                prefix = f"{sec}\n" if not doc_type.startswith("Control") and group_by == "model" else ""
                all_labels_text.append(f"{prefix}{pct:.0f}%\n({s}/{t})")
            else:
                all_scores.append(0)
                all_labels_text.append("")
                
            xtick_pos.append(pos)
            xtick_labels.append(doc_type)

    x_positions = np.array(x_positions)
    all_scores = np.array(all_scores)

    fig, ax = plt.subplots(figsize=(max(14, n_groups * 6), 8))

    for i in range(len(x_positions)):
        idx_series = i % n_series
        doc_type, sec = active_series[idx_series]
        style = get_bar_style(doc_type, sec)
        
        rects = ax.bar(
            x_positions[i],
            all_scores[i],
            bar_width,
            color=style["color"],
            alpha=style["alpha"],
            hatch=style["hatch"],
            edgecolor=style["edgecolor"],
        )
        ax.bar_label(rects, labels=[all_labels_text[i]], padding=2, fontsize=7.5, fontweight="bold")

    ax.set_ylabel("Taux de succès (%)")
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 120)
    
    legend_handles = [
        mpatches.Patch(facecolor='gray', hatch='//', edgecolor='black', label='Control Conditions'),
    ]
    if group_by == "model":
        legend_handles.extend([
            mpatches.Patch(facecolor='darkorange', edgecolor='black', label='Doc: perturbation = v2'),
            mpatches.Patch(facecolor='crimson', edgecolor='black', label='Doc: perturbation = capitalize'),
            mpatches.Patch(facecolor='forestgreen', edgecolor='black', label='Doc: perturbation = underscore'),
            mpatches.Patch(facecolor='white', hatch='..', edgecolor='black', label="Injection (éval. lib d'origine)")
        ])
    ax.legend(handles=legend_handles, title="Type de Condition & Perturbation", loc='upper right', bbox_to_anchor=(0.99, 0.95), framealpha=0.9)
    
    top_y = ax.get_ylim()[1] - 3
    for g_idx, g in enumerate(groups):
        center_x = group_centers[g_idx] + (n_series * step - bar_gap) / 2 - bar_width / 2
        ax.text(center_x, top_y, str(g), fontsize=10, fontweight="bold", ha="center", va="top")
        
    ax.set_title(f"Performance globale par {group_by} — run_control vs injection", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out = os.path.join(output_dir, f"plot_global_{group_by}_all.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"✅ Sauvegardé : {out}")


def _secondary_data_one(data: dict, primary_keys: list, *doc_labels: str) -> set:
    all_sec = set()
    for p in primary_keys:
        for d in doc_labels:
            if d in data[p]:
                all_sec.update(data[p][d].keys())
    return all_sec


def _row_has_data(data: dict, primary_keys: list, control_label: str | None, doc_label: str, doc_orig_label: str) -> bool:
    labels = [doc_label, doc_orig_label]
    if control_label is not None:
        labels.append(control_label)
    for p in primary_keys:
        for label in labels:
            if label in data[p] and any(st["total"] > 0 for st in data[p][label].values()):
                return True
    return False


DOC_ROWS_CONFIG = [
    ("Control minimal", "Doc minimal", "Doc minimal (éval. lib d'origine)", "Minimal"),
    ("Control ultra_minimal", "Doc ultra_minimal", "Doc ultra_minimal (éval. lib d'origine)", "Ultra_minimal"),
    (None, "Doc explanation", "Doc explanation (éval. lib d'origine)", "Explanation"),
]


def plot_detailed_breakdown(data: dict, output_dir: str, group_by: str) -> None:
    primary_keys = sorted(data.keys())
    if not primary_keys:
        return

    rows_config = [
        (c_lbl, d_lbl, do_lbl, r_title)
        for c_lbl, d_lbl, do_lbl, r_title in DOC_ROWS_CONFIG
        if _row_has_data(data, primary_keys, c_lbl, d_lbl, do_lbl)
    ]
    if not rows_config:
        return

    all_sec = set()
    for c_lbl, d_lbl, do_lbl, _t in rows_config:
        labels = [d_lbl, do_lbl]
        if c_lbl is not None: labels.append(c_lbl)
        all_sec |= _secondary_data_one(data, primary_keys, *labels)
        
    secondary_keys = sorted(all_sec)
    if not secondary_keys:
        return

    sec_legend_title = "Perturbation" if group_by == "model" else "Modèle" if group_by == "perturbation" else "Modèle/Perturbation"

    try:
        cmap = plt.colormaps.get_cmap('tab10')
    except AttributeError:
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(secondary_keys))]

    n_rows = len(rows_config)
    n_cols = len(primary_keys)
    fig, axes = plt.subplots(nrows=n_rows, ncols=max(1, n_cols), figsize=(max(14, 4 * n_cols), 4 * n_rows))
    
    if n_cols == 1 and n_rows == 1: axes = np.array([[axes]])
    elif n_cols == 1: axes = axes.reshape(-1, 1)
    elif n_rows == 1: axes = axes.reshape(1, -1)

    for row, (control_label, doc_label, doc_orig_label, row_title) in enumerate(rows_config):
        active_lbls = []
        display_lbls = []
        if control_label is not None:
            active_lbls.append(control_label)
            display_lbls.append("Control")
        active_lbls.append(doc_label)
        display_lbls.append("Injection")
        active_lbls.append(doc_orig_label)
        display_lbls.append("Injection\n(Eval Orig)")
        
        x = np.arange(len(active_lbls))
        n_sec = len(secondary_keys)
        width = 0.8 / max(1, n_sec)

        for col, p_key in enumerate(primary_keys):
            ax = axes[row, col]
            
            for i, sec in enumerate(secondary_keys):
                pos = x - 0.4 + width/2 + i*width
                y_vals = []
                texts = []
                
                for lbl in active_lbls:
                    if lbl in data[p_key] and sec in data[p_key][lbl]:
                        st = data[p_key][lbl][sec]
                        s, t = st["success"], st["total"]
                        pct = (s / t * 100) if t else 0
                        y_vals.append(pct)
                        texts.append(f"{pct:.0f}%\n({s}/{t})" if t else "")
                    else:
                        y_vals.append(0)
                        texts.append("")
                        
                rects = ax.bar(pos, y_vals, width, label=str(sec), color=colors[i], alpha=0.9, edgecolor='black', linewidth=0.6)
                ax.bar_label(rects, labels=[t if v > 0 else "" for t, v in zip(texts, y_vals)], padding=1, fontsize=6)

            ax.set_title(str(p_key), fontsize=10, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(display_lbls, rotation=0, fontsize=9)
            ax.set_ylabel("Précision (%)")
            ax.set_ylim(0, 115)
            
            if col == 0:
                ax.legend(title=sec_legend_title, loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize="x-small", title_fontsize="small")
            
            ax.grid(axis="y", linestyle=":", alpha=0.5)

    plt.suptitle(f"Détail par {group_by} (Légende : {sec_legend_title}) — run_control vs injection", fontsize=12, fontweight="bold", y=1.002)
    plt.tight_layout()
    out = os.path.join(output_dir, f"plot_detailed_{group_by}_breakdown.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"✅ Sauvegardé : {out}")


def main():
    repo = find_repo_root()
    default_output = os.path.join(repo, "results")

    parser = argparse.ArgumentParser(
        description="Plot des résultats : supporte des fichiers JSONL, assignation manuelle de perturbation, et filtrage croisé."
    )
    parser.add_argument(
        "input_args",
        nargs="+",
        help="Fichiers ou dossiers (scan de .jsonl), suivis optionnellement de leur type de perturbation (ex: folder1 capitalize file.jsonl v2)",
    )
    parser.add_argument(
        "--group-by",
        choices=["model", "perturbation", "doc_type"],
        default="model",
        help="Axe principal d'agrégation (défaut: model).",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Répertoire de sortie des graphiques",
    )
    # Nouvelles options de filtrage
    parser.add_argument(
        "--exclude-neutral-inj",
        action="store_true",
        help="Ne comptabilise pas les lignes d'injection où passed=True ET control_passed=True.",
    )
    parser.add_argument(
        "--exclude-neutral-ctrl",
        action="store_true",
        help="Ne comptabilise pas les lignes de contrôle correspondant aux problem_id exclus par --exclude-neutral-inj.",
    )
    args = parser.parse_args()

    file_pert_pairs = []
    i = 0
    while i < len(args.input_args):
        item = args.input_args[i]
        pert_override = None

        if ":" in item and not os.path.exists(item):
            parts = item.rsplit(":", 1)
            item = parts[0]
            pert_override = parts[1]
            i += 1
        else:
            if i + 1 < len(args.input_args):
                next_item = args.input_args[i + 1]
                if not next_item.endswith(".jsonl") and not os.path.exists(next_item):
                    pert_override = next_item
                    i += 2
                else:
                    i += 1
            else:
                i += 1

        if os.path.isdir(item):
            for root, _, files in os.walk(item):
                for file in files:
                    if file.endswith(".jsonl"):
                        file_pert_pairs.append((os.path.join(root, file), pert_override))
        elif os.path.isfile(item):
            file_pert_pairs.append((item, pert_override))
        else:
            print(f"⚠ Ignoré : '{item}' n'est ni un fichier ni un dossier valide.")

    if not file_pert_pairs:
        print("❌ Aucun fichier .jsonl trouvé ou fourni.")
        sys.exit(1)

    first_file = os.path.abspath(file_pert_pairs[0][0])
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else (os.path.dirname(first_file) or default_output)
    os.makedirs(output_dir, exist_ok=True)

    data = load_data(file_pert_pairs, args.group_by, args.exclude_neutral_inj, args.exclude_neutral_ctrl)
    if not data:
        print("❌ Aucune donnée valide chargée.")
        sys.exit(1)

    plot_global_all_conditions(data, output_dir, args.group_by)
    plot_detailed_breakdown(data, output_dir, args.group_by)


if __name__ == "__main__":
    main()