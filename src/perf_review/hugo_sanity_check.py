#!/usr/bin/env python3
"""
Advanced sanity check: side-by-side comparison of CONTROL vs INJECTION outputs.

Goal:
  From a single results.jsonl file, generate comparisons CONTROL vs INJECTION:
    - CONTROL is restricted to doc_name="nothing" by default (baseline).
    - INJECTION is grouped by doc_name (each doc analyzed separately).
  The report(s) help inspect failures where CONTROL passed but INJECTION failed.
  Works even if control entries are missing.

Output:
  Writes a .txt report showing both solutions next to each other (as sections),
  including doc_name, passed/control_passed flags and execution errors.
  Optionally plots difficulty vs frequency of parsed functions with an L2 linear regression fit.
"""

import argparse
import ast
import json
import os
import re
import sys
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    plt = None
    np = None


def load_results(path: str) -> list[dict]:
    entries: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "metadata" not in e or e["metadata"] is None:
                e["metadata"] = {}
            entries.append(e)
    return entries


def _problem_id(e: dict) -> int | None:
    meta = e.get("metadata") or {}
    pid = meta.get("problem_id", e.get("task_id"))
    try:
        return int(pid)
    except Exception:
        return None


def _short(s: str, limit: int = 220) -> str:
    s = (s or "").strip()
    if len(s) <= limit:
        return s
    return s[:limit] + " …"


def fmt_entry(e: dict, *, kind: str) -> str:
    meta = e.get("metadata") or {}
    pid = meta.get("problem_id", e.get("task_id"))
    pert = meta.get("perturbation_type", "Unknown")
    doc = meta.get("doc_name", "")
    passed = e.get("passed", False)
    cp = e.get("control_passed", None)
    err = e.get("error")

    lines = []
    lines.append(f"[{kind}] problem_id={pid} perturbation={pert} doc_name={doc!r}")
    lines.append(f"[{kind}] passed={passed} control_passed={cp} error={err!r}")
    if kind == "CONTROL":
        lines.append(f"[{kind}] stdout={_short(e.get('stdout', ''))!r}")
        lines.append(f"[{kind}] stderr={_short(e.get('stderr', ''))!r}")
    else:
        lines.append(f"[{kind}] stdout(injection)={_short(e.get('stdout', ''))!r}")
        lines.append(f"[{kind}] stderr(injection)={_short(e.get('stderr', ''))!r}")
        if "stdout_control" in e or "stderr_control" in e:
            lines.append(f"[{kind}] stdout_control={_short(e.get('stdout_control', ''))!r}")
            lines.append(f"[{kind}] stderr_control={_short(e.get('stderr_control', ''))!r}")

    code = (e.get("llm_code") or "").rstrip()
    lines.append(f"[{kind}] llm_code:\n{code if code else '<EMPTY>'}")
    return "\n".join(lines)


def _safe_filename(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "unknown"
    out = []
    for ch in s:
        if ch.isalnum() or ch in {".", "-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:120]


def _doc_name(e: dict) -> str:
    return ((e.get("metadata") or {}).get("doc_name") or "").strip()


def _model_name(e: dict) -> str:
    return ((e.get("metadata") or {}).get("model_name") or "").strip()


def _is_control_nothing(e: dict, control_doc: str) -> bool:
    if not e.get("is_control", False):
        return False
    dn = _doc_name(e)
    if control_doc == "nothing":
        return dn in ("", "nothing")
    return dn == control_doc


def _extract_functions_from_code(code: str) -> list[str]:
    if not code:
        return []
        
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]

    funcs = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        full_name = f"{node.func.value.id}.{node.func.attr}"
                        if full_name.startswith(("np.", "numpy.")):
                            funcs.append(full_name)
    except Exception:
        matches = re.findall(r'([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)?)\s*\(', code)
        funcs.extend([m for m in matches if m.startswith(("np.", "numpy."))])
        
    return funcs


def _is_confused(e: dict) -> bool:
    is_passed = e.get("passed", False)
    is_control_passed = e.get("control_passed", False)
    
    if not is_passed and not is_control_passed:
        stdout_inj = e.get("stdout", "") or ""
        stdout_ctrl = e.get("stdout_control", "") or ""
        target_err = "TEST_FAILED: Assertion incorrecte"
        
        if target_err in stdout_inj or target_err in stdout_ctrl:
            return True
    return False


def _write_report_for_model_doc(
    *,
    out_path: str,
    results_path: str,
    model: str,
    inj_doc: str,
    control_doc: str,
    control_by_pid: dict[int, dict],
    inj_entries: list[dict],
    only_mismatches: bool,
    only_both_failed: bool,
    max_pairs: int,
    count_confusion: bool,
) -> tuple[int, int, int]:
    pairs_written = 0
    candidates_seen = 0
    confusion_count = 0

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Advanced sanity check — CONTROL vs INJECTION\n")
        f.write(f"- results_file: {results_path}\n")
        f.write(f"- model: {model}\n")
        f.write(f"- control_doc_name: {control_doc!r}\n")
        f.write(f"- injection_doc_name: {inj_doc!r}\n")
        f.write(f"- only_mismatches: {only_mismatches}\n")
        if count_confusion:
            f.write(f"- count_confusion: True\n")
        f.write("\n" + "=" * 90 + "\n\n")

        inj_entries_sorted = sorted(
            inj_entries, key=lambda e: (_problem_id(e) or 0, int(bool(e.get("passed", False))))
        )

        for inj in inj_entries_sorted:
            pid = _problem_id(inj)
            if pid is None:
                continue
            ctrl = control_by_pid.get(pid)

            candidates_seen += 1

            if count_confusion and _is_confused(inj):
                confusion_count += 1

            if only_mismatches:
                if not ctrl or inj.get("passed", False):
                    continue
                    
            if only_both_failed:
                if inj.get("passed", False) or inj.get("control_passed", False):
                    continue

            f.write(f"PROBLEM {pid}\n")
            f.write("-" * 90 + "\n")
            
            if ctrl:
                f.write(fmt_entry(ctrl, kind="CONTROL") + "\n")
            else:
                f.write(f"[CONTROL] <AUCUN CONTRÔLE RÉUSSI TROUVÉ POUR LE PROBLÈME {pid}>\n")
                
            f.write("\n" + "-" * 90 + "\n")
            f.write(fmt_entry(inj, kind="INJECTION") + "\n")
            f.write("\n" + "=" * 90 + "\n\n")

            pairs_written += 1
            if max_pairs and pairs_written >= max_pairs:
                break

        f.write("\nSummary\n")
        f.write(f"- candidates_seen: {candidates_seen}\n")
        f.write(f"- pairs_written: {pairs_written}\n")
        if count_confusion:
            f.write(f"- confusion_count: {confusion_count}\n")

    return pairs_written, candidates_seen, confusion_count


def _generate_difficulty_plot(entries: list[dict], out_dir: str, bin_size: int = 0):
    if not plt or not np:
        print("⚠ matplotlib ou numpy non installé. Impossible de générer le graphique.", file=sys.stderr)
        return

    freq_counts = defaultdict(int)
    diff_counts = defaultdict(int)

    # Comptage strict sur les données d'injection uniquement
    for e in entries:
        if e.get("is_control", False):
            continue
            
        code = e.get("llm_code", "") or ""
        # set() garantit qu'une fonction répétée 10 fois dans le même code compte pour 1 "tentative"
        funcs = set(_extract_functions_from_code(code)) 
        
        is_passed = e.get("passed", False)
        is_confused = _is_confused(e)
        
        for f in funcs:
            freq_counts[f] += 1
            if not is_passed and not is_confused:
                diff_counts[f] += 1

    if not freq_counts:
        print("⚠ Aucune fonction numpy parsée dans le code LLM. Le graphique sera vide.", file=sys.stderr)
        return

    x_vals = []
    y_vals = []
    annotations = []

    # Liste des fonctions trouvées
    funcs = list(freq_counts.keys())

    if bin_size > 0:
        grouped_diffs = defaultdict(list)
        grouped_names = defaultdict(list)
        for f_name in funcs:
            freq = freq_counts[f_name]
            # Calcul du Taux de difficulté (%)
            difficulty_rate = (diff_counts[f_name] / freq) * 100
            
            b_start = (freq // bin_size) * bin_size
            grouped_diffs[b_start].append(difficulty_rate)
            grouped_names[b_start].append(f_name)
            
        b_starts = sorted(grouped_diffs.keys())
        x_vals = [b + bin_size / 2 for b in b_starts]
        y_vals = [np.mean(grouped_diffs[b]) for b in b_starts]
        
        for i, b in enumerate(b_starts):
            if len(grouped_names[b]) == 1:
                annotations.append((x_vals[i], y_vals[i], grouped_names[b][0]))
                
        xlabel_text = f"Fréquence d'apparition (Bacs de {bin_size})"
        ylabel_text = "Difficulté Moyenne Normalisée (%)"
        out_name = f"difficulty_vs_frequency_grouped_bin{bin_size}.png"
    else:
        for f_name in funcs:
            freq = freq_counts[f_name]
            # Calcul du Taux de difficulté (%)
            difficulty_rate = (diff_counts[f_name] / freq) * 100
            
            x_vals.append(freq)
            y_vals.append(difficulty_rate)
            
            # N'annoter que les fonctions qui posent un minimum de problème pour ne pas surcharger
            if difficulty_rate > 0:
                annotations.append((freq, difficulty_rate, f_name))
                
        xlabel_text = "Fréquence d'apparition dans les injections"
        ylabel_text = "Difficulté Normalisée (Taux d'échec en %)"
        out_name = "difficulty_vs_frequency_normalized.png"

    if len(x_vals) < 2:
        print("⚠ Pas assez de points pour calculer une régression.", file=sys.stderr)
        return

    x_array = np.array(x_vals)
    y_array = np.array(y_vals)

    # Régression Linéaire (L2 OLS)
    coeffs = np.polyfit(x_array, y_array, 1)
    poly_fn = np.poly1d(coeffs)
    y_pred = poly_fn(x_array)

    # Calcul du R^2
    ss_res = np.sum((y_array - y_pred) ** 2)
    ss_tot = np.sum((y_array - np.mean(y_array)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    plt.figure(figsize=(14, 8))
    
    # Rendre la taille des points proportionnelle à la fréquence peut aussi aider la lecture
    plt.scatter(x_array, y_array, alpha=0.7, edgecolors='k', label='Données observées')
    
    # Tracé de la ligne de régression
    x_line = np.linspace(min(x_array), max(x_array), 100)
    y_line = poly_fn(x_line)
    plt.plot(x_line, y_line, color='red', linestyle='--', 
             label=f'Ajustement L2 : y = {coeffs[0]:.3f}x + {coeffs[1]:.1f} ($R^2$ = {r_squared:.3f})')

    # Annotations des fonctions
    for x_pos, y_pos, txt in annotations:
        plt.annotate(txt, (x_pos, y_pos), xytext=(5, 5), textcoords='offset points', fontsize=8)

    plt.xlabel(xlabel_text, fontsize=11)
    plt.ylabel(ylabel_text, fontsize=11)
    plt.title("Difficulté de Perturbation selon la Fréquence d'Utilisation des Fonctions", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✅ Graphique de difficulté généré : {out_path} (R² = {r_squared:.4f})")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CONTROL vs INJECTION comparison reports from a single results.jsonl."
    )
    parser.add_argument("results_file", help="Path to results.jsonl")
    parser.add_argument(
        "--model",
        default=None,
        help="Optional: restrict to one model (exact match). If omitted, process all models found.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory (default: alongside results_file)",
    )
    parser.add_argument(
        "--doc-prefix",
        default=None,
        help="Optional: only keep injection entries whose doc_name starts with this prefix",
    )
    parser.add_argument(
        "--control-doc",
        default="nothing",
        help="Control doc_name to use as baseline (default: 'nothing' meaning doc_name in {'', 'nothing'}).",
    )
    parser.add_argument(
        "--only-mismatches",
        action="store_true",
        help="Only include pairs where control passed but injection did NOT pass.",
    )
    parser.add_argument(
        "--only-both-failed",
        action="store_true",
        help="Only include pairs where injection failed AND control_passed is also false.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Optional max number of pairs to write (0 = no limit).",
    )
    parser.add_argument(
        "--count-confusion",
        action="store_true",
        help="Counts occurrences where passed=False, control_passed=False, and stdout contains 'TEST_FAILED: Assertion incorrecte'.",
    )
    parser.add_argument(
        "--plot-difficulty",
        action="store_true",
        help="Generates a matplotlib scatter plot mapping perturbation difficulty against parsed function frequency.",
    )
    parser.add_argument(
        "--group-frequencies",
        type=int,
        default=0,
        metavar="BIN_SIZE",
        help="If set to > 0, groups functions into frequency bins and plots a scatter with L2 regression.",
    )
    parser.add_argument(
        "--print-responses",
        action="store_true",
        help="Prints all LLM responses (code) to the standard output.",
    )
    args = parser.parse_args()

    results_path = os.path.abspath(args.results_file)
    out_dir = args.output_dir or os.path.dirname(results_path)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    entries = load_results(results_path)
    if not entries:
        print("❌ No entries loaded.", file=sys.stderr)
        raise SystemExit(1)

    if args.print_responses:
        print("\n" + "=" * 80)
        print("DUMPING ALL LLM RESPONSES")
        print("=" * 80)
        for e in entries:
            pid = _problem_id(e)
            doc = _doc_name(e)
            kind = "CONTROL" if e.get("is_control") else "INJECTION"
            code = (e.get("llm_code") or "").rstrip()
            print(f"\n--- problem_id={pid} | type={kind} | doc_name={doc!r} ---")
            print(code if code else "<EMPTY>")
        print("=" * 80 + "\n")

    models = sorted({m for m in (_model_name(e) for e in entries) if m})
    if args.model:
        models = [m for m in models if m == args.model]
    if not models:
        print("❌ No matching models found in results.", file=sys.stderr)
        raise SystemExit(1)

    total_reports = 0
    global_confusion_counts = defaultdict(int)

    for model in models:
        model_entries = [e for e in entries if _model_name(e) == model]

        control_by_pid: dict[int, dict] = {}
        for e in model_entries:
            if not _is_control_nothing(e, args.control_doc):
                continue
            if not e.get("passed", False):
                continue
            pid = _problem_id(e)
            if pid is None:
                continue
            if pid not in control_by_pid:
                control_by_pid[pid] = e

        if not control_by_pid:
            print(f"⚠ Aucun contrôle réussi trouvé pour model={model!r} et control_doc={args.control_doc!r}. Génération des rapports d'injection de manière isolée.")

        inj_by_doc: dict[str, list[dict]] = defaultdict(list)
        for e in model_entries:
            if e.get("is_control", False):
                continue
            dn = _doc_name(e)
            if args.doc_prefix and not dn.startswith(args.doc_prefix):
                continue
            inj_by_doc[dn].append(e)

        for inj_doc, inj_entries in sorted(inj_by_doc.items(), key=lambda kv: kv[0]):
            if not inj_entries:
                continue
            out_name = f"advanced_sanity__model={_safe_filename(model)}__control={_safe_filename(args.control_doc)}__doc={_safe_filename(inj_doc)}.txt"
            out_path = os.path.join(out_dir, out_name)
            
            pairs_written, candidates_seen, conf_count = _write_report_for_model_doc(
                out_path=out_path,
                results_path=results_path,
                model=model,
                inj_doc=inj_doc,
                control_doc=args.control_doc,
                control_by_pid=control_by_pid,
                inj_entries=inj_entries,
                only_mismatches=args.only_mismatches,
                only_both_failed=args.only_both_failed,
                max_pairs=args.max_pairs,
                count_confusion=args.count_confusion,
            )
            total_reports += 1
            
            if args.count_confusion:
                if inj_doc in ("minimal", "ultra_minimal"):
                    global_confusion_counts[inj_doc] += conf_count
                else:
                    global_confusion_counts["other"] += conf_count

            msg = f"✅ Wrote report: {out_path} (pairs_written={pairs_written}, candidates_seen={candidates_seen})"
            if args.count_confusion:
                msg += f" [confusion_count={conf_count}]"
            print(msg)

    if args.plot_difficulty or args.group_frequencies > 0:
        _generate_difficulty_plot(entries, out_dir, bin_size=args.group_frequencies)

    if total_reports == 0:
        print("⚠ No reports generated (no matching injection docs).", file=sys.stderr)
        return

    if args.count_confusion:
        print("\n" + "=" * 40)
        print(" GLOBAL CONFUSION SUMMARY ")
        print("=" * 40)
        print(f"ultra_minimal : {global_confusion_counts['ultra_minimal']}")
        print(f"minimal       : {global_confusion_counts['minimal']}")
        
        if global_confusion_counts["other"] > 0:
            print(f"other         : {global_confusion_counts['other']}")
            
        total_confusion = sum(global_confusion_counts.values())
        print("-" * 40)
        print(f"TOTAL         : {total_confusion}")
        print("=" * 40 + "\n")


if __name__ == "__main__":
    main()