#!/usr/bin/env python3
"""
Advanced sanity check: side-by-side comparison of CONTROL vs INJECTION outputs.

Goal:
  From a single results.jsonl file, generate comparisons CONTROL vs INJECTION:
    - CONTROL is restricted to doc_name="nothing" by default (baseline).
    - INJECTION is grouped by doc_name (each doc analyzed separately).
  The report(s) help inspect failures where CONTROL passed but INJECTION failed.

Output:
  Writes a .txt report showing both solutions next to each other (as sections),
  including doc_name, passed/control_passed flags and execution errors.

Usage:
  # Generate reports for all models (one file per model+doc_name):
  python3 src/perf_review/advanced_sanity_check.py results/run_xxx/results.jsonl

  # Restrict to one model:
  python3 src/perf_review/advanced_sanity_check.py results/run_xxx/results.jsonl --model codestral

  # Only compare injection docs starting with "explanation":
  python3 src/perf_review/advanced_sanity_check.py results/run_xxx/results.jsonl --doc-prefix explanation

  # Only keep cases where control passed but injection did NOT pass:
  python3 src/perf_review/advanced_sanity_check.py results/run_xxx/results.jsonl --only-mismatches
"""

import argparse
import json
import os
import sys
from collections import defaultdict


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
    """Sanitize a string to be used as a filename component."""
    s = (s or "").strip()
    if not s:
        return "unknown"
    # Keep letters, digits, dot, dash, underscore. Replace others with underscore.
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
) -> tuple[int, int]:
    """Return (pairs_written, candidates_seen)."""
    pairs_written = 0
    candidates_seen = 0

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Advanced sanity check — CONTROL vs INJECTION\n")
        f.write(f"- results_file: {results_path}\n")
        f.write(f"- model: {model}\n")
        f.write(f"- control_doc_name: {control_doc!r}\n")
        f.write(f"- injection_doc_name: {inj_doc!r}\n")
        f.write(f"- only_mismatches: {only_mismatches}\n")
        f.write("\n" + "=" * 90 + "\n\n")

        # deterministic order
        inj_entries_sorted = sorted(
            inj_entries, key=lambda e: (_problem_id(e) or 0, int(bool(e.get("passed", False))))
        )

        for inj in inj_entries_sorted:
            pid = _problem_id(inj)
            if pid is None:
                continue
            ctrl = control_by_pid.get(pid)
            if not ctrl:
                continue

            candidates_seen += 1
            if only_mismatches and inj.get("passed", False):
                continue
            if only_both_failed:
                # Keep only cases where injection fails AND the same code also fails in CONTROL mode.
                if inj.get("passed", False):
                    continue
                if inj.get("control_passed", False):
                    continue

            f.write(f"PROBLEM {pid}\n")
            f.write("-" * 90 + "\n")
            f.write(fmt_entry(ctrl, kind="CONTROL") + "\n")
            f.write("\n" + "-" * 90 + "\n")
            f.write(fmt_entry(inj, kind="INJECTION") + "\n")
            f.write("\n" + "=" * 90 + "\n\n")

            pairs_written += 1
            if max_pairs and pairs_written >= max_pairs:
                break

        f.write("\nSummary\n")
        f.write(f"- candidates_seen: {candidates_seen}\n")
        f.write(f"- pairs_written: {pairs_written}\n")

    return pairs_written, candidates_seen


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
        help="Optional: only keep injection entries whose doc_name starts with this prefix (e.g. 'explanation')",
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
        help="Only include pairs where injection failed AND control_passed is also false (passed=false and control_passed=false).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Optional max number of pairs to write (0 = no limit).",
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

    # Group entries by model
    models = sorted({m for m in (_model_name(e) for e in entries) if m})
    if args.model:
        models = [m for m in models if m == args.model]
    if not models:
        print("❌ No matching models found in results.", file=sys.stderr)
        raise SystemExit(1)

    total_reports = 0
    for model in models:
        model_entries = [e for e in entries if _model_name(e) == model]

        # Build baseline control map: pid -> first passing control entry for doc_name=control_doc
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
            print(f"⚠ No passing CONTROL entries for model={model!r} and control_doc={args.control_doc!r}. Skipping.")
            continue

        # Group injection entries by doc_name
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
            pairs_written, candidates_seen = _write_report_for_model_doc(
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
            )
            total_reports += 1
            print(
                f"✅ Wrote report: {out_path} (pairs_written={pairs_written}, candidates_seen={candidates_seen})"
            )

    if total_reports == 0:
        print("⚠ No reports generated (no matching injection docs or no passing control baseline).", file=sys.stderr)


if __name__ == "__main__":
    main()

