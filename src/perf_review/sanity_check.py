"""
Sanity check on results.jsonl — detects anomalies, produces statistics,
and writes a human-readable report.

Usage (from project root):
    python3 src/perf_review/sanity_check.py results/run_xxx/results.jsonl
    python3 src/perf_review/sanity_check.py results/run_xxx/results.jsonl -o report.txt
    python3 src/perf_review/sanity_check.py results/run_xxx/results.jsonl results/run_yyy/results.jsonl
"""

import json
import re
import argparse
import os
import sys
from collections import defaultdict, Counter
from datetime import datetime


# ============================================================
# 1. PARSING
# ============================================================

def load_results(path):
    """Load all entries from a results.jsonl file. Normalise chaque entrée avec metadata au moins {}."""
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if "metadata" not in e:
                e["metadata"] = {}
            entries.append(e)
    return entries


def extract_np_calls(code):
    """Extract all np.xxx calls from code. Returns a set of function names."""
    if not code:
        return set()
    return set(re.findall(r'(?:np|numpy)\.(\w+)', code))


def uses_only_methods(code):
    """Check if code uses zero np.xxx / numpy.xxx module-level calls."""
    if not code:
        return True
    return len(re.findall(r'(?:np|numpy)\.(\w+)', code)) == 0


# ============================================================
# 2. ANOMALY DETECTORS
# ============================================================

def detect_both_passed(entries):
    """
    Injection entries where passed=True AND control_passed=True.
    The LLM answer works both with wrapper and real numpy,
    meaning it likely bypassed the perturbation.
    """
    hits = []
    for e in entries:
        if e.get('is_control'):
            continue
        if e.get('passed') and e.get('control_passed'):
            np_calls = extract_np_calls(e.get('llm_code', ''))
            hits.append({
                'task_id': e['task_id'],
                'model': e['metadata'].get('model_name', '?'),
                'doc': e['metadata'].get('doc_name', '?'),
                'np_calls': sorted(np_calls),
                'uses_no_np': len(np_calls) == 0,
                'code': e.get('llm_code', ''),
            })
    return hits


def detect_no_np_passed(entries):
    """
    Entries where passed=True but the code uses no np.xxx calls.
    These pass "through the cracks" regardless of the wrapper.
    """
    hits = []
    for e in entries:
        if not e.get('passed'):
            continue
        code = e.get('llm_code', '')
        if uses_only_methods(code):
            hits.append({
                'task_id': e['task_id'],
                'model': e['metadata'].get('model_name', '?'),
                'doc': e['metadata'].get('doc_name', '?'),
                'mode': e['metadata'].get('mode', '?'),
                'is_control': e.get('is_control', False),
                'code': code,
            })
    return hits


def detect_injection_pass_control_fail(entries):
    """
    Injection entries where passed=True but control_passed=False.
    The code works with the wrapper but not real numpy — suspicious.
    """
    hits = []
    for e in entries:
        if e.get('is_control'):
            continue
        if e.get('passed') and not e.get('control_passed'):
            hits.append({
                'task_id': e['task_id'],
                'model': e['metadata'].get('model_name', '?'),
                'doc': e['metadata'].get('doc_name', '?'),
                'code': e.get('llm_code', ''),
                'np_calls': sorted(extract_np_calls(e.get('llm_code', ''))),
            })
    return hits


def detect_perturbation_adoption(entries):
    """
    For injection entries, check if the LLM actually adopted the perturbation
    in its code (e.g. used _v2 suffix, _ suffix, or capitalized names).
    """
    results = []
    for e in entries:
        if e.get('is_control'):
            continue
        code = e.get('llm_code', '')
        if not code:
            continue

        has_v2 = bool(re.search(r'np\.\w+_v2', code))
        has_underscore = bool(re.search(r'np\.\w+_(?!\w)', code))
        has_capitalize = bool(re.search(r'np\.[A-Z]\w+', code))
        has_normal_np = bool(re.search(r'np\.[a-z]\w+(?<!_v2)(?<!_)\b', code))

        results.append({
            'task_id': e['task_id'],
            'model': e['metadata'].get('model_name', '?'),
            'doc': e['metadata'].get('doc_name', '?'),
            'passed': e.get('passed', False),
            'has_v2': has_v2,
            'has_underscore': has_underscore,
            'has_capitalize': has_capitalize,
            'has_normal_np': has_normal_np,
            'uses_no_np': uses_only_methods(code),
        })
    return results


# ============================================================
# 3. STATISTICS
# ============================================================

def compute_global_stats(entries):
    """Overall pass rates split by mode."""
    stats = defaultdict(lambda: {'total': 0, 'passed': 0})
    for e in entries:
        mode = e['metadata'].get('mode', 'unknown')
        stats[mode]['total'] += 1
        if e.get('passed'):
            stats[mode]['passed'] += 1
    return dict(stats)


def compute_stats_by_model_doc(entries):
    """Pass rates broken down by (model, doc_name, mode)."""
    stats = defaultdict(lambda: {'total': 0, 'passed': 0})
    for e in entries:
        key = (
            e['metadata'].get('model_name', '?'),
            e['metadata'].get('doc_name', '?'),
            e['metadata'].get('mode', '?'),
        )
        stats[key]['total'] += 1
        if e.get('passed'):
            stats[key]['passed'] += 1
    return dict(stats)


def compute_np_function_usage(entries, only_passed=False):
    """Count which np.xxx functions are used across all entries."""
    counter = Counter()
    for e in entries:
        if only_passed and not e.get('passed'):
            continue
        np_calls = extract_np_calls(e.get('llm_code', ''))
        for func in np_calls:
            counter[func] += 1
    return counter


def detect_unknown_model_and_api_failures(entries):
    """
    Détecte les lignes à problème de qualité :
    - model_name manquant ou "Unknown"
    - error == "LLM_API_FAILURE"
    Retourne (unknown_model_entries, api_failure_entries).
    """
    unknown_model = []
    api_failure = []
    for e in entries:
        meta = e.get("metadata") or {}
        model = meta.get("model_name")
        if model is None or (isinstance(model, str) and (not model.strip() or model.strip().lower() == "unknown")):
            unknown_model.append({
                "task_id": e.get("task_id"),
                "error": e.get("error"),
                "has_metadata": bool(meta),
            })
        if e.get("error") == "LLM_API_FAILURE":
            api_failure.append({
                "task_id": e.get("task_id"),
                "model": meta.get("model_name", "?"),
                "doc": meta.get("doc_name", "?"),
                "mode": meta.get("mode", "?"),
            })
    return unknown_model, api_failure


def compute_error_categories(entries):
    """Categorize errors from failed entries by (model, doc, mode)."""
    cats = defaultdict(lambda: Counter())
    for e in entries:
        if e.get('passed'):
            continue
        meta = e.get('metadata') or {}
        key = (
            meta.get('model_name', '?'),
            meta.get('doc_name', '?'),
            meta.get('mode', '?'),
        )
        if e.get('error') == 'LLM_API_FAILURE':
            cats[key]['llm_api_failure'] += 1
            continue
        stdout = e.get('stdout', '')
        stderr = e.get('stderr', '')
        combined = stdout + stderr

        if 'has no attribute' in combined:
            cats[key]['attribute_error'] += 1
        elif 'SyntaxError' in combined:
            cats[key]['syntax_error'] += 1
        elif 'TypeError' in combined:
            cats[key]['type_error'] += 1
        elif 'ValueError' in combined:
            cats[key]['value_error'] += 1
        elif 'IndexError' in combined:
            cats[key]['index_error'] += 1
        elif 'TEST_FAILED' in combined:
            cats[key]['wrong_answer'] += 1
        elif 'TIMEOUT' in combined:
            cats[key]['timeout'] += 1
        else:
            cats[key]['other'] += 1
    return dict(cats)


# ============================================================
# 4. REPORT WRITER
# ============================================================

def write_report(entries, output_path, source_path):
    """Write a complete sanity check report."""
    lines = []

    def w(text=""):
        lines.append(text)

    def section(title):
        w()
        w("=" * 70)
        w(f"  {title}")
        w("=" * 70)
        w()

    def subsection(title):
        w()
        w(f"--- {title} ---")
        w()

    # Header
    w("SANITY CHECK REPORT")
    w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"Source:    {source_path}")
    w(f"Entries:   {len(entries)}")
    w()

    n_control = sum(1 for e in entries if e.get('is_control'))
    n_injection = len(entries) - n_control
    w(f"  Control entries:   {n_control}")
    w(f"  Injection entries: {n_injection}")

    # ---- 0. DATA QUALITY: Unknown model / LLM_API_FAILURE ----
    section("0. DATA QUALITY — Unknown model & LLM_API_FAILURE")

    unknown_model_entries, api_failure_entries = detect_unknown_model_and_api_failures(entries)
    w(f"  Entries with missing or 'Unknown' model_name: {len(unknown_model_entries)}")
    if unknown_model_entries:
        subsection("Détail (task_id, error)")
        for u in unknown_model_entries[:30]:
            w(f"    task_id={u['task_id']}  error={u.get('error', '')!r}  has_metadata={u['has_metadata']}")
        if len(unknown_model_entries) > 30:
            w(f"    ... et {len(unknown_model_entries) - 30} autres")
    w()
    w(f"  Entries with error LLM_API_FAILURE: {len(api_failure_entries)}")
    if api_failure_entries:
        subsection("Répartition par (model, doc, mode)")
        by_key = defaultdict(int)
        for a in api_failure_entries:
            by_key[(a['model'], a['doc'], a['mode'])] += 1
        for (model, doc, mode), count in sorted(by_key.items()):
            w(f"    {model:25s} | {doc:15s} | {mode:10s} : {count}")
        subsection("Exemples (task_id, model, doc) — premiers 20")
        for a in api_failure_entries[:20]:
            w(f"    task_id={a['task_id']}  model={a['model']}  doc={a['doc']}  mode={a['mode']}")
        if len(api_failure_entries) > 20:
            w(f"    ... et {len(api_failure_entries) - 20} autres")

    # ---- 1. GLOBAL STATS ----
    section("1. GLOBAL PASS RATES")

    global_stats = compute_global_stats(entries)
    for mode, s in sorted(global_stats.items()):
        rate = s['passed'] / s['total'] * 100 if s['total'] else 0
        w(f"  {mode:12s}: {s['passed']:4d}/{s['total']:4d}  ({rate:5.1f}%)")

    # ---- 2. PER MODEL/DOC STATS ----
    section("2. PASS RATES BY MODEL / DOC / MODE")

    by_model_doc = compute_stats_by_model_doc(entries)
    current_model = None
    for (model, doc, mode), s in sorted(by_model_doc.items()):
        if model != current_model:
            w()
            w(f"  [{model}]")
            current_model = model
        rate = s['passed'] / s['total'] * 100 if s['total'] else 0
        w(f"    {mode:10s} | {doc:15s} : {s['passed']:4d}/{s['total']:4d}  ({rate:5.1f}%)")

    # ---- 3. ANOMALY: both passed ----
    section("3. ANOMALY: passed=True AND control_passed=True (injection)")

    both = detect_both_passed(entries)
    both_with_np = [h for h in both if not h['uses_no_np']]
    both_without_np = [h for h in both if h['uses_no_np']]

    w(f"  Total: {len(both)} entries pass both wrapper and real numpy")
    w(f"    - Using np.xxx calls (might be method-only or compatible): {len(both_with_np)}")
    w(f"    - Using NO np.xxx calls (pure array methods/python):       {len(both_without_np)}")

    subsection("Breakdown by model / doc")
    by_key = defaultdict(int)
    for h in both:
        by_key[(h['model'], h['doc'])] += 1
    for (model, doc), count in sorted(by_key.items()):
        w(f"    {model:25s} | {doc:15s} : {count}")

    if both_without_np:
        subsection("Examples: passed both, no np.xxx calls (first 10)")
        for h in both_without_np[:10]:
            w(f"    task={h['task_id']}, model={h['model']}, doc={h['doc']}")
            w(f"      code: {h['code'][:120]}")
            w()

    if both_with_np:
        subsection("Examples: passed both, WITH np.xxx calls (first 10)")
        for h in both_with_np[:10]:
            w(f"    task={h['task_id']}, model={h['model']}, doc={h['doc']}")
            w(f"      np calls: {h['np_calls']}")
            w(f"      code: {h['code'][:120]}")
            w()

    # ---- 4. ANOMALY: no np calls but passed ----
    section("4. PASSED WITHOUT ANY np.xxx CALL (all modes)")

    no_np = detect_no_np_passed(entries)
    w(f"  Total: {len(no_np)} entries passed without np.xxx calls")

    subsection("Breakdown by mode / model / doc")
    by_key = defaultdict(int)
    for h in no_np:
        by_key[(h['mode'], h['model'], h['doc'])] += 1
    for (mode, model, doc), count in sorted(by_key.items()):
        w(f"    {mode:10s} | {model:25s} | {doc:15s} : {count}")

    subsection("Unique task_ids that never use np (across all entries)")
    task_counts = Counter(h['task_id'] for h in no_np)
    frequent = task_counts.most_common(20)
    for tid, cnt in frequent:
        w(f"    task {tid}: {cnt} times")

    # ---- 5. ANOMALY: injection pass, control fail ----
    section("5. ANOMALY: INJECTION passed=True, control_passed=False")

    inj_pass = detect_injection_pass_control_fail(entries)
    w(f"  Total: {len(inj_pass)} entries")
    if inj_pass:
        subsection("Breakdown by model / doc")
        by_key = defaultdict(int)
        for h in inj_pass:
            by_key[(h['model'], h['doc'])] += 1
        for (model, doc), count in sorted(by_key.items()):
            w(f"    {model:25s} | {doc:15s} : {count}")

        subsection("Examples (first 5)")
        for h in inj_pass[:5]:
            w(f"    task={h['task_id']}, model={h['model']}, doc={h['doc']}")
            w(f"      np calls: {h['np_calls']}")
            w(f"      code: {h['code'][:120]}")
            w()

    # ---- 6. PERTURBATION ADOPTION ----
    section("6. PERTURBATION ADOPTION IN INJECTION CODE")

    adoption = detect_perturbation_adoption(entries)
    if adoption:
        subsection("By model / doc: how often does the LLM use perturbation patterns?")
        by_key = defaultdict(lambda: {
            'total': 0, 'v2': 0, 'underscore': 0,
            'capitalize': 0, 'normal_np': 0, 'no_np': 0,
        })
        for a in adoption:
            k = (a['model'], a['doc'])
            by_key[k]['total'] += 1
            if a['has_v2']:
                by_key[k]['v2'] += 1
            if a['has_underscore']:
                by_key[k]['underscore'] += 1
            if a['has_capitalize']:
                by_key[k]['capitalize'] += 1
            if a['has_normal_np']:
                by_key[k]['normal_np'] += 1
            if a['uses_no_np']:
                by_key[k]['no_np'] += 1

        w(f"  {'model':25s} | {'doc':15s} | {'total':>5s} | {'_v2':>4s} | {'_':>4s} | {'Cap':>4s} | {'norm':>4s} | {'no_np':>5s}")
        w(f"  {'-'*25}-+-{'-'*15}-+-{'-'*5}-+-{'-'*4}-+-{'-'*4}-+-{'-'*4}-+-{'-'*4}-+-{'-'*5}")
        for (model, doc), s in sorted(by_key.items()):
            w(f"  {model:25s} | {doc:15s} | {s['total']:5d} | {s['v2']:4d} | {s['underscore']:4d} | {s['capitalize']:4d} | {s['normal_np']:4d} | {s['no_np']:5d}")

    # ---- 7. ERROR CATEGORIES ----
    section("7. ERROR CATEGORIES (failed entries)")

    error_cats = compute_error_categories(entries)
    if error_cats:
        all_cat_names = sorted(set(
            cat for cats in error_cats.values() for cat in cats
        ))
        header = f"  {'model':25s} | {'doc':15s} | {'mode':10s}"
        for cat in all_cat_names:
            header += f" | {cat[:8]:>8s}"
        w(header)
        w(f"  {'-' * len(header)}")
        for (model, doc, mode), cats in sorted(error_cats.items()):
            row = f"  {model:25s} | {doc:15s} | {mode:10s}"
            for cat in all_cat_names:
                row += f" | {cats.get(cat, 0):8d}"
            w(row)

    # ---- 8. MOST USED NP FUNCTIONS ----
    section("8. MOST USED np.xxx FUNCTIONS (in passed entries)")

    func_usage = compute_np_function_usage(entries, only_passed=True)
    w(f"  Top 30 functions used in passing solutions:")
    w()
    for func, count in func_usage.most_common(30):
        w(f"    np.{func:30s} : {count:4d} times")

    # Write output
    report = "\n".join(lines) + "\n"

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report written to: {output_path}")
    else:
        print(report)

    return report


# ============================================================
# 5. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sanity check on results.jsonl files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 src/perf_review/sanity_check.py results/run_xxx/results.jsonl
  python3 src/perf_review/sanity_check.py results/run_xxx/results.jsonl -o report.txt
  python3 src/perf_review/sanity_check.py results/a/results.jsonl results/b/results.jsonl
        """,
    )
    parser.add_argument(
        "results",
        nargs="+",
        help="Path(s) to results.jsonl files",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output path for the report (default: auto-generated in same dir as first results file)",
    )

    args = parser.parse_args()

    all_entries = []
    source_paths = []
    for path in args.results:
        if not os.path.isfile(path):
            print(f"Warning: file not found: {path}", file=sys.stderr)
            continue
        entries = load_results(path)
        all_entries.extend(entries)
        source_paths.append(path)
        print(f"Loaded {len(entries)} entries from {path}")

    if not all_entries:
        print("No entries loaded, nothing to do.", file=sys.stderr)
        sys.exit(1)

    output = args.output
    if output is None:
        result_dir = os.path.dirname(args.results[0])
        output = os.path.join(result_dir, "sanity_check_report.txt")

    source_str = " + ".join(source_paths)
    write_report(all_entries, output, source_str)


if __name__ == "__main__":
    main()
