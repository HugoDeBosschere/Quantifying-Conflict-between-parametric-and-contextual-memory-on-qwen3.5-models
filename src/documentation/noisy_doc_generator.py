"""
Generates numpy documentation with controlled noise levels for evaluating
LLM performance with varying documentation density.

Concept:
    - Interest functions: numpy functions actually used in DS1000 exercises
      (extracted from reference_code, prompt, and optionally LLM results)
      → these are MODIFIED according to the perturbation (e.g. _v2, capitalize)
    - Noise functions: other numpy functions not directly used in exercises
      → these appear with their REAL (unmodified) numpy documentation
    - noise_ratio controls what fraction of non-interest functions are included:
        0.0 → only interest functions (modified), no noise
        0.5 → interest functions (modified) + 50% of other functions (unmodified)
        1.0 → interest functions (modified) + all other functions (unmodified)
    - The goal is to test whether LLMs lose track of the modifications when
      they are drowned in large volumes of normal (unmodified) documentation.

Available perturbations: v2, underscore, capitalize

Usage:
    python noisy_doc_generator.py \\
        --perturbation v2 \\
        --noise 0.75 \\
        --ds1000 ../../data/ds1000_npyOnly.jsonl \\
        --results ../../results/run_xxx/results.jsonl \\
        --output-dir .

    This generates 3 files:
        corrupted_full_numpy_v2_noise75.txt
        corrupted_minimal_numpy_v2_noise75.txt
        corrupted_ultra_minimal_numpy_v2_noise75.txt
"""

import numpy
import inspect
import types
import re
import json
import random
import argparse
import os
import tqdm


# ============================================================
# 1. EXTRACT INTEREST FUNCTIONS FROM DS1000 + RESULTS
# ============================================================

def extract_numpy_calls(code_text):
    """Extract np.xxx / numpy.xxx dotted attribute chains from code."""
    pattern = r'(?:np|numpy)\.(\w+(?:\.\w+)*)'
    return set(re.findall(pattern, code_text))


def extract_interest_functions(ds1000_path, results_paths=None):
    """
    Extract all numpy functions referenced in DS1000 exercises.

    Sources:
        - reference_code : the ground-truth solution
        - prompt          : setup code visible to the LLM
        - llm_code        : (optional) functions LLMs actually use in results

    Returns a set of relative function paths, e.g. {"isnan", "array", "linalg.norm"}
    """
    all_funcs = set()

    with open(ds1000_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            for field in ('reference_code', 'prompt'):
                if field in data and data[field]:
                    all_funcs.update(extract_numpy_calls(data[field]))

    if results_paths:
        for rpath in results_paths:
            with open(rpath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get('llm_code'):
                        all_funcs.update(extract_numpy_calls(data['llm_code']))

    return all_funcs


# ============================================================
# 2. PERTURBATION DEFINITIONS
# ============================================================

def _capitalize_match(text, shorthand):
    """Capitalize first letter of function names after shorthand."""
    pattern = re.escape(shorthand) + r'\.(\w+)'
    def _repl(m, _sh=shorthand):
        name = m.group(1)
        return f"{_sh}.{name[0].upper()}{name[1:]}"
    return re.sub(pattern, _repl, text)


def _capitalize_signature_start(text):
    """Capitalize first letter at start of signature."""
    return re.sub(r'^(\w)', lambda m: m.group(1).upper(), text)


def _capitalize_extract_sig(docstring):
    """Extract and capitalize first-line signature."""
    first_line = re.search(".*?(?=\n)", docstring)
    if first_line:
        sig = re.search(r"(\w+)\(.*", first_line.group(0))
        if sig:
            matched = sig.group(0)
            return matched[0].upper() + matched[1:]
    return ""


def _v2_extract_sig(docstring):
    """Extract and add _v2 to first-line signature."""
    first_line = re.search(".*?(?=\n)", docstring)
    if first_line:
        sig = re.search(r"(\w+)\(.*", first_line.group(0))
        if sig:
            return re.sub(r"^(\w+)\(", r"\1_v2(", sig.group(0))
    return ""


def _underscore_extract_sig(docstring):
    """Extract and add _ to first-line signature."""
    first_line = re.search(".*?(?=\n)", docstring)
    if first_line:
        sig = re.search(r"(\w+)\(.*", first_line.group(0))
        if sig:
            return re.sub(r"^(\w+)\(", r"\1_(", sig.group(0))
    return ""


PERTURBATIONS = {
    "v2": {
        "modify_func_name": lambda name: name + "_v2",
        "modify_doc_text": lambda text, sh: re.sub(
            re.escape(sh) + r'\.(\w+)', sh + r'.\1_v2', text
        ),
        "modify_signature_full": lambda text: re.sub(
            r'^(\w+)\(', r'\1_v2(', text
        ),
        "extract_modify_signature": _v2_extract_sig,
    },
    "underscore": {
        "modify_func_name": lambda name: name + "_",
        "modify_doc_text": lambda text, sh: re.sub(
            re.escape(sh) + r'\.(\w+)', sh + r'.\1_', text
        ),
        "modify_signature_full": lambda text: re.sub(
            r'^(\w+)\(', r'\1_(', text
        ),
        "extract_modify_signature": _underscore_extract_sig,
    },
    "capitalize": {
        "modify_func_name": lambda name: name[0].upper() + name[1:],
        "modify_doc_text": _capitalize_match,
        "modify_signature_full": _capitalize_signature_start,
        "extract_modify_signature": _capitalize_extract_sig,
    },
}


# ============================================================
# 3. NUMPY CRAWLER
# ============================================================

def crawl_numpy_functions(list_module):
    """
    Crawl numpy module tree and return a list of
    (prefix, name, obj) tuples for all functions/ufuncs found.
    """
    functions = []
    seen = set()

    for base_module in list_module:
        stack = [(base_module, base_module.__name__)]
        visited = set()

        while stack:
            current_mod, prefix = stack.pop()
            if current_mod in visited:
                continue
            visited.add(current_mod)

            for name in sorted(dir(current_mod)):
                if name.startswith("_"):
                    continue
                try:
                    obj = getattr(current_mod, name)
                except Exception:
                    continue

                if inspect.isfunction(obj) or isinstance(obj, numpy.ufunc):
                    if obj in seen:
                        continue
                    seen.add(obj)
                    functions.append((prefix, name, obj))

                elif isinstance(obj, types.ModuleType):
                    if hasattr(obj, '__name__') and 'numpy' in obj.__name__:
                        stack.append((obj, f"{prefix}.{name}"))

    return functions


def get_relative_path(prefix, name, base_module_name="numpy"):
    """
    Compute relative function path for matching against interest set.
    e.g. prefix="numpy.linalg", name="norm" → "linalg.norm"
         prefix="numpy", name="isnan" → "isnan"
    """
    rel_prefix = prefix[len(base_module_name):].lstrip(".")
    if rel_prefix:
        return f"{rel_prefix}.{name}"
    return name


# ============================================================
# 4. DOC HELPERS
# ============================================================

def supress_see_also(docstring):
    """Remove See Also sections from docstring."""
    result = re.sub(
        "See Also\n-*\n*.*?(?=\n\n)", "", docstring, flags=re.DOTALL
    )
    result = re.sub(
        r"See also\n-*\n*.*?(?=\n\s*\n|$)", "", result, flags=re.DOTALL
    )
    result = re.sub(
        r"See Also\n-*\n*.*?(?=\n\s*\n|$)", "", result, flags=re.DOTALL
    )
    return result


# ============================================================
# 5. DOCUMENTATION GENERATORS WITH NOISE CONTROL
# ============================================================

def select_functions(all_functions, interest_set, noise_ratio, seed=42):
    """
    Split functions into interest/noise, select noise subset,
    and return the combined list to include in the doc.

    Interest functions will be MODIFIED (perturbation applied).
    Noise functions will appear with their REAL numpy documentation.

    Returns:
        selected : list of (prefix, name, obj, is_interest) tuples
        stats    : dict with counts
    """
    interest_funcs = []
    noise_funcs = []

    for prefix, name, obj in all_functions:
        rel_path = get_relative_path(prefix, name)
        if rel_path in interest_set:
            interest_funcs.append((prefix, name, obj))
        else:
            noise_funcs.append((prefix, name, obj))

    n_noise = int(len(noise_funcs) * noise_ratio)
    if n_noise >= len(noise_funcs):
        selected_noise = noise_funcs
    else:
        rng = random.Random(seed)
        selected_noise = rng.sample(noise_funcs, n_noise)

    selected = (
        [(p, n, o, True) for p, n, o in interest_funcs]
        + [(p, n, o, False) for p, n, o in selected_noise]
    )

    stats = {
        "interest": len(interest_funcs),
        "noise_total": len(noise_funcs),
        "noise_selected": len(selected_noise),
        "total_included": len(interest_funcs) + len(selected_noise),
        "noise_ratio": noise_ratio,
    }

    return selected, stats


def generate_noisy_full_doc(selected_functions, perturbation_key,
                            list_shorthand, output_file):
    """
    Generate full doc (name + signature + full docstring).
    Interest functions are MODIFIED, noise functions use REAL numpy doc.
    """
    p = PERTURBATIONS[perturbation_key]

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Reference Documentation for numpy \n")
        f.write("=" * 60 + "\n\n")

        for prefix, name, obj, is_interest in tqdm.tqdm(selected_functions, desc="full"):
            raw_doc = obj.__doc__
            if not raw_doc:
                continue

            if is_interest:
                display_name = p["modify_func_name"](name)
                new_doc = p["modify_signature_full"](raw_doc)
                new_doc = supress_see_also(new_doc)
                for sh in list_shorthand:
                    new_doc = p["modify_doc_text"](new_doc, sh)
            else:
                display_name = name
                new_doc = supress_see_also(raw_doc)

            full_name = f"{prefix}.{display_name}"
            f.write(f"FUNCTION: {full_name}\n")
            f.write("-" * (10 + len(full_name)) + "\n")
            f.write(new_doc + "\n")
            f.write("\n" + "#" * 40 + "\n\n")

    print(f"  Full doc generated: {output_file}")


def _extract_real_signature(docstring):
    """Extract the first-line signature from a docstring, unmodified."""
    first_line = re.search(".*?(?=\n)", docstring)
    if first_line:
        sig = re.search(r"(\w+)\(.*", first_line.group(0))
        if sig:
            return sig.group(0)
    return ""


def generate_noisy_minimal_doc(selected_functions, perturbation_key,
                               output_file):
    """
    Generate minimal doc (name + first-line signature).
    Interest functions are MODIFIED, noise functions use REAL signature.
    """
    p = PERTURBATIONS[perturbation_key]

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Reference Documentation for numpy \n")
        f.write("=" * 60 + "\n\n")

        for prefix, name, obj, is_interest in tqdm.tqdm(selected_functions, desc="minimal"):
            raw_doc = obj.__doc__
            if not raw_doc:
                continue

            if is_interest:
                display_name = p["modify_func_name"](name)
                sig = p["extract_modify_signature"](raw_doc)
            else:
                display_name = name
                sig = _extract_real_signature(raw_doc)

            full_name = f"{prefix}.{display_name}"

            f.write(f"FUNCTION: {full_name}\n")
            f.write("-" * (10 + len(full_name)) + "\n")
            f.write((sig or "") + "\n")
            f.write("#" * 40 + "\n")

    print(f"  Minimal doc generated: {output_file}")


def generate_noisy_ultra_minimal_doc(selected_functions, perturbation_key,
                                     output_file):
    """
    Generate ultra-minimal doc (name only).
    Interest functions are MODIFIED, noise functions use REAL name.
    """
    p = PERTURBATIONS[perturbation_key]

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Reference Documentation for numpy \n")
        f.write("=" * 60 + "\n\n")

        for prefix, name, obj, is_interest in tqdm.tqdm(selected_functions, desc="ultra_minimal"):
            raw_doc = obj.__doc__
            if not raw_doc:
                continue

            if is_interest:
                display_name = p["modify_func_name"](name)
            else:
                display_name = name

            full_name = f"{prefix}.{display_name}"
            f.write(f"FUNCTION: {full_name}\n")
            f.write("-" * (10 + len(full_name)) + "\n")

    print(f"  Ultra-minimal doc generated: {output_file}")


# ============================================================
# 6. MAIN PIPELINE
# ============================================================

def generate_all_docs(perturbation_key, noise_ratio, ds1000_path,
                      results_paths=None, output_dir=".", seed=42):
    """
    Full pipeline: extract interest functions → crawl numpy →
    select with noise → generate 3 doc levels.

    Returns the stats dict and the set of interest function names.
    """
    print(f"[1/4] Extracting interest functions from DS1000...")
    interest_set = extract_interest_functions(ds1000_path, results_paths)
    print(f"       Found {len(interest_set)} unique numpy function references")

    print(f"[2/4] Crawling numpy module tree...")
    all_functions = crawl_numpy_functions([numpy])
    print(f"       Found {len(all_functions)} total functions/ufuncs")

    print(f"[3/4] Selecting functions (noise_ratio={noise_ratio})...")
    selected, stats = select_functions(
        all_functions, interest_set, noise_ratio, seed
    )
    print(f"       Interest: {stats['interest']} | "
          f"Noise: {stats['noise_selected']}/{stats['noise_total']} | "
          f"Total: {stats['total_included']}")

    noise_pct = int(noise_ratio * 100)
    base = f"corrupted_{{level}}_numpy_{perturbation_key}_noise{noise_pct}.txt"

    print(f"[4/4] Generating docs with '{perturbation_key}' perturbation...")

    full_path = os.path.join(output_dir, base.format(level="full"))
    minimal_path = os.path.join(output_dir, base.format(level="minimal"))
    ultra_path = os.path.join(output_dir, base.format(level="ultra_minimal"))

    generate_noisy_full_doc(
        selected, perturbation_key, ["np", "numpy"], full_path
    )
    generate_noisy_minimal_doc(
        selected, perturbation_key, minimal_path
    )
    generate_noisy_ultra_minimal_doc(
        selected, perturbation_key, ultra_path
    )

    interest_file = os.path.join(
        output_dir, f"interest_functions_{perturbation_key}.txt"
    )
    with open(interest_file, 'w') as f:
        for func_name in sorted(interest_set):
            f.write(func_name + "\n")
    print(f"\nInterest functions list saved to: {interest_file}")

    print(f"\nDone! Files generated in {output_dir}/")
    print(f"  - {os.path.basename(full_path)}")
    print(f"  - {os.path.basename(minimal_path)}")
    print(f"  - {os.path.basename(ultra_path)}")

    return stats, interest_set


# ============================================================
# 7. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate numpy docs with controlled noise levels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # v2 perturbation, 75%% noise
  python noisy_doc_generator.py --perturbation v2 --noise 0.75 \\
      --ds1000 ../../data/ds1000_npyOnly.jsonl

  # capitalize perturbation, 50%% noise, with LLM results
  python noisy_doc_generator.py --perturbation capitalize --noise 0.5 \\
      --ds1000 ../../data/ds1000_npyOnly.jsonl \\
      --results ../../results/run_xxx/results.jsonl

  # Generate multiple noise levels at once
  python noisy_doc_generator.py --perturbation v2 \\
      --noise 0.0 0.25 0.5 0.75 1.0 \\
      --ds1000 ../../data/ds1000_npyOnly.jsonl
        """,
    )
    parser.add_argument(
        "--perturbation", "-p",
        choices=list(PERTURBATIONS.keys()),
        required=True,
        help="Type of perturbation to apply",
    )
    parser.add_argument(
        "--noise", "-n",
        type=float,
        nargs="+",
        required=True,
        help="Noise ratio(s) between 0.0 and 1.0 (can specify multiple)",
    )
    parser.add_argument(
        "--ds1000",
        required=True,
        help="Path to ds1000_npyOnly.jsonl",
    )
    parser.add_argument(
        "--results", "-r",
        nargs="*",
        default=None,
        help="Path(s) to results.jsonl files for broader function coverage",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=".",
        help="Output directory (default: current directory)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for noise selection (default: 42)",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for noise_ratio in args.noise:
        if not 0.0 <= noise_ratio <= 1.0:
            parser.error(f"Noise ratio must be between 0.0 and 1.0, got {noise_ratio}")

        print(f"\n{'='*60}")
        print(f"Generating: perturbation={args.perturbation}, noise={noise_ratio}")
        print(f"{'='*60}")

        generate_all_docs(
            perturbation_key=args.perturbation,
            noise_ratio=noise_ratio,
            ds1000_path=args.ds1000,
            results_paths=args.results,
            output_dir=args.output_dir,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
