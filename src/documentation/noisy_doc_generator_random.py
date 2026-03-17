"""
Variant of noisy_doc_generator.py where interest and noise functions are
**shuffled together** in the final documentation.

Goal:
    In the original generator, all functions of interest are listed first
    and then the noise functions. Here we keep exactly the same selection
    logic (interest_set + noise_ratio) and the same perturbations (v2,
    underscore, capitalize), but we **randomly mix** interest and noise
    before writing the docs, so that interest functions are not grouped
    all together.

Usage is identical, e.g.:

    python noisy_doc_generator_random.py \\
        --perturbation v2 \\
        --noise 0.75 \\
        --ds1000 ../../data/ds1000_npyOnly.jsonl \\
        --results ../../results/run_xxx/results.jsonl \\
        --output-dir .

It will generate:
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
    pattern = r"(?:np|numpy)\.(\w+(?:\.\w+)*)"
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

    with open(ds1000_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            for field in ("reference_code", "prompt"):
                if field in data and data[field]:
                    all_funcs.update(extract_numpy_calls(data[field]))

    if results_paths:
        for rpath in results_paths:
            with open(rpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("llm_code"):
                        all_funcs.update(extract_numpy_calls(data["llm_code"]))

    return all_funcs


# ============================================================
# 2. PERTURBATION DEFINITIONS (identiques à noisy_doc_generator.py)
# ============================================================


def _capitalize_match(text, shorthand):
    """Capitalize first letter of function names after shorthand."""
    pattern = re.escape(shorthand) + r"\.(\w+)"

    def _repl(m, _sh=shorthand):
        name = m.group(1)
        return f"{_sh}.{name[0].upper()}{name[1:]}"

    return re.sub(pattern, _repl, text)


def _capitalize_signature_start(text):
    """Capitalize first letter at start of signature."""
    return re.sub(r"^(\w)", lambda m: m.group(1).upper(), text)


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
            re.escape(sh) + r"\.(\w+)", sh + r".\1_v2", text
        ),
        "modify_signature_full": lambda text: re.sub(
            r"^(\w+)\(", r"\1_v2(", text
        ),
        "extract_modify_signature": _v2_extract_sig,
    },
    "underscore": {
        "modify_func_name": lambda name: name + "_",
        "modify_doc_text": lambda text, sh: re.sub(
            re.escape(sh) + r"\.(\w+)", sh + r".\1_", text
        ),
        "modify_signature_full": lambda text: re.sub(
            r"^(\w+)\(", r"\1_(", text
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
                    if hasattr(obj, "__name__") and "numpy" in obj.__name__:
                        stack.append((obj, f"{prefix}.{name}"))

    return functions


def get_relative_path(prefix, name, base_module_name="numpy"):
    """
    Compute relative function path for matching against interest set.
    e.g. prefix="numpy.linalg", name="norm" → "linalg.norm"
         prefix="numpy", name="isnan" → "isnan"
    """
    rel_prefix = prefix[len(base_module_name) :].lstrip(".")
    if rel_prefix:
        return f"{rel_prefix}.{name}"
    return name


# ============================================================
# 4. DOC HELPERS
# ============================================================


def supress_see_also(docstring):
    """Remove See Also sections from docstring."""
    result = re.sub("See Also\n-*\n*.*?(?=\n\n)", "", docstring, flags=re.DOTALL)
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

    IMPORTANT DIFFERENCE WITH THE BASE GENERATOR:
    ------------------------------------------------
    Here, after selecting interest + noise, we **shuffle** them together
    so that interest functions are not grouped at the top of the file.

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

    # Mélange global des fonctions d'intérêt et de bruit
    rng = random.Random(seed)
    rng.shuffle(selected)

    stats = {
        "interest": len(interest_funcs),
        "noise_total": len(noise_funcs),
        "noise_selected": len(selected_noise),
        "total_included": len(interest_funcs) + len(selected_noise),
        "noise_ratio": noise_ratio,
    }

    return selected, stats


def generate_noisy_full_doc(selected_functions, perturbation_key, list_shorthand, output_file):
    """
    Generate full doc (name + signature + full docstring).
    All included functions (interest + noise) are MODIFIED according to the
    perturbation.
    """
    p = PERTURBATIONS[perturbation_key]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Reference Documentation for numpy \n")
        f.write("=" * 60 + "\n\n")

        for prefix, name, obj, is_interest in tqdm.tqdm(
            selected_functions, desc="full"
        ):
            raw_doc = obj.__doc__
            if not raw_doc:
                continue

            display_name = p["modify_func_name"](name)
            new_doc = p["modify_signature_full"](raw_doc)
            new_doc = supress_see_also(new_doc)
            for sh in list_shorthand:
                new_doc = p["modify_doc_text"](new_doc, sh)

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


def generate_noisy_minimal_doc(selected_functions, perturbation_key, output_file):
    """
    Generate minimal doc (name + first-line signature).
    All included functions (interest + noise) are MODIFIED according to the
    perturbation.
    """
    p = PERTURBATIONS[perturbation_key]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Reference Documentation for numpy \n")
        f.write("=" * 60 + "\n\n")

        for prefix, name, obj, is_interest in tqdm.tqdm(
            selected_functions, desc="minimal"
        ):
            raw_doc = obj.__doc__
            if not raw_doc:
                continue

            display_name = p["modify_func_name"](name)
            sig = p["extract_modify_signature"](raw_doc)

            full_name = f"{prefix}.{display_name}"

            f.write(f"FUNCTION: {full_name}\n")
            f.write("-" * (10 + len(full_name)) + "\n")
            f.write((sig or "") + "\n")
            f.write("#" * 40 + "\n")

    print(f"  Minimal doc generated: {output_file}")


def generate_noisy_ultra_minimal_doc(selected_functions, perturbation_key, output_file):
    """
    Generate ultra-minimal doc (name only).
    All included functions (interest + noise) are MODIFIED according to the
    perturbation.
    """
    p = PERTURBATIONS[perturbation_key]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Reference Documentation for numpy \n")
        f.write("=" * 60 + "\n\n")

        for prefix, name, obj, is_interest in tqdm.tqdm(
            selected_functions, desc="ultra_minimal"
        ):
            raw_doc = obj.__doc__
            if not raw_doc:
                continue

            display_name = p["modify_func_name"](name)

            full_name = f"{prefix}.{display_name}"
            f.write(f"FUNCTION: {full_name}\n")
            f.write("-" * (10 + len(full_name)) + "\n")

    print(f"  Ultra-minimal doc generated: {output_file}")


# ============================================================
# 6. MAIN
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate noisy numpy documentation with shuffled interest/noise functions."
    )
    parser.add_argument(
        "--perturbation",
        choices=["v2", "underscore", "capitalize"],
        required=True,
        help="Type de perturbation à appliquer (v2, underscore, capitalize).",
    )
    parser.add_argument(
        "--noise",
        "-n",
        type=float,
        nargs="+",
        required=True,
        help="Ratio(s) de fonctions de bruit à inclure (0.0 à 1.0). Peut en prendre plusieurs.",
    )
    parser.add_argument(
        "--ds1000",
        required=True,
        help="Chemin vers ds1000_npyOnly.jsonl (dataset de base).",
    )
    parser.add_argument(
        "--results",
        nargs="*",
        default=None,
        help="Chemins optionnels vers un ou plusieurs results.jsonl pour élargir l'ensemble de fonctions d'intérêt.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Répertoire de sortie pour les fichiers de documentation bruitée.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed pour la sélection et le mélange des fonctions.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Extracting interest functions from {args.ds1000} ...")
    interest_set = extract_interest_functions(args.ds1000, args.results)
    print(f"  -> {len(interest_set)} interest functions")

    print("[INFO] Crawling numpy functions ...")
    all_functions = crawl_numpy_functions([numpy])
    print(f"  -> {len(all_functions)} total numpy functions crawled")

    # Shorthands for modifying textual doc (np. / numpy.)
    list_shorthand = ["np", "numpy"]

    for noise_ratio in args.noise:
        if not 0.0 <= noise_ratio <= 1.0:
            raise SystemExit(
                f"Noise ratio must be between 0.0 and 1.0, got {noise_ratio}"
            )

        print(f"\n{'=' * 60}")
        print(
            f"Generating (shuffled): perturbation={args.perturbation}, noise={noise_ratio}"
        )
        print(f"{'=' * 60}")

        print(f"[INFO] Selecting functions with noise ratio={noise_ratio} ...")
        selected_functions, stats = select_functions(
            all_functions, interest_set, noise_ratio, seed=args.seed
        )
        print(f"  Stats: {stats}")

        suffix = f"noise{int(noise_ratio * 100)}"

        full_path = os.path.join(
            args.output_dir,
            f"corrupted_full_numpy_{args.perturbation}_{suffix}.txt",
        )
        minimal_path = os.path.join(
            args.output_dir,
            f"corrupted_minimal_numpy_{args.perturbation}_{suffix}.txt",
        )
        ultra_minimal_path = os.path.join(
            args.output_dir,
            f"corrupted_ultra_minimal_numpy_{args.perturbation}_{suffix}.txt",
        )

        print("[INFO] Generating FULL doc ...")
        generate_noisy_full_doc(
            selected_functions, args.perturbation, list_shorthand, full_path
        )

        print("[INFO] Generating MINIMAL doc ...")
        generate_noisy_minimal_doc(
            selected_functions, args.perturbation, minimal_path
        )

        print("[INFO] Generating ULTRA-MINIMAL doc ...")
        generate_noisy_ultra_minimal_doc(
            selected_functions, args.perturbation, ultra_minimal_path
        )

    print("[DONE] Noisy shuffled docs generated for all requested noise ratios.")


if __name__ == "__main__":
    main()

