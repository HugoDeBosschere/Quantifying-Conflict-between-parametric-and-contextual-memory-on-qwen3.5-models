#!/usr/bin/env python3
"""
Affiche et collecte les stderr des entrées d'un fichier de résultats JSONL.
"""
import argparse
import json
import os


def find_repo_root():
    path = os.path.abspath(os.path.dirname(__file__))
    for _ in range(5):
        if os.path.isdir(os.path.join(path, ".git")):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


def main():
    repo = find_repo_root()
    default_data = os.path.join(repo, "results", "qwen_result3.jsonl")
    default_out = os.path.join(repo, "src", "perf_review", "error_gathered.txt")

    parser = argparse.ArgumentParser(description="Affiche et enregistre les erreurs (stderr) d'un JSONL de résultats.")
    parser.add_argument("input_file", nargs="?", default=default_data, help="Fichier JSONL de résultats")
    parser.add_argument("-o", "--output", default=default_out, help="Fichier texte de sortie pour les erreurs")
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        print(f"Fichier introuvable : {args.input_file}")
        return

    cpt = 0
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        with open(args.input_file, "r", encoding="utf-8") as f_data:
            for line in f_data:
                line = line.strip()
                if not line:
                    continue
                try:
                    res = json.loads(line)
                except json.JSONDecodeError:
                    continue
                stderr = res.get("stderr", "")
                if stderr:
                    print(stderr)
                    f.write(stderr)
                    f.write("\n\n")
                    cpt += 1
        f.write(f"Total errors : {cpt}\n")
    print(cpt)


if __name__ == "__main__":
    main()
