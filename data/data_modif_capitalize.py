import json
import os
import re

_SKIP_SINGLE_LETTER_ATTR = frozenset({("i", "e"), ("e", "g")})


def _capitalize_first(name: str) -> str:
    if not name:
        return name
    return name[0].upper() + name[1:]


def _transform_code_block_capitalize(block: str) -> str:
    """
    Corrompt les accès attributaires dans un bloc de code:
    - np.mean -> np.Mean
    - a.max -> a.Max
    - A.shape -> A.Shape
    - f(...).reshape -> f(...).Reshape (chaînage après parenthèse / crochets)

    Cas spécial: .T est conservé tel quel.
    """
    attr_pattern = re.compile(
        r"(?<![\w])(?P<id>[A-Za-z_]\w*)\.(?P<idattr>[A-Za-z_]\w*)"
        r"|(?P<closepar>\))\.(?P<parenattr>[A-Za-z_]\w*)"
        r"|(?P<closebr>\])\.(?P<brackattr>[A-Za-z_]\w*)"
    )

    def _maybe_cap(obj: str, attr: str) -> str:
        if len(obj) == 1 and len(attr) == 1 and (obj, attr) in _SKIP_SINGLE_LETTER_ATTR:
            return f"{obj}.{attr}"
        if attr.startswith("__") and attr.endswith("__"):
            return f"{obj}.{attr}"
        if attr == "T":
            return f"{obj}.{attr}"
        if attr and attr[0].isupper():
            return f"{obj}.{attr}"
        return f"{obj}.{_capitalize_first(attr)}"

    def repl(match: re.Match) -> str:
        if match.group("id") is not None:
            return _maybe_cap(match.group("id"), match.group("idattr"))
        if match.group("closepar") is not None:
            return _maybe_cap(match.group("closepar"), match.group("parenattr"))
        if match.group("closebr") is not None:
            return _maybe_cap(match.group("closebr"), match.group("brackattr"))
        return match.group(0)

    return attr_pattern.sub(repl, block)


def _corrupt_prompt_capitalize(prompt: str) -> str:
    """
    Corrompt le code dans les blocs <code> (voir data_modif_v2 pour les variantes DS1000).
    Corrompt aussi l'énoncé avant « \\nA:\\n » (sessions >>> / In [..]:).
    """
    marker = "\nA:\n"
    if marker in prompt:
        stem, rest = prompt.split(marker, 1)
        stem = _transform_code_block_capitalize(stem)
        prompt = stem + marker + rest

    code_block_pattern = re.compile(
        r"<code>\n?(?P<body>.*?)(?:(?P<close>\n?</code>)|(?=\n\s*#{1,3}\s+BEGIN\s+SOLUTION)|(?P<eos>\Z))",
        flags=re.DOTALL,
    )

    def repl(match: re.Match) -> str:
        transformed = _transform_code_block_capitalize(match.group("body"))
        if match.group("close"):
            return f"<code>\n{transformed}\n</code>"
        return f"<code>\n{transformed}\n"

    return code_block_pattern.sub(repl, prompt)


def capitalize_prompt_module(filename, list_shorthand):
    """
    Lit un fichier JSONL, modifie uniquement le champ 'prompt' pour mettre en
    majuscule la première lettre des méthodes des bibliothèques spécifiées,
    et sauvegarde le résultat.
    """
    print(f"Traitement du fichier : {filename}")

    out_dir = os.path.dirname(os.path.abspath(filename))
    new_filename = os.path.join(out_dir, "ds1000_npyOnly_corrupted_capitalize.jsonl")

    with open(filename, "r", encoding="utf-8") as f_in, \
         open(new_filename, "w", encoding="utf-8") as f_out:

        count = 0
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            if "prompt" in data:
                current_prompt = data["prompt"]

                current_prompt = _corrupt_prompt_capitalize(current_prompt)

                data["prompt"] = current_prompt

            new_line = json.dumps(data, ensure_ascii=False)
            f_out.write(new_line + "\n")
            count += 1

    print(f"Terminé ! {count} lignes traitées. Sauvegardé dans {new_filename}")


if __name__ == "__main__":
    capitalize_prompt_module("data/ds1000_npyOnly.jsonl", ["numpy", "np"])
