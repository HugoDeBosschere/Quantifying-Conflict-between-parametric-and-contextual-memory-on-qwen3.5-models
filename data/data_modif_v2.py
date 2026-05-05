import json
import os
import re

# Abréviations latines courantes : « i.e. », « e.g. » — sinon i.e → i.e_v2 (faux positif).
_SKIP_SINGLE_LETTER_ATTR = frozenset({("i", "e"), ("e", "g")})


def _transform_code_block_v2(block: str) -> str:
    """
    Corrompt les accès attributaires dans un bloc de code:
    - np.mean -> np.mean_v2
    - a.max -> a.max_v2
    - A.shape -> A.shape_v2
    - f(...).reshape -> f(...).reshape_v2 (chaînage après parenthèse)
    - a[i].reshape -> a[i].reshape_v2 (chaînage après crochets)

    Ne modifie pas les attributs déjà suffixés _v2 ni les dunder.
    """
    attr_pattern = re.compile(
        r"(?<![\w])(?P<id>[A-Za-z_]\w*)\.(?P<idattr>[A-Za-z_]\w*)"
        r"|(?P<closepar>\))\.(?P<parenattr>[A-Za-z_]\w*)"
        r"|(?P<closebr>\])\.(?P<brackattr>[A-Za-z_]\w*)"
    )

    def _maybe_v2(obj: str, attr: str) -> str:
        if len(obj) == 1 and len(attr) == 1 and (obj, attr) in _SKIP_SINGLE_LETTER_ATTR:
            return f"{obj}.{attr}"
        if attr.startswith("__") and attr.endswith("__"):
            return f"{obj}.{attr}"
        if attr.endswith("_v2"):
            return f"{obj}.{attr}"
        return f"{obj}.{attr}_v2"

    def repl(match: re.Match) -> str:
        if match.group("id") is not None:
            return _maybe_v2(match.group("id"), match.group("idattr"))
        if match.group("closepar") is not None:
            return _maybe_v2(match.group("closepar"), match.group("parenattr"))
        if match.group("closebr") is not None:
            return _maybe_v2(match.group("closebr"), match.group("brackattr"))
        return match.group(0)

    return attr_pattern.sub(repl, block)


def _corrupt_prompt_v2(prompt: str) -> str:
    """
    Corrompt le code dans les blocs <code>.

    Cas DS1000 :
    - paire classique <code> ... </code>
    - bloc ouvert sans </code>, qui se termine par une ligne du type
      « ### BEGIN SOLUTION » (indentation possible) — le marqueur n'est pas modifié.

    Corrompt aussi l'énoncé avant « \\nA:\\n » (ex. sessions >>> ou In [..]:),
    qui n'est pas dans des balises <code>.
    """
    marker = "\nA:\n"
    if marker in prompt:
        stem, rest = prompt.split(marker, 1)
        stem = _transform_code_block_v2(stem)
        prompt = stem + marker + rest

    code_block_pattern = re.compile(
        r"<code>\n?(?P<body>.*?)(?:(?P<close>\n?</code>)|(?=\n\s*#{1,3}\s+BEGIN\s+SOLUTION)|(?P<eos>\Z))",
        flags=re.DOTALL,
    )

    def repl(match: re.Match) -> str:
        transformed = _transform_code_block_v2(match.group("body"))
        if match.group("close"):
            return f"<code>\n{transformed}\n</code>"
        return f"<code>\n{transformed}\n"

    return code_block_pattern.sub(repl, prompt)


def v2_prompt_module(filename, list_shorthand):
    """
    Lit un fichier JSONL, modifie uniquement le champ 'prompt' pour ajouter
    le suffixe "_v2" aux méthodes des bibliothèques spécifiées, et sauvegarde le résultat.
    """
    print(f"Traitement du fichier : {filename}")

    # Sortie à côté du fichier source (ex. data/ds1000_npyOnly.jsonl → data/ds1000_..._v2.jsonl)
    out_dir = os.path.dirname(os.path.abspath(filename))
    new_filename = os.path.join(out_dir, "ds1000_npyOnly_corrupted_v2.jsonl")

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

                current_prompt = _corrupt_prompt_v2(current_prompt)

                data["prompt"] = current_prompt

            new_line = json.dumps(data, ensure_ascii=False)
            f_out.write(new_line + "\n")
            count += 1

    print(f"Terminé ! {count} lignes traitées. Sauvegardé dans {new_filename}")


if __name__ == "__main__":
    v2_prompt_module("data/ds1000_npyOnly.jsonl", ["numpy", "np"])
