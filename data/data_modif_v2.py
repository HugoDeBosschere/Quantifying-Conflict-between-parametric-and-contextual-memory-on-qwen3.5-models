import json
import re


def _transform_code_block_v2(block: str) -> str:
    """
    Corrompt les accès attributaires dans un bloc de code:
    - np.mean -> np.mean_v2
    - a.max -> a.max_v2
    - A.shape -> A.shape_v2
    Ne modifie pas les attributs déjà suffixés _v2 ni les dunder.
    """
    attr_pattern = re.compile(r"(?<![\w])([A-Za-z_]\w*)\.([A-Za-z_]\w*)")

    def repl(match: re.Match) -> str:
        obj = match.group(1)
        attr = match.group(2)
        if attr.startswith("__") and attr.endswith("__"):
            return match.group(0)
        if attr.endswith("_v2"):
            return match.group(0)
        return f"{obj}.{attr}_v2"

    return attr_pattern.sub(repl, block)


def _corrupt_prompt_v2(prompt: str) -> str:
    """
    Corrompt uniquement le code dans les balises <code>...</code> du prompt.
    """
    code_block_pattern = re.compile(r"<code>\n?(.*?)\n?</code>", flags=re.DOTALL)

    def repl(match: re.Match) -> str:
        original = match.group(1)
        transformed = _transform_code_block_v2(original)
        return f"<code>\n{transformed}\n</code>"

    return code_block_pattern.sub(repl, prompt)


def v2_prompt_module(filename, list_shorthand):
    """
    Lit un fichier JSONL, modifie uniquement le champ 'prompt' pour ajouter
    le suffixe "_v2" aux méthodes des bibliothèques spécifiées, et sauvegarde le résultat.
    """
    print(f"Traitement du fichier : {filename}")

    new_filename = "ds1000_npyOnly_corrupted_v2.jsonl"

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
