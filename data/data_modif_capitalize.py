import json
import re


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
    Cas spécial: .T est conservé tel quel.
    """
    attr_pattern = re.compile(r"(?<![\w])([A-Za-z_]\w*)\.([A-Za-z_]\w*)")

    def repl(match: re.Match) -> str:
        obj = match.group(1)
        attr = match.group(2)
        if attr.startswith("__") and attr.endswith("__"):
            return match.group(0)
        if attr == "T":
            return match.group(0)
        if attr and attr[0].isupper():
            return match.group(0)
        return f"{obj}.{_capitalize_first(attr)}"

    return attr_pattern.sub(repl, block)


def _corrupt_prompt_capitalize(prompt: str) -> str:
    """
    Corrompt uniquement le code dans les balises <code>...</code> du prompt.
    """
    code_block_pattern = re.compile(r"<code>\n?(.*?)\n?</code>", flags=re.DOTALL)

    def repl(match: re.Match) -> str:
        original = match.group(1)
        transformed = _transform_code_block_capitalize(original)
        return f"<code>\n{transformed}\n</code>"

    return code_block_pattern.sub(repl, prompt)


def capitalize_prompt_module(filename, list_shorthand):
    """
    Lit un fichier JSONL, modifie uniquement le champ 'prompt' pour mettre en
    majuscule la première lettre des méthodes des bibliothèques spécifiées,
    et sauvegarde le résultat.
    """
    print(f"Traitement du fichier : {filename}")

    new_filename = "ds1000_npyOnly_corrupted_capitalize.jsonl"

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
