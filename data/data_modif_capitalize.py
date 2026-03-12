import json
import re


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

                for sh in list_shorthand:
                    pattern = re.escape(sh) + r"\.(\w+)"

                    def capitalize_match(m, _sh=sh):
                        func_name = m.group(1)
                        return f"{_sh}.{func_name[0].upper()}{func_name[1:]}"

                    current_prompt = re.sub(pattern, capitalize_match, current_prompt)

                data["prompt"] = current_prompt

            new_line = json.dumps(data, ensure_ascii=False)
            f_out.write(new_line + "\n")
            count += 1

    print(f"Terminé ! {count} lignes traitées. Sauvegardé dans {new_filename}")


if __name__ == "__main__":
    capitalize_prompt_module("data/ds1000_npyOnly.jsonl", ["numpy", "np"])
