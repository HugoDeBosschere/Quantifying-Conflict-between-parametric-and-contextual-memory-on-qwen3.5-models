import json
import re


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

                for sh in list_shorthand:
                    # np.array -> np.array_v2, numpy.mean -> numpy.mean_v2
                    pattern = re.escape(sh) + r"\.(\w+)"
                    current_prompt = re.sub(pattern, rf"{sh}.\1_v2", current_prompt)

                data["prompt"] = current_prompt

            new_line = json.dumps(data, ensure_ascii=False)
            f_out.write(new_line + "\n")
            count += 1

    print(f"Terminé ! {count} lignes traitées. Sauvegardé dans {new_filename}")


if __name__ == "__main__":
    v2_prompt_module("data/ds1000_npyOnly.jsonl", ["numpy", "np"])
