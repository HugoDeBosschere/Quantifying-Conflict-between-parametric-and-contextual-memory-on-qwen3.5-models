import json
import os
import re


def _transform_code_block_underscore(block: str) -> str:
    """
    Corrompt les accès attributaires dans un bloc de code:
    - np.mean -> np.mean_
    - a.max -> a.max_
    - A.shape -> A.shape_
    - f(...).reshape -> f(...).reshape_ (chaînage après parenthèse / crochets)

    Ne modifie pas les attributs déjà suffixés "_" ni les dunder.
    """
    attr_pattern = re.compile(
        r"(?<![\w])(?P<id>[A-Za-z_]\w*)\.(?P<idattr>[A-Za-z_]\w*)"
        r"|(?P<closepar>\))\.(?P<parenattr>[A-Za-z_]\w*)"
        r"|(?P<closebr>\])\.(?P<brackattr>[A-Za-z_]\w*)"
    )

    def _maybe_us(obj: str, attr: str) -> str:
        if attr.startswith("__") and attr.endswith("__"):
            return f"{obj}.{attr}"
        if attr.endswith("_"):
            return f"{obj}.{attr}"
        return f"{obj}.{attr}_"

    def repl(match: re.Match) -> str:
        if match.group("id") is not None:
            return _maybe_us(match.group("id"), match.group("idattr"))
        if match.group("closepar") is not None:
            return _maybe_us(match.group("closepar"), match.group("parenattr"))
        if match.group("closebr") is not None:
            return _maybe_us(match.group("closebr"), match.group("brackattr"))
        return match.group(0)

    return attr_pattern.sub(repl, block)


def _corrupt_prompt_underscore(prompt: str) -> str:
    """
    Corrompt uniquement le code dans les balises <code>...</code> du prompt.
    """
    code_block_pattern = re.compile(r"<code>\n?(.*?)\n?</code>", flags=re.DOTALL)

    def repl(match: re.Match) -> str:
        original = match.group(1)
        transformed = _transform_code_block_underscore(original)
        return f"<code>\n{transformed}\n</code>"

    return code_block_pattern.sub(repl, prompt)


def underscore_prompt_module(filename, list_shorthand):
    """
    Lit un fichier JSONL, modifie uniquement le champ 'prompt' pour ajouter
    des underscores aux méthodes des bibliothèques spécifiées, et sauvegarde le résultat.
    """
    print(f"Traitement du fichier : {filename}")
    
    out_dir = os.path.dirname(os.path.abspath(filename))
    new_filename = os.path.join(out_dir, "ds1000_npy_modif_prompt.jsonl")
    
    with open(filename, "r", encoding="utf-8") as f_in, \
         open(new_filename, "w", encoding="utf-8") as f_out:
        
        count = 0
        for line in f_in:
            line = line.strip()
            if not line: continue # Sauter les lignes vides
            
            # 1. On transforme la ligne de texte en Dictionnaire Python
            data = json.loads(line)
            
            # 2. On vérifie si la clé 'prompt' existe et on la modifie
            if "prompt" in data:
                current_prompt = data["prompt"]
                
                current_prompt = _corrupt_prompt_underscore(current_prompt)
                
                # On met à jour le dictionnaire
                data["prompt"] = current_prompt
            
            # 3. On retransforme le Dictionnaire en ligne JSON (string)
            # ensure_ascii=False permet de garder les accents lisibles si il y en a
            new_line = json.dumps(data, ensure_ascii=False)
            f_out.write(new_line + "\n")
            count += 1

    print(f"Terminé ! {count} lignes traitées. Sauvegardé dans {new_filename}")

if __name__ == '__main__':
    underscore_prompt_module("data/ds1000_npy.jsonl", ["numpy", "np"])