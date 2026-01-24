import re
import textwrap



# def extract_code_and_fix(llm_response):
#     print("\n--- Réponse Brut du LLM ---")
#     print(llm_response)
#     print("---------------------------\n")

#     pattern = r"(?:```(?:\w*)|<code.*?>)\n?(.*?)(?:```|</code>)|((?:(?!```|<code).)*?)</code>"
    
#     matches = re.findall(pattern, llm_response, re.DOTALL)
    
#     code_blocks = []
    
#     if matches:
#         for m in matches:
#             raw_block = m[0] or m[1]
            
#             if not raw_block or not raw_block.strip():
#                 continue
                
#             block = raw_block.strip("\n")
            
#             try:
#                 block = textwrap.dedent(block)
#             except Exception:
#                 pass
            
#             code_blocks.append(block)
#     else:
#         code_blocks.append(llm_response.strip())

#     full_code = "\n".join(code_blocks)

#     # for when the result= part is not between markdowns
#     if "result =" not in full_code and "result=" not in full_code:
#         rescue_pattern = r"^\s*result\s*=.*"
        
#         potential_lines = re.findall(rescue_pattern, llm_response, re.MULTILINE)
        
#         if potential_lines:
#             rescued_line = potential_lines[-1].strip()
            
#             if not rescued_line.startswith("#"):
#                 full_code += "\n" + rescued_line

#     full_code = fix_unexpected_indent(full_code)

#     lines = full_code.split('\n')
#     cleaned_lines = []
    
#     for line in lines:
#         line_clean = line.rstrip() 
        
#         if not line_clean.strip(): continue
#         if line_clean.strip().startswith(("import ", "from ")): continue
        
#         if line_clean.strip() in ["```", "```python", "<code>", "</code>"]: continue

#         cleaned_lines.append(line_clean)
        
#     final_code = "\n".join(cleaned_lines)
#     print(f"code final : {final_code}")
#     return final_code


import re
import textwrap
import ast

# ... (Garde ta fonction fix_unexpected_indent inchangée) ...

def extract_code_and_fix(llm_response):
    print("\n--- Réponse Brut du LLM ---")
    print(llm_response)
    print("---------------------------\n")

    # 1. Extraction (Ta regex robuste)
    pattern = r"(?:```(?:\w*)|<code.*?>)\n?(.*?)(?:```|</code>)|((?:(?!```|<code).)*?)</code>"
    matches = re.findall(pattern, llm_response, re.DOTALL)
    
    code_blocks = []
    if matches:
        for m in matches:
            raw_block = m[0] or m[1]
            if not raw_block or not raw_block.strip(): continue
            block = raw_block.strip("\n")
            try:
                block = textwrap.dedent(block)
            except: pass
            code_blocks.append(block)
    else:
        code_blocks.append(llm_response.strip())

    full_code = "\n".join(code_blocks)

    # =================================================================
    # ### --- SAUVETAGE INTELLIGENT (SMARTER RESCUE) ---
    # =================================================================
    if "result =" not in full_code and "result=" not in full_code:
        rescue_pattern = r"^\s*result\s*=.*"
        potential_lines = re.findall(rescue_pattern, llm_response, re.MULTILINE)
        
        if potential_lines:
            # ON TRIE LES CANDIDATS
            # On cherche d'abord une ligne "propre" (sans array, sans crochets, sans DataFrame)
            # Car souvent "result = array([...])" est juste un print du LLM.
            
            clean_candidates = [
                line.strip() for line in potential_lines 
                if "array(" not in line and "[" not in line and "DataFrame" not in line
            ]
            
            best_candidate = None
            
            if clean_candidates:
                # Si on a des lignes propres (ex: "result = a"), on prend la dernière trouvée
                best_candidate = clean_candidates[-1]
            else:
                # Si on n'a que des lignes "sales" (avec crochets), on prend la PREMIÈRE trouvée
                # (Car la dernière est souvent celle qui est coupée/tronquée)
                best_candidate = potential_lines[0].strip()

            if best_candidate and not best_candidate.startswith("#"):
                print(f"🚑 Sauvetage réussi avec : '{best_candidate}'")
                full_code += "\n" + best_candidate

    # 2. Fix Indentation Escalier
    full_code = fix_unexpected_indent(full_code)

    # 3. Nettoyage Final
    lines = full_code.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line_clean = line.rstrip() 
        if not line_clean.strip(): continue
        
        # Filtre Imports
        if line_clean.strip().startswith(("import ", "from ")): 
            # Petit bonus : on filtre explicitement sklearn/scipy ici aussi
            continue
            
        if line_clean.strip() in ["```", "```python", "<code>", "</code>"]: continue
        if "END SOLUTION" in line_clean: continue 
        
        # Conversion return -> result = (Bonus vu tes erreurs précédentes)
        if line_clean.startswith("return "):
             line_clean = line_clean.replace("return ", "result = ", 1)

        cleaned_lines.append(line_clean)
        
    final_code = "\n".join(cleaned_lines)
    print(f"code final : {final_code}")
    return final_code




def ensure_result_assignment(code):
    if "result =" not in code and "result=" not in code:
        lines = code.split('\n')
        last_line_idx = -1
        
        # Trouver la dernière ligne de code réel
        for i in range(len(lines) -1, -1, -1):
            if lines[i].strip() and not lines[i].strip().startswith("#"):
                last_line_idx = i
                break
        
        if last_line_idx != -1:
            line = lines[last_line_idx]
            # On récupère l'indentation actuelle de la ligne
            indent = line[:len(line) - len(line.lstrip())]
            content = line.lstrip()
            
            # On reconstruit : Indentation + "result = " + Contenu
            lines[last_line_idx] = f"{indent}result = {content}"
            return "\n".join(lines)
            
    return code




def modify_lib(file_content, new_import_statement):
    """
    modify the lib to import in the exec_context from the ds1000 dataset
    """
    # EXPLICATION DU PATTERN :
    # 1. (exec_context\s*=\s*r?""")  -> Groupe 1 : Capture le nom de la variable et l'ouverture des guillemets (r""" ou """)
    # 2. (.*?)                       -> Groupe 2 : Capture tout le texte AVANT l'import (non-gourmand)
    # 3. (import\s+numpy\s+as\s+np)  -> Groupe 3 : Cible spécifiquement l'import que tu veux changer
    # 4. (.*?""")                    -> Groupe 4 : Capture tout le reste jusqu'à la fermeture des guillemets
    
    pattern = r'(exec_context\s*=\s*r?""")(.*?)(import\s+numpy\s+as\s+np)(.*?""")'
    
    # On utilise re.DOTALL pour que le point (.) matche aussi les retours à la ligne (\n)
    match = re.search(pattern, file_content, flags=re.DOTALL)

    if match:
        new_content = re.sub(
            pattern, 
            rf'\1\2{new_import_statement}\4', 
            file_content, 
            flags=re.DOTALL
        )
        return new_content
    else:
        print("Aucune occurrence trouvée dans exec_context.")
        return ""



import ast

def fix_unexpected_indent(code):
    """
    Corrige le cas où la première ligne est collée à gauche, 
    mais la suite est indentée sans raison (pas de ':', '(', etc. à la fin de la 1ère ligne).
    """
    # 1. Vérif rapide : Si le code est valide, on ne touche à rien
    try:
        ast.parse(code)
        return code
    except (SyntaxError, IndentationError):
        pass # On continue pour essayer de réparer

    lines = code.split('\n')
    
    # Trouver la première ligne non vide
    idx1 = -1
    for i, line in enumerate(lines):
        if line.strip():
            idx1 = i
            break
    
    if idx1 == -1: return code # Code vide

    line1 = lines[idx1]
    indent1 = len(line1) - len(line1.lstrip())

    # Vérifier si la ligne 1 "appelle" une indentation (finie par :, (, [, { ou \)
    # Si oui, l'indentation suivante est légitime, on arrête.
    if line1.strip().endswith( (':', '(', '[', '{', '\\', ',') ):
        return code

    # Trouver la deuxième ligne non vide
    idx2 = -1
    for i in range(idx1 + 1, len(lines)):
        if lines[i].strip():
            idx2 = i
            break
            
    if idx2 == -1: return code # Une seule ligne de code

    line2 = lines[idx2]
    indent2 = len(line2) - len(line2.lstrip())

    # --- DÉTECTION DU BUG ---
    # Si la ligne 2 est plus indentée que la 1, alors que la 1 n'est pas un bloc...
    if indent2 > indent1:
        offset = indent2 - indent1
        # On garde le début (la première ligne) tel quel
        repaired_lines = lines[:idx2]
        
        # Pour tout le reste, on retire l'offset (l'indentation en trop)
        for line in lines[idx2:]:
            if not line.strip():
                repaired_lines.append(line)
                continue
                
            # On calcule l'indentation actuelle
            curr_indent = len(line) - len(line.lstrip())
            
            # Si la ligne a assez d'espaces, on coupe
            if curr_indent >= offset:
                # On coupe 'offset' caractères au début, mais on préserve le reste
                # (ex: indentation relative if/else plus loin dans le code)
                repaired_lines.append(line[offset:])
            else:
                # Cas rare : ligne bizarrement moins indentée, on lstrip tout par sécurité
                repaired_lines.append(line.lstrip())
        
        return "\n".join(repaired_lines)

    return code