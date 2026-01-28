import re
import textwrap
import ast

# ... (Garde ta fonction fix_unexpected_indent inchangée) ...

# def extract_code_and_fix(llm_response):
#     print("\n--- Réponse Brut du LLM ---")
#     print(llm_response)
#     print("---------------------------\n")

#     # 1. Extraction (Ta regex robuste)
#     pattern = r"(?:```(?:\w*)|<code.*?>)\n?(.*?)(?:```|</code>)|((?:(?!```|<code).)*?)</code>"
#     matches = re.findall(pattern, llm_response, re.DOTALL)
    
#     code_blocks = []
#     if matches:
#         for m in matches:
#             raw_block = m[0] or m[1]
#             if not raw_block or not raw_block.strip(): continue
#             block = raw_block.strip("\n")
#             try:
#                 block = textwrap.dedent(block)
#             except: pass
#             code_blocks.append(block)
#     else:
#         code_blocks.append(llm_response.strip())

#     full_code = "\n".join(code_blocks)

#     # =================================================================
#     # ### --- SAUVETAGE INTELLIGENT (SMARTER RESCUE) ---
#     # =================================================================
#     if "result =" not in full_code and "result=" not in full_code:
#         rescue_pattern = r"^\s*result\s*=.*"
#         potential_lines = re.findall(rescue_pattern, llm_response, re.MULTILINE)
        
#         if potential_lines:
#             # ON TRIE LES CANDIDATS
#             # On cherche d'abord une ligne "propre" (sans array, sans crochets, sans DataFrame)
#             # Car souvent "result = array([...])" est juste un print du LLM.
            
#             clean_candidates = [
#                 line.strip() for line in potential_lines 
#                 if "array(" not in line and "[" not in line and "DataFrame" not in line
#             ]
            
#             best_candidate = None
            
#             if clean_candidates:
#                 # Si on a des lignes propres (ex: "result = a"), on prend la dernière trouvée
#                 best_candidate = clean_candidates[-1]
#             else:
#                 # Si on n'a que des lignes "sales" (avec crochets), on prend la PREMIÈRE trouvée
#                 # (Car la dernière est souvent celle qui est coupée/tronquée)
#                 best_candidate = potential_lines[0].strip()

#             if best_candidate and not best_candidate.startswith("#"):
#                 print(f"🚑 Sauvetage réussi avec : '{best_candidate}'")
#                 full_code += "\n" + best_candidate

#     # 2. Fix Indentation Escalier
#     full_code = fix_unexpected_indent(full_code)

#     # 3. Nettoyage Final
#     lines = full_code.split('\n')
#     cleaned_lines = []
    
#     for line in lines:
#         line_clean = line.rstrip() 
#         if not line_clean.strip(): continue
        
#         # Filtre Imports
#         if line_clean.strip().startswith(("import ", "from ")): 
#             # Petit bonus : on filtre explicitement sklearn/scipy ici aussi
#             continue
            
#         if line_clean.strip() in ["```", "```python", "<code>", "</code>"]: continue
#         if "END SOLUTION" in line_clean: continue 
        
#         # Conversion return -> result = (Bonus vu tes erreurs précédentes)
#         if line_clean.startswith("return "):
#              line_clean = line_clean.replace("return ", "result = ", 1)

#         cleaned_lines.append(line_clean)
        
#     final_code = "\n".join(cleaned_lines)
#     print(f"code final : {final_code}")
#     return final_code

import re
import ast
import textwrap

# ==============================================================================
# 1. OUTILS BAS NIVEAU (INDENTATION & VALIDATION)
# ==============================================================================

def fix_unexpected_indent(code):
    """
    Corrige le cas où la première ligne est collée à gauche, 
    mais la suite est indentée sans raison (escalier).
    """
    try:
        ast.parse(code)
        return code
    except (SyntaxError, IndentationError):
        pass

    lines = code.split('\n')
    # Recherche des deux premières lignes non vides
    indices = [i for i, l in enumerate(lines) if l.strip()]
    
    if len(indices) < 2:
        return code

    idx1, idx2 = indices[0], indices[1]
    indent1 = len(lines[idx1]) - len(lines[idx1].lstrip())
    indent2 = len(lines[idx2]) - len(lines[idx2].lstrip())

    # Si la ligne 1 ne déclenche pas d'indentation (pas de ':', '(', etc.)
    # mais que la ligne 2 est plus indentée -> C'est un bug du LLM.
    if indent2 > indent1 and not lines[idx1].strip().endswith((':','(', '[', '{', '\\')):
        offset = indent2 - indent1
        repaired = lines[:idx2]
        for line in lines[idx2:]:
            if not line.strip():
                repaired.append(line)
                continue
            curr_indent = len(line) - len(line.lstrip())
            if curr_indent >= offset:
                repaired.append(line[offset:])
            else:
                repaired.append(line.lstrip())
        return "\n".join(repaired)

    return code

def is_valid_and_useful_line(line):
    stripped = line.strip()
    
    # 1. On garde tout ce qui a une assignation ou un mot clé
    if "=" in stripped or stripped.startswith(("def ", "class ", "if ", "for ", "while ", "return", "import", "from", "with ")):
        return True

    # --- SAUVETAGE MULTI-LIGNES (NOUVEAU) ---
    # Si la ligne finit par une virgule, c'est probablement un élément de liste/tuple sur plusieurs lignes.
    # On la garde pour ne pas casser la structure.
    if stripped.endswith(","):
        return True
    # ----------------------------------------

    # 2. Détection des déchets (Arrays, Nombres isolés)
    if re.match(r"^\s*(\[|array|[\d\.-]|nan|inf|\(|$)", stripped):
        try:
            tree = ast.parse(stripped)
            # Si c'est une expression isolée SANS virgule à la fin, on considère que c'est un print inutile
            if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
                return False
        except SyntaxError:
            return False

    return True

# ==============================================================================
# 2. LOGIQUE D'EXTRACTION ET DE SAUVETAGE
# ==============================================================================

def get_raw_code_block(llm_response):
    """
    Extrait le code entre les balises markdown ou <code>.
    Gère le cas 'multi-blocs' (Chain of Thought).
    """
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
        # Fallback si pas de markdown
        code_blocks.append(llm_response.strip())

    return "\n".join(code_blocks)

def rescue_missing_result(code, full_response):
    """
    Si 'result =' est absent du code extrait, cherche dans le texte brut autour.
    """
    if "result =" not in code and "result=" not in code:
        rescue_pattern = r"^\s*result\s*=.*"
        potential_lines = re.findall(rescue_pattern, full_response, re.MULTILINE)
        
        if potential_lines:
            # On cherche la meilleure ligne candidate (qui n'est pas un commentaire ou un array affiché)
            clean_candidates = [
                line.strip() for line in potential_lines 
                if is_valid_and_useful_line(line) and not line.strip().startswith("#")
            ]
            
            # On prend la dernière trouvée (souvent la conclusion) ou la première par défaut
            best_candidate = clean_candidates[-1] if clean_candidates else potential_lines[0].strip()
            
            print(f"🚑 Sauvetage réussi : '{best_candidate}'")
            return code + "\n" + best_candidate
            
    return code

# ==============================================================================
# 3. TRANSFORMATIONS AST (SUPPRESSION IMPORT + RETURN -> RESULT)
# ==============================================================================

def apply_ast_transformations(code):
    """
    Parse le code, supprime les imports, convertit les returns globaux,
    et régénère le code propre.
    """
    try:
        tree = ast.parse(code)
        new_body = []
        
        for node in tree.body:
            # --- SUPPRESSION DES IMPORTS ---
            # Si le noeud est un import, on ne l'ajoute pas à la nouvelle liste.
            # Il disparaît donc purement et simplement.
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue 

            # --- TRANSFORMATION RETURN -> RESULT ---
            elif isinstance(node, ast.Return) and node.value:
                assign = ast.Assign(
                    targets=[ast.Name(id='result', ctx=ast.Store())],
                    value=node.value
                )
                new_body.append(assign)

            # --- LE RESTE ON GARDE ---
            else:
                new_body.append(node)
        
        tree.body = new_body
        
        # On régénère le code sous forme de string (Python 3.9+)
        if hasattr(ast, "unparse"):
            return ast.unparse(tree)
            
    except SyntaxError:
        # Si le code est cassé, on ne peut pas le transformer via AST.
        # On le renvoie tel quel pour que le 'Slow Path' (nettoyage ligne par ligne) s'en occupe.
        pass
    
    return code


# ==============================================================================
# 4. LE PIPELINE PRINCIPAL (L'ENTONNOIR)
# ==============================================================================

def extract_code_and_fix(llm_response):
    print("\n--- Réponse Brut du LLM ---")
    print(llm_response)
    print("---------------------------\n")

    # 1. Extraction brute
    raw_code = get_raw_code_block(llm_response)
    
    # 2. Sauvetage éventuel (si 'result' manque)
    code_with_result = rescue_missing_result(raw_code, llm_response)

    # 3. Tentative de Validation Globale (Fast Path)
    # Si le code est propre, on passe directement à la transformation sémantique
    clean_code = code_with_result
    try:
        ast.parse(clean_code)
    except SyntaxError:
        print("Code invalide. Nettoyage ligne par ligne activé.")
        
        # 4. Nettoyage Ligne par Ligne (Slow Path / Fallback)
        lines = clean_code.split('\n')
        cleaned_lines = []
        for line in lines:
            line_clean = line.rstrip()
            
            # Filtres Textuels basiques
            if not line_clean.strip(): continue
            if line_clean.strip().startswith(("import ", "from ")): continue
            if "END SOLUTION" in line_clean: continue
            if line_clean.strip() in ["```", "```python", "<code>", "</code>"]: continue
            
            # --- FILTRE IMPORT MANUEL (Backup indispensable ici) ---
            if line_clean.strip().startswith(("import ", "from ")): 
                continue

            # Filtre AST (Le Gardien)
            if not is_valid_and_useful_line(line_clean):
                continue
            
            cleaned_lines.append(line_clean)
        
        clean_code = "\n".join(cleaned_lines)
        
        # Fix indentation (Indispensable après un nettoyage ligne par ligne)
        clean_code = fix_unexpected_indent(clean_code)

    # 5. Transformation Sémantique (Return -> Result)
    # On le fait via AST pour être sûr de ne pas casser les fonctions
    final_code = apply_ast_transformations(clean_code)

    print(f"--- Code Final ---\n{final_code}\n------------------")
    return final_code



#*---------------------------------------------*
# Verify if "result = " is in the llm response #
#*---------------------------------------------*


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


#*-------------------------------------------------------------------------------------------------*
# Function to modify the lib before execution and therefore test if the counterfactual information #
# were taken into account or not                                                                   #
#*-------------------------------------------------------------------------------------------------*


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
