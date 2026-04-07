import re
import ast
import textwrap
import traceback 

# ==============================================================================
# 1. OUTILS BAS NIVEAU (INDENTATION & VALIDATION)
# ==============================================================================

def fix_unexpected_indent(code):
    try:
        try:
            ast.parse(code)
            return code
        except (SyntaxError, IndentationError):
            pass

        lines = code.split('\n')
        indices = [i for i, l in enumerate(lines) if l.strip()]
        
        if len(indices) < 2:
            return code

        idx1, idx2 = indices[0], indices[1]
        indent1 = len(lines[idx1]) - len(lines[idx1].lstrip())
        indent2 = len(lines[idx2]) - len(lines[idx2].lstrip())

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
            
    except Exception as e:
        print(f"⚠️ Warning: fix_unexpected_indent failed: {e}")
        return code

    return code



# def is_valid_and_useful_line(line):
#     """
#     Vérifie si une ligne de code générée par le LLM est utile à l'exécution.
#     Conserve les assignations ET les expressions brutes (ex: np.sum(x)).
#     Filtre les commentaires, les prints, les asserts, et les lignes vides.
#     """
#     line_clean = line.strip()
    
#     # 1. Rejeter les lignes vides
#     if not line_clean:
#         return False
        
#     # 2. Rejeter les commentaires purs
#     if line_clean.startswith("#"):
#         return False
        
#     # 3. Rejeter les instructions d'affichage (print)
#     # Les LLMs ajoutent souvent print(result) pour "montrer" la réponse, 
#     # ce qui pollue l'évaluation via exec()
#     if line_clean.startswith("print(") or line_clean.startswith("print "):
#         return False
        
#     # 4. Rejeter les assertions (assert)
#     # Parfois, le LLM s'auto-évalue en générant des assert
#     if line_clean.startswith("assert "):
#         return False
        
#     # 5. Rejeter les balises markdown résiduelles (au cas où la regex a raté)
#     if line_clean.startswith("```"):
#         return False

#     # Si la ligne passe tous ces filtres, on la considère valide.
#     # On autorise donc : `result = x` MAIS AUSSI `np.insert(arr, 0, element)`
#     return True




def is_valid_and_useful_line(line):
    """
    Filtre ligne par ligne (Slow Path) quand l'AST global échoue.
    """
    try:
        stripped = line.strip()
        
        # 1. On garde les assignations, définitions, et mots-clés
        if "=" in stripped or stripped.startswith(("def ", "class ", "if ", "for ", "while ", "return", "import", "from", "with ")):
            return True

        # SAUVETAGE MULTI-LIGNES (ex: éléments de liste terminant par virgule)
        if stripped.endswith(","):
            return True

        # 2. Détection des déchets (Arrays, Nombres isolés, Appels de fonctions orphelins)
        # Regex élargie pour attraper 'array', 'tensor', '[', '(', chiffres...
        if re.match(r"^\s*(\[|array|tensor|DataFrame|[\d\.-]|nan|inf|\(|$)", stripped):
            try:
                tree = ast.parse(stripped)
                # Si c'est une expression isolée (pas d'assignation)
                if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
                    # On vérifie si c'est un print autorisé
                    expr = tree.body[0].value
                    if isinstance(expr, ast.Call):
                        func_name = ""
                        if isinstance(expr.func, ast.Name): func_name = expr.func.id
                        elif isinstance(expr.func, ast.Attribute): func_name = expr.func.attr
                        
                        # Si c'est print, on garde. Sinon (array, tensor...), on jette.
                        if func_name in {'print', 'show', 'plot', 'display'}:
                            return True
                        else:
                            return False # C'est un array(...) inutile
                    
                    return False # C'est une valeur brute (ex: 50)
            except SyntaxError:
                return False # Syntaxe pourrie -> Poubelle
                
    except Exception as e:
        print(f"⚠️ Warning: is_valid_and_useful_line failed on '{line}': {e}")
        return True

    return True

# ==============================================================================
# 2. LOGIQUE D'EXTRACTION
# ==============================================================================

def get_raw_code_block(llm_response):
    try:
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

        return "\n".join(code_blocks)
        
    except Exception as e:
        print(f"⚠️ Warning: get_raw_code_block failed: {e}")
        return llm_response

# ==============================================================================
# 3. TRANSFORMATIONS AST (LE COEUR DU NETTOYAGE)
# ==============================================================================

def is_static_definition(value_node):
    """
    Détermine si une assignation (a = ...) est du hardcode pur (statique).
    Retourne True si l'expression ne dépend d'aucune variable externe inconnue.
    """
    LIBRARY_WHITELIST = {
        'np', 'numpy', 'pd', 'pandas', 'plt', 'math', 'scipy', 'sklearn', 
        'tf', 'torch', 'datetime', 'random', 'json', 're', 'itertools', 'collections'
    }
    
    for node in ast.walk(value_node):
        # Si on trouve un nom de variable (Name)
        if isinstance(node, ast.Name):
            # Si ce nom n'est pas une librairie connue, c'est une dépendance (ex: 'a', 'x')
            if node.id not in LIBRARY_WHITELIST:
                return False # Ce n'est pas statique, ça dépend d'une variable
                
    return True

def apply_ast_transformations(code):
    """
    Nettoyage intelligent via AST.
    A. Supprime les imports.
    B. Transforme 'return' en 'result ='.
    C. Supprime les assignations statiques écrasantes (a = [1,2]).
    D. Supprime les expressions orphelines (array([1,2])) sauf print/plot.
    """
    try:
        tree = ast.parse(code)
        new_body = []
        
        for node in tree.body:
            # A. Suppression des Imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue 

            # B. Transformation du Return
            elif isinstance(node, ast.Return) and node.value:
                assign = ast.Assign(
                    targets=[ast.Name(id='result', ctx=ast.Store())],
                    value=node.value
                )
                ast.copy_location(assign, node)
                new_body.append(assign)

            # C. Traitement des Assignations (Suppression constantes)
            elif isinstance(node, ast.Assign):
                # Assignation à un subscript (ex: b[i] = 1, b[np.arange(n), a] = 1) → toujours garder
                if any(not isinstance(t, ast.Name) for t in node.targets):
                    new_body.append(node)
                    continue
                targets_names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if 'result' in targets_names:
                    new_body.append(node)
                elif is_static_definition(node.value):
                    continue  # constante statique (ex: a = 1, a = [1,2])
                else:
                    new_body.append(node)

            # D. Suppression des Expressions Orphelines (Le cas array([...]))
            elif isinstance(node, ast.Expr):
                keep_expr = False
                
                # On ne garde que les appels de fonctions autorisés (effets de bord)
                if isinstance(node.value, ast.Call):
                    func = node.value.func
                    func_name = ""
                    if isinstance(func, ast.Name): func_name = func.id
                    elif isinstance(func, ast.Attribute): func_name = func.attr
                    
                    # WHITELIST : On ne garde que ça
                    if func_name in {'print', 'show', 'plot', 'seed', 'compile', 'fit', 'append', 'extend', 'write', 'display'}:
                        keep_expr = True
                
                if keep_expr:
                    new_body.append(node)
                else:
                    # Ici, array([1,2]) est jeté car 'array' n'est pas dans la whitelist
                    # print(f"Suppression expression orpheline L{node.lineno}")
                    continue

            # E. Le reste (For, While, If, Def...) on garde tout
            else:
                new_body.append(node)
        
        tree.body = new_body
        ast.fix_missing_locations(tree)
        
        if hasattr(ast, "unparse"):
            return ast.unparse(tree)
            
    except SyntaxError:
        pass 
    except Exception as e:
        print(f"⚠️ Warning: apply_ast_transformations failed: {e}")
    
    return code



# ==============================================================================
# 4. PIPELINE PRINCIPAL
# ==============================================================================


# import re
# import ast
# import traceback

# def extract_code_and_fix(llm_response):
#     try:
#         print("\n--- Réponse Brut du LLM ---")
#         print(llm_response)
#         print("---------------------------\n")

#         # 1. Extraction robuste du code via Regex
#         # Accepte ```python, ```markdown, ```py ou juste ```
#         # re.DOTALL permet à (.*?) de capturer les retours à la ligne
#         pattern = r'
# http://googleusercontent.com/immersive_entry_chip/0
# http://googleusercontent.com/immersive_entry_chip/1




# def extract_code_and_fix_vLyonnais(llm_response):
def extract_code_and_fix(llm_response):
    try:
        print("\n--- Réponse Brut du LLM ---")
        print(llm_response)
        print("---------------------------\n")

        raw_code = get_raw_code_block(llm_response)
        code_with_result = raw_code  # pas de sauvetage : si le LLM omet l'assignation, c'est une erreur

        # 3. Fast Path (AST Immédiat)
        potential_clean_code = apply_ast_transformations(code_with_result)

        is_valid = False
        try:
            ast.parse(potential_clean_code)
            is_valid = True
            clean_code = potential_clean_code
        except SyntaxError:
            is_valid = False

        if not is_valid:
            print("⚠️ Code invalide. Nettoyage ligne par ligne activé.")
            lines = code_with_result.split('\n')
            cleaned_lines = []
            for line in lines:
                line_clean = line.rstrip()
                if not line_clean.strip(): continue
                if line_clean.strip().startswith(("import ", "from ")): continue
                if "END SOLUTION" in line_clean: continue
                if line_clean.strip() in ["```", "```python", "<code>", "</code>"]: continue
                
                if not is_valid_and_useful_line(line_clean):
                    continue
                
                cleaned_lines.append(line_clean)
            
            clean_code = "\n".join(cleaned_lines)
            clean_code = fix_unexpected_indent(clean_code)
            
            # On réessaie l'AST clean sur le résultat
            clean_code = apply_ast_transformations(clean_code)

        print(f"--- Code Final ---\n{clean_code}\n------------------")
        return clean_code

    except Exception as e:
        print(f"❌ CRITICAL ERROR in extract_code_and_fix: {e}")
        traceback.print_exc()
        return llm_response

# ==============================================================================
# AUTRES OUTILS
# ==============================================================================


def modify_lib(file_content, new_import_statement):
    try:
        pattern = r'(exec_context\s*=\s*r?""")(.*?)(import\s+numpy\s+as\s+np)(.*?""")'
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
    except Exception as e:
        print(f"❌ CRITICAL ERROR in modify_lib: {e}")
        return ""