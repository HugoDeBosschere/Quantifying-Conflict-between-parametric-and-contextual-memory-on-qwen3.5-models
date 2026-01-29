import re
import ast
import textwrap
import traceback # Indispensable pour voir les erreurs sans crasher

# ==============================================================================
# 1. OUTILS BAS NIVEAU (INDENTATION & VALIDATION)
# ==============================================================================

def fix_unexpected_indent(code):
    """
    Corrige le cas où la première ligne est collée à gauche, 
    mais la suite est indentée sans raison (escalier).
    """
    try:
        # Check rapide : si c'est déjà valide, on ne touche à rien
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
        # En cas de crash, on renvoie le code tel quel
        return code

    return code

def is_valid_and_useful_line(line):
    try:
        stripped = line.strip()
        
        # 1. On garde tout ce qui a une assignation ou un mot clé
        if "=" in stripped or stripped.startswith(("def ", "class ", "if ", "for ", "while ", "return", "import", "from", "with ")):
            return True

        # --- SAUVETAGE MULTI-LIGNES ---
        if stripped.endswith(","):
            return True

        # 2. Détection des déchets (Arrays, Nombres isolés)
        if re.match(r"^\s*(\[|array|[\d\.-]|nan|inf|\(|$)", stripped):
            try:
                tree = ast.parse(stripped)
                # Si c'est une expression isolée SANS virgule à la fin, on considère que c'est un print inutile
                if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
                    return False
            except SyntaxError:
                return False
                
    except Exception as e:
        print(f"⚠️ Warning: is_valid_and_useful_line failed on '{line}': {e}")
        # Dans le doute, on GARDE la ligne (True) pour ne pas supprimer du code utile
        return True

    return True

# ==============================================================================
# 2. LOGIQUE D'EXTRACTION ET DE SAUVETAGE
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
        # Fallback ultime : on renvoie tout
        return llm_response

def rescue_missing_result(code, full_response):
    try:
        if "result =" not in code and "result=" not in code:
            rescue_pattern = r"^\s*result\s*=.*"
            potential_lines = re.findall(rescue_pattern, full_response, re.MULTILINE)
            
            if potential_lines:
                clean_candidates = [
                    line.strip() for line in potential_lines 
                    if is_valid_and_useful_line(line) and not line.strip().startswith("#")
                ]
                
                best_candidate = clean_candidates[-1] if clean_candidates else potential_lines[0].strip()
                
                print(f"🚑 Sauvetage réussi : '{best_candidate}'")
                return code + "\n" + best_candidate
                
    except Exception as e:
        print(f"⚠️ Warning: rescue_missing_result failed: {e}")
        
    return code

# ==============================================================================
# 3. TRANSFORMATIONS AST (SUPPRESSION IMPORT + RETURN -> RESULT)
# ==============================================================================

def apply_ast_transformations(code):
    """
    1. Supprime les imports (import x, from x import y) au niveau global.
    2. Transforme les 'return x' de niveau 0 en 'result = x'.
    """
    try:
        tree = ast.parse(code)
        new_body = []
        
        for node in tree.body:
            # A. Suppression des Imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue # On ne l'ajoute pas (suppression)

            # B. Transformation du Return (seulement niveau 0)
            elif isinstance(node, ast.Return) and node.value:
                # Création manuelle du noeud d'assignation
                assign = ast.Assign(
                    targets=[ast.Name(id='result', ctx=ast.Store())],
                    value=node.value
                )
                # IMPORTANT : On copie la localisation pour éviter le crash lineno
                ast.copy_location(assign, node)
                new_body.append(assign)

            # C. Conservation du reste
            else:
                new_body.append(node)
        
        tree.body = new_body
        
        # --- FIX CRUCIAL ---
        # Remplit les infos de lignes manquantes pour tous les noeuds créés/bougés
        ast.fix_missing_locations(tree)
        # -------------------
        
        if hasattr(ast, "unparse"):
            return ast.unparse(tree)
            
    except SyntaxError:
        pass # Code non parsable, normal on passe la main
    except Exception as e:
        # Erreur technique (ex: bug AST interne), on log mais on ne plante pas
        print(f"⚠️ Warning: apply_ast_transformations failed: {e}")
        # traceback.print_exc() # Décommenter si besoin de debug intense
    
    return code # On retourne le code original en cas d'échec

# ==============================================================================
# 4. LE PIPELINE PRINCIPAL (L'ENTONNOIR)
# ==============================================================================

def extract_code_and_fix(llm_response):
    # Sécurité maximale : Si tout plante, on renvoie au moins le raw string
    try:
        print("\n--- Réponse Brut du LLM ---")
        print(llm_response)
        print("---------------------------\n")

        # 1. Extraction brute
        raw_code = get_raw_code_block(llm_response)
        
        # 2. Sauvetage éventuel (si 'result' manque)
        code_with_result = rescue_missing_result(raw_code, llm_response)

        # 3. Tentative Fast Path (Nettoyage AST immédiat)
        # On essaie d'enlever les imports tout de suite
        potential_clean_code = apply_ast_transformations(code_with_result)

        # On vérifie si le résultat est valide
        is_valid = False
        try:
            ast.parse(potential_clean_code)
            is_valid = True
            clean_code = potential_clean_code
        except SyntaxError:
            is_valid = False

        if not is_valid:
            print("Code invalide ou nettoyage AST échoué. Nettoyage ligne par ligne activé.")
            
            # 4. Nettoyage Ligne par Ligne (Slow Path / Fallback)
            lines = code_with_result.split('\n')
            cleaned_lines = []
            for line in lines:
                line_clean = line.rstrip()
                
                # Filtres Textuels basiques
                if not line_clean.strip(): continue
                if line_clean.strip().startswith(("import ", "from ")): continue
                if "END SOLUTION" in line_clean: continue
                if line_clean.strip() in ["```", "```python", "<code>", "</code>"]: continue
                
                # Filtre AST (Le Gardien)
                if not is_valid_and_useful_line(line_clean):
                    continue
                
                cleaned_lines.append(line_clean)
            
            clean_code = "\n".join(cleaned_lines)
            
            # Fix indentation
            clean_code = fix_unexpected_indent(clean_code)
            
            # 5. Transformation Sémantique Finale (Return -> Result)
            # On réessaie l'AST une dernière fois sur le code propre
            clean_code = apply_ast_transformations(clean_code)

        print(f"--- Code Final ---\n{clean_code}\n------------------")
        return clean_code

    except Exception as e:
        print(f"CRITICAL ERROR in extract_code_and_fix: {e}")
        traceback.print_exc()
        # En cas de catastrophe nucléaire, on renvoie une réponse qui ne fera pas planter l'exec (mais échouera le test)
        return llm_response # Ou "" si tu préfères

# ==============================================================================
# AUTRES OUTILS
# ==============================================================================



def ensure_result_assignment(code):
    try:
        if "result =" not in code and "result=" not in code:
            lines = code.split('\n')
            last_line_idx = -1
            
            for i in range(len(lines) -1, -1, -1):
                if lines[i].strip() and not lines[i].strip().startswith("#"):
                    last_line_idx = i
                    break
            
            if last_line_idx != -1:
                line = lines[last_line_idx]
                indent = line[:len(line) - len(line.lstrip())]
                content = line.lstrip()
                lines[last_line_idx] = f"{indent}result = {content}"
                return "\n".join(lines)
    except Exception as e:
        print(f"⚠️ Warning: ensure_result_assignment failed: {e}")
            
    return code



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