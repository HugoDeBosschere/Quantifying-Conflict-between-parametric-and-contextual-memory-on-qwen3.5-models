import re
import ast
import textwrap
import traceback

# ==============================================================================
# 1. LOGIQUE D'EXTRACTION
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


def apply_ast_transformations(code):
    """
    Nettoyage via AST.
    A. Supprime les imports.
    """
    try:
        tree = ast.parse(code)
        new_body = []
        removed_any = False

        for node in tree.body:
            # A. Suppression des Imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                removed_any = True
                continue
            else:
                new_body.append(node)

        # Si aucun import supprimé, retourner le code original sans ast.unparse
        # (ast.unparse normalise le code, ex: "a, b = x" -> "(a, b) = x")
        if not removed_any:
            return code

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
            clean_code = potential_clean_code

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
        # Format ds1000 : exec_context = r"""...import numpy as np..."""
        pattern_triple = r'(exec_context\s*=\s*r?""")(.*?)(import\s+numpy\s+as\s+np)(.*?""")'
        if re.search(pattern_triple, file_content, flags=re.DOTALL):
            return re.sub(
                pattern_triple,
                rf'\1\2{new_import_statement}\4',
                file_content,
                flags=re.DOTALL
            )

        # Format NumpyEval : exec_context = "import numpy as np\n..."
        pattern_single = r'(exec_context\s*=\s*")(import\s+numpy\s+as\s+np)(\\n)'
        if re.search(pattern_single, file_content):
            return re.sub(
                pattern_single,
                rf'\g<1>{new_import_statement}\3',
                file_content
            )

        print("Aucune occurrence trouvée dans exec_context.")
        return ""
    except Exception as e:
        print(f"❌ CRITICAL ERROR in modify_lib: {e}")
        return ""