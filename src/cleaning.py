import re
import textwrap



def extract_code_and_fix(llm_response):
    print("\n--- Réponse Brut du LLM ---")
    print(llm_response)
    print("---------------------------\n")

    pattern = r"```(?:python|markdown|Markdown|Python|code)?\n(.*?)```|(.*?)</code>"
    matches = re.findall(pattern, llm_response, re.DOTALL)
    
    code = ""
    if matches:
        valid_matches = [m[0] or m[1] for m in matches if m[0] or m[1]]
        if valid_matches:
            code = max(valid_matches, key=len)
        else:
            code = llm_response
    else:
        code = llm_response

    code = code.strip("\n")
    code = textwrap.dedent(code)

    lines = code.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line_clean = line.rstrip() 
        if not line_clean.strip():
            continue

        if line_clean.strip().startswith(("import ", "from ")):
            continue
            
        cleaned_lines.append(line_clean)
        
    code = "\n".join(cleaned_lines)
    return code


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
