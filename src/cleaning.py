import re



def extract_code_and_fix(llm_response):
    print("\n--- Réponse Brut du LLM ---")
    print(llm_response)
    print("---------------------------\n")

    # find the code between the markdown
    pattern = r"```(?:python|Python|code)?\n(.*?)```"
    matches = re.findall(pattern, llm_response, re.DOTALL)
    
    if matches:
        code = max(matches, key=len).strip()
    else:
        code = llm_response.strip()

    # take the imports out of the code 
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
            
        cleaned_lines.append(line)
        
    code = "\n".join(cleaned_lines)

    return code


def ensure_result_assignment(code):
    """
    Heuristique : Le moteur de test s'attend souvent à trouver une variable 'result'.
    Si le LLM renvoie juste 'np.percentile(...)', on rajoute 'result = ' devant.
    """
    if "result =" not in code and "result=" not in code:
        # On prend la dernière ligne non vide
        lines = code.split('\n')
        last_line_idx = -1
        for i in range(len(lines) -1, -1, -1):
            if lines[i].strip():
                last_line_idx = i
                break
        
        if last_line_idx != -1:
            lines[last_line_idx] = "result = " + lines[last_line_idx]
            return "\n".join(lines)
            
    return code