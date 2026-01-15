import requests

class LLMClient:
    def __init__(self, config):
        """
        On injecte la config à l'initialisation.
        config: dict contenant la section "llm" et "new_lib_injection"
        """
        self.model_name = config["llm"]["model"]
        self.api_url = config["llm"]["api_url"]
        self.temperature = config["llm"]["temperature"]
        self.system_prompt = config["new_lib_injection"]["system_prompt"]
        self.context_prompt = config["new_lib_injection"]["context_prompt"]

    def query_llm(self, prompt_text, context_prompt_type="description"):
        print(f"Interrogation de {self.model_name}...")
        
        full_prompt = f"{self.system_prompt}\n\n{self.context_prompt[context_prompt_type]}\n\n{prompt_text}"
        
        try:
            response = requests.post(self.api_url, json={
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                }
            })
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            print(f"Erreur API : {e}")
            return None

