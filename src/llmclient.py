import requests
import os

class LLMClient:
    def __init__(self, config, model_name, doc_name, mode="injection"):
        """
        config: dict containing "llm", "new_lib_injection" and "real_lib" sections
        mode: "injection" (counterfactual lib) or "control" (real lib)
        """
        lib_key = "new_lib_injection" if mode == "injection" else "real_lib"
        lib_config = config[lib_key]

        self.model_name = model_name
        self.mode = mode
        self.api_url = config["llm"]["api_url"]
        self.temperature = config["llm"]["temperature"]
        self.num_ctx = config["llm"].get("num_ctx", 20000)

        self.system_prompt = lib_config["system_prompt"]
        self.custom_lib_path = lib_config.get("custom_lib_path")
        self.lib_name = lib_config["name"]

        self.documentation = self._load_doc(lib_config, doc_name)
        self.model_metadata = self._build_metadata(doc_name)

    def _build_metadata(self, doc_name):
        return {
            "model_name": self.model_name,
            "doc_name": doc_name,
            "mode": self.mode,
            "temperature": self.temperature
        }

    def _load_doc(self, lib_config, doc_name):
        doc_info = lib_config.get("documentation", {}).get(doc_name, {})
        intro = doc_info.get("intro", "")
        doc_path = doc_info.get("path", "")

        if not doc_path:
            return intro

        try:
            if os.path.exists(doc_path):
                with open(doc_path, mode="r", encoding="utf-8") as f:
                    return intro + f.read()
        except FileNotFoundError:
            print("THE DOCUMENTATION was not found")

        return intro

    def warm_up(self):
        """
        Envoie une requête minimale pour que Ollama charge le modèle en VRAM
        avant le premier vrai task. Évite que le chargement (~2 min) bloque pendant le premier task.
        """
        print(f"[Warm-up] Chargement du modèle {self.model_name} sur Ollama...")
        _, _ = self.query_llm("Say OK.")
        print(f"[Warm-up] Modèle {self.model_name} prêt.")

    def query_llm(self, prompt_text):
        """
        Envoie le prompt au LLM via Ollama.
        Le system prompt est envoyé dans le champ 'system' (pas dupliqué dans 'prompt').
        La documentation est incluse dans le prompt uniquement si elle existe.
        """
        print(f"Interrogation de {self.model_name}...")

        if self.documentation:
            full_prompt = f"{self.documentation}\n\n{prompt_text}"
        else:
            full_prompt = prompt_text

        count_token = self.get_token_count(full_prompt)

        try:
            response = requests.post(f"{self.api_url}/generate", json={
                "model": self.model_name,
                "system": self.system_prompt,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_ctx": self.num_ctx
                }
            })
            response.raise_for_status()
            return response.json()['response'], count_token
        except requests.exceptions.HTTPError as e:
            body = (response.text[:500] if getattr(response, "text", None) else "") or ""
            code = getattr(response, "status_code", "?")
            print(f"API Error [{self.model_name}] HTTP {code}: {e}")
            if body:
                print(f"  Response: {body}")
            return None, 0
        except Exception as e:
            print(f"API Error [{self.model_name}]: {e}")
            return None, 0

    def get_token_count(self, prompt):
        try:
            response = requests.post(f"{self.api_url}/tokenize", json={
                "model": self.model_name,
                "content": prompt
            })
            if response.status_code == 200:
                tokens = response.json().get('tokens', [])
                return len(tokens)
            else:
                print(f"Token count error: {response.text}")
                return len(prompt) // 3
        except Exception as e:
            print(f"Token count exception: {e}")
            return 0
