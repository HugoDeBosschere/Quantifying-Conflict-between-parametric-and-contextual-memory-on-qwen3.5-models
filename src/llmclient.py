import requests
import os
import time
import sys

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
        self.temperature_config = config["llm"]["temperature"]
        self.temperature = self._resolve_temperature(self.temperature_config)
        self.num_ctx = config["llm"].get("num_ctx", 20000)
        self.seed = self._resolve_seed(config["llm"].get("seed"))

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
            "temperature": self.temperature,
            "seed": self.seed,
        }

    def _resolve_temperature(self, temperature_cfg):
        """Résout la température à utiliser pour `self.model_name`."""
        if isinstance(temperature_cfg, (int, float)):
            return float(temperature_cfg)
        if isinstance(temperature_cfg, dict):
            # Priorité : clé exacte du modèle, puis `default`.
            if self.model_name in temperature_cfg:
                return float(temperature_cfg[self.model_name])
            if "default" in temperature_cfg:
                return float(temperature_cfg["default"])
            # Fallback : première valeur numérique trouvée (évite crash si config incomplète)
            for v in temperature_cfg.values():
                if isinstance(v, (int, float)):
                    return float(v)
            return 0.0
        # Type inattendu : on fallback à 0
        return 0.0

    def _resolve_seed(self, seed_cfg):
        """Résout la seed pour `self.model_name`, ou None si non configurée."""
        if seed_cfg is None:
            return None
        if isinstance(seed_cfg, bool):
            return None
        if isinstance(seed_cfg, (int, float)):
            return int(seed_cfg)
        if isinstance(seed_cfg, dict):
            if self.model_name in seed_cfg:
                v = seed_cfg[self.model_name]
                return None if v is None else int(v)
            if "default" in seed_cfg:
                v = seed_cfg["default"]
                return None if v is None else int(v)
            return None
        return None

    def _ollama_options(self):
        opts = {"temperature": self.temperature, "num_ctx": self.num_ctx}
        if self.seed is not None:
            opts["seed"] = self.seed
        return opts

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

    def _wait_for_server(self, timeout_sec=120, poll_interval=5):
        """Attend qu'Ollama réponde (GET sur l'API). Retourne True si prêt, False si timeout."""
        base_url = self.api_url.rstrip("/").replace("/api", "") or "http://localhost:11434"
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            try:
                r = requests.get(f"{base_url}/api/tags", timeout=5)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            print(f"[Warm-up] Ollama pas encore prêt, nouvel essai dans {poll_interval}s...")
            time.sleep(poll_interval)
        return False

    def warm_up(self, max_retries=10, retry_delay_sec=30, first_request_timeout=300):
        """
        Attend qu'Ollama soit prêt, puis envoie une requête minimale pour charger le modèle.
        En cas d'échec après max_retries, quitte le processus (sys.exit(1)).
        """
        print(f"[Warm-up] Attente du serveur Ollama...")
        if not self._wait_for_server(timeout_sec=120):
            print("[Warm-up] ERREUR: Ollama n'a pas répondu à temps. Arrêt.")
            sys.exit(1)
        print(f"[Warm-up] Serveur OK. Chargement du modèle {self.model_name}...")

        for attempt in range(1, max_retries + 1):
            response, _ = self._generate_minimal(timeout=first_request_timeout if attempt == 1 else 60)
            if response is not None:
                print(f"[Warm-up] Modèle {self.model_name} prêt.")
                return
            print(f"[Warm-up] Tentative {attempt}/{max_retries} échouée, nouvel essai dans {retry_delay_sec}s...")
            if attempt < max_retries:
                time.sleep(retry_delay_sec)

        print(f"[Warm-up] ERREUR: Le modèle {self.model_name} n'a pas répondu après {max_retries} tentatives. Arrêt.")
        sys.exit(1)

    def _generate_minimal(self, timeout=300):
        """
        Une seule requête /api/generate avec prompt minimal (pas de tokenize avant).
        Retourne (response_text, 0) ou (None, 0). Utilisé pour le warm-up.
        """
        try:
            response = requests.post(
                f"{self.api_url}/generate",
                json={
                    "model": self.model_name,
                    "system": self.system_prompt,
                    "prompt": "Say OK.",
                    "stream": False,
                    "options": self._ollama_options(),
                },
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json().get("response"), 0
        except requests.exceptions.Timeout:
            print(f"API Error [{self.model_name}] Timeout après {timeout}s")
            return None, 0
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

        try:
            # Timeout dur : 600s pour éviter de bloquer un job entier sur une seule requête
            response = requests.post(
                f"{self.api_url}/generate",
                json={
                    "model": self.model_name,
                    "system": self.system_prompt,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": self._ollama_options(),
                },
                timeout=600,
            )
            response.raise_for_status()
            data = response.json()
            # Token count : préférer la réponse Ollama (prompt_eval_count), sinon estimation
            count_token = data.get("prompt_eval_count")
            if count_token is None:
                count_token = len(full_prompt) // 3

            return data["response"], count_token
        except requests.exceptions.Timeout:
            print(f"API Error [{self.model_name}] Timeout après 600s")
            return None, 0
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
        """
        Comptage de tokens via /api/tokenize (Ollama récent).
        Utilisé uniquement en secours si besoin ; le flux principal utilise
        prompt_eval_count renvoyé par /api/generate.
        """
        try:
            response = requests.post(f"{self.api_url}/tokenize", json={
                "model": self.model_name,
                "content": prompt
            })
            if response.status_code == 200:
                tokens = response.json().get("tokens", [])
                return len(tokens)
            return len(prompt) // 3
        except Exception:
            return len(prompt) // 3
