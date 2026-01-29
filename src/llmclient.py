import requests
import os

class LLMClient:
    def __init__(self, config, model_name, doc_name):
        """
        Initialize the LLM client with support for both local CUDA inference and Ollama API.
        config: dict containing "llm" and "new_lib_injection" sections
        """
        self.model_name = model_name
        self.api_url = config["llm"]["api_url"]
        self.temperature = config["llm"]["temperature"]
        self.system_prompt = config["new_lib_injection"]["system_prompt"]
        self.custom_lib_path = config["new_lib_injection"]["custom_lib_path"]
        self.new_lib_name = config["new_lib_injection"]["name"]
        self.documentation = self.load_doc(config, doc_name)
        self.model_metadata = self.load_model_metadata(doc_name)
    

    def load_model_metadata(self, doc_name) :
        
        return {"model_name" : self.model_name,
                "doc_name" : doc_name,
                "temperature" : self.temperature}


    def load_doc(self, config, doc_name) :
        doc = []
        doc.append(config.get("new_lib_injection", {}).get("documentation", {}).get(doc_name, {}).get("intro", ""))
        doc_path = config.get("new_lib_injection", {}).get("documentation", {}).get(doc_name, {}).get("path", "")
        
        try :
            if os.path.exists(doc_path) :
                with open(doc_path, mode = "r", encoding="utf-8") as f :
                    content = f.read()
                    doc.append(content)
                    return ''.join(doc)
        except FileNotFoundError:
            print("THE DOCUMENTATION was not found")
            return ""


    def query_llm(self, prompt_text, new_lib_name):
        print(f"Interrogation de {self.model_name}...")
        if self.new_lib_name == new_lib_name :
            full_prompt = f"{self.system_prompt}\n\n{self.documentation}\n\n{prompt_text}"
        else :
            full_prompt = f"{self.system_prompt}\n\n{prompt_text}"

        try:
            response = requests.post(self.api_url, json={
                "model": self.model_name,
                "system":self.system_prompt,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                }
            })
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            print(f"API Error: {e}")
            return None

