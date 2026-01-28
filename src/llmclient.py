import requests

class LLMClient:
    def __init__(self, config):
        """
        Initialize the LLM client with support for both local CUDA inference and Ollama API.
        config: dict containing "llm" and "new_lib_injection" sections
        """
        self.model_name = config["llm"]["model"]
        self.api_url = config["llm"].get("api_url", "http://127.0.0.1:11434/api/generate")
        self.temperature = config["llm"]["temperature"]
        self.system_prompt = config["new_lib_injection"]["system_prompt"]
        self.context_prompt = config["new_lib_injection"]["context_prompt"]
        self.custom_lib_path = config["new_lib_injection"]["custom_lib_path"]

        # CUDA inference settings
        self.use_local_inference = config["llm"].get("use_local_inference", False)
        self.device = config["llm"].get("device", "cuda")
        self.max_new_tokens = config["llm"].get("max_new_tokens", 1024)

        # Initialize local model if using local inference
        self.model = None
        self.tokenizer = None

        if self.use_local_inference:
            self._init_local_model()

    def _init_local_model(self):
        """Initialize the local model with CUDA support."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Determine device
        if self.device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
            print(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
        elif self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Auto-detected device: {self.device}")
        else:
            if self.device == "cuda" and not torch.cuda.is_available():
                print("Warning: CUDA requested but not available. Falling back to CPU.")
                self.device = "cpu"

        print(f"Loading model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with appropriate settings for GPU
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model = self.model.to(self.device)

        self.model.eval()
        print(f"Model loaded successfully on {self.device}")

    def _query_local(self, full_prompt):
        """Run inference locally using the loaded model on GPU."""
        import torch

        # Format prompt for Llama-3 Instruct
        messages = [
            {"role": "user", "content": full_prompt}
        ]

        # Use chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = full_prompt

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response

    def _query_api(self, full_prompt):
        """Query the Ollama API (original implementation)."""
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

    def query_llm(self, prompt_text, context_prompt_type="description"):
        """
        Query the LLM with the given prompt.
        Uses local CUDA inference if enabled, otherwise falls back to Ollama API.
        """
        full_prompt = f"{self.system_prompt}\n\n{self.context_prompt[context_prompt_type]}\n\n{prompt_text}"

        if self.use_local_inference and self.model is not None:
            try:
                return self._query_local(full_prompt)
            except Exception as e:
                print(f"Local inference error: {e}")
                return None
        else:
            return self._query_api(full_prompt)
