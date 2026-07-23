"""
Inference Pipeline — loads Llama-3 Base + custom LoRA adapters using HuggingFace.
"""

import os
import time
import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
from peft import PeftModel

class DevAIPipeline:
    def __init__(
        self,
        base_model_id: str = "unsloth/llama-3-8b-Instruct-bnb-4bit",
        lora_path: str = "data/checkpoints/dev-ai-llama3-lora",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ):
        self.base_model_id = base_model_id
        self.lora_path = lora_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        # Determine best device for Mac M1 or GPU
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        self.model = None
        self.tokenizer = None
        self._model_ready = False
        
        # Mock retriever to avoid breaking server.py
        class MockRetriever:
            chunks = []
            idf = []
        self.retriever = MockRetriever()

    def load(self) -> bool:
        print(f"[Pipeline] Loading Tokenizer: {self.base_model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)

        if not os.path.exists(self.lora_path):
            print(f"[Pipeline] WARNING: LoRA adapters not found at {self.lora_path}.")
            print("  → Upload to Kaggle and run kaggle_train.ipynb to train, then download the adapters.")
            return False

        print(f"[Pipeline] Loading Base Model: {self.base_model_id} on {self.device}...")
        # Note: 4-bit loading doesn't work on MPS natively yet, so we load in standard precision on Mac
        # and rely on the unified memory (16GB RAM can fit an 8B model in fp16).
        torch_dtype = torch.float16 if self.device in ["mps", "cuda"] else torch.float32
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=torch_dtype,
            device_map=self.device
        )

        print(f"[Pipeline] Loading LoRA Adapters from {self.lora_path}...")
        self.model = PeftModel.from_pretrained(base_model, self.lora_path)
        self.model.eval()
        self._model_ready = True
        
        self.hf_pipe = hf_pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map=self.device
        )
        print("[Pipeline] Status: READY")
        return True

    def query(self, question: str, topic_hint: Optional[str] = None) -> dict:
        start = time.perf_counter()
        
        if not self._model_ready:
            return {
                "answer": "Model not loaded. Please train and download the LoRA adapters first.",
                "topic": topic_hint or "general",
                "confidence": 0.0,
                "retrieval_used": False,
                "retrieved_chunks": [],
                "tokens_generated": 0,
                "time_seconds": 0.0,
                "tokens_per_second": 0.0,
                "model_used": False,
            }

        messages = [
            {"role": "system", "content": f"You are a master developer AI assistant. Focus on {topic_hint if topic_hint else 'programming'}."},
            {"role": "user", "content": question}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        outputs = self.hf_pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        generated_text = outputs[0]["generated_text"][len(prompt):].strip()
        elapsed = time.perf_counter() - start
        # Approximate tokens
        n_gen = len(generated_text) // 4
        
        return {
            "answer": generated_text,
            "topic": topic_hint or "general",
            "confidence": 0.95,
            "retrieval_used": False,
            "retrieved_chunks": [],
            "tokens_generated": n_gen,
            "time_seconds": round(elapsed, 3),
            "tokens_per_second": round(n_gen / elapsed if elapsed > 0 else 0, 1),
            "model_used": True,
        }

    @property
    def is_ready(self) -> bool:
        return self._model_ready
