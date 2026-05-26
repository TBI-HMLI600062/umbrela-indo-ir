"""
Model Utilities Module
---------------------
Handles model loading and initialization for different types of language models:
- Together AI models (API-based)
- OpenAI-compatible models: ChatGPT (OpenAI) and DeepSeek
- Flan-T5 models (sequence-to-sequence)
- Causal language models (like LLaMA)

All models are configured for optimal performance with appropriate data types
and device mapping.
"""

from typing import Optional, List, Dict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
    BitsAndBytesConfig
)
import torch
from together import Together
import together
import os


class APIBasePipeline:
    """Base class for all API-based (non-local) pipelines."""
    pass


class TogetherPipeline(APIBasePipeline):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_key = os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY environment variable is not set.")
        self.client = Together(api_key=self.api_key)

    def __call__(self, messages: List[Dict], max_new_tokens=100, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        return [{"generated_text": response.choices[0].message.content}]


class OpenAIPipeline(APIBasePipeline):
    """OpenAI-compatible pipeline — handles ChatGPT (OpenAI) and DeepSeek."""

    _PROVIDER_CONFIG = {
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "api_key_env": "OPENAI_API_KEY",
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com",
            "api_key_env": "DEEPSEEK_API_KEY",
        },
    }

    def __init__(self, model_name: str, provider: str):
        from openai import OpenAI
        cfg = self._PROVIDER_CONFIG[provider]
        api_key = os.getenv(cfg["api_key_env"])
        if not api_key:
            raise ValueError(f"{cfg['api_key_env']} environment variable is not set.")
        self.model_name = model_name
        self.client = OpenAI(base_url=cfg["base_url"], api_key=api_key)

    def __call__(self, messages: List[Dict], max_new_tokens=512, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0,
        )
        return [{"generated_text": response.choices[0].message.content}]




def get_model_baseline(name_or_path_to_model: str, use_together: bool = False,
                       provider: str = "hf"):
    """
    Load and configure a language model for text generation.

    Args:
        name_or_path_to_model: Model ID or local path
        use_together: Legacy flag — equivalent to provider="together"
        provider: "hf" (default) | "together" | "openai" | "deepseek"

    Returns:
        Configured pipeline ready for text generation
    """
    # Legacy compat
    if use_together:
        provider = "together"

    if provider == "together":
        return TogetherPipeline(model_name=name_or_path_to_model)

    if provider in ("openai", "deepseek"):
        return OpenAIPipeline(model_name=name_or_path_to_model, provider=provider)

    # Local HF model — Flan-T5 sequence-to-sequence
    if "flan-t5" in name_or_path_to_model.lower():
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(name_or_path_to_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            name_or_path_to_model,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for efficient memory usage
            device_map="auto"  # Automatically handle device placement
        )
        
        # Create sequence-to-sequence pipeline
        return pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer
        )
    
    # Standard causal language model
    else:
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(name_or_path_to_model)
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path_to_model,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Create text generation pipeline
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )

def get_model_lora(base_model_id: str, lora_adapter_path: str):
    """
    Load a base model with 4-bit quantization and merge/attach a LoRA adapter.

    Returns a HuggingFace pipeline compatible with the existing UMBRELA
    inference flow (grade_each_pq_pair / get_relevance_score_baseline).
    """
    from peft import PeftModel
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    model = model.merge_and_unload()

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )


# Note: The quantized model loading function below is commented out but preserved
# for potential future use. It demonstrates how to load models with 4-bit quantization
# for memory-efficient inference.

"""
def get_model_quantized(name_or_path_to_model: str) -> Tuple:
    tokenizer = AutoTokenizer.from_pretrained(name_or_path_to_model)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path_to_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    return model, tokenizer
"""