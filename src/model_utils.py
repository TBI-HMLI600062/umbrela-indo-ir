"""
Model Utilities Module
---------------------
Handles model loading and initialization for different types of language models:
- Together AI models (API-based)
- Flan-T5 models (sequence-to-sequence)
- Causal language models (like LLaMA)

All models are configured for optimal performance with appropriate data types
and device mapping.
"""

from typing import Optional
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
from typing import *



class TogetherPipeline:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_key = os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY environment variable is not set.")
        self.client = Together(api_key=self.api_key)
    
    def __call__(self, messages: List[Dict], max_new_tokens=100, **kwargs):
        # Use Together API to generate responses
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            do_sample=False,
            temperature=None,
            top_p=None,
            
        )
        return [{"generated_text": response.choices[0].message.content}]




def get_model_baseline(name_or_path_to_model: str, use_together: bool = False, use_4bit: bool = False):
    """
    Load and configure a language model for text generation.
    
    This function handles three types of models:
    1. Together AI models accessed through API
    2. Flan-T5 models using sequence-to-sequence architecture
    3. Standard causal language models (like LLaMA)
    
    Args:
        name_or_path_to_model: Model identifier or path (e.g., "meta-llama/Llama-3-70b")
        use_together: Whether to use Together AI API
        
    Returns:
        Configured pipeline ready for text generation
        
    Notes:
        - Models are loaded with bfloat16 precision for efficiency
        - Device mapping is automatic based on available hardware
        - Flan-T5 models use text2text-generation pipeline
        - Other models use standard text-generation pipeline
    """
    
    # Together AI API-based model
    if use_together:
        return TogetherPipeline(model_name=name_or_path_to_model)
    
    # Flan-T5 sequence-to-sequence model
    elif "flan-t5" in name_or_path_to_model.lower():
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
        tokenizer = AutoTokenizer.from_pretrained(name_or_path_to_model)

        if use_4bit:
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
        else:
            model = AutoModelForCausalLM.from_pretrained(
                name_or_path_to_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

        # Reset max_length in generation_config to avoid conflict with max_new_tokens
        if hasattr(model, "generation_config"):
            model.generation_config.max_length = None

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