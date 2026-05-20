from pathlib import Path

def get_umbrella_prompt(query: str, passage: str, mode: str) -> str:
    """
    Generates a prompt for evaluating passage relevance to a query.
    
    Args:
        query (str): The search query to evaluate against
        passage (str): The passage to evaluate
        mode (str): The prompt mode to use. Options:
            - "zeroshot_bing": Zero-shot Bing-style prompt
            - "zeroshot_basic": Zero-shot basic prompt
            - "fewshot_bing": Few-shot Bing-style prompt
            - "fewshot_basic": Few-shot basic prompt
            - "zeroshot_bing_strict": Strict zero-shot Bing-style prompt
        
    Returns:
        str: Formatted prompt for relevance evaluation
        
    Score meanings:
        0: No relevance
        1: Related but doesn't answer
        2: Contains answer but unclear/with extra info
        3: Dedicated, exact answer
        
    Raises:
        ValueError: If mode is not one of the supported options
    """
    valid_modes = [
        "zeroshot_bing",
        "zeroshot_basic",
        "fewshot_bing",
        "fewshot_basic",
        "zeroshot_bing_strict",
    ]
    
    if mode not in valid_modes:
        raise ValueError(f"Mode must be one of {valid_modes}")
    
    file_mapping = {
        "zeroshot_bing": "qrel_zeroshot_bing.txt",
        "zeroshot_basic": "qrel_zeroshot_basic.txt",
        "fewshot_bing": "qrel_fewshot_bing.txt",
        "fewshot_basic": "qrel_fewshot_basic.txt",
        "zeroshot_bing_strict": "qrel_zeroshot_bing_strict.txt",
    }
    
    file_name = file_mapping[mode]
    template_path = Path(__file__).parent.parent / "prompts" / file_name
    
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template file not found: {template_path}")
    
    with open(template_path, "r") as f:
        prompt_template = f.read()
    
    return prompt_template.format(query=query, passage=passage)
