

from typing import Dict, Tuple, Optional
import json
import re
import torch
from model_utils import TogetherPipeline
from prompts import get_umbrella_prompt



def get_relevance_score_baseline(prompt: str, pipeline, system_message: str) -> str:
    """
    Get model response for a given prompt, handling both Together AI and standard pipelines.
    
    The function includes first-run logging to help with debugging and verification.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    
    # First-run logging for debugging
    if not hasattr(get_relevance_score_baseline, "called"):
        get_relevance_score_baseline.called = True
        print("Initial messages for verification:")
        print(messages)

    # Handle Together AI models
    if isinstance(pipeline, TogetherPipeline):
        if not hasattr(get_relevance_score_baseline, "output_from_together"):
            get_relevance_score_baseline.output_from_together = True
            print("Using Together AI model for inference")
        
        outputs = pipeline(messages)
        output = outputs[0]["generated_text"]
    
    # Handle standard pipeline models
    else:
        terminators = [t for t in [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ] if t is not None]
        
        # Process chat template if available
        if hasattr(pipeline.tokenizer, "apply_chat_template"):
            if hasattr(pipeline.tokenizer, 'chat_template') and pipeline.tokenizer.chat_template is not None:
                prompt = pipeline.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                if not hasattr(get_relevance_score_baseline, "warning"):
                    get_relevance_score_baseline.warning = True
                    print("Warning: No chat template available, using only the prompt (and not the system message).")
                prompt = f"{prompt}"
        else:
            prompt = f"{prompt}"

        # Generate model output
        outputs = pipeline(
            prompt,
            max_new_tokens=100,
            eos_token_id=terminators,
            pad_token_id=128009,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        output = outputs[0]["generated_text"]

        # Return generated text without the prompt if chat template was used, otherwise return full text
        if hasattr(pipeline.tokenizer, 'chat_template') and pipeline.tokenizer.chat_template is not None:
            output =  outputs[0]["generated_text"][len(prompt):]
        else:
            output = outputs[0]["generated_text"]
    if not hasattr(get_relevance_score_baseline, "print_one_output"):
        get_relevance_score_baseline.print_one_output = True
        print(f"sample output: {output}")   
    return output




def find_first_number(text: str) -> Optional[int]:
    """
    Extract the first valid UMBRELA relevance score (0-3) from text.
    
    Args:
        text (str): The text to search for a score
        
    Returns:
        Optional[int]: First valid score found, or 0 if none found
    """
    # Try patterns in order of most specific to most general
    
    # 1. Try "## final score: X" format first
    final_score_match = re.search(r'(?:^|\n)##\s*final score:\s*([0-3])', text, re.IGNORECASE | re.MULTILINE)
    if final_score_match:
        return int(final_score_match.group(1))
    
    # 2. Try "O: X" format next
    o_score_match = re.search(r'O:\s*([0-3])', text)
    if o_score_match:
        return int(o_score_match.group(1))
    
    # 3. Fall back to any valid score number
    general_match = re.search(r'\b[0-3]\b', text)
    if general_match:
        return int(general_match.group())
    
    return 0  # Default score if no valid score found



def grade_each_pq_pair(query: str, passage: str, pipeline, 
                  log_file_path: str, system_message: str,
                  qidx: str, docidx: str, mode: str ) -> Tuple[Optional[int], Dict[str, int]]:
    """
    Grade the relevance of a passage-query pair using UMBRELA methodology.
    
    Args:
        query (str): The search query
        passage (str): The passage to evaluate
        pipeline: The model pipeline (Together AI or standard)
        log_file_path (str): Path to log results
        system_message (str): System message (empty for UMBRELA)
        qidx (str): Query ID
        docidx (str): Document ID
        mode (str): UMBRELA prompt mode (default: "zeroshot_bing")
        
    Returns:
        Tuple[Optional[int], Dict[str, int]]: Final relevance score and scoring log
    """
    # Generate UMBRELA prompt
    prompt = get_umbrella_prompt(query=query, passage=passage, mode=mode)
    
    # Get model response
    # print(prompt)
    llms_output = get_relevance_score_baseline(prompt, pipeline, system_message)
    # Extract score using regex pattern specific to UMBRELA format
    score_pattern = r'##final score:\s*([0-3])'
    match = re.search(score_pattern, llms_output)
    
    if match:
        final_score = int(match.group(1))
    else:
        # Fallback to finding first number if UMBRELA format isn't found
        final_score = find_first_number(llms_output)
    
    # Ensure score is within valid range
    if final_score not in [0, 1, 2, 3]:
        final_score = 0  # Default to 0 for invalid scores
    
    # Log results for analysis
    scoring_log = {
        "prompt_mode":mode,
        "qidx": qidx,
        "docidx": docidx,
        "query": query,
        "passage": passage,
        "LLMs_output": llms_output,
        "final_relevance_score": final_score,
        "prompt_mode": mode
    }

    # Append to log file
    with open(log_file_path, "a") as f:
        f.write(json.dumps(scoring_log) + "\n")  # Write each log on new line

    return final_score, scoring_log