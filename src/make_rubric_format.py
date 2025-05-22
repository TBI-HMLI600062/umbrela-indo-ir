import json
import os
import gzip
from typing import Dict, Optional
from pathlib import Path



def make_qrel_dict(qrel_file_path: str) -> Dict:
    """Load qrel file into a dictionary for ground truth lookup."""
    qrel_dict = {}
    with open(qrel_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:  # qid 0 docid score format
                qid, _, docid, score = parts
                try:
                    qrel_dict[(qid, docid)] = int(score)
                except ValueError:
                    print(f"Warning: Invalid score in qrel file: {score}")
    return qrel_dict


def generate_umbrella_json_line(query_id: str, paragraph_id: str, text: str, 
                              query_text: str, ground_truth_relevance_label: int,
                              model_output: str, final_score: int, mode: str,
                              model_name: str,
                              passage_to_msmarco: Optional[Dict] = None, 
                              qidtomsmarcoqids: Optional[Dict] = None) -> str:
    """Generate JSON line for UMBRELA format output."""
    json_line = [
        str(qidtomsmarcoqids[query_id]) if qidtomsmarcoqids else str(query_id),
        [{
            "paragraph_id": str(passage_to_msmarco[paragraph_id]) if passage_to_msmarco else str(paragraph_id),
            "text": text,
            "paragraph": "",
            "paragraph_data": {
                "judgments": [{
                    "paragraphId": str(passage_to_msmarco[paragraph_id]) if passage_to_msmarco else str(paragraph_id),
                    "query": str(qidtomsmarcoqids[query_id]) if qidtomsmarcoqids else str(query_id),
                    "relevance": ground_truth_relevance_label,
                    "titleQuery": query_text
                }],
                "rankings": []
            },
            "grades": [{
                "correctAnswered": True,
                "answer": model_output,  # Changed to use raw model output
                "llm": model_name,
                "llm_options": {},
                "prompt_info": {
                    "prompt_class": "umbrella",
                    "prompt_style": mode,
                    "context_first": False,
                    "check_unanswerable": False,
                    "check_answer_key": False,
                    "is_self_rated": True
                },
                "self_ratings": final_score,
                "prompt_type": "direct_grading",
            }]
        }]
    ]
    return json.dumps(json_line)

def make_mapping_dict(doc_mapping_path: str) -> Dict:
    """Load mapping dictionary from file."""
    mapping = {}
    with open(doc_mapping_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                key, value = parts
                mapping[value] = key
    return mapping



def process_log_to_rubric(input_file: str, output_file: str, qrel_file_path: str,
                         is_dl23: bool = False, doc_mapping_path: Optional[str] = "./data/dl2023/docid_to_docidx.txt",
                         query_mapping_path: Optional[str] = "./data/dl2023/qid_to_qidx.txt",
                         model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"):
    '''
    Converts a JSONL log file of LLM relevance judgments to UMBRELA rubric format.

    This function processes a JSONL log file containing relevance judgments from a language model,
    optionally maps query and document IDs to MS MARCO IDs for the TREC DL23 dataset, and generates
    a gzip-compressed JSONL file in the UMBRELA rubric format. The output is compatible with the
    Autograder Workbench (https://github.com/laura-dietz/rubric-internal) for trec style evaluation, 
    such as computing metrics like nDCG or MAP, and leaderboard corelation.

    See Also:
        - Autograder Workbench: https://github.com/laura-dietz/rubric-internal
        - TREC Deep Learning Track: https://trec.nist.gov/tracks.html

        
    input_file: Path to JSONL log file with keys: qidx, docidx, passage, query, LLMs_output,final_relevance_score, prompt_mode.
    output_file: Path to output gzip-compressed JSONL file.
    qrel_file_path: Path to qrel file (format: qid 0 docid score).
    is_dl23: If True, maps IDs to MS MARCO IDs for TREC DL23. Defaults to False.
    doc_mapping_path: Path to document ID mapping file for TREC DL23. Defaults to "./data/dl2023/docid_to_docidx.txt".
    query_mapping_path: Path to query ID mapping file for TREC DL23. Defaults to "./data/dl2023/qid_to_qidx.txt".
    model_name: Name of the LLM used. Defaults to "meta-llama/Llama-3.3-70B-Instruct-Turbo".    
    '''
    # Load ground truth qrels
    qrel_dict = make_qrel_dict(qrel_file_path)
    
    # Load mappings if needed
    passage_to_msmarco = make_mapping_dict(doc_mapping_path) if is_dl23 and doc_mapping_path else None
    qidtomsmarcoqids = make_mapping_dict(query_mapping_path) if is_dl23 and query_mapping_path else None
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    visited_pairs = set()
    processed_count = 0
    error_count = 0
    
    with gzip.open(output_file, 'wt', encoding='utf-8') as out_f:
        with open(input_file, 'r', encoding='utf-8') as in_f:
            for line in in_f:
                try:
                    log = json.loads(line.strip())
                    qid = log["qidx"]
                    docid = log["docidx"]

                    
                    if is_dl23:
                        pair_key = (qidtomsmarcoqids[qid],passage_to_msmarco[docid])
                    else:
                        pair_key = (qid, docid)
                    # Try to get ground truth score
                    try:
                        ground_truth = qrel_dict[(qid, docid)]
                    except KeyError:
                        try:
                            ground_truth = qrel_dict[(str(qid), str(docid))]
                        except KeyError:
                            print(f"Warning: No ground truth found for pair {qid}, {docid}")
                            ground_truth = 0
                            
                    # print(ground_truth)
                    if pair_key not in visited_pairs:
                        visited_pairs.add(pair_key)
                        
                        json_line = generate_umbrella_json_line(
                            query_id=qid,
                            paragraph_id=docid,
                            text=log["passage"],
                            query_text=log["query"],
                            ground_truth_relevance_label=ground_truth,
                            model_output=log["LLMs_output"],
                            final_score=log["final_relevance_score"],
                            mode=log["prompt_mode"],
                            model_name=model_name,
                            passage_to_msmarco=passage_to_msmarco,
                            qidtomsmarcoqids=qidtomsmarcoqids
                        )
                        out_f.write(json_line + '\n')
                        print(processed_count)
                        processed_count += 1
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
                    error_count += 1
                except KeyError as e:
                    print(f"Missing key in entry: {e}")
                    error_count += 1

    print(f"Processing complete:\n"
          f"Processed entries: {processed_count}\n"
          f"Errors encountered: {error_count}")
