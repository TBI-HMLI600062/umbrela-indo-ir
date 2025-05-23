import torch
import argparse
import os
import logging
from typing import Optional
from pathlib import Path
from data_processing import load_data_files, clean_files
from model_utils import get_model_baseline
# from prompts import create_system_message
from relevance_processors import grade_pq_pairs
from make_rubric_format import process_log_to_rubric

def setup_logging_and_device():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    logging.basicConfig(level=logging.WARNING)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Add these print statements to confirm
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    print(f"Current device: {device}")
    
    # If on CUDA, you can also print additional details
    if torch.cuda.is_available():
        print(f"Current CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return logging.getLogger(__name__), device

def parse_arguments():
    
    parser = argparse.ArgumentParser(description="Process relevance judgments using multi-criteria evaluation.")
    parser.add_argument("--model_id", type=str, required=True, 
                      help="Model ID (e.g., meta-llama/Llama-3.3-70B-Instruct-Turbo)")
    parser.add_argument("--test_qrel_path", type=str, required=True,
                      help="Path to relevance judgments file")
    parser.add_argument("--queries_path", type=str, required=True,
                      help="Path to queries file")
    parser.add_argument("--docs_path", type=str, required=True,
                      help="Path to documents file")
    parser.add_argument("--result_file_path", type=str, required=True,
                      help="Output path for results")
    parser.add_argument("--prompt_mode", type=str, default="zeroshot_basic",
                      help="Valid values: 'zeroshot_bing', 'zeroshot_basic', 'fewshot_bing', 'fewshot_basic'")
    parser.add_argument("--is_dl23", action="store_true", 
                        help="Set if using DL23 (adds mapping conversion)")
    parser.add_argument("-together", action="store_true",
                      help="Use together.ai API")
    parser.add_argument("--max_pairs", type=int, default=None,
                      help="Maximum number of pairs to process (default: process all)")

    return parser.parse_args()
    
    

def main():
    logger, device = setup_logging_and_device()
    args = parse_arguments()
    if ("2019" not in args.docs_path and "2020" not in args.docs_path) or args.is_dl23:
        is_dl23 = True
    else:
        is_dl23 = False
    docid_to_doc, qid_to_query, test_qrel = load_data_files(
        args.docs_path, args.queries_path, args.test_qrel_path
    )

    system_message = ""
    model = get_model_baseline(args.model_id, args.together)


    grade_pq_pairs(
        test_qrel, docid_to_doc, qid_to_query,
        args.result_file_path, model, system_message,args.prompt_mode, args.max_pairs)

    log_file ="."/ Path(args.result_file_path).parent / "logs" / Path(args.result_file_path).name.replace(".txt", ".jsonl")
    rubric_file = "."/ Path(args.result_file_path).parent / "rubric_format" / Path(args.result_file_path).name.replace(".txt", "_rubric.jsonl.gz")
    print(f"is_dl23:{is_dl23}")
    # # Convert relevance judgment logs to UMBRELLA rubric format for TREC-style evaluation
    process_log_to_rubric(
        input_file=log_file,
        output_file=rubric_file,
        qrel_file_path=args.test_qrel_path,
        is_dl23=is_dl23,  
        model_name=args.model_id
    )




if __name__ == "__main__":
    main()