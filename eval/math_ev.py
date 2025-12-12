import argparse
import json
import re
import signal
import regex as re
import ast
import multiprocessing

import deepspeed
from functools import partial
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def extract_boxed_answer(text):
    """
    Extract answers from \boxed{} using regular expressions.
    """
    match = re.findall(r"boxed\{((?:[^\{\}]|\{(?:[^\{\}]|\{[^\{\}]*\})*\})*)\}", text)
    # if not match:
    #     print("None!", text)
    return match[-1] if match else None


def verify_with_model(extracted_ans, target_answer, model, tokenizer):
    model.eval()
    prompt = f"<|start_header_id|>system<|end_header_id|>\n\nYou are a math expert.<|eot_id|><|start_header_id|>user<|end_header_id|>I have an answer to a problem that needs verification. The answer may involve complexity, and you should disregard the loss of units (if any) in both answers. My answer is {extracted_ans}, and the correct answer is {target_answer}. Please tell me whether my answer is correct or not in one word: 'Correct' or 'Incorrect'.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    source_text_res = tokenizer.encode_plus(
        prompt, max_length=1024, truncation=True, add_special_tokens=False
    )
    with torch.no_grad():
        outputs = model.module.generate(
            torch.tensor([source_text_res["input_ids"]]).to(
                torch.cuda.current_device()
            ),
            attention_mask=torch.tensor([source_text_res["attention_mask"]]).to(
                torch.cuda.current_device()
            ),
            max_new_tokens=128,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
    decoded_output = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    result = decoded_output[0].split("<|end_header_id|>\n\n")[-1]
    if "incorrect" in result.lower():
        print(result, "extracted_ans", extracted_ans, "target_answer", target_answer)
        return False
    elif "correct" in result.lower():
        print(result, "extracted_ans", extracted_ans, "target_answer", target_answer)
        return True
    else:
        print(
            "Strange Verify Result",
            result,
            "extracted_ans",
            extracted_ans,
            "target_answer",
            target_answer,
        )
        return False


def verifying_math(response, target_answer, model, tokenizer):
    extracted_ans = extract_boxed_answer(response)
    return (
        verify_with_model(extracted_ans, target_answer, model, tokenizer),
        extracted_ans,
    )


def eval_math(data, passk, model, tokenizer):
    total_correctness = 0
    total_num = len(data)
    if not isinstance(data, list):
        data = [data]

    for data_i, item in enumerate(data):
        # print(data_i)
        item = item[0]
        answer = item["answer_value"]
        recorded = False
        for i, pred in enumerate(item["predictions"]):
            # print(i)
            # if data_i == 69 and i == 95:
            #     pred["solving_res"] = None
            #     pred["correctness"] = False
            #     continue
            response = pred["completion"]

            if False:
                correctness, solving_res = verifying_math(
                    response, answer, model, tokenizer
                )
                pred["solving_res"] = solving_res
                pred["correctness"] = correctness
            else:
                solving_res = pred["solving_res"]
                correctness = pred["correctness"]
            pred["solving_res"] = solving_res
            pred["correctness"] = correctness
            if i < passk and correctness and not recorded:
                total_correctness += 1
                recorded = True
            if i >= passk:
                break
    acc = round(100 * total_correctness / total_num, 1)
    return acc, total_num, data


def eval_file_math(file_path, passk=1, model_name=None):
    with open(file_path, "r") as f:
        data = json.load(f)

    if model_name is None:
        model_name = "meta-llama/Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if "llama" in model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    # )
    # model = model.eval()

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model = model.eval()

    inf_config = {
        "replace_with_kernel_inject": False,
        "dtype": torch.bfloat16,
        "enable_cuda_graph": False,
        "tensor_parallel": {"tp_size": 1},
        "max_out_tokens": 128,
        "min_out_tokens": 1,
    }
    model = deepspeed.init_inference(model=model, config=inf_config)

    acc, total_num, data = eval_math(data, passk, model, tokenizer)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    return acc, total_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        default="",
        required=True,
        help="Path to the JSON file containing math reasoning results",
    )
    parser.add_argument(
        "--passk",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Path to the model directory or HuggingFace model name for verification",
    )
    args = parser.parse_args()
    acc, total_num = eval_file_math(args.file_path, args.passk, args.model_name)
    print(f"Pass@{args.passk} Accuracy: {acc}%   Total number: {total_num}")
