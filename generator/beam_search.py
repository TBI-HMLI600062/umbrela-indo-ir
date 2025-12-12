import os
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams
from vllm.beam_search import BeamSearchOutput
import torch
import json
from utils import (
    format_data,
    format_prompt,
    save_beam_search,
    input_dict,
    get_output_dict,
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    default="",
    required=True,
    help="Path to the model directory or HuggingFace model name",
)
parser.add_argument("--max_new_tokens", type=int, default=300)
parser.add_argument(
    "--task", type=str, default="tool", choices=["tool", "code", "math"]
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=10,
    help="batch size, -1 means full batch and dispatch with vllm",
)
parser.add_argument(
    "--beam",
    type=int,
    default=10,
)
parser.add_argument("--temperature", type=float, default=0)
parser.add_argument("--output_path", type=str, default="data_gen/beam_search")
args = parser.parse_args()


def sampling(
    llm,
    prompt_list,
    data_list,
    batch_size,
    beam_search_params,
    output_path,
    input_path,
):
    if batch_size == -1:
        batch_size = len(prompt_list)
    data = []
    outputs = []
    prompts = []
    for batch_idx in tqdm(range(0, len(prompt_list), batch_size), desc="Beam Search"):
        batch_promts = prompt_list[batch_idx : batch_idx + batch_size]
        batch_data = data_list[batch_idx : batch_idx + batch_size]
        batch_outputs = llm.beam_search(
            batch_promts,
            beam_search_params,
        )
        data.extend(batch_data)
        outputs.extend(batch_outputs)
        prompts.extend(batch_promts)
        save_beam_search(data, outputs, prompts, output_path, input_path)


def main():
    model_name = args.model_name
    batch_size = args.batch_size
    gpu_num = torch.cuda.device_count()
    output_path = os.path.join(args.output_path, model_name.split("/")[-1], args.task)

    beam_search_params = BeamSearchParams(
        beam_width=args.beam,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "llama" in model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    llm = LLM(
        model=model_name,
        dtype="bfloat16",  #
        tensor_parallel_size=gpu_num,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
    )

    input_path = input_dict[args.task][-1]

    with open(input_path, "r") as f:
        raw_data = json.load(f)
    data_list, messages_list = format_data(raw_data, args.task)
    prompt_list = format_prompt(messages_list, tokenizer)
    prompt_list = [{"prompt": prompt} for prompt in prompt_list]
    sampling(
        llm,
        prompt_list,
        data_list,
        batch_size,
        beam_search_params,
        output_path,
        input_path,
    )


if __name__ == "__main__":
    main()
