import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from accelerate import load_checkpoint_and_dispatch
import torch
import json
from utils import (
    format_data,
    format_prompt,
    save_generations,
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
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--num_return_sequences", type=int, default=100)
parser.add_argument(
    "--task", type=str, default="tool", choices=["tool", "code", "math"]
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=-1,
    help="batch size, -1 means full batch and dispatch with vllm",
)
parser.add_argument("--temperature", type=float, default=1.5)
args = parser.parse_args()


def sampling(
    llm,
    prompt_list,
    data_list,
    batch_size,
    num_return_sequences,
    sampling_params,
    output_path,
):
    if batch_size == -1:
        batch_size = len(prompt_list)
    for batch_idx in range(0, len(prompt_list), batch_size):
        batch_prompts = prompt_list[batch_idx : batch_idx + batch_size]
        batch_data = data_list[batch_idx : batch_idx + batch_size]
        batch_outputs = llm.generate(batch_prompts, sampling_params=sampling_params)
        final_data_list = []
        for output, data in zip(batch_outputs, batch_data):
            final_data = save_generations(
                data, output, output_path, num_return_sequences
            )
            final_data_list.append(final_data)
        with open(output_path + ".json", "w") as f:
            json.dump(final_data_list, f, indent=4)


def main():
    model_name = args.model_name
    batch_size = args.batch_size
    num_return_sequences = args.num_return_sequences
    gpu_num = torch.cuda.device_count()
    output_dict = get_output_dict(model_name.split("/")[-1])

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        n=args.num_return_sequences,
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

    for input_path, output_path in zip(input_dict[args.task], output_dict[args.task]):
        with open(input_path, "r") as f:
            raw_data = json.load(f)
        data_list, messages_list = format_data(raw_data, args.task)
        prompt_list = format_prompt(messages_list, tokenizer)
        sampling(
            llm,
            prompt_list,
            data_list,
            batch_size,
            num_return_sequences,
            sampling_params,
            output_path,
        )


if __name__ == "__main__":
    main()
