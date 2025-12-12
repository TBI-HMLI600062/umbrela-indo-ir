import argparse
import gc
import json
from pathlib import Path
import os
from typing import List, Dict
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

json_file_name = "results.json"
hidden_file_name = "hidden_states.pt"
sample_num = 100
LAYER_RATIOS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    type=str,
    default="",
    required=True,
    help="Path to the model directory or HuggingFace model name",
)
parser.add_argument(
    "--task", type=str, default="tool", choices=["tool", "code", "math"]
)
parser.add_argument("--batch_size", type=int, default=50)
parser.add_argument(
    "--data_root",
    type=str,
    default="data_gen",
    help="Root directory for generated data",
)
args = parser.parse_args()


def get_layer_indices(total_layers):
    """Calculate target layer indices"""
    return sorted([round(ratio * total_layers) for ratio in LAYER_RATIOS])


@torch.inference_mode()
def get_hidden_states(
    model,
    tokenizer,
    prompt: str,
    response_list: List[str],
    batch_size: int = 8,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Extract hidden states for the **last token** of:
      1) the stand‑alone `prompt`;
      2) every string obtained by concatenating `prompt + response`
         for `response` in `response_list`.

    Returns
    -------
    {
        "prompt_hidden": {layer_num: tensor(1, hidden_size)},
        "cand_hidden": {layer_num: tensor(sample_num, hidden_size)},
    }
    """

    # the tuple returned by `outputs.hidden_states` has length:
    #     1 (embeddings) + num_hidden_layers
    # so transformer layer `i` is at index `i+1`
    device = model.device
    target_layers = get_layer_indices(model.config.num_hidden_layers)

    def process_batch(texts):
        """Process text batches and collect hidden states"""
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        last_indices = inputs.attention_mask.sum(dim=1) - 1
        outputs = model(**inputs, output_hidden_states=True)
        hiddens = {
            layer: outputs.hidden_states[layer - 1][
                torch.arange(len(last_indices)), last_indices
            ].cpu()
            for layer in target_layers
        }
        del inputs, outputs, last_indices
        torch.cuda.empty_cache()
        gc.collect()
        return hiddens

    # Process prompt
    prompt_hidden = process_batch([prompt])

    # Process responses
    cand_hidden = {layer: [] for layer in target_layers}
    for i in range(0, len(response_list), batch_size):
        batch = [prompt + resp for resp in response_list[i : i + batch_size]]
        batch_hidden = process_batch(batch)
        for layer in target_layers:
            cand_hidden[layer].append(batch_hidden[layer])

        # Memory cleanup
        del batch, batch_hidden
        torch.cuda.empty_cache()
        gc.collect()

    for layer in target_layers:
        cand_hidden[layer] = torch.cat(cand_hidden[layer], dim=0)

    return {"prompt_hidden": prompt_hidden, "cand_hidden": cand_hidden}


def generate_hidden(path_list, model, tokenizer, batch_size=8):
    for path in tqdm(path_list, total=len(path_list)):
        json_path = os.path.join(path, json_file_name)
        with open(json_path, "r") as f:
            data = json.load(f)
        data = data[0]
        prompt = data["prompt"]
        response_list = [item["completion"] for item in data["predictions"]]

        # Keep only the first sample_num responses
        response_list = response_list[:sample_num]

        hidden_dict = get_hidden_states(
            model, tokenizer, prompt, response_list, batch_size
        )
        hidden_path = os.path.join(path, hidden_file_name)
        torch.save(hidden_dict, hidden_path)
        del hidden_dict
        torch.cuda.empty_cache()
        gc.collect()


def main():
    pure_model_name = args.model_name.split("/")[-1]
    model_name = args.model_name
    data_dir = os.path.join(args.data_root, pure_model_name, args.task)
    batch_size = args.batch_size

    print(f"Loading model from {model_name}")
    # Model loading
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    if "llama" in model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_path_list = []
    test_path_list = []
    for root, dirs, files in os.walk(data_dir):
        if json_file_name in files and hidden_file_name not in files:
            # if json_file_name in files:
            if "train" in root:
                train_path_list.append(root)
            elif "test" in root:
                test_path_list.append(root)
    print(f"unprocessed training data: {len(train_path_list)}")
    print(f"unprocessed test data: {len(test_path_list)}")

    train_path_list.sort()
    test_path_list.sort()

    print("Generating hidden states for training data...")
    generate_hidden(train_path_list, model, tokenizer, batch_size)
    print("Generating hidden states for test data...")
    generate_hidden(test_path_list, model, tokenizer, batch_size)


if __name__ == "__main__":
    main()
