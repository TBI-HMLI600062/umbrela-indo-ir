import os
import json
from prompt_template import (
    SYSTEM_CODE,
    SYSTEM_MATH,
    SYSTEM_TOOL,
    USER_TOOL,
    USER_CODE,
    USER_MATH,
)

input_dict = {
    "math": ["data/math/math_train.json", "data/math/math_test.json"],
    # "math": ["data/math/math_test.json"],
    "code": ["data/code/mbpp_train.json", "data/code/mbpp_test.json"],
    "tool": ["data/tool/xlam_train.json", "data/tool/xlam_test.json"],
}


def get_output_dict(model_name):
    output_dict = {
        "math": [
            f"data_gen/{model_name}/math/math_train",
            f"data_gen/{model_name}/math/math_test",
        ],
        "code": [
            f"data_gen/{model_name}/code/mbpp_train",
            f"data_gen/{model_name}/code/mbpp_test",
        ],
        "tool": [
            f"data_gen/{model_name}/tool/xlam_train",
            f"data_gen/{model_name}/tool/xlam_test",
        ],
    }
    return output_dict


def convert_messages_to_prompt(messages, tokenizer):
    """
    Convert OpenAI messages to model prompt (based on tokenizer.chat_template).
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("This tokenizer does not support chat templates.")

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # Add assistant header after user's last message, waiting for generation
    )
    return prompt


def format_data(data, task):
    if task == "math":
        messages_list = []
        for item in data:
            item["id"] = str(item["item_id"])
            messages = []
            system_prompt = SYSTEM_MATH
            user_prompt = USER_MATH.replace("{question}", item["question"])
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            messages_list.append(messages)

        return data, messages_list
    elif task == "code":
        messages_list = []
        for item in data:
            item["id"] = str(item["item_id"])[-3:].zfill(4)
            messages = []
            system_prompt = SYSTEM_CODE
            user_prompt = (
                USER_CODE.replace("{question}", item["question"])
                .replace("{function_name}", item["function_name"])
                .replace("{test_list[0]}", item["test_list"][0])
            )
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            messages_list.append(messages)

        return data, messages_list
    elif task == "tool":
        messages_list = []
        for item in data:
            item["id"] = str(item["id"]).zfill(5)
            messages = []
            system_prompt = SYSTEM_TOOL.replace("{function_list}", item["tools"])
            user_prompt = USER_TOOL.replace("{query}", item["query"])
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            messages_list.append(messages)

        return data, messages_list

    else:
        raise ValueError(f"Invalid task: {task}")


def format_prompt(messages_list, tokenizer):
    """
    Convert OpenAI messages to model prompt (based on tokenizer.chat_template).
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("This tokenizer does not support chat templates.")

    prompt_list = []
    for messages in messages_list:
        prompt = convert_messages_to_prompt(messages, tokenizer)
        prompt_list.append(prompt)
    return prompt_list


def save_generations(data, output, output_path, num_return_sequences):
    save_path = os.path.join(output_path, f"{data['id']}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data["prompt"] = output.prompt
    data["predictions"] = []
    for i in range(num_return_sequences):
        pred = {}
        pred["completion"] = output.outputs[i].text
        data["predictions"].append(pred)

    save_path = os.path.join(save_path, "results.json")
    if not isinstance(data, list):
        data = [data]
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)
    return data


def save_beam_search(data, outputs, prompts, output_path, input_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    save_path = os.path.join(output_path, input_path.split("/")[-1])

    final_data = []
    for item, output, prompt in zip(data, outputs, prompts):

        item["prompt"] = prompt["prompt"]
        item["predictions"] = []
        pred = {}
        pred["completion"] = output.sequences[0].text[len(prompt["prompt"]) :]
        item["predictions"].append(pred)
        final_data.append(item)
    with open(save_path, "w") as f:
        json.dump(final_data, f, indent=4)


def save_greedy(data, outputs, output_path, input_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    save_path = os.path.join(output_path, input_path.split("/")[-1])
    final_data = []
    for item, output in zip(data, outputs):

        item["prompt"] = output.prompt
        item["predictions"] = []
        for i in range(len(output.outputs)):
            pred = {}
            pred["completion"] = output.outputs[i].text
            item["predictions"].append(pred)
        final_data.append(item)
    with open(save_path, "w") as f:
        json.dump(final_data, f, indent=4)
