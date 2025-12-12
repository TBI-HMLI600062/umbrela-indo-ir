import argparse
import json
import os

from tqdm import tqdm
from tool_ev import eval_file_tool
from math_ev import eval_file_math
from code_ev import eval_file_code


def eval_file(file_path, task):
    # return: bool (correctness), str (solving_res)
    if task == "tool":
        return eval_file_tool(file_path)
    elif task == "math":
        return eval_file_math(file_path)
    elif task == "code":
        return eval_file_code(file_path)
    else:
        raise ValueError(f"task {task} not found")


def eval_dir(dir_path, task):
    query_ids = []
    for dir in os.listdir(dir_path):
        query_ids.append(dir)
    query_ids.sort()
    for query_id in tqdm(query_ids, total=len(query_ids), desc="labeling"):
        file_path = os.path.join(dir_path, query_id, "results.json")
        eval_file(file_path, task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="",
        required=True,
        help="Path to the data directory containing results.json files",
    )
    parser.add_argument("--task", type=str, default="tool")
    args = parser.parse_args()

    eval_dir(args.data_path, args.task)
