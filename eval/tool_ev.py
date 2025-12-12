import argparse
import json
import re
from collections import defaultdict


def get_last_outer_bracket_content(s):
    stack = []
    result = []
    for i, char in enumerate(s):
        if char == "[":
            stack.append(i)
        elif char == "]":
            if stack:
                start = stack.pop()
                if not stack:  # Stack is empty, meaning this is the outermost []
                    result.append(s[start : i + 1])
    return result[-1] if result else None


def get_function_call(response):
    fc_string = get_last_outer_bracket_content(response)
    if fc_string:
        try:
            fc_json = json.loads(fc_string)
            if not isinstance(fc_json, list):
                fc_json = json.loads(fc_json)
            return fc_json
        except Exception as e:
            # print(e)
            return f"[json decode error] {fc_string}"
    else:
        return f"[fc extract error] {response}"


def compare_function_call(fc_json, answer):
    if isinstance(fc_json, str):
        try:
            fc_json = json.loads(fc_json)
        except Exception as e:
            return False
    answer_json = json.loads(answer)
    if len(fc_json) != len(answer_json):
        return False
    for ans in answer_json:
        flag = False
        for fc in fc_json:
            if ans == fc:
                flag = True
                break
        if not flag:
            return False
    return True


def verifying_tool(response, answer):
    fc_json = get_function_call(response)
    if not isinstance(fc_json, list):
        return False, ""
    fc_string = json.dumps(fc_json)
    return compare_function_call(fc_json, answer), fc_string


def eval_tool(data, passk=1):
    total_correctness = 0
    mv_correctness = 0
    total_num = len(data)
    if not isinstance(data, list):
        data = [data]

    for item in data:
        stat_dict = defaultdict(int)

        if isinstance(item, list):
            item = item[0]
        answer = item["answers"]
        recorded = False
        for i, pred in enumerate(item["predictions"]):
            response = pred["completion"]
            correctness, solving_res = verifying_tool(response, answer)
            pred["solving_res"] = solving_res
            pred["correctness"] = correctness
            if i < passk and solving_res:
                stat_dict[solving_res] += 1
            if i < passk and correctness and not recorded:
                total_correctness += 1
                recorded = True

        if list(stat_dict.keys()):
            mv_res = max(stat_dict, key=stat_dict.get)
            mv_correctness += compare_function_call(mv_res, answer)

    acc = round(100 * total_correctness / total_num, 1)
    mv_acc = round(100 * mv_correctness / total_num, 1)
    return acc, total_num, data, mv_acc


def eval_file_tool(file_path, passk=1):
    with open(file_path, "r") as f:
        data = json.load(f)

    acc, total_num, data, mv_acc = eval_tool(data, passk)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    return acc, total_num, mv_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        default="",
        required=True,
        help="Path to the JSON file containing tool calling results",
    )
    parser.add_argument(
        "--passk",
        type=int,
        default=10,
    )
    args = parser.parse_args()
    acc, total_num, mv_acc = eval_file_tool(args.file_path, args.passk)
    print(
        f"Pass@{args.passk} Accuracy: {acc}%   MV@{args.passk} Accuracy: {mv_acc}%   Total number: {total_num}"
    )
