import argparse
import json
import re
import signal
import regex as re
import ast
import multiprocessing


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def timeout_handler(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.timeout_handler)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def extract_boxed_answer(text):
    a = text.split("# START OF CODE")
    if len(a) <= 1:
        return None
    b = a[-1].split("# END OF CODE")
    if len(b) <= 1:
        return None
    # if not match:
    #     print("None!", text)
    return b[0]


def verify_code(extracted_answer, test_list):
    if extracted_answer is None:
        return False
    exec_globals = {}  # Only allow specific variables
    try:
        with timeout(seconds=2):
            exec(extracted_answer, exec_globals)
    except Exception as e:
        print(e)
        return False

    # Get the parsed function
    function_name = test_list[0].split("assert ")[-1].split("(")[0]
    if function_name == "":
        function_name = test_list[0].split("assert ")[-1].split("(")[1]
    function = exec_globals.get(function_name)

    if not callable(function):
        print("Not callable!")
        return False

    # Parse and execute `assert` statements
    for test_case in test_list:
        try:
            # Parse test statement
            parsed = ast.parse(test_case, mode="exec")

            # Ensure the AST has only one `assert` statement
            if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.Assert):
                raise ValueError(f"Invalid test case: {test_case}")

            # Get the test expression from `assert`
            test_expr = parsed.body[0].test  # `test_expr` is of type `Compare`

            # Ensure it is a comparison expression (e.g., my_function(1) == 2)
            if (
                not isinstance(test_expr, ast.Compare)
                or len(test_expr.comparators) != 1
            ):
                raise ValueError(f"Unsupported test format: {test_case}")

            # Calculate left expression
            left_expr = compile(ast.Expression(test_expr.left), "<test>", "eval")
            queue = multiprocessing.Queue()

            def eval_with_timeout(queue):
                try:
                    queue.put(eval(left_expr, exec_globals))
                except Exception as e:
                    queue.put(e)

            process = multiprocessing.Process(target=eval_with_timeout, args=(queue,))

            process.start()
            process.join(2)
            if process.is_alive():
                print("Too long! Killed!")
                process.terminate()
                return False

            result = queue.get_nowait()
            if isinstance(result, Exception):
                return False
            left_value = result

            # Calculate right expression (expected value)
            right_expr = compile(
                ast.Expression(test_expr.comparators[0]), "<test>", "eval"
            )
            right_value = eval(right_expr, exec_globals)

            print(left_value, right_value)

            # Perform comparison
            if left_value == right_value:
                continue
            else:
                return False

        except Exception as e:
            print(e)
            return False

    return True


def verifying_code(response, test_list):
    code = extract_boxed_answer(response)
    return verify_code(code, test_list), code


def eval_code(data, passk=10):
    total_correctness = 0
    total_num = len(data)
    if not isinstance(data, list):
        data = [data]

    for data_i, item in enumerate(data):
        # print(data_i)
        if isinstance(item, list):
            item = item[0]
        test_list = item["test_list"]
        recorded = False
        for i, pred in enumerate(item["predictions"]):
            # print(i)
            if i == 95:
                pred["solving_res"] = None
                pred["correctness"] = False
                continue
            response = pred["completion"]
            if False:
                correctness, solving_res = verifying_code(response, test_list)
                pred["solving_res"] = solving_res
                pred["correctness"] = correctness
            else:
                solving_res = pred["solving_res"]
                correctness = pred["correctness"]
            if i < passk and correctness and not recorded:
                total_correctness += 1
                recorded = True
    acc = round(100 * total_correctness / total_num, 1)
    return acc, total_num, data


def eval_file_code(file_path, passk=1):
    with open(file_path, "r") as f:
        data = json.load(f)

    acc, total_num, data = eval_code(data, passk)

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
        help="Path to the JSON file containing code generation results",
    )
    parser.add_argument(
        "--passk",
        type=int,
        default=1,
    )
    args = parser.parse_args()
    acc, total_num = eval_file_code(args.file_path, args.passk)
    print(f"Pass@{args.passk} Accuracy: {acc}%   Total number: {total_num}")
