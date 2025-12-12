import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import os
import gc
import argparse
import random
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from datetime import datetime
import logging
from ranker_structure import RankModel_Transformer, RankModel_MLP
import concurrent.futures
import wandb
import itertools


timestamp = datetime.now().strftime("%Y%m%d%H%M%S")


# Done: Introduce candidate_mask, consider whether random sampling of positive samples is needed
# Should need to sample positive samples
# ToDo: Validation and test sets have bias, need detailed analysis
class ProgressStep:
    def __init__(self):
        self.num = 0


class RankDataset(Dataset):
    def __init__(
        self,
        folders,
        layer,
        cand_num=10,
        istrain=True,
        filtered=True,
        sample_num=500,
        correct_sample="random",
    ):
        self.context_hstates = []
        self.candidate_hstates = []
        self.labels = []
        self.key_padding_mask = []
        self.filtered = filtered
        self.cand_num = cand_num
        self.sample_num = sample_num
        self.correct_sample = correct_sample
        self.layer = layer

        # Use concurrent futures to process folders in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Process each folder in parallel
            if istrain:
                results = list(
                    tqdm(
                        executor.map(self.process_folder, folders),
                        desc="Loading train data",
                        total=len(folders),
                    )
                )
            else:
                results = list(
                    tqdm(
                        executor.map(self.process_test_folder, folders),
                        desc="Loading test data",
                        total=len(folders),
                    )
                )

        # Unzip the results into respective lists

        def transfer(nested_list):
            return list(itertools.chain(*filter(lambda x: x is not None, nested_list)))

        a, b, c, d = zip(*results)
        self.context_hstates = transfer(a)
        self.candidate_hstates = transfer(b)
        self.labels = transfer(c)
        self.key_padding_mask = transfer(d)

        # Convert lists to tensors
        old_num_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(8)  # Temporarily set thread count
            self.context_hstates = torch.concat(
                self.context_hstates, dim=0
            )  # (batch, feature_dim)
            self.candidate_hstates = torch.stack(self.candidate_hstates, dim=0)
            self.key_padding_mask = torch.stack(self.key_padding_mask, dim=0)
            self.labels = torch.stack(self.labels, dim=0)
        finally:
            # Restore original thread count setting
            torch.set_num_threads(old_num_threads)

        print("Dataset Init Finished!")
        # self.labels = torch.stack(self.labels)  # (batch, cand_num)

        # self.candidate_hstates = torch.stack(self.candidate_hstates)  # (batch, cand_num, feature_dim)
        # self.candidate_hstates, self.key_padding_mask, self.labels = self.pad_and_create_mask()

    def process_test_folder(self, folder):
        context_hstates, cand_hstates, desc = load_single_data(folder, self.layer)
        labels = [
            item["correctness"] for item in desc["predictions"]
        ]  # Extract labels (list of 300)
        labels = torch.tensor(
            labels, dtype=torch.bool
        )  # Convert label list to a tensor of shape (300,)
        if self.filtered:
            remain_labels = [
                item["solving_res"] is not None for item in desc["predictions"]
            ]
            remain_labels = torch.tensor(remain_labels, dtype=torch.bool)
        else:
            remain_labels = [True for item in desc["predictions"]]
            remain_labels = torch.tensor(remain_labels, dtype=torch.bool)

        # sample_num = len(correct_indices)
        context_hstates_list = []
        cand_hstates_list = []
        labels_list = []
        mask_list = []

        for _ in range(self.sample_num):

            # sampled_indices=random.sample(range(len(labels)), cand_num)
            sampled_indices = [i for i in range(self.cand_num)]
            sampled_indices = torch.tensor(sampled_indices, dtype=torch.int64)
            elected = remain_labels[sampled_indices]
            sampled_indices = sampled_indices[elected]
            sampled_indices = sampled_indices.tolist()
            # random.shuffle(sampled_indices)

            # Create candidate hstates for the current group
            current_candidate_hstates = cand_hstates[
                sampled_indices
            ]  # Shape (10, 1536)
            current_label = labels[sampled_indices]

            padded_tensor = torch.zeros(
                self.cand_num, context_hstates.size(-1), dtype=torch.bfloat16
            )
            X = current_candidate_hstates.size(0)
            padded_tensor[:X] = current_candidate_hstates

            mask = torch.ones(self.cand_num + 1)
            mask[: X + 1] = 0

            padded_label = torch.zeros(self.cand_num)
            padded_label[:X] = current_label

            # Add to the dataset's context, candidates, and labels
            context_hstates_list.append(context_hstates)  # Same context for all groups
            cand_hstates_list.append(padded_tensor)
            labels_list.append(padded_label)
            mask_list.append(mask)

        return (context_hstates_list, cand_hstates_list, labels_list, mask_list)

    def process_folder(self, folder):
        context_hstates, cand_hstates, desc = load_single_data(folder, self.layer)
        labels = [
            item["correctness"] for item in desc["predictions"]
        ]  # Extract labels (list of 300)
        labels = torch.tensor(
            labels, dtype=torch.bool
        )  # Convert label list to a tensor of shape (300,)
        if self.filtered:
            remain_labels = [
                item["solving_res"] is not None for item in desc["predictions"]
            ]
            remain_labels = torch.tensor(remain_labels, dtype=torch.bool)
        else:
            remain_labels = [True for item in desc["predictions"]]
            remain_labels = torch.tensor(remain_labels, dtype=torch.bool)

        # Find indices of correct and incorrect candidate hstates
        correct_indices = torch.nonzero(labels, as_tuple=True)[
            0
        ].tolist()  # Indices where labels are True
        incorrect_indices = torch.nonzero(~labels, as_tuple=True)[
            0
        ].tolist()  # Indices where labels are False

        context_hstates_list = []
        cand_hstates_list = []
        mask_list = []
        labels_list = []

        if len(correct_indices) == 0:
            return None, None, None, None

        for _ in range(self.sample_num):
            # if len(correct_indices) == 0:
            #     break  #  Considered in if sample_num == 0

            # Randomly sample correct and incorrect candidates
            if self.correct_sample == "random":
                correct_index = random.choice(correct_indices)
                # sample 1 or multiple positive samples
                # sampled_indices= [correct_index] + random.sample(incorrect_indices, cand_num - 1)
                sampled_indices = [correct_index] + random.sample(
                    range(len(labels)), self.cand_num - 1
                )
                sampled_indices = list(set(sampled_indices))
            elif self.correct_sample == "fixed":
                correct_index = random.choice(correct_indices)
                sampled_indices = [correct_index] + [
                    random.choice(incorrect_indices) for _ in range(self.cand_num - 1)
                ]

            sampled_indices = torch.tensor(sampled_indices, dtype=torch.int64)
            elected = remain_labels[sampled_indices]
            sampled_indices = sampled_indices[elected]
            sampled_indices = sampled_indices.tolist()
            random.shuffle(sampled_indices)

            # Create candidate hstates for the current group
            current_candidate_hstates = cand_hstates[
                sampled_indices
            ]  # Shape (10, 1536)
            current_label = labels[sampled_indices]

            if current_label.sum() == 0:
                print("Warning: All samples are incorrect, skipping this sample")
                continue
            padded_tensor = torch.zeros(
                self.cand_num, context_hstates.size(-1), dtype=torch.bfloat16
            )
            X = current_candidate_hstates.size(0)
            padded_tensor[:X] = current_candidate_hstates

            mask = torch.ones(self.cand_num + 1)
            mask[: X + 1] = 0

            padded_label = torch.zeros(self.cand_num)
            padded_label[:X] = current_label

            # Add to the dataset's context, candidates, and labels
            context_hstates_list.append(context_hstates)  # Same context for all groups
            cand_hstates_list.append(padded_tensor)
            mask_list.append(mask.bool())
            labels_list.append(padded_label.bool())

        return (context_hstates_list, cand_hstates_list, labels_list, mask_list)

    def pad_and_create_mask(self):
        padded_hstates = []
        key_padding_mask = []
        padded_labels = []
        feature_dim = self.candidate_hstates[0].size(-1)
        for i, (hstate, label) in enumerate(zip(self.candidate_hstates, self.labels)):
            # Create a zero tensor (cand_num, feature_dim)
            padded_tensor = torch.zeros(
                self.cand_num, feature_dim, dtype=torch.bfloat16
            )
            X = hstate.size(0)
            padded_tensor[:X] = hstate

            # Create key_padding_mask considering context_hstates
            mask = torch.ones(self.cand_num + 1)
            mask[: X + 1] = 0

            padded_label = torch.zeros(self.cand_num)
            padded_label[:X] = label

            padded_hstates.append(padded_tensor)
            key_padding_mask.append(mask)
            padded_labels.append(padded_label)

        # Stack results
        padded_hstates = torch.stack(padded_hstates)  # (N, cand_num, feature_dim)
        key_padding_mask = torch.stack(key_padding_mask)  # (N, cand_num)
        padded_labels = torch.stack(padded_labels)  # (N, cand_num)

        return padded_hstates, key_padding_mask.bool(), padded_labels.bool()

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return (
            self.context_hstates[idx],
            self.candidate_hstates[idx],
            self.labels[idx],
            self.key_padding_mask[idx],
        )


def load_single_data(path, layer):
    data = torch.load(
        os.path.join(path, "hidden_states.pt"),
        weights_only=False,
        map_location="cpu",
    )
    context_hstates = data["prompt_hidden"][layer]
    cand_hstates = data["cand_hidden"][layer]
    with open(os.path.join(path, "results.json"), "r", encoding="utf-8") as f:
        desc = json.load(f)
    desc = desc[0]
    # Need to match the number of sampled hidden states
    desc["predictions"] = desc["predictions"][:100]
    # print(context_hstates.shape)
    return context_hstates, cand_hstates, desc


# def get_folders_in_batches(root_dir, file_size=100):
#     all_folders = [
#         os.path.join(root_dir, folder)
#         for folder in os.listdir(root_dir)
#         if os.path.isdir(os.path.join(root_dir, folder))
#     ]
#     for i in range(0, len(all_folders), file_size):
#         if i + file_size > len(all_folders):
#             yield all_folders[i:]
#         else:
#             yield all_folders[i : i + file_size]


def get_folders_in_batches(root_dir, dataset_type, split=None):
    if dataset_type == "single":
        all_folders = [
            os.path.join(root_dir, folder)
            for folder in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, folder))
        ]
        random.shuffle(all_folders)
        return all_folders
    elif dataset_type[:3] == "all":
        layer = dataset_type[3:]
        all_folders = list(
            itertools.chain(
                *[
                    [
                        os.path.join(root_dir, category, folder)
                        for folder in os.listdir(os.path.join(root_dir, category))
                        if os.path.isdir(os.path.join(root_dir, category, folder))
                    ]
                    for category in os.listdir(root_dir)
                    if f"_{layer}" in category
                    and split in category
                    and "100" in category
                    and "math" in category
                ]
            )
        )
        random.shuffle(all_folders)
        return all_folders


def evaluate_ranker(ranker_path, test_dir, dataset_layer, cand_num, batch_size):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    ranker = torch.load(ranker_path, weights_only=False)

    test_folders = get_folders_in_batches(test_dir, dataset_type="single", split="test")

    test_dataset = RankDataset(
        test_folders, dataset_layer, istrain=False, cand_num=cand_num, sample_num=1
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    test_accuracy = evaluate_accuracy(ranker, test_loader, device)
    print(f"Cand Num: {cand_num}, Test accuracy: {test_accuracy}")
    return test_accuracy


def evaluate_accuracy(ranker, test_loader, device):
    ranker.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for context_hstates, candidate_hstates, labels, key_padding_mask in test_loader:
            context_hstates, candidate_hstates, labels, key_padding_mask = (
                context_hstates.to(device),
                candidate_hstates.to(device),
                labels.to(device),
                key_padding_mask.to(device),
            )
            key_padding_mask = None

            if isinstance(ranker, RankModel_Transformer):
                outputs = ranker(candidate_hstates, context_hstates, key_padding_mask)
            elif isinstance(ranker, RankModel_MLP):
                outputs = ranker(
                    torch.cat([context_hstates.unsqueeze(1), candidate_hstates], dim=1)
                )
            _, predicted = torch.max(outputs, dim=1)
            equal_elements = torch.gather(labels, 1, predicted.unsqueeze(1))
            correct += equal_elements.sum().item()
            total += candidate_hstates.size(0)
    accuracy = 100 * correct / total
    return accuracy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a ranker with cross-attention for ranking"
    )
    parser.add_argument(
        "--ranker_path",
        type=str,
        default="",
        required=True,
        help="Path to the trained ranker model file",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="",
        required=True,
        help="Root directory containing the test data folders",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=500,
        help="Batch size",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ranker_info = args.ranker_path.split("/")[-2]
    dataset_layer = int(ranker_info.split("dataset_layer")[-1].split("-")[0])
    results = {}

    cand_list = [1, 3, 5, 10]
    for cand_num in cand_list:
        test_accuracy = evaluate_ranker(
            args.ranker_path, args.test_dir, dataset_layer, cand_num, args.batchsize
        )
        results[cand_num] = test_accuracy
    print("=" * 50)
    for cand_num, test_accuracy in results.items():
        print(f"Cand Num: {cand_num}, Test accuracy: {test_accuracy}")
    print("=" * 50)

    # Parent directory of args.ranker_path
    save_path = os.path.dirname(args.ranker_path)
    with open(os.path.join(save_path, "scaling_law.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
