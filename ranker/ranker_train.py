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


def train_ranker_in_batches(args):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    train_info = f"{args.ranker_structure}-multipos-layers{args.num_layers}-lr{args.lr}-epoch{args.num_epochs}-{args.num_splits}-numheads{args.num_attention_heads}-bz{args.batch_size}-dp{args.dropout_prob}-{args.opt}-momentum{args.momentum}-wd{args.weight_decay}-cand{args.cand_num}-correct-{args.correct_sample}-sample{args.sample_num}-filtered-{args.filtered}-criterion{args.criterion}-schedule{args.lr_schedule}-dataset_layer{args.dataset_layer}"

    if "trans" in args.ranker_structure:
        if args.ranker_structure == "trans":
            ranker = RankModel_Transformer(
                args.hidden_size,
                args.hidden_size,
                args.num_attention_heads,
                args.dropout_prob,
                args.num_layers,
                args.half,
            ).to(device)
        elif "*" in args.ranker_structure:
            scale = float(args.ranker_structure.split("*")[-1])
            print(f"Set hidden size as {scale} times as the input size!")
            ranker = RankModel_Transformer(
                args.hidden_size,
                int(args.hidden_size * scale),
                args.num_attention_heads,
                args.dropout_prob,
                args.num_layers,
                args.half,
            ).to(device)
    if "mlp" in args.ranker_structure:
        if args.ranker_structure == "mlp":
            ranker = RankModel_MLP(
                args.hidden_size,
                int(args.hidden_size / 4.0),
                1,
                args.num_layers,
                args.dropout_prob,
                None,
                args.half,
            ).to(device)
        elif "_o" in args.ranker_structure:
            # mlp_o128_cri_cos_t0.2
            information = args.ranker_structure.split("_o")[-1]
            output_size = int(information.split("_cri_")[0])
            criterion = information.split("_cri_")[-1]
            ranker = RankModel_MLP(
                args.hidden_size,
                output_size,
                args.num_layers,
                args.dropout_prob,
                criterion,
                args.half,
            ).to(device)

    test_folders = get_folders_in_batches(
        args.test_dir, dataset_type=args.dataset_type, split="test"
    )

    test_dataset = RankDataset(
        test_folders,
        args.dataset_layer,
        istrain=False,
        cand_num=args.cand_num,
        sample_num=1,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )
    log_dir="log/ranker_train/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, train_info + "-" + timestamp + ".log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="")

    # criterion = nn.CrossEntropyLoss()
    if args.opt == "sgd":
        optimizer = optim.SGD(
            ranker.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.opt == "adam":
        optimizer = optim.Adam(ranker.parameters(), lr=args.lr)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(
            ranker.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
            eps=1e-4,
        )

    if args.lr_schedule == "constant":
        scheduler = optim.lr_scheduler.ConstantLR(
            optimizer=optimizer, factor=1.0, total_iters=args.num_splits
        )
    elif args.lr_schedule == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.num_splits * args.num_epochs,
            eta_min=args.lr * 0.01,
        )

    for epoch in range(args.num_epochs):
        train_folders = get_folders_in_batches(
            args.train_dir, dataset_type=args.dataset_type, split="train"
        )
        used_folders_num = len(train_folders) // args.num_splits
        for split in range(args.num_splits):
            used_folders = train_folders[:used_folders_num]
            train_folders = train_folders[used_folders_num:]
            train_dataset = RankDataset(
                used_folders,
                args.dataset_layer,
                filtered=args.filtered,
                cand_num=args.cand_num,
                sample_num=args.sample_num,
                correct_sample=args.correct_sample,
            )
            logging.info("Dataset Finished!")
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=12,
                pin_memory=True,
            )
            logging.info("Dataloader Finished!")
            train_loss, test_accuracy = train(
                ranker,
                train_loader,
                args.criterion,
                optimizer,
                device,
                split + epoch * args.num_splits,
                test_loader,
                scheduler,
            )
            logging.info(
                "***###------------------------------------------------------------###***"
            )
            logging.info(
                f"Epoch {epoch}/{args.num_epochs} Split {epoch+1}/{args.num_splits}, Train Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%"
            )
            logging.info(
                "***###------------------------------------------------------------###***"
            )

            ranker_dir = os.path.join(args.save_dir, f"{train_info}-{timestamp}")
            if not os.path.exists(ranker_dir):
                os.makedirs(ranker_dir)
            ranker_save_path = os.path.join(
                ranker_dir,
                f"Split_{epoch * args.num_splits + split+1}-TestA_{test_accuracy:.1f}-TrainL_{train_loss:.1f}.pth",
            )
            torch.save(ranker, ranker_save_path)
            logging.info(f"ranker saved to {ranker_save_path}")

            del train_dataset, train_loader
            gc.collect()
            torch.cuda.empty_cache()


def train(
    ranker,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch,
    test_loader,
    scheduler,
):
    ranker.train()
    total_loss = 0.0
    for step, (
        context_hstates,
        candidate_hstates,
        labels,
        key_padding_mask,
    ) in enumerate(train_loader):
        # pdb.set_trace()
        context_hstates, candidate_hstates, labels, key_padding_mask = (
            context_hstates.to(device),
            candidate_hstates.to(device),
            labels.to(device),
            key_padding_mask.to(device),
        )
        optimizer.zero_grad()
        if isinstance(ranker, RankModel_Transformer):
            outputs = ranker(candidate_hstates, context_hstates, key_padding_mask)
        elif isinstance(ranker, RankModel_MLP):
            outputs = ranker(
                torch.cat([context_hstates.unsqueeze(1), candidate_hstates], dim=1)
            )
        if criterion == "kl_div":
            positive_value = 1.0 / labels.sum(dim=1).float()
            targets = positive_value.unsqueeze(1) * labels.float()
            targets.to(dtype=torch.bfloat16)
            log_outputs = F.log_softmax(outputs, dim=1)
            loss = F.kl_div(log_outputs, targets + 1e-4, reduction="mean")

        elif criterion == "logistic":
            labels = labels.to(dtype=torch.bfloat16)
            loss = F.binary_cross_entropy_with_logits(outputs, labels)
        elif criterion == "dpo":
            outputs_matrix = outputs.unsqueeze(-1) - outputs.unsqueeze(-2)
            labels_matrix = labels.to(dtype=torch.bfloat16).unsqueeze(-1) - labels.to(
                dtype=torch.bfloat16
            ).unsqueeze(-2)
            loss = F.binary_cross_entropy_with_logits(
                outputs_matrix, labels_matrix, reduction="none"
            )
            loss = (loss * (labels_matrix > 0.5).to(dtype=torch.bfloat16)).sum() / (
                (labels_matrix > 0.5).to(dtype=torch.bfloat16)
            ).sum()
        elif criterion == "sce":
            positive_value = 1.0 / labels.sum(dim=1).float()
            targets = positive_value.unsqueeze(1) * labels.float()
            targets.to(dtype=torch.bfloat16)
            outputs = F.softmax(outputs, dim=1)
            loss = -torch.mean(
                torch.log(outputs + 1e-4) * targets
                + torch.log(targets + 1e-4) * outputs
            )
        else:
            raise NotImplementedError
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(
            ranker.parameters(), max_norm=3.0, norm_type=2
        )
        optimizer.step()

        total_loss += loss.item()
        if step % 20 == 0:
            print(f"Split {epoch}, Step {step}, Loss: {loss.item()}")
            logging.info(f"Split {epoch}, Step {step}, Loss: {loss.item()}")
        if step % 100 == 0 or step == len(train_loader) - 1:
            test_accuracy = evaluate_accuracy(ranker, test_loader, device)

    scheduler.step()

    return total_loss / len(train_loader), test_accuracy


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
    ranker.train()
    return accuracy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a ranker with cross-attention for ranking"
    )
    # model
    parser.add_argument(
        "--hidden_size", type=int, default=4096, help="Hidden size of the ranker"
    )
    parser.add_argument(
        "--num_attention_heads", type=int, default=1, help="Number of attention heads"
    )
    parser.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout probability"
    )
    parser.add_argument(
        "--num_layers", type=int, default=1, help="Number of cross-attention layers"
    )
    parser.add_argument(
        "--ranker_structure", type=str, default="trans", help="ranker structure"
    )
    parser.add_argument("--half", action="store_true", default=False)

    # opt
    parser.add_argument("--opt", type=str, default="sgd")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_schedule", type=str, default="constant")

    # training
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--num_splits", type=int, default=50, help="Number of data splits"
    )
    # parser.add_argument("--gpu", type=int, default=0, help="GPU device number to use")
    parser.add_argument("--criterion", type=str, default="kl_div")

    # dataset
    parser.add_argument("--filtered", action="store_true", default=False)
    parser.add_argument(
        "--train_dir",
        type=str,
        default="",
        required=True,
        help="Root directory containing the training data folders",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="",
        required=True,
        help="Root directory containing the test data folders",
    )
    parser.add_argument("--model_path", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--file_size", type=int, default=100, help="Number of folders to load at once"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="model/ranker",
        help="Directory to save the trained ranker",
    )
    parser.add_argument("--cand_num", type=int, default=10)
    parser.add_argument("--sample_num", type=int, default=500)
    parser.add_argument("--correct_sample", type=str, default="random")
    parser.add_argument("--dataset_layer", type=int, default=32)
    parser.add_argument("--dataset_type", type=str, default="single")
    parser.add_argument("--task_name", type=str, default="tool")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model_name = args.model_path.split("/")[-1]
    args.save_dir = os.path.join(
        args.save_dir, f"{model_name}_cand{args.cand_num}_half{args.half}"
    )

    train_ranker_in_batches(args)


if __name__ == "__main__":
    main()
