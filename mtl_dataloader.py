

from torch.utils.data import Dataset
import torch
import random
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler, Dataset
import torch
import pandas as pd
from utils import get_base_path
import os


class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.targets = labels

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        input_id = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        label = self.targets[idx]

        return input_id, attention_mask, label


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
    axis=0,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=axis)
    if attention_mask is None:
        return input_ids[:, keep_column_mask] if axis == 0 else input_ids[keep_column_mask, :]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class MTLDataloader(DataLoader):
    def __init__(self, dummy_dataset, *args, **kwargs):
        data_loaders = kwargs.pop('data_loaders')
        mtl_bal_sampling = kwargs.pop('mtl_bal_sampling')
        self.task_keys = kwargs.pop('task_keys')
        self.task_num = kwargs.pop('task_num')
        self.pad_token_id = kwargs.pop('pad_token_id')
        self.sqrt = kwargs.pop('sqrt')
        super().__init__(dummy_dataset, *args, **kwargs)
        self.data_loaders = data_loaders
        self.mtl_bal_sampling = mtl_bal_sampling
        self.loader_len = None

        # just for random sampling of examples from a task
        self.task_iterators = [iter(x) for x in self.data_loaders]

    def __iter__(self):

        def mtl_data_iterator():
            draws = []
            for i in range(self.task_num):
                draws.extend([i] * len(self.data_loaders[i]))
            iterators = [iter(_) for _ in self.data_loaders]
            random.shuffle(draws)
            self.loader_len = len(draws)
            for loader_id in draws:
                iterator = iterators[loader_id]
                yield next(iterator)

        def mtl_bal_data_iterator():
            draws = []
            max_dataloader_len = max([len(x) for x in self.data_loaders])
            for i in range(self.task_num):
                if self.sqrt:
                    # x : max_dataloader_len = sqrt(len(x)) : sqrt(len(max_dataloader_len))
                    batch_num = int(
                        max_dataloader_len * (len(self.data_loaders[i]) ** 0.5) // (max_dataloader_len ** 0.5))
                    draws.extend([i] * batch_num)
                else:
                    draws.extend([i] * max_dataloader_len)
            iterators = [iter(_) for _ in self.data_loaders]
            random.shuffle(draws)
            self.loader_len = len(draws)
            for loader_id in draws:
                task_name = self.task_keys[loader_id]
                iterator = iterators[loader_id]
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterators[loader_id] = iter(self.data_loaders[loader_id])
                    iterator = iterators[loader_id]
                    batch = next(iterator)
                yield (loader_id, task_name), batch

        if self.mtl_bal_sampling:
            return mtl_bal_data_iterator()
        else:
            return mtl_data_iterator()

    def __len__(self):
        return self.loader_len
        # return len(dummy_dataset)


def select_hig_dis(args, df):
    if args.dataset == 'brexit':
        df = df[df['aggrement'] <= 2].reset_index(drop=True)
        # df = df[df['agree_mv'] == 0]
    elif args.dataset == 'mfrc':
        df = df[df['aggrement'] <= 11.75].reset_index(drop=True)

    return df


def get_dataset(args, task_key, split, tokenizer):
    # import IPython; IPython.embed()
    data_file = f"data/{args.dataset}/{args.label}/annotators/{task_key}/{split}.csv"
    data_file = os.path.join(get_base_path(), data_file)
    df = pd.read_csv(data_file)

    if args.dry_run:
        df = df.sample(100, random_state=args.seed).reset_index(drop=True)
    # if running for baseline then sample the data as budget / number of tasks
    if split == 'train':
        if args.baseline:
            sample_size = float(args.budget) * len(args.mtl_tasks.split(",")) * \
                args.dataset_train_size // len(args.mtl_tasks.split(","))
            sample_size = int(sample_size)

            df = df.sample(
                sample_size, random_state=args.seed).reset_index(drop=True)

    elif split == 'test':
        if args.test_high_dis:
            df = select_hig_dis(args, df)

    texts = df[args.text_col].tolist()
    labels = df[args.label].tolist()
    encoded_texts = tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt")
    input_ids = encoded_texts["input_ids"]
    attention_mask = encoded_texts['attention_mask']
    # labels = torch.tensor(labels)
    dataset = CustomDataset(input_ids, attention_mask, labels)

    return dataset


class MTLTasks:

    def __init__(self, mtl_tasks, args, tokenizer, few_shot=False):
        self.args = args
        self.dataset = args.dataset
        # self.k_shot = args.k_shot
        self.tasks = mtl_tasks
        self.tokenizer = tokenizer
        self.encoded_dataset = defaultdict(dict)
        self.data_loader_maps = defaultdict(dict)

        self.few_shot = few_shot

        # this is for balancing the dataset
        self.balance_ratio = args.balance_ratio
        splits = ['train', 'test', 'val']

        self.creat_dataset_dict(args, splits)

        self.creat_dataloader_map(args, splits)

    def creat_dataset_dict(self, args, splits):
        for task_key in self.tasks:
            for split in splits:
                dataset = get_dataset(args, task_key, split, self.tokenizer)
                self.encoded_dataset[task_key][split] = dataset

    def creat_dataloader_map(self, args, splits):
        for task_key in self.encoded_dataset:
            for split in splits:
                shuffle = split == 'train'

                batch_size = args.train_batch_size if split == 'train' else args.predict_batch_size

                data = self.encoded_dataset[task_key][split]

                # weighted sampler to handle heavy dataset imbalance

                pos_ratio = np.sum(data.targets) / len(data)
                if self.balance_ratio > 0 and split == 'train' and pos_ratio < self.balance_ratio:
                    # logger.info(
                    #     f"Using weighted random sampler for Task {task_key}, positive ratio ={pos_ratio}")
                    # # 50/50
                    perfect_balance_weights = [
                        1.0/(1-pos_ratio), 1.0/pos_ratio]
                    class_wieghts = [(1-self.balance_ratio)*perfect_balance_weights[0],
                                     self.balance_ratio*perfect_balance_weights[1]]
                    sample_weights = [class_wieghts[t]
                                      for t in data.targets]

                    w_sampler = WeightedRandomSampler(
                        sample_weights, len(data.targets), replacement=True)
                    data_loader = DataLoader(
                        data, batch_size=batch_size, sampler=w_sampler)
                else:
                    data_loader = DataLoader(
                        data, shuffle=shuffle, batch_size=batch_size)
                self.data_loader_maps[task_key][split] = data_loader

    def get_mtl_dataloader(self, split):
        dummy_dataset = list(self.encoded_dataset.values())[0][split]
        shuffle = split == 'train'
        mtl_bal_sampling = split == 'train'
        batch_size = self.args.train_batch_size if split == 'train' else self.args.predict_batch_size
        data_loaders = []
        for task_id, task_key in enumerate(self.data_loader_maps):
            if task_id < len(self.tasks):
                data_loader = self.data_loader_maps[task_key][split]
                data_loaders.append(data_loader)

        mtl_dataloader = MTLDataloader(dummy_dataset, shuffle=shuffle, batch_size=batch_size,
                                       data_loaders=data_loaders, task_keys=self.tasks, mtl_bal_sampling=mtl_bal_sampling,
                                       task_num=len(self.tasks), pad_token_id=self.tokenizer.pad_token_id, sqrt=self.args.sqrt
                                       )
        return mtl_dataloader

    def get_dataloader_sequence_iterator(self):
        for task_key in self.data_loader_maps:
            if 'val' in self.data_loader_maps[task_key]:
                data_loaders = [self.data_loader_maps[task_key][split]
                                for split in ['train', 'val', 'test']]
            else:
                data_loaders = [self.data_loader_maps[task_key][split]
                                for split in ['train', 'test', 'test']]
            yield task_key, data_loaders
