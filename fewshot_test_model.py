from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, AdamW
from mtl_model import MTLRobertaForSequenceClassification
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import json
import random
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler, Dataset
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from utils import get_base_path
import os
import sys
from pathlib import Path
from fewshot_dataloader import load_data, get_dataset
from configs import get_args
from utils import get_fewshot_result_path, create_logger, map_n_to_budget
import wandb


def evaluate(data_loader, model, device):
    predictions = []
    test_labels = []
    test_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            # TODO set classification head
            outputs = model.forward(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            test_loss += loss.item()

            logits = outputs.logits
            predicted = torch.argmax(logits, dim=1)
            predictions.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    # import IPython; IPython.embed()
    avg_loss = test_loss / len(data_loader)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average='binary')
    auc = roc_auc_score(test_labels, predictions)
    return [precision, recall, f1, auc, test_loss]


# Generate and save the evaluation result

def main(args, base_models_path):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    config = RobertaConfig.from_pretrained(
        args.model_name, num_labels=2)  # add label2id id2label!

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    df_test = load_data(args, 'test',  annotators=args.few_shot_task)

    few_shot_test_dataset = get_dataset(
        args, df_test, split='test', tokenizer=tokenizer)
    test_data_loader = DataLoader(
        few_shot_test_dataset, shuffle=True, batch_size=32)
    test_file_name = "test_high_dis_result.json" if args.test_high_dis else "test_result.json"
    # the shot to run for
    k_shots = args.k_shot.split(",")

    for shot in k_shots:
        args.k_shot = int(shot)
        task_result_path = get_fewshot_result_path(args)
        
        for model_path in Path(task_result_path).rglob('*/model'):
            model_path = str(model_path)
            report_path = Path(model_path).parent / test_file_name

            if not os.path.exists(report_path):
                model = RobertaForSequenceClassification.from_pretrained(model_path, config=config)
                model.to(device)
                test_precision, test_recall, test_f1, test_auc, test_loss = evaluate(
                    test_data_loader, model, device)
                test_result = {'precision': test_precision, 'recall': test_recall,
                            'f1': test_f1, 'auc': test_auc, 'loss': test_loss}
                with open(report_path, "w") as report_file:
                    json.dump(test_result, report_file, indent=4)
                print(f"Finished evaluating {model_path}")
        
if __name__ == "__main__":
    args = get_args()
    args.budget = map_n_to_budget(args.dataset, args.n_mtl_tasks)

    base_models_path = f"results/{args.model_name}/{args.dataset}/{args.label}/seed_{args.seed}/budget_{args.budget}/mtl_{args.n_mtl_tasks}"

    # wandb_run = wandb.init(tags=[
    #     f"DATASET_{args.dataset}",
    #     f"annotator_{args.few_shot_task}",
    #     f"budget{args.budget}",
    #     f"shot_{args.k_shot}"], project="MTL2DIS")

    main(args, base_models_path)

    # wandb_run.finish()
