from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from configs import get_args, merge_args_into_config
import argparse
import json
import os
from pathlib import Path    
import datasets
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (classification_report, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (AdamW, AutoTokenizer, BertTokenizer, Trainer)
from transformers import RobertaPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaClassificationHead, RobertaConfig, RobertaPreTrainedModel
from mtl_model import MTLRobertaForSequenceClassification
from utils import (create_logger, set_seed, get_base_path,
                   get_hp_tuning_result_path, keep_best_model,
                   load_yaml_file, get_mtl_best_model_path, get_mtl_result_path)
from mtl_dataloader import MTLDataloader, MTLTasks
import wandb
from mtl_main import get_eval_metrics
from tqdm import tqdm


def main(args, model_path, result_path):
    # print(args.mtl_tasks.split(","))
    config = RobertaConfig.from_pretrained(
        args.model_name, num_labels=2)  # add label2id id2label!

    merge_args_into_config(args, config)

    model = MTLRobertaForSequenceClassification.from_pretrained(
        model_path, config=config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # create mtl tasks
    mtl_tasks = args.mtl_tasks.split(",")

    main_tasks = MTLTasks(args=args, mtl_tasks=mtl_tasks,
                          tokenizer=tokenizer, few_shot=False)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # tasks dataloader iterator
    task_iterator = main_tasks.get_dataloader_sequence_iterator()
    model.eval()
    task_metrics_dict = {'task_name': [], 'f1': [], 'val_loss': [], 'auc': [],
                         'precision': [], 'recall': []}

    for (task_name, (train_loader, dev_loader, test_loader)) in tqdm(task_iterator):
        model.set_classification_head(task_name)
        if args.split == "test":
            precision, recall, f1, auc, val_loss = get_eval_metrics(
                model, test_loader, device)
        elif args.split == "val":
            precision, recall, f1, auc, val_loss = get_eval_metrics(
                model, dev_loader, device)

        # Store metrics in the dictionary
        task_metrics_dict['task_name'].append(task_name)
        task_metrics_dict['f1'].append(f1)
        task_metrics_dict['val_loss'].append(val_loss)
        task_metrics_dict['auc'].append(auc)
        task_metrics_dict['precision'].append(precision)
        task_metrics_dict['recall'].append(recall)

    avg_val_loss = np.mean(task_metrics_dict['val_loss'])
    avg_val_f1 = np.mean(task_metrics_dict['f1'])

    result_dict = {"avg_val_loss": avg_val_loss,
                   "avg_val_precision": np.mean(task_metrics_dict['precision']),
                   "avg_val_recall": np.mean(task_metrics_dict['recall']),
                   "avg_val_f1": avg_val_f1,
                   "avg_val_auc": np.mean(task_metrics_dict['auc'])}

    result_dict.update(task_metrics_dict)

    if args.split == "test":
        if args.test_high_dis:
            file_name = "test_high_dis_result.json"
        else:
            file_name = "test_result.json"
    elif args.split == "val":
        file_name = "val_result.json"

    with open(f"{result_path}/{file_name}", "w") as report_file:
        json.dump(result_dict, report_file, indent=4)


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)
     
    models_path = f"results/{args.model_name}/{args.dataset}/{args.label}/seed_{args.seed}/budget_{args.budget}/"
    models_path = os.path.join(get_base_path(), models_path) 
    for model_path in Path(models_path).rglob('*/best_model'):
        report_file = model_path.parent / 'test_result.json'
    
        if not os.path.exists(report_file):
            args.mtl_tasks = report_file.parent.name.split('_')[1]
            result_path = model_path.parent
            main(args, model_path, result_path)
        else : 
            print(f"already evaluated {report_file}")
   
   
   
