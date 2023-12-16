
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
from utils import (create_logger, set_seed, get_mtl_result_path,
                   get_hp_tuning_result_path, keep_best_model,
                   load_yaml_file, get_mtl_best_model_path)
from mtl_dataloader import MTLDataloader, MTLTasks
import wandb
from collections import defaultdict


def mtl_train(args, model, mtl_train_dataloader, device, optimizer):
    losses = []
    for batch in tqdm(mtl_train_dataloader):
        (model_task_id, model_task_name), batch = batch
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        model.set_classification_head(model_task_name)
        outputs = model.forward(input_ids=input_ids,
                                attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return model, np.mean(losses)


def get_eval_metrics(model, data_loader, device):
    test_predictions = []
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
            test_predictions.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    # import IPython; IPython.embed()
    avg_loss = test_loss / len(data_loader)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_predictions,
                                                               average='binary')
    auc = roc_auc_score(test_labels, test_predictions)
    return [precision, recall, f1, auc, test_loss]


def evaluate(task_iterator, model, device, split, logger):
    model.eval()
    task_metrics_dict = {'task_name': [], 'f1': [], 'val_loss': [], 'auc': [],
                         'precision': [], 'recall': []}
    for (task_name, (train_loader, dev_loader, test_loader)) in tqdm(task_iterator):
        model.set_classification_head(task_name)

        if split == "test":
            precision, recall, f1, auc, val_loss = get_eval_metrics(
                model, test_loader, device)
        elif split == "val":
            precision, recall, f1, auc, val_loss = get_eval_metrics(
                model, dev_loader, device)
           # Print metrics for the current task
        logger.info(
            f"{task_name} - F1: {f1:.4f}, {split}_loss: {val_loss:.4f}")
        # Store metrics in the dictionary
        task_metrics_dict['task_name'].append(task_name)
        task_metrics_dict['f1'].append(f1)
        task_metrics_dict['val_loss'].append(val_loss)
        task_metrics_dict['auc'].append(auc)
        task_metrics_dict['precision'].append(precision)
        task_metrics_dict['recall'].append(recall)

    avg_val_loss = np.mean(task_metrics_dict['val_loss'])
    avg_val_f1 = np.mean(task_metrics_dict['f1'])

    logger.info(f"Avergage {split} Loss: {avg_val_loss:.4f}")

    logger.info(
        f"{split} result:  precision {np.mean(task_metrics_dict['precision']):.4f},recall {np.mean(task_metrics_dict['recall']):.4f}, f1  {np.mean(task_metrics_dict['f1']):.4f}, auc {np.mean(task_metrics_dict['auc']):.4f}")

    result_dict = {"avg_val_loss": avg_val_loss,
                   "avg_val_precision": np.mean(task_metrics_dict['precision']),
                   "avg_val_recall": np.mean(task_metrics_dict['recall']),
                   "avg_val_f1": avg_val_f1,
                   "avg_val_auc": np.mean(task_metrics_dict['auc'])}

    result_dict.update(task_metrics_dict)

    return result_dict


def run(args, result_path):

    logger = create_logger(save_path=os.path.join(result_path, "log.log"))

    config = RobertaConfig.from_pretrained(
        args.model_name, num_labels=2)  # add label2id id2label!

    merge_args_into_config(args, config)

    model = MTLRobertaForSequenceClassification.from_pretrained(
        args.model_name, config=config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # create mtl tasks
    mtl_tasks = args.mtl_tasks.split(",")
    main_tasks = MTLTasks(args=args, mtl_tasks=mtl_tasks,
                          tokenizer=tokenizer, few_shot=False)

    mtl_train_dataloader = main_tasks.get_mtl_dataloader(
        split='train')

    # tasks dataloader iterator

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)

    # Initialize variables to keep track of the best model and its validation loss
    # best_model_state_dict = model.state_dict()
    best_avg_val_f1 = float('-inf')
    best_result_dict = defaultdict()
    best_task_metrics_dict = defaultdict()
    # start training
    for epoch in range(args.epochs):
        log_dict = {}
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")

        model.train()
        model, loss = mtl_train(
            args, model, mtl_train_dataloader, device, optimizer)

        logger.info(f"Train Loss: {loss:.4f}")
        log_dict.update({"epoch": epoch,
                         "train_loss": loss})
        # wandb.log({"epoch": epoch, "train_loss": loss})

        # validation loop

        model.eval()
        task_iterator = main_tasks.get_dataloader_sequence_iterator()
        result_dict = evaluate(task_iterator, model, device, "val", logger)
        log_dict.update(result_dict)
        # wandb.log(result_dict)

        # Check if the current model has a lower validation loss than the best model
        if result_dict['avg_val_f1'] > best_avg_val_f1:
            best_avg_val_f1 = result_dict['avg_val_f1']
            best_result_dict = result_dict
            model.save_pretrained(f"{result_path}/model")
            # best_model_state_dict = model.state_dict()

        wandb.log(log_dict)

    config_dict = {'lr': args.lr, 'train_batch_size': args.train_batch_size,
                   'balance_ratio': args.balance_ratio, 'weight_decay': args.weight_decay}

    best_result_dict.update(config_dict)

    with open(f"{result_path}/{args.val_result_file_name}", "w") as report_file:
        json.dump(best_result_dict, report_file, indent=4)


def sweep_main():

    model_with_current_hps = get_hp_tuning_result_path(args)
    if not os.path.exists(os.path.join(model_with_current_hps, 'model')):
        wandb_run = wandb.init(tags=[
            f"DATASET_{args.dataset}",
            # f"tasks_{args.mtl_tasks}",
            f"budget{args.budget}"], project="MTL2DIS")
        config = wandb.config
        args.lr = config.lr
        args.weight_decay = config.weight_decay

        # args.balance_ratio = config.balance_ratio

        wandb.log({"lr": args.lr, "train_batch_size": args.train_batch_size,
                   'balance_ratio': args.balance_ratio, 'tasks': args.mtl_tasks})

        run(args, model_with_current_hps)

        wandb_run.finish()


def main():
    model_with_current_hps = get_hp_tuning_result_path(args)
    if not os.path.exists(os.path.join(model_with_current_hps, 'model')):
        wandb_run = wandb.init(tags=[
            f"DATASET_{args.dataset}",
            # f"tasks_{args.mtl_tasks}",
            f"budget{args.budget}"], project="MTL2DIS")

        # args.balance_ratio = config.balance_ratio
        wandb.log({"lr": args.lr, "train_batch_size": args.train_batch_size,
                   'balance_ratio': args.balance_ratio, 'tasks': args.mtl_tasks})

        run(args, model_with_current_hps)

        # wandb_run.finish()


if __name__ == "__main__":

    args = get_args()
    combinations = args.mtl_tasks.split("-")
    for comb in combinations:
        args.mtl_tasks = comb
        # check if the best mtl model for currect combination of tasks exist then skip running
        best_model_for_current_tasks = get_mtl_best_model_path(args)
        # import IPython; IPython.embed()
        # print(args.seed)
        if not os.path.exists(best_model_for_current_tasks):

            set_seed(args.seed)

            if args.run_sweep:
                # initialize wandb
                sweep_config = load_yaml_file("yaml_configs/fix_br.yaml")

                sweep_config['name'] = f"{args.dataset}_{args.budget}_{args.mtl_tasks}"
                sweep_id = wandb.sweep(sweep_config, project="MTL2DIS")
                # run the sweep
                wandb.agent(sweep_id, function=sweep_main)

                keep_best_model(args)
            else:
                if args.load_hp:
                    hp_path = get_mtl_result_path(args)
                    hp_path = hp_path.replace(f'seed_{args.seed}', 'seed_0')
                    with open(Path(hp_path)/'val_result.json') as f:
                        hp = json.load(f)
                    args.lr = hp['lr']
                    args.weight_decay = hp['weight_decay']
                    print(args.lr, args.weight_decay)
                main()
                keep_best_model(args)
        else:
            print(f"Best model for {args.mtl_tasks} already exists!")
