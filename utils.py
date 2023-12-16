import logging
import numpy as np
import random
import torch
import os
import yaml
from yaml.loader import SafeLoader
import os
import json
from shutil import copyfile, rmtree, copytree


def create_logger(save_path, log_level=logging.INFO):
    # Create a logger instance
    logger = logging.getLogger(__name__)

    # Set the log level
    logger.setLevel(log_level)

    # Create a formatter for the log messages
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a file handler to log to the specified file
    file_handler = logging.FileHandler(save_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    # file_handler.flush = True
    # Create a console handler to log to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def map_n_to_budget(dataset, n):
    if dataset == 'mfrc':
        if n == 6:
            return 0.25
        elif n == 12:
            return 0.5
        elif n == 18:
            return 0.75
        elif n == 24:
            return 1.0
    elif dataset == 'brexit':
        if n == 3:
            return 0.5
        elif n == 4:
            return 0.66
        elif n == 5:
            return 0.83
        elif n == 6:
            return 1.0


def load_yaml_file(file):
    with open(file, "r") as f:
        data = yaml.load(f, Loader=SafeLoader)
    return data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def get_hp_tuning_result_path(args):
    if not args.few_shot:
        result_path = os.path.join(
            f"results/{args.model_name}/{args.dataset}/{args.label}/seed_{args.seed}/budget_{args.budget}/mtl_{len(args.mtl_tasks.split(','))}/mtl_{args.mtl_tasks}_{args.k_shot}",
            "hyperparam_tuning",
            f"{args.train_batch_size}/balanced_{args.balance_ratio}/{args.lr}/{args.weight_decay}")
        result_path = os.path.join(get_base_path(), result_path)
    else:
        NotImplementedError

    os.makedirs(result_path, exist_ok=True)
    return result_path


def get_mtl_best_model_path(args):
    result_path = os.path.join(f"results/{args.model_name}/{args.dataset}/{args.label}/seed_{args.seed}/budget_{args.budget}/mtl_{len(args.mtl_tasks.split(','))}/mtl_{args.mtl_tasks}_{args.k_shot}",
                               'best_model')
    result_path = os.path.join(get_base_path(), result_path)
    return result_path


def get_mtl_result_path(args):
    result_path = f"results/{args.model_name}/{args.dataset}/{args.label}/seed_{args.seed}/budget_{args.budget}/mtl_{len(args.mtl_tasks.split(','))}/mtl_{args.mtl_tasks}_{args.k_shot}"
    result_path = os.path.join(get_base_path(), result_path)
    return result_path


def get_fewshot_result_path(args):
    task_base_result_path = f"results/{args.model_name}/{args.dataset}/{args.label}/seed_{args.seed}/budget_{args.budget}/few_shot/strategy_{args.few_shot_sample_strategy}/{'sampler' if args.balance_ratio > 0 else 'wo_sampler'}/{args.k_shot}/{args.few_shot_task}"
    task_base_result_path = os.path.join(
        get_base_path(), task_base_result_path)
    os.makedirs(task_base_result_path, exist_ok=True)
    return task_base_result_path


def get_base_path():

    script_directory = os.path.dirname(os.path.abspath(__name__))

    # Go up to the main directory from the script directory
    main_directory = script_directory
    while not main_directory.endswith('MTLDIS_NAACL_2024'):
        main_directory = os.path.abspath(os.path.join(main_directory, '..'))

    return main_directory


def keep_best_model(args):
    destination_dir = f"results/{args.model_name}/{args.dataset}/{args.label}/seed_{args.seed}/budget_{args.budget}/mtl_{len(args.mtl_tasks.split(','))}/mtl_{args.mtl_tasks}_{args.k_shot}"
    source_dir = os.path.join(destination_dir, "hyperparam_tuning")

    best_model_path = None
    best_avg_val_f1 = float('-inf')

    # Traverse through the subdirectories
    for current_directory, subdirectories, files in os.walk(source_dir):
        # Check if there is a "model" file in the current directory
        if 'model' in subdirectories:
            # Load the "test_result.json" file
            test_result_path = os.path.join(
                current_directory, 'val_result.json')
            if os.path.exists(test_result_path):
                with open(test_result_path, 'r') as f:
                    test_result_data = json.load(f)

                # Extract the relevant metric (e.g., avg_val_f1)
                avg_val_f1 = test_result_data.get('avg_val_f1', 0.0)

                # Update the best model if needed
                if avg_val_f1 > best_avg_val_f1:
                    best_avg_val_f1 = avg_val_f1
                    best_model_path = current_directory

    # Copy the best model and "test_result.json" to the destination directory
    if best_model_path:
        copytree(os.path.join(best_model_path, 'model'),
                 os.path.join(destination_dir, 'best_model'))
        copyfile(os.path.join(best_model_path, 'val_result.json'),
                 os.path.join(destination_dir, 'val_result.json'))
        copyfile(os.path.join(best_model_path, 'log.log'),
                 os.path.join(destination_dir, 'log.log'))

    # Delete subdirectories and their contents
    rmtree(source_dir)
