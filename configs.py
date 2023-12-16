import argparse


def merge_args_into_config(args, config):
    config.tasks = args.mtl_tasks.split(',')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dry_run', action='store_true')

    # names
    parser.add_argument('--model_name', default='roberta-base',
                        choices=['roberta-base', 'bert-base-uncased', 'roberta-large'])
    parser.add_argument(
        '--dataset', choices=['ghc', 'brexit', 'mfrc'], default='brexit')
    parser.add_argument('--label', default='Hate')
    parser.add_argument('--text_col', default='text')
    parser.add_argument('--id_col', default='text_id')
    parser.add_argument(
        '--criteria', help='a list of comma separated criteria', default=None)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--val_result_file_name', default='val_result.json')

    # mtl tasks
    parser.add_argument('--mtl_tasks', default="Ann1,Ann2,Ann3,Ann4,Ann5,Ann6",
                        help='a list of comma seperated tasks')
    parser.add_argument('--baseline', action='store_true', help='run baseline')

    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--predict_batch_size', default=64, type=int)
    parser.add_argument('--load_val', action='store_true')

    # hyperparams
    parser.add_argument('--run_sweep', action='store_true')
    parser.add_argument('--load_hp', action='store_true',
                        help='enable loading saved hyperparameters')
    parser.add_argument('--lr', default=2.0e-05, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--balance_ratio', default=0.5, type=float,
                        help='put a ration to balance the sampler with, if 0 not balanced')
    parser.add_argument('--sqrt', action='store_true')

    # fewshot params
    parser.add_argument('--few_shot', action='store_true')
    parser.add_argument('--k_shot', default="64",
                        help='a list of comma seperated shots')
    parser.add_argument('--n_fewshot_tasks', default=3, type=int)
    parser.add_argument('--n_mtl_tasks', default=3, type=int)
    parser.add_argument('--few_shot_sample_strategy',
                        default='mv')
    parser.add_argument('--few_shot_task',
                        help='a list of comma seperated tasks')
    parser.add_argument('--freeze_roberta', action='store_true')

    parser.add_argument('--split', default='test')
    # budget of annotation
    parser.add_argument('--budget', default=2352, type=float)
    parser.add_argument('--dataset_train_size', default=784, type=float)

    # test params
    parser.add_argument('--test_high_dis', action='store_true',
                        help='test on high disagreement samples')
    args = parser.parse_args()

    return args
