import argparse
import json
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        type=str,
                        default="../data",
                        required=True,
                        )
    parser.add_argument("--task_params",
                        type=str,
                        required=True,
                        help="JSON file path"
                        )
    parser.add_argument("--output_dir",
                        type=str,
                        default="./output",
                        required=True,
                        )
    parser.add_argument("--per_gpu_batch_size",
                        default=8,
                        type=int,
                        help="Batch size per GPU",
                        )
    parser.add_argument("--cuda",
                        action="store_true",
                        )
    parser.add_argument("--tokenizer_name",
                        type=str,
                        required=True,
                        )
    parser.add_argument("--do_lower_case",
                        action="store_true",
                        )
    parser.add_argument("--model_type",
                        default=None,
                        type=str,
                        required=True,
                        )
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        )
    parser.add_argument("--num_train_epochs",
                        default=1,
                        type=int,
                        )
    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        )
    parser.add_argument("--ewc",
                        action="store_true",
                        default=False
                        )

    args = parser.parse_args()
    with open(args.task_params) as file:
        args.task_params = json.load(file)
    if args.cuda:
        args.device = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
    else:
        args.device = torch.device("cpu")
        args.n_gpu = 0
    return args
