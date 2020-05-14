import random
import numpy as np
import torch
import logging
import os
import json
import copy

from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from arguments import parse_args
from tqdm import tqdm, trange
from collections import OrderedDict
from torch.utils.data.dataloader import default_collate

from transformers import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes
from transformers import glue_processors

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from utils import convert_dict
from MAS.mas_utils import init_reg_params, shared_model, exp_lr_scheduler, compute_omega_grads_norm, \
    init_reg_params_across_tasks, consolidate_reg_params, sanity_model
from MAS.optimizer_lib import omega_update_Adam, local_AdamW

logger = logging.getLogger(__name__)


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def partial_name(args):
    task_type = ""
    for key in args.task_params:
        task_type += str(key)+"_"
    if args.ewc:
        part_name = "ewc_tasktype_{}lambda{}_seed_{}_epochs_{}_ngpu_{}".format(task_type,
                                                                             args.ewc_lamda,
                                                                             args.seed,
                                                                             args.num_train_epochs,
                                                                             args.n_gpu)
    elif args.mas:
        part_name = "mas_tasktype_{}lambda{}_initlr_{}_seed_{}_epochs_{}_ngpu_{}".format(task_type,
                                                                             args.mas_lamda,
                                                                             args.mas_init_lr,
                                                                             args.seed,
                                                                             args.num_train_epochs,
                                                                             args.n_gpu)
    else:
        part_name = "normal_tasktype_{}seed_{}_epochs_{}_ngpu_{}".format(task_type,
                                                                      args.seed,
                                                                      args.num_train_epochs,
                                                                      args.n_gpu)
    return part_name

def save_model(args, task_num, model):
    part_name = partial_name(args)
    if not os.path.exists(os.path.join(args.output_dir, part_name)):
        os.makedirs(os.path.join(args.output_dir, part_name))

    task_paramters = OrderedDict()
    if args.mas:
        bert_paramters = model.tmodel.state_dict().copy()
    else:
        bert_paramters = model.state_dict().copy()

    for layer, weights in bert_paramters.items():
        if layer == "classifier.weight" or layer == 'classifier.bias':
            task_paramters[layer] = weights
    del bert_paramters["classifier.weight"]
    del bert_paramters['classifier.bias']

    torch.save(bert_paramters, os.path.join(args.output_dir, part_name, "bert_paramters_" + str(task_num) + ".pt"))
    torch.save(task_paramters, os.path.join(args.output_dir, part_name, "task_paramters_" + str(task_num) + ".pt"))

    print()
    print("***** Parameters Saved for task", task_num, "*****")
    print()


def compute_ewc_loss(args, lamda, task_num, task, model, consolidate_fisher, consolidate_mean):
    # EWC Loss is computed only on BERT parameters (not on task specific parameters)
    # bert_paramters = model.state_dict()
    # del bert_paramters["classifier.weight"]
    # del bert_paramters['classifier.bias']

    loss_ewc = torch.tensor(0.0, requires_grad=True).to(args.device)
    for name, params in model.named_parameters():
        if not (name == "classifier.weight" or name == "classifier.bias"):
            # print(name)
            mean = consolidate_mean[task][name]
            fisher = consolidate_fisher[task][name]
            loss_ewc_layer = fisher * (params - mean) ** 2
            loss_ewc += loss_ewc_layer.sum()

    return (lamda / 2) * loss_ewc


def estimate_fisher_mean(args, train_dataset, model, task, fisher_consolidate, mean_consolidate, batch_size=32,
                         collate_fn=None):
    set_seed(args.seed, args.n_gpu)
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=(collate_fn or default_collate))

    mean_consolidate[task] = {}
    fisher_consolidate[task] = {}

    for n, p in model.named_parameters():
        mean_consolidate[task][n] = p.data.to(args.device)

    for n, p in model.named_parameters():
        fisher_consolidate[task][n] = torch.zeros(p.size()).to(args.device)

    model.eval()
    # new_model = copy.deepcopy(model).cpu()
    for x, p, q, y in data_loader:
        model.zero_grad()
        x = Variable(x).to(args.device)
        y = Variable(y).to(args.device)
        p = Variable(p).to(args.device)
        q = Variable(q).to(args.device)
        # x = tuple(t.to(args.device) for t in x)
        inputs = {"input_ids": x, "attention_mask": p, "token_type_ids": q, "labels": y}
        outputs = model(**inputs)
        loss = outputs[0]
        loss.backward()
        for n, p in model.named_parameters():
            fisher_consolidate[task][n] += p.grad.data ** 2 / len(data_loader)
    # fisher_consolidate[task] = {n: p for n, p in fisher_consolidate[task].items()}

    return fisher_consolidate, mean_consolidate


def train(args, train_dataset, task, all_tasks, model, task_num, tokenizer, accuracy_matrix,
          consolidate_fisher, consolidate_mean):
    """ Train the model """
    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    t_total = len(train_dataloader) * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    if args.mas:
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.tmodel.named_parameters() if not any(nd in n for nd in no_decay)],
                "names": [n for n, p in model.tmodel.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.tmodel.named_parameters() if any(nd in n for nd in no_decay)],
             "names": [n for n, p in model.tmodel.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
    else:
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

    if args.mas:
        if task_num == 0:
            model = init_reg_params(model, args.device)
        else:
            init_reg_params_across_tasks(model, args.device)
        optimizer = local_AdamW(optimizer_grouped_parameters, args.mas_lambda, lr=args.mas_init_lr, eps=args.adam_epsilon)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size,
    )
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=False,
    )

    set_seed(args.seed, args.n_gpu)

    for _ in train_iterator:
        # epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):
            if args.mas:
                model.tmodel.zero_grad()
                model.tmodel.train()
            else:
                model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3]
            }
            '''if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                )'''  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            if args.mas:
                outputs = model.tmodel(**inputs)
            else:
                outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            # Compute EWC loss & Update the total loss (For first task it is just zero)
            if task_num > 0 and args.ewc:
                lamda = args.ewc_lamda
                for i in range(task_num):
                    ewc_loss = compute_ewc_loss(args, lamda/task_num, task_num, all_tasks[i], model,
                                                consolidate_fisher, consolidate_mean)
                # print("EWC:{}, Normal:{}, Total:{}".format(ewc_loss, loss, loss+ewc_loss))
                    loss += ewc_loss
            loss.backward()

            tr_loss += loss.item()

            if args.mas:
                torch.nn.utils.clip_grad_norm_(model.tmodel.parameters(), args.max_grad_norm)
                optimizer.step(model.reg_params)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                model.zero_grad()

            scheduler.step()  # Update learning rate schedule
            global_step += 1
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                current_accuracy = accuracy_matrix[task_num][task_num]
                evaluate(args, model, task, tokenizer, accuracy_matrix, task_num, task_num, "Current Task")
                if current_accuracy>=accuracy_matrix[task_num][task_num]:
                    accuracy_matrix[task_num][task_num] = current_accuracy
                else:
                    save_model(args, task_num, model)
                    new_model_state = model.state_dict().copy()
                    # Evaluating on all tasks - both forward and backward transfer
                    for i in range(len(all_tasks)):
                        part_name = partial_name(args)
                        # Previous tasks
                        if (i < task_num):
                            if args.mas:
                                model.tmodel.load_state_dict(
                                    torch.load(os.path.join(args.output_dir, part_name, "task_paramters_" + str(i) + ".pt")),
                                    strict=False)
                            else:
                                model.load_state_dict(
                                    torch.load(os.path.join(args.output_dir, part_name, "task_paramters_" + str(i) + ".pt")),
                                    strict=False)
                            results, accuracy_matrix = evaluate(args, model, all_tasks[i], tokenizer, accuracy_matrix,
                                                                task_num, i,
                                                                "Previous Task (Continual)")
                        # Future tasks
                        elif (i > task_num):
                            if args.mas:
                                model.tmodel.load_state_dict(
                                    torch.load(os.path.join(args.output_dir, part_name, "task_paramters_" + str(i) + ".pt")),
                                    strict=False)
                            else:
                                model.load_state_dict(
                                    torch.load(os.path.join(args.output_dir, part_name, "task_paramters_" + str(i) + ".pt")),
                                    strict=False)
                            results, accuracy_matrix = evaluate(args, model, all_tasks[i], tokenizer, accuracy_matrix,
                                                                task_num, i,
                                                                "Future Task (Continual)")
                    model.load_state_dict(new_model_state)

    if args.mas:
        # optimizer_ft = omega_update_Adam(model.reg_params)
        # model = compute_omega_grads_norm(model, train_dataloader, optimizer_ft, args.device)
        # sanity_model(model)
        if task_num >= 1:
            model = consolidate_reg_params(model)
    logs = {}
    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
    learning_rate_scalar = scheduler.get_lr()[0]
    logs["learning_rate"] = learning_rate_scalar
    logs["loss"] = loss_scalar
    logging_loss = tr_loss

    for key, value in logs.items():
        tb_writer.add_scalar(key, value, global_step)
    print(json.dumps({**logs, **{"step": global_step}}))

    '''if args.save_steps > 0 and global_step % args.save_steps == 0:
        # Save model checkpoint
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  
        # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)

        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to %s", output_dir)'''
    tb_writer.close()
    part_name = partial_name(args)
    model.load_state_dict(torch.load(os.path.join(args.output_dir, part_name, "bert_paramters_" + str(task_num) + ".pt")),
                            strict=False)
    model.load_state_dict(torch.load(os.path.join(args.output_dir, part_name, "task_paramters_" + str(task_num) + ".pt")),
                            strict=False)
    return global_step, tr_loss / global_step, accuracy_matrix, model


def evaluate(args, model, task, tokenizer, accuracy_matrix, train_task_num, current_task_num, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if task == "mnli" else (task,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if task == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(os.path.join(eval_output_dir, prefix)):
            os.makedirs(os.path.join(eval_output_dir, prefix))

        args.eval_batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = RandomSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation:: Task : {}, Prefix : {} *****".format(task, prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in eval_dataloader:
            if args.mas:
                model.tmodel.eval()
            else:
                model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2],
                          "labels": batch[3]}
                '''if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids '''
                if args.mas:
                    outputs = model.tmodel(**inputs)
                else:
                    outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            print(result)
            accuracy_matrix[train_task_num][current_task_num] = format(result['acc'], ".2f")
        # for key in sorted(result.keys()):
        #	logger.info("  %s = %s", key, str(result[key]))
        #	writer.write("%s = %s\n" % (key, str(result[key])))

    return results, accuracy_matrix


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    processor = glue_processors[task]()
    output_mode = glue_output_modes[task]

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            args.model_type,
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(os.path.join(args.data_dir, task)) if evaluate else processor.get_train_examples(
                os.path.join(args.data_dir, task))
        )
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def main():
    args = parse_args()

    print(args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    set_seed(args.seed, args.n_gpu)

    # Prepare GLUE tasks
    processors = {}
    output_modes = {}
    label_lists = {}
    num_label_list = {}
    for key in args.task_params:
        print(processors)
        processors[key] = glue_processors[key]()
        output_modes[key] = glue_output_modes[key]
        label_lists[key] = processors[key].get_labels()
        num_label_list[key] = len(label_lists[key])

    # Configs
    configs = {}
    for key in args.task_params:
        configs[key] = AutoConfig.from_pretrained(args.model_type,
                                                  # args.config_name if args.config_name else args.model_name_or_path,
                                                  num_labels=num_label_list[key],
                                                  finetuning_task=key,
                                                  cache_dir=None,
                                                  )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_type,
        do_lower_case=args.do_lower_case,
        cache_dir=None,
    )

    # Continual Learning
    n = len(configs)
    accuracy_matrix = np.zeros((n, n))
    transfer_matrix = np.zeros((n, n))

    tasks = list(args.task_params.keys())
    models = []
    # Model
    for key in args.task_params:
        if args.n_gpu<=1:
            if args.mas:
                models.append((key, shared_model(AutoModelForSequenceClassification.from_pretrained(args.model_type, config=configs[key]))))
            else:
                models.append((key, AutoModelForSequenceClassification.from_pretrained(args.model_type, config=configs[key])))
        else:
            if args.mas:
                models.append((key, shared_model(torch.nn.DataParallel(AutoModelForSequenceClassification.from_pretrained(args.model_type, config=configs[key])))))
            else:
                models.append((key, torch.nn.DataParallel(AutoModelForSequenceClassification.from_pretrained(args.model_type, config=configs[key]))))

    for i in range(n):
        models[i][1].to(args.device)
        save_model(args, i, models[i][1])
    if args.cuda == torch.device('cuda') and torch.cuda.device_count() > 1:
        for key in models:
            models[key] = torch.nn.DataParallel(models[key])
    consolidate_fisher = {}
    consolidate_mean = {}

    for i in range(len(configs)):
        if (i > 0):
            part_name = partial_name(args)
            # Always load the BERT parameters of previous model
            if args.mas:
                models[i][1].tmodel.load_state_dict(
                    torch.load(os.path.join(args.output_dir, part_name, "bert_paramters_" + str(i - 1) + ".pt")), strict=False)
                models[i][1].reg_params = models[i - 1][1].reg_params
            else:
                models[i][1].load_state_dict(
                    torch.load(os.path.join(args.output_dir, part_name, "bert_paramters_" + str(i - 1) + ".pt")), strict=False)
        new_args = convert_dict(args.task_params[tasks[i]], args)
        train_dataset = load_and_cache_examples(args, tasks[i], tokenizer, evaluate=False)
        # --- No randomness till here ---
        global_step, tr_loss, accuracy_matrix, model = train(new_args, train_dataset, tasks[i], tasks, models[i][1], i,
                                                            tokenizer, accuracy_matrix, consolidate_fisher, consolidate_mean)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        if args.ewc:
            if (i == (len(configs)-1)):
                print("\n ***** Last Task : No need to estimate fisher matrix ***** \n")
                break
            else:
                print()
                print("***** Estimating Diagonals of Fisher Information Matrix *****")
                print()
                consolidate_fisher, consolidate_mean = estimate_fisher_mean(new_args, train_dataset, model, tasks[i],
                                                                            consolidate_fisher, consolidate_mean,
                                                                            args.per_gpu_batch_size)

    print("\n ***** Accuracy Matrix *****\n")
    print(accuracy_matrix)
    print("\n***** Transfer Matrix *****")
    print("Future Transfer => Upper Triangular Matrix  ||  Backward Transfer => Lower Triangular Matrix\n")

    for i in range(n):
        for j in range(n):
            transfer_matrix[j][i] = accuracy_matrix[j][i] - accuracy_matrix[i][i]

    print(transfer_matrix)


if __name__ == '__main__':
    main()
