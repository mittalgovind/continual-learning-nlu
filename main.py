import random
import numpy as np
import torch
import logging
import os
import json

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from arguments import parse_args
from tqdm import tqdm, trange
from collections import OrderedDict 

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
from models.bert_finetuning import BertForContinualFineTuning

logger = logging.getLogger(__name__)


def set_seed(seed, n_gpu):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if n_gpu > 0:
		torch.cuda.manual_seed_all(seed)


def save_model(args, task_num, model):

	if not os.path.exists(args.output_dir): 
		os.makedirs(args.output_dir)

	task_paramters = OrderedDict()
	bert_paramters = model.state_dict().copy()

	for layer, weights in bert_paramters.items():
		if layer == "classifier.weight" or layer == 'classifier.bias':
			task_paramters[layer] = weights
	del bert_paramters["classifier.weight"]
	del bert_paramters['classifier.bias']

	torch.save(bert_paramters, os.path.join(args.output_dir, "bert_paramters_" + str(task_num) + ".pt"))
	torch.save(task_paramters, os.path.join(args.output_dir, "task_paramters_" + str(task_num) + ".pt"))

	print()
	print("***** Parameters Saved for task", task_num ,"*****")
	print()


def train(args, train_dataset, task, all_tasks, model, task_num, tokenizer):
	""" Train the model """
	tb_writer = SummaryWriter()

	args.train_batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
	train_sampler = RandomSampler(train_dataset)
	train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
	t_total = len(train_dataloader) * args.num_train_epochs

	# Prepare optimizer and schedule (linear warmup and decay)
	no_decay = ["bias", "LayerNorm.weight"]
	optimizer_grouped_parameters = [
		{
			"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
			"weight_decay": args.weight_decay,
		},
		{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
	]

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
	steps_trained_in_current_epoch = 0

	tr_loss, logging_loss = 0.0, 0.0
	model.zero_grad()
	train_iterator = trange(
		epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=False,
	)
	set_seed(args.seed, args.n_gpu)
	
	for _ in train_iterator:
		epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
		for step, batch in enumerate(epoch_iterator):

			# Skip past any already trained steps if resuming training
			if steps_trained_in_current_epoch > 0:
				steps_trained_in_current_epoch -= 1
				continue

			model.train()
			batch = tuple(t.to(args.device) for t in batch)
			inputs = {"input_ids": batch[0], "attention_mask": batch[1],"token_type_ids":batch[2] , "labels": batch[3]}
			'''if args.model_type != "distilbert":
				inputs["token_type_ids"] = (
					batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
				)'''  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
			outputs = model(**inputs)
			loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

			if args.n_gpu > 1:
				loss = loss.mean()  # mean() to average on multi-gpu parallel training
			loss.backward()

			tr_loss += loss.item()
		
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

			optimizer.step()
			scheduler.step()  # Update learning rate schedule
			model.zero_grad()
			global_step += 1

	logs = {}

	results = evaluate(args, model, task, tokenizer, "Current Task")
	for key, value in results.items():
		eval_key = "eval_{}".format(key)
		logs[eval_key] = value

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

	save_model(args, task_num, model)

	#Evaluating on all tasks - both forward and backward transfer

	for i in range(len(all_tasks)):
		#Previous tasks
		if (i < task_num):
			model.load_state_dict(torch.load(os.path.join(args.output_dir, "task_paramters_" + str(i) + ".pt")), strict=False)
			evaluate(args, model, all_tasks[i], tokenizer, "Previous Task (Continual)")

	tb_writer.close()

	return global_step, tr_loss / global_step


def evaluate(args, model, task, tokenizer, prefix=""):
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
		eval_sampler = SequentialSampler(eval_dataset)
		eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

		# Eval!
		logger.info("***** Running evaluation:: Task : {}, Prefix : {} *****".format(task, prefix))
		logger.info("  Num examples = %d", len(eval_dataset))
		logger.info("  Batch size = %d", args.eval_batch_size)
		eval_loss = 0.0
		nb_eval_steps = 0
		preds = None
		out_label_ids = None
		for batch in tqdm(eval_dataloader, desc="Evaluating"):
			model.eval()
			batch = tuple(t.to(args.device) for t in batch)

			with torch.no_grad():
				inputs = {"input_ids": batch[0], "attention_mask": batch[1],"token_type_ids":batch[2] , "labels": batch[3]}
				'''if args.model_type != "distilbert":
					inputs["token_type_ids"] = (
						batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
					)  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids '''
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
			for key in sorted(result.keys()):
				logger.info("  %s = %s", key, str(result[key]))
				writer.write("%s = %s\n" % (key, str(result[key])))

	return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
	processor = glue_processors[task]()
	output_mode = glue_output_modes[task]
	
	logger.info("Creating features from dataset file at %s", args.data_dir)
	label_list = processor.get_labels()
	if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
		# HACK(label indices are swapped in RoBERTa pretrained model)
		label_list[1], label_list[2] = label_list[2], label_list[1]
	examples = (
		processor.get_dev_examples(os.path.join(args.data_dir, task)) if evaluate else processor.get_train_examples(os.path.join(args.data_dir, task))
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

	# Setup logging
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO,
	)
	
	# multi-gpu training (should be after apex fp16 initialization)
	if args.n_gpu > 1:
		model = torch.nn.DataParallel(model)



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
	accuracy_matrix = np.zeros((n,n))

	tasks = list(args.task_params.keys())
	models = []
	# Model
	for key in args.task_params:
		models.append((key, AutoModelForSequenceClassification.from_pretrained(args.model_type)))
	for i in range(n):
		models[i][1].to(args.device)
		save_model(args, i, models[i][1])


	for i in range(len(configs)):
		if (i>0):
			#Always load the BERT parameters of previous model
			models[i][1].load_state_dict(torch.load(os.path.join(args.output_dir, "bert_paramters_" + str(i-1) + ".pt")), strict=False)
		new_args = convert_dict(args.task_params[tasks[i]], args)
		train_dataset = load_and_cache_examples(args, tasks[i], tokenizer, evaluate=False)
		global_step, tr_loss = train(new_args, train_dataset, tasks[i], tasks, models[i][1], i, tokenizer)
		logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)



if __name__ == '__main__':
	main()