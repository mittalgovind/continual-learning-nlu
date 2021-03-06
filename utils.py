class AttrDict(dict):
	def __init__(self, *args, **kwargs):
		super(AttrDict, self).__init__(*args, **kwargs)
		self.__dict__ = self

def convert_dict(config, args):
	out = AttrDict()
	out.update(config)
	out.update({
				"num_train_epochs": args.num_train_epochs,
				"seed": args.seed,
				"n_gpu": args.n_gpu,
				"per_gpu_batch_size": args.per_gpu_batch_size,
				"warmup_steps": 0,
				"model_name_or_path": "",
				"device": args.device,
				"model_type": args.model_type,
				"output_dir": args.output_dir,
				"data_dir": args.data_dir,
				"max_seq_length": args.max_seq_length,
				"ewc": args.ewc,
				"ewc_lamda": args.ewc_lamda,
				"mas": args.mas,
				"mas_lambda": args.mas_lambda,
				"mas_init_lr": args.mas_init_lr,
				"task_params": args.task_params
				})
	return out