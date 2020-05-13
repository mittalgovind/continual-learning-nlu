import os

init = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=70GB
#SBATCH --output=slurm_{file_name}.out
#SBATCH --error=slurm_{file_name}.err
#SBATCH --qos=batch
#SBATCH --time=20:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cpk290@nyu.edu
#SBATCH --constraint=gpu_12gb
#SBATCH --job-name={job_name}

cd /misc/vlgscratch4/BrunaGroup/chandu
source cv_tutor/env/bin/activate

cd continual-learning-nlu
"""

CONFIGS = ["example_task_1.json", "example_task_2.json", "example_task_3.json",
"example_task_4.json", "example_task_5.json", "example_task_6.json", "example_task_7.json"]

CONFIG_DIR = "configs"
DATA_DIR = "data"
TASK_PARAMS = ""
OUTPUT_DIR = "out_2"
NUM_EPOCHS = 3
SEEDS = [42,43,44,45,46,47,48,49,50]
EWC_LAMBDAS = [1000000, 1500000, 500000, 2000000]

python_command = "python main.py --data_dir {} --task_params {} --tokenizer_name bert --do_lower_case --model_type bert-base-uncased --output_dir {} --num_train_epochs {} --seed {} --ewc --ewc_lamda {} --cuda"


for i,config in enumerate(CONFIGS):
	for ewc_lambda in EWC_LAMBDAS:
		f = open("sbatch_ewc_{}_lambda_{}".format(config, ewc_lambda), "w")
		f.write(init.format(file_name="ewc_{}_lambda_{}".format(config, ewc_lambda),
							job_name="{}.json".format(str(i+1))))
		for seed in SEEDS:
			new_command = python_command.format(DATA_DIR,
												os.path.join(CONFIG_DIR ,config),
												OUTPUT_DIR,
												str(NUM_EPOCHS),
												str(seed),
												ewc_lambda
												)
			f.write(new_command+"\n")
		f.close()