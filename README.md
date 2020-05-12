## Baseline

python main.py --data_dir "data" --task_params "configs/example_task_1.json" --cuda --tokenizer_name "bert" --do_lower_case --model_type "bert-base-uncased" --output_dir "out" --num_train_epochs 2 --seed 42

## EWC

python main.py --data_dir "data" --task_params configs/example_task_1.json --tokenizer_name "bert" --do_lower_case --model_type "bert-base-uncased" --output_dir "out" --num_train_epochs 2 --seed 42 --ewc --ewc_lamda 1000000 --cuda

## MAS

Go into ./MAS and run

python main.py --data_dir "../data" --task_params "example_task.json" --cuda --tokenizer_name "bert" --do_lower_case --model_type "bert-base-uncased" --output_dir "out" --num_train_epochs 1 --seed 42 --init_lr=0.00005 --reg_lambda 50
