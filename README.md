## Baseline

python main.py --data_dir "data" --task_params "configs/example_task_1.json" --cuda --tokenizer_name "bert" --do_lower_case --model_type "bert-base-uncased" --output_dir "out" --num_train_epochs 2 --seed 42

## EWC

python main.py --data_dir "data" --task_params configs/example_task_1.json --tokenizer_name "bert" --do_lower_case --model_type "bert-base-uncased" --output_dir "out" --num_train_epochs 2 --seed 42 --ewc --lamda 1000000
