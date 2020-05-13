from subprocess import Popen, PIPE
import os
from hyperopt import hp, Trials, rand, fmin, partial
from main import loader


def train_network(lamda):
    print('regularization lambda = {}'.format(lamda))
    # args = ['python', 'main.py', '--data_dir', '../data', '--task_params', 'single_task.json', '--cuda',
    #         '--tokenizer_name', 'bert',
    #         '--do_lower_case', '--model_type', 'bert-base-uncased', '--output_dir', 'out', '--num_train_epochs', '1',
    #         '--seed',
    #         '42', '--init_lr', '0.00005', '--reg_lambda', str(lamda)]
    acc_mat, t_mat = loader(lamda)
    print(acc_mat, t_mat)

    return t_mat[1][0]
    # output[-1]


space = (hp.uniformint('l', 50, 500))
best = fmin(train_network, space, algo=rand.suggest, max_evals=50)
print(best)
# print(lambda_space())
