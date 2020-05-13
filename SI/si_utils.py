import torch


class shared_model(torch.nn.Module):
    def __init__(self, model):
        super(shared_model, self).__init__()
        self.tmodel = model
        self.reg_params = {}

    def forward(self, x):
        return self.tmodel(x)


def init_reg_params(model, device, freeze_layers=['classifier.weight', 'classifier.bias']):
    """
    Input:
    1) model: A reference to the model that is being trained
    2) device: which device to use (GPU or not)
    3) freeze_layers: A list containing the layers for which omega is not calculated. Useful in the
        case of computational limitations where computing the importance parameters for the entire model
        is not feasible

    Output:
    1) model: A dictionary containing importance weights (omega), init_val (keep a reference
    to the initial values of the parameters) for all trainable parameters is calculated and the updated
    model with these reg_params is returned.


    Function: Initializes the reg_params for a model for the initial task (task = 1)

    """
    reg_params = dict()
    for name, param in model.tmodel.named_parameters():
        if not name in freeze_layers:
            # print("Initializing omega values for layer", name)
            init_val = param.data.clone()
            param_dict = dict()

            # for first task, omega is initialized to zero
            param_dict['small_omega'] = torch.zeros(param.size(), dtype=torch.float64)
            param_dict['big_omega'] = torch.zeros(param.size(), dtype=torch.float64)
            param_dict['init_val'] = init_val

            # the key for this dictionary is the name of the layer
            reg_params[name] = param_dict

    model.reg_params = reg_params
    return model


def init_reg_params_across_tasks(model, device, freeze_layers=['classifier.weight', 'classifier.bias']):
    """
    Input:
    1) model: A reference to the model that is being trained
    2) device: GPU
    3) freeze_layers: A list containing the layers for which omega is not calculated. Useful in the
        case of computational limitations where computing the importance parameters for the entire model
        is not feasible

    Output:
    1) model: A dictionary containing importance weights (omega), init_val (keep a reference
    to the initial values of the parameters) for all trainable parameters is calculated and the updated
    model with these reg_params is returned.


    Function: Initializes the reg_params for a model for other tasks in the sequence (task != 1)
    """

    # Get the reg_params for the model
    reg_params = model.reg_params
    for name, param in model.tmodel.named_parameters():
        if not name in freeze_layers:
            param_dict = reg_params[name]
            print("Initializing the omega values for layer for the new task", name)

            param_dict['small_omega'] = torch.zeros(param.data.size(), dtype=torch.float64).cuda()
            # Store the previous values of omega
            param_dict['prev_big_omega'] = param_dict['big_omega']
            # Initialize a new omega
            param_dict['big_omega'] = torch.zeros(param.data.size(), dtype=torch.float64).cuda()

            # store the initial values of the parameters
            init_val = param.data.clone()
            init_val = init_val.to(device)
            param_dict['init_val'] = init_val

            # the key for this dictionary is the name of the layer
            reg_params[name] = param_dict
    return model


def exp_lr_scheduler(optimizer, epoch, init_lr=0.0008, lr_decay_epoch=20):
    """
    Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.
    """
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))
    print('lr is ' + str(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


# sanity check for the model to check if the omega values are getting updated
def sanity_model(model, freeze_layers=['classifier.weight', 'classifier.bias']):
    for name, param in model.tmodel.named_parameters():
        if name not in freeze_layers:
            print(name)
            param_dict = model.reg_params[name]
            print("Mean value of omega is", param_dict['big_omega'].mean())
            print("Max omega is", param_dict['big_omega'].max())
            print("Min omega is", param_dict['big_omega'].min())


def consolidate_reg_params(model):
    """
    Input:
    1) model: A reference to the model that is being trained

    Output:
    1) reg_params: A dictionary containing importance weights (omega), init_val (keep a reference
    to the initial values of the parameters) for all trainable parameters


    Function: This function updates the value (adds the value) of omega across the tasks that the model is
    exposed to

    """
    # Get the reg_params for the model
    reg_params = model.reg_params

    for name, param in model.tmodel.named_parameters():
        if param in reg_params:
            param_dict = reg_params[param]
            print("Consolidating the omega values for layer", name)

            # Store the previous values of omega
            prev_big_omega = param_dict['prev_big_omega']
            new_big_omega = param_dict['big_omega']

            new_big_omega = torch.add(prev_big_omega, new_big_omega)
            del param_dict['prev_big_omega']

            param_dict['big_omega'] = new_big_omega

            # the key for this dictionary is the name of the layer
            reg_params[param] = param_dict

    model.reg_params = reg_params

    return model


def compute_omega_grads_norm(model, dataloader, optimizer, device):
    """
    Inputs:
    1) model: A reference to the model for which omega is to be calculated
    2) dataloader: A dataloader to feed the data to the model
    3) optimizer: An instance of the "omega_update" class
    4) use_gpu: Flag is set to True if the model is to be trained on the GPU

    Outputs:
    1) model: An updated reference to the model is returned

    Function: Global version for computing the l2 norm of the function (neural network's) outputs. In
    addition to this, the function also accumulates the values of omega across the items of a task

    """
    model.tmodel.eval()

    index = 0
    for data in dataloader:
        # get the inputs and labels
        batch = tuple(t.to(device) for t in data)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3]}
        # Zero the parameter gradients

        optimizer.zero_grad()

        outputs = model.tmodel(**inputs)
        # get the function outputs

        del inputs

        # compute the squared l2 norm of the function outputs
        l2_norm = torch.norm(outputs[1], 2, dim=1)
        del outputs

        squared_l2_norm = l2_norm ** 2
        del l2_norm

        sum_norm = torch.sum(squared_l2_norm)
        del squared_l2_norm

        # compute gradients for these parameters
        sum_norm.backward()

        # optimizer.step computes the omega values for the new batches of data
        optimizer.step(model.reg_params, index, len(batch[0]), device)
        # optimizer.step(model.reg_params)
        del batch
        index = index + 1

    return model
