#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import math
import torch
import torch.optim as optim


class local_AdamW(optim.Optimizer):
    """ Implements Adam algorithm with weight decay fix.
    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
    """

    def __init__(self, params, reg_lambda, freeze_layers=['classifier.weight', 'classifier.bias'],
                 lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, correct_bias=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)
        self.reg_lambda = reg_lambda
        self.freeze_layers = freeze_layers

    def step(self, reg_params, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for index, p in enumerate(group["params"]):
                name = group["names"][index]
                if p.grad is None:
                    continue
                grad = p.grad.data
                zero = torch.FloatTensor(p.data.size()).zero_()
                if grad.equal(zero.cuda()):
                    print('omega after zero')
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                ###### BEGIN MAS CODE########
                # if p in reg_params:
                if name not in self.freeze_layers:
                    param_dict = reg_params[name]

                    omega = param_dict['omega'].cuda()
                    init_val = param_dict['init_val'].cuda()
                    curr_param_value = p.data.cuda()

                    # get the difference
                    param_diff = curr_param_value - init_val

                    # get the gradient for the penalty term for change in the weights of the parameters
                    importance_grad = torch.mul(param_diff, 2 * self.reg_lambda * omega)
                    del param_diff, omega, init_val, curr_param_value
                    grad += importance_grad
                    del importance_grad
                ###### END MAS CODE#######

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)


                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(-group["lr"] * group["weight_decay"], p.data)

        return loss


class omega_update_Adam(optim.Optimizer):
    """ Implements Adam algorithm with weight decay fix.
    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
    """

    def __init__(self, params, freeze_layers=['classifier.weight', 'classifier.bias'],
                 lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, correct_bias=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta para"
                             "meter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)
        self.freeze_layers = freeze_layers

    def step(self, reg_params, batch_index, batch_size, device):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        i = 0
        total = 0
        for group in self.param_groups:
            for index, p in enumerate(group["params"]):
                name = group["names"][index]
                if p.grad is None:
                    continue

                grad = p.grad.data
                # grad = torch.tensor(p.grad.data, dtype=torch.float64).to(device)
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                if p.grad is None:
                    continue
                zero = torch.FloatTensor(p.data.size()).zero_()
                ###### BEGIN MAS CODE ######
                # if p in reg_params:
                if name not in self.freeze_layers:

                    grad_copy = grad.clone().abs()
                    if grad_copy.equal(zero.cuda()):
                        # print('grad has become zero')
                        i += 1
                    param_dict = reg_params[name]
                    omega = param_dict['omega'].to(device)
                    current_size = (batch_index + 1) * batch_size
                    omega += (grad_copy - batch_size*batch_index*omega)/float(current_size)
                    param_dict['omega'] = omega
                    reg_params[name] = param_dict
                    total += 1
                ##### END MAS CODE ####

        # we don't want to update anything and using this step to only access the gradients, so loss is not calculated.

        return None
