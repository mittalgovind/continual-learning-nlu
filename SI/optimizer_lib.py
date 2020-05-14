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

    def step(self, reg_params, batch_index, batch_size, closure=None):
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
                lr = group["lr"]
                state["step"] += 1

                ###### BEGIN SI CODE########
                if name not in self.freeze_layers:
                    # update gradient with SI regularization
                    param_dict = reg_params[name]

                    small_omega = param_dict['small_omega'].cuda()
                    big_omega = param_dict['big_omega'].cuda()
                    init_val = param_dict['init_val'].cuda()
                    curr_param_value = p.data.cuda()

                    # get the difference
                    param_diff = torch.sub(curr_param_value, init_val)

                    # get the gradient for the penalty term for change in the weights of the parameters
                    importance_grad = torch.mul(param_diff, 2 * self.reg_lambda * big_omega)

                    # update small omega
                    current_size = (batch_index + 1) * batch_size
                    small_omega_update = torch.mul(grad, param_diff)
                    small_omega_update = torch.sub(small_omega_update, batch_size * batch_index * small_omega)
                    small_omega_update = torch.div(small_omega_update, float(current_size))
                    param_dict['small_omega'] = torch.sub(small_omega, small_omega_update)
                    reg_params[name] = param_dict
                    # add the surrogate loss
                    grad = torch.add(grad, importance_grad)
                    del param_diff, big_omega, init_val, curr_param_value, small_omega_update
                    del importance_grad, param_dict
                ###### END SI CODE#######

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                grad = grad.float()
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    lr = lr * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-lr, exp_avg, denom)

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
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)
        self.freeze_layers = freeze_layers
        self.xi = 0.1

    def step(self, reg_params, batch_index, batch_size, device, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
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
                ###### BEGIN SI CODE ######
                if name not in self.freeze_layers:
                    param_dict = reg_params[name]

                    # UPDATE BIG OMEGA
                    big_omega = param_dict['big_omega'].to(device)
                    small_omega = param_dict['small_omega'].to(device)
                    init_val = param_dict['init_val']
                    curr_wt = p.data.cuda()

                    delta = torch.sub(curr_wt, init_val)
                    big_omega_update = small_omega / (delta ** 2 + self.xi)
                    current_size = (batch_index + 1) * batch_size
                    big_omega_update = torch.sub(big_omega_update, batch_size * batch_index * big_omega)
                    big_omega_update = torch.div(big_omega_update, float(current_size))
                    big_omega = torch.add(big_omega, big_omega_update)
                    big_omega += (big_omega_update - batch_size * batch_index * big_omega) / float(current_size)
                    param_dict['big_omega'] = big_omega
                    reg_params[p] = param_dict
                    del delta, big_omega, big_omega_update, current_size, param_dict

        return None
