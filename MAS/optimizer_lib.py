#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import copy
import os
import shutil


class local_AdamW(optim.Optimizer):
    """ Implements Adam algorithm with weight decay fix.
    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
    """

    def __init__(self, params, reg_lambda, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, correct_bias=True):
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
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
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
                if p in reg_params:
                    param_dict = reg_params[p]

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


class local_sgd(optim.SGD):
    def __init__(self, params, reg_lambda, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(local_sgd, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        self.reg_lambda = reg_lambda

    def __setstate__(self, state):
        super(local_sgd, self).__setstate__(state)

    def step(self, reg_params, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if p in reg_params:
                    param_dict = reg_params[p]

                    omega = param_dict['omega']
                    init_val = param_dict['init_val']

                    curr_param_value = p.data
                    curr_param_value = curr_param_value.cuda()

                    init_val = init_val.cuda()
                    omega = omega.cuda()

                    # get the difference
                    param_diff = curr_param_value - init_val

                    # get the gradient for the penalty term for change in the weights of the parameters
                    local_grad = torch.mul(param_diff, 2 * self.reg_lambda * omega)

                    del param_diff
                    del omega
                    del init_val
                    del curr_param_value

                    d_p = d_p + local_grad

                    del local_grad

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

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

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, correct_bias=True):
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

    def step(self, reg_params, batch_index, batch_size, device, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
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
                if p.grad is None:
                    continue

                ###### BEGIN MAS CODE########
                if p in reg_params:
                    grad_copy = grad.clone().abs()
                    # grad_copy = grad.clone()
                    param_dict = reg_params[p]
                    omega = param_dict['omega'].to(device)
                    current_size = (batch_index + 1) * batch_size
                    omega += (grad_copy - batch_size*batch_index*omega)/float(current_size)
                    zero = torch.FloatTensor(p.data.size()).zero_()
                    if omega.equal(zero.cuda()):
                        print('omega after zero')

                    param_dict['omega'] = omega
                    reg_params[p] = param_dict
                ###### END MAS CODE#######

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                # exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                # exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                # denom = exp_avg_sq.sqrt().add_(group["eps"])
                #
                # step_size = group["lr"]
                # if group["correct_bias"]:  # No bias correction for Bert
                #     bias_correction1 = 1.0 - beta1 ** state["step"]
                #     bias_correction2 = 1.0 - beta2 ** state["step"]
                #     step_size = step_size * math.sqrt(bias_correction2) / bias_correction1
                #
                # p.data.addcdiv_(-step_size, exp_avg, denom)
                #
                #
                # # Just adding the square of the weights to the loss function is *not*
                # # the correct way of using L2 regularization/weight decay with Adam,
                # # since that will interact with the m and v parameters in strange ways.
                # #
                # # Instead we want to decay the weights in a manner that doesn't interact
                # # with the m/v parameters. This is equivalent to adding the square
                # # of the weights to the loss with plain (non-momentum) SGD.
                # # Add weight decay at the end (fixed version)
                # if group["weight_decay"] > 0.0:
                #     p.data.add_(-group["lr"] * group["weight_decay"], p.data)

        return loss


class omega_update(optim.SGD):

    def __init__(self, params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(omega_update, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)

    def __setstate__(self, state):
        super(omega_update, self).__setstate__(state)

    def step(self, reg_params, batch_index, batch_size, device, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                if p in reg_params:
                    grad_data = p.grad.data

                    # The absolute value of the grad_data that is to be added to omega
                    grad_data_copy = p.grad.data.clone()
                    grad_data_copy = grad_data_copy.abs()

                    param_dict = reg_params[p]

                    omega = param_dict['omega']
                    omega = omega.to(device)

                    current_size = (batch_index + 1) * batch_size
                    step_size = 1 / float(current_size)

                    # Incremental update for the omega
                    omega = omega + step_size * (grad_data_copy - batch_size * omega)

                    param_dict['omega'] = omega

                    reg_params[p] = param_dict

        return loss


# class omega_vector_update(optim.SGD):
#
#     def __init__(self, params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False):
#         super(omega_vector_update, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)
#
#     def __setstate__(self, state):
#         super(omega_vector_update, self).__setstate__(state)
#
#     def step(self, reg_params, finality, batch_index, batch_size, device, closure=None):
#         loss = None
#
#         if closure is not None:
#             loss = closure()
#
#         for group in self.param_groups:
#             weight_decay = group['weight_decay']
#             momentum = group['momentum']
#             dampening = group['dampening']
#             nesterov = group['nesterov']
#
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
#
#                 if p in reg_params:
#
#                     grad_data = p.grad.data
#
#                     # The absolute value of the grad_data that is to be added to omega
#                     grad_data_copy = p.grad.data.clone()
#                     grad_data_copy = grad_data_copy.abs()
#
#                     param_dict = reg_params[p]
#
#                     if not finality:
#
#                         if 'temp_grad' in reg_params.keys():
#                             temp_grad = param_dict['temp_grad']
#
#                         else:
#                             temp_grad = torch.FloatTensor(p.data.size()).zero_()
#                             temp_grad = temp_grad.to(device)
#
#                         temp_grad = temp_grad + grad_data_copy
#                         param_dict['temp_grad'] = temp_grad
#
#                         del temp_data
#
#
#                     else:
#
#                         # temp_grad variable
#                         temp_grad = param_dict['temp_grad']
#                         temp_grad = temp_grad + grad_data_copy
#
#                         # omega variable
#                         omega = param_dict['omega']
#                         omega.to(device)
#
#                         current_size = (batch_index + 1) * batch_size
#                         step_size = 1 / float(current_size)
#
#                         # Incremental update for the omega
#                         omega = omega + step_size * (temp_grad - batch_size * (omega))
#
#                         param_dict['omega'] = omega
#
#                         reg_params[p] = param_dict
#
#                         del omega
#                         del param_dict
#
#                     del grad_data
#                     del grad_data_copy
#
#         return loss
