import torch
from torch import Tensor
from torch.optim import Optimizer
from typing import List, Optional
import collections
import math

from tensorboardX import SummaryWriter


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()

def _use_grad_for_differentiable(func):
    def _use_grad(self, *args, **kwargs):
        prev_grad = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(self.defaults['differentiable'])
            ret = func(self, *args, **kwargs)
        finally:
            torch.set_grad_enabled(prev_grad)
        return ret
    return _use_grad

class SGDcustom(Optimizer):
    def __init__(self, params, lr=required, momentum=0,
                 weight_decay=0, *,
                 differentiable=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, 
                        weight_decay=weight_decay, differentiable=differentiable)

        super(SGDcustom, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('differentiable', False)

    @_use_grad_for_differentiable
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            self._update_params(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'])

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

    def _update_params(self, params: List[Tensor],
            d_p_list: List[Tensor],
            momentum_buffer_list: List[Optional[Tensor]],
            *,
            weight_decay: float,
            momentum: float,
            lr: float):

        for i, param in enumerate(params):
            d_p = d_p_list[i]

            if weight_decay != 0:
                d_p = d_p.add(param, alpha=weight_decay)

            if momentum != 0:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = torch.clone(d_p).detach()
                    momentum_buffer_list[i] = buf
                else:
                    buf.mul_(momentum).add_(d_p, alpha=1)


                d_p = buf

            param.add_(d_p, alpha=-lr)


class SignSGD(Optimizer):
    def __init__(self, params, lr=required, betas=(0.9, 0.99),
                 weight_decay=0, *,
                 differentiable=False):
        
        momentum = betas[0]
        momentum_interp = betas[1]

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if momentum_interp < 0.0:
            raise ValueError("Invalid momentum_interp value: {}".format(momentum_interp))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, momentum_interp=momentum_interp,
                        weight_decay=weight_decay, differentiable=differentiable)

        super(SignSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('differentiable', False)

    @_use_grad_for_differentiable
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            error_residuals_list = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            self._update_params(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'],
                momentum_interp=group['momentum_interp'])

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss

    def _update_params(self, params: List[Tensor],
            d_p_list: List[Tensor],
            momentum_buffer_list: List[Optional[Tensor]],
            *,
            weight_decay: float,
            momentum: float,
            momentum_interp: float,
            lr: float,
            ):

        for i, param in enumerate(params):
            d_p = d_p_list[i]

            if momentum > 1e-8:
                buf = momentum_buffer_list[i]

                if buf is None:
                    buf = torch.clone(d_p).detach()
                    momentum_buffer_list[i] = buf
                else:
                    grad = buf.mul(momentum).add(d_p, alpha=1-momentum)
                    buf.mul_(momentum_interp).add_(d_p, alpha=1-momentum_interp)
                    d_p = grad
            
            # decouple momentum and weight decay
            if weight_decay != 0:
                d_p.add_(param, alpha=weight_decay)

            # if noise_scale > 1e-8:
            #     d_p.add_(torch.randn_like(d_p), alpha = noise_scale)

            d_p.sign_()
            param.add_(d_p, alpha=-lr)
        #     elif sign_type == "EF":
        #         error_residuals = error_residuals_list[i]
        #         if error_residuals is not None:
        #             d_p.add_(error_residuals,alpha=1)

        #         total1norm += d_p.norm(p=1)
        #         totaldim += d_p.numel()
        #         d_p_list[i] = d_p

        #     elif sign_type == "layerEF":
        #         error_residuals = error_residuals_list[i]
        #         if error_residuals is not None:
        #             d_p.add_(error_residuals,alpha=1)
                    
        #         layer1norm = d_p.norm(p=1) / d_p.numel()
        #         d_p_sign = d_p.sign()
        #         d_p_sign.mul_(layer1norm)
        #         error_residuals_list[i] = d_p - d_p_sign
        #         d_p = d_p_sign

        #         if momentum > 1e-8:
        #             buf = momentum_buffer_list[i]

        #             if buf is None:
        #                 buf = torch.clone(d_p).detach()
        #                 momentum_buffer_list[i] = buf
        #             else:
        #                 buf.mul_(momentum).add_(d_p, alpha=1)

        #             d_p = buf

        #     elif sign_type == "1norm":
        #         total1norm += d_p.norm(p=1)
        #         totaldim += d_p.numel()
        #         d_p_list[i] = d_p

        #     elif sign_type == "layer1norm":
        #         layer1norm = d_p.norm(p=1) / d_p.numel()
        #         d_p.sign_().mul_(layer1norm)


        #     if (sign_type != "EF") and (sign_type != "1norm"):
        #         param.add_(d_p, alpha=-lr)

        # if (sign_type == "EF") or (sign_type == "1norm"):
        #     for i, param in enumerate(params):
        #         d_p = d_p_list[i]
        #         if sign_type == "EF":
        #             d_p_sign = d_p.sign()
        #             d_p_sign.mul_(total1norm / totaldim)
        #             error_residuals_list[i] = d_p - d_p_sign

        #             if momentum > 1e-8:
        #                 buf = momentum_buffer_list[i]

        #                 if buf is None:
        #                     buf = torch.clone(d_p).detach()
        #                     momentum_buffer_list[i] = buf
        #                 else:
        #                     buf.mul_(momentum).add_(d_p_sign, alpha=1)

        #                 d_p_sign = buf

        #             param.add_(d_p_sign, alpha=-lr)                
        #         else:
        #             d_p.sign_().mul_(total1norm/totaldim)
        #             param.add_(d_p, alpha=-lr)


"""Lamb optimizer."""




def log_lamb_rs(optimizer: Optimizer, event_writer: SummaryWriter, token_count: int):
    """Log a histogram of trust ratio scalars in across layers."""
    results = collections.defaultdict(list)
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            for i in ('weight_norm', 'adam_norm', 'trust_ratio'):
                if i in state:
                    results[i].append(state[i])

    for k, v in results.items():
        event_writer.add_histogram(f'lamb/{k}', torch.tensor(v), token_count)

class Lamb(Optimizer):
    r"""Implements Lamb algorithm.
    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.
    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0, adam=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        self.adam = adam
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Paper v3 does not use debiasing.
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                # Apply bias to lr to avoid broadcast.
                step_size = group['lr'] # * math.sqrt(bias_correction2) / bias_correction1

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])

                adam_norm = adam_step.pow(2).sum().sqrt()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(adam_step, alpha=-step_size * trust_ratio)

        return loss