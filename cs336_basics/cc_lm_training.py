import math

import torch
import torch.nn as nn


def cross_entropy(inputs, targets):
    """
    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.
    """
    mx = torch.max(inputs, dim=-1, keepdim=True).values
    new_inputs = inputs - mx

    target_logit = torch.gather(new_inputs, new_inputs.dim()-1, targets.unsqueeze(-1))
    exp_sum = torch.log(torch.exp(new_inputs).sum(dim=-1, keepdim=True))

    return torch.mean(exp_sum - target_logit)


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": eps
        }

        super().__init__(params, defaults=defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
        
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                state = self.state[p]
                m = state.get("m", 0)
                v = state.get("v", 0)
                t = state.get("t", 1)
                grad = p.grad.data

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad**2)
                alpha_t = lr * ((1 - beta2**t)**0.5) / (1 - beta1**t)
                p.data -= alpha_t * m / (torch.sqrt(v)+eps)
                p.data = p.data - lr * weight_decay * p.data

                state["m"] = m
                state["v"] = v
                state["t"] = t+1


def cosine_anneal_scheduling(t, alpha_max, alpha_min, tw, tc):
    if t < tw:
        return t * alpha_max / tw
    
    if t > tc:
        return alpha_min

    return alpha_min + 0.5 * (1 + math.cos(math.pi * (t - tw) / (tc - tw))) * (alpha_max - alpha_min)


def gradient_clipping(parameters, max_l2_norm, eps=1e-6):
    param_with_grad = [p for p in parameters if p.grad is not None]

    if len(param_with_grad) == 0:
        return

    total_norm = 0
    for p in param_with_grad:
        total_norm += torch.sum(p.grad**2)
    
    total_norm = total_norm ** 0.5

    if total_norm >= max_l2_norm:
        for p in param_with_grad:
            p.grad.data = p.grad.data * max_l2_norm / (total_norm + eps)
