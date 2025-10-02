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