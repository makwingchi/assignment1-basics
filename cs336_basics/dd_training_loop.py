import random

import numpy as np
import torch


def load_data(dataset, batch_size, context_length, device):
    """
    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.
    """
    dataset_size = dataset.shape[0]
    indices = random.sample(range(dataset_size-context_length), batch_size)
    
    train, valid = [], []
    for idx in indices:
        curr_train = torch.tensor(dataset[idx:idx+context_length], device=device)
        curr_valid = torch.tensor(dataset[idx+1:idx+1+context_length], device=device)

        train.append(curr_train)
        valid.append(curr_valid)

    return torch.stack(train, dim=0), torch.stack(valid, dim=0)


def save_checkpoint(model, optimizer, iteration, out):
    model_params = model.state_dict()
    optim_params = optimizer.state_dict()

    to_save = {
        "model_params": model_params,
        "optim_params": optim_params,
        "iteration": iteration
    }

    torch.save(to_save, out)


def load_checkpoint(src, model, optimizer):
    to_load = torch.load(src)
    model.load_state_dict(to_load["model_params"])
    optimizer.load_state_dict(to_load["optim_params"])

    return to_load["iteration"]