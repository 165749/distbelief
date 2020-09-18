import torch
import numpy as np
from distbelief.utils.tracer import tracer


def ravel_model_params(model, grads=False):
    """
    Squash model parameters or gradients into a single tensor.
    """
    with tracer.start_active_span('ravel_model_params'):
        if grads:
            buffer = torch.zeros(sum([para.grad.numel() for para in model.parameters()]))
            current_index = 0
            for parameter in model.parameters():
                numel = parameter.grad.numel()
                # Update data (but not grad) in model.parameter()
                buffer[current_index:current_index + numel] = parameter.grad.view(-1)
                current_index += numel
            return buffer
        else:
            buffer = torch.zeros(sum([para.data.numel() for para in model.parameters()]))
            current_index = 0
            for parameter in model.parameters():
                numel = parameter.data.numel()
                # Update data (but not grad) in model.parameter()
                buffer[current_index:current_index + numel] = parameter.data.view(-1)
                current_index += numel
            return buffer


def unravel_model_params(model, parameter_update):
    """
    Assigns parameter_update params to model.parameters.
    This is done by iterating through model.parameters() and assigning the relevant params in parameter_update.
    NOTE: this function manipulates model.parameters.
    """
    with tracer.start_active_span('unravel_model_params'):
        current_index = 0 # keep track of where to read from parameter_update
        for parameter in model.parameters():
            numel = parameter.data.numel()
            size = parameter.data.size()
            # Update data (but not grad) in model.parameter()
            parameter.data.copy_(parameter_update[current_index:current_index+numel].view(size))
            current_index += numel
