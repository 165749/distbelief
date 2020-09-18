import logging
import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer, required
from distbelief.utils.serialization import ravel_model_params, unravel_model_params
from distbelief.utils.messaging import send_message
from distbelief.utils.tracer import tracer

_LOGGER = logging.getLogger(__name__)


class DownpourSGD(Optimizer):
    """DownpourSGD"""

    def __init__(self, params, lr=required, model=required):
        """__init__

        :param params:
        :param lr:
        :param model:
        """
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr,)
        self.gradients_shape = ravel_model_params(model).numel()
        self.gradients_buffer = torch.zeros(self.gradients_shape)
        self.model = model

        # Send a empty gradients to fetch the initial model from the server
        send_message(self.gradients_buffer)
        dist.recv(tensor=self.gradients_buffer)
        unravel_model_params(self.model, self.gradients_buffer)

        super(DownpourSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Collect gradients from self.model
        gradients = ravel_model_params(self.model, grads=True)

        with tracer.start_active_span('multiply -lr'):
            # Multiple gradients by learning rate
            gradients = gradients * (-self.param_groups[0]['lr'])
        # Send gradients to the server
        send_message(gradients)

        # Will pull parameters from the server, so no need to update internal parameters

        # Wait the server to send back the updated model
        dist.recv(tensor=self.gradients_buffer)
        with tracer.start_active_span('local update') as scope:
            scope.span.set_tag('size', self.gradients_buffer.element_size() * self.gradients_buffer.nelement())
            scope.span.set_tag('worker', dist.get_rank())
            # Update parameters in self.model
            unravel_model_params(self.model, self.gradients_buffer)

        return loss

    def stop(self):
        # Inform the server about completion (by setting tensor[0] to inf)
        tensor = torch.zeros(self.gradients_shape)
        tensor[0] = float('inf')
        send_message(tensor)
