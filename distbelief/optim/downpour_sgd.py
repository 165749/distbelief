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
        self.model = model

        # Send a empty gradients to fetch the initial model from the server
        # Send gradients to the server layer by layer
        for para in self.model.parameters():
            with tracer.start_active_span('send') as scope:
                scope.span.set_tag('size', para.data.nelement() * para.data.element_size())
                scope.span.set_tag('worker', dist.get_rank())
                dist.send(torch.zeros(para.data.size()), 0)
        # Wait the server to send back the updated model
        for para in self.model.parameters():
            with tracer.start_active_span('recv') as scope:
                scope.span.set_tag('size', para.data.nelement() * para.data.element_size())
                scope.span.set_tag('worker', dist.get_rank())
                dist.recv(para.data)

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

        # Learning rate
        lr = -self.param_groups[0]['lr']
        # Send gradients to the server layer by layer
        with tracer.start_active_span('send'):
            for i, para in enumerate(self.model.parameters()):
                with tracer.start_active_span('layer {}'.format(i)) as scope:
                    scope.span.set_tag('size', para.grad.nelement() * para.grad.element_size())
                    scope.span.set_tag('worker', dist.get_rank())
                    dist.send(lr * para.grad, 0)

        # Will pull parameters from the server, so no need to update internal parameters

        # Wait the server to send back the updated model
        with tracer.start_active_span('recv'):
            for i, para in enumerate(self.model.parameters()):
                with tracer.start_active_span('layer {}'.format(i)) as scope:
                    scope.span.set_tag('size', para.data.nelement() * para.data.element_size())
                    scope.span.set_tag('worker', dist.get_rank())
                    dist.recv(para.data)

        return loss

    def stop(self):
        # Inform the server about completion (by setting tensor[0] to inf)
        tensor = torch.zeros(list(self.model.parameters())[0].data.size())
        # TODO (zhuojin): Fix hardcoded
        tensor[0][0][0][0] = float('inf')
        send_message(tensor)
