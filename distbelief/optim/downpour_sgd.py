import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim.optimizer import Optimizer, required
from distbelief.utils.serialization import ravel_model_params, unravel_model_params
from distbelief.utils.messaging import send_message
from distbelief.utils.tracer import tracer

_LOGGER = logging.getLogger(__name__)


def build_distributed_model(model, lr, cuda=False):
    class DistributedModel(model):
        def __init__(self, *args, **kwargs):
            super(DistributedModel, self).__init__(*args, **kwargs)
            self.parameters_generator = self.parameters()
            self.parameters_buffer = []  # For receivers collecting model parameters
            self.senders = []
            self.receivers = []
            self.current_receiver = 0
            self.span = None
            self.root_span = None
            self.hooks = []
            self.register_hooks()
            for para in self.parameters():
                self.parameters_buffer.append(torch.zeros(para.data.size()))
            for name, module in self.named_modules():
                module.name = name

            # Send a empty gradients to fetch the initial model from the server
            # Send gradients to the server layer by layer
            with tracer.start_active_span('init'):
                for name, para in reversed(list(self.named_parameters())):
                    with tracer.start_active_span('send') as scope:
                        scope.span.set_tag('size', para.data.nelement() * para.data.element_size())
                        name = name.rsplit('.', maxsplit=1)
                        scope.span.set_tag('layer', name[0])
                        scope.span.set_tag('type', name[1])
                        dist.send(torch.zeros(para.data.size()), dist.get_rank() + 1)
            self.reset_and_start_receivers()

        def register_hooks(self):
            print('Register hooks!')
            for layer in self.modules():
                hook = layer.register_forward_pre_hook(self.forward_pre_hook_fn)
                self.hooks.append(hook)
                hook = layer.register_backward_hook(self.backward_hook_fn)
                self.hooks.append(hook)

        def remove_hooks(self):
            print('Remove hooks!')
            for hook in self.hooks:
                hook.remove()

        def init_tracer_span(self, root_span):
            self.root_span = root_span
            self.span = tracer.start_span('compute', child_of=root_span)

        def finish_tracer_span(self):
            self.span.finish()

        def reset_senders(self):
            self.senders = []

        def send(self, tensor, layer, type):
            with tracer.start_active_span('send') as scope:
                scope.span.set_tag('size', tensor.nelement() * tensor.element_size())
                scope.span.set_tag('layer', layer)
                scope.span.set_tag('type', type)
                sender = dist.isend(tensor, dist.get_rank() + 1)
                self.senders.append(sender)

        def wait_all_senders(self):
            for i, sender in enumerate(self.senders):
                sender.wait()

        def reset_and_start_receivers(self):
            self.parameters_generator = self.parameters()  # Reset generator
            self.receivers = []
            self.current_receiver = 0
            for para in self.parameters_buffer:
                receiver = dist.irecv(para, dist.get_rank() + 1)
                self.receivers.append(receiver)

        def wait_receiver(self):
            with tracer.start_active_span('recv', child_of=self.span) as scope:
                size = self.parameters_buffer[self.current_receiver].nelement() * self.parameters_buffer[self.current_receiver].element_size()
                self.receivers[self.current_receiver].wait()
                scope.span.set_tag('size', self.parameters_buffer[self.current_receiver].nelement() *
                                   self.parameters_buffer[self.current_receiver].element_size())
                para = next(self.parameters_generator)
                if cuda:
                    para.data = self.parameters_buffer[self.current_receiver].cuda()
                else:
                    para.data = self.parameters_buffer[self.current_receiver]
                self.current_receiver += 1

        def forward_pre_hook_fn(self, module, input):
            self.span.finish()
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm2d):
                self.wait_receiver()
                if module.bias is not None:
                    self.wait_receiver()
            self.span = tracer.start_span('compute', child_of=self.root_span)
            self.span.set_tag('layer', module.name)

        def backward_hook_fn(self, module, input, output):
            self.span.set_tag('layer', module.name)
            self.span.finish()
            weight = None
            bias = None
            if isinstance(module, nn.Conv2d):
                weight = input[1]
                # Note: For GPU training, the argument input will only contain bias if bias=True for nn.Conv2d,
                # which is caused by a well-known issue of backward hooks in PyTorch. To bypass the issue, needs
                # to disable all the bias of nn.Conv2d in the model.
                if not cuda:
                    bias = input[2]
            elif isinstance(module, nn.BatchNorm2d):
                weight = input[1]
                bias = input[2]
            elif isinstance(module, nn.Linear):
                weight = input[2].t()
                bias = input[0]
            # Reverse order in the backward
            if bias is not None:
                with tracer.start_active_span('lr', child_of=self.root_span) as scope:
                    scope.span.set_tag('layer', module.name)
                    scope.span.set_tag('type', 'bias')
                    grad = (-lr) * bias
                with tracer.start_active_span('copy', child_of=self.root_span) as scope:
                    scope.span.set_tag('layer', module.name)
                    scope.span.set_tag('type', 'bias')
                    grad = grad.cpu()
                    self.send(grad, module.name, 'bias')
            if weight is not None:
                with tracer.start_active_span('lr', child_of=self.root_span) as scope:
                    scope.span.set_tag('layer', module.name)
                    scope.span.set_tag('type', 'weight')
                    grad = (-lr) * weight
                with tracer.start_active_span('copy', child_of=self.root_span) as scope:
                    scope.span.set_tag('layer', module.name)
                    scope.span.set_tag('type', 'weight')
                    grad = grad.cpu()
                    self.send(grad, module.name, 'weight')
            self.span = tracer.start_span('compute', child_of=self.root_span)

        def step_begin(self):
            # Inform the server starting next step (by setting tensor[0] to 0)
            tensor = torch.zeros(1)
            dist.send(tensor, dist.get_rank() + 1)

    return DistributedModel


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

        ''' Synchronous send
        # Learning rate
        lr = -self.param_groups[0]['lr']
        # Send gradients to the server layer by layer
        with tracer.start_active_span('send'):
            for i, para in enumerate(self.model.parameters()):
                with tracer.start_active_span('layer {}'.format(i)) as scope:
                    scope.span.set_tag('size', para.grad.nelement() * para.grad.element_size())
                    scope.span.set_tag('worker', dist.get_rank())
                    dist.send(lr * para.grad, 0)
        '''
        self.model.wait_all_senders()

        # Will pull parameters from the server, so no need to update internal parameters

        '''Synchronous recv
        # Wait the server to send back the updated model
        with tracer.start_active_span('recv'):
            for i, para in enumerate(self.model.parameters()):
                with tracer.start_active_span('layer {}'.format(i)) as scope:
                    scope.span.set_tag('size', para.data.nelement() * para.data.element_size())
                    scope.span.set_tag('worker', dist.get_rank())
                    dist.recv(para.data)
        '''
        self.model.reset_and_start_receivers()

        return loss

    def stop(self):
        # Inform the server about completion (by setting tensor[0] to inf)
        tensor = torch.zeros(1)
        tensor[0] = float('inf')
        dist.send(tensor, dist.get_rank() + 1)
