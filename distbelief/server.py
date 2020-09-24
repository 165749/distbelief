# 
"""
Parameter server for distbelief
"""
import logging
import torch
import torch.optim
import torch.distributed as dist
from distbelief.utils.messaging import MessageListener, send_message
from distbelief.utils.serialization import ravel_model_params, unravel_model_params
from distbelief.utils.tracer import tracer

_LOGGER = logging.getLogger(__name__)


class ParameterServer(MessageListener):
    """ParameterServer"""
    def __init__(self, model, active_worker):
        _LOGGER.info("Creating ParameterServer")
        self.global_model = [para.data for para in model.parameters()]
        self.gradient_buffers = [torch.zeros(para.data.size()) for para in model.parameters()]
        self.active_worker = [i for i in range(1, active_worker+1)]
        # Init superclass
        super().__init__(model)

    def receive(self):
        for worker in self.active_worker.copy():
            for i, buffer in enumerate(self.gradient_buffers):
                with tracer.start_active_span('recv') as scope:
                    scope.span.set_tag('size', buffer.nelement() * buffer.element_size())
                    dist.recv(tensor=buffer, src=worker)
                    # TODO (zhuojin): Fix hardcoded
                    if i == 0 and buffer[0][0][0][0] == float('inf'):
                        self.active_worker.remove(worker)
                        break
                    with tracer.start_active_span('add'):
                        self.global_model[i].add(buffer)
            else:
                with tracer.start_active_span('send'):
                    for i, para in enumerate(self.global_model):
                        with tracer.start_active_span('layer {}'.format(i)) as scope:
                            scope.span.set_tag('size', para.data.nelement() * para.data.element_size())
                            dist.send(para.data, dst=worker)

        if len(self.active_worker) == 0:
            self.stop()
