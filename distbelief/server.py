# 
"""
Parameter server for distbelief
"""
import logging
import torch
import torch.optim
from distbelief.utils.messaging import MessageListener, send_message
from distbelief.utils.serialization import ravel_model_params, unravel_model_params
from distbelief.utils.tracer import tracer

_LOGGER = logging.getLogger(__name__)


class ParameterServer(MessageListener):
    """ParameterServer"""
    def __init__(self, model, active_worker):
        _LOGGER.info("Creating ParameterServer")
        # By default grads=False in ravel_model_params(), meaning that will only return parameter.data here
        self.parameter_shard = ravel_model_params(model)
        self.active_worker = active_worker
        # Init superclass
        super().__init__(model)

    def receive(self, sender, parameter):
        with tracer.start_active_span('receive') as scope:
            scope.span.set_tag('size', parameter.element_size() * parameter.nelement())
            scope.span.set_tag('worker', sender)
            print("Processing message from sender {}".format(sender))

            if parameter[0] == float('inf'):
                # Notified when a worker is complete
                self.active_worker -= 1
                if self.active_worker == 0:
                    self.stop()
                return

            with tracer.start_active_span('update'):
                self.parameter_shard.add_(parameter)
            # Send the current model back to the sender
            send_message(self.parameter_shard, dst=sender)
