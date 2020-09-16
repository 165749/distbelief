# 
"""
Parameter server for distbelief
"""
import logging
import torch
import torch.optim
from distbelief.utils.messaging import MessageCode, MessageListener, send_message
from distbelief.utils.serialization import ravel_model_params, unravel_model_params
from distbelief.utils.tracer import tracer

_LOGGER = logging.getLogger(__name__)

class ParameterServer(MessageListener):
    """ParameterServer"""
    def __init__(self, model, active_worker):
        _LOGGER.info("Creating ParameterServer")
        # By default grads=False in ravel_model_params(), meaning that will only return parameter.data here
        self.parameter_shard = torch.rand(ravel_model_params(model).numel())
        self.model = model
        self.active_worker = active_worker
        #init superclass
        super().__init__(model)

    def receive(self, sender, message_code, parameter):
        with tracer.start_active_span('receive'):
            print("Processing message: {} from sender {}".format(message_code.name, sender))

            if message_code == MessageCode.ParameterUpdate:
                # TODO (zhuojin): Think about the case of multiple workers
                #be sure to clone here
                self.parameter_shard = parameter.clone()

            elif message_code == MessageCode.ParameterRequest:
                send_message(MessageCode.ParameterUpdate, self.parameter_shard, dst=sender)

            elif message_code == MessageCode.GradientUpdate:
                self.parameter_shard.add_(parameter)

            elif message_code == MessageCode.Complete:
                # Confirm with worker
                # TODO (zhuojin): dummy val for the second argument
                send_message(MessageCode.Complete, torch.zeros(self.parameter_shard.size()), dst=sender)
                self.active_worker -= 1
                if self.active_worker == 0:
                    self.stop()
