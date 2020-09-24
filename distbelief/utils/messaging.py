from enum import Enum
import logging
import torch
import torch.distributed as dist
from threading import Thread
from distbelief.utils.serialization import ravel_model_params
from distbelief.utils.tracer import tracer

_LOGGER = logging.getLogger(__name__)


class MessageListener(Thread):
    """MessageListener
   
    base class for message listeners, extends pythons threading Thread
    """
    def __init__(self, model):
        """__init__

        :param model: nn.Module to be defined by the user
        """
        self.model = model
        self.running = True
        _LOGGER.info("Setting m_parameter")
        super(MessageListener, self).__init__()

    def receive(self):
        """receive

        :param sender: rank id of the sender
        :param parameter: the data payload
        """
        raise NotImplementedError()

    def run(self):
        _LOGGER.info("Started Running!")
        self.running = True
        while self.running:
            _LOGGER.info("Polling for message...")
            self.receive()

    def stop(self):
        self.running = False


def send_message(payload, dst=0):
    """Sends a message to a destination
    Concatenates both the message code and destination with the payload into a single tensor and then sends that as a tensor
    """
    _LOGGER.info("SENDING MESSAGE FROM RANK: {}".format(dist.get_rank()))
    with tracer.start_active_span('send') as scope:
        scope.span.set_tag('size', payload.element_size() * payload.nelement())
        scope.span.set_tag('worker', dist.get_rank())
        # TODO (zhuojin): Temporarily choose synchronous sending here
        dist.send(tensor=payload, dst=dst)
