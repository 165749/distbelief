from enum import Enum
import logging
import torch
import torch.distributed as dist
from threading import Thread
from distbelief.utils.serialization import ravel_model_params
from distbelief.utils.tracer import tracer

_LOGGER = logging.getLogger(__name__)


class MessageCode(Enum):
    """Different types of messages between client and server that we support go here."""
    ParameterRequest = 0
    GradientUpdate = 1
    ParameterUpdate = 2
    Complete = 3

    @staticmethod
    def to_string(code):
        if code == MessageCode.ParameterRequest:
            return 'ParameterRequest'
        elif code == MessageCode.GradientUpdate:
            return 'GradientUpdate'
        elif code == MessageCode.ParameterUpdate:
            return 'ParameterUpdate'
        elif code == MessageCode.Complete:
            return 'Complete'
        else:
            return 'Error'


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
        self.m_parameter = torch.zeros(ravel_model_params(model).numel() + 2)
        super(MessageListener, self).__init__()

    def receive(self, sender, message_code, parameter):
        """receive

        :param sender: rank id of the sender
        :param message_code: Enum code 
        :param parameter: the data payload
        """
        raise NotImplementedError()

    def run(self):
        _LOGGER.info("Started Running!")
        self.running = True
        while self.running:
            _LOGGER.info("Polling for message...")
            dist.recv(tensor=self.m_parameter)
            self.receive(int(self.m_parameter[0].item()),
                         MessageCode(self.m_parameter[1].item()),
                         self.m_parameter[2:])

    def stop(self):
        self.running = False


def send_message(message_code, payload, dst=0):
    """Sends a message to a destination
    Concatenates both the message code and destination with the payload into a single tensor and then sends that as a tensor
    """
    _LOGGER.info("SENDING MESSAGE: {} RANK: {}".format(message_code, dist.get_rank()))
    with tracer.start_active_span('send') as scope:
        m_parameter = torch.Tensor([dist.get_rank(), message_code.value])
        m_parameter = torch.cat((m_parameter, payload))
        scope.span.set_tag('type', MessageCode.to_string(message_code))
        scope.span.set_tag('size', m_parameter.element_size() * m_parameter.nelement())
        scope.span.set_tag('worker', dist.get_rank())
        # Temporarily use synchronous sending here
        dist.send(tensor=m_parameter, dst=dst)
