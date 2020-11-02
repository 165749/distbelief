import logging
import time
import datetime
import numpy as np
import torch
import torch.optim
import torch.distributed as dist
import multiprocessing
from multiprocessing.managers import BaseManager
from distbelief.utils.trace import Tracer

_LOGGER = logging.getLogger(__name__)


class GlobalModel:
    def __init__(self, parameters_with_names):
        self.global_model = [para.data for name, para in parameters_with_names]
        self.locks = [multiprocessing.Lock() for _ in range(len(parameters_with_names))]

    def update(self, gradient_buffer):
        for i, gradients in enumerate(gradient_buffer):
            self.locks[i].acquire()
            self.global_model[i].add_(gradients)
            self.locks[i].release()

    def collect(self, gradient_buffer):
        for i, gradients in enumerate(gradient_buffer):
            self.locks[i].acquire()
            gradients.copy_(self.global_model[i])
            self.locks[i].release()


class ParameterServer:
    def __init__(self, parameters_with_names, worker_num):
        _LOGGER.info("Creating ParameterServer")
        BaseManager.register("GlobalModel", GlobalModel)
        manager = BaseManager()
        manager.start()
        self.parameters_with_names = parameters_with_names
        self.global_model = manager.GlobalModel(self.parameters_with_names)
        self.worker_num = worker_num

    def run(self):
        threads = []
        layer_name = [name.rsplit('.', maxsplit=1) for name, _ in self.parameters_with_names]
        layer_shape = [list(para.data.size()) for name, para in self.parameters_with_names]
        for server_id in range(2, 2*self.worker_num + 1, 2):
            thread = multiprocessing.Process(target=ParameterServer.receive, args=(self.global_model, layer_name, layer_shape, server_id, self.worker_num))
            thread.start()
            threads.append(thread)

        # Initialize communication group and root span
        dist.init_process_group('gloo', rank=0, world_size=2 * self.worker_num + 1, timeout=datetime.timedelta(days=1))
        print("server 0 initialized")

        # Wait for everyone to be ready
        for idx in range(1, 2 * self.worker_num + 1):
            dist.recv(tensor=torch.zeros(1), src=idx)
        for idx in range(1, 2 * self.worker_num + 1):
            dist.send(tensor=torch.zeros(1), dst=idx)  # Send to everyone

        for thread in threads:
            thread.join()

        print("server 0 finished")

    @classmethod
    def receive(cls, global_model, layer_name, layer_shape, server_id, worker_num):
        # Set up communication group
        dist.init_process_group('gloo', rank=server_id, world_size=2 * worker_num + 1, timeout=datetime.timedelta(days=1))
        print("server {} initialized".format(server_id))

        # Start tracer for each server
        tracer = Tracer()

        # Note (zhuojin): A potential issue may appear after forking. Temporarily solve it by set_num_threads to 1.
        torch.set_num_threads(1)

        with tracer.start_active_span('server {}'.format(server_id)):
            span_step = tracer.start_span("init")
            gradient_buffers = [torch.zeros(shape) for shape in layer_shape]
            step_num = 0

            # Wait for starting up
            with tracer.start_active_span('wait'):
                dist.send(tensor=torch.zeros(1), dst=0)
                dist.recv(tensor=torch.zeros(1), src=0)
            while True:
                # Receive gradients in the reverse order
                for i in range(len(gradient_buffers)-1, -1, -1):
                    buffer = gradient_buffers[i]
                    with tracer.start_active_span('recv') as span:
                        span.set_tag('layer', layer_name[i][0])
                        span.set_tag('type', layer_name[i][1])
                        dist.recv(tensor=buffer, src=server_id - 1)
                else:
                    with tracer.start_active_span('update'):
                        # Update global_model
                        global_model.update(gradient_buffers)
                    tensor = torch.zeros(1)
                    dist.recv(tensor=tensor, src=server_id - 1)
                    span_step.finish()
                    if tensor[0] == float('inf'):
                        break
                    span_step = tracer.start_span('step {}'.format(step_num))
                    step_num += 1
                    with tracer.start_active_span('collect'):
                        # Collect global_model in gradient_buffer
                        global_model.collect(gradient_buffers)
                    for i, para in enumerate(gradient_buffers):
                        with tracer.start_active_span('send') as span:
                            span.set_tag('size', para.nelement() * para.element_size())
                            span.set_tag('layer', layer_name[i][0])
                            span.set_tag('type', layer_name[i][1])
                            dist.send(para, dst=server_id - 1)
        dist.destroy_process_group()
        tracer.export_traces("server{}.json".format(server_id))
        print("server {} finished".format(server_id))
