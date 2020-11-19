import logging
import time
import datetime
import numpy as np
import torch
import torch.optim
import torch.distributed as dist
from distbelief.utils.trace import Tracer
import psutil

_LOGGER = logging.getLogger(__name__)


class ParameterServer:
    def __init__(self, args, model, worker_num):
        _LOGGER.info("Creating ParameterServer")
        # Store model in the shared memory
        model.share_memory()
        if args.ignore_bn:
            bn_names = [name for name, module in model.named_modules() if isinstance(module, torch.nn.BatchNorm2d)]
            parameters_with_names = [(name, para) for name, para in model.named_parameters() if
                                     name.rsplit('.', maxsplit=1)[0] not in bn_names]
        else:
            parameters_with_names = [(name, para) for name, para in model.named_parameters()]
        self.parameters_with_names = parameters_with_names
        self.worker_num = worker_num

    def run(self):
        threads = []
        torch.multiprocessing.set_start_method('spawn')
        for server_id in range(2, 2*self.worker_num + 1, 2):
            thread = torch.multiprocessing.Process(target=ParameterServer.receive, args=(self.parameters_with_names, server_id, self.worker_num))
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
    def receive(cls, parameters_with_names, server_id, worker_num):
        # Set up communication group
        dist.init_process_group('gloo', rank=server_id, world_size=2 * worker_num + 1, timeout=datetime.timedelta(days=1))
        print("server {} initialized".format(server_id))

        # Start tracer for each server
        tracer = Tracer()

        # # Set CPU affinity for the server thread
        # proc = psutil.Process()
        # proc.cpu_affinity([server_id//2])
        # print("Server {} is pinned to core {}".format(server_id, server_id//2))

        global_model = [para.data for _, para in parameters_with_names]
        layer_name = [name.rsplit('.', maxsplit=1) for name, _ in parameters_with_names]

        with tracer.start_active_span('server {}'.format(server_id)):
            gradient_buffers = [torch.zeros(para.data.size()) for _, para in parameters_with_names]
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
                        for i, gradients in enumerate(gradient_buffers):
                            global_model[i].add_(gradients)
                    tensor = torch.zeros(1)
                    dist.recv(tensor=tensor, src=server_id - 1)
                    span_step.finish()
                    if tensor[0] == float('inf'):
                        break
                    span_step = tracer.start_span('step {}'.format(step_num))
                    step_num += 1
                    with tracer.start_active_span('collect'):
                        for i, gradients in enumerate(gradient_buffers):
                            gradients.copy_(global_model[i])
                    for i, para in enumerate(gradient_buffers):
                        with tracer.start_active_span('send') as span:
                            span.set_tag('size', para.nelement() * para.element_size())
                            span.set_tag('layer', layer_name[i][0])
                            span.set_tag('type', layer_name[i][1])
                            dist.send(para, dst=server_id - 1)
        dist.destroy_process_group()
        tracer.export_traces("server{}.json".format(server_id))
        print("server {} finished".format(server_id))
