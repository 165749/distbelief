import logging
import time
import datetime
import numpy as np
import torch
import torch.optim
import torch.distributed as dist
from distbelief.utils.trace import Tracer

_LOGGER = logging.getLogger(__name__)


class ParameterServer:
    def __init__(self, args, model, worker_num):
        _LOGGER.info("Creating ParameterServer")
        # Store model in the shared memory
        if args.cuda:
            model = model.cuda()
        model.share_memory()
        if args.ignore_bn:
            bn_names = [name for name, module in model.named_modules() if isinstance(module, torch.nn.BatchNorm2d)]
            parameters_with_names = [(name, para) for name, para in model.named_parameters() if
                                     name.rsplit('.', maxsplit=1)[0] not in bn_names]
        else:
            parameters_with_names = [(name, para) for name, para in model.named_parameters()]
        self.parameters_with_names = parameters_with_names
        self.worker_num = worker_num
        self.cuda = args.cuda
        self.sync = args.sync
        self.all_reduce = args.all_reduce

    def run(self):
        threads = []
        torch.multiprocessing.set_start_method('spawn')
        barrier = torch.multiprocessing.Barrier(self.worker_num) if self.sync else None
        for server_id in range(2, 2*self.worker_num + 1, 2):
            thread = torch.multiprocessing.Process(target=ParameterServer.receive, args=(self.parameters_with_names, server_id, self.worker_num, self.cuda, barrier, self.all_reduce))
            thread.start()
            threads.append(thread)

        # Initialize communication group
        dist.init_process_group('gloo', rank=0, world_size=2 * self.worker_num + 1)
        if self.all_reduce:
            # Create new group for all workers to perform all_reduce
            dist.new_group([i for i in range(1, dist.get_world_size(), 2)])
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
    def receive(cls, parameters_with_names, server_id, worker_num, cuda, barrier, all_reduce):
        # Set up communication group
        dist.init_process_group('gloo', rank=server_id, world_size=2 * worker_num + 1)
        if all_reduce:
            # Create new group for all workers to perform all_reduce
            dist.new_group([i for i in range(1, dist.get_world_size(), 2)])
        print("server {} initialized".format(server_id))

        # Start tracer for each server
        tracer = Tracer(cuda=cuda)

        global_model = [para.data for _, para in parameters_with_names]
        layer_name = [name.rsplit('.', maxsplit=1) for name, _ in parameters_with_names]

        with tracer.start_active_span('server {}'.format(server_id)):
            span_step = tracer.start_span("init")
            gradient_buffers = [torch.zeros(para.data.size()) for _, para in parameters_with_names]
            step_num = 0

            # Wait for starting up
            with tracer.start_active_span('wait'):
                dist.send(tensor=torch.zeros(1), dst=0)
                dist.recv(tensor=torch.zeros(1), src=0)
            while True:
                if all_reduce:
                    tensor = torch.zeros(1)
                    dist.recv(tensor=tensor, src=server_id - 1)
                    span_step.finish()
                    if tensor[0] == float('inf'):
                        break
                    span_step = tracer.start_span('step {}'.format(step_num))
                    step_num += 1
                else:
                    receivers = []
                    # Receive gradients in the reverse order
                    for buffer in reversed(gradient_buffers):
                        receivers.append(dist.irecv(tensor=buffer, src=server_id - 1))
                    for i, receiver in enumerate(receivers):
                        with tracer.start_active_span('recv') as span:
                            span.set_tag('layer', layer_name[-1 - i][0])
                            span.set_tag('type', layer_name[-1 - i][1])
                            receiver.wait()
                            if barrier is None:  # For async
                                if cuda:
                                    global_model[-1 - i].add_(gradient_buffers[-1 - i].cuda())
                                    gradient_buffers[-1 - i].copy_(global_model[-1 - i])
                                else:
                                    global_model[-1 - i].add_(gradient_buffers[-1 - i])
                                    gradient_buffers[-1 - i].copy_(global_model[-1 - i])
                    if barrier is not None:  # For sync
                        with tracer.start_active_span('update'):
                            for i, gradients in enumerate(gradient_buffers):
                                if cuda:
                                    global_model[i].add_(gradients.cuda())
                                else:
                                    global_model[i].add_(gradients)
                        with tracer.start_active_span('collect'):
                            for i, gradients in enumerate(gradient_buffers):
                                if cuda:
                                    gradient_buffers[i] = global_model[i].cpu()
                                else:
                                    gradients.copy_(global_model[i])
                    tensor = torch.zeros(1)
                    dist.recv(tensor=tensor, src=server_id - 1)
                    if barrier is not None:
                        with tracer.start_active_span('barrier'):
                            barrier.wait()
                    span_step.finish()
                    if tensor[0] == float('inf'):
                        break
                    span_step = tracer.start_span('step {}'.format(step_num))
                    step_num += 1
                    for i, para in enumerate(gradient_buffers):
                        with tracer.start_active_span('send') as span:
                            span.set_tag('size', para.nelement() * para.element_size())
                            span.set_tag('layer', layer_name[i][0])
                            span.set_tag('type', layer_name[i][1])
                            dist.send(para, dst=server_id - 1)
        dist.destroy_process_group()
        tracer.export_traces("server{}.json".format(server_id))
        print("server {} finished".format(server_id))
