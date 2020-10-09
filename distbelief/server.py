import logging
import time
import datetime
import numpy as np
import torch
import torch.optim
import torch.distributed as dist
import multiprocessing
from multiprocessing.managers import BaseManager
from jaeger_client import Config
from distbelief.utils.tracer import tracer, init_tracer, numbers_to_trace_context, trace_context_to_numbers
from opentracing.propagation import Format

_LOGGER = logging.getLogger(__name__)


class GlobalModel:
    def __init__(self, model):
        self.global_model = [para.data for para in model.parameters()]
        self.lock = multiprocessing.Lock()

    def update(self, gradient_buffer):
        self.lock.acquire()
        for i, gradients in enumerate(gradient_buffer):
            self.global_model[i].add_(gradients)
        self.lock.release()

    def collect(self, gradient_buffer):
        self.lock.acquire()
        for buffer, para in zip(gradient_buffer, self.global_model):
            buffer.copy_(para)
        self.lock.release()

    @classmethod
    def get_tracer(cls):
        return tracer


class ParameterServer:
    def __init__(self, model, worker_num):
        _LOGGER.info("Creating ParameterServer")
        BaseManager.register("GlobalModel", GlobalModel)
        manager = BaseManager()
        manager.start()
        self.global_model = manager.GlobalModel(model)
        self.model = model
        self.worker_num = worker_num

    def run(self):
        threads = []
        layer_name = [name.rsplit('.', maxsplit=1) for name, _ in self.model.named_parameters()]
        layer_shape = [para.data.size() for para in self.model.parameters()]
        for server_id in range(2, 2*self.worker_num + 1, 2):
            thread = multiprocessing.Process(target=ParameterServer.receive, args=(self.global_model, layer_name, layer_shape, server_id, self.worker_num))
            thread.start()
            threads.append(thread)

        # Initialize communication group and root span
        dist.init_process_group('gloo', rank=0, world_size=2 * self.worker_num + 1, timeout=datetime.timedelta(days=1))
        print("server 0 initialized")
        root_span = tracer.start_span('distbelief')
        context = {}
        tracer.inject(root_span, Format.TEXT_MAP, context)
        numbers = trace_context_to_numbers(context)
        tensor = torch.from_numpy(np.array(numbers, dtype=np.int64))
        for idx in range(1, 2 * self.worker_num + 1):
            dist.send(tensor=tensor, dst=idx)  # send to everyone

        for thread in threads:
            thread.join()

        root_span.finish()
        print("server 0 finished")

    @classmethod
    def receive(cls, global_model, layer_name, layer_shape, server_id, worker_num):
        # Set up communication group
        dist.init_process_group('gloo', rank=server_id, world_size=2 * worker_num + 1, timeout=datetime.timedelta(days=1))
        print("server {} initialized".format(server_id))

        # Note (zhuojin): Because currently Jaeger is still incompatible with multiprocessing,
        # have to initialize a new tracer in the subprocess.
        Config._initialized = False
        tracer = init_tracer("distbelief")

        tensor = torch.zeros(4, dtype=torch.int64)  # TODO (zhuojin): Remove hard-code
        dist.recv(tensor=tensor)
        context = numbers_to_trace_context(tensor.tolist())
        span_ctx = tracer.extract(Format.TEXT_MAP, context)
        with tracer.start_active_span('server {}'.format(server_id), child_of=span_ctx):
            span = tracer.start_span("init")
            gradient_buffers = [torch.zeros(shape) for shape in layer_shape]
            step_num = 0

            while True:
                # Receive gradients in the reverse order
                for i in range(len(gradient_buffers)-1, -1, -1):
                    buffer = gradient_buffers[i]
                    with tracer.start_active_span('recv', child_of=span) as scope:
                        scope.span.set_tag('size', buffer.nelement() * buffer.element_size())
                        scope.span.set_tag('layer', layer_name[i][0])
                        scope.span.set_tag('type', layer_name[i][1])
                        dist.recv(tensor=buffer, src=server_id - 1)
                else:
                    with tracer.start_active_span('update', child_of=span):
                        # Update global_model
                        global_model.update(gradient_buffers)
                    tensor = torch.zeros(1)
                    dist.recv(tensor=tensor, src=server_id - 1)
                    span.finish()
                    if tensor[0] == float('inf'):
                        break
                    span = tracer.start_span('Step {}'.format(step_num))
                    step_num += 1
                    with tracer.start_active_span('collect', child_of=span):
                        # Collect global_model in gradient_buffer
                        global_model.collect(gradient_buffers)
                    for i, para in enumerate(gradient_buffers):
                        with tracer.start_active_span('send', child_of=span) as scope:
                            scope.span.set_tag('size', para.nelement() * para.element_size())
                            scope.span.set_tag('layer', layer_name[i][0])
                            scope.span.set_tag('type', layer_name[i][1])
                            dist.send(para, dst=server_id - 1)
        dist.destroy_process_group()
        # Wait for trace collection
        time.sleep(2)
        print("server {} finished".format(server_id))
