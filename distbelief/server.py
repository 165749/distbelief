import logging
import time
import datetime
import numpy as np
import torch
import torch.optim
import torch.distributed as dist
import multiprocessing
from multiprocessing import shared_memory
from distbelief.utils.trace import Tracer
import psutil

_LOGGER = logging.getLogger(__name__)


class GlobalModel:
    def __init__(self, parameters_with_names):
        tensors = [para.data for name, para in parameters_with_names]
        buffer_shapes = [tensor.numpy().shape for tensor in tensors]
        self.shared_memories = []
        for i, tensor in enumerate(tensors):
            try:
                self.shared_memories.append(shared_memory.SharedMemory(create=True, size=tensor.numpy().nbytes, name=f"layer{i}"))
            except FileExistsError:
                memory = shared_memory.SharedMemory(name=f"layer{i}")
                memory.close()
                memory.unlink()
                self.shared_memories.append(shared_memory.SharedMemory(create=True, size=tensor.numpy().nbytes, name=f"layer{i}"))
        for memory, shape in zip(self.shared_memories, buffer_shapes):
            buffer = np.ndarray(shape, dtype=np.float32, buffer=memory.buf)
            buffer.fill(0)

    def clear(self):
        for memory in self.shared_memories:
            memory.close()
            memory.unlink()


class ParameterServer:
    def __init__(self, parameters_with_names, worker_num):
        _LOGGER.info("Creating ParameterServer")
        self.parameters_with_names = parameters_with_names
        self.global_model = GlobalModel(self.parameters_with_names)
        self.worker_num = worker_num

    def run(self):
        threads = []
        layer_locks = [multiprocessing.Lock() for _ in self.parameters_with_names]
        layer_name = [name.rsplit('.', maxsplit=1) for name, _ in self.parameters_with_names]
        layer_shape = [list(para.data.size()) for name, para in self.parameters_with_names]
        for server_id in range(2, 2*self.worker_num + 1, 2):
            thread = multiprocessing.Process(target=ParameterServer.receive, args=(layer_locks, layer_name, layer_shape, server_id, self.worker_num))
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

        self.global_model.clear()
        print("server 0 finished")

    @classmethod
    def receive(cls, layer_locks, layer_name, layer_shape, server_id, worker_num):
        # Set up communication group
        dist.init_process_group('gloo', rank=server_id, world_size=2 * worker_num + 1, timeout=datetime.timedelta(days=1))
        print("server {} initialized".format(server_id))

        # Start tracer for each server
        tracer = Tracer()

        # Note (zhuojin): A potential issue may appear after forking. Temporarily solve it by set_num_threads to 1.
        torch.set_num_threads(1)

        # Set CPU affinity for the server thread
        proc = psutil.Process()
        proc.cpu_affinity([server_id//2])
        print("Server {} is pinned to core {}".format(server_id, server_id//2))

        # Create tensors mapped to the shared memory
        shared_memories = [shared_memory.SharedMemory(name=f"layer{i}") for i in range(len(layer_shape))]
        global_model = [torch.from_numpy(np.ndarray(shape, dtype=np.float32, buffer=memory.buf)) for memory, shape in zip(shared_memories, layer_shape)]

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
                        for i, gradients in enumerate(gradient_buffers):
                            layer_locks[i].acquire()
                            global_model[i].add_(gradients)
                            layer_locks[i].release()
                    tensor = torch.zeros(1)
                    dist.recv(tensor=tensor, src=server_id - 1)
                    span_step.finish()
                    if tensor[0] == float('inf'):
                        break
                    span_step = tracer.start_span('step {}'.format(step_num))
                    step_num += 1
                    with tracer.start_active_span('collect'):
                        for i, gradients in enumerate(gradient_buffers):
                            layer_locks[i].acquire()
                            gradients.copy_(global_model[i])
                            layer_locks[i].release()
                    for i, para in enumerate(gradient_buffers):
                        with tracer.start_active_span('send') as span:
                            span.set_tag('size', para.nelement() * para.element_size())
                            span.set_tag('layer', layer_name[i][0])
                            span.set_tag('type', layer_name[i][1])
                            dist.send(para, dst=server_id - 1)
        for memory in shared_memories:
            memory.close()
        dist.destroy_process_group()
        tracer.export_traces("server{}.json".format(server_id))
        print("server {} finished".format(server_id))
