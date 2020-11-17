import random
import time
import collections
import gzip
import json
import torch


class Span:
    def __init__(self, tracer, parent_id, span_id, name):
        self.tracer = tracer
        self.parent_id = parent_id
        self.span_id = span_id
        self.tags = []
        self.start_time = None
        self.end_time = None
        self.name = name

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

    def start(self):
        if self.tracer.cuda:
            torch.cuda.synchronize()
        self.start_time = time.time_ns()

    def finish(self):
        if self.tracer.cuda:
            torch.cuda.synchronize()
        if self.end_time is None:
            self.end_time = time.time_ns()
            self.tracer.finish_current_span()
        else:
            print("warning: span has already been finished")

    def set_tag(self, key, value):
        self.tags.append(f'{key}#{value}')


# Note: tracer is not thread-safe
class Tracer:
    def __init__(self, cuda=False):
        self.cuda = cuda
        self.traces_collection = []
        self.current_span = Span(tracer=self, parent_id=0, span_id=0, name="root")
        self.context_stack = collections.deque()
        self.context_stack.append(self.current_span)

    def start_span(self, name):
        span_id = random.randrange(1, 18446744073709551616)  # random between (0, 16^16)
        self.context_stack.append(self.current_span)
        span = Span(tracer=self, parent_id=self.current_span.span_id, span_id=span_id, name=name)
        self.current_span = span
        span.start()
        return span

    def start_active_span(self, name):
        span_id = random.randrange(1, 18446744073709551616)  # random between (0, 16^16)
        self.context_stack.append(self.current_span)
        span = Span(tracer=self, parent_id=self.current_span.span_id, span_id=span_id, name=name)
        self.current_span = span
        return span

    def finish_current_span(self):
        self.traces_collection.append(self.current_span)
        self.current_span = self.context_stack.pop()

    def export_traces(self, filename):
        with gzip.open(filename + ".gz", "wt") as file:
            json.dump([{
                "id": "{:016x}".format(span.span_id),
                "parent": "{:016x}".format(span.parent_id),
                "start": span.start_time//1000,
                "end": span.end_time//1000,
                "op": span.name,
                "tags": {k: v for k, v in map(lambda x: x.split("#"), span.tags)},
            } for span in self.traces_collection], file)
