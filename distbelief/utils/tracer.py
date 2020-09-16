from jaeger_client import Config
import logging
import struct


def init_tracer(service):
    logging.getLogger('').handlers = []
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    config = Config(
        config={
            'sampler': {
                'type': 'const',
                'param': 1,
            },
            'logging': True,
        },
        service_name=service,
    )

    # this call also sets opentracing.tracer
    return config.initialize_tracer()


def trace_context_to_numbers(context):
    assert 'uber-trace-id' in context.keys()
    return [struct.unpack('>q', struct.pack('>Q', int(id_str, 16)))[0] for id_str in context['uber-trace-id'].split(':')]


def numbers_to_trace_context(numbers):
    return {'uber-trace-id': ':'.join(['{:x}'.format(struct.unpack('>Q', struct.pack('>q', id))[0]) for id in numbers])}


tracer = init_tracer('distbelief')
