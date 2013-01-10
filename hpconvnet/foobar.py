"""
foobar.py - numeric tracing utilites

"""
import copy
import logging
import numpy as np

logger = logging.getLogger(__name__)

_trace_target = []

_trace = []

trace_enabled = True
trace_verify = False

def trace_eq(a, b):
    if len(a) != len(b):
        raise ValueError('Length mismatch', (a, b))
    rval = True
    for i, (ai, bi) in enumerate(zip(a, b)):
        if isinstance(ai, basestring):
            if i == 0:
                if ai != bi:
                    logger.error('TRACE: %s != %s' % (ai, bi))
                    rval = False
            else:
                if ai != bi:
                    logger.warn('TRACE: %s != %s' % (ai, bi))
                    # -- we're letting this go for now...
        elif isinstance(ai, float):
            if not np.allclose(ai, bi, atol=1e-3, rtol=1e-3):
                logger.error('TRACE: %s != %s' % (ai, bi))
                rval = False
        elif isinstance(ai, int):
            if not ai == bi:
                logger.error('TRACE: %s != %s' % (ai, bi))
                rval = False
        elif isinstance(ai, (tuple, list)):
            if not trace_eq(ai, bi):
                pass # -- letting this go for now
    return rval



def verify_last_trace():
    if trace_verify:
        target = _trace_target[len(_trace)-1]
        logger.info("Verifying trace: %s" % str(target))
        assert trace_eq(target, _trace[-1])


def append_trace(*args):
    if trace_enabled:
        logger.info("Appending trace: %s" % str(args))
        _trace.append(args)
        verify_last_trace()


def append_ndarray_signature(x, *args):
    assert isinstance(x, np.ndarray)
    sig = (str(x.dtype), x.shape)
    if x.size:
        sig = sig + (x.min(), x.max(), x.mean())
    return append_trace(*(args + sig))


def append_randomstate(msg, rng, *args):
    sample = copy.deepcopy(rng).randn()
    return append_trace(msg, sample, *args)


def reset_trace():
    _trace[:] = []


def set_trace_target(trace_target):
    _trace_target[:] = trace_target

