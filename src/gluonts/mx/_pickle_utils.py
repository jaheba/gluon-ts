import io
import pickle
import sys
from functools import partial
from multiprocessing.reduction import ForkingPickler

import mxnet as mx

register_pickler = partial(ForkingPickler.register, mx.nd.NDArray)

# ForkingPickler related functions:
if sys.platform == "darwin" or sys.platform == "win32":

    def rebuild_ndarray(*args):
        """Rebuild ndarray from pickled shared memory"""
        # pylint: disable=no-value-for-parameter
        return mx.nd.NDArray(mx.nd.ndarray._new_from_shared_mem(*args))

    @register_pickler
    def reduce_ndarray(data):
        """Reduce ndarray to shared memory handle"""
        return rebuild_ndarray, data._to_shared_mem()


else:

    def rebuild_ndarray(pid, fd, shape, dtype):
        """Rebuild ndarray from pickled shared memory"""
        # pylint: disable=no-value-for-parameter
        fd = fd.detach()
        return mx.nd.NDArray(
            nd.ndarray._new_from_shared_mem(pid, fd, shape, dtype)
        )

    @register_pickler
    def reduce_ndarray(data):
        """Reduce ndarray to shared memory handle"""
        # keep a local ref before duplicating fd
        data = data.as_in_context(context.Context("cpu_shared", 0))
        pid, fd, shape, dtype = data._to_shared_mem()
        fd = multiprocessing.reduction.DupFd(fd)
        return rebuild_ndarray, (pid, fd, shape, dtype)


def encode(data):
    buf = io.BytesIO()
    ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(data)
    return buf.getvalue()


decode = pickle.loads
