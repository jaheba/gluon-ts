import random

from functools import partial

from toolz.dicttoolz import valmap
from toolz.functoolz import curry
from toolz.itertoolz import concat as flatten

import mxnet as mx
import numpy as np

from gluonts.dataset.util import batcher, cycle, dct_reduce as merge_with
from gluonts.nursery import glide

from ._pickle_utils import encode, decode

batcher = curry(batcher)


def shuffled(xs):
    random.shuffle(xs)
    return xs


def create_shuffler(pipe, batch_size, num_shuffle_batches):
    into_batches = batcher(batch_size=batch_size * num_shuffle_batches)
    return (
        pipe.and_then(into_batches).and_then_each(shuffled).and_then(flatten)
    )


@curry
def as_in_context(val, ctx):
    if isinstance(val, mx.nd.NDArray):
        return val.as_in_context(ctx)
    return val


def stack(data, ctx=None, dtype=np.float32):
    peek = data[0]
    if isinstance(peek, (mx.nd.NDArray, np.ndarray)) and not 0 in peek.shape:
        return mx.nd.array(data, dtype=dtype, ctx=ctx)

    return data


def _decode(batch, context):
    decoded = decode(batch)
    return valmap(as_in_context(ctx=context), decoded)


def batchify(batch):
    return merge_with(stack, batch,)


def print_return(x):
    print("XXX")
    return x


def inference_data_loader(data, transform, context, batch_size):
    pipe = (
        glide.Pipeline()
        .and_then(transform)
        .and_then(batcher(batch_size=batch_size))
        .and_then_each(batchify)
    )

    return pipe.parapply(
        data, encode=encode, decode=partial(_decode, context=context)
    )


def train_data_loader(
    data,
    transform,
    context,
    batch_size,
    num_shuffle_batches=0,
    cache_data=False,
):

    pipe = glide.Pipeline([cycle, transform])

    if cache_data:
        pipe = pipe.but_first(list)

    if num_shuffle_batches:
        pipe = create_shuffler(pipe, batch_size, num_shuffle_batches)

    pipe = pipe.and_then(batcher(batch_size=batch_size))
    pipe = pipe.and_then_each(batchify)

    return pipe.parapply(
        data, encode=encode, decode=partial(_decode, context=context)
    )
