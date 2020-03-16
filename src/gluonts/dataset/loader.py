# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
import itertools
import pickle
import io
import sys
import multiprocessing
import multiprocessing.queues
import time
from multiprocessing.reduction import ForkingPickler, DupFd, recv_handle
from multiprocessing.pool import ThreadPool, Pool
from multiprocessing import Queue
from typing import Callable, Iterable
import copy

import numpy as np
from gluonts.core.component import DType
from gluonts.dataset.common import Dataset, FileDataset, ListDataset
from gluonts.transform import Transformation
from mxnet.ndarray import NDArray

try:
    import multiprocessing.resource_sharer
except ImportError:
    pass

from mxnet import nd, context
import mxnet as mx

if sys.platform == "darwin" or sys.platform == "win32":

    def rebuild_ndarray(*args):
        """Rebuild ndarray from pickled shared memory"""
        # pylint: disable=no-value-for-parameter
        return nd.NDArray(nd.ndarray._new_from_shared_mem(*args))

    def reduce_ndarray(data):
        """Reduce ndarray to shared memory handle"""
        return rebuild_ndarray, data._to_shared_mem()


else:

    def rebuild_ndarray(pid, fd, shape, dtype):
        """Rebuild ndarray from pickled shared memory"""
        # pylint: disable=no-value-for-parameter
        if sys.version_info[0] == 2:
            fd = multiprocessing.reduction.rebuild_handle(fd)
        else:
            fd = fd.detach()
        return nd.NDArray(
            nd.ndarray._new_from_shared_mem(pid, fd, shape, dtype)
        )

    def reduce_ndarray(data):
        """Reduce ndarray to shared memory handle"""
        # keep a local ref before duplicating fd
        data = data.as_in_context(context.Context("cpu_shared", 0))
        pid, fd, shape, dtype = data._to_shared_mem()
        if sys.version_info[0] == 2:
            fd = multiprocessing.reduction.reduce_handle(fd)
        else:
            fd = DupFd(fd)
        return rebuild_ndarray, (pid, fd, shape, dtype)


ForkingPickler.register(nd.NDArray, reduce_ndarray)


# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Standard library imports
import functools
from typing import Any, Dict, Iterable, Iterator

# Third-party imports
import mxnet as mx
import numpy as np

# First-party imports
from gluonts.core.component import DType
from gluonts.dataset.common import DataEntry, Dataset
from gluonts.transform import Transformation

from .util import take, batcher, dct_reduce, shuffler, cycle

DataBatch = Dict[str, Any]


class DataLoader(Iterable[DataEntry]):
    """
    An abstract Iterable type for iterating and transforming a dataset,
    in batches of a prescribed size.

    Parameters
    ----------
    dataset
        The dataset from which to load data.
    transform
        A transformation to apply to each entry in the dataset.
    batch_size
        The size of the batches to emit.
    ctx
        MXNet context to use to store data.
    dtype
        Floating point type to use.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        transform: Transformation,
        is_train: bool,
        batch_size: int,
        ctx: mx.Context,
        dtype: DType = np.float32,
    ) -> None:
        self.batch_size = batch_size
        self.ctx = ctx
        self.dtype = dtype
        self.is_train = is_train
        self.dataset = dataset
        self.transform = transform

    @property
    def stream(self) -> Iterable:
        return self.transform(self.dataset, is_train=self.is_train)

    def make_batch_iter(self):
        batches = batcher(self.stream, self.batch_size)
        stack = functools.partial(dct_reduce, self.stack)
        return map(stack, batches)

    def stack(self, xs):
        if isinstance(xs[0], np.ndarray):
            data = np.asarray(xs)
            if data.dtype.kind == "f":
                data = data.astype(self.dtype)
            return mx.nd.array(data, dtype=data.dtype, ctx=self.ctx)

        if isinstance(xs[0], mx.nd.NDArray):
            return mx.nd.stack(*xs)

        if isinstance(xs[0], list):
            return list(self.stack(t) for t in zip(*xs))

        if isinstance(xs[0], tuple):
            return tuple(self.stack(t) for t in zip(*xs))

        return xs

    def __iter__(self) -> Iterator[DataBatch]:
        return self.make_batch_iter()


class TrainDataLoader(DataLoader):
    """
    An Iterable type for iterating and transforming a dataset, in batches of a
    prescribed size, until a given number of batches is reached.

    The transformation are applied with in training mode, i.e. with the flag
    `is_train = True`.

    Parameters
    ----------
    dataset
        The dataset from which to load data.
    transform
        A transformation to apply to each entry in the dataset.
    batch_size
        The size of the batches to emit.
    ctx
        MXNet context to use to store data.
    num_batches_per_epoch
        Number of batches to return in one complete iteration over this object.
    dtype
        Floating point type to use.
    """

    def __init__(
        self,
        dataset: Dataset,
        transform: Transformation,
        batch_size: int,
        ctx: mx.Context,
        num_batches_per_epoch: int,
        dtype: DType = np.float32,
        shuffle_for_training: bool = True,
        num_batches_for_shuffling: int = 10,
    ) -> None:
        assert dataset, "empty dataset"

        super().__init__(
            dataset=cycle(dataset),
            transform=transform,
            batch_size=batch_size,
            ctx=ctx,
            dtype=dtype,
            is_train=True,
        )

        self.num_batches_per_epoch = num_batches_per_epoch
        self.shuffle_for_training = shuffle_for_training
        self.num_batches_for_shuffling = num_batches_for_shuffling

    @property
    def stream(self) -> Iterable:
        s = self.transform(self.dataset, is_train=self.is_train)
        if self.shuffle_for_training:
            return shuffler(s, self.num_batches_for_shuffling)
        return s

    def __len__(self) -> int:
        return self.num_batches_per_epoch

    def __iter__(self) -> Iterator[DataBatch]:
        batches = self.make_batch_iter()
        return take(batches, self.num_batches_per_epoch)


class ValidationDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        *,
        transform: Transformation,
        batch_size: int,
        ctx: mx.Context,
        dtype: DType = np.float32,
    ) -> None:
        super().__init__(
            dataset,
            transform=transform,
            is_train=True,
            batch_size=batch_size,
            ctx=ctx,
            dtype=dtype,
        )


class InferenceDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        *,
        transform: Transformation,
        batch_size: int,
        ctx: mx.Context,
        dtype: DType = np.float32,
    ) -> None:
        super().__init__(
            dataset,
            transform=transform,
            is_train=False,
            batch_size=batch_size,
            ctx=ctx,
            dtype=dtype,
        )
