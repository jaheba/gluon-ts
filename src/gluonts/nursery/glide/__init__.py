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

"""
gluonts.nursery.glide
~~~~~~~~~~~~~~~~~~~~~

Glide is a simple pipeline, which is able to map functions in parallel.

The core idea is to work on already sharded input-data, since inter-process
communication is expensive using multiprocessing. Thus, glide's only
syncronisation point is a final result queue to which each process emits its
data.

To help splitting the input preemptively glides offers a `partition`-method.

    from gluonts.nursery import glide

    data = range(100)
    parts = glide.partition(data, 3)

    def double(n):
        return n * 2

    assert set(glide.ParApply(glide.lift(double), parts)) == set(range(0, 200, 2))

"""
__all__ = ["partition", "Apply", "ParApply", "lift", "Map", "Pipeline"]

from functools import partial

from toolz.functoolz import curry

from ._partition import partition, divide_into
from .parallel import ParApply
from .sequential import Apply
from .pipeline import Pipeline


@curry
class Map:
    def __init__(self, fn, xs):
        self.fn = fn
        self.xs = xs

    def __iter__(self):
        yield from map(self.fn, self.xs)


lift = Map
