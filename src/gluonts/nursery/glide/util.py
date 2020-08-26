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

from toolz.functoolz import curry

from ._partition import partition


class M:
    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class Map(M):
    def __init__(self, fn, xs):
        self.fn = fn
        self.xs = xs

    def __iter__(self):
        yield from map(self.fn, self.xs)

    def __len__(self):
        return len(self.xs)


lift = curry(Map)


class Filter(M):
    def __init__(self, fn, xs):
        self.fn = fn
        self.xs = xs

    def __iter__(self):
        for el in self.xs:
            if self.fn(el):
                yield el

    def __len__(self):
        return sum(1 for _ in self)


@partition.register
def partition_filter(xs: M, n):
    return [type(xs)(xs.fn, part) for part in partition(xs.xs, n)]
