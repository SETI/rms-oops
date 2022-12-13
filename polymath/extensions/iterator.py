################################################################################
# polymath/extensions/iterator.py: iterator over Qube objects
################################################################################

import itertools
import numpy as np
import sys

from ..qube import Qube

class QubeIterator(object):
    """Iterator across the first axis of an object."""

    def __init__(self, obj):

        if not obj._shape_:
            self.obj = [obj]
            self.stop = 1
        else:
            self.obj = obj
            self.stop = obj._shape_[0]

        self.index = -1

    def __iter__(self):
        self.index = -1
        return self

    def __next__(self):
        self.index += 1
        if self.index >= self.stop:
            raise StopIteration

        return self.obj[self.index]

    # Python 2 support
    if sys.version_info.major < 3:
        def next(self):
            return QubeIterator.__next__(self)


class QubeNDIterator(object):
    """Iterator across all the axes of an object. Returns (index_tuple, item).
    """

    def __init__(self, obj):

        if not obj._shape_:
            self.obj = np.array([obj], dtype='object')
            self.shape = (1,)
        else:
            self.obj = obj
            self.shape = obj._shape_

        self.iterator = None

    def __iter__(self):
        self.iterator = itertools.product(*[range(s) for s in self.shape])
        return self

    def __next__(self):
        indx = self.iterator.__next__()
        return (indx, self.obj[indx])

    # Python 2 support
    if sys.version_info.major < 3:
        def next(self):
            indx = self.iterator.next()
            return (indx, self.obj[indx])


def __iter__(self):
    return QubeIterator(obj=self)

def ndenumerate(self):
    """Iterate across all the axes of an object. Returned values are (index
    tuple, item at index).
    """

    return QubeNDIterator(obj=self)

################################################################################
