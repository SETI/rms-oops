################################################################################
# Empty
#
# Modified 1/2/11 (MRS) -- Refactored Scalar.py to eliminate circular loading
#   dependencies with Array and Empty classes. Array.py, Scalar.py and Empty.py
#   are now separate files.
################################################################################

import numpy as np
import unittest

from oops.broadcastable.Array import Array

class Empty(Array):
    """An empty array, needed in some situations so the moral equivalent of a
    "None" can still respond to basic Array operations."""

    OOPS_CLASS = "Empty"
    IS_EMPTY = True

    def __init__(self, vals=None):

        self.shape = []
        self.rank  = 0
        self.item  = []
        self.vals  = np.array(0)

    # Overrides of standard Array methods
    def __repr__(self): return "Empty()"

    def __str__(self): return "Empty()"

    def __nonzero__(self): return False

    def __getitem__(self, i): return self

    def __getslice__(self, i, j): return self

################################################################################
