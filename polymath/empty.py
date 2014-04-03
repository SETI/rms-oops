################################################################################
# polymath/modules/empty.py: Empty subclass of the PolyMath base class
#
# Mark Showalter, PDS Rings Node, SETI Institute, February 2014
################################################################################

from __future__ import division
import numpy as np

from qube import Qube

class Empty(Qube):
    """An Empty object needed in some situations as the moral equivalent of
    "None", indicating that something is undefined or inapplicable. However, an
    Empty object still responds to basic PolyMath methods. Any operation
    involving an Empty object returns an Empty object."""

    NRANK = 0           # the number of numerator axes.
    NUMER = ()          # shape of the numerator.

    FLOATS_OK = False   # True to allow floating-point numbers.
    INTS_OK = False     # True to allow integers.
    BOOLS_OK = False    # True to allow booleans.

    UNITS_OK = False    # True to allow units; False to disallow them.
    MASKS_OK = False    # True to allow masks; False to disallow them.
    DERIVS_OK = False   # True to disallow derivatives; False to allow them.

    def __init__(self, *arg, **keywords):
        Qube.__init__(self, 0, False, None, {}, nrank=0, drank=0, example=None)

    # Overrides of standard Qube methods

    def clone(self, recursive=True): return Empty()

    def __repr__(self): return "Empty()"

    def __str__(self): return "Empty()"

    def __nonzero__(self): return False

    def __getitem__(self, i): return self

    def __setitem__(self, i, arg):
        raise TypeError('Empty class does not support item assignment')

    # All arithmetic operations involving Empty return Empty

    def __pos__(self, recursive=True): return self
    def __neg__(self, recursive=True): return self
    def __abs__(self, recursive=True): return self

    def __add__(self,  arg, recursive=True): return self
    def __radd__(self, arg, recursive=True): return self
    def __iadd__(self, arg): return self

    def __sub__(self,  arg, recursive=True): return self
    def __rsub__(self, arg, recursive=True): return self
    def __isub__(self, arg): return self

    def __mul__(self,  arg, recursive=True): return self
    def __rmul__(self, arg, recursive=True): return self
    def __imul__(self, arg): return self

    def __div__(self,  arg, recursive=True): return self
    def __rdiv__(self, arg, recursive=True): return self
    def __idiv__(self, arg): return self

    def __truediv__(self,  arg, recursive=True): return self
    def __rtruediv__(self, arg, recursive=True): return self
    def __itruediv__(self, arg): return self

    def __floordiv__(self,  arg): return self
    def __rfloordiv__(self, arg): return self
    def __ifloordiv__(self, arg): return self

    def __mod__(self,  arg): return self
    def __rmod__(self, arg): return self
    def __imod__(self, arg): return self

    def __pow__(self, arg, recursive=True): return self

    def reciprocal(self, recursive=True, nozeros=False): return self

    def __invert__(self): return self
    def __and__(self, arg): return self
    def __or__(self, arg): return self
    def __xor__(self, arg): return self
    def __iand__(self, arg): return self
    def __ior__(self, arg): return self
    def __ixor__(self, arg): return self

# Useful class constants

Empty.EMPTY = Empty()

################################################################################
# Once defined, register with the base class
################################################################################

Qube.EMPTY_CLASS = Empty

################################################################################
