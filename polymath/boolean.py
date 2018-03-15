################################################################################
# polymath/boolean.py: Boolean subclass of PolyMath base class
#
# Mark Showalter, PDS Ring-Moon Systems Node, SETI Institute
################################################################################

from __future__ import division
import numpy as np

from .qube   import Qube
from .scalar import Scalar

class Boolean(Scalar):
    """A PolyMath subclass involving booleans. Masked values are unknown,
    neither True nor False.

    Units and derivatives are disallowed."""

    NRANK = 0           # the number of numerator axes.
    NUMER = ()          # shape of the numerator.

    FLOATS_OK = False   # True to allow floating-point numbers.
    INTS_OK = False     # True to allow integers.
    BOOLS_OK = True     # True to allow booleans.

    UNITS_OK = False    # True to allow units; False to disallow them.
    MASKS_OK = True     # True to allow masks; False to disallow them.
    DERIVS_OK = False   # True to disallow derivatives; False to allow them.

    DEFAULT_VALUE = False

    @staticmethod
    def as_boolean(arg, recursive=True):
        """Return the argument converted to Boolean if possible."""

        if type(arg) == Boolean: return arg

        return Boolean(arg, units=False, derivs={})

    def as_int(self):
        """Return a Scalar equal to one where True, zero where False.

        This method overrides the default behavior defined in the base class to
        return a Scalar of ints instead of a Boolean. True become one; False
        becomes zero.
        """

        if np.shape(self.values) == ():
            result = Scalar(int(self.values))
        else:
            result = Scalar(self.values.astype('int'))

        return result

    def as_float(self):
        """Return a floating-point numeric version of this object.

        This method overrides the default behavior defined in the base class to
        return a Scalar of floats instead of a Boolean. True become one; False
        becomes zero.
        """

        if np.shape(self.values) == ():
            result = Scalar(float(self.values))
        else:
            result = Scalar(self.values.astype('float'))

        return result

    def as_numeric(self):
        """Return a numeric version of this object.

        This method overrides the default behavior defined in the base class to
        return a Scalar of ints instead of a Boolean.
        """

        return self.as_int()

    def as_index(self):
        """Return an object suitable for indexing a NumPy ndarray."""

        return (self.values & self.antimask)

    def as_index_and_mask(self):
        """Objects suitable for indexing an N-dimensional array and its mask.

        Return: (indx, mask_indx)
            indx        the index to apply to an array.
            mask_indx1  the index to apply to the mask before the array has
                        already been indexed.
            mask_indx2  the index to apply to the mask after the array has
                        already been indexed.
        """

        if self.mask is True:
            return (False, False)
        elif self.mask is False:
            return (self.values, None)
        else:
            return (self.values, self.mask[self.values])

    def is_numeric(self):
        """Return True if this object is numeric; False otherwise.

        This method overrides the default behavior in the base class to return
        return False. Every other subclass is numeric.
        """

        return False

    def sum(self, axis=None, value=True):
        """Return the number of items matching True or False.

        Input:
            axis        an integer axis or a tuple of axes. The sum is
                        determined across these axes, leaving any remaining axes
                        in the returned value. If None (the default), then the
                        sum is performed across all axes if the object.
            value       value to match.
        """

        if value:
            return self.as_int().sum(axis=axis)
        else:
            return (Scalar.ONE - self.as_int()).sum(axis=axis)

    def identity(self):
        """An object of this subclass equivalent to the identity."""

        return Boolean(True).as_readonly()

    ############################################################################
    # Arithmetic operators
    ############################################################################

    def __pos__(self, recursive=True):
        return self.as_int()

    def __neg__(self, recursive=True):
        return -self.as_int()

    def __abs__(self, recursive=True):
        return self.as_int()

    def __add__(self, arg, recursive=True):
        return self.as_int() + arg

    def __radd__(self, arg, recursive=True):
        return self.as_int() + arg

    def __iadd__(self, arg):
        Qube.raise_unsupported_op('+=', self)

    def __sub__(self, arg, recursive=True):
        return self.as_int() - arg

    def __rsub__(self, arg, recursive=True):
        return -self.as_int() + arg

    def __isub__(self, arg):
        Qube.raise_unsupported_op('-=', self)

    def __mul__(self, arg, recursive=True):
        return self.as_int() * arg

    def __rmul__(self, arg, recursive=True):
        return self.as_int() * arg

    def __imul__(self, arg):
        Qube.raise_unsupported_op('*=', self)

    def __truediv__(self, arg, recursive=True):
        return self.as_int() / arg

    def __rtruediv__(self, arg, recursive=True):
        if not isinstance(arg, Qube): arg = Scalar(arg)
        return arg / self.as_int()

    def __itruediv__(self, arg):
        Qube.raise_unsupported_op('/=', self)

    def __floordiv__(self, arg):
        return self.as_int() // arg

    def __rfloordiv__(self, arg):
        if not isinstance(arg, Qube): arg = Scalar(arg)
        return arg // self.as_int()

    def __ifloordiv__(self, arg):
        Qube.raise_unsupported_op('//=', self)

    def __mod__(self, arg):
        return self.as_int() % arg

    def __rmod__(self, arg):
        if not isinstance(arg, Qube): arg = Scalar(arg)
        return arg % self.as_int()

    def __imod__(self, arg):
        Qube.raise_unsupported_op('%=', self)

    def __pow__(self, arg):
        return self.as_int()**arg

# Useful class constants

Boolean.TRUE = Boolean(True).as_readonly()
Boolean.FALSE = Boolean(False).as_readonly()
Boolean.MASKED = Boolean(False,True).as_readonly()

################################################################################
# Once the load is complete, we can fill in a reference to the Boolean class
# inside the Qube object.
################################################################################

Qube.BOOLEAN_CLASS = Boolean

################################################################################
