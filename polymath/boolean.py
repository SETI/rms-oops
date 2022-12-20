################################################################################
# polymath/boolean.py: Boolean subclass of PolyMath base class
################################################################################

from __future__ import division
import numpy as np

from polymath.qube   import Qube
from polymath.scalar import Scalar

class Boolean(Scalar):
    """A PolyMath subclass involving booleans. Masked values are unknown,
    neither True nor False.

    Units and derivatives are disallowed.
    """

    NRANK = 0           # the number of numerator axes.
    NUMER = ()          # shape of the numerator.

    FLOATS_OK = False   # True to allow floating-point numbers.
    INTS_OK = False     # True to allow integers.
    BOOLS_OK = True     # True to allow booleans.

    UNITS_OK = False    # True to allow units; False to disallow them.
    DERIVS_OK = False   # True to allow derivatives and to allow this class to
                        # have a denominator; False to disallow them.

    DEFAULT_VALUE = False

    #===========================================================================
    @staticmethod
    def as_boolean(arg, recursive=True):
        """The argument converted to Boolean if possible."""

        if isinstance(arg, Boolean):
            return arg

        if isinstance(arg, np.bool_):   # np.bool_ is not a subclass of bool
            arg = bool(arg)

        return Boolean(arg, units=False, derivs={})

    #===========================================================================
    def as_int(self):
        """A Scalar equal to one where True, zero where False.

        This method overrides the default behavior defined in the base class to
        return a Scalar of ints instead of a Boolean. True become one; False
        becomes zero.
        """

        if np.isscalar(self._values_):
            result = Scalar(int(self._values_))
        else:
            result = Scalar(self._values_.astype('int'))

        return result

    #===========================================================================
    def as_float(self):
        """A floating-point numeric version of this object.

        This method overrides the default behavior defined in the base class to
        return a Scalar of floats instead of a Boolean. True become one; False
        becomes zero.
        """

        if np.isscalar(self._values_):
            result = Scalar(float(self._values_))
        else:
            result = Scalar(self._values_.astype('float'))

        return result

    #===========================================================================
    def as_numeric(self):
        """A numeric version of this object.

        This method overrides the default behavior defined in the base class to
        return a Scalar of ints instead of a Boolean.
        """

        return self.as_int()

    #===========================================================================
    def as_index(self):
        """An object suitable for indexing a NumPy ndarray."""

        return (self._values_ & self.antimask)

    #===========================================================================
    def is_numeric(self):
        """True if this object is numeric; False otherwise.

        This method overrides the default behavior in the base class to return
        return False. Every other subclass is numeric.
        """

        return False

    #===========================================================================
    def sum(self, axis=None, value=True, out=None):
        """The number of items matching True or False.

        Input:
            axis        an integer axis or a tuple of axes. The sum is
                        determined across these axes, leaving any remaining axes
                        in the returned value. If None (the default), then the
                        sum is performed across all axes if the object.
            value       value to match.
            out         Ignored. Enables "np.sum(Boolean)" to work.
        """

        if value:
            return self.as_int().sum(axis=axis)
        else:
            return (Scalar.ONE - self.as_int()).sum(axis=axis)

    #===========================================================================
    def identity(self):
        """An object of this subclass equivalent to the identity."""

        return Boolean(True).as_readonly()

    #===========================================================================
    def logical_not(self):
        """The negation of this object."""

        return Boolean(np.logical_not(self._values_), self._mask_)

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
        Qube._raise_unsupported_op('+=', self)

    def __sub__(self, arg, recursive=True):
        return self.as_int() - arg

    def __rsub__(self, arg, recursive=True):
        return -self.as_int() + arg

    def __isub__(self, arg):
        Qube._raise_unsupported_op('-=', self)

    def __mul__(self, arg, recursive=True):
        return self.as_int() * arg

    def __rmul__(self, arg, recursive=True):
        return self.as_int() * arg

    def __imul__(self, arg):
        Qube._raise_unsupported_op('*=', self)

    def __truediv__(self, arg, recursive=True):
        return self.as_int() / arg

    def __rtruediv__(self, arg, recursive=True):
        if not isinstance(arg, Qube):
            arg = Scalar(arg)
        return arg / self.as_int()

    def __itruediv__(self, arg):
        Qube._raise_unsupported_op('/=', self)

    def __floordiv__(self, arg):
        return self.as_int() // arg

    def __rfloordiv__(self, arg):
        if not isinstance(arg, Qube):
            arg = Scalar(arg)
        return arg // self.as_int()

    def __ifloordiv__(self, arg):
        Qube._raise_unsupported_op('//=', self)

    def __mod__(self, arg):
        return self.as_int() % arg

    def __rmod__(self, arg):
        if not isinstance(arg, Qube):
            arg = Scalar(arg)
        return arg % self.as_int()

    def __imod__(self, arg):
        Qube._raise_unsupported_op('%=', self)

    def __pow__(self, arg):
        return self.as_int()**arg

################################################################################
# Useful class constants
################################################################################

Boolean.TRUE = Boolean(True).as_readonly()
Boolean.FALSE = Boolean(False).as_readonly()
Boolean.MASKED = Boolean(False,True).as_readonly()

################################################################################
# Once the load is complete, we can fill in a reference to the Boolean class
# inside the Qube object.
################################################################################

Qube.BOOLEAN_CLASS = Boolean

################################################################################
