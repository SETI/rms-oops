################################################################################
# polymath/modules/boolean.py: Boolean subclass of PolyMath base class
#
# Mark Showalter, PDS Rings Node, SETI Institute, February 2014
################################################################################

from __future__ import division
import numpy as np

from qube   import Qube
from scalar import Scalar

class Boolean(Qube):
    """A PolyMath subclass involving dimensionless booleans.

    Masks, units and derivatives are disallowed."""

    NRANK = 0           # the number of numerator axes.
    NUMER = ()          # shape of the numerator.

    FLOATS_OK = False   # True to allow floating-point numbers.
    INTS_OK = False     # True to allow integers.
    BOOLS_OK = True     # True to allow booleans.

    UNITS_OK = False    # True to allow units; False to disallow them.
    MASKS_OK = False    # True to allow masks; False to disallow them.
    DERIVS_OK = False   # True to disallow derivatives; False to allow them.

    def __init__(self, arg=None, mask=None, units=None, derivs={},
                       nrank=None, drank=None, example=None):
        """Default constructor; True where nonzero and unmasked."""

        original_arg = arg

        # Interpret the example
        if example is not None:
            if mask is None: mask = example.mask

            if arg is None:
                arg = (example.values != 0)
                for r in range(example.rank):
                    arg = np.any(arg, axis=-1)

        # Interpret the arg if it is a PolyMath object
        if isinstance(arg, Qube):
            if mask is None or mask is False:
                mask = arg.mask
            else:
                mask = arg.mask | mask

            rank = arg.rank
            arg = (arg.values != 0)
            for r in range(rank):
                arg = np.any(arg, axis=-1)

        # Interpret the arg if it is a NumPy MaskedArray
        if isinstance(arg, np.ma.MaskedArray):
            if arg.mask is not np.ma.nomask:
                if mask is None or mask is False:
                    mask = arg.mask
                else:
                    mask = arg.mask | mask

            arg = (arg.data != 0)

        # Convert a list or tuple to a NumPy ndarray
        if type(arg) in (list,tuple):
            arg = np.asarray(arg)
            arg = (arg != 0)

        # Value is False where mask is True
        if mask is None:
            mask = False

        if np.shape(arg) == ():
            if mask:
                arg = False
            else:
                arg = bool(arg)
        elif np.shape(mask) == ():
            if mask:
                if arg is original_arg:
                    arg = arg.copy()

                arg.fill(False)
        else:
            if arg is original_arg:
                arg = arg.copy()

            arg[mask] = False

        Qube.__init__(self, arg, mask=False, units=units, derivs=derivs,
                            nrank=0, drank=0, example=None)

    @staticmethod
    def as_boolean(arg, recursive=True):
        """Return the argument converted to Boolean if possible."""

        if type(arg) == Boolean: return arg

        return Boolean(arg)

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

        if self.readonly:
            result = result.as_readonly(nocopy='vm')

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

        if self.readonly:
            result = result.as_readonly(nocopy='vm')

        return result

    def as_numeric(self):
        """Return a numeric version of this object.

        This method overrides the default behavior defined in the base class to
        return a Scalar of ints instead of a Boolean.
        """

        return self.as_int()

    def as_index(self, remove_masked=False):
        """Return an object suitable for indexing a NumPy ndarray plus a mask."""

        return self.values, None

    def is_numeric(self):
        """Return True if this object is numeric; False otherwise.

        This method overrides the default behavior in the base class to return
        return False. Every other subclass is numeric.
        """

        return False

    def sum(self, value=True):
        """Return the number of items matching True or False."""

        if value:
            return self.as_int().sum()
        else:
            return self.size - self.as_int().sum()

    def masked_single(self):
        """Return an object of this subclass containing one masked value."""

        raise TypeError("class 'Boolean' does not support masking")

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
        if Qube.is_empty(arg): return arg
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

    ############################################################################
    # Logical operators
    ############################################################################

    # (~) operator
    def __invert__(self):

        obj = Boolean(~self.values)
        if self.readonly: obj = obj.as_readonly(nocopy='vm')

        return obj

    # (&) operator
    def __and__(self, arg):
        if Qube.is_empty(arg): return arg

        obj = Boolean(self.values & Boolean.as_boolean(arg).values)
        if self.readonly and arg.readonly: obj = obj.as_readonly(nocopy='vm')

        return obj

    # (|) operator
    def __or__(self, arg):
        if Qube.is_empty(arg): return arg

        obj = Boolean(self.values | Boolean.as_boolean(arg).values)
        if self.readonly and arg.readonly: obj = obj.as_readonly(nocopy='vm')

        return obj

    # (^) operator
    def __xor__(self, arg):
        if Qube.is_empty(arg): return arg

        obj = Boolean(self.values ^ Boolean.as_boolean(arg).values)
        if self.readonly and arg.readonly: obj = obj.as_readonly(nocopy='vm')

        return obj

    # (&=) operator
    def __iand__(self, arg):
        if Qube.is_empty(arg): return arg

        self.require_writable()
        self._Qube__values_ &= Boolean.as_boolean(arg).values
        return self

    # (|=) operator
    def __ior__(self, arg):
        if Qube.is_empty(arg): return arg

        self.require_writable()
        self._Qube__values_ |= Boolean.as_boolean(arg).values
        return self

    # (^=) operator
    def __ixor__(self, arg):
        if Qube.is_empty(arg): return arg

        self.require_writable()
        self._Qube__values_ ^= Boolean.as_boolean(arg).values
        return self

# Useful class constants

Boolean.TRUE = Boolean(True).as_readonly()
Boolean.FALSE = Boolean(False).as_readonly()

################################################################################
# Once the load is complete, we can fill in a reference to the Boolean class
# inside the Qube object.
################################################################################

Qube.BOOLEAN_CLASS = Boolean

################################################################################
