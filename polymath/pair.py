################################################################################
# polymath/modules/pair.py: Pair subclass of PolyMath Vector
#
# Mark Showalter, PDS Rings Node, SETI Institute, February 2014
################################################################################

from __future__ import division
import numpy as np
import numbers

from qube   import Qube
from scalar import Scalar
from vector import Vector
from units  import Units

class Pair(Vector):
    """A PolyMath subclass containing coordinate pairs or 2-vectors.
    """

    NRANK = 1           # the number of numerator axes.
    NUMER = (2,)        # shape of the numerator.

    FLOATS_OK = True    # True to allow floating-point numbers.
    INTS_OK = True      # True to allow integers.
    BOOLS_OK = False    # True to allow booleans.

    UNITS_OK = True     # True to allow units; False to disallow them.
    MASKS_OK = True     # True to allow masks; False to disallow them.
    DERIVS_OK = True    # True to disallow derivatives; False to allow them.

    @staticmethod
    def as_pair(arg, recursive=True):
        """Return the argument converted to Pair if possible.

        If recursive is True, derivatives will also be converted.

        As a special case as_pair() of a single value returns a Pair with the
        value repeated.
        """

        if type(arg) == Pair:
            if recursive: return arg
            return arg.without_derivs()

        if isinstance(arg, Qube):

            # Collapse a 1x2 or 2x1 Matrix down to a Pair
            if arg.numer in ((1,2), (2,1)):
                return arg.flatten_numer(Pair, recursive)

            # For any suitable Qube, move numerator items to the denominator
            if arg.rank > 1 and arg.numer[0] == 2:
                arg = arg.split_items(1, Pair)

            arg = Pair(arg, example=arg)
            if recursive: return arg
            return arg.without_derivs()

        # Special case of a single number
        if isinstance(arg, numbers.Number):
            return Pair((arg,arg))

        return Pair(arg)

    @staticmethod
    def from_scalars(x, y, recursive=True):
        """A Pair constructed by combining two scalars.

        Inputs:
            args        any number of Scalars or arguments that can be casted
                        to Scalars. They need not have the same shape, but it
                        must be possible to cast them to the same shape. A value
                        of None is converted to a zero-valued Scalar that
                        matches the denominator shape of the other arguments.

            recursive   True to include all the derivatives. The returned object
                        will have derivatives representing the union of all the
                        derivatives found amongst the scalars. Default is True.
        """

        return Vector.from_scalars(x, y, recursive=recursive, classes=[Pair])

    def swapxy(self, recursive=True):
        """A pair object in which the first and second values are switched.

        If recursive is True, derivatives will also be swapped.
        """

        # Roll the array axis to the end
        lshape = len(self.values.shape)
        new_values = np.rollaxis(self.values, lshape - self.drank - 1, lshape)

        # Swap the axes
        new_values = new_values[..., ::-1]

        # Roll the axis back
        new_values = np.rollaxis(new_values, -1, lshape - self.drank - 1)

        # Construct the object
        obj = Pair(new_values, derivs={}, example=self)
        if self.readonly: obj.as_readonly()

        # Fill in the derivatives if necessary
        if recursive:
            for (key, deriv) in self.derivs.iteritems():
                obj.insert_deriv(key, deriv.swapxy(False))

        return obj

# A useful class constant

Pair.ZERO   = Pair((0.,0.)).as_readonly()
Pair.ONES   = Pair((1.,1.)).as_readonly()
Pair.XAXIS  = Pair((1.,0.)).as_readonly()
Pair.YAXIS  = Pair((0.,1.)).as_readonly()
Pair.MASKED = Pair((1,1), True).as_readonly()

Pair.IDENTITY = Pair([(1.,0.),(0.,1.)], drank=1).as_readonly()

################################################################################
# Once defined, register with Qube class
################################################################################

Qube.PAIR_CLASS = Pair

################################################################################
