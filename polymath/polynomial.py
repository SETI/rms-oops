################################################################################
# polymath/polynomial.py: Polynomial subclass of Vector
################################################################################

from __future__ import division
import numpy as np

from polymath.qube   import Qube
from polymath.scalar import Scalar
from polymath.vector import Vector
from polymath.units  import Units

class Polynomial(Vector):
    """This is a Vector subclass in which the elements are interpreted as the
    coefficients of a polynomial in a single variable x. Coefficients appear
    in order of decreasing exponent. Mathematical operations, polynomial
    root-solving are supported. Coefficients can have derivatives and these can
    be used to determine derivatives of the values or roots.
    """

    INTS_OK = False     # Only floating-point coefficients are allowed
    UNIT_OK = False     # Units are disallowed

    #===========================================================================
    def __init__(self, *args, **keywords):
        """Constructor for a Polynomial.

        If a single argument is a subclass of Vector, it is quickly converted to
        class Polynomial.

        Otherwise, the constructor takes the same inputs as the constructor for
        class Vector.
        """

        # For a subclass of Vector, transfer all attributes
        if (len(args) == 1 and len(keywords) == 0 and
            isinstance(args[0], Vector)):

                for (key, value) in args[0].__dict__.items():
                    self.__dict__[key] = value

                # Convert derivatives to class Polynomial if necessary
                if type(self) != Polynomial:
                    derivs = {}
                    for (key,value) in args[0].derivs.items():
                        derivs[key] = Polynomial(value)

                    self._derivs_ = derivs

        # Otherwise use the Vector class constructor
        else:
            super(Polynomial, self).__init__(*args, **keywords)

    #===========================================================================
    @property
    def order(self):
        """The order of the polynomial, i.e., the largest exponent."""

        return self.item[-self._drank_-1] - 1

    #===========================================================================
    @staticmethod
    def as_polynomial(arg, recursive=True):
        """The object converted to class Polynomial.

        Input:
            arg         object to convert to Polynomial.
            recursive   True to include derivatives in the conversion.
        """

        if isinstance(arg, Vector):
            if not recursive:
                arg = arg.wod

            return Polynomial(arg)

        vector = Vector.as_vector(arg)
        if recursive:
            return Polynomial(vector)
        else:
            return Polynomial(vector.wod)

    #===========================================================================
    def as_vector(self, recursive=True):
        """This object converted to class Vector.
        """

        obj = Qube.__new__(Vector)

        for (key, value) in self.__dict__.items():
            obj.__dict__[key] = value

        derivs = {}
        if recursive:
            for (key, value) in self._derivs_.items():
                derivs[key] = self.as_vector(recursive=False)

        obj.insert_derivs(derivs)
        return obj

    #===========================================================================
    def at_least_order(self, order, recursive=True):
        """A shallow copy of this object with at least this minimum order.
        Extra leading polynomial coefficients are filled with zeros.

        Input:
            order       minimum order of the Polynomial.
            recursive   True to include derivatives in the conversion.
        """

        if self.order >= order:
            if recursive:
                return self
            else:
                return self.wod

        new_values = np.zeros(self._shape_ + (order+1,))
        new_values[...,-self.order-1:] = self._values_

        result = Polynomial(new_values, self._mask_, derivs={}, example=self)

        if recursive and self._derivs_:
            for (key, value) in self._derivs_.items():
                result.insert_deriv(key, value.at_least_order(order,
                                                              recursive=False))

        return result

    #===========================================================================
    def set_order(self, order, recursive=True):
        """This Polynomial expressed with exactly this order. Extra polynomial
        coefficients are filled with zeros. If this Polynomial exceeds this
        order requested, raise an exception.

        Input:
            order       minimum number of the Polynomial.
            recursive   True to include derivatives in the conversion.
        """

        if self.order > order:
            raise ValueError('Polynomial of order %d ' % self.order +
                             'exceeds intended order %d' % order)

        return self.at_least_order(order, recursive=recursive)

    #===========================================================================
    def invert_line(self, recursive=True):
        """The inversion of this linear polynomial."""

        if self.order != 1:
            raise ValueError('invert_line requires a first-order polynomial')

        # y = a x + b
        # y - b = a x
        # y/a - b/a = x

        (a,b) = self.to_scalars(recursive=recursive)

        a_inv = 1./a
        return Polynomial(Vector.from_scalars(a_inv, -b * a_inv),
                          recursive=recursive)

    ############################################################################
    # Math operations
    ############################################################################

    def __neg__(self):
        return Polynomial(-self.as_vector())

    def __add__(self, arg):
        arg  = Polynomial.as_polynomial(arg ).at_least_order(self.order)
        self = Polynomial.as_polynomial(self).at_least_order(arg.order)
        return Polynomial(self.as_vector() + arg.as_vector())

    def __radd__(self, arg):
        return self.__add__(arg)

    def __iadd__(self, arg):
        arg = Polynomial.as_polynomial(arg).set_order(self.order)
        super(Polynomial,self).__iadd__(arg.as_vector())
        return Polynomial(self)

    def __sub__(self, arg):
        arg  = Polynomial.as_polynomial(arg ).at_least_order(self.order)
        self = Polynomial.as_polynomial(self).at_least_order(arg.order)
        return Polynomial(self.as_vector() - arg.as_vector())

    def __rsub__(self, arg):
        arg  = Polynomial.as_polynomial(arg ).at_least_order(self.order)
        self = Polynomial.as_polynomial(self).at_least_order(arg.order)
        return Polynomial(arg.as_vector() - self.as_vector())

    def __isub__(self, arg):
        arg = Polynomial.as_polynomial(arg).set_order(self.order)
        super(Polynomial,self).__isub__(arg.as_vector())
        return Polynomial(self)

    def __mul__(self, arg):

        # Support for Polynomial multiplication
        if isinstance(arg, Polynomial):
            if self._drank_ != arg._drank_:
                raise ValueError('incompatible denominators for multiply')

            new_order = self.order + arg.order
            new_shape = Qube.broadcasted_shape(self._shape_, arg._shape_)
            new_values = np.zeros(new_shape + (new_order+1,))
            new_mask = Qube.or_(self._mask_, arg._mask_)

            # It's simpler to work in order of increasing powers
            tail_indx = self._drank_ * (slice(None),)
            indx = (Ellipsis, slice(None,None,-1)) + tail_indx
            self_values = self._values_[indx]
            arg_values  = arg._values_[indx]

            # Perform the multiplication
            kstop = arg._values_.shape[-self._drank_-1]
            dk    = self._values_.shape[-self._drank_-1]
            for k in range(kstop):
                new_indx = (Ellipsis, slice(k,k+dk)) + tail_indx
                arg_indx = (Ellipsis, slice(k,k+1))  + tail_indx
                new_values[new_indx] += arg_values[arg_indx] * self_values

            result = Polynomial(new_values[indx], new_mask, derivs={},
                                units=Units.mul_units(self._units_,arg._units_))

            # Deal with derivatives
            derivs = {}
            for (key, value) in self._derivs_.items():
                derivs[key] = arg.wod * value

            for (key, value) in arg._derivs_.items():
                if key in derivs:
                    derivs[key] = derivs[key] + self.wod * value
                else:
                    derivs[key] = self.wod * value

            result.insert_derivs(derivs)

            return result

        return Polynomial(self.as_vector() * arg)

    def __rmul__(self, arg):
        return self.__mul__(arg)

    def __imul__(self, arg):

        # Multiplying by a zero-order Polynomial is valid
        if isinstance(arg, Vector) and arg.item == (1,):
            arg = arg.to_scalar(0)

        super(Polynomial,self).__imul__(arg)
        return Polynomial(self)

    def __truediv__(self, arg):

        # Dividing by a zero-order Polynomial is valid
        if isinstance(arg, Vector) and arg.item == (1,):
            arg = arg.to_scalar(0)

        return Polynomial(self.as_vector() / arg)

    def __itruediv__(self, arg):

        # Dividing by a zero-order Polynomial is valid
        if isinstance(arg, Vector) and arg.item == (1,):
            arg = arg.to_scalar(0)

        super(Polynomial,self).__itruediv__(arg)
        return Polynomial(self)

    def __pow__(self, arg):
        if arg < 0 or arg != int(arg):
            raise ValueError('Polynomial exponents must be non-negative '
                             'integers')

        if arg == 0:
            return Polynomial([1.])

        if arg == 1:
            return self

        result = self * self
        for k in range(2,arg):
            result = result * self

        return result

    def __eq__(self, arg):
        arg  = Polynomial.as_polynomial(arg ).at_least_order(self.order)
        self = Polynomial.as_polynomial(self).at_least_order(arg.order)
        return arg.as_vector() == self.as_vector()

    def __ne__(self, arg):
        arg  = Polynomial.as_polynomial(arg ).at_least_order(self.order)
        self = Polynomial.as_polynomial(self).at_least_order(arg.order)
        return arg.as_vector() != self.as_vector()

    ############################################################################
    # Special Polynomial operations
    ############################################################################

    def deriv(self, recursive=True):
        """The first derivative of this Polynomial."""

        if self.order <= 0:
            new_values = np.zeros(self._values_.shape)
        else:
           indx1 = (Ellipsis, slice(0,-1)) + self._drank_ * (slice(None,None),)
           indx2 = (Ellipsis,)             + self._drank_ * (np.newaxis,)
           new_values = self._values_[indx1] * np.arange(self.order,0,-1)[indx2]

        result = Polynomial(new_values, self._mask_, derivs={}, example=self)

        if recursive and self._derivs_:
            for (key,value) in self._derivs_.items():
                result.insert_deriv(key, value.deriv(recursive=False))

        return result

    #===========================================================================
    def eval(self, x, recursive=True):
        """Evaluate the polynomial at x.

        Inputs:
            x           Scalar at which to evaluate the Polynomial.
            recursive   True to evaluate derivatives as well.

        Return:         A Scalar of values. Note that the shapes of self and x
                        are broadcasted together.
        """

        if self.order == 0:
            if recursive:
                return Scalar(example=self)
            else:
                return Scalar(example=self.wod)

        x = Scalar.as_scalar(x, recursive=recursive)
        x_powers = [1., x]
        x_power = x
        for k in range(1,self.order):
            x_power *= x
            x_powers.append(x_power)

        x_powers = Vector.from_scalars(*(x_powers[::-1]))

        return Qube.dot(self, x_powers, 0, 0, (Scalar,), recursive=recursive)

    #===========================================================================
    def roots(self, recursive=True):
        """Find the roots of the polynomial.

        Inputs:
            recursive   True to evaluate derivatives at the roots as well.

        Return:         A Scalar of roots. This has the same shape as self but
                        an extra leading axis matching the order of the
                        polynomial. The leading index selects among the roots of
                        the polynomial. Roots appear in increasing order and
                        without any duplicates. If fewer real roots exist, the
                        set of roots is padded at the end with masked values.
        """

        # Constant case is easy
        if self.order == 0:
            # a = 0
            raise ValueError('no roots of a order-zero Polynomial')

        # Linear case is easy
        if self.order == 1:
            # a x + b = 0
            (a,b) = self.to_scalars(recursive=recursive)
            result = -b/a
            return result.reshape((1,) + result._shape_)

        # Quadratic case is easy
        if self.order == 2:
            # a x^2 + b x + c = 0
            (a,b,c) = self.to_scalars(recursive=recursive)
            (x0,x1) = Scalar.solve_quadratic(a, b, c, recursive=recursive)
            x1 = x1.mask_where(x1 == x0)        # mask duplicated solutions
            return Qube.stack(x0,x1).sort(axis=0)

        # Method for higher-order polynomials stolen from np.roots; see:
        #    https://github.com/numpy/numpy
        #               /blob/v1.14.0/numpy/lib/polynomial.py#L153-L235
        #     p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]

        # Copy polynomial coefficients
        coefficients = self._values_.copy()

        # Convert the mask to an array
        if np.isscalar(self._mask_):
            if self._mask_:
                poly_mask = np.ones(self._shape_, dtype='bool')
            else:
                poly_mask = np.zeros(self._shape_, dtype='bool')
        else:
            poly_mask = self._mask_.copy()

    # Method stolen from np.roots; see
    # https://github.com/numpy/numpy/blob/v1.14.0/numpy/lib/polynomial.py#L153-L235
    #     p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]

        # Mask out any cases where all coefficients are zero
        all_zeros = np.all(coefficients == 0., axis=-1)
        if np.any(all_zeros):

            # Problem is now 1 * x**n = 0 so solution is no longer undefined
            coefficients[all_zeros,0] = 1.
            poly_mask |= all_zeros

#     N = len(p)
#     if N > 1:
#         # build companion matrix and find its eigenvalues (the roots)
#         A = diag(np.ones((N-2,), p.dtype), -1)
#         A[0,:] = -p[1:] / p[0]
#         roots = np.linalg.eigvals(A)
#     else:
#         roots = np.array([])

        # Shift coefficients till the leading coefficient is nonzero
        shifts = (coefficients[...,0] == 0.)
        total_shifts = np.zeros(shifts._shape_, dtype='int')
        while np.any(shifts):
            coefficients[shifts,:-1] = coefficients[shifts,1:]
            coefficients[shifts,-1] = 0.
            total_shifts += shifts
            shifts = (coefficients[...,0] == 0.)

        # Implement the NumPy solution, array-based
        matrix = np.empty(self._shape_ + (self.order,self.order))
        matrix[...,:,:] = np.diag(np.ones((self.order-1,)), -1)
        matrix[...,0,:] = -coefficients[...,1:] / coefficients[...,0:1]
        roots = np.linalg.eigvals(matrix)
        roots = np.rollaxis(roots,-1,0)

        # Convert the roots to a real Scalar
        is_complex = np.imag(roots) != 0.
        root_values = np.real(roots)
        root_mask = poly_mask[np.newaxis,...] | is_complex

        # Mask extraneous zeros
        # Handily, they always show up first in the array of roots
        max_shifts = total_shifts.max()
        for k in range(max_shifts):
            root_mask[total_shifts > k, k] = True

        roots = Scalar(root_values, Qube.as_one_bool(root_mask))
        roots = roots.sort(axis=0)

        # Mask duplicated values
        mask_changed = False
        for k in range(1,self.order):
            mask = ((roots._values_[k,...] == roots._values_[k-1,...]) &
                     ~roots._mask_[k,...])
            if np.any(mask):
                root_mask[k,...] |= mask
                mask_changed = True

        if mask_changed:
            roots = Scalar(root_values, Qube.as_one_bool(root_mask))
            roots = roots.sort(axis=0)

        # Deal with derivatives if necessary
        #
        # Sum_j c[j] x**j = 0
        #
        # Sum_j dc[j]/dt x**j + Sum_j c[j] j x**(j-1) dx/dt = 0
        #
        # dx/dt = -Sum_j dc[j]/dt x**j / Sum_j c[j] j x**(j-1)

        if recursive:
            for (key, value) in self._derivs_.items():
                deriv = (-value.eval(roots, recursive=False) /
                         self.deriv.eval(roots, recursive=False))
                roots.insert_deriv(key, deriv)

        return roots

################################################################################
