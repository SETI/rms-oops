################################################################################
# polymath/scalar.py: Scalar subclass of PolyMath base class
################################################################################

from __future__ import division
import numpy as np
import numbers
import sys
import warnings

from polymath.qube  import Qube
from polymath.units import Units

# Maximum argument to exp()
EXP_CUTOFF = np.log(sys.float_info.max)
TWOPI = np.pi * 2.

class Scalar(Qube):
    """A PolyMath subclass involving dimensionless scalars."""

    NRANK = 0           # the number of numerator axes.
    NUMER = ()          # shape of the numerator.

    FLOATS_OK = True    # True to allow floating-point numbers.
    INTS_OK = True      # True to allow integers.
    BOOLS_OK = False    # True to allow booleans.

    UNITS_OK = True     # True to allow units; False to disallow them.
    DERIVS_OK = True    # True to allow derivatives and to allow this class to
                        # have a denominator; False to disallow them.

    DEFAULT_VALUE = 1

    #===========================================================================
    def _maxval(self):
        """Internal method returns the maximum value associated with a dtype."""

        dtype = self._values_.dtype
        if dtype.kind == 'f':
            return np.inf
        elif dtype.kind == 'u':
            return 256**dtype.itemsize - 1
        elif dtype.kind == 'i':
            return 256**dtype.itemsize//2 - 1
        elif dtype.kind == 'b':
            return 1
        else:
            raise ValueError('invalid dtype %s' % str(dtype))

    #===========================================================================
    def _minval(self):
        """Internal method returns the minimum value associated with a dtype."""

        # Define constant maxval
        dtype = self._values_.dtype
        if dtype.kind == 'f':
            return -np.inf
        elif dtype.kind == 'u':
            return 0
        elif dtype.kind == 'i':
            return -256**dtype.itemsize//2
        elif dtype.kind == 'b':
            return 0
        else:
            raise ValueError('invalid dtype %s' % str(dtype))

    #===========================================================================
    @staticmethod
    def as_scalar(arg, recursive=True):
        """The argument converted to Scalar if possible.

        If recursive is True, derivatives will also be converted.
        """

        if isinstance(arg, Scalar):
            if arg.is_bool():
                return arg.as_int()
            if recursive:
                return arg
            return arg.wod

        if isinstance(arg, Qube):
            if type(arg) == Qube.BOOLEAN_CLASS:
                return Qube.BOOLEAN_CLASS(arg).as_int()

            arg = Scalar(arg)
            if recursive:
                return arg
            return arg.wod

        if isinstance(arg, Units):
            return Scalar(arg.from_units_factor, units=arg)

        return Scalar(arg)

    #===========================================================================
    def to_scalar(self, indx, recursive=True):
        """Duplicates the behavior of Vector.to_scalar, in this case returning
        self.

        Input:
            indx        index of the vector component; must be zero
            recursive   True to include the derivatives.
        """

        if indx != 0:
            raise ValueError('Scalar.to_scalar() index out of range')

        if recursive:
            return self

        return self.wod

    #===========================================================================
    def as_index(self, masked=None):
        """This object made suitable for indexing an N-dimensional NumPy array.

        Input:
            masked      the value to insert in the place of a masked item. If
                        None and the object contains masked elements, the array
                        will be flattened and masked elements will be skipped.
        """

        (index, mask) = self.as_index_and_mask((masked is None), masked)
        return index

    #===========================================================================
    def as_index_and_mask(self, purge=False, masked=None):
        """This object made suitable for indexing and masking an N-dimensional
        array.

        Input:
            purge           True to eliminate masked elements from the index;
                            False to retain them but leave them masked.
            masked          the index value to insert in place of any masked.
                            item. This may be needed because each value in the
                            returned index array must be an integer and in
                            range. If None (the default), then masked values
                            in the index will retain their unmasked values when
                            the index is applied.
        """

        if self.is_float():
            raise IndexError('floating-point indexing is not permitted')

        if (self._drank_ > 0):
            raise ValueError('an indexing object cannot have a denominator')

        # If nothing is masked, this is easy
        if not np.any(self._mask_):
            if np.shape(self._values_):
                return (self._values_.astype(np.intp), False)
            else:
                return (int(self._values_), False)

        # If purging...
        if purge:

            # If all masked...
            if Qube.is_one_true(self._mask_):
                return ((), False)

            # If partially masked...
            return (self._values_[self.antimask].astype(np.intp), False)

        # Without a replacement...
        if masked is None:
            new_values = self._values_.astype(np.intp)

        # If all masked...
        elif Qube.is_one_true(self._mask_):
            new_values = np.empty(self._values_.shape, dtype=np.intp)
            new_values[...] = masked

        # If partially masked...
        else:
            new_values = self._values_.copy().astype(np.intp)
            new_values[self._mask_] = masked

        return (new_values, self._mask_)

    #===========================================================================
    def int(self, top=None, remask=False, clip=False, inclusive=True,
                  shift=None, builtins=None, masked=None):
        """An integer (floor) version of this Scalar.

        If this object already contains integers, it is returned as is.
        Otherwise, a copy is returned. Derivatives are always removed. Units
        are disallowed.

        Inputs:
            top         Optional nominal maximum integer value.
            remask      If True, values less than zero or greater than the
                        specified top value (if provided) are masked.
            clip        If True, values less than zero or greater than the
                        specified top value are clipped.
            inclusive   True to leave the top value unmasked; False to mask it.
            shift       True to shift any occurrences of the top value down by
                        one; False to leave them unchanged. Default None lets
                        shift match the input value of inclusive.
            builtins    If True and the result is a single unmasked scalar,
                        the result is returned as a Python int instead of an
                        instance of Scalar. Default is the value specified
                        by Qube.PREFER_BUILTIN_TYPES.
            masked      value to return if builtins is True but the returned
                        value is masked. Default is to return a masked value
                        instead of a builtin type.
        """

        if self._drank_:
            raise ValueError('Scalar.int() does not support denominators')

        Units.require_unitless(self._units_)

        # For compatibility with Vector.int, where the first arg is the shape
        if isinstance(top, (list, tuple)):
            top = top[0]

        if top is not None:

            # Convert to int; be sure it's a copy before modifying
            if self.is_int():
                copied = self.copy()
                values = copied._values_
                mask = copied._mask_
            else:
                values = self.wod.as_int()._values_
                mask = self._mask_
                if not np.isscalar(mask):
                    mask = mask.copy()

            if shift is None:
                shift = inclusive

            if shift and not clip:      # shift is unneeded if clip is True
                if self._shape_:
                    values[self._values_ == top] -= 1
                elif self._values_ == top:
                    values = top - 1

            if remask:
                is_outside = Scalar.is_outside(self._values_, 0, top, inclusive)

            if clip:
                values = np.clip(values, 0, top-1)

            if remask and np.any(is_outside):
                mask = Qube.or_(mask, is_outside)

            result = Scalar(values, mask, example=self)

        else:
            result = self.wod.as_int()
            if clip:
                result = result.mask_where_lt(0, replace=0, remask=remask)

            elif remask:
                result = result.mask_where_lt(0, remask=remask)

        # Convert result to a Python int if necessary
        if builtins is None:
            builtins = Qube.PREFER_BUILTIN_TYPES

        if builtins:
            return result.as_builtin(masked=masked)

        return result

    #===========================================================================
    def frac(self, recursive=True):
        """Return an object containing the fractional components of all values.

        The returned object is an instance of the same subclass as this object.

        Inputs:
            recursive   True to include the derivatives of the returned object.
                        frac() leaves the derivatives unchanged.
        """

        if self._drank_:
            raise ValueError('Scalar.frac() does not support denominators')

        Units.require_unitless(self._units_)

        # Convert to fractional values
        if isinstance(self._values_, np.ndarray):
            new_values = (self._values_ % 1.)
        else:
            new_values = self._values_ % 1.

        # Construct a new copy
        obj = Qube.__new__(type(self))
        obj.__init__(new_values, mask=self._mask_, derivs=self._derivs_)

        return obj

    #===========================================================================
    def abs(self, recursive=True):
        """Return the absolute value of each value.

        Inputs:
            recursive   True to include the derivatives of the absolute values
                        inside the returned object.
        """

        if recursive:
            return abs(self)
        else:
            return abs(self.wod)

    #===========================================================================
    def sin(self, recursive=True):
        """Return the sine of each value.

        Inputs:
            recursive   True to include the derivatives of the sine inside the
                        returned object.
        """

        if self._drank_:
            raise ValueError('Scalar.sin() does not support denominators')

        Units.require_angle(self._units_)
        obj = Scalar(np.sin(self._values_), mask=self._mask_)

        if recursive and self._derivs_:
            factor = self.wod.cos()
            for (key, deriv) in self._derivs_.items():
                obj.insert_deriv(key, factor * deriv)

        return obj

    #===========================================================================
    def cos(self, recursive=True):
        """Return the cosine of each value.

        Inputs:
            recursive   True to include the derivatives of the cosine inside the
                        returned object.
        """

        if self._drank_:
            raise ValueError('Scalar.cos() does not support denominators')

        Units.require_angle(self._units_)
        obj = Scalar(np.cos(self._values_), mask=self._mask_)

        if recursive and self._derivs_:
            factor = -self.wod.sin()
            for (key, deriv) in self._derivs_.items():
                obj.insert_deriv(key, factor * deriv)

        return obj

    #===========================================================================
    def tan(self, recursive=True):
        """Return the tangent of each value.

        Inputs:
            recursive   True to include the derivatives of the tangent inside
                        the returned object.
        """

        if self._drank_:
            raise ValueError('Scalar.tan() does not support denominators')

        Units.require_angle(self._units_)

        obj = Scalar(np.tan(self._values_), mask=self._mask_)

        if recursive and self._derivs_:
            inv_sec_sq = self.wod.cos()**(-2)
            for (key, deriv) in self._derivs_.items():
                obj.insert_deriv(key, inv_sec_sq * deriv)

        return obj

    #===========================================================================
    def arcsin(self, recursive=True, check=True):
        """Return the arcsine of each value.

        Inputs:
            recursive   True to include the derivatives of the arcsine inside
                        the returned object.
            check       True to mask out the locations of any values outside the
                        domain [-1,1]. If False, a ValueError will be raised if
                        any value is encountered where the arcsine is undefined.
                        Check=True is slightly faster if we already know at the
                        time of the call that all input values are valid.
        """

        if self._drank_:
            raise ValueError('Scalar.arcsin() does not support denominators')

        # Limit domain to [-1,1] if necessary
        if check:
            Units.require_unitless(self._units_)

            temp_mask = (self._values_ < -1) | (self._values_ > 1)
            if np.any(temp_mask):
                if Qube.is_one_true(temp_mask):
                    temp_values = 0.
                else:
                    temp_values = self._values_.copy()
                    temp_values[temp_mask] = 0.
                    temp_mask = Qube.or_(self._mask_, temp_mask)
            else:
                temp_values = self._values_
                temp_mask = self._mask_

            obj = Scalar(np.arcsin(temp_values), temp_mask)

        else:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    func_values = np.arcsin(self._values_)
                except RuntimeWarning:
                    raise ValueError('Scalar.arcsin() of value outside domain '
                                     '(-1,1)')

            obj = Scalar(func_values, mask=self._mask_)

        if recursive and self._derivs_:
            factor = (1. - self.wod**2)**(-0.5)
            for (key, deriv) in self._derivs_.items():
                obj.insert_deriv(key, factor * deriv)

        return obj

    #===========================================================================
    def arccos(self, recursive=True, check=True):
        """Return the arccosine of each value.

        Inputs:
            recursive   True to include the derivatives of the arccosine
                        inside the returned object.
            check       True to mask out the locations of any values outside the
                        domain [-1,1]. If False, a ValueError will be raised if
                        any value is encountered where the arccosine is
                        undefined. Check=True is slightly faster if we already
                        know at the time of the call that all input values are
                        valid.
        """

        if self._drank_:
            raise ValueError('Scalar.arccos() does not support denominators')

        # Limit domain to [-1,1] if necessary
        if check:
            Units.require_unitless(self._units_)

            temp_mask = (self._values_ < -1) | (self._values_ > 1)
            if np.any(temp_mask):
                if Qube.is_one_true(temp_mask):
                    temp_values = 0.
                else:
                    temp_values = self._values_.copy()
                    temp_values[temp_mask] = 0.
                    temp_mask = Qube.or_(self._mask_, temp_mask)
            else:
                temp_values = self._values_
                temp_mask = self._mask_

            obj = Scalar(np.arccos(temp_values), temp_mask)

        else:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    func_values = np.arccos(self._values_)
                except RuntimeWarning:
                    raise ValueError('Scalar.arccos() of value outside domain '
                                     '(-1,1)')

            obj = Scalar(func_values, mask=self._mask_)

        if recursive and self._derivs_:
            factor = -(1. - self.wod**2)**(-0.5)
            for (key, deriv) in self._derivs_.items():
                obj.insert_deriv(key, factor * deriv)

        return obj

    #===========================================================================
    def arctan(self, recursive=True):
        """Return the arctangent of each value.

        Inputs:
            recursive   True to include the derivatives of the arctangent
                        inside the returned object.
        """

        if self._drank_:
            raise ValueError('Scalar.arctan() does not support denominators')

        Units.require_unitless(self._units_)

        obj = Scalar(np.arctan(self._values_), mask=self._mask_)

        if recursive and self._derivs_:
            factor = 1. / (1. + self.wod**2)
            for (key, deriv) in self._derivs_.items():
                obj.insert_deriv(key, factor * deriv)

        return obj

    #===========================================================================
    def arctan2(self, arg, recursive=True):
        """Return the four-quadrant value of arctan2(y,x).

        If this object is read-only, the returned object will also be read-only.

        Inputs:
            arg         The second argument to arctan2().
            recursive   True to include the derivatives of the arctangent
                        inside the returned object. This is the result of
                        merging the derivatives in both this object and the
                        argument object.
        """

        y = self
        x = Scalar.as_scalar(arg)
        Units.require_compatible(y._units_, x._units_)

        if x._drank_ or y._drank_:
            raise ValueError('Scalar.arctan2() does not support denominators')

        obj = Scalar(np.arctan2(y._values_, x._values_),
                     Qube.or_(x._mask_, y._mask_))

        if recursive and (x._derivs_ or y._derivs_):
            denom_inv = (x.wod**2 + y.wod**2).reciprocal()

            new_derivs = {}
            for (key, y_deriv) in y._derivs_.items():
                new_derivs[key] = x.wod * denom_inv * y_deriv

            for (key, x_deriv) in x._derivs_.items():
                term = y.wod * denom_inv * x_deriv
                if key in new_derivs:
                    new_derivs[key] -= term
                else:
                    new_derivs[key] = -term

            obj.insert_derivs(new_derivs)

        return obj

    #===========================================================================
    def sqrt(self, recursive=True, check=True):
        """Return the square root, masking imaginary values.

        If this object is read-only, the returned object will also be read-only.

        Inputs:
            recursive   True to include the derivatives of the square root
                        inside the returned object.
            check       True to mask out the locations of any values < 0 before
                        taking the square root. If False, a ValueError will be
                        raised any negative value encountered. Check=True is
                        slightly faster if we already know at the time of the
                        call that all input values are valid.
        """

        if self._drank_:
            raise ValueError('Scalar.sqrt() does not support denominators')

        if check:
            no_negs = self.mask_where_lt(0., replace=1.)
            sqrt_vals = np.sqrt(no_negs._values_)

        else:
            no_negs = self
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    sqrt_vals = np.sqrt(no_negs._values_)
                except RuntimeWarning:
                    raise ValueError('Scalar.sqrt() of negative value')

        obj = Scalar(sqrt_vals, mask=no_negs._mask_,
                                units=Units.sqrt_units(no_negs._units_))

        if recursive and no_negs._derivs_:
            factor = 0.5 / obj
            for (key, deriv) in self._derivs_.items():
                obj.insert_deriv(key, factor * deriv)

        return obj

    #===========================================================================
    def log(self, recursive=True, check=True):
        """Return the natural log, masking undefined values.

        If this object is read-only, the returned object will also be read-only.

        Inputs:
            recursive   True to include the derivatives of the log inside the
                        returned object.
            check       True to mask out the locations of any values <= 0 before
                        taking the log. If False, a ValueError will be raised
                        any value <= 0 is encountered. Check=True is slightly
                        faster if we already know at the time of the call that
                        all input values are valid.
        """

        if self._drank_:
            raise ValueError('Scalar.log() does not support denominators')

        if check:
            no_negs = self.mask_where_le(0., replace=1.)
            log_values = np.log(no_negs._values_)
        else:
            no_negs = self
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    log_values = np.log(no_negs._values_)
                except RuntimeWarning:
                    raise ValueError('Scalar.log() of non-positive value')

        obj = Scalar(log_values, mask=no_negs._mask_)

        if recursive and no_negs._derivs_:
            for (key, deriv) in self._derivs_.items():
                obj.insert_deriv(key, deriv / no_negs)

        return obj

    #===========================================================================
    def exp(self, recursive=True, check=False):
        """Return e raised to the power of each value.

        If this object is read-only, the returned object will also be read-only.

        Inputs:
            recursive   True to include the derivatives of the function inside
                        the returned object.
            check       True to mask out the locations of any values that will
                        overflow to infinity. If False, a ValueError will be
                        raised any value overflows. Check=True is slightly
                        faster if we already know at the time of the call that
                        all input values are valid.
        """

        global EXP_CUTOFF

        if self._drank_:
            raise ValueError('Scalar.exp() does not support denominators')

        Units.require_angle(self._units_)

        if check:
            no_oflow = self.mask_where_gt(EXP_CUTOFF, replace=EXP_CUTOFF)
            exp_values = np.exp(no_oflow._values_)

        else:
            no_oflow = self
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    exp_values = np.exp(no_oflow._values_)
                except (ValueError, TypeError):
                    raise ValueError('Scalar.exp() overflow encountered')

        obj = Scalar(exp_values, mask=no_oflow._mask_)

        if recursive and self._derivs_:
            for (key, deriv) in self._derivs_.items():
                obj.insert_deriv(key, deriv * exp_values)

        return obj

    #===========================================================================
    def sign(self, zeros=True, builtins=None, masked=None):
        """The sign of each value as +1, -1 or 0.

        Inputs:
            zeros       If zeros is False, then only values of +1 and -1 are
                        returned; sign(0) = +1 instead of 0.
            builtins    If True and the result is a single unmasked scalar,
                        the result is returned as a Python int instead of an
                        instance of Scalar. Default is the value specified
                        by Qube.PREFER_BUILTIN_TYPES.
            masked      value to return if builtins is True but the returned
                        value is masked. Default is to return a masked value
                        instead of a builtin type.
        """

        result = Scalar(np.sign(self._values_), mask=self._mask_)

        if not zeros:
            result[result == 0] = 1

        # Convert result to a Python int if necessary
        if builtins is None:
            builtins = Qube.PREFER_BUILTIN_TYPES

        if builtins:
            return result.as_builtin(masked=masked)

        return result

    #===========================================================================
    @staticmethod
    def solve_quadratic(a, b, c, recursive=True, include_antimask=False):
        """Return a tuple containing the two results of a quadratic equation as
        Scalars. Duplicates and complex values are masked.

        The formula solved is:
            a * x**2 + b * x + c = 0

        The solution is implemented to provide maximal precision.

        If include_mask is True, a Boolean is also returned containing True
        where the solution exists (because the discriminant is nonnegative).
        """

        a          = Scalar.as_scalar(a, recursive=recursive)
        neg_half_b = Scalar.as_scalar(b, recursive=recursive) * (-0.5)
        c          = Scalar.as_scalar(c, recursive=recursive)

        discr = neg_half_b**2 - a*c

        term = neg_half_b + neg_half_b.sign(zeros=False) * discr.sqrt()
        x0 = c / term
        x1 = term / a

        a_zeros = (a == 0)
        if isinstance(a_zeros, (bool, np.bool_)):
            if a_zeros:
                x0 = c / (neg_half_b * 2)

        else:
            if a_zeros.any():
                linear_x0 = c / (neg_half_b * 2)
                linear_x0 = linear_x0.broadcast_into_shape(x0._shape_)
                a_zeros   = a_zeros.broadcast_into_shape(x0._shape_)
                x0[a_zeros] = linear_x0[a_zeros]

        if include_antimask:
            return (x0, x1, Qube.BOOLEAN_CLASS(discr._values_ >= 0,
                                               discr._mask_))
        else:
            return (x0, x1)

    #===========================================================================
    def eval_quadratic(self, a, b, c, recursive=True):
        """Evaluate a quadratic function for this Scalar.

        The value returned is:
            a * self**2 + b * self + c
        """

        if not recursive:
            self = self.wod
            a = Scalar.as_scalar(a, recursive=False)
            b = Scalar.as_scalar(b, recursive=False)
            c = Scalar.as_scalar(c, recursive=False)

        return self * (self * a + b) + c

    #===========================================================================
    def max(self, axis=None, builtins=None, masked=None, out=None):
        """The maximum of the unmasked values.

        Input:
            axis        an integer axis or a tuple of axes. The maximum is
                        determined across these axes, leaving any remaining axes
                        in the returned value. If None (the default), then the
                        maximum is performed across all axes if the object.
            builtins    If True and the result is a single unmasked scalar,
                        the result is returned as a Python int or float instead
                        of an instance of Scalar. Default is the value specified
                        by Qube.PREFER_BUILTIN_TYPES.
            masked      value to return if builtins is True but the returned
                        value is masked. Default is to return a masked value
                        instead of a builtin type.
            out         Ignored. Enables "np.max(Scalar)" to work.
        """

        if self._drank_:
            raise ValueError('Scalar.max() does not support denominators')

        self._check_axis(axis, 'max()')         # make sure axis input is valid

        if self._size_ == 0:
            return self.wod._zero_sized_result(axis)

        if self._shape_ == ():
            result = self.wod

        elif not np.any(self._mask_):
            result = Scalar(np.max(self._values_, axis=axis), mask=False,
                            example=self)

        # If all masked, use the unmasked values but leave the result masked
        elif np.all(self._mask_):
            result = Scalar(np.max(self._values_, axis=axis), mask=True,
                            example=self)

        else:
            # In this case, the values and mask are both arrays

            min_possible = self._minval()       # smallest possible value
            new_values = self._values_.copy()
            new_values[self._mask_] = min_possible
            max_values = np.max(new_values, axis=axis)

            # Deal with completely masked items. Here, use the max of the
            # unmasked values.
            mask = np.all(self._mask_, axis=axis)
            if np.any(mask):
                alt_values = np.max(self._values_, axis=axis)
                if np.shape(mask):
                    max_values[mask] = alt_values[mask]
                else:
                    max_values = alt_values
            else:
                mask = False

            result = Scalar(max_values, mask, example=self)

        # Convert result to a Python type if necessary
        if builtins is None:
            builtins = Qube.PREFER_BUILTIN_TYPES

        if builtins:
            return result.as_builtin(masked=masked)

        return result

    #===========================================================================
    def min(self, axis=None, builtins=None, masked=None, out=None):
        """The minimum of the unmasked values.

        Input:
            axis        an integer axis or a tuple of axes. The minimum is
                        determined across these axes, leaving any remaining axes
                        in the returned value. If None (the default), then the
                        minimum is performed across all axes if the object.
            builtins    If True and the result is a single unmasked scalar,
                        the result is returned as a Python int or float instead
                        of an instance of Scalar. Default is the value specified
                        by Qube.PREFER_BUILTIN_TYPES.
            masked      value to return if builtins is True but the returned
                        value is masked. Default is to return a masked value
                        instead of a builtin type.
            out         Ignored. Enables "np.min(Scalar)" to work.
        """

        if self._drank_:
            raise ValueError('Scalar.min() does not support denominators')

        self._check_axis(axis, 'min()')         # make sure axis input is valid

        if self._size_ == 0:
            return self.wod._zero_sized_result(axis)

        if self._shape_ == ():
            result = self.wod

        elif not np.any(self._mask_):
            result = Scalar(np.min(self._values_, axis=axis), mask=False,
                            example=self)

        # If all masked, use the unmasked values but leave the result masked
        elif np.all(self._mask_):
            result = Scalar(np.min(self._values_, axis=axis), mask=True,
                            example=self)

        else:
            # In this case, the values and mask are both arrays

            max_possible = self._maxval()       # largest possible value
            new_values = self._values_.copy()
            new_values[self._mask_] = max_possible
            min_values = np.min(new_values, axis=axis)

            # Deal with completely masked items. Here, use the min of the
            # unmasked values.
            mask = np.all(self._mask_, axis=axis)
            if np.any(mask):
                alt_values = np.min(self._values_, axis=axis)
                if np.shape(mask):
                    min_values[mask] = alt_values[mask]
                else:
                    min_values = alt_values
                    mask = True
            else:
                mask = False

            result = Scalar(min_values, mask, example=self)

        # Convert result to a Python type if necessary
        if builtins is None:
            builtins = Qube.PREFER_BUILTIN_TYPES

        if builtins:
            return result.as_builtin(masked=masked)

        return result

    #===========================================================================
    def argmax(self, axis=None, builtins=None, masked=None):
        """The index of the maximum of the unmasked values along the specified
        axis.

        This returns an integer Scalar array of the same shape as self, except
        that the specified axis has been removed. Each value indicates the
        index of the maximum along that axis. The index is masked where the
        values along the axis are all masked.

        If axis is None, then it returns the index of the maximum argument after
        flattening the array.

        Input:
            axis        an optional integer axis. If None, it returns the index
                        of the maximum argument in the flattened the array.
            builtins    If True and the result is a single unmasked scalar,
                        the result is returned as a Python int instead of an
                        instance of Scalar. Default is the value specified
                        by Qube.PREFER_BUILTIN_TYPES.
            masked      value to return if builtins is True but the returned
                        value is masked. Default is to return a masked value
                        instead of a builtin type.
        """

        if self._drank_:
            raise ValueError('Scalar.argmax() does not support denominators')

        self._check_axis(axis, 'argmax()')      # make sure axis input is valid

        if self._shape_ == ():
            raise ValueError('no Scalar.argmax() for object with shape ()')

        if self._size_ == 0:
            ints = self.zeros(self.shape, dtype='int')
            return ints._zero_sized_result(axis)

        if not np.any(self._mask_):
            result = Scalar(np.argmax(self._values_, axis=axis), mask=False)

        # If all masked, use the argmax values but leave the result masked
        elif np.all(self._mask_):
            result = Scalar(np.argmax(self._values_, axis=axis), mask=True)

        # In this case, the values and mask are both arrays

        else:
            min_possible = self._minval()
            new_values = self._values_.copy()
            new_values[self._mask_] = min_possible
            argmax = np.argmax(new_values, axis=axis)

            # Deal with completely masked items. Here, use the argmax of the
            # unmasked values.
            mask = np.all(self._mask_, axis=axis)
            if np.any(mask):
                alt_argmax = np.argmax(self._values_, axis=axis)
                if np.shape(mask):
                    argmax[mask] = alt_argmax[mask]
                else:
                    argmax = alt_argmax
                    mask = True
            else:
                mask = False

            result = Scalar(argmax, mask)

        # Convert result to a Python type if necessary
        if builtins is None:
            builtins = Qube.PREFER_BUILTIN_TYPES

        if builtins:
            return result.as_builtin(masked=masked)

        return result

    #===========================================================================
    def argmin(self, axis=None, builtins=None, masked=None):
        """The index of the minimum of the unmasked values along the specified
        axis.

        This returns an integer Scalar array of the same shape as self, except
        that the specified axis has been removed. Each value indicates the
        index of the minimum along that axis. The index is masked where the
        values along the axis are all masked.

        Input:
            axis        an optional integer axis. If None, it returns the index
                        of the maximum argument in the flattened the array.
            builtins    If True and the result is a single unmasked scalar,
                        the result is returned as a Python int instead of an
                        instance of Scalar. Default is the value specified
                        by Qube.PREFER_BUILTIN_TYPES.
            masked      value to return if builtins is True but the returned
                        value is masked. Default is to return a masked value
                        instead of a builtin type.
        """

        if self._drank_:
            raise ValueError('Scalar.argmin() does not support denominators')

        self._check_axis(axis, 'argmin()')      # make sure axis input is valid

        if self._shape_ == ():
            raise ValueError('no Scalar.argmin() for object with shape ()')

        if self._size_ == 0:
            ints = self.zeros(self.shape, dtype='int')
            return ints._zero_sized_result(axis)

        if not np.any(self._mask_):
            result = Scalar(np.argmin(self._values_, axis=axis), mask=False)

        # If all masked, use the argmin values but leave the result masked
        elif np.all(self._mask_):
            result = Scalar(np.argmin(self._values_, axis=axis), mask=True)

        # In this case, the values and mask are both arrays

        else:
            max_possible = self._maxval()
            new_values = self._values_.copy()
            new_values[self._mask_] = max_possible
            argmin = np.argmin(new_values, axis=axis)

            # Deal with completely masked items. Here, use the argmin of the
            # unmasked values.
            mask = np.all(self._mask_, axis=axis)
            if np.any(mask):
                alt_argmin = np.argmin(self._values_, axis=axis)
                if np.shape(mask):
                    argmin[mask] = alt_argmin[mask]
                else:
                    argmin = alt_argmin
                    mask = True
            else:
                mask = False

            result = Scalar(argmin, mask)

        # Convert result to a Python type if necessary
        if builtins is None:
            builtins = Qube.PREFER_BUILTIN_TYPES

        if builtins:
            return result.as_builtin(masked=masked)

        return result

    #===========================================================================
    @staticmethod
    def maximum(*args):
        """A Scalar composed of the maximum among the given Scalars after they
        are all broadcasted to the same shape.

        Masked values are ignored in the comparisons. Derivatives are removed.
        """

        if len(args) == 0:
            raise ValueError('missing arguments to Scalar.maximum()')

        # Convert to scalars of the same shape
        scalars = []
        for arg in args:
            scalars.append(Scalar.as_scalar(arg, recursive=False))

        scalars = Qube.broadcast(*scalars, _protected=False)

        # Make sure there are no denominators
        for scalar in scalars:
          if scalar._drank_:
            raise ValueError('Scalar.maximum() does not support denominators')

        # len == 1 case is easy
        if len(scalars) == 1:
            return scalars[0]

        # Convert to floats if any scalar uses floats
        floats_found = False
        ints_found = False
        for scalar in scalars:
            if scalar.is_float():
                floats_found = True
            if scalar.is_int():
                ints_found = True

        if floats_found and ints_found:
            scalars = [s.as_float() for s in scalars]

        # Create the scalar containing maxima
        result = scalars[0].copy()
        for scalar in scalars[1:]:
            antimask = Qube.and_(scalar._values_ > result._values_,
                                 scalar.antimask)
            antimask = Qube.or_(antimask, result._mask_)
            result[antimask] = scalar[antimask]

        result._clear_cache()
        return result

    #===========================================================================
    @staticmethod
    def minimum(*args):
        """A Scalar composed of the minimum among the given Scalars after they
        are all broadcasted to the same shape.
        """

        if len(args) == 0:
            raise ValueError('missing arguments to Scalar.minimum()')

        # Convert to scalars of the same shape
        scalars = []
        for arg in args:
            scalars.append(Scalar.as_scalar(arg, recursive=False))

        scalars = Qube.broadcast(*scalars, _protected=False)

        # Make sure there are no denominators
        for scalar in scalars:
          if scalar._drank_:
            raise ValueError('Scalar.minimum() does not support denominators')

        # len == 1 case is easy
        if len(scalars) == 1:
            return scalars[0]

        # Convert to floats if any scalar uses floats
        floats_found = False
        ints_found = False
        for scalar in scalars:
            if scalar.is_float():
                floats_found = True
            if scalar.is_int():
                ints_found = True

        if floats_found and ints_found:
            scalars = [s.as_float() for s in scalars]

        # Create the scalar containing minima
        result = scalars[0].copy()
        for scalar in scalars[1:]:
            antimask = Qube.and_(scalar._values_ < result._values_,
                                 scalar.antimask)
            antimask = Qube.or_(antimask, result._mask_)
            result[antimask] = scalar[antimask]

        return result

    #===========================================================================
    def median(self, axis=None, builtins=None, masked=None, out=None):
        """The median of the unmasked values.

        Input:
            axis        an integer axis or a tuple of axes. The median is
                        determined across these axes, leaving any remaining axes
                        in the returned value. If None (the default), then the
                        median is performed across all axes if the object.
            builtins    if True and the result is a single unmasked scalar, the
                        result is returned as a Python int or float instead of
                        as an instance of Scalar. Default is that specified by
                        Qube.PREFER_BUILTIN_TYPES.
            masked      value to return if builtins is True but the returned
                        value is masked. Default is to return a masked value
                        instead of a builtin type.
            out         Ignored. Enables "np.median(Scalar)" to work.
        """

        if self._drank_:
            raise ValueError('Scalar.median() does not support denominators')

        self._check_axis(axis, 'median()')      # make sure axis input is valid

        if self._size_ == 0:
            return self.wod._zero_sized_result(axis)

        if self._shape_ == ():
            result = self.wod.as_float()

        elif not np.any(self._mask_):
            result = Scalar(np.median(self._values_, axis=axis), mask=False,
                            example=self)

        # If all masked, use the unmasked values but leave the result masked
        elif np.all(self._mask_):
            result = Scalar(np.median(self._values_, axis=axis), mask=True,
                            example=self)

        elif axis is None:
            result = Scalar(np.median(self._values_[self.antimask]), mask=False,
                            example=self)

        else:
            # Interpret the axis selection
            len_shape = len(self._shape_)
            if isinstance(axis, int):
                axis = (axis,)

            # Force all indices to be positive
            axes = tuple([a % len_shape for a in axis])

            # Move them to the front
            dest = tuple(range(len(axes)))
            new_scalar = self.move_axis(axes, dest)

            # Flatten the leading axes
            new_scalar = new_scalar.reshape((-1,)
                                            + new_scalar._shape_[len(axes):])

            # Sort along the leading axis, with masked values at the top
            max_possible = self._maxval()
            new_values = new_scalar._values_.copy()
            new_values[new_scalar._mask_] = max_possible
            new_values = np.sort(new_values, axis=0)

            # Count the number of unmasked values for each trailing index
            if isinstance(new_scalar._mask_, (bool, np.bool_)):
                if new_scalar._mask_:
                    count = 0
                else:
                    count = self._values_.size // new_values[0].size
            else:
                count = (self._values_.size // new_values[0].size
                         - np.sum(new_scalar._mask_, axis=0))

            # Define the indices of the middle one or two
            klo = np.maximum((count - 1) // 2, 0)
            khi = count // 2
            indices = tuple(np.indices(new_values.shape[1:]))
            values_lo = new_values[(klo,) + indices]
            values_hi = new_values[(khi,) + indices]

            # Derive the median
            new_values = 0.5 * (values_lo + values_hi)
            new_mask = (count == 0)

            # Fill in masked items using unmasked medians
            if np.any(new_mask):
                if np.shape(new_values):
                    new_values[new_mask] = np.median(self._values_,
                                                     axis=axis)[new_mask]
                else:
                    new_values = np.median(self._values_, axis=axis)

            result = Scalar(new_values, new_mask, units=self._units_)

        result = result.wod

        # Convert result to a Python type if necessary
        if builtins is None:
            builtins = Qube.PREFER_BUILTIN_TYPES

        if builtins:
            return result.as_builtin(masked=masked)

        return result

    #===========================================================================
    def sort(self, axis=0):
        """The array sorted along the speficied axis from minimum to maximum.
        Masked values appear at the end.

        Input:
            axis        an integer axis.
        """

        if self._drank_:
            raise ValueError('Scalar.sort() does not support denominators')

        self._check_axis(axis, 'sort()')        # make sure axis input is valid

        if self._size_ == 0:
            return self.wod._zero_sized_result(axis)

        if not np.any(self._mask_):
            result = Scalar(np.sort(self._values_, axis=axis), mask=False,
                            units=self._units_)

        else:
            maxval = self._maxval()
            new_values = self._values_.copy()
            new_values[self._mask_] = maxval
            new_values = np.sort(new_values, axis=axis)

            # Create the new mask
            if np.shape(self._mask_) == ():
                new_mask = self._mask_
            else:
                new_mask = self._mask_.copy()
                new_mask = np.sort(new_mask, axis=axis)

            # Construct the result
            result = Scalar(new_values, new_mask, units=self._units_)

            # Replace the masked values by the max
            new_values[new_mask] = result.max()

        return result.wod

    ############################################################################
    # Overrides of arithmetic operators
    ############################################################################

    def reciprocal(self, recursive=True, nozeros=False):
        """An object equivalent to the reciprocal of this object.

        Input:
            recursive   True to return the derivatives of the reciprocal too;
                        otherwise, derivatives are removed.
            nozeros     False (the default) to mask out any zero-valued items in
                        this object prior to the divide. Set to True only if you
                        know in advance that this object has no zero-valued
                        items.
        """

        if self._rank_:
           raise ValueError('Scalar.reciprocal() does not support denominators')

        # mask out zeros if necessary
        if nozeros:
            denom = self
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    denom_inv_values = 1. / denom._values_
                    denom_inv_mask = denom._mask_
                except (ZeroDivisionError, RuntimeWarning):
                    raise ValueError('divide by zero in Scalar.reciprocal()')

        else:
            denom = self.mask_where_eq(0, replace=1)
            denom_inv_values = 1. / denom._values_
            denom_inv_mask = denom._mask_

        # Construct the object
        obj = Qube.__new__(type(self))
        obj.__init__(denom_inv_values, denom_inv_mask,
                     units = Units.units_power(self._units_, -1))

        # Fill in derivatives if necessary
        if recursive and self._derivs_:
            factor = -obj*obj       # At this point it has no derivs
            for (key,deriv) in self._derivs_.items():
                obj.insert_deriv(key, factor * deriv)

        return obj

    #===========================================================================
    def identity(self):
        """An object of this subclass equivalent to the identity."""

        # Scalar case
        if self.is_float():
            new_value = 1.
        else:
            new_value = 1

        # Construct the object
        return Scalar(new_value).as_readonly()

    ############################################################################
    # Logical operators
    #   All comparisons involving masked values return False.
    ############################################################################

    def __lt__(self, arg, builtins=True):

        arg = Scalar.as_scalar(arg)
        Units.require_compatible(self._units_, arg._units_)
        if self._denom_ or arg._denom_:
            raise ValueError('Scalar "<" operator does not support '
                             'denominators')

        compare = (self._values_ < arg._values_)

        # Return a Python bool if possible
        if np.isscalar(compare) and builtins:
            if self._mask_ or arg._mask_:
                return False
            return bool(compare)

        compare &= (self.antimask & arg.antimask)

        result = Qube.BOOLEAN_CLASS(compare)
        result._truth_if_all_ = True
        return result

    def __gt__(self, arg, builtins=True):

        arg = Scalar.as_scalar(arg)
        Units.require_compatible(self._units_, arg._units_)
        if self._denom_ or arg._denom_:
            raise ValueError('Scalar ">" operator does not support '
                             'denominators')

        compare = (self._values_ > arg._values_)

        # Return a Python bool if possible
        if np.isscalar(compare) and builtins:
            if self._mask_ or arg._mask_:
                return False
            return bool(compare)

        compare &= (self.antimask & arg.antimask)

        result = Qube.BOOLEAN_CLASS(compare)
        result._truth_if_all_ = True
        return result

    def __le__(self, arg, builtins=True):

        arg = Scalar.as_scalar(arg)
        Units.require_compatible(self._units_, arg._units_)
        if self._denom_ or arg._denom_:
            raise ValueError('Scalar "<=" operator does not support '
                             'denominators')

        compare = (self._values_ <= arg._values_)

        # Return a Python bool if possible
        if np.isscalar(compare) and builtins:
            if self._mask_ or arg._mask_:
                return False
            return bool(compare)

        compare &= (self.antimask & arg.antimask)

        result = Qube.BOOLEAN_CLASS(compare)
        result._truth_if_all_ = True
        return result

    def __ge__(self, arg, builtins=True):

        arg = Scalar.as_scalar(arg)
        Units.require_compatible(self._units_, arg._units_)
        if self._denom_ or arg._denom_:
            raise ValueError('Scalar ">=" operator does not support '
                             'denominators')

        compare = (self._values_ >= arg._values_)

        # Return a Python bool if possible
        if np.isscalar(compare) and builtins:
            if self._mask_ or arg._mask_:
                return False
            return bool(compare)

        compare &= (self.antimask & arg.antimask)

        result = Qube.BOOLEAN_CLASS(compare)
        result._truth_if_all_ = True
        return result

    def __round__(self, digits):

        return Scalar(np.round(self._values_, digits), example=self)

    ############################################################################
    # Exponentiation operator
    ############################################################################

    def _power_0(self, recursive=True):
        x = self.ones(self._shape_, dtype=Qube._dtype(self), mask=self._mask_)
        if recursive:
            for key, deriv in self._derivs_.items():
                x.insert_deriv(key, deriv.zeros(deriv._shape_,
                                                numer=deriv._numer_,
                                                denom=deriv._denom_,
                                                mask=deriv._mask_))
        return x

    def _power_1(self, recursive=True):
        return self if recursive else self.wod

    def _power_2(self, recursive=True):
        result = Scalar(self._values_ * self._values_, self._mask_,
                        units = Units.units_power(self._units_, 2))

        if recursive and self._derivs_:
            factor = 2. * self.wod
            for key, deriv in self._derivs_.items():
                result.insert_deriv(key, factor * deriv)

        return result

    def _power_3(self, recursive=True):
        x_sq = self._values_ * self._values_
        result = Scalar(self._values_ * x_sq, self._mask_,
                        units = Units.units_power(self._units_, 3))

        if recursive and self._derivs_:
            factor = Scalar(3. * x_sq, self._mask_,
                            units = Units.units_power(self._units_, 2))
            for key, deriv in self._derivs_.items():
                result.insert_deriv(key, factor * deriv)

        return result

    def _power_4(self, recursive=True):
        x_sq = self._values_ * self._values_
        result = Scalar(x_sq * x_sq, self._mask_,
                        units = Units.units_power(self._units_, 4))

        if recursive and self._derivs_:
            factor = Scalar(4. * x_sq * self._values_, self._mask_,
                            units = Units.units_power(self._units_, 3))
            for key, deriv in self._derivs_.items():
                result.insert_deriv(key, factor * deriv)

        return result

    def _power_neg_1(self, recursive=True):
        return self.reciprocal(recursive=recursive)

    def _power_half(self, recursive=True):
        return self.sqrt(recursive=recursive)

    def _power_neg_half(self, recursive=True):
        return self.sqrt(recursive=recursive).reciprocal(recursive=recursive)

    _EASY_INT_POWERS = {
         0: _power_0,
         1: _power_1,
         2: _power_2,
         3: _power_3,
         4: _power_4,
        -1: _power_neg_1,
    }

    _EASY_FLOAT_POWERS = {
         0.5: _power_half,
        -0.5: _power_neg_half,
    }

    # Generic exponentiation, PolyMath scalar to a single scalar power
    def __pow__(self, expo, recursive=True):

        if self._denom_:
            raise ValueError('Scalar "**" operator does not support '
                             'denominators')

        # Handle the common and easy cases where there is only a single exponent
        if isinstance(expo, numbers.Integral):
            try:
                return Scalar._EASY_INT_POWERS[expo](self, recursive)
            except KeyError:
                pass

        if isinstance(expo, numbers.Real):
            try:
                return Scalar._EASY_FLOAT_POWERS[expo](self, recursive)
            except KeyError:
                pass

        # Interpret the exponent and mask if any
        if isinstance(expo, Scalar):
            if expo._rank_:
                raise ValueError('Scalar "**" exponent must be scalar')
            Units.require_unitless(expo._units_)

            if expo._derivs_:
                raise ValueError('Scalar "**" exponent derivatives are not '
                                 'supported')

        else:
            expo = Scalar(expo)

        # 0-D case
        if not self._shape_ and not expo._shape_:
            try:
                new_values = self._values_ ** expo._values_
            except (ValueError, ZeroDivisionError):
                return self.masked_single(recursive)

            if not isinstance(new_values, numbers.Real):
                return self.masked_single(recursive)

            new_mask = False
            new_units = Units.units_power(self._units_, expo._values_)

        # Array case
        else:

            # Without this step, negative int exponents on int values truncate
            # to 0.
            if expo.is_int():
                if expo._shape_:
                    if np.any(expo._values_ < 0):
                        expo = expo.as_float()
                elif expo._values_ < 0:
                    expo = expo.as_float()

            # Plow forward with the results blindly, then mask nan and inf.
            # Zero to a negative power creates a RuntTimeWarning, which needs to
            # be suppressed.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                new_values = self._values_ ** expo._values_

            new_mask = Qube.or_(self._mask_, expo._mask_)
            invalid = np.isnan(new_values) | np.isinf(new_values)
            if np.any(invalid):
                new_values[invalid] = 1.
                new_mask = Qube.or_(new_mask, invalid)

            # Check units and exponent
            if Units.is_unitless(self._units_):
                new_units = None
            elif np.isscalar(expo._values_):
                new_units = Units.units_power(self._units_, expo._values_)
            else:
                raise ValueError('Scalar with units cannot be raised to '
                                 'multiple powers')

        obj = Scalar.__new__(type(self))
        obj.__init__(new_values, new_mask, units=new_units, example=self)

        # Evaluate the derivatives if necessary
        if recursive and self._derivs_:
            factor = expo * self.__pow__(expo-1, recursive=False)
            for (key, deriv) in self._derivs_.items():
                obj.insert_deriv(key, factor * deriv)

        return obj

################################################################################
# Useful class constants
################################################################################

Scalar.ZERO   = Scalar(0).as_readonly()
Scalar.ONE    = Scalar(1).as_readonly()
Scalar.TWO    = Scalar(2).as_readonly()
Scalar.THREE  = Scalar(3).as_readonly()

Scalar.PI     = Scalar(np.pi).as_readonly()
Scalar.TWOPI  = Scalar(2*np.pi).as_readonly()
Scalar.HALFPI = Scalar(np.pi/2).as_readonly()

Scalar.MASKED = Scalar(1, True).as_readonly()

Scalar.INF    = Scalar(np.inf).as_readonly()
Scalar.NEGINF = Scalar(-np.inf).as_readonly()

################################################################################
# Once the load is complete, we can fill in a reference to the Scalar class
# inside the Qube object.
################################################################################

Qube.SCALAR_CLASS = Scalar

################################################################################
