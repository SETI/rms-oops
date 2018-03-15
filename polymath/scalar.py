################################################################################
# polymath/scalar.py: Scalar subclass of PolyMath base class
#
# Mark Showalter, PDS Ring-Moon Systems Node, SETI Institute
################################################################################

from __future__ import division
import numpy as np
import sys
import warnings

from .qube  import Qube
from .units import Units

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
    MASKS_OK = True     # True to allow masks; False to disallow them.
    DERIVS_OK = True    # True to disallow derivatives; False to allow them.

    DEFAULT_VALUE = 1

    def _maxval(self):
        """Internal method returns the maximum value associated with a dtype."""

        dtype = self.values.dtype
        if dtype.kind == 'f':
            return np.inf
        elif dtype.kind == 'u':
            return 256**dtype.itemsize - 1
        elif dtype.kind == 'i':
            return 256**dtype.itemsize//2 - 1
        else:
            raise ValueError('invalid dtype %s' % str(dtype))

    def _minval(self):
        """Internal method returns the minimum value associated with a dtype."""

        # Define constant maxval
        dtype = self.values.dtype
        if dtype.kind == 'f':
            return -np.inf
        elif dtype.kind == 'u':
            return 0
        elif dtype.kind == 'i':
            return -256**dtype.itemsize//2
        else:
            raise ValueError('invalid dtype %s' % str(dtype))

    @staticmethod
    def as_scalar(arg, recursive=True):
        """Return the argument converted to Scalar if possible.

        If recursive is True, derivatives will also be converted.
        """

        if type(arg) == Scalar:
            if recursive: return arg
            return arg.without_derivs()

        if isinstance(arg, Qube):
            if type(arg) == Qube.BOOLEAN_CLASS:
                return Qube.BOOLEAN_CLASS(arg).as_int()

            arg = Scalar(arg)
            if recursive: return arg
            return arg.without_derivs()

        if isinstance(arg, Units):
            return Scalar(arg.from_units_factor, units=arg)

        return Scalar(arg)

    def as_index(self, masked=None):
        """Return an object suitable for indexing a NumPy ndarray.

        Input:
            masked      the value to insert in the place of a masked item. If
                        None and the object contains masked elements, the array
                        will be flattened and masked elements will be skipped.
        """

        obj = self.as_int()

        if not np.any(self.mask):
            return obj.values

        if np.shape(obj.mask) == ():
            raise ValueError('object is entirely masked')

        if masked is None:
            obj = obj.flatten()
            if np.shape(obj.values) == ():
                if obj.mask:
                    return None
                else:
                    return obj.values
            return obj.values[obj.antimask]
        else:
            obj = obj.copy()
            obj.values[obj.mask] = masked
            return obj.values

    def as_index_and_mask(self, purge=False, masked=None):
        """Objects suitable for indexing an N-dimensional array and its mask.

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

        ints = self.as_int()

        # If nothing is masked, this is easy
        if not np.any(self.mask):
            return (ints.values, None)

        # If purging...
        if purge:
            # If all masked...
            if ints.mask is True:
                return ((), None)

            # If partially masked...
            return (ints.values[ints.antimask], None)

        # Without a replacement...
        if masked is None:
            new_values = ints.values

        # If all masked...
        elif ints.mask is True:
            if type(ints.values) == np.ndarray:
                new_values = np.empty(ints.shape, dtype='int').fill(masked)
            else:
                new_values = masked

        # If partially masked...
        else:
            new_values = ints.values.copy()
            new_values[ints.mask] = masked

        # Return results
        return (new_values, ints.mask)

    def int(self):
        """Return an integer (floor) version of this Scalar.

        If this object already contains integers, it is returned as is.
        Otherwise, a copy is returned. Derivatives are always removed. Units
        are disallowed.

        If the result is a single unmasked scalar, it is returned as a Python
        scalar rather than as an instance of Scalar.
        """

        Units.require_unitless(self.units)
        return self.without_derivs().as_int()

    def frac(self, recursive=True):
        """Return an object containing the fractional components of all values.

        The returned object is an instance of the same subclass as this object.

        Inputs:
            recursive   True to include the derivatives of the returned object.
                        frac() leaves the derivatives unchanged.
        """

        Units.require_unitless(self.units)

        # Convert to fractional values
        if isinstance(self.values, np.ndarray):
            new_values = (self.values % 1.)
        else:
            new_values = self.values % 1.

        # Construct a new copy
        obj = Qube.__new__(type(self))
        obj.__init__(new_values, mask=self.mask, derivs=self.derivs)

        return obj

    def abs(self, recursive=True):
        """Return the absolute value of each value.

        Inputs:
            recursive   True to include the derivatives of the absolute values
                        inside the returned object.
        """

        if recursive:
            return abs(self)
        else:
            return abs(self.without_derivs())

    def sin(self, recursive=True):
        """Return the sine of each value.

        Inputs:
            recursive   True to include the derivatives of the sine inside the
                        returned object.
        """

        Units.require_angle(self.units)
        obj = Scalar(np.sin(self.values), mask=self.mask)

        if recursive and self.derivs:
            factor = self.without_derivs().cos()
            for (key, deriv) in self.derivs.items():
                obj.insert_deriv(key, factor * deriv)

        return obj

    def cos(self, recursive=True):
        """Return the cosine of each value.

        Inputs:
            recursive   True to include the derivatives of the cosine inside the
                        returned object.
        """

        Units.require_angle(self.units)
        obj = Scalar(np.cos(self.values), mask=self.mask)

        if recursive and self.derivs:
            factor = -self.without_derivs().sin()
            for (key, deriv) in self.derivs.items():
                obj.insert_deriv(key, factor * deriv)

        return obj

    def tan(self, recursive=True):
        """Return the tangent of each value.

        Inputs:
            recursive   True to include the derivatives of the tangent inside
                        the returned object.
        """

        Units.require_angle(self.units)

        obj = Scalar(np.tan(self.values), mask=self.mask)

        if recursive and self.derivs:
            inv_sec_sq = self.without_derivs().cos()**(-2)
            for (key, deriv) in self.derivs.items():
                obj.insert_deriv(key, inv_sec_sq * deriv)

        return obj

    def arcsin(self, recursive=True, check=True):
        """Return the arcsine of each value.

        If this object is read-only, the returned object will also be read-only.

        Inputs:
            recursive   True to include the derivatives of the arcsine inside
                        the returned object.
            check       True to mask out the locations of any values outside the
                        domain [-1,1]. If False, a ValueError will be raised if
                        any value is encountered where the arcsine is undefined.
                        Check=True is slightly faster if we already know at the
                        time of the call that all input values are valid.
        """

        # Limit domain to [-1,1] if necessary
        if check:
            Units.require_unitless(self.units)

            temp_mask = (self.values < -1) | (self.values > 1)
            if np.any(temp_mask):
                if temp_mask is True:
                    temp_values = 0.
                else:
                    temp_values = self.values.copy()
                    temp_values[temp_mask] = 0.
                    temp_mask |= self.mask
            else:
                temp_values = self.values
                temp_mask = self.mask

            obj = Scalar(np.arcsin(temp_values), temp_mask)

        else:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    func_values = np.arcsin(self.values)
                except:
                    raise ValueError('arcsin of value outside domain (-1,1)')

            obj = Scalar(func_values, mask=self.mask)

        if recursive and self.derivs:
            x = self.without_derivs()
            factor = (1. - x*x)**(-0.5)
            for (key, deriv) in self.derivs.items():
                obj.insert_deriv(key, factor * deriv)

        return obj

    def arccos(self, recursive=True, check=True):
        """Return the arccosine of each value.

        If this object is read-only, the returned object will also be read-only.

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

        # Limit domain to [-1,1] if necessary
        if check:
            Units.require_unitless(self.units)

            temp_mask = (self.values < -1) | (self.values > 1)
            if np.any(temp_mask):
                if temp_mask is True:
                    temp_values = 0.
                else:
                    temp_values = self.values.copy()
                    temp_values[temp_mask] = 0.
                    temp_mask |= self.mask
            else:
                temp_values = self.values
                temp_mask = self.mask

            obj = Scalar(np.arccos(temp_values), temp_mask)

        else:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    func_values = np.arccos(self.values)
                except:
                    raise ValueError('arccos of value outside domain (-1,1)')

            obj = Scalar(func_values, mask=self.mask)

        if recursive and self.derivs:
            x = self.without_derivs()
            factor = -(1. - x*x)**(-0.5)
            for (key, deriv) in self.derivs.items():
                obj.insert_deriv(key, factor * deriv)

        return obj

    def arctan(self, recursive=True):
        """Return the arctangent of each value.

        If this object is read-only, the returned object will also be read-only.

        Inputs:
            recursive   True to include the derivatives of the arctangent
                        inside the returned object.
        """

        Units.require_unitless(self.units)

        obj = Scalar(np.arctan(self.values), mask=self.mask)

        if recursive and self.derivs:
            factor = 1. / (1. + self.without_derivs()**2)
            for (key, deriv) in self.derivs.items():
                obj.insert_deriv(key, factor * deriv)

        return obj

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
        Units.require_compatible(y.units, x.units)

        obj = Scalar(np.arctan2(y.values, x.values), x.mask | y.mask)

        if recursive and (x.derivs or y.derivs):
            x_wod = x.without_derivs()
            y_wod = y.without_derivs()
            denom_inv = (x_wod**2 + y_wod**2).reciprocal()

            new_derivs = {}
            for (key, y_deriv) in y.derivs.items():
                new_derivs[key] = x_wod * denom_inv * y_deriv

            for (key, x_deriv) in x.derivs.items():
                term = y_wod * denom_inv * x_deriv
                if key in new_derivs:
                    new_derivs[key] -= term
                else:
                    new_derivs[key] = -term

            obj.insert_derivs(new_derivs)

        return obj

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

        if check:
            no_negs = self.mask_where_lt(0.,1.)
            sqrt_vals = np.sqrt(no_negs.values)

        else:
            no_negs = self
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    sqrt_vals = np.sqrt(no_negs.values)
                except:
                    raise ValueError('sqrt of value negative value')

        obj = Scalar(sqrt_vals, mask=no_negs.mask,
                                units=Units.sqrt_units(no_negs.units))

        if recursive and no_negs.derivs:
            factor = 0.5 / obj
            for (key, deriv) in self.derivs.items():
                obj.insert_deriv(key, factor * deriv)

        return obj

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

        if check:
            no_negs = self.mask_where_le(0., 1.)
            log_values = np.log(no_negs.values)
        else:
            no_negs = self
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    log_values = np.log(no_negs.values)
                except:
                    raise ValueError('log of non-positive value')

        obj = Scalar(log_values, mask=no_negs.mask)

        if recursive and no_negs.derivs:
            for (key, deriv) in self.derivs.items():
                obj.insert_deriv(key, deriv / no_negs)

        return obj

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

        Units.require_angle(self.units)

        if check:
            no_oflow = self.mask_where_gt(EXP_CUTOFF, EXP_CUTOFF)
            exp_values = np.exp(no_oflow.values)

        else:
            no_oflow = self
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    exp_values = np.exp(no_oflow.values)
                except:
                    raise ValueError('overflow encountered in exp')

        obj = Scalar(exp_values, mask=no_oflow.mask)

        if recursive and self.derivs:
            for (key, deriv) in self.derivs.items():
                obj.insert_deriv(key, deriv * exp_values)

        return obj

    def sign(self, zeros=True):
        """Return the sign of each value as +1, -1 or 0.

        If zeros is False, then only values of +1 and -1 will be returned;
        sign(0) = +1 instead of 0.
        """

        result = Scalar(np.sign(self.values), mask=self.mask)

        if not zeros:
            result[result == 0] = 1

        return result

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

        discr = neg_half_b*neg_half_b - a*c

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
                linear_x0 = linear_x0.broadcast_into_shape(x0.shape)
                a_zeros   = a_zeros.broadcast_into_shape(x0.shape)
                x0[a_zeros] = linear_x0[a_zeros]

        if include_antimask:
            return (x0, x1, Boolean(discr.values >= 0, discr.mask))
        else:
            return (x0, x1)

    def eval_quadratic(self, a, b, c, recursive=True):
        """Evaluate a quadratic function for this Scalar.

        The value returned is:
            a * self**2 + b * self + c
        """

        if not recursive:
            self = self.without_derivs()
            a = Scalar.as_scalar(a, recursive=False)
            b = Scalar.as_scalar(b, recursive=False)
            c = Scalar.as_scalar(c, recursive=False)

        return self * (self * a + b) + c

    def max(self, axis=None):
        """Return the maximum of the unmasked values.

        NOTE: This only occurs if Qube.PREFER_BUILTIN_TYPES is True:
        If the result is a single scalar, it is returned as a Python value
        rather than as a Scalar. Denominators are not supported.

        Input:
            axis        an integer axis or a tuple of axes. The maximum is
                        determined across these axes, leaving any remaining axes
                        in the returned value. If None (the default), then the
                        maximum is performed across all axes if the object.
        """

        if self.drank:
            raise ValueError('denominators are not supported in max()')

        if self.shape == ():
            result = self

        elif not np.any(self.mask):
            result = Scalar(np.max(self.values, axis=axis), mask=False,
                            units=self.units)

        elif axis is None:
            if np.shape(self.mask) == ():
                result = Scalar(np.max(self.values), mask=self.mask,
                                units=self.units)
            elif np.all(self.mask):
                result = Scalar(np.max(self.values), mask=True,
                                units=self.units)
            else:
                result = Scalar(np.max(self.values[self.antimask]),
                                mask=False, units=self.units)

        else:

            # Create new array
            minval = self._minval()

            new_values = self.values.copy()
            new_values[self.mask] = minval
            new_values = np.max(new_values, axis=axis)

            # Create new mask
            if self.mask is True:
                new_mask = True
            elif self.mask is False:
                new_mask = False
            else:
                new_mask = np.all(self.mask, axis=axis)

            # Use the max of the unmasked values if all are masked
            if np.all(new_mask):
                unmasked_maxes = np.max(self.values, axis=axis)
                new_values = unmasked_maxes
                new_mask = True
            elif np.any(new_mask):
                unmasked_maxes = np.max(self.values, axis=axis)
                new_values[new_mask] = unmasked_maxes[new_mask]
            else:
                new_mask = False

            result = Scalar(new_values, new_mask, units=self.units)

        result = result.without_derivs()

        # Convert result to a Python constant if necessary
        if Qube.PREFER_BUILTIN_TYPES:
            return result.as_builtin()

        return result

    def min(self, axis=None):
        """Return the minimum of the unmasked values.

        NOTE: This only occurs if Qube.PREFER_BUILTIN_TYPES is True:
        If the result is a single scalar, it is returned as a Python value
        rather than as a Scalar. Denominators are not supported.

        Input:
            axis        an integer axis or a tuple of axes. The minimum is
                        determined across these axes, leaving any remaining axes
                        in the returned value. If None (the default), then the
                        minimum is performed across all axes if the object.
        """

        if self.drank:
            raise ValueError('denominators are not supported in min()')

        if self.shape == ():
            result = self

        elif not np.any(self.mask):
            result = Scalar(np.min(self.values, axis=axis), mask=False,
                            derivs={}, units=self.units)

        elif axis is None:
            if np.shape(self.mask) == ():
                result = Scalar(np.min(self.values), mask=self.mask,
                                derivs={}, units=self.units)
            elif np.all(self.mask):
                result = Scalar(np.min(self.values), mask=True,
                                derivs={}, units=self.units)
            else:
                result = Scalar(np.min(self.values[self.antimask]),
                                mask=False, derivs={}, units=self.units)

        else:
            # Create new array
            maxval = self._maxval()

            new_values = self.values.copy()
            new_values[self.mask] = maxval
            new_values = np.min(new_values, axis=axis)

            # Create new mask
            if self.mask is True:
                new_mask = True
            elif self.mask is False:
                new_mask = False
            else:
                new_mask = np.all(self.mask, axis=axis)

            # Use the min of the unmasked values if all are masked
            if np.all(new_mask):
                unmasked_mins = np.min(self.values, axis=axis)
                new_values = unmasked_mins
                new_mask = True
            elif np.any(new_mask):
                unmasked_mins = np.min(self.values, axis=axis)
                new_values[new_mask] = unmasked_mins[new_mask]
            else:
                new_mask = False

            result = Scalar(new_values, new_mask, derivs={}, units=self.units)

        # Convert result to a Python constant if necessary
        if Qube.PREFER_BUILTIN_TYPES:
            return result.as_builtin()

        return result

    def argmax(self, axis=None):
        """Return the index of the maximum of the unmasked values.

        Input:
            axis        an optional integer axis.

        If axis is None, then it returns the index of the maximum argument in
        the flattened array. Otherwise, it returns an integer Scalar array of
        the same shape as self except that the specified axis has been removed.
        Each value indicates the index of the maximum along that axis. The index
        is masked where the axis is entirely masked.
        """

        if self.drank:
            raise ValueError('denominators are not supported in argmax()')

        if self.shape == ():
            raise ValueError('no argmax for Scalar with shape = ()')

        if not np.any(self.mask):
            result = Scalar(np.argmax(self.values, axis=axis), mask=False)

        elif axis is None:
            if np.shape(self.mask) == ():
                result = Scalar(np.argmax(self.values), mask=self.mask)
            elif np.all(self.mask):
                result = Scalar(np.argmax(self.values), mask=True)
            else:
                minval = self._minval()
                values = self.values.copy()
                values[self.mask] = minval
                if np.all(values == minval):
                    result = Scalar(np.argmin(self.mask), mask=False)
                else:
                    result = Scalar(np.argmax(values), mask=False)

        else:

            # Create new array
            minval = self._minval()
            new_values = self.values.copy()
            new_values[self.mask] = minval
            argmaxes = np.argmax(new_values, axis=axis)

            # Create new mask
            if self.mask is True:
                new_mask = True
            elif self.mask is False:
                new_mask = False
            else:
                new_mask = np.all(self.mask, axis=axis)

            # Use the argmax of the unmasked values if all are masked
            if np.all(new_mask):
                unmasked_argmaxes = np.argmax(self.values, axis=axis)
                argmaxes = unmasked_argmaxes
                new_mask = True
            elif np.any(new_mask):
                unmasked_argmaxes = np.argmax(self.values, axis=axis)
                argmaxes[new_mask] = unmasked_argmaxes[new_mask]
            else:
                new_mask = False

            result = Scalar(argmaxes, new_mask)

        # Convert result to a Python constant if necessary
        if result.shape == () and not result.mask:
            return int(result.values)

        return result

    def argmin(self, axis=None):
        """Return the index of the minimum of the unmasked values.

        Input:
            axis        an optional integer axis.

        If axis is None, then it returns the index of the minimum argument in
        the flattened array. Otherwise, it returns an integer Scalar array of
        the same shape as self except that the specified axis has been removed.
        Each value indicates the index of the minimum along that axis. The index
        is masked where the axis is entirely masked.
        """

        if self.drank:
            raise ValueError('denominators are not supported in argmin()')

        if self.shape == ():
            raise ValueError('no argmin for Scalar with shape = ()')

        if not np.any(self.mask):
            result = Scalar(np.argmin(self.values, axis=axis), mask=False)

        elif axis is None:
            if np.shape(self.mask) == ():
                result = Scalar(np.argmin(self.values), mask=self.mask)
            elif np.all(self.mask):
                result = Scalar(np.argmin(self.values), mask=True)
            else:
                maxval = self._maxval()
                values = self.values.copy()
                values[self.mask] = maxval
                if np.all(values == maxval):
                    result = Scalar(np.argmin(self.mask), mask=False)
                else:
                    result = Scalar(np.argmin(values), mask=False)

        else:

            # Create new array
            maxval = self._maxval()
            new_values = self.values.copy()
            new_values[self.mask] = maxval
            argmins = np.argmin(new_values, axis=axis)

            # Create new mask
            if self.mask is True:
                new_mask = True
            elif self.mask is False:
                new_mask = False
            else:
                new_mask = np.all(self.mask, axis=axis)

            # Use the argmin of the unmasked values if all are masked
            if np.all(new_mask):
                unmasked_argmins = np.argmin(self.values, axis=axis)
                argmins = unmasked_argmins
                new_mask = True
            elif np.any(new_mask):
                unmasked_argmins = np.argmin(self.values, axis=axis)
                argmins[new_mask] = unmasked_argmins[new_mask]
            else:
                new_mask = False

            result = Scalar(argmins, new_mask)

        # Convert result to a Python constant if necessary
        if result.shape == () and not result.mask:
            return int(result.values)

        return result

    @staticmethod
    def maximum(*scalars):
        """Return a Scalar composed of the maximum among the given Scalars after
        they are all broadcasted to the same shape."""

        if len(scalars) == 0:
            raise ValueError('invalid number of arguments')

        # Convert to scalars of the same shape
        scalars = [Scalar.as_scalar(s,recursive=False) for s in scalars]
        scalars = Qube.broadcast(*scalars)

        # Make sure there are no denominators
        for scalar in scalars:
          if scalar.drank:
            raise ValueError('denominators are not supported in maximum()')

        # len == 1 case is easy
        if len(scalars) == 1:
            return scalars[0]

        # Convert to floats if any scalar uses floats
        floats_found = False
        ints_found = False
        for scalar in scalars:
            if scalar.is_float(): floats_found = True
            if scalar.is_int(): ints_found = True

        if floats_found and ints_found:
            scalars = [s.as_float() for s in scalars]

        # Create the scalar containing maxima
        maxes = scalars[0].copy()
        for scalar in scalars[1:]:
            scalar = scalar.mask_where(scalar <= maxes)
            maxes._set_values_(scalar.values, False, scalar.antimask)

        return maxes

    @staticmethod
    def minimum(*scalars):
        """Return a Scalar composed of the minimum among the given Scalars after
        they are all broadcasted to the same shape."""

        if len(scalars) == 0:
            raise ValueError('invalid number of arguments')

        # Convert to scalars of the same shape
        scalars = [Scalar.as_scalar(s,recursive=False) for s in scalars]
        scalars = Qube.broadcast(*scalars)

        # Make sure there are no denominators
        for scalar in scalars:
          if scalar.drank:
            raise ValueError('denominators are not supported in minimum()')

        # len == 1 case is easy
        if len(scalars) == 1:
            return scalars[0]

        # Convert to floats if any scalar uses floats
        floats_found = False
        ints_found = False
        for scalar in scalars:
            if scalar.is_float(): floats_found = True
            if scalar.is_int(): ints_found = True

        if floats_found and ints_found:
            scalars = [s.as_float() for s in scalars]

        # Create the scalar containing minima
        mins = scalars[0].copy()
        for scalar in scalars[1:]:
            scalar = scalar.mask_where(scalar >= mins)
            mins._set_values_(scalar.values, False, scalar.antimask)

        return mins

    def sum(self, axis=None):
        """Return the sum of the unmasked values.

        NOTE: This only occurs if Qube.PREFER_BUILTIN_TYPES is True:
        If the result is a single scalar, it is returned as a Python value
        rather than as a Scalar. Denominators are not supported.

        Input:
            axis        an integer axis or a tuple of axes. The sum is
                        determined across these axes, leaving any remaining axes
                        in the returned value. If None (the default), then the
                        sum is performed across all axes if the object.
        """

        if self.drank:
            raise ValueError('denominators are not supported in sum()')

        if self.shape == ():
            result = self

        elif not np.any(self.mask):
            result = Scalar(np.sum(self.values, axis=axis), units=self.units)

        elif axis is None:
            if np.shape(self.mask) == ():
                result = Scalar(np.sum(self.values), mask=self.mask,
                                units=self.units)
            elif np.all(self.mask):
                result = Scalar(np.sum(self.values), mask=True,
                                units=self.units)
            else:
                result = Scalar(np.sum(self.values[self.antimask]),
                                mask=False, units=self.units)

        else:
            # Create new array and mask
            new_values = self.values.copy()
            new_values[self.mask] = 0
            new_values = np.sum(new_values, axis=axis)

            if np.shape(self.mask):
                new_mask = np.all(self.mask, axis=axis)
            else:
                new_mask = self.mask

            # Fill in masked items using unmasked sums
            if np.any(new_mask):
                if np.shape(new_values):
                    new_values[new_mask] = np.sum(self.values,
                                                  axis=axis)[new_mask]
                else:
                    new_values = np.sum(self.values, axis=axis)

            result = Scalar(new_values, new_mask, units=self.units)

        # Convert result to a Python constant if necessary
        if Qube.PREFER_BUILTIN_TYPES:
            return result.as_builtin()

        return result

    def mean(self, axis=None):
        """Return the mean of the unmasked values.

        NOTE: This only occurs if Qube.PREFER_BUILTIN_TYPES is True:
        If the result is a single scalar, it is returned as a Python value
        rather than as a Scalar. Denominators are not supported.

        Input:
            axis        an integer axis or a tuple of axes. The mean is
                        determined across these axes, leaving any remaining axes
                        in the returned value. If None (the default), then the
                        mean is performed across all axes if the object.
        """

        if self.drank:
            raise ValueError('denominators are not supported in mean()')

        if self.shape == ():
            result = self.as_float()

        elif not np.any(self.mask):
            result = Scalar(np.mean(self.values, axis=axis), mask=False,
                            units=self.units)

        elif axis is None:
            if np.shape(self.mask) == ():
                result = Scalar(np.mean(self.values), mask=self.mask,
                                units=self.units)
            elif np.all(self.mask):
                result = Scalar(np.mean(self.values), mask=True,
                                units=self.units)
            else:
                result = Scalar(np.mean(self.values[self.antimask]),
                                mask=False, units=self.units)

        else:
            # Create new array and mask
            new_values = self.values.copy()
            new_values[self.mask] = 0
            new_values = np.sum(new_values, axis=axis).astype('float')

            if self.mask is True:
                count = 0.
            elif self.mask is False:
                count = np.size(self.values) / float(np.size(new_values))
            else:
                count = np.sum(self.mask == False, axis=axis).astype('float')

            new_mask = (count == 0.)
            new_values /= np.maximum(count, 1.)

            # Fill in masked items using unmasked means
            if np.any(new_mask):
                if np.shape(new_values):
                    new_values[new_mask] = np.mean(self.values,
                                                   axis=axis)[new_mask]
                else:
                    new_values = np.mean(self.values, axis=axis)

            result = Scalar(new_values, new_mask, units=self.units)

        # Convert result to a Python constant if necessary
        if Qube.PREFER_BUILTIN_TYPES:
            return result.as_builtin()

        return result

    def median(self, axis=None):
        """Return the median of the unmasked values.

        NOTE: This only occurs if Qube.PREFER_BUILTIN_TYPES is True:
        If the result is a single scalar, it is returned as a Python value
        rather than as a Scalar. Denominators are not supported.

        Input:
            axis        an integer axis or a tuple of axes. The median is
                        determined across these axes, leaving any remaining axes
                        in the returned value. If None (the default), then the
                        median is performed across all axes if the object.
        """

        if self.drank:
            raise ValueError('denominators are not supported in median()')

        if self.shape == ():
            result = self.as_float()

        elif not np.any(self.mask):
            result = Scalar(np.median(self.values, axis=axis), mask=False,
                            units=self.units)

        elif axis is None:
            if np.shape(self.mask) == ():
                result = Scalar(np.median(self.values), mask=self.mask,
                                units=self.units)
            elif np.all(self.mask):
                result = Scalar(np.median(self.values), mask=True,
                                units=self.units)
            else:
                result = Scalar(np.median(self.values[self.antimask]),
                                mask=False, units=self.units)

        else:
            # Interpret the axis selection
            len_shape = len(self.shape)
            if isinstance(axis, int):
                axis = (axis,)

            axes = list(axis)
            axes = [a % len_shape for a in axes]
            axes = list(set(axes))
            axes.sort(reverse=True)

            # Reorganize so that the leading axis is a flattened version of all
            # the axes over which the median is to be performed. Remaining axes
            # stay in their original order.
            new_scalar = self.roll_axis(axes[0], 0, recursive=False)
            for k in axes[1:]:
                new_scalar.roll_axis(k+1, 0)
                shape = new_scalar.shape
                new_scalar = new_scalar.reshape((shape[0] * shape[1],) +
                                                shape[2:])

            # Sort along the leading axis, with masked values at the top
            maxval = self._maxval()

            new_values = new_scalar.values.copy()
            new_values[new_scalar.mask] = maxval
            new_values = np.sort(new_values, axis=0)

            # Count the number of unmasked values
            if new_scalar.mask is True:
                count = 0
            elif new_scalar.mask is False:
                count = self.values.size // new_values[0].size
            else:
                count = np.sum(new_scalar.mask == False, axis=0)

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
                    new_values[new_mask] = np.median(self.values,
                                                     axis=axis)[new_mask]
                else:
                    new_values = np.median(self.values, axis=axis)

            result = Scalar(new_values, new_mask, units=self.units)

        result = result.without_derivs()

        # Convert result to a Python constant if necessary
        if Qube.PREFER_BUILTIN_TYPES:
            return result.as_builtin()

        return result

    def sort(self, axis=0):
        """Return the array sorted along the speficied axis from minimum to
        maximum. Masked values appear at the end.

        Input:
            axis        an integer axis.
        """

        if self.drank:
            raise ValueError('denominators are not supported in sort()')

        if self.shape == ():
            result = self

        elif not np.any(self.mask):
            result = Scalar(np.sort(self.values, axis=axis), mask=False,
                            units=self.units)

        else:
            maxval = self._maxval()
            new_values = self.values.copy()
            new_values[self.mask] = maxval
            new_values = np.sort(new_values, axis=axis)

            # Create the new mask
            if np.shape(self.mask) == ():
                new_mask = self.mask
            else:
                new_mask = self.mask.copy()
                new_mask = np.sort(new_mask, axis=axis)

            # Construct the result
            result = Scalar(new_values, new_mask, units=self.units)

            # Replace the masked values by the max
            new_values[new_mask] = result.max()

        result = result.without_derivs()
        return result

    ############################################################################
    # Overrides of arithmetic operators
    ############################################################################

    def reciprocal(self, recursive=True, nozeros=False):
        """Return an object equivalent to the reciprocal of this object.

        Input:
            recursive   True to return the derivatives of the reciprocal too;
                        otherwise, derivatives are removed.
            nozeros     False (the default) to mask out any zero-valued items in
                        this object prior to the divide. Set to True only if you
                        know in advance that this object has no zero-valued
                        items.
        """

        assert self.drank == 0

        # mask out zeros if necessary
        if nozeros:
          denom = self
          with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                denom_inv_values = 1. / denom.values
                denom_inv_mask = denom.mask
            except:
                raise ValueError('divide by zero encountered in reciprocal()')

        else:
            denom = self.mask_where_eq(0,1)
            denom_inv_values = 1. / denom.values
            denom_inv_mask = denom.mask

        # Construct the object
        obj = Qube.__new__(type(self))
        obj.__init__(denom_inv_values, denom_inv_mask,
                     Units.units_power(self.units, -1))

        # Fill in derivatives if necessary
        if recursive and self.derivs:
            factor = -obj*obj       # At this point it has no derivs
            for (key,deriv) in self.derivs.items():
                obj.insert_deriv(key, factor * deriv)

        return obj

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
    ############################################################################

    # (<) operator
    def __lt__(self, arg):
        """True where self < arg; also False where either value is masked."""

        arg = Scalar.as_scalar(arg)
        Units.require_compatible(self.units, arg.units)

        compare = (self.values < arg.values)
        mask = self.mask | arg.mask

        if np.shape(compare) == ():
            if mask: compare = False
            return bool(compare)

        if np.shape(mask) == ():
            if mask: compare.fill(False)
        else:
            compare[mask] = False

        return Qube.BOOLEAN_CLASS(compare)

    # (>) operator
    def __gt__(self, arg):
        """True where self > arg; also False where either value is masked."""

        arg = Scalar.as_scalar(arg)
        Units.require_compatible(self.units, arg.units)

        compare = (self.values > arg.values)
        mask = self.mask | arg.mask

        if np.shape(compare) == ():
            if mask: compare = False
            return bool(compare)

        if np.shape(mask) == ():
            if mask: compare.fill(False)
        else:
            compare[mask] = False

        return Qube.BOOLEAN_CLASS(compare)

    # (<=) operator
    def __le__(self, arg):
        """True where self <= arg or where both values are masked."""

        arg = Scalar.as_scalar(arg)
        Units.require_compatible(self.units, arg.units)

        compare = (self.values <= arg.values)
        one_masked = self.mask ^ arg.mask
        both_masked = self.mask & arg.mask

        if np.shape(compare) == ():
            if both_masked: compare = True
            if one_masked: compare = False
            return bool(compare)

        if np.shape(both_masked) == ():
            if both_masked: compare.fill(True)
            if one_masked: compare.fill(False)
        else:
            compare[both_masked] = True
            compare[one_masked] = False

        return Qube.BOOLEAN_CLASS(compare)

    # (>=) operator
    def __ge__(self, arg):
        """True where self >= arg or where both values are masked."""

        arg = Scalar.as_scalar(arg)
        Units.require_compatible(self.units, arg.units)

        compare = (self.values >= arg.values)
        one_masked = self.mask ^ arg.mask
        both_masked = self.mask & arg.mask

        if np.shape(compare) == ():
            if both_masked: compare = True
            if one_masked: compare = False
            return bool(compare)

        if np.shape(both_masked) == ():
            if both_masked: compare.fill(True)
            if one_masked: compare.fill(False)
        else:
            compare[both_masked] = True
            compare[one_masked] = False

        return Qube.BOOLEAN_CLASS(compare)

# Useful class constants

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
