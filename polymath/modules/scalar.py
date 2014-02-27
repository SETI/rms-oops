################################################################################
# polymath/modules/scalar.py: Scalar subclass of PolyMath base class
#
# Mark Showalter, PDS Rings Node, SETI Institute, February 2014
################################################################################

from __future__ import division
import numpy as np
import sys
import warnings

from qube    import Qube
from units   import Units

# Maximum argument to exp()
EXP_CUTOFF = np.log(sys.float_info.max)

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

    @staticmethod
    def as_scalar(arg, recursive=True):
        """Return the argument converted to Scalar if possible.

        If recursive is True, derivatives will also be converted.
        """

        if type(arg) == Scalar: return arg

        if type(arg) == Qube.BOOLEAN_CLASS:
            return Qube.BOOLEAN_CLASS(arg).as_int()

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
            return obj.values[~obj.mask]
        else:
            obj = obj.copy()
            obj.values[obj.mask] = masked
            return obj.values

    def int(self):
        """Return an integer (floor) version of this Scalar.

        If this object already contains integers, it is returned as is.
        Otherwise, a copy is returned. Derivatives are always removed. Units
        are disallowed. If this object is read-only, the returned object will
        also be read-only.

        If the result is a single unmasked scalar, it is returned as a Python
        scalar rather than as an instance of Scalar.
        """

        Units.require_unitless(self.units)
        obj = self.without_derivs().as_int()

        return obj

    def frac(self):
        """Return an object containing the fractional components of all values.

        The returned object is an instance of the same subclass as this object.
        Derivatives are removed. If this object is read-only, the return object
        is as well.
        """

        Units.require_unitless(self.units)

        # Convert to fractional values
        if isinstance(self.values, np.ndarray):
            new_values = (self.values % 1.)
        else:
            new_values = self.values % 1.

        # Construct a new copy
        obj = Qube.__new__(type(self))
        obj.__init__(new_values, derivs={}, example=self)

        # Preserve the read-only status
        if self.readonly: obj = obj.as_readonly(nocopy='vm')

        return obj

    def sin(self, recursive=True):
        """Return the sine of each value.

        If this object is read-only, the returned object will also be read-only.

        Inputs:
            recursive   True to include the derivatives of the sine inside the
                        returned object.
        """

        Units.require_angle(self.units)
        obj = Scalar(np.sin(self.values), self.mask)

        if self.readonly:
            obj = obj.as_readonly(nocopy='vm')

        if recursive and self.derivs:
            factor = self.without_derivs().cos()
            for (key, deriv) in self.derivs.iteritems():
                obj.insert_deriv(key, factor * deriv,
                                 override=True, nocopy='vm')

        return obj

    def cos(self, recursive=True):
        """Returns the cosine of each value.

        If this object is read-only, the returned object will also be read-only.

        Inputs:
            recursive   True to include the derivatives of the cosine inside the
                        returned object.
        """

        Units.require_angle(self.units)
        obj = Scalar(np.cos(self.values), self.mask)

        if self.readonly:
            obj = obj.as_readonly(nocopy='vm')

        if recursive and self.derivs:
            factor = -self.without_derivs().sin()
            for (key, deriv) in self.derivs.iteritems():
                obj.insert_deriv(key, factor * deriv,
                                 override=True, nocopy='vm')

        return obj

    def tan(self, recursive=True):
        """Returns the tangent of each value.

        If this object is read-only, the returned object will also be read-only.

        Inputs:
            recursive   True to include the derivatives of the tangent inside
                        the returned object.
        """

        Units.require_angle(self.units)

        obj = Scalar(np.tan(self.values), self.mask)

        if self.readonly:
            obj = obj.as_readonly(nocopy='vm')

        if recursive and self.derivs:
            inv_sec_sq = self.without_derivs().cos()**(-2)
            for (key, deriv) in self.derivs.iteritems():
                obj.insert_deriv(key, inv_sec_sq * deriv,
                                 override=True, nocopy='vm')

        return obj

    def arcsin(self, recursive=True, check=True):
        """Returns the arcsine of each value.

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

            obj = Scalar(func_values, self.mask)

        if self.readonly:
            obj = obj.as_readonly(nocopy='vm')

        if recursive and self.derivs:
            x = self.without_derivs()
            factor = (1. - x*x)**(-0.5)
            for (key, deriv) in self.derivs.iteritems():
                obj.insert_deriv(key, factor * deriv,
                                 override=True, nocopy='vm')

        return obj

    def arccos(self, recursive=True, check=True):
        """Returns the arccosine of each value.

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

            obj = Scalar(func_values, self.mask)

        if self.readonly:
            obj = obj.as_readonly(nocopy='vm')

        if recursive and self.derivs:
            x = self.without_derivs()
            factor = -(1. - x*x)**(-0.5)
            for (key, deriv) in self.derivs.iteritems():
                obj.insert_deriv(key, factor * deriv,
                                 override=True, nocopy='vm')

        return obj

    def arctan(self, recursive=True):
        """Returns the arctangent of each value.

        If this object is read-only, the returned object will also be read-only.

        Inputs:
            recursive   True to include the derivatives of the arctangent
                        inside the returned object.
        """

        Units.require_unitless(self.units)

        obj = Scalar(np.arctan(self.values), self.mask)

        if self.readonly:
            obj = obj.as_readonly(nocopy='vm')

        if recursive and self.derivs:
            factor = 1. / (1. + self.without_derivs()**2)
            for (key, deriv) in self.derivs.iteritems():
                obj.insert_deriv(key, factor * deriv,
                                 override=True, nocopy='vm')

        return obj

    def arctan2(self, arg, recursive=True):
        """Returns the four-quadrant value of arctan2(y,x).

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

        if y.readonly and x.readonly:
            obj = obj.as_readonly(nocopy='vm')

        if recursive and (x.derivs or y.derivs):
            x_wod = x.without_derivs()
            y_wod = y.without_derivs()
            denom_inv = (x_wod**2 + y_wod**2).reciprocal()

            new_derivs = {}
            for (key, y_deriv) in y.derivs.iteritems():
                new_derivs[key] = x_wod * denom_inv * y_deriv

            for (key, x_deriv) in x.derivs.iteritems():
                if key in new_derivs:
                    new_derivs[key] -= y_wod * denom_inv * x_deriv
                else:
                    new_derivs[key] = -y_wod * denom_inv * x_deriv

            obj.insert_derivs(new_derivs, override=True, nocopy='vm')

        return obj

    def sqrt(self, recursive=True, check=True):
        """Returns the square root, masking imaginary values.

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

        obj = Scalar(sqrt_vals, no_negs.mask, Units.sqrt_units(no_negs.units))

        if self.readonly:
            obj = obj.as_readonly(nocopy='vm')

        if recursive and no_negs.derivs:
            factor = Scalar(0.5 / sqrt_vals, no_negs.mask)
            for (key, deriv) in self.derivs.iteritems():
                obj.insert_deriv(key, factor * deriv,
                                 override=True, nocopy='vm')

        return obj

    def log(self, recursive=True, check=True):
        """Returns the log, masking undefined values.

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

        obj = Scalar(log_values, no_negs.mask)

        if self.readonly:
            obj = obj.as_readonly(nocopy='vm')

        if recursive and no_negs.derivs:
            for (key, deriv) in self.derivs.iteritems():
                obj.insert_deriv(key, deriv / no_negs, nocopy='vm')

        return obj

    def exp(self, recursive=True, check=False):
        """Returns the e raised to the power of each value.

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

        obj = Scalar(exp_values, no_oflow.mask)

        if self.readonly:
            obj = obj.as_readonly(nocopy='vm')

        if recursive and self.derivs:
            for (key, deriv) in self.derivs.iteritems():
                obj.insert_deriv(key, deriv * exp_values,
                                 override=True, nocopy='vm')

        return obj

    def sign(self):
        """Returns the sign of each value as +1, -1 or 0.

        If this object is read-only, the returned object will also be read-only.
        """

        obj = Scalar(np.sign(self.values), self.mask)
        if self.readonly: obj = obj.as_readonly(nocopy='vm')

        return obj

    def max(self):
        """Returns the maximum of the unmasked values."""

        if np.all(self.mask):
            if self.is_float():
                return Scalar(1., True, self.units)
            else:
                return Scalar(1, True, self.units)

        if not np.any(self.mask):
            maxval = np.max(self.values)
        else:
            maxval = np.max(self.values[~self.mask])

        if self.units is None or self.units == Units.UNITLESS:
            if self.is_float():
                return float(maxval)
            else:
                return int(maxval)
        else:
            return Scalar(maxval, False, self.units)

    def min(self):
        """Returns the minimum of the unmasked values."""

        if np.all(self.mask):
            if self.is_float():
                return Scalar(1., True, self.units)
            else:
                return Scalar(1, True, self.units)

        if not np.any(self.mask):
            minval = np.min(self.values)
        else:
            minval = np.min(self.values[~self.mask])

        if self.units is None or self.units == Units.UNITLESS:
            if self.is_float():
                return float(minval)
            else:
                return int(minval)
        else:
            return Scalar(minval, False, self.units)

    def mean(self):
        """Returns the mean of the unmasked values."""

        if np.all(self.mask):
            if self.is_float():
                return Scalar(1., True, self.units)
            else:
                return Scalar(1, True, self.units)

        if not np.any(self.mask):
            meanval = np.mean(self.values)
        else:
            meanval = np.mean(self.values[~self.mask])

        if self.units is None or self.units == Units.UNITLESS:
            if self.is_float() or (meanval % 1. != 0):
                return float(meanval)
            else:
                return int(meanval)
        else:
            return Scalar(meanval, False, self.units)

    def sum(self, recursive=False):
        """Returns the sum of the unmasked values.

        If recursive, the derivatives are summed too. """

        if np.all(self.mask):
            if self.is_float():
                return Scalar(0., True, self.units)
            else:
                return Scalar(0, True, self.units)

        if not np.any(self.mask):
            sumval = np.sum(self.values)
        else:
            sumval = np.sum(self.values[~self.mask])

        if self.units is None or self.units == Units.UNITLESS:
            if self.is_float():
                return float(sumval)
            else:
                return int(sumval)
        else:
            return Scalar(sumval, False, self.units)

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
          with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                denom_inv_values = 1. / self.values
                denom_inv_mask = self.mask
            except:
                raise ValueError('divide by zero encountered in reciprocal()')

        else:
            denom = self.mask_where_eq(0,1)
            denom_inv_values = 1. / denom.values
            denom_inv_mask = denom.mask

        # Construct the object
        obj = Qube.__new__(type(self))
        obj.__init__(denom_inv_values, denom_inv_mask,
                     Units.units_power(self.units, -1),
                     derivs={}, example=self)

        # Inherit the read-only status from the source
        if self.readonly:
            obj = obj.as_readonly()

        # Fill in derivatives if necessary
        if recursive and self.derivs:
            factor = -obj*obj       # At this point it has no derivs
            for (key,deriv) in self.derivs.iteritems():
                obj.insert_deriv(key, factor * deriv, override=True,
                                 nocopy='vm')

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

        arg_wasnt_qube = not isinstance(arg, Qube)

        arg = Scalar.as_scalar(arg)
        Units.require_compatible(self.units, arg.units)

        compare = (self.values < arg.values)
        mask = self.mask | arg.mask

        if np.shape(compare) == ():
            if mask: compare = False
            return compare

        if np.shape(mask) == ():
            if mask: compare.fill(False)
        else:
            compare[mask] = False

        obj = Qube.BOOLEAN_CLASS(compare)
        if self.readonly and (arg.readonly or arg_wasnt_qube):
            obj = obj.as_readonly(nocopy='vm')

        return obj

    # (>) operator
    def __gt__(self, arg):
        """True where self > arg; also False where either value is masked."""

        arg_wasnt_qube = not isinstance(arg, Qube)

        arg = Scalar.as_scalar(arg)
        Units.require_compatible(self.units, arg.units)

        compare = (self.values > arg.values)
        mask = self.mask | arg.mask

        if np.shape(compare) == ():
            if mask: compare = False
            return compare

        if np.shape(mask) == ():
            if mask: compare.fill(False)
        else:
            compare[mask] = False

        obj = Qube.BOOLEAN_CLASS(compare)
        if self.readonly and (arg.readonly or arg_wasnt_qube):
            obj = obj.as_readonly(nocopy='vm')

        return obj

    # (<=) operator
    def __le__(self, arg):
        """True where self <= arg or where both values are masked."""

        arg_wasnt_qube = not isinstance(arg, Qube)

        arg = Scalar.as_scalar(arg)
        Units.require_compatible(self.units, arg.units)

        compare = (self.values <= arg.values)
        one_masked = self.mask ^ arg.mask
        both_masked = self.mask & arg.mask

        if np.shape(compare) == ():
            if both_masked: compare = True
            if one_masked: compare = False
            return compare

        if np.shape(both_masked) == ():
            if both_masked: compare.fill(True)
            if one_masked: compare.fill(False)
        else:
            compare[both_masked] = True
            compare[one_masked] = False

        obj = Qube.BOOLEAN_CLASS(compare)
        if self.readonly and (arg.readonly or arg_wasnt_qube):
            obj = obj.as_readonly(nocopy='vm')

        return obj

    # (>=) operator
    def __ge__(self, arg):
        """True where self >= arg or where both values are masked."""

        arg_wasnt_qube = not isinstance(arg, Qube)

        arg = Scalar.as_scalar(arg)
        Units.require_compatible(self.units, arg.units)

        compare = (self.values >= arg.values)
        one_masked = self.mask ^ arg.mask
        both_masked = self.mask & arg.mask

        if np.shape(compare) == ():
            if both_masked: compare = True
            if one_masked: compare = False
            return compare

        if np.shape(both_masked) == ():
            if both_masked: compare.fill(True)
            if one_masked: compare.fill(False)
        else:
            compare[both_masked] = True
            compare[one_masked] = False

        obj = Qube.BOOLEAN_CLASS(compare)
        if self.readonly and (arg.readonly or arg_wasnt_qube):
            obj = obj.as_readonly(nocopy='vm')

        return obj

# Useful class constants

Scalar.ONE = Scalar(1)
Scalar.ZERO = Scalar(0)
Scalar.MASKED = Scalar(1, True)

################################################################################
# Once the load is complete, we can fill in a reference to the Scalar class
# inside the Qube object.
################################################################################

Qube.SCALAR_CLASS = Scalar

################################################################################
