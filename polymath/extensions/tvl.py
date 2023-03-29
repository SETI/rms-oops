################################################################################
# polymath/extensions/tvl.py: Three-valued logic operations
################################################################################

import numpy as np
from polymath.qube import Qube

def tvl_and(self, arg, builtins=None):
    """Three-valued logic "and" operator.

    Masked values are treated as indeterminate rather than being ignored.
    These are the rules:
        - False and anything = False
        - True and True = True
        - True and Masked = Masked

    If builtins is True and the result is a single scalar True or False, it is
    returned as a Python boolean instead of an instance of Boolean. Default is
    the value specified by Qube.PREFER_BUILTIN_TYPES.
    """

    # Truth table...
    #           False       Masked      True
    # False     False       False       False
    # Masked    False       Masked      Masked
    # True      False       Masked      True

    self = Qube.BOOLEAN_CLASS.as_boolean(self)
    arg = Qube.BOOLEAN_CLASS.as_boolean(arg)

    if Qube.is_one_false(self._mask_):
        self_is_true = self._values_
        self_is_not_false = self._values_
    else:
        self_is_true = self._values_ & self.antimask
        self_is_not_false = self._values_ | self._mask_

    if Qube.is_one_false(arg._mask_):
        arg_is_true = arg._values_
        arg_is_not_false  = arg._values_
    else:
        arg_is_true = arg._values_ & arg.antimask
        arg_is_not_false  = arg._values_ | arg._mask_

    result_is_true = self_is_true & arg_is_true
    result_is_not_false = self_is_not_false & arg_is_not_false

    result_is_masked = Qube.and_(np.logical_not(result_is_true),
                                 result_is_not_false)

    result = Qube.BOOLEAN_CLASS(result_is_true, result_is_masked)

    # Convert result to a Python bool if necessary
    if builtins is None:
        builtins = Qube.PREFER_BUILTIN_TYPES

    if builtins:
        return result.as_builtin()

    return result

#===============================================================================
def tvl_or(self, arg, builtins=None):
    """Three-valued logic "or" operator.

    Masked values are treated as indeterminate rather than being ignored.
    These are the rules:
        - True or anything = True
        - False or False = False
        - False or Masked = Masked

    If builtins is True and the result is a single scalar True or False, it is
    returned as a Python boolean instead of an instance of Boolean. Default is
    the value specified by Qube.PREFER_BUILTIN_TYPES.
    """

    # Truth table...
    #           False       Masked      True
    # False     False       Masked      True
    # Masked    Masked      Masked      True
    # True      True        True        True

    self = Qube.BOOLEAN_CLASS.as_boolean(self)
    arg = Qube.BOOLEAN_CLASS.as_boolean(arg)

    if Qube.is_one_false(self._mask_):
        self_is_true = self._values_
        self_is_not_false = self._values_
    else:
        self_is_true = self._values_ & self.antimask
        self_is_not_false = self._values_ | self._mask_

    if Qube.is_one_false(arg._mask_):
        arg_is_true = arg._values_
        arg_is_not_false  = arg._values_
    else:
        arg_is_true = arg._values_ & arg.antimask
        arg_is_not_false  = arg._values_ | arg._mask_

    result_is_true = self_is_true | arg_is_true
    result_is_not_false = self_is_not_false | arg_is_not_false

    result_is_masked = Qube.and_(np.logical_not(result_is_true),
                                 result_is_not_false)

    result = Qube.BOOLEAN_CLASS(result_is_not_false, result_is_masked)

    # Convert result to a Python bool if necessary
    if builtins is None:
        builtins = Qube.PREFER_BUILTIN_TYPES

    if builtins:
        return result.as_builtin()

    return result

#===============================================================================
def tvl_any(self, axis=None, builtins=None):
    """Three-valued logic "any" operator.

    Masked values are treated as indeterminate rather than being ignored.
    These are the rules:
        - True if any unmasked value is True;
        - False if and only if all the items are False and unmasked;
        - otherwise, Masked.

    Input:
        axis        an integer axis or a tuple of axes. The any operation is
                    performed across these axes, leaving any remaining axes in
                    the returned value. If None (the default), then the any
                    operation is performed across all axes of the object.
        builtins    if True and the result is a single scalar True or False, the
                    result is returned as a Python boolean instead of an
                    instance of Boolean. Default is that specified by
                    Qube.PREFER_BUILTIN_TYPES.
    """

    self = Qube.BOOLEAN_CLASS.as_boolean(self)

    if not self._shape_:
        args = (self,)                  # make a copy

    elif isinstance(self._mask_, (bool, np.bool_)):
        args = (np.any(self._values_, axis=axis), self._mask_)

    else:
        # True where any value is True AND its antimask is True
        new_values = np.any(self._values_ & self.antimask, axis=axis)

        # Masked if any value is masked unless new_values is True
        masked_found = np.any(self._mask_, axis=axis)
        new_mask = np.logical_not(new_values) & masked_found

        args = (new_values, new_mask)

    result = Qube.BOOLEAN_CLASS(*args)

    # Convert result to a Python bool if necessary
    if builtins is None:
        builtins = Qube.PREFER_BUILTIN_TYPES

    if builtins:
        return result.as_builtin()

    return result

#===============================================================================
def tvl_all(self, axis=None, builtins=None):
    """Three-valued logic "all" operator.

    Masked values are treated as indeterminate rather than being ignored.
    These are the rules:
        - True if and only if all the items are True and unmasked.
        - False if any unmasked value is False.
        - otherwise, Masked.

    Input:
        axis        an integer axis or a tuple of axes. The all operation is
                    performed across these axes, leaving any remaining axes in
                    the returned value. If None (the default), then the all
                    operation is performed across all axes of the object.
        builtins    if True and the result is a single scalar True or False, the
                    result is returned as a Python boolean instead of an
                    instance of Boolean. Default is that specified by
                    Qube.PREFER_BUILTIN_TYPES.
    """

    self = Qube.BOOLEAN_CLASS.as_boolean(self)

    if not self._shape_:
        args = (self,)                  # make a copy

    elif isinstance(self._mask_, (bool, np.bool_)):
        args = (np.all(self._values_, axis=axis), self._mask_)

    else:
        # False where any value is False AND its antimask is True
        # Therefore, True where every value is True OR its mask is True
        new_values = np.all(self._values_ | self._mask_, axis=axis)

        # Masked where any value is masked unless new_values is False
        mask_found = np.any(self._mask_, axis=axis)
        new_mask = new_values & mask_found

        args = (new_values, new_mask)

    result = Qube.BOOLEAN_CLASS(*args)

    # Convert result to a Python bool if necessary
    if builtins is None:
        builtins = Qube.PREFER_BUILTIN_TYPES

    if builtins:
        return result.as_builtin()

    return result

#===============================================================================
def tvl_eq(self, arg, builtins=None):
    """Three-valued logic "equals" operator.

    Masked values are treated as indeterminate, so if either value is masked,
    the returned value is masked.

    If builtins is True and the result is a single scalar True or False, the
    result is returned as a Python boolean instead of an instance of Boolean.
    Default is the value specified by Qube.PREFER_BUILTIN_TYPES.
    """

    return self._tvl_op(arg, (self == arg), builtins=builtins)

#===============================================================================
def tvl_ne(self, arg, builtins=None):
    """Three-valued logic "not equal" operator.

    Masked values are treated as indeterminate, so if either value is masked,
    the returned value is masked.

    If builtins is True and the result is a single scalar True or False, the
    result is returned as a Python boolean instead of an instance of Boolean.
    Default is the value specified by Qube.PREFER_BUILTIN_TYPES.
    """

    return self._tvl_op(arg, (self != arg), builtins=builtins)

#===============================================================================
def tvl_lt(self, arg, builtins=None):
    """Three-valued logic "less than" operator.

    Masked values are treated as indeterminate, so if either value is masked,
    the returned value is masked.

    If builtins is True and the result is a single scalar True or False, the
    result is returned as a Python boolean instead of an instance of
    Boolean. Default is the value specified by Qube.PREFER_BUILTIN_TYPES.
    """

    return self._tvl_op(arg, (self < arg), builtins=builtins)

#===============================================================================
def tvl_gt(self, arg, builtins=None):
    """Three-valued logic "greater than" operator.

    Masked values are treated as indeterminate, so if either value is masked,
    the returned value is masked.

    If builtins is True and the result is a single scalar True or False, the
    result is returned as a Python boolean instead of an instance of Boolean.
    Default is the value specified by Qube.PREFER_BUILTIN_TYPES.
    """

    return self._tvl_op(arg, (self > arg), builtins=builtins)

#===============================================================================
def tvl_le(self, arg, builtins=None):
    """Three-valued logic "less than or equal to" operator.

    Masked values are treated as indeterminate, so if either value is masked,
    the returned value is masked.

    If builtins is True and the result is a single scalar True or False, the
    result is returned as a Python boolean instead of an instance of Boolean.
    Default is the value specified by Qube.PREFER_BUILTIN_TYPES.
    """

    return self._tvl_op(arg, (self <= arg), builtins=builtins)

#===============================================================================
def tvl_ge(self, arg, builtins=None):
    """Three-valued logic "greater than or equal to" operator.

    Masked values are treated as indeterminate, so if either value is masked,
    the returned value is masked.

    If builtins is True and the result is a single scalar True or False, the
    result is returned as a Python boolean instead of an instance of Boolean.
    Default is the value specified by Qube.PREFER_BUILTIN_TYPES.
    """

    return self._tvl_op(arg, (self >= arg), builtins=builtins)

#===============================================================================
def _tvl_op(self, arg, comparison, builtins=None):
    """Three-valued logic version of any boolean operator.

    Masked values are treated as indeterminate, so if either value is masked,
    the returned value is masked.

    If builtins is True and the result is a single scalar True or False, the
    result is returned as a Python boolean instead of an instance of Boolean.
    Default is the value specified by Qube.PREFER_BUILTIN_TYPES.
    """

    # Return a Python bool if appropriate
    if isinstance(comparison, bool):
        if builtins is None:
            builtins = Qube.PREFER_BUILTIN_TYPES
        if builtins:
            return comparison

        comparison = Qube.BOOLEAN_CLASS(comparison)

    # Determine arg_mask, if any
    if isinstance(arg, Qube):
        arg_mask = arg._mask_
    elif isinstance(arg, np.ma.MaskedArray):
        arg_mask = arg.mask
    else:
        arg_mask = False

    comparison._set_mask_(Qube.or_(self._mask_, arg_mask))

    return comparison

################################################################################
