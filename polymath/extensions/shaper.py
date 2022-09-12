################################################################################
# polymath/extensions/shaper.py: re-shaping operations
################################################################################

import numpy as np
import numbers
from ..qube import Qube

def reshape(self, shape, recursive=True):
    """A shallow copy of the object with a new leading shape.

    Input:
        shape       a tuple defining the new leading shape. A value of -1 can
                    appear at one location in the new shape, and the size of
                    that shape will be determined based on this object's size.
        recursive   True to apply the same shape to the derivatives.
                    Otherwise, derivatives are deleted from the returned object.
    """

    if np.isscalar(shape):
        shape = (shape,)

    if shape == self._shape_:
        return self

    if np.isscalar(self._values_):
        new_values = np.array([self._values_]).reshape(shape)
    else:
        new_values = self._values_.reshape(shape + self.item)

    if np.isscalar(self._mask_):
        new_mask = self._mask_
    else:
        new_mask = self._mask_.reshape(shape)

    obj = Qube.__new__(type(self))
    obj.__init__(new_values, new_mask, example=self)
    obj._readonly_ = self._readonly_

    if recursive:
        for (key, deriv) in self._derivs_.items():
            obj.insert_deriv(key, deriv.reshape(shape, False))

    return obj

#===============================================================================
def flatten(self, recursive=True):
    """A shallow copy of the object flattened to one dimension."""

    if len(self._shape_) < 2:
        return self

    count = np.product(self._shape_)
    return self.reshape((count,), recursive)

#===============================================================================
def swap_axes(self, axis1, axis2, recursive=True):
    """A shallow copy of the object with two leading axes swapped.

    Input:
        axis1       the first index of the swap. Negative indices are relative
                    to the last index before the numerator items begin.
        axis2       the second index of the swap.
        recursive   True to perform the same swap on the derivatives.
                    Otherwise, derivatives are deleted from the returned object.
    """

    # Validate first axis
    len_shape = len(self._shape_)
    if axis1 < 0:
        a1 = axis1 + len_shape
    else:
        a1 = axis1

    if a1 < 0 or a1 >= len_shape:
        raise ValueError('axis1 argument out of range (%d,%d): %d' %
                         (-len_shape, len_shape, axis1))

    # Validate second axis
    if axis2 < 0:
        a2 = axis2 + len_shape
    else:
        a2 = axis2

    if a2 < 0 or a2 >= len_shape:
        raise ValueError('axis2 argument out of range (%d,%d): %d' %
                         (-len_shape, len_shape, axis2))

    if a1 == a2:
        return self

    new_values = self._values_.swapaxes(a1, a2)

    if np.isscalar(self._mask_):
        new_mask = self._mask_
    else:
        new_mask = self._mask_.swapaxes(a1, a2)

    obj = Qube.__new__(type(self))
    obj.__init__(new_values, new_mask, example=self)
    obj._readonly_ = self._readonly_

    if recursive:
        for (key, deriv) in self._derivs_.items():
            obj.insert_deriv(key, deriv.swap_axes(a1, a2, False))

    return obj

#===============================================================================
def roll_axis(self, axis, start=0, recursive=True, rank=None):
    """A shallow copy of the object with the specified axis rolled to a new
    position.

    Input:
        axis        the axis to roll.
        start       the axis will be rolled to fall in front of this axis;
                    default is zero.
        recursive   True to perform the same axis roll on the derivatives.
                    Otherwise, derivatives are deleted from the returned object.
        rank        rank to assume for the object, which could be larger then
                    len(self.shape) because of broadcasting.
    """

    # Validate the rank
    len_shape = len(self._shape_)
    if rank is None:
        rank = len_shape
    if rank < len_shape:
        raise ValueError('roll rank %d is too small for object' % rank)

    if len_shape == 0:
        rank = 1

    # Identify the axis to roll, which could be negative
    if axis < 0:
        a1 = axis + rank
    else:
        a1 = axis

    if a1 < 0 or a1 >= rank:
        raise ValueError('roll axis %d out of range' % axis)

    # Identify the start axis, which could be negative
    if start < 0:
        a2 = start + rank
    else:
        a2 = start

    if a2 < 0 or a2 >= rank + 1:
        raise ValueError('roll axis %d out of range' % start)

    # No need to modify a shapeless object
    if not self._shape_:
        return self

    # Add missing axes if necessary
    if len_shape < rank:
        self = self.reshape((rank - len_shape) * (1,) + self._shape_,
                            recursive=recursive)

    # Roll the values and mask of the object
    new_values = np.rollaxis(self._values_, a1, a2)

    if np.shape(self._mask_):
        new_mask = np.rollaxis(self._mask_, a1, a2)
    else:
        new_mask = self._mask_

    obj = Qube.__new__(type(self))
    obj.__init__(new_values, new_mask, example=self)
    obj._readonly_ = self._readonly_

    if recursive:
        for (key, deriv) in self._derivs_.items():
            obj.insert_deriv(key, deriv.roll_axis(a1, a2, False, rank))

    return obj

#===============================================================================
@staticmethod
def stack(*args, **keywords):
    """Stack objects of the same class into one with a new leading axis.

    Inputs:
        args        any number of Scalars or arguments that can be casted to
                    Scalars. They need not have the same shape, but it must be
                    possible to cast them to the same shape. A value of None is
                    converted to a zero-valued Scalar that matches the
                    denominator shape of the other arguments.

        recursive   True to include all the derivatives. The returned object
                    will have derivatives representing the union of all the
                    derivatives found amongst the scalars. Default is True.

    Note that the 'recursive' input is handled as a keyword argument in order to
    distinguish it from the Qube inputs.
    """

    # Search the keywords for "recursive"
    recursive = True
    if 'recursive' in keywords:
        recursive = keywords['recursive']
        del keywords['recursive']

    # No other keyword is allowed
    if keywords:
      raise TypeError(('stack() got an unexpected keyword argument ' +
                       '"%s"') % keywords.keys()[0])

    args = list(args)

    # Get the type and units if any
    # Only use class Qube if no suitable subclass was found
    floats_found = False
    ints_found = False
    bools_found = False

    float_arg = None
    int_arg = None
    bool_arg = None

    units = None
    denom = None
    subclass_indx = None

    for (i,arg) in enumerate(args):
        if arg is None:
            continue

        qubed = False
        if not isinstance(arg, Qube):
            arg = Qube(arg)
            args[i] = arg
            qubed = True

        if denom is None:
            denom = arg._denom_
        elif denom != arg._denom_:
            raise ValueError('incompatible denominators in stack()')

        if arg.is_float():
            floats_found = True
            if float_arg is None or not qubed:
                float_arg = arg
                subclass_indx = i
        elif arg.is_int() and float_arg is None:
            ints_found = True
            if int_arg is None or not qubed:
                int_arg = arg
                subclass_indx = i
        elif arg.is_bool() and int_arg is None and float_arg is None:
            bools_found = True
            if bool_arg is None or not qubed:
                bool_arg = arg
                subclass_indx = i

        if arg._units_ is not None:
            if units is None:
                units = arg._units_
            else:
                arg.confirm_units(units)

    drank = len(denom)

    # Convert to subclass and type
    for (i,arg) in enumerate(args):
        if arg is None:                 # Used as placehold for derivs
            continue

        args[i] = args[subclass_indx].as_this_type(arg, recursive=recursive,
                                                   coerce=False)

    # Broadcast all inputs into a common shape
    args = Qube.broadcast(*args, recursive=True)

    # Determine what type of mask is needed:
    mask_true_found = False
    mask_false_found = False
    mask_array_found = False
    for arg in args:
        if arg is None:
            continue
        elif Qube.is_one_true(arg._mask_):
            mask_true_found = True
        elif Qube.is_one_false(arg._mask_):
            mask_false_found = True
        else:
            mask_array_found = True

    # Construct the mask
    if  mask_array_found or (mask_false_found and mask_true_found):
        mask = np.zeros((len(args),) + args[subclass_indx].shape,
                        dtype=np.bool_)
        for i in range(len(args)):
            if args[i] is None:
                mask[i] = False
            else:
                mask[i] = args[i]._mask_
    else:
        mask = mask_true_found

    # Construct the array
    if floats_found:
        dtype = np.float_
    elif ints_found:
        dtype = np.int_
    else:
        dtype = np.bool_

    values = np.empty((len(args),) + np.shape(args[subclass_indx]._values_),
                      dtype=dtype)
    for i in range(len(args)):
        if args[i] is None:
            values[i] = 0
        else:
            values[i] = args[i]._values_

    # Construct the result
    result = Qube.__new__(type(args[subclass_indx]))
    result.__init__(values, mask, units=units, drank=drank)

    # Fill in derivatives if necessary
    if recursive:
        keys = []
        for arg in args:
            if arg is None:
                continue
            keys += arg.derivs.keys()

        keys = set(keys)        # remove duplicates

        derivs = {}
        for key in keys:
            deriv_list = []
            for arg in args:
                if arg is None:
                    deriv_list.append(None)
                else:
                    deriv_list.append(arg.derivs.get(key, None))

            derivs[key] = Qube.stack(*deriv_list, recursive=False)

        result.insert_derivs(derivs)

    return result

################################################################################
