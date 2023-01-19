################################################################################
# polymath/extensions/item_ops.py: item restructuring operations
################################################################################

import numpy as np
from polymath.qube import Qube

def extract_numer(self, axis, index, classes=(), recursive=True):
    """An object extracted from one numerator axis.

    Input:
        axis        the item axis from which to extract a slice.
        index       the index value at which to extract the slice.
        classes     a single class or list or tuple of classes. The class of the
                    object returned will be the first suitable class in the
                    list. Otherwise, a generic Qube object will be returned.
        recursive   True to include matching slices of the derivatives in the
                    returned object; otherwise, the returned object will not
                    contain derivatives.
    """

    # Position axis from left
    if axis >= 0:
        a1 = axis
    else:
        a1 = axis + self._nrank_
    if a1 < 0 or a1 >= self._nrank_:
        raise ValueError('axis is out of range (%d,%d): %d',
                         (-self._nrank_, self._nrank_, axis))
    k1 = len(self._shape_) + a1

    # Roll this axis to the beginning and slice it out
    new_values = np.rollaxis(self._values_, k1, 0)
    new_values = new_values[index]

    # Construct and cast
    obj = Qube(new_values, self._mask_, nrank=self._nrank_ - 1,
               example=self)
    obj = obj.cast(classes)
    obj._readonly_ = self._readonly_

    # Slice the derivatives if necessary
    if recursive:
        for (key, deriv) in self._derivs_.items():
            obj.insert_deriv(key, deriv.extract_numer(a1, index, classes,
                                                      False))

    return obj

#===============================================================================
def slice_numer(self, axis, index1, index2, classes=(), recursive=True):
    """An object sliced from one numerator axis.

    Input:
        axis        the item axis from which to extract a slice.
        index1      the starting index value at which to extract the slice.
        index2      the ending index value at which to extract the slice.
        classes     a single class or list or tuple of classes. The class of the
                    object returned will be the first suitable class in the
                    list. Otherwise, a generic Qube object will be returned.
        recursive   True to include matching slices of the derivatives in the
                    returned object; otherwise, the returned object will not
                    contain derivatives.
    """

    # Position axis from left
    if axis >= 0:
        a1 = axis
    else:
        a1 = axis + self._nrank_
    if a1 < 0 or a1 >= self._nrank_:
        raise ValueError('axis is out of range (%d,%d): %d',
                         (-self._nrank_, self._nrank_, axis))
    k1 = len(self._shape_) + a1

    # Roll this axis to the beginning and slice it out
    new_values = np.rollaxis(self._values_, k1, 0)
    new_values = new_values[index1:index2]
    new_values = np.rollaxis(new_values, 0, k1+1)

    # Construct and cast
    obj = Qube(new_values, self._mask_, example=self)
    obj = obj.cast(classes)
    obj._readonly_ = self._readonly_

    # Slice the derivatives if necessary
    if recursive:
        for (key, deriv) in self._derivs_.items():
            obj.insert_deriv(key, deriv.slice_numer(a1, index1, index2,
                                                    classes, False))

    return obj

################################################################################
# Numerator shaping operations
################################################################################

def transpose_numer(self, axis1=0, axis2=1, recursive=True):
    """A copy of this object with two numerator axes transposed.

    Inputs:
        axis1       the first axis to transpose from among the numerator axes.
                    Negative values count backward from the last numerator axis.
        axis2       the second axis to transpose.
        recursive   True to transpose the same axes of the derivatives;
                    False to return an object without derivatives.
    """

    len_shape = len(self._shape_)

    # Position axis1 from left
    if axis1 >= 0:
        a1 = axis1
    else:
        a1 = axis1 + self._nrank_
    if a1 < 0 or a1 >= self._nrank_:
        raise ValueError('first axis is out of range (%d,%d): %d',
                         (-self._nrank_, self._nrank_, axis1))
    k1 = len_shape + a1

    # Position axis2 from item left
    if axis2 >= 0:
        a2 = axis2
    else:
        a2 = axis2 + self._nrank_
    if a2 < 0 or a2 >= self._nrank_:
        raise ValueError('second axis out of range (%d,%d): %d',
                         (-self._nrank_, self._nrank_, axis2))
    k2 = len_shape + a2

    # Swap the axes
    new_values = np.swapaxes(self._values_, k1, k2)

    # Construct the result
    obj = Qube.__new__(type(self))
    obj.__init__(new_values, self._mask_, example=self)
    obj._readonly_ = self._readonly_

    if recursive:
        for (key, deriv) in self._derivs_.items():
            obj.insert_deriv(key, deriv.transpose_numer(a1, a2, False))

    return obj

#===============================================================================
def reshape_numer(self, shape, classes=(), recursive=True):
    """This object with a new shape for numerator items.

    Input:
        shape       the new shape.
        classes     a single class or list or tuple of classes. The class of the
                    object returned will be the first suitable class in the
                    list. Otherwise, a generic Qube object will be returned.
        recursive   True to reshape the derivatives in the same way;
                    otherwise, the returned object will not contain derivatives.
    """

    # Validate the shape
    shape = tuple(shape)
    if self.nsize != int(np.prod(shape)):
        raise ValueError('item size must be unchanged: %s, %s' %
                         (str(self._numer_), str(shape)))

    # Reshape
    full_shape = self._shape_ + shape + self._denom_
    new_values = np.asarray(self._values_).reshape(full_shape)

    # Construct and cast
    obj = Qube(new_values, self._mask_, nrank=len(shape), example=self)
    obj = obj.cast(classes)
    obj._readonly_ = self._readonly_

    # Reshape the derivatives if necessary
    if recursive:
      for (key, deriv) in self._derivs_.items():
        obj.insert_deriv(key, deriv.reshape_numer(shape, classes, False))

    return obj

#===============================================================================
def flatten_numer(self, classes=(), recursive=True):
    """This object with a new numerator shape such that nrank == 1.

    Input:
        classes     a single class or list or tuple of classes. The class of the
                    object returned will be the first suitable class in the
                    list. Otherwise, a generic Qube object will be returned.
        recursive   True to include matching slices of the derivatives in the
                    returned object; otherwise, the returned object will not
                    contain derivatives.
    """

    return self.reshape_numer((self.nsize,), classes, recursive)

################################################################################
# Denominator shaping operations
################################################################################

def transpose_denom(self, axis1=0, axis2=1):
    """A copy of this object with two denominator axes transposed.

    Inputs:
        axis1       the first axis to transpose from among the denominator axes.
                    Negative values count backward from the last axis.
        axis2       the second axis to transpose.
    """

    len_shape = len(self._shape_)

    # Position axis1 from left
    if axis1 >= 0:
        a1 = axis1
    else:
        a1 = axis1 + self._drank_
    if a1 < 0 or a1 >= self._drank_:
        raise ValueError('first axis is out of range (%d,%d): %d',
                         (-self._drank_, self._drank_, axis1))
    k1 = len_shape + self._nrank_ + a1

    # Position axis2 from item left
    if axis2 >= 0:
        a2 = axis2
    else:
        a2 = axis2 + self._drank_
    if a2 < 0 or a2 >= self._drank_:
        raise ValueError('second axis out of range (%d,%d): %d',
                         (-self._drank_, self._drank_, axis2))
    k2 = len_shape + self._nrank_ + a2

    # Swap the axes
    new_values = np.swapaxes(self._values_, k1, k2)

    # Construct the result
    obj = Qube.__new__(type(self))
    obj.__init__(new_values, self._mask_, example=self)
    obj._readonly_ = self._readonly_

    return obj

#===============================================================================
def reshape_denom(self, shape):
    """This object with a new shape for denominator items.

    Input:
        shape       the new denominator shape.
    """

    # Validate the shape
    shape = tuple(shape)
    if self.dsize != int(np.prod(shape)):
        raise ValueError('denominator size must be unchanged: %s, %s' %
                         (str(self._denom_), str(shape)))

    # Reshape
    full_shape = self._shape_ + self._numer_ + shape
    new_values = np.asarray(self._values_).reshape(full_shape)

    # Construct and cast
    obj = Qube.__new__(type(self))
    obj.__init__(new_values, self._mask_, drank=len(shape), example=self)
    obj._readonly_ = self._readonly_

    return obj

#===============================================================================
def flatten_denom(self):
    """This object with a new denominator shape such that drank == 1.
    """

    return self.reshape_denom((self.dsize,))

################################################################################
# Numerator/denominator operations
################################################################################

def join_items(self, classes):
    """The object with denominator axes joined to the numerator.

    Derivatives are removed.

    Input:
        classes     either a single subclass of Qube or a list or tuple of
                    subclasses. The returned object will be an instance of the
                    first suitable subclass in the list.
    """

    if not self._drank_:
        return self.wod

    obj = Qube(self._values_, self._mask_,
               nrank=(self._nrank_ + self._drank_), drank=0,
               example=self)
    obj = obj.cast(classes)
    obj._readonly_ = self._readonly_

    return obj

#===============================================================================
def split_items(self, nrank, classes):
    """The object with numerator axes converted to denominator axes.

    Derivatives are removed.

    Input:
        nrank       number of numerator axes to retain.
        classes     either a single subclass of Qube or a list or tuple of
                    subclasses. The returned object will be an instance of the
                    first suitable subclass in the list.
    """

    obj = Qube(self._values_, self._mask_,
               nrank=nrank, drank=(self._rank_ - nrank),
               example=self)
    obj = obj.cast(classes)
    obj._readonly_ = self._readonly_

    return obj

#===============================================================================
def swap_items(self, classes):
    """A new object with the numerator and denominator axes exchanged.

    Derivatives are removed.

    Input:
        classes     either a single subclass of Qube or a list or tuple of
                    subclasses. The returned object will be an instance of the
                    first suitable subclass in the list.
    """

    new_values = self._values_
    len_shape = new_values.ndim

    for r in range(self._nrank_):
        new_values = np.rollaxis(new_values, -self._drank_-1, len_shape)

    obj = Qube(new_values, self._mask_,
               nrank=self._drank_, drank=self._nrank_, example=self)
    obj = obj.cast(classes)
    obj._readonly_ = self._readonly_

    return obj

#===============================================================================
def chain(self, arg):
    """Chain multiplication of this derivative by another.

    Returns the denominator of the first object times the numerator of the
    second argument. The result will be an instance of the same class. This
    operation is never recursive.

    Inputs:
        arg         the right-hand term in the chain multiplication.
    """

    left = self.flatten_denom().join_items(Qube)
    right = arg.flatten_numer(Qube)

    return Qube.dot(left, right, -1, 0, type(self), False)

################################################################################
