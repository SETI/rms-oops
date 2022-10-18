################################################################################
# polymath/extensions/pickle.py: Support for pickling/serialization
################################################################################

import bz2
import numpy as np
import numbers
import sys

from ..qube import Qube

SINGLE_DIGITS = 6.92
DOUBLE_DIGITS = 15.65

#===============================================================================
@staticmethod
def _interpret_digits(digits):
    """Validate value as one of 'single', 'double', or a number 7-15; return the
    value converted to a number as the second value.
    """

    if digits is None:
        digits = Qube.DEFAULT_PICKLE_DIGITS

    if isinstance(digits, numbers.Real):
        digits = min(max(SINGLE_DIGITS, digits, DOUBLE_DIGITS))
        return (digits, digits)
    elif digits == 'double':
        return (digits, DOUBLE_DIGITS)
    elif digits == 'single':
        return (digits, SINGLE_DIGITS)
    else:
        raise ValueError('unrecognized precision: %s' % repr(digits))

#===============================================================================
def set_pickle_digits(self, digits=None, deriv_digits=None):
    """Set the desired number of decimal digits of precision in the storage of
    this object's floating-point values and their derivatives.

    This attribute is ignored for integer and boolean values.

    Input:
        digits          if a number is specified, this is the number of decimal
                        digits to preserve when this object is pickled. Use
                        "double" for full double precision; "single" for single
                        precision. If None or unspecified, the values of class
                        constant DEFAULT_PICKLE_DIGITS is used.

        deriv_digits    same as above, for for derivatives. If None or
                        unspecified, the value of class constant
                        DEFAULT_DERIV_PICKLE_DIGITS is used, or else the
                        precision indicated by the value above, whichever is
                        less precise.
    """

    (digits, digits_as_number) = Qube._interpret_digits(digits)
    (deriv_digits, deriv_as_number) = Qube._interpret_digits(deriv_digits)

    if deriv_as_number > digits_as_number:
        deriv_digits = digits

    self._pickle_digits_ = digits
    self._deriv_pickle_digits_ = deriv_digits

    # Recurse into derivatives
    for deriv in self.derivs.values():
        deriv._pickle_digits = deriv_digits
        deriv._deriv_pickle_digits = deriv_digits

#===============================================================================
LOG10 = np.log(10)
BYTES_INFO = [
    ('uint8' ,  8, 2**8 , np.log(2**8)  / LOG10),
    ('uint16', 16, 2**16, np.log(2**16) / LOG10),
    ('uint32', 32, 2**32, np.log(2**32) / LOG10),
]

@staticmethod
def _encode_one_float(values, digits):
    """Encode one 1-D array into a tuple for the specified digits precision.

    For a small array, the format is ('literal', values)
    For single or double precision, the format is (dtype, bz2_bytes).
    For integer encodings, it is (dtype, scale_factor, offset, bz2_bytes).
    For purely constant values, it is ('constant', value)
    """

    # Deal with a small object quickly
    if values.size <= 6:
        return ('literal', values)

    # Determine the range
    minval = np.min(values)
    maxval = np.max(values)
    span = maxval - minval

    # Handle a constant
    if span == 0.:
        return ('constant', minval)

    # Handle single and double
    if digits == 'double':
        return ('float64', bz2.compress(values))

    if digits == 'single':
        return ('float32', bz2.compress(values.astype('float32')))

    # Determine the number of digits to be accommodated by an offset
    smallest = min(abs(minval), abs(maxval))
    min_digits_from_scaling = np.log(smallest / span) / LOG10

    if min_digits_from_scaling > 1:
        digits_needed = digits - min_digits_from_scaling

        # Find the smallest int encoding to accommodate the remaining digits
        for (dtype, nbytes, power_of_two, dtype_digits) in BYTES_INFO:
          if digits_needed < dtype_digits:

            # Set the offset as the minimum; scale for an unsigned int
            scale_factor = power_of_two / span
            scale_factor *= (1. - sys.float_info.epsilon)   # avoid round up
            new_values = (scale_factor * (values - minval)).astype(dtype)

            # To reverse: values = new_values/scale_factor + minval + 0.5
            # The "+ 0.5" is to make sure the restored values are not
            # systematically smaller than they were originally

            return (dtype, 1./scale_factor, minval + 0.5,
                    bz2.compress(new_values))

    if digits <= SINGLE_DIGITS:
        return ('float32', bz2.compress(values.as_type('float32')))

    return ('float64', bz2.compress(values))

#===============================================================================
@staticmethod
def _encode_float_array(values, item, digits):
    """Complete encoding of a floating-point array.

    Creates a list of encoding tuples, one for each item. This is worthwhile
    because the individual items might have very different ranges.
    """

    # Isolate the leading and trailing items into a 2-D array
    values = values.reshape((-1,) + item)
    values = values.reshape(values.shape[0],-1)

    results = []
    for k in range(values.shape[-1]):
        # copy() to ensure contiguous
        results.append(Qube._encode_one_float(values[:,k].copy(), digits))

    return results

#===============================================================================
@staticmethod
def _decode_one_float(encoding, destination, k):
    """Decode one float from an encoded tuple, write to the destination at the
    index."""

    dtype = encoding[0]
    if dtype[0] == 'f':
        values = np.frombuffer(bz2.decompress(encoding[1]), dtype=dtype)
        destination[:,k] = values

    elif dtype[0] == 'u':
        (_, scale_factor, offset, bz2_bytes) = encoding
        unscaled = np.frombuffer(bz2.decompress(bz2_bytes), dtype=dtype)
        destination[:,k] = scale_factor * unscaled + offset

    elif dtype == 'literal':
        destination[:,k] = encoding[1]

    elif dtype == 'constant':
        (_, value) = encoding
        destination[:,k].fill(value)

    else:
        raise ValueError('unrecognized method for decoding: %s' % dtype)

#===========================================================================
@staticmethod
def _decode_float_array(encoded, shape, item):
    """Complete decoding of a floating-point array.

    Each encoding tuple in the list applies to one of the item axes.
    """

    # Create the empty buffer
    values = np.empty(shape + item, dtype='float')

    # New version with a shape and items each flattened, sharing memory.
    # Note: In the case of a Scalar, this adds one extra axis of size 1.
    destination = values.reshape(shape + (-1,))
    destination = destination.reshape((-1,) + (destination.shape[-1],))

    for k in range(destination.shape[-1]):
        Qube._decode_one_float(encoded[k], destination, k)

    return values

#===============================================================================
def __getstate__(self):

    # This state is defined by a dictionary containing many of the Qube
    # attributes. "_cache_" is removed. "_mask_", and "_values_" are replaced by
    # encodings, as discussed below. "PICKLE_VERSION" is added, with a value
    # defined by the Qube class constant. "VALS_ENCODING" and "MASK_ENCODING"
    # are also added.

    # Start with a shallow clone; save derivatives for later
    clone = self.clone(recursive=False)

    # Add the new items (or placeholders)
    clone.PICKLE_VERSION = Qube.PICKLE_VERSION
    clone.VALS_ENCODING = []
    clone.MASK_ENCODING = []

    # For a single value, nothing changes
    if np.isscalar(self._values_):
        antimask = None             # used below

    # For a fully masked object, remove the values
    elif np.all(self._mask_):
        clone._mask_ = True         # convert to bool if it's an array
        clone._values_ = None
        clone.VALS_ENCODING += ['ALL_MASKED']
        antimask = None

    # Otherwise, values is an array and not fully masked
    else:

        #---------------------------
        # Encode the mask array
        #---------------------------

        if not np.any(self._mask_):
            clone._mask_ = False    # convert to bool if it's an array

        mask_shape = np.shape(clone._mask_)

        if mask_shape:
            # If any "edges" of the mask array are all True, save the corners
            # and reduce the mask size
            corners = self.corners
            if Qube._shape_from_corners(corners) != self.shape:
                clone.MASK_ENCODING += [corners, 'CORNERS']
                clone._mask_ = self._mask_[self._slicer].copy()

            clone.MASK_ENCODING += [clone._mask_.shape, 'BZ2']
            clone._mask_ = bz2.compress(clone._mask_)

        #---------------------------
        # Encode the values array
        #---------------------------

        # Select the antimasked values; otherwise, flatten the shape axes.
        # At this point, the values array is always 2-D.
        if mask_shape:
            antimask = self.antimask
            clone._values_ = clone._values_[antimask]
            clone.VALS_ENCODING += ['ANTIMASKED']
            active_shape = (clone._values_.shape[0],)
        else:
            antimask = None
            active_shape = self._shape_

        # Floating-point arrays receive special handling for improved
        # compression
        if self.is_float():
            clone.VALS_ENCODING += [active_shape, 'FLOAT']
            clone._values_ = Qube._encode_float_array(clone._values_,
                                                      self._item_,
                                                      self.pickle_digits)

        # Otherwise, the values array is BZ2-encoded
        else:
            shape = clone._values_.shape
            dtype = clone._values_.dtype
            clone.VALS_ENCODING += [shape, dtype, 'BZ2']
            clone._values_ = bz2.compress(clone._values_)

    #---------------------------
    # Process the derivatives
    #---------------------------

    # We replace the each derivative by its __getstate__ value. However, we
    # first modify each derivative, applying the antimask, if any, and removing
    # the mask. This avoids the duplication of masks.

    for key, deriv in self._derivs_.items():
        if antimask is not None:
            new_deriv = Qube.__new__(type(deriv))
            new_values = deriv._values_[antimask]
            new_deriv.__init__(new_values, False, example=deriv)
        else:
            new_deriv = deriv.clone()
            new_deriv._mask_ = False

        clone._derivs_[key] = new_deriv     # will be pickled recursively

    return clone.__dict__

#===============================================================================
def __setstate__(self, state):

    self.__dict__ = state

    # Save copies of the encoding info if needed for debugging
    if not Qube.PICKLE_DEBUG:
        VALS_ENCODING = list(self.VALS_ENCODING)
        MASK_ENCODING = list(self.MASK_ENCODING)

    #---------------------------
    # Decode the mask
    #---------------------------

    mask_is_writable = not bool(self.MASK_ENCODING)  # True if not encoded
    while self.MASK_ENCODING:
        encoding = self.MASK_ENCODING.pop()

        if encoding == 'BZ2':
            shape = self.MASK_ENCODING.pop()
            bz2_bytes = bz2.decompress(self._mask_)
            self._mask_ = np.frombuffer(bz2_bytes, 'bool').reshape(shape)

        elif encoding == 'CORNERS':
            corners = self.MASK_ENCODING.pop()
            new_mask = np.ones(self._shape_, dtype='bool')
            slicer = Qube._slicer_from_corners(corners)
            new_mask[slicer] = self._mask_
            self._mask_ = new_mask
            mask_is_writable = True

        else:
            raise ValueError('unrecognized mask encoding: ' + str(encoding))

    # Define the antimask
    if np.shape(self._mask_):
        antimask = np.logical_not(self._mask_)
    else:
        antimask = None

    #---------------------------
    # Decode the values
    #---------------------------

    values_is_writable = not bool(self.VALS_ENCODING)   # True if not encoded
    while self.VALS_ENCODING:
        encoding = self.VALS_ENCODING.pop()

        if encoding == 'BZ2':
            dtype = self.VALS_ENCODING.pop()
            shape = self.VALS_ENCODING.pop()
            bz2_bytes = bz2.decompress(self._values_)
            self._values_ = np.frombuffer(bz2_bytes, dtype).reshape(shape)

        elif encoding == 'FLOAT':
            active_shape = self.VALS_ENCODING.pop()
            self._values_ = Qube._decode_float_array(self._values_,
                                                     active_shape,
                                                     self._item_)
            values_is_writable = True

        elif encoding == 'ANTIMASKED':
            if antimask is None:
                raise ValueError('missing antimask for decoding')
            new_values = np.empty(self._shape_ + self._item_, dtype='float')
            new_values[...] = self._default_
            new_values[antimask] = self._values_
            self._values_ = new_values
            values_is_writable = True

        elif encoding == 'ALL_MASKED':
            new_values = np.empty(self._shape_ + self._item_, dtype='float')
            new_values[...] = self._default_
            self._values_ = new_values
            values_is_writable = True

        else:
            raise ValueError('unrecognized values encoding: ' + str(encoding))

    #---------------------------
    # Set readonly status
    #---------------------------

    if self._readonly_:
        self.as_readonly()
    else:
        if not mask_is_writable:
            self._mask_ = self._mask_.copy()
        if not values_is_writable:
            self._values_ = self._values_.copy()

    #---------------------------
    # Expand the derivatives
    #---------------------------

    if antimask is not None:

      for key, deriv in self._derivs_.items():

        # Expand the derivative; share the parent object's mask
        new_values = np.empty(self._shape_ + deriv._item_, dtype='float64')
        new_values[...] = deriv._default_
        new_values[antimask] = deriv._values_

        new_deriv = Qube.__new__(type(deriv))
        new_deriv.__init__(new_values, self._mask_, example=deriv)
        if deriv._readonly_:
            new_deriv.as_readonly()

        self.insert_deriv(key, new_deriv, override=True)

    # In DEBUG mode, put the encoding info back
    if not Qube.PICKLE_DEBUG:
        self.VALS_ENCODING = VALS_ENCODING
        self.MASK_ENCODING = MASK_ENCODING

    # Otherwise, delete extraneous attributes
    else:
        del self.__dict__['PICKLE_VERSION']
        del self.__dict__['VALS_ENCODING']
        del self.__dict__['MASK_ENCODING']

################################################################################
