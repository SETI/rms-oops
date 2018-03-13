################################################################################
# polymath/qube.py: Base class for all PolyMath subclasses.
#
# Mark Showalter, PDS Ring-Moon Systems Node, SETI Institute
################################################################################

from __future__ import division
import numpy as np
import numbers
import sys

from units import Units

class Qube(object):
    """The base class for all PolyMath subclasses.

    The PolyMath subclasses, e.g., Scalar, Vector3, Matrix3, etc., define one
    or more possibly multidimensional items. Unlike NumPy ndarrays, this class
    makes a clear distinction between the dimensions associated with the items
    and any additional, leading dimensions that define an array of such items.

    The "shape" is defined by the leading axes only, so a 2x2 array of 3x3
    matrices would have shape (2,2,3,3) according to NumPy but has shape (2,2)
    according to PolyMath. Standard NumPy rules of broadcasting apply, but only
    on the array dimensions, not on the item dimensions. In other words, you can
    multiply a (2,2) array of 3x3 matrices by a (5,1,2) array of 3-vectors,
    yielding a (5,2,2) array of 3-vectors. 

    PolyMath objects are designed as lightweight wrappers on NumPy ndarrays.
    All standard mathematical operators and indexing/slicing options are
    defined. One can generally mix PolyMath arithmetic with scalars, NumPy
    ndarrays, NumPy MaskedArrays, or anything array-like.

    In every object, a boolean mask is maintained in order to identify undefined
    array elements. Operations that would otherwise raise errors such as 1/0 and
    sqrt(-1) are masked out so that run-time errors can be avoided.

    PolyMath objects also support embedded units using the Units class. However,
    the internal values in a PolyMath object are always held in standard units
    of kilometers, seconds and radians, or arbitrary combinations thereof. The
    units are primarily used for input and output.

    PolyMath objects can be either read-only or read-write. Read-only objects
    are prevented from modification to the extent that Python makes this
    possible. Operations on read-only objects should always return read-only
    objects.

    PolyMath objects can track associated derivatives and partial derivatives,
    which are represented by other PolyMath objects. Mathematical operations
    generally carry all derivatives along so that, for example, if x.d_dt is the
    derivative of x with respect to t, then x.sin().d_dt will be the derivative
    of sin(x) with respect to t.

    The denominators of partial derivatives are represented by splitting the
    item axes into numerator and denominator axes. As a result, for example, the
    partial derivatives of a Vector3 object (item shape (3,)) with respect to a
    Pair (item shape (2,)) will have overall item shape (3,2).

    The PolyMath subclasses generally do not constrain the shape of the
    denominator, just the numerator. As a result, the aforementioned partial
    derivatives can still be represented by a Vector3 object.

    The following internal attributes are used:
        __shape_    a tuple representing the leading axes, which are not
                    considered part of the items.

        __rank_     the number of axes belonging to the items.
        __nrank_    the number of numerator axes associated with the items.
        __drank_    the number of denominator axes associated with the items.

        __item_     a tuple representing the shape of the individual items.
        __numer_    a tuple representing the shape of the numerator items.
        __denom_    a tuple representing the shape of the denominator items.

        __values_   the array's data as a NumPy array or a Python scalar. The
                    shape of this array is object.shape + object.item. If the
                    object has units, then the values are in in default units
                    instead.
        __mask_     the array's mask as a NumPy boolean array. The array value
                    is True if the Array value at the same location is masked.
                    A scalar value of False indicates that the entire object is
                    unmasked; a scalar value of True indicates that it is
                    entirely masked.
        __units_    the units of the array, if any. None indicates no units.
        __derivs_   a dictionary of the names and values of any derivatives,
                    represented by additional PolyMath objects.

    For each of these, there exists a read-only property that has the same name
    minus the leading and trailing underscores.

    Additional attributes are filled in as needed
        __readonly_ True if the object cannot (or at least should not) be
                    modified. A determined user can probably alter a read-only
                    object, but the API makes this more difficult. Initially
                    False.

    Every instance also has these read-only properties:
        size        the number of elements in the shape.
        isize       the number of elements in the item array.
        nsize       the number of elements in the numerator item array.
        dsize       the number of elements in the denominator item array.

    """

    # This prevents binary operations of the form:
    #   <np.ndarray> <op> <Qube>
    # from executing the ndarray operation instead of the polymath operation
    __array_priority__ = 1

    # Default class constants, to be overridden as needed by subclasses...
    NRANK = None        # the number of numerator axes; None to leave this
                        # unconstrained.
    NUMER = None        # shape of the numerator; None to leave unconstrained.

    FLOATS_OK = True    # True to allow floating-point numbers.
    INTS_OK = True      # True to allow integers.
    BOOLS_OK = False    # True to allow booleans.

    UNITS_OK = True     # True to allow units; False to disallow them.
    MASKS_OK = True     # True to allow masks; False to disallow them.
    DERIVS_OK = True    # True to disallow derivatives; False to allow them.

    def __new__(subtype, *values, **keywords):
        """"Create a new, un-initialized object."""

        return object.__new__(subtype)

    def __init__(self, arg=None, mask=None, units=None, derivs=None,
                       nrank=None, drank=None, example=None, default=None):
        """Default constructor.

        arg         an object to define the numeric value(s) of the returned 
                    object. It could be another PolyMath object, a number, a
                    Numpy ndarray, a NumPy MaskedArray, or anything array-like.
                    If this object is read-only, then the returned object will
                    be entirely read-only. Otherwise, the object will be
                    read-writable. The values are generally given in standard
                    units of km, seconds and radians, regardless of the
                    specified units.

        mask        the mask for the object, as a single boolean, an array, or
                    anything array-like. Use None (the default) to copy the
                    value from the example object, or False (unmasked)
                    otherwise.

        units       the units of the object. Use None to infer the units from
                    the example object; use False to suppress units.

        derivs      a dictionary of derivatives represented as PolyMath objects.
                    Use None to employ a copy of the derivs attribute of the
                    example object, or else {} for no derivatives.

        nrank       optionally, the number of numerator axes in the returned
                    object; None to derive the rank from the input data and/or
                    the subclass.

        drank       optionally, the number of denominator axes in the returned
                    object; None to derive it from the input data and/or the
                    subclass.

        example     optionally, another Qube object from which to copy any input
                    arguments that have not been explicitly specified.

        default     value to use where masked. Typically a nonzero Qube constant
                    that will not "break" most arithmetic calculations. Default
                    is None, in which case the class constant DEFAULT_VALUE is
                    used, or else a reasonable value is constructed.
        """

        if type(arg) in (list, tuple):
            arg = np.array(arg)

        if type(mask) in (list, tuple):
            mask = np.array(mask)

        if self.NRANK is not None: nrank = self.NRANK

        # Interpret the example
        if example is not None:
            if arg     is None: arg     = example.__values_
            if mask    is None: mask    = example.__mask_
            if units   is None: units   = example.__units_
            if derivs  is None: derivs  = example.__derivs_
            if nrank   is None: nrank   = example.__nrank_
            if drank   is None: drank   = example.__drank_
            if default is None: default = example.__default_

        # Interpret the arg if it is a PolyMath object
        if isinstance(arg, Qube):
            if (type(self).__name__ == 'Empty') ^ \
               (type(arg).__name__ == 'Empty'):
                raise TypeError('class %s cannot be converted to class %s' %
                                 (type(arg).__name__, type(self).__name__))
            obj = arg
            arg = obj.__values_

            if mask is False or mask is None:
                mask = obj.__mask_
            else:
                mask = mask | obj.__mask_

            if units is None:
                units = obj.__units_

            Units.require_compatible(units, obj.__units_)

            if derivs == {} or derivs is None:
                derivs = obj.__derivs_.copy()   # shallow copy
            else:
                derivs = obj.__derivs_.copy().update(derivs)

            if nrank is None:
                nrank = obj.__nrank_
            elif nrank != obj.__nrank_:         # nranks _must_ be compatible
                raise ValueError('numerator ranks are incompatible: %d, %d' %
                                 (nrank, obj.__nrank_))

            if drank is None:                   # Override drank if undefined
                drank = obj.__drank_

        # Interpret the arg if it is a NumPy MaskedArray
        mask_from_array = None
        if isinstance(arg, np.ma.MaskedArray):
            if arg.mask is not np.ma.nomask:
                mask_from_array = arg.mask

            arg = arg.data

        # Convert a list or tuple to a NumPy ndarray
        if type(arg) in (list,tuple):
            arg = np.asarray(arg)

        # Fill in the denominator rank if undefined
        if drank is None:
            drank = 0

        # Fill in the numerator rank if undefined
        if nrank is None:
            if self.NRANK is None:
                if self.NUMER is None:
                    nrank = 0
                else:
                    nrank = len(self.NUMER)
            else:
                nrank = self.NRANK
        elif self.NRANK is not None:
            if nrank != self.NRANK:
                raise ValueError(("incompatible array rank for class '%s': " +
                                  "%d, %d") % (type(self).__name__,
                                               self.NRANK, nrank))

        rank = nrank + drank

        # Check the shape against nrank and drank
        full_shape = np.shape(arg)
        if len(full_shape) < rank:
            raise ValueError(("incompatible array shape for class '%s': " +
                              "%s; minimum rank = %d + %d") %
                              (type(self).__name__, str(full_shape),
                               nrank, drank))

        dd = len(full_shape) - drank
        nn = dd - nrank
        denom = full_shape[dd:]
        numer = full_shape[nn:dd]
        item  = full_shape[nn:]
        shape = full_shape[:nn]

        if self.NUMER is not None and self.NUMER != numer:
            raise ValueError(("incompatible numerator shape for class '%s': " +
                              "%s, %s") % (type(self).__name__,
                                           str(self.NUMER), str(numer)))

        # Check unit compatibility
        test_units = None if (units is False) else units
        if not self.UNITS_OK and not Units.is_angle(test_units): # allow radians
            raise TypeError("units are disallowed in class '%s': %s" %
                            (type(self).__name__, str(units)))

        # Fill in the values
        if isinstance(arg, np.ndarray):
            if arg.dtype.kind == 'b':
                if not self.BOOLS_OK:
                    if self.INTS_OK:
                        arg = arg.astype('int')
                    else:
                        arg = np.asfarray(arg)
            elif arg.dtype.kind in ('i','u'):
                if not self.INTS_OK:
                    if self.FLOATS_OK:
                        arg = np.asfarray(arg)
                    else:
                        arg = (arg != 0)
            elif arg.dtype.kind == 'f':
                if not self.FLOATS_OK:
                    if self.BOOLS_OK:
                        arg = (arg != 0.)
                    else:
                        raise TypeError("floats are disallowed in class '%s'" %
                                        type(self).__name__)
            else:
                raise ValueError("unsupported data type: %s" % str(arg.dtype))

        else:
            if isinstance(arg, bool) or type(arg) == np.bool_:
                if not self.BOOLS_OK:
                    if self.INTS_OK:
                        arg = int(arg)
                    else:
                        arg = float(arg)
            elif isinstance(arg, int) or isinstance(arg, long):
                if not self.INTS_OK:
                    if self.FLOATS_OK:
                        arg = float(arg)
                    else:
                        arg = (arg != 0)
            elif isinstance(arg, float):
                if not self.FLOATS_OK:
                    if self.BOOLS_OK:
                        arg = (arg != 0.)
                    else:
                        raise TypeError("floats are disallowed in class '%s'" %
                                        type(self).__name__)
            else:
                raise ValueError("unsupported data type: '%s'" % str(arg.dtype))

        self.__values_ = arg

        # Fill in the mask
        if isinstance(mask, Qube):
            mask = mask.as_mask_where_nonzero_or_masked()

        if mask is None:
            mask = False

        if np.shape(mask) == ():
            mask = bool(mask)
        else:
            mask = np.asarray(mask).astype('bool')

        if mask_from_array is not None:
            for r in range(rank):
                mask_from_array = np.any(mask_from_array)

            mask = mask | mask_from_array

        if not self.MASKS_OK and mask is not False:
            raise TypeError("masks are disallowed in class '%s'" %
                            type(self).__name__)

        # Broadcast the mask to the shape of the values if necessary
        if np.shape(mask) not in ((), shape):
            mask_mismatch = True
            try:
                dummy = np.empty(shape, dtype='bool')
                new_mask = np.broadcast_arrays(mask, dummy)[0]

                if new_mask.shape == shape:
                    mask = new_mask.copy()
                    mask_mismatch = False

            except:
                pass

            if mask_mismatch:
                raise ValueError(("object shape and mask shape are " +
                                  "incompatible: %s, %s") %
                                  (str(shape), str(np.shape(mask))))

        self.__mask_ = mask
        self.__antimask_ = None
        self.__corners_  = None
        self.__slicer_   = None

        if np.shape(self.__mask_) == ():    # Avoid type np.bool_ if possible
            if self.__mask_:
                self.__mask_ = True
                self.__antimask_ = False
            else:
                self.__mask_ = False
                self.__antimask_ = True

        # Fill in the default
        if default is not None and np.shape(default) == item:
            self.__default_ = default

        elif hasattr(self, 'DEFAULT_VALUE') and drank == 0:
            self.__default_ = self.DEFAULT_VALUE

        elif item:
            self.__default_ = np.ones(item)

        else:
            self.__default_ = 1

        if self.is_float():
            if isinstance(self.__default_, np.ndarray):
                self.__default_ = self.__default_.astype('float')
            else:
                self.__default_ = float(self.__default_)

        elif self.is_int():
            if isinstance(self.__default_, np.ndarray):
                self.__default_ = self.__default_.astype('int')
            else:
                self.__default_ = int(self.__default_)

        # Fill in the units
        if self.UNITS_OK:
            self.__units_ = None if (units is False) else Units.as_units(units)
        else:
            if not Units.is_angle(units):
                raise TypeError("units are disallowed in class '%s'" %
                                 type(self).__name__)
            self.__units_ = None

        # Fill in the remaining shape info
        self.__rank_  = rank
        self.__nrank_ = nrank
        self.__drank_ = drank

        self.__item_  = item
        self.__numer_ = numer
        self.__denom_ = denom

        self.__shape_ = shape

        # The object is read-only if the values array is read-only
        self.__readonly_ = Qube._array_is_readonly(self.__values_)

        if self.__readonly_:
            Qube._array_to_readonly(self.__mask_)

        # Install the derivs (converting to read-only if necessary)
        self.__derivs_ = {}
        if derivs:
            self.insert_derivs(derivs)

        self.__is_deriv_ = False    # gets changed by insert_derivs()

        # Used only for if clauses
        self.__truth_if_any_ = False
        self.__truth_if_all_ = False

        return

    def clone(self, recursive=True, preserve=None):
        """Fast construction of a shallow copy.

        Inputs:
            recursive   True to copy the derivatives from the example; False to
                        ignore them.
            preserve    an optional list of derivative names to include even if
                        recursive is False.
        """

        obj = object.__new__(type(self))

        for (attr, value) in self.__dict__.iteritems():
            if attr.startswith('d_d') and attr[3:] in self.__derivs_:
                continue

            obj.__dict__[attr] = value

        # Clone the derivs
        obj.__derivs_ = {}
        if recursive:
            for (key, deriv) in self.__derivs_.iteritems():
                obj.insert_deriv(key, deriv.clone(recursive=False))

        elif preserve:
            for (key, deriv) in self.__derivs_.iteritems():
                if key not in preserve:
                    obj.insert_deriv(key, deriv.clone(recursive=False))

        return obj

    ############################################################################
    # For Packrat serialization
    ############################################################################

    def PACKRAT__args__(self):
        """Return the list of attributes to write into the Packrat file."""

        args = ['_Qube__shape_', '_Qube__item_',
                '_Qube__nrank_', '_Qube__drank_',
                '_Qube__readonly_', '_Qube__default_']

        # For a fully masked object, no need to save values
        if self.mask is True:
            args.append('_Qube__mask_')

        # For an unmasked object, save the array values as they are
        elif self.mask is False:
            args.append('_Qube__mask_')
            args.append('_Qube__values_')

        # For a partially masked object, take advantage of the corners and also
        # only save the unmasked values
        else:
            _ = self.corners
            args.append('_Qube__corners_')

            self.__sliced_mask_ = self.__mask_[self.slicer]
            args.append('_Qube__sliced_mask_')

            self.__unmasked_values_ = self.__values_[self.antimask]
            args.append('_Qube__unmasked_values_')

        # Include derivatives and units as needed
        if self.__derivs_:
            args.append('_Qube__derivs_{"single":True}')

        if self.__units_:
            args.append('_Qube__units_')

        return args

    @staticmethod
    def PACKRAT__init__(cls, **args):
        """Construct an object from the subobjects extracted from the XML."""

        shape = args['shape']
        item  = args['item']
        nrank = args['nrank']
        drank = args['drank']
        readonly = args['readonly']
        default  = args['default']
        derivs   = args.get('derivs', None)
        units    = args.get('units', None)

        try:
            dtype = default.dtype
        except AttributeError:
            if type(default) == int:
                dtype = 'int'
            else:
                dtype = 'float'

        # If the dictionary contains 'mask', it is either fully masked or
        # fully unmasked
        try:
            mask = args['mask']
            if mask:
                values = np.empty(shape + item, dtype=dtype)
                repeater = (Ellipsis,) + len(item) * (slice(None),)
                values[repeater] = default
            else:
                values = args['values']

        # Otherwise, take advantage of the corners and of the fact that only
        # unmasked values were saved
        except KeyError:
    
            values = np.empty(shape + item, dtype=dtype)
            repeater = (Ellipsis,) + len(item) * (slice(None),)
            values[repeater] = default

            corners = args['corners']
            slicer = Qube.slicer_from_corners(args['corners'])
            mask = np.ones(shape, dtype='bool')
            mask[slicer] = args['sliced_mask']

            values[np.logical_not(mask)] = args['unmasked_values']

        # Construct the object
        obj = Qube.__new__(cls)
        obj.__init__(values, mask, units, derivs, nrank, drank)

        # Set the readonly state as needed
        if readonly:
            obj = obj.as_readonly()

        return obj

    ############################################################################
    # Properties and low-level access
    ############################################################################

    def _set_values_(self, values, mask=None, antimask=None):
        """Low-level method to update the values of an array.

        The read-only status of the object is defined by that of the given
        value.
        If a mask is provided, it is also updated.
        If antimask is not None, then only the array locations associated with
        the antimask are modified.
        """

        # Confirm shapes
        if antimask is None:
            assert np.shape(values) == np.shape(self.__values_), \
                'shape mismatch'
            if type(mask) == np.ndarray:
                assert np.shape(mask) == self.shape, 'mask shape mismatch'
        else:
            if np.shape(antimask):
                assert np.shape(antimask) == self.shape, \
                    'antimask shape mismatch'

        # Update values
        if antimask is None:
            self.__values_ = values
        elif np.shape(values) == ():
            self.__values_ = values
        else:
            self.__values_[antimask] = values[antimask]

        self.__readonly_ = Qube._array_is_readonly(self.__values_)

        # Update the mask if necessary
        if mask is not None:

            if antimask is None:
                self.__mask_ = mask
            elif np.shape(mask) == ():
                if np.shape(self.__mask_) == ():
                    old_mask = self.__mask_
                    self.__mask_ = np.empty(self.shape, dtype='bool')
                    self.__mask_.fill(old_mask)
                self.__mask_[antimask] = mask
            else:
                self.__mask_[antimask] = mask[antimask]

            self.__antimask_ = None
            self.__corners_  = None
            self.__slicer_   = None

        # Set the readonly state based on the values given
        if self.__readonly_:
            self.__mask_ = Qube._array_to_readonly(self.__mask_)

        elif Qube._array_is_readonly(self.__mask_):
            self.__mask_ = self.__mask_.copy()

        # Avoid type np.bool_ if possible
        if np.shape(self.__mask_) == ():
            if self.__mask_:
                self.__mask_ = True
                self.__antimask_ = False
            else:
                self.__mask_ = False
                self.__antimask_ = True

        return self

    def _set_mask_(self, mask, antimask=None):
        """Low-level method to update the mask of an array.

        The read-only status of the object will be preserved.
        If antimask is not None, then only the mask locations associated with
        the antimask are modified
        """

        # Confirm the shape
        assert type(mask)==bool or mask.shape == self.shape, \
            'mask shape mismatch'

        is_readonly = self.__readonly_

        # Update the mask
        if antimask is None:
            self.__mask_ = mask
        elif np.shape(mask) == ():
            if np.shape(self.__mask_) == ():
                old_mask = self.__mask_
                self.__mask_ = np.empty(self.shape, dtype='bool')
                self.__mask_.fill(old_mask)
            self.__mask_[antimask] = mask
        else:
            self.__mask_[antimask] = mask[antimask]

        self.__antimask_ = None
        self.__corners_  = None
        self.__slicer_   = None

        if isinstance(self.__mask_, np.ndarray):
            if is_readonly:
                self.__mask_ = Qube._array_to_readonly(self.__mask_)

            elif Qube._array_is_readonly(self.__mask_):
                self.__mask_ = self.__mask_.copy()

        # Avoid type np.bool_ if possible
        if np.shape(self.__mask_) == ():
            if self.__mask_:
                self.__mask_ = True
                self.__antimask_ = False
            else:
                self.__mask_ = False
                self.__antimask_ = True

        return self

    @property
    def values(self): return self.__values_

    @property
    def vals(self): return self.__values_        # Handy shorthand

    @property
    def mvals(self):

        # Deal with a scalar
        if self.__values_.shape == ():
            if self.__mask_:
                return np.ma.masked
            else:
                return self.__values_

        # Construct something that behaves as a suitable mask
        if self.mask is False:
            newmask = np.ma.nomask
        elif self.mask is True:
            newmask = np.ones(self.__values_.shape, dtype='bool')
        elif self.__rank_ > 0:
            newmask = self.__mask_.reshape(self.__shape_ + self.__rank_ * (1,))
            (newmask, newvals) = np.broadcast_arrays(newmask, self.__values_)
        else:
            newmask = self.__mask_

        return np.ma.MaskedArray(self.__values_, newmask)

    @property
    def mask(self):
        # Annoyingly, type np.bool is not a subclass of bool
        if np.shape(self.__mask_) == () and type(self.__mask_) != bool:
            if self.__mask_:
                self.__mask_ = True
                self.__antimask_ = False
            else:
                self.__mask_ = False
                self.__antimask_ = True

        return self.__mask_

    @property
    def antimask(self):
        if self.__antimask_ is None:
            self.__antimask_ = np.logical_not(self.mask)

        return self.__antimask_

    @property
    def default(self): return self.__default_

    @property
    def units(self): return self.__units_

    @property
    def derivs(self): return self.__derivs_

    @property
    def shape(self): return self.__shape_

    @property
    def rank(self): return self.__rank_

    @property
    def nrank(self): return self.__nrank_

    @property
    def drank(self): return self.__drank_

    @property
    def item(self): return self.__item_

    @property
    def numer(self): return self.__numer_

    @property
    def denom(self): return self.__denom_

    @property
    def size(self):
        return int(np.prod(self.__shape_))

    @property
    def isize(self):
        return int(np.prod(self.__item_))

    @property
    def nsize(self):
        return int(np.prod(self.__numer_))

    @property
    def dsize(self):
        return int(np.prod(self.__denom_))

    @property
    def readonly(self):
        return self.__readonly_

    @property
    def is_deriv(self):
        return self.__is_deriv_

    def find_corners(self):
        """Update the corner indices such that everything outside this defined
        "hypercube" is masked."""

        shape = self.__shape_
        lshape = len(shape)
        index0 = lshape * (0,)

        self.__slicer_  = None

        if lshape == 0:
            self.__corners_ = None

        if self.mask is False:
            self.__corners_ = (index0, shape)
            return

        if self.mask is True:
            self.__corners_ = (index0, index0)
            return

        lower = []
        upper = []
        antimask = self.antimask
        self.__masked_value_ = None

        for axis in range(lshape):
            other_axes = range(lshape)
            del other_axes[axis]

            occupied = np.any(antimask, tuple(other_axes))
            indices = np.where(occupied)[0]
            if len(indices) == 0:
                self.__corners_ = (index0, index0)
                return

            lower.append(indices[0])
            upper.append(indices[-1] + 1)

        self.__corners_ = (tuple(lower), tuple(upper))

    @property
    def corners(self):
        """Corners of a "hypercube" that contain all the unmasked array
        elements."""

        if self.__corners_ is None:
            self.find_corners()

        return self.__corners_

    @staticmethod
    def slicer_from_corners(corners):
        """A slice object based on corners specified as a tuple of indices.
        """

        slice_objects = []
        for axis in range(len(corners[0])):
            slice_objects.append(slice(corners[0][axis], corners[1][axis]))

        return tuple(slice_objects)

    @property
    def slicer(self):
        """A slice object containing all the array elements inside the current
        corners."""

        if self.__slicer_ is None:
            self.__slicer_ = Qube.slicer_from_corners(self.corners)

        return self.__slicer_

    ############################################################################
    # Derivative operations
    ############################################################################

    def insert_deriv(self, key, deriv, override=True):
        """Insert or replace a derivative in this object.

        To prevent recursion, any internal derivatives of a derivative object
        are stripped away. If the object is read-only, then derivatives will
        also be converted to read-only.

        Derivatives cannot be integers. They are converted to floating-point if
        necessary.

        You cannot replace the pre-existing value of a derivative in a read-only
        object unless you explicit set override=True. However, inserting a new
        derivative into a read-only object is not prevented.

        Input:
            key         the name of the derivative. Each derivative also becomes
                        accessible as an object attribute with "d_d" in front of
                        the name. For example, the time-derivative of this
                        object might be keyed by "t", in which case it can also
                        be accessed as attribute "d_dt".

            deriv       the derivative as a Qube subclass. Derivatives must have
                        the same leading shape and the same numerator as the
                        object; denominator items are used for partial
                        derivatives.

            override    True to allow the value of a pre-existing derivative to
                        be replaced.
        """

        if not self.DERIVS_OK:
            raise TypeError("derivatives are disallowed in class '%s'" %
                            type(self).__name__)

        # Make sure the derivative is compatible with the object
        if not isinstance(deriv, Qube):
            raise ValueError("invalid class for derivative '%s': '%s'" %
                             (key, type(deriv).__name__))

        if self.__numer_ != deriv.__numer_:
            raise ValueError(("shape mismatch for numerator of derivative " +
                              "'%s': %s, %s") % (key, str(deriv.__numer_),
                                                      str(self.__numer_)))

        # Prevent recursion, convert to floating point
        deriv = deriv.without_derivs().as_float()

        # Broadcast the shape to match the parent object if necessary
        if deriv.shape != self.shape:
            deriv = deriv.broadcast_into_shape(self.shape, False).as_readonly()
        elif self.__readonly_ and not deriv.__readonly_:
            deriv = deriv.clone(recursive=False).as_readonly()

        # Save in the derivative dictionary and as an attribute
        if self.readonly and (key in self.__derivs_) and not override:
            raise ValueError('derivative d_d' + key + ' cannot be replaced ' +
                             'in a read-only object')

        self.__derivs_[key] = deriv
        setattr(self, 'd_d' + key, deriv)

        deriv.__is_deriv_ = True
        return self

    def insert_derivs(self, dict, override=False):
        """Insert or replace the derivatives in this object from a dictionary.

        You cannot replace the pre-existing values of any derivative in a
        read-only object unless you explicit set override=True. However,
        inserting a new derivative into a read-only object is not prevented.

        Input:
            dict        the dictionary of derivatives, keyed by their names.

            override    True to allow the value of a pre-existing derivative to
                        be replaced.
        """

        # Check every insert before proceeding with any
        if self.readonly and not override:
            for key in dict:
                if key in self.__derivs_:
                    raise ValueError('derivative d_d' + key + ' cannot be ' +
                                     'replaced in a read-only object')

        # Insert derivatives
        for (key, deriv) in dict.iteritems():
            self.insert_deriv(key, deriv, override)

    def delete_deriv(self, key, override=False):
        """Delete a single derivative from this object, given the key.

        Derivatives cannot be deleted from a read-only object without explicitly
        setting override=True.

        Input:
            key         the name of the derivative to delete.
            override    True to allow the deleting of derivatives from a
                        read-only object.
        """

        if not override: self.require_writable()

        if key in self.__derivs_.keys():
            del self.__derivs_[key]
            del self.__dict__['d_d' + key]

    def delete_derivs(self, override=False, preserve=None):
        """Delete all derivatives from this object.

        Derivatives cannot be deleted from a read-only object without explicitly
        setting override=True.

        Input:
            override    True to allow the deleting of derivatives from a
                        read-only object.
            preserve    an optional list, tuple or set of the names of
                        derivatives to retain. All others are removed.
        """

        if not override: self.require_writable()

        # If something is being preserved...
        if preserve:

            # Delete derivatives not on the list
            for key in self.__derivs_.keys():
                if key not in preserve:
                    self.delete_deriv(key, override)

            return

        # Delete all derivatives
        for key in self.__derivs_.keys():
            delattr(self, 'd_d' + key)

        self.__derivs_ = {}
        return

    def without_derivs(self, preserve=None):
        """Return a shallow copy of this object without derivatives.

        A read-only object remains read-only, and is cached for later use.

        Input:
            preserve    an optional list, tuple or set of the names of
                        derivatives to retain. All others are removed.
        """

        if self.__derivs_ == {}: return self

        # If something is being preserved...
        if preserve:

            # Create a fast copy with derivatives
            obj = self.clone(recursive=True)

            # Delete derivatives not on the list
            deletions = []
            for key in obj.__derivs_:
                if key not in preserve:
                    deletions.append(key)

            for key in deletions:
                obj.delete_deriv(key, True)

            return obj

        # Return a fast copy without derivatives
        return self.clone(recursive=False)

    def without_deriv(self, key):
        """Return a shallow copy of this object without a particular derivative.

        A read-only object remains read-only.

        Input:
            key         the key of the derivative to remove.
        """

        if key not in self.__derivs_: return self

        result = self.clone(recursive=True)
        del result.__derivs_[key]

        return result

    def with_deriv(self, key, value, method='insert'):
        """Return a shallow copy of this object with a derivative inserted or
        added.

        A read-only object remains read-only.

        Input:
            key         the key of the derivative to insert.
            value       value for this derivative.
            method      how to insert the derivative:
                        'insert'  inserts the new derivative, raising a
                                  ValueError if a derivative of the same name
                                  already exists.
                        'replace' replaces an existing derivative of the same
                                  name.
                        'add'     adds this derivative to an existing derivative
                                  of the same name.
        """

        result = self.clone(recursive=True)

        assert method in ('insert', 'replace', 'add')

        if key in result.__derivs_:
            if method == 'insert':
                raise ValueError('derivative d_d%s already exists' % key)

            if method == 'add':
                value = value + result.__derivs_[key]

        result.insert_deriv(key, value)
        return result

    def unique_deriv_name(self, key, *objects):
        """Return the given name for a deriv if it does not exist in this
        object or any of the given objects; otherwise return a variant that is
        unique."""

        # Make a list of all the derivative keys
        all_keys = set(self.derivs.keys())
        for obj in objects:
            if not hasattr(obj, 'derivs'): continue
            all_keys |= set(obj.derivs.keys())

        # Return the proposed key if it is unused
        if key not in all_keys:
            return key

        # Otherwise, tack on a number and iterate until the name is unique
        i = 0
        while True:
            unique = key + str(i)
            if unique not in all_keys:
                return unique

            i += 1

    ############################################################################
    # Unit operations
    ############################################################################

    def set_units(self, units, override=False):
        """Set the units of this object.

        Input:
            units       the new units.
            override    if True, the units can be modified on a read-only
                        object.
        """

        if not self.UNITS_OK and units is not None:
            raise TypeError("units are disallowed in class '%s'" %
                            type(self).__name__)

        if not override: self.require_writable()

        units = Units.as_units(units)

        Units.require_compatible(units, self.__units_)
        self.__units_ = units

        self.__without_derivs = None

    def without_units(self, recursive=True):
        """Return a shallow copy of this object without derivatives.

        A read-only object remains read-only. If recursive is True, derivatives
        are also stripped of their units.
        """

        if self.__units_ is None and self.__derivs_ == {}: return self

        obj = self.clone(recursive)
        obj.__units_ = None

        return obj

    def into_units(self, recursive=True):
        """Return a copy of this object with values scaled to its units.

        Normally, the values stored internally in standard units of km, radians,
        and seconds. This method overrides that standard and converts the
        internal values to their intended units. Used primarily for I/O.

        If this object has no units, or is already in standard units, it is
        returned as is.

        Inputs:
            recursive   if True, the derivatives are also converted; otherwise
                        derivatives are removed.
        """

        # Handle easy cases first
        if recursive or not self.__derivs_:
            if self.__units_ is None: return self
            if self.__units_.into_units_factor == 1.: return self

        # Construct the new object
        obj = self.clone(recursive)
        obj._set_values_(Units.into_units(self.__units_, self.__values_))

        # Fill in derivatives if necessary
        if recursive:
            for (key, deriv) in self.__derivs_.iteritems():
                obj.insert_deriv(key, deriv.into_units(recursive=False))

        return obj

    def from_units(self, recursive=True):
        """Return a copy of this object with values scaled to standard units.

        This method undoes the conversion done by into_units().

        Inputs:
            recursive       if True, the derivatives are also converted;
                            otherwise, derivatives are removed.
        """

        # Handle easy cases first
        if recursive or not self.__derivs_:
            if self.__units_ is None: return self
            if self.__units_.from_units_factor == 1.: return self

        # Construct the new object
        obj = self.clone(recursive)
        obj._set_values_(Units.from_units(self.__units_, self.__values_))

        # Fill in derivatives if necessary
        if recursive:
            for (key, deriv) in self.__derivs_.iteritems():
                obj.insert_deriv(key, deriv.from_units(recursive=False))

        return obj

    def confirm_units(self, units):
        """Raises a ValueError if the units are not compatible with this object.

        Input:
            units       the new units.
        """

        if not Units.can_match(self.units, units):
            raise ValueError('units are not compatible')

        return self

    ############################################################################
    # Read-only/read-write operations
    ############################################################################

    @staticmethod
    def _array_is_readonly(arg):
        """Return True if the argument is a read-only NumPy ndarray.

        False means that it is either a writable array or a scalar."""

        if type(arg) != np.ndarray: return False

        return (not arg.flags['WRITEABLE'])

    @staticmethod
    def _array_to_readonly(arg):
        """Make the given array read-only. Returns the array."""

        if type(arg) != np.ndarray: return arg

        arg.flags['WRITEABLE'] = False
        return arg

    def as_readonly(self, recursive=True):
        """Convert this object to read-only. It is modified and returned.

        If recursive is False, the derivatives are removed. Otherwise, they are
        also converted to read-only.

        If this object is already read-only, it is returned as is. Otherwise,
        the internal __values_ and __mask_ arrays are modified as necessary.
        Once this happens, the internal arrays will also cease to be writable in
        any other object that shares them.

        Note that as_readonly() cannot be undone. Use copy() to create a
        writable copy of a readonly object.
        """

        # If it is already read-only, return
        if self.__readonly_: return self

        # Update the value if it is an array
        Qube._array_to_readonly(self.__values_)
        Qube._array_to_readonly(self.__mask_)
        self.__readonly_ = True

        # Update the derivatives
        if recursive:
            for key in self.__derivs_:
                self.__derivs_[key].as_readonly()

        return self

    def match_readonly(self, arg):
        """Sets the read-only status of this object equal to that of another."""

        if arg.__readonly_:
            return self.as_readonly()
        elif self.__readonly_:
            raise ValueError('object is read-only')

        return self

    def require_writable(self):
        """Raises a ValueError if the object is read-only.

        Used internally at the beginning of methods that will modify this
        object.
        """

        if self.__readonly_:
            raise ValueError('object is read-only')

    ############################################################################
    # Copying operations and conversions
    ############################################################################

    def copy(self, recursive=True, readonly=False):
        """Deep copy operation with additional options.

        Input:
            recursive   if True, derivatives will also be copied; if False,
                        the copy will not contain any derivatives.

            readonly    if True, the copy will be read-only. If the source is
                        already read-only, no array duplication is required.
                        if False, the returned copy is guaranteed to be an
                        entirely new copy, suitable for modification.
        """

        # Create a shallow copy
        obj = self.clone(recursive=False)

        # Copying a readonly object is easy
        if self.__readonly_ and readonly:
            return obj

        # Copy the values
        if isinstance(self.__values_, np.ndarray):
            obj.__values_ = self.__values_.copy()
        else:
            obj.__values_ = self.__values_

        # Copy the mask
        if isinstance(self.__mask_, np.ndarray):
            obj.__mask_ = self.__mask_.copy()
        else:
            obj.__mask_ = self.__mask_

        obj.__antimask_ = None
        obj.__corners_  = None
        obj.__slicer_   = None

        # Set the read-only state
        if readonly:
            obj.as_readonly()
        else:
            obj.__readonly_ = False

       # Make the derivatives read-only if necessary
        if recursive:
            for (key, deriv) in self.__derivs_.iteritems():
                obj.insert_deriv(key, deriv.copy(recursive=False,
                                                 readonly=readonly))

        return obj

    # Python-standard copy function
    def __copy__(self):
        """Return a deep copy of this object unless it is read-only."""

        return self.copy(recursive=True, readonly=False)

    ############################################################################
    # Floats vs. integers vs. booleans
    ############################################################################

    def is_numeric(self):
        """Return True if this object contains numbers; False if boolean.

        This method returns True. It is overridden by the Boolean subclass to
        return False.
        """

        return True

    def as_numeric(self):
        """Return a numeric version of this object.

        This method normally returns the object itself without modification. It
        is overridden by the Boolean subclass to return an integer version equal
        to one where True and zero where False.
        """

        return self

    def is_float(self):
        """True if this object contains floats; False if ints or booleans."""

        # Array case
        if isinstance(self.__values_, np.ndarray):
            return np.issubdtype(self.__values_.dtype, float)

        # Scalar case
        return not isinstance(self.__values_, numbers.Rational)

    def as_float(self, recursive=True):
        """Return a floating-point version of this object.

        If this object already contains floating-point values, it is returned
        as is. Otherwise, a copy is returned. Derivatives are not modified.

        Derivatives are not modified.
        """

        # If already floating, return as is
        if self.is_float():
            if recursive: return self
            return self.without_derivs()

        # Handle a Boolean
        if isinstance(self, Qube.BOOLEAN_CLASS):
            self = self.as_float()

        # If object cannot contain floats, raise an error
        if not self.FLOATS_OK:
            raise TypeError("floats are disallowed in class '%s'" %
                            type(self).__name__)

        # Convert values to float
        if isinstance(self.__values_, np.ndarray):
            new_values = self.__values_.astype('float')
        else:
            new_values = float(self.__values_)

        # Construct a new copy
        obj = self.clone(recursive)
        obj._set_values_(new_values)
        return obj

    def is_int(self):
        """True if this object contains ints; False if floats or booleans."""

        # Array case
        if isinstance(self.__values_, np.ndarray):
            return np.issubdtype(self.__values_.dtype, int)

        # Scalar case
        if isinstance(self.__values_, bool):
            return False

        return isinstance(self.__values_, numbers.Rational)

    def as_int(self, recursive=True):
        """Return an integer version of this object.

        If this object already contains integers, it is returned as is.
        Otherwise, a copy is returned.
        """

        # If already integers, return as is
        if self.is_int():
            if recursive: return self
            return self.without_derivs()

        # Handle a Boolean
        if isinstance(self, Qube.BOOLEAN_CLASS):
            self = self.as_int()

        # If object cannot contain ints, raise an error
        if not self.INTS_OK:
            raise TypeError("ints are disallowed in class '%s'" %
                            type(self).__name__)

        # Truncate the integers (as a floor operation)
        if isinstance(self.__values_, np.ndarray):
            new_values = (self.__values_ // 1.).astype('int')
        else:
            new_values = int(self.__values_ // 1.)

        # Construct a new copy
        obj = self.clone(recursive)
        obj._set_values_(new_values)
        return obj

    def is_bool(self):
        """True if this object contains booleans; False otherwise."""

        return isinstance(self, Qube.BOOLEAN_CLASS)

    def as_bool(self):
        """Return an boolean version of this object."""

        # If already boolean, return as is
        if self.is_bool():
            return self

        return Qube.BOOLEAN_CLASS.as_boolean(self)

    def is_all_masked(self):
        """Return True if this is entirely masked."""

        return np.all(self.__mask_)

    ############################################################################
    # Subclass operations
    ############################################################################

    @staticmethod
    def is_empty(arg):
        return (type(arg) == Qube.EMPTY_CLASS)

    @staticmethod
    def is_real_number(arg):
        """Return True if arg is of a Python numeric or NumPy numeric type.""" 
        return isinstance(arg, numbers.Real) or isinstance(arg, np.number) 

    def masked_single(self):
        """Return an object of this subclass containing one masked value."""

        if self.__item_ == ():
            if self.is_float():
                new_value = 1.
            else:
                new_value = 1
        else:
            if self.is_float():
                new_value = np.ones(self.__item_, dtype='float')
            else:
                new_value = np.ones(self.__item_, dtype='int')

        obj = Qube.__new__(type(self))
        obj.__init__(new_value, True, derivs={}, example=self)
        obj.as_readonly()
        return obj

    def as_this_type(self, arg, recursive=True, nrank=None, drank=None):
        """Return the argument converted to this class.

        If the object is already of this class, it is returned unchanged. If the
        argument is a scalar or NumPy ndarray, a new instance of this object's
        class is created.

        Input:
            arg         the object (scalar, NumPy ndarray or Qube subclass) to
                        convert to the class of this object.
            recursive   True to convert the derivatives as well.
            nrank       numerator rank, overriding value in this object.
            drank       denominator rank, overriding value in this object.
        """

        # If the classes already match, return the argument as is
        if type(arg) == type(self): return arg

        # Construct the new object
        obj = Qube.__new__(type(self))
        if isinstance(arg, Qube):
            obj.__init__(arg.__values_, arg.__mask_, derivs={},
                         nrank=nrank, drank=drank, example=arg)

            # Copy the derivatives if necessary
            if recursive:
                for (key, deriv) in arg.__derivs_.iteritems():
                    obj.insert_deriv(key, self.as_this_type(deriv, False))

        else:
            obj.__init__(arg, mask=False, units=None, derivs={},
                              nrank=nrank, drank=drank, example=self)

        return obj

    def as_this_type_unless_boolean(self, arg, recursive=True,
                                          nrank=None, drank=None):
        """Works the same as as_this_type() except when arg contains int or
        float values, the value returned is a Scalar instead.
        """

        if not isinstance(self, Qube.BOOLEAN_CLASS):
            return self.as_this_type(arg, recursive, nrank, drank)

        arg_type = None

        if isinstance(arg, Qube):
            if arg.is_int():
                arg_type = 'int'
            if arg.is_float():
                arg_type = 'float'

        elif isinstance(arg, np.ndarray):
            if arg.dtype.kind in 'ui':
                arg_type = 'int'
            elif arg.dtype.kind == 'f':
                arg_type = 'float'
            elif arg.dtype.kind == 'b':
                arg_type = 'bool'

        elif isinstance(arg, bool):
            arg_type = 'bool'

        elif isinstance(arg, numbers.Rational):
            arg_type = 'int'

        elif isinstance(arg, numbers.Real):
            arg_type = 'float'

        if arg_type == 'int':
            return self.as_int().as_this_type(arg, recursive, nrank, drank)

        if arg_type == 'float':
            return self.as_float().as_this_type(arg, recursive, nrank, drank)

        return self.as_this_type(arg, recursive, nrank, drank)

    def cast(self, classes):
        """Return a shallow copy of this object casted to another subclass.

        If a list or tuple of classes is provided, the object will be casted to
        the first suitable class in the list. If this object is already has the
        selected class, it is returned without modification.

        Input:
            classes     a single Qube subclass, or a list or tuple of Qube
                        subclasses.
        """

        # Convert a single class to a tuple
        if isinstance(classes, type):
            classes = (classes,)

        # For each class in the list...
        for cls in classes:

            # If this is already the class of this object, return it as is
            if cls is type(self): return self

            # Exclude the class if it is incompatible
            if cls.NUMER is not None and cls.NUMER != self.__numer_: continue
            if cls.NRANK is not None and cls.NRANK != self.__nrank_: continue

            # Construct the new object
            obj = Qube.__new__(cls)
            obj.__init__(self, example=self)
            return obj

        # If no suitable class was found, return this object unmodified
        return self

    ############################################################################
    # Masking operations
    ############################################################################

    def mask_where(self, mask, replace=None, remask=True):
        """Return a copy of this object after a mask has been applied.

        If the mask is empty, this object is returned unchanged.

        Inputs:
            mask            the mask to apply as a boolean array.
            replace         a single replacement value or, an object of the same
                            shape and class as this object, containing
                            replacement values. These are inserted into returned
                            object at every masked location. Use None (the
                            default) to leave values unchanged.
            remask          True to include the new mask in the object's mask;
                            False to replace the values but leave them unmasked.
        """

        # Convert to boolean array if necessary
        if isinstance(mask, np.ndarray):
            mask = mask.astype('bool')
        elif isinstance(mask, Qube):
            mask = mask.as_mask_where_nonzero_or_masked()
        else:
            mask = Qube.BOOLEAN_CLASS.as_boolean(mask).__values_

        # If the mask is empty, return the object as is
        if not np.any(mask): return self

        # Get the replacement value as this type
        if replace is not None:
            replace = self.as_this_type(replace, recursive=False,
                                        nrank=self.nrank, drank=self.drank)
            if replace.shape != () and replace.shape != self.shape:
                raise ValueError('shape of replacement is incompatible with ' +
                                 'shape of object being masked')

        # Shapeless case
        if np.shape(self.__values_) == ():
            if replace is None:
                new_values = self.__values_
            else:
                new_values = replace.__values_

            if remask or replace.__mask_:
                new_mask = True
            else:
                new_mask = self.__mask_

            obj = self.clone(recursive=True)
            obj._set_values_(new_values, new_mask)
            return obj

        # Construct the new mask
        if remask:
            new_mask = self.__mask_ | mask
        elif np.shape(self.__mask_) == ():
            new_mask = self.__mask_
        else:
            new_mask = self.__mask_.copy()

        # Construct the new array of values
        if replace is None:
            new_values = self.__values_

        # If replacement is a single value...
        elif replace.shape == ():
            new_values = self.__values_.copy()
            new_values[mask] = replace.__values_

            # Update the mask if replacement values are masked
            if replace.__mask_:
                if np.shape(new_mask) == ():
                    new_mask = True
                else:
                    new_mask[mask] = True

        # If replacement is an array of values...
        else:
            new_values = self.__values_.copy()
            new_values[mask] = replace.__values_[mask]

            # Update the mask if replacement values are masked
            if new_mask is True:
                pass
            elif replace.mask is False:
                pass
            else:
                if new_mask is False:
                    new_mask = np.zeros(self.shape, dtype='bool')

                if replace.mask is True:
                    new_mask[mask] = True
                else:
                    new_mask[mask] = replace.__mask_[mask]

        # Construct the new object and return
        obj = self.clone(recursive=True)
        obj._set_values_(new_values, new_mask)
        return obj

    def mask_where_eq(self, match, replace=None, remask=True):
        """Return a copy of this object with items equal to a value masked.

        Instead of or in addition to masking the items, the values can be
        replaced. If no items need to be masked, this object is returned
        unchanged.

        Inputs:
            match           the item value to match.
            replace         a single replacement value or, an object of the same
                            shape and class as this object, containing
                            replacement values. These are inserted into returned
                            object at every masked location. Use None (the
                            default) to leave values unchanged.
            remask          True to include the new mask into the object's mask;
                            False to replace the values but leave them unmasked.
        """

        match = self.as_this_type(match, recursive=False)

        mask = (self.__values_ == match.__values_)
        for r in range(self.__rank_):
            mask = np.all(mask, axis=-1)

        return self.mask_where(mask, replace, remask)

    def mask_where_ne(self, match, replace=None, remask=True):
        """Return a copy of this object with items not equal to a value masked.

        Instead of or in addition to masking the items, the values can be
        replaced. If no items need to be masked, this object is returned
        unchanged.

        Inputs:
            match           the item value to match.
            replace         a single replacement value or, an object of the same
                            shape and class as this object, containing
                            replacement values. These are inserted into returned
                            object at every masked location. Use None (the
                            default) to leave values unchanged.
            remask          True to include the new mask into the object's mask;
                            False to replace the values but leave them unmasked.
        """

        match = self.as_this_type(match, recursive=False)

        mask = (self.__values_ != match.__values_)
        for r in range(self.__rank_):
            mask = np.any(mask, axis=-1)

        return self.mask_where(mask, replace, remask)

    def mask_where_le(self, limit, replace=None, remask=True):
        """Return a copy of this object with items <= a limit value masked.

        Instead of or in addition to masking the items, the values can be
        replaced. If no items need to be masked, this object is returned
        unchanged.

        Inputs:
            limit           the limiting value.
            replace         a single replacement value or, an object of the same
                            shape and class as this object, containing
                            replacement values. These are inserted into returned
                            object at every masked location. Use None (the
                            default) to leave values unchanged.
            remask          True to include the new mask into the object's mask;
                            False to replace the values but leave them unmasked.
        """

        assert self.item == ()

        if isinstance(limit, Qube):
            limit = limit.__values_

        return self.mask_where(self.__values_ <= limit, replace, remask)

    def mask_where_ge(self, limit, replace=None, remask=True):
        """Return a copy of this object with items >= a limit value masked.

        Instead of or in addition to masking the items, the values can be
        replaced. If no items need to be masked, this object is returned
        unchanged.

        Inputs:
            limit           the limiting value.
            replace         a single replacement value or, an object of the same
                            shape and class as this object, containing
                            replacement values. These are inserted into returned
                            object at every masked location. Use None (the
                            default) to leave values unchanged.
            remask          True to include the new mask into the object's mask;
                            False to replace the values but leave them unmasked.
        """

        assert self.item == ()

        if isinstance(limit, Qube):
            limit = limit.__values_

        return self.mask_where(self.__values_ >= limit, replace, remask)

    def mask_where_lt(self, limit, replace=None, remask=True):
        """Return a copy with items less than a limit value masked.

        Instead of or in addition to masking the items, the values can be
        replaced. If no items need to be masked, this object is returned
        unchanged.

        Inputs:
            limit           the limiting value.
            replace         a single replacement value or, an object of the same
                            shape and class as this object, containing
                            replacement values. These are inserted into returned
                            object at every masked location. Use None (the
                            default) to leave values unchanged.
            remask          True to include the new mask into the object's mask;
                            False to replace the values but leave them unmasked.
        """

        assert self.item == ()

        if isinstance(limit, Qube):
            limit = limit.__values_

        return self.mask_where(self.__values_ < limit, replace, remask)

    def mask_where_gt(self, limit, replace=None, remask=True):
        """Return a copy with items greater than a limit value masked.

        Instead of or in addition to masking the items, the values can be
        replaced. If no items need to be masked, this object is returned
        unchanged.

        Inputs:
            limit           the limiting value.
            replace         a single replacement value or, an object of the same
                            shape and class as this object, containing
                            replacement values. These are inserted into returned
                            object at every masked location. Use None (the
                            default) to leave values unchanged.
            remask          True to include the new mask into the object's mask;
                            False to replace the values but leave them unmasked.
        """

        assert self.item == ()

        if isinstance(limit, Qube):
            limit = limit.__values_

        return self.mask_where(self.__values_ > limit, replace, remask)

    def mask_where_between(self, lower, upper, mask_endpoints=False,
                                 replace=None, remask=True):
        """Return a copy with values between two limits masked.

        Instead of or in addition to masking the items, the values can be
        replaced. If no items need to be masked, this object is returned
        unchanged.

        Inputs:
            lower           the lower limit.
            upper           the upper limit.
            mask_endpoints  True to mask the endpoints, where values are equal
                            to the lower or upper limits; False to exclude the
                            endpoints.
            replace         a single replacement value or, an object of the same
                            shape and class as this object, containing
                            replacement values. These are inserted into returned
                            object at every masked location. Use None (the
                            default) to leave values unchanged.
            remask          True to include the new mask into the object's mask;
                            False to replace the values but leave them unmasked.
        """

        assert self.item == ()

        if isinstance(lower, Qube):
            lower = lower.__values_

        if isinstance(upper, Qube):
            upper = upper.__values_

        if mask_endpoints:      # end points are included in the mask
            mask = (self.__values_ >= lower) & (self.__values_ <= upper)
        else:                   # end points are not included in the mask
            mask = (self.__values_ > lower) & (self.__values_ < upper)

        return self.mask_where(mask, replace, remask)

    def mask_where_outside(self, lower, upper, mask_endpoints=False,
                                 replace=None, remask=True):
        """Return a copy with values outside two limits masked.

        Instead of or in addition to masking the items, the values can be
        replaced. If no items need to be masked, this object is returned
        unchanged.

        Inputs:
            lower           the lower limit.
            upper           the upper limit.
            mask_endpoints  True to mask the endpoints, where values are equal
                            to the lower or upper limits; False to exclude the
                            endpoints.
            replace         a single replacement value or, an object of the same
                            shape and class as this object, containing
                            replacement values. These are inserted into returned
                            object at every masked location. Use None (the
                            default) to leave values unchanged.
            remask          True to include the new mask into the object's mask;
                            False to replace the values but leave them unmasked.
        """

        assert self.item == ()

        if isinstance(lower, Qube):
            lower = lower.__values_

        if isinstance(upper, Qube):
            upper = upper.__values_

        if mask_endpoints:      # end points are included in the mask
            mask = (self.__values_ <= lower) | (self.__values_ >= upper)
        else:                   # end points are not included in the mask
            mask = (self.__values_ < lower) | (self.__values_ > upper)

        return self.mask_where(mask, replace, remask)

    def clip(self, lower, upper, remask=True):
        """Return a copy with values clipped to fall within a pair of limits.

        Values below the lower limit become equal to the lower limit; values
        above the upper limit become equal to the upper limit.

        Instead of or in addition to masking the items, the values can be
        replaced. If no items need to be masked, this object is returned
        unchanged.

        Inputs:
            lower           the lower limit or an object of the same shape and
                            type as this, containing lower limits. None or
                            masked values to ignore.
            upper           the upper limit or an object of the same shape and
                            type as this, containing upper limits. None or
                            masked values to ignore.
            remask          True to include the new mask into the object's mask;
                            False to replace the values but leave them unmasked.
        """

        result = self

        if lower is not None:
            if isinstance(lower, Qube):
                lower = lower.__values_
            result = result.mask_where(result.__values_ < lower, lower, remask)

        if upper is not None:
            if isinstance(upper, Qube):
                upper = upper.__values_

            result = result.mask_where(result.__values_ > upper, upper, remask)

        return result

    def count_masked(self):
        """Return the number of masked items in this object."""

        if self.mask is True:
            return self.size
        elif self.mask is False:
            return 0
        else:
            return np.count_nonzero(self.__mask_)

    def masked(self):
        """Return the number of masked items in this object. DEPRECATED NAME;
        use count_masked()."""

        return self.count_masked()

    def count_unmasked(self):
        """Return the number of unmasked items in this object."""

        if self.mask is True:
            return 0
        elif self.mask is False:
            return self.size
        else:
            return self.size - np.count_nonzero(self.__mask_)

    def unmasked(self):
        """Return the number of unmasked items in this object. DEPRECATED NAME;
        use count_unmasked()"""

        return self.count_unmasked()

    def without_mask(self, recursive=True):
        """Return a shallow copy of this object without its mask."""

        if self.mask is False: return self

        obj = self.clone(recursive=False)
        obj._set_mask_(False)

        if recursive:
            for (key,deriv) in self.__derivs_.iteritems():
                obj.insert_deriv(key, deriv.without_mask())

        return obj

    def as_all_masked(self, recursive=True):
        """Return a shallow copy of this object with everything masked."""

        if self.mask is True: return self

        obj = self.clone(recursive=False)
        obj._set_mask_(True)

        if recursive:
            for (key,deriv) in self.__derivs_.iteritems():
                obj.insert_deriv(key, deriv.all_masked())

        return obj

    def all_masked(self, recursive=True):
        """Return a shallow copy of this object with everything masked.
        DEPRECATED NAME; use as_all_masked()"""

        return self.as_all_masked(recursive)

    def remask(self, mask, recursive=True):
        """Return a shallow copy of this object with a replaced mask.

        This is much quicker than masked_where(), for cases where only the mask
        is changing.
        """

        if np.shape(mask) not in (self.shape, ()):
            raise ValueError('mask shape is incompatible with object: ' +
                             str(np.shape(mask)) + ', ' + str(self.shape))

        # Construct the new object
        obj = self.clone(recursive=False)
        obj._set_mask_(mask)

        if recursive:
            for (key,deriv) in self.__derivs_.iteritems():
                obj.insert_deriv(key, deriv.remask(mask))

        return obj

    def as_all_constant(self, constant=None, recursive=True):
        """Return a shallow, read-only copy of this object with constant values.

        Derivatives are all set to zero. The mask is unchanged.
        """

        if constant is None:
            constant = self.zero()

        constant = self.as_this_type(constant, recursive=False)

        obj = self.clone(recursive=False)
        obj._set_values_(Qube.broadcast(constant, obj)[0].__values_)
        obj.as_readonly()

        if recursive:
            for (key,deriv) in self.__derivs_.iteritems():
                obj.insert_deriv(key, deriv.all_constant(recursive=False))

        return obj

    def all_constant(self, constant=None, recursive=True):
        """Return a shallow, read-only copy of this object with constant values.
        DEPRECATED NAME; use as_all_constant().

        Derivatives are all set to zero. The mask is unchanged.
        """

        return self.as_all_constant(constant, recursive)

    def as_size_zero(self, recursive=True):
        """Return a shallow, read-only copy of this object with size zero.
        """

        obj = Qube.__new__(type(self))

        if self.shape:
            new_values = self.__values_[:0]

            if np.shape(self.mask):
                new_mask = self.__mask_[:0]
            else:
                new_mask = np.array([self.mask])[:0]

        else:
            new_values = np.array([self.__values_])[:0]
            new_mask = np.array([self.mask])[:0]

        obj.__init__(new_values, new_mask, derivs={}, example=self)

        if recursive:
            for (key,deriv) in self.__derivs_.iteritems():
                obj.insert_deriv(key, deriv.as_size_zero(recursive=False))

        return obj

    def as_mask_where_nonzero(self):
        """A scalar or NumPy array where values are nonzero and unmasked."""

        if self.mask is True:
            return False

        if self.mask is False:
            if self.__rank_:
                axes = tuple(range(-self.__rank_, 0))
                return np.any(self.__values_, axis=axes)
            else:
                return (self.__values_ != 0)

        if self.__rank_:
            axes = tuple(range(-self.__rank_, 0))
            return np.any(self.__values_, axis=axes) & self.antimask
        else:
            return (self.__values_ != 0) & self.antimask

    def as_mask_where_zero(self):
        """A scalar or NumPy array where values are zero and unmasked."""

        if self.mask is True:
            return False

        if self.mask is False:
            if self.__rank_:
                axes = tuple(range(-self.__rank_, 0))
                return np.all(self.__values_ == 0, axis=axes)
            else:
                return (self.__values_ == 0)

        if self.__rank_:
            axes = tuple(range(-self.__rank_, 0))
            return np.all(self.__values_ == 0, axis=axes) & self.antimask
        else:
            return (self.__values_ == 0) & self.antimask

    def as_mask_where_nonzero_or_masked(self):
        """A scalar or NumPy array where values are nonzero or masked."""

        if self.mask is True:
            return True

        if self.mask is False:
            if self.__rank_:
                axes = tuple(range(-self.__rank_, 0))
                return np.any(self.__values_, axis=axes)
            else:
                return (self.__values_ != 0)

        if self.__rank_:
            axes = tuple(range(-self.__rank_, 0))
            return np.any(self.__values_, axis=axes) | self.__mask_
        else:
            return (self.__values_ != 0) | self.__mask_

    def as_mask_where_zero_or_masked(self):
        """A scalar or NumPy array where values are zero or masked."""

        if self.mask is True:
            return True

        if self.mask is False:
            if self.__rank_:
                axes = tuple(range(-self.__rank_, 0))
                return np.all(self.__values_ == 0, axis=axes)
            else:
                return (self.__values_ == 0)

        if self.__rank_:
            axes = tuple(range(-self.__rank_, 0))
            return np.all(self.__values_ == 0, axis=axes) | self.__mask_
        else:
            return (self.__values_ == 0) | self.__mask_

    def shrink(self, antimask=None):
        """Return a 1-D version of this object, containing only the samples
        in the antimask provided.

        The antimask is ignored if it is None. Otherwise, a value of True
        indicates that a value should be included; False means that is should be
        discarded. A scalar value of True or False applies to the entire object.
        If this object has no shape, then it is returned unchanged.
        """

        if antimask is None or antimask is True or self.shape == (): return self

        if antimask is False:
            obj = self.flatten[:0]
            obj.__shrink_source_ = self
            return obj

        if antimask.shape == (): return self

        if self.shape != antimask.shape:
            raise ValueError('antimask shape %s ' % str(np.shape(antimask)) +
                             'does not match object shape %s' % str(self.shape))

        if self.mask is False:
            mask = np.zeros(self.shape, dtype='bool')
        elif self.mask is True:
            mask = np.ones(self.shape, dtype='bool')
        else:
            mask = self.__mask_

        obj = Qube.__new__(type(self))
        obj.__init__(self.__values_[antimask], mask[antimask], derivs={},
                     example=self)

        for (key, deriv) in self.__derivs_.iteritems():
            obj.insert_deriv(key, deriv.shrink(antimask))

        obj.match_readonly(self)

        obj.__shrink_source_ = self       # Save a pointer to the original
        return obj

    def unshrink(self, antimask=None):
        """Return the results of a shrink operation to its original dimensions.
        """

        if np.shape(antimask) == () or self.shape == ():
            try:
                return self.__shrink_source_
            except AttributeError:
                return self

        # Copy the source if it's still available
        try:
            clone = self.__shrink_source_.clone(recursive=True)
            new_values = clone.__values_.copy()
            new_mask = clone.__mask_.copy()

        # Otherwise, fill in default values
        except AttributeError:
            default = self.__default_

            new_values = np.empty(antimask.shape + self.__item_,
                                  dtype=self.__values_.dtype)

            if isinstance(default, Qube): default = default.__values_
            new_values[...] = default
            new_values[antimask] = self.__values_

            new_mask = np.ones(antimask.shape, dtype='bool')
            new_mask[antimask] = self.__mask_

        # Create the new object
        obj = Qube.__new__(type(self))
        obj.__init__(new_values, new_mask, example=self.without_derivs())

        # Unshrink the derivatives
        for (key, deriv) in self.__derivs_.iteritems():
            obj.insert_deriv(key, deriv.unshrink(antimask))

        obj.match_readonly(self)
        return obj

    ############################################################################
    # I/O operations
    ############################################################################

    def __repr__(self):
        """Express the value as a string.

        The format of the returned string is:
            Class([value, value, ...], suffices, ...)
        where the quanity inside square brackets is the result of str() applied
        to a NumPy ndarray.

        The suffices are, in order...
            - "mask" if the object has a mask
            - the name of the units of the object has units
            - the names of all the derivatives, in alphabetical order

        """

        return self.__str__()


    def __str__(self):
        """Express the value as a string.

        The format of the returned string is:
            Class(value, value, ...; suffices, ...)
        where the quanity inside square brackets is the result of str() applied
        to a NumPy ndarray.

        The suffices are, in order...
            - "denom=(shape)" if the object has a denominator;
            - "mask" if the object has a mask;
            - the name of the units of the object has units;
            - the names of all the derivatives, in alphabetical order.
        """

        suffix = []

        # Apply the units if necessary
        obj = self.into_units(recursive=False)

        # Indicate the denominator shape if necessary
        if self.__denom_ != ():
            suffix += ['denom=' + str(self.__denom_)]

        # Masked objects have a suffix ', mask'
        is_masked = np.any(self.__mask_)
        if is_masked:
            suffix += ['mask']

        # Objects with units include the units in the suffix
        if self.__units_ is not None and self.__units_ != Units.UNITLESS:
            suffix += [str(self.__units_)]

        # Objects with derivatives include a list of the names
        if self.__derivs_:
            keys = self.__derivs_.keys()
            keys.sort()
            for key in keys:
                suffix += ['d_d' + key]

        # Generate the value string
        if np.shape(self.__values_) == ():
            if is_masked:
                string = '--'
            else:
                string = str(self.__values_)
        elif is_masked:
            string = str(self.mvals)[1:-1]
        else:
            string = str(self.__values_)[1:-1]

        # Add an extra set of brackets around derivatives
        if self.__denom_ != ():
            string = '[' + string + ']'

        # Concatenate the results
        if len(suffix) == 0:
            suffix = ''
        else:
            suffix = '; ' + ', '.join(suffix)

        return type(self).__name__ + '(' + string + suffix + ')'
    
    ############################################################################
    # Numerator slicing operations
    ############################################################################

    def extract_numer(self, axis, index, classes=(), recursive=True):
        """Return an object extracted from one numerator axis.

        Input:
            axis        the item axis from which to extract a slice.
            index       the index value at which to extract the slice.
            classes     a single class or list or tuple of classes. The class
                        of the object returned will be the first suitable class
                        in the list. Otherwise, a generic Qube object will be
                        returned.
            recursive   True to include matching slices of the derivatives in
                        the returned object; otherwise, the returned object will
                        not contain derivatives.
        """

        # Position axis from left
        if axis >= 0:
            a1 = axis
        else:
            a1 = axis + self.__nrank_
        if a1 < 0 or a1 >= self.__nrank_:
            raise ValueError('axis is out of range (%d,%d): %d',
                             (-self.__nrank_, self.__nrank_, axis))
        k1 = len(self.__shape_) + a1

        # Roll this axis to the beginning and slice it out
        new_values = np.rollaxis(self.__values_, k1, 0)
        new_values = new_values[index]

        # Construct and cast
        obj = Qube(new_values, derivs={}, nrank=self.__nrank_-1, example=self)
        obj = obj.cast(classes)
        obj.__readonly_ = self.__readonly_

        # Slice the derivatives if necessary
        if recursive:
            for (key, deriv) in self.__derivs_.iteritems():
                obj.insert_deriv(key, deriv.extract_numer(a1, index, classes,
                                                          False))

        return obj

    def slice_numer(self, axis, index1, index2, classes=(), recursive=True):
        """Return an object sliced from one numerator axis.

        Input:
            axis        the item axis from which to extract a slice.
            index1      the starting index value at which to extract the slice.
            index2      the ending index value at which to extract the slice.
            classes     a single class or list or tuple of classes. The class
                        of the object returned will be the first suitable class
                        in the list. Otherwise, a generic Qube object will be
                        returned.
            recursive   True to include matching slices of the derivatives in
                        the returned object; otherwise, the returned object will
                        not contain derivatives.
        """

        # Position axis from left
        if axis >= 0:
            a1 = axis
        else:
            a1 = axis + self.__nrank_
        if a1 < 0 or a1 >= self.__nrank_:
            raise ValueError('axis is out of range (%d,%d): %d',
                             (-self.__nrank_, self.__nrank_, axis))
        k1 = len(self.__shape_) + a1

        # Roll this axis to the beginning and slice it out
        new_values = np.rollaxis(self.__values_, k1, 0)
        new_values = new_values[index1:index2]
        new_values = np.rollaxis(new_values, 0, k1+1)

        # Construct and cast
        obj = Qube(new_values, derivs={}, example=self)
        obj = obj.cast(classes)
        obj.__readonly_ = self.__readonly_

        # Slice the derivatives if necessary
        if recursive:
            for (key, deriv) in self.__derivs_.iteritems():
                obj.insert_deriv(key, deriv.slice_numer(a1, index1, index2,
                                                        classes, False))

        return obj

    ############################################################################
    # Numerator shaping operations
    ############################################################################

    def transpose_numer(self, axis1=0, axis2=1, recursive=True):
        """Return a copy of this object with two numerator axes transposed.

        Inputs:
            axis1       the first axis to transpose from among the numerator
                        axes. Negative values count backward from the last
                        numerator axis.
            axis2       the second axis to transpose.
            recursive   True to transpose the same axes of the derivatives;
                        False to return an object without derivatives.
        """

        len_shape = len(self.__shape_)

        # Position axis1 from left
        if axis1 >= 0:
            a1 = axis1
        else:
            a1 = axis1 + self.__nrank_
        if a1 < 0 or a1 >= self.__nrank_:
            raise ValueError('first axis is out of range (%d,%d): %d',
                             (-self.__nrank_, self.__nrank_, axis1))
        k1 = len_shape + a1

        # Position axis2 from item left
        if axis2 >= 0:
            a2 = axis2
        else:
            a2 = axis2 + self.__nrank_
        if a2 < 0 or a2 >= self.__nrank_:
            raise ValueError('second axis out of range (%d,%d): %d',
                             (-self.__nrank_, self.__nrank_, axis2))
        k2 = len_shape + a2

        # Swap the axes
        new_values = np.swapaxes(self.__values_, k1, k2)

        # Construct the result
        obj = Qube.__new__(type(self))
        obj.__init__(new_values, derivs={}, example=self)
        obj.__readonly_ = self.__readonly_

        if recursive:
            for (key, deriv) in self.__derivs_.iteritems():
                obj.insert_deriv(key, deriv.transpose_numer(a1, a2, False))

        return obj

    def reshape_numer(self, shape, classes=(), recursive=True):
        """Return this object with a new shape for numerator items.

        Input:
            shape       the new shape.
            classes     a single class or list or tuple of classes. The class
                        of the object returned will be the first suitable class
                        in the list. Otherwise, a generic Qube object will be
                        returned.
            recursive   True to reshape the derivatives in the same way;
                        otherwise, the returned object will not contain
                        derivatives.
        """

        # Validate the shape
        shape = tuple(shape)
        if self.nsize != int(np.prod(shape)):
            raise ValueError('item size must be unchanged: %s, %s' %
                             (str(self.__numer_), str(shape)))

        # Reshape
        full_shape = self.__shape_ + shape + self.__denom_
        new_values = self.__values_.reshape(full_shape)

        # Construct and cast
        obj = Qube(new_values, derivs={}, nrank=len(shape), example=self)
        obj = obj.cast(classes)
        obj.__readonly_ = self.__readonly_

        # Reshape the derivatives if necessary
        if recursive:
          for (key, deriv) in self.__derivs_.iteritems():
            obj.insert_deriv(key, deriv.reshape_numer(shape, classes, False))

        return obj

    def flatten_numer(self, classes=(), recursive=True):
        """Return this object with a new numerator shape such that nrank == 1.

        Input:
            classes     a single class or list or tuple of classes. The class
                        of the object returned will be the first suitable class
                        in the list. Otherwise, a generic Qube object will be
                        returned.
            recursive   True to include matching slices of the derivatives in
                        the returned object; otherwise, the returned object will
                        not contain derivatives.
        """

        return self.reshape_numer((self.nsize,), classes, recursive)

    ############################################################################
    # Denominator shaping operations
    ############################################################################

    def transpose_denom(self, axis1=0, axis2=1):
        """Return a copy of this object with two denominator axes transposed.

        Inputs:
            axis1       the first axis to transpose from among the denominator
                        axes. Negative values count backward from the last
                        axis.
            axis2       the second axis to transpose.
        """

        len_shape = len(self.__shape_)

        # Position axis1 from left
        if axis1 >= 0:
            a1 = axis1
        else:
            a1 = axis1 + self.__drank_
        if a1 < 0 or a1 >= self.__drank_:
            raise ValueError('first axis is out of range (%d,%d): %d',
                             (-self.__drank_, self.__drank_, axis1))
        k1 = len_shape + self.__nrank_ + a1

        # Position axis2 from item left
        if axis2 >= 0:
            a2 = axis2
        else:
            a2 = axis2 + self.__drank_
        if a2 < 0 or a2 >= self.__drank_:
            raise ValueError('second axis out of range (%d,%d): %d',
                             (-self.__drank_, self.__drank_, axis2))
        k2 = len_shape + self.__nrank_ + a2

        # Swap the axes
        new_values = np.swapaxes(self.__values_, k1, k2)

        # Construct the result
        obj = Qube.__new__(type(self))
        obj.__init__(new_values, derivs={}, example=self)
        obj.__readonly_ = self.__readonly_

        return obj

    def reshape_denom(self, shape):
        """Return this object with a new shape for denominator items.

        Input:
            shape       the new denominator shape.
        """

        # Validate the shape
        shape = tuple(shape)
        if self.dsize != int(np.prod(shape)):
            raise ValueError('denominator size must be unchanged: %s, %s' %
                             (str(self.__denom_), str(shape)))

        # Reshape
        full_shape = self.__shape_ + self.__numer_ + shape
        new_values = self.__values_.reshape(full_shape)

        # Construct and cast
        obj = Qube.__new__(type(self))
        Qube.__init__(obj, new_values, derivs={}, drank=len(shape),
                           example=self)
        obj.__readonly_ = self.__readonly_

        return obj

    def flatten_denom(self):
        """Return this object with a new denominator shape such that drank == 1.
        """

        return self.reshape_denom((self.dsize,))

    ############################################################################
    # Numerator/denominator operations
    ############################################################################

    def join_items(self, classes):
        """Return the object with denominator axes joined to the numerator.

        Derivatives are removed.

        Input:
            classes     either a single subclass of Qube or a list or tuple of
                        subclasses. The returned object will be an instance of
                        the first suitable subclass in the list.
        """

        if not self.__drank_: return self.without_derivs()

        obj = Qube(derivs={}, nrank=(self.__nrank_ + self.__drank_), drank=0,
                   example=self)
        obj = obj.cast(classes)
        obj.__readonly_ = self.__readonly_

        return obj

    def split_items(self, nrank, classes):
        """Return the object with numerator axes converted to denominator axes.

        Derivatives are removed.

        Input:
            classes     either a single subclass of Qube or a list or tuple of
                        subclasses. The returned object will be an instance of
                        the first suitable subclass in the list.
        """

        obj = Qube(derivs={}, nrank=nrank, drank=(self.__rank_ - nrank),
                   example=self)
        obj = obj.cast(classes)
        obj.__readonly_ = self.__readonly_

        return obj

    def swap_items(self, classes):
        """A new object with the numerator and denominator axes exchanged.

        Derivatives are removed.

        Input:
            classes     either a single subclass of Qube or a list or tuple of
                        subclasses. The returned object will be an instance of
                        the first suitable subclass in the list.
        """

        new_values = self.__values_
        len_shape = len(new_values.shape)

        for r in range(self.__nrank_):
            new_values = np.rollaxis(new_values, -self.__drank_ - 1, len_shape)

        obj = Qube(new_values, derivs={},
                   nrank=self.__drank_, drank=self.__nrank_, example=self)
        obj = obj.cast(classes)
        obj.__readonly_ = self.__readonly_

        return obj

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

    ############################################################################
    # Unary operators
    ############################################################################

    def __pos__(self):
        return self.clone(recursive=True)

    def __neg__(self, recursive=True):

        # Construct a copy with negative values
        obj = self.clone(recursive=False)
        obj._set_values_(-self.__values_)

        # Fill in the negative derivatives
        if recursive and self.__derivs_:
            for (key, deriv) in self.__derivs_.iteritems():
                obj.insert_deriv(key, -deriv)

        return obj

    def __abs__(self, recursive=True):

        # Check rank
        if self.__nrank_ != 0:
            Qube.raise_unsupported_op('abs()', self)

        # Construct a copy with absolute values
        obj = self.clone(recursive=False)
        obj._set_values_(np.abs(self.__values_))

        # Fill in the derivatives, multiplied by sign(self)
        if recursive and self.__derivs_:
            sign = self.without_derivs().sign()
            for (key, deriv) in self.__derivs_.iteritems():
                obj.insert_deriv(key, deriv * sign)

        return obj

    ############################################################################
    # Addition
    ############################################################################

    # Default method for left addition, element by element
    def __add__(self, arg, recursive=True):

        if Qube.is_empty(arg): return arg

        # Convert arg to another Qube if necessary
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = self.as_this_type(arg)
            except:
                Qube.raise_unsupported_op('+', self, original_arg)

        # Verify compatibility
        if not Units.can_match(self.__units_, arg.__units_):
            raise ValueError('operand units are incompatible: %s, %s' %
                             (str(self.__units_), str(arg.__units_)))

        if self.__numer_ != arg.__numer_:
            if type(self) != type(arg):
                Qube.raise_unsupported_op('+', self, original_arg)

            raise ValueError('item shapes are incompatible: %s, %s' %
                             (str(self.__numer_), str(arg.__numer_)))

        if self.__denom_ != arg.__denom_:
            raise ValueError('denominator shapes are incompatible: %s, %s' %
                             (str(self.__denom_), str(arg.__denom_)))

        # Construct the result
        obj = Qube.__new__(type(self))
        obj.__init__(self.__values_ + arg.__values_, self.__mask_ | arg.__mask_,
                     self.__units_ or arg.__units_, derivs={}, example=self)

        if recursive:
            obj.insert_derivs(self.add_derivs(arg))

        return obj

    # Default method for right addition, element by element
    def __radd__(self, arg, recursive=True):
        return self.__add__(arg, recursive)

    # Default method for in-place addition, element by element
    def __iadd__(self, arg):
        self.require_writable()
        if Qube.is_empty(arg): return arg

        # Convert arg to another Qube if necessary
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = self.as_this_type(arg)
            except:
                Qube.raise_unsupported_op('+=', self, original_arg)

        # Verify compatibility
        if not Units.can_match(self.__units_, arg.__units_):
            raise ValueError('operand units are incompatible: %s, %s' %
                             (str(self.__units_), str(arg.__units_)))

        if self.__numer_ != arg.__numer_:
            if type(self) != type(arg):
                Qube.raise_unsupported_op('+=', self, original_arg)

            raise ValueError('item shapes are incompatible: %s, %s' %
                             (str(self.__numer_), str(arg.__numer_)))

        if self.__denom_ != arg.__denom_:
            raise ValueError('denominator shapes are incompatible: %s, %s' %
                             (str(self.__denom_), str(arg.__denom_)))

        # Perform the operation
        if self.is_int() and not arg.is_int():
            raise TypeError('"+=" operation returns non-integer result')

        new_derivs = self.add_derivs(arg)   # if this raises exception, stop
        self.__values_ += arg.__values_     # on exception here, no harm done
        self.__mask_ = self.__mask_ | arg.__mask_
        self.__units_ = self.__units_ or arg.__units_
        self.insert_derivs(new_derivs)

        self.__antimask_ = None
        self.__corners_  = None
        self.__slicer_   = None
        return self

    def add_derivs(self, arg):
        """Return a dictionary of added derivatives."""

        set1 = set(self.__derivs_)
        set2 = set(arg.__derivs_)
        set12 = set1 & set2
        set1 -= set12
        set2 -= set12

        new_derivs = {}
        for key in set12:
            new_derivs[key] = self.__derivs_[key] + arg.__derivs_[key]
        for key in set1:
            new_derivs[key] = self.__derivs_[key]
        for key in set2:
            new_derivs[key] = arg.__derivs_[key]

        return new_derivs

    ############################################################################
    # Subtraction
    ############################################################################

    # Default method for left subtraction, element by element
    def __sub__(self, arg, recursive=True):

        if Qube.is_empty(arg): return arg

        # Convert arg to the same subclass and try again
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = self.as_this_type(arg)
            except:
                Qube.raise_unsupported_op('-', self, original_arg)

        # Verify compatibility
        if not Units.can_match(self.__units_, arg.__units_):
            raise ValueError('operand units are incompatible: %s, %s' %
                             (str(self.__units_), str(arg.__units_)))

        if self.__numer_ != arg.__numer_:
            if type(self) != type(arg):
                Qube.raise_unsupported_op('-', self, original_arg)

            raise ValueError('item shapes are incompatible: %s, %s' %
                             (str(self.__numer_), str(arg.__numer_)))

        if self.__denom_ != arg.__denom_:
            raise ValueError('denominator shapes are incompatible: %s, %s' %
                             (str(self.__denom_), str(arg.__denom_)))

        # Construct the result
        obj = Qube.__new__(type(self))
        obj.__init__(self.__values_ - arg.__values_, self.__mask_ | arg.__mask_,
                     self.__units_ or arg.__units_, derivs={}, example=self)

        if recursive:
            obj.insert_derivs(self.sub_derivs(arg))

        return obj

    # Default method for right subtraction, element by element
    def __rsub__(self, arg, recursive=True):

        # Convert arg to the same subclass and try again
        if not isinstance(arg, Qube):
            arg = self.as_this_type(arg)
            return arg.__sub__(self, recursive)

    # In-place subtraction
    def __isub__(self, arg):
        self.require_writable()
        if Qube.is_empty(arg): return arg

        # Convert arg to another Qube if necessary
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = self.as_this_type(arg)
            except:
                Qube.raise_unsupported_op('-=', self, original_arg)

        # Verify compatibility
        if not Units.can_match(self.__units_, arg.__units_):
            raise ValueError('operand units are incompatible: %s, %s' %
                             (str(self.__units_), str(arg.__units_)))

        if self.__numer_ != arg.__numer_:
            if type(self) != type(arg):
                Qube.raise_unsupported_op('-=', self, original_arg)

            raise ValueError('item shapes are incompatible: %s, %s' %
                             (str(self.__numer_), str(arg.__numer_)))

        if self.__denom_ != arg.__denom_:
            raise ValueError('denominator shapes are incompatible: %s, %s' %
                             (str(self.__denom_), str(arg.__denom_)))

        # Perform the operation
        if self.is_int() and not arg.is_int():
            raise TypeError('"-=" operation returns non-integer result')

        new_derivs = self.sub_derivs(arg)   # if this raises exception, stop
        self.__values_ -= arg.__values_     # on exception here, no harm done
        self.__mask_ = self.__mask_ | arg.__mask_
        self.__units_ = self.__units_ or arg.__units_
        self.insert_derivs(new_derivs)

        self.__antimask_ = None
        self.__corners_  = None
        self.__slicer_   = None
        return self

    def sub_derivs(self, arg):
        """Return a dictionary of subtracted derivatives."""

        set1 = set(self.__derivs_)
        set2 = set(arg.__derivs_)
        set12 = set1 & set2
        set1 -= set12
        set2 -= set12

        new_derivs = {}
        for key in set12:
            new_derivs[key] = self.__derivs_[key] - arg.__derivs_[key]
        for key in set1:
            new_derivs[key] = self.__derivs_[key]
        for key in set2:
            new_derivs[key] = -arg.__derivs_[key]

        return new_derivs

    ############################################################################
    # Multiplication
    ############################################################################

    # Generic left multiplication
    def __mul__(self, arg, recursive=True):

        if Qube.is_empty(arg): return arg

        # Handle multiplication by a number
        if Qube.is_real_number(arg):
            return self.mul_by_number(arg, recursive)

        # Convert arg to a Scalar if necessary
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = Qube.SCALAR_CLASS.as_scalar(arg)
            except:
                Qube.raise_unsupported_op('*', self, original_arg)

        # Check denominators
        if self.__drank_ and arg.__drank_:
            raise ValueError("dual operand denominators for '*': %s, %s" %
                             (str(self.__denom_), str(arg.__denom_)))

        # Multiply by scalar...
        if arg.__nrank_ == 0:
            try:
                return self.mul_by_scalar(arg, recursive)
 
            # Revise the exception if the arg was modified
            except:
                if arg is not original_arg:
                    Qube.raise_unsupported_op('*', self, original_arg)
                raise

        # Swap and try again
        if self.__nrank_ == 0:
            return arg.mul_by_scalar(self, recursive)

        # Multiply by matrix...
        if self.__nrank_ == 2 and arg.__nrank_ in (1,2):
            return Qube.dot(self, arg, -1, 0, (type(arg), type(self)),
                            recursive)

        # Give up
        Qube.raise_unsupported_op('*', self, original_arg)

    # Generic right multiplication
    def __rmul__(self, arg, recursive=True):

        # Handle multiplication by a number
        if Qube.is_real_number(arg):
            return self.mul_by_number(arg, recursive)

        # Convert arg to a Scalar and try again
        original_arg = arg
        try:
            arg = Qube.SCALAR_CLASS.as_scalar(arg)
            return self.mul_by_scalar(arg, recursive)

        # Revise the exception if the arg was modified
        except:
            if arg is not original_arg:
                Qube.raise_unsupported_op('*', original_arg, self)
            raise

    # Generic in-place multiplication
    def __imul__(self, arg):
        self.require_writable()
        if Qube.is_empty(arg): return arg

        # Convert arg to a Scalar if necessary
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = Qube.SCALAR_CLASS.as_scalar(arg)
            except:
                Qube.raise_unsupported_op('*=', self, original_arg)

        # Scalar case
        if arg.__rank_ == 0:

            # Align axes
            arg_values = arg.__values_
            if self.__rank_ and np.shape(arg_values) != ():
                arg_values = arg_values.reshape(np.shape(arg_values) +
                                                self.__rank_ * (1,))

            # Multiply...
            if self.is_int() and not arg.is_int():
                raise TypeError('"*=" operation returns non-integer result')

            new_derivs = self.mul_derivs(arg)   # on exception, stop
            self.__values_ *= arg_values        # on exception, no harm done
            self.__mask_ = self.__mask_ | arg.__mask_
            self.__units_ = Units.mul_units(self.__units_, arg.__units_)
            self.insert_derivs(new_derivs)

            self.__antimask_ = None
            self.__corners_  = None
            self.__slicer_   = None
            return self

        # Matrix multiply case
        if self.__nrank_ == 2 and arg.__nrank_ == 2 and arg.__drank_ == 0:
            result = Qube.dot(self, arg, -1, 0, type(self), recursive=True)
            self._set_values_(result.__values_, result.__mask_)
            self.insert_derivs(result.__derivs_)
            return self

        # return NotImplemented
        Qube.raise_unsupported_op('*=', self, original_arg)

    def mul_by_number(self, arg, recursive=True):
        """Internal multiply op when the arg is a Python scalar."""

        obj = self.clone(recursive=False)
        obj._set_values_(self.__values_ * arg)

        if recursive and self.__derivs_:
            for (key, deriv) in self.__derivs_.iteritems():
                obj.insert_deriv(key, deriv.mul_by_number(arg, False))

        return obj

    def mul_by_scalar(self, arg, recursive=True):
        """Internal multiply op when the arg is a Qube with nrank == 0 and no
        more than one object has a denominator."""

        # Align axes
        self_values = self.__values_
        self_shape = np.shape(self_values)
        if arg.__drank_ > 0 and self_shape != ():
            self_values = self_values.reshape(self_shape + arg.__drank_ * (1,))

        arg_values = arg.__values_
        arg_shape = (arg.__shape_ + self.__rank_ * (1,) + arg.__denom_)
        if np.shape(arg_values) not in ((), arg_shape):
            arg_values = arg_values.reshape(arg_shape)

        # Construct object
        obj = Qube.__new__(type(self))
        obj.__init__(self_values * arg_values,
                     self.__mask_ | arg.__mask_,
                     Units.mul_units(self.__units_, arg.__units_),
                     derivs = {},
                     drank = max(self.__drank_, arg.__drank_),
                     example = self)

        obj.insert_derivs(self.mul_derivs(arg))

        return obj

    def mul_derivs(self, arg):
        """Return a dictionary of multiplied derivatives."""

        new_derivs = {}

        if self.__derivs_:
            arg_wod = arg.without_derivs()
            for (key, self_deriv) in self.__derivs_.iteritems():
                new_derivs[key] = self_deriv * arg_wod

        if arg.__derivs_:
            self_wod = self.without_derivs()
            for (key, arg_deriv) in arg.__derivs_.iteritems():
                term = self_wod * arg_deriv
                if key in new_derivs:
                    new_derivs[key] = new_derivs[key] + term
                else:
                    new_derivs[key] = term

        return new_derivs

    ############################################################################
    # Division
    ############################################################################

    def __div__(self, arg, recursive=True):
        return self.__truediv__(arg, recursive)

    def __rdiv__(self, arg, recursive=True):
        return self.__rtruediv__(arg)

    def __idiv__(self, arg, recursive=True):
        return self.__itruediv__(arg)

    # Generic left true division
    def __truediv__(self, arg, recursive=True):
        """Cases of divide-by-zero are masked."""

        if Qube.is_empty(arg): return arg

        # Handle division by a number
        if Qube.is_real_number(arg):
            return self.div_by_number(arg, recursive)

        # Convert arg to a Scalar if necessary
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = Qube.SCALAR_CLASS.as_scalar(arg)
            except:
                Qube.raise_unsupported_op('/', self, original_arg)

        # Check right denominator
        if arg.__drank_ > 0:
            raise ValueError("right operand denominator for '/': %s" %
                             str(arg.__denom_))

        # Divide by scalar...
        if arg.__nrank_ == 0:
            try:
                return self.div_by_scalar(arg, recursive)
 
            # Revise the exception if the arg was modified
            except:
                if arg is not original_arg:
                    Qube.raise_unsupported_op('/', self, original_arg)
                raise

        # Swap and multiply by reciprocal...
        if self.__nrank_ == 0:
            return self.reciprocal(recursive).mul_by_scalar(arg, recursive)

        # Matrix / matrix is multiply by inverse matrix
        if self.__rank_ == 2 and arg.__rank_ == 2:
            return self.__mul__(arg.reciprocal(recursive))

        # Give up
        Qube.raise_unsupported_op('/', self, original_arg)

    # Generic right division
    def __rtruediv__(self, arg, recursive=True):

        # Handle right division by a number
        if Qube.is_real_number(arg):
            return self.reciprocal(recursive).__mul__(arg, recursive)

        # Convert arg to a Scalar and try again
        original_arg = arg
        try:
            arg = Qube.SCALAR_CLASS.as_scalar(arg)
            return arg.__div__(self, recursive)

        # Revise the exception if the arg was modified
        except:
            if arg is not original_arg:
                Qube.raise_unsupported_op('/', original_arg, self)
            raise

    # Generic in-place division
    def __itruediv__(self, arg):
        if Qube.is_empty(arg): return arg

        if not self.is_float():
            raise TypeError('"/=" operation returns non-integer result')

        self.require_writable()

        # Convert arg to a Scalar if necessary
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = Qube.SCALAR_CLASS.as_scalar(arg)
            except:
                Qube.raise_unsupported_op('/=', self, original_arg)

        # In-place multiply by the reciprocal
        try:
            return self.__imul__(arg.reciprocal())

        # Revise the exception if the arg was modified
        except:
            if arg is not original_arg:
                Qube.raise_unsupported_op('/=', self, original_arg)
            raise

    def div_by_number(self, arg, recursive=True):
        """Internal division op when the arg is a Python scalar."""

        obj = self.clone(recursive=False)

        # Mask out zeros
        if arg == 0:
            obj._set_mask_(True)
        else:
            obj._set_values_(self.__values_ / arg)

        if recursive and self.__derivs_:
            for (key, deriv) in self.__derivs_.iteritems():
                obj.insert_deriv(key, deriv.div_by_number(arg, False))

        return obj

    def div_by_scalar(self, arg, recursive):
        """Internal division op when the arg is a Qube with rank == 0."""

        # Mask out zeros
        arg = arg.mask_where_eq(0.,1.)

        # Align axes
        arg_values = arg.__values_
        if np.shape(arg_values) != () and self.__rank_:
            arg_values = arg_values.reshape(arg.shape + self.__rank_ * (1,))

        # Construct object
        obj = Qube.__new__(type(self))
        obj.__init__(self.__values_ / arg_values,
                     self.__mask_ | arg.__mask_,
                     Units.div_units(self.__units_, arg.__units_),
                     derivs = {},
                     example = self)

        if recursive:
            obj.insert_derivs(self.div_derivs(arg, nozeros=True))

        return obj

    def div_derivs(self, arg, nozeros=False):
        """Return a dictionary of divided derivatives.

        if nozeros is True, the arg is assumed not to contain any zeros, so
        divide-by-zero errors are not checked."""

        new_derivs = {}

        if not self.__derivs_ and not arg.__derivs_:
            return new_derivs

        if not nozeros:
            arg = arg.mask_where_eq(0., 1.)

        arg_wod_inv = arg.without_derivs().reciprocal(nozeros=True)

        for (key, self_deriv) in self.__derivs_.iteritems():
            new_derivs[key] = self_deriv * arg_wod_inv

        if arg.__derivs_:
            self_wod = self.without_derivs()
            for (key, arg_deriv) in arg.__derivs_.iteritems():
                term = self_wod * (arg_deriv * arg_wod_inv**2)
                if key in new_derivs:
                    new_derivs[key] -= term
                else:
                    new_derivs[key] = -term

        return new_derivs

    ############################################################################
    # Floor Division (with no support for derivatives)
    ############################################################################

    # Generic left floor division
    def __floordiv__(self, arg):
        """Cases of divide-by-zero become masked. Derivatives are ignored."""

        if Qube.is_empty(arg): return arg

        # Convert arg to a Scalar if necessary
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = Qube.SCALAR_CLASS.as_scalar(arg)
            except:
                Qube.raise_unsupported_op('//', self, original_arg)

        # Check right denominator
        if arg.__drank_ > 0:
            raise ValueError("right operand denominator for '//': %s" %
                             str(arg.__denom_))

        # Floor divide by scalar...
        if arg.__nrank_ == 0:
            try:
                return self.floordiv_by_scalar(arg)

            # Revise the exception if the arg was modified
            except:
                if arg is not original_arg:
                    Qube.raise_unsupported_op('//', original_arg, self)
                raise

        # Give up
        Qube.raise_unsupported_op('//', self, original_arg)

    # Generic right floor division
    def __rfloordiv__(self, arg):

        # Convert arg to a Scalar and try again
        original_arg = arg
        try:
            arg = Qube.SCALAR_CLASS.as_scalar(arg)
            return arg.__floordiv__(self)

        # Revise the exception if the arg was modified
        except:
            if arg is not original_arg:
                Qube.raise_unsupported_op('//', original_arg, self)
            raise

   # Generic in-place floor division
    def __ifloordiv__(self, arg):
        self.require_writable()
        if Qube.is_empty(arg): return arg

        # Convert arg to a Scalar if necessary
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = Qube.SCALAR_CLASS.as_scalar(arg)
            except:
                Qube.raise_unsupported_op('//=', self, original_arg)

        # Handle floor division by a scalar
        if arg.__rank_ == 0:
            divisor = arg.mask_where_eq(0, 1)
            div_values = divisor.__values_

            # Align axes
            if self.__rank_:
                div_values = np.reshape(div_values, np.shape(div_values) +
                                                    self.__rank_ * (1,))
            self.__values_ //= div_values
            self.__mask_ = self.__mask_ | divisor.__mask_
            self.__units_ = Units.div_units(self.__units_, arg.__units_)
            self.delete_derivs()

            self.__antimask_ = None
            self.__corners_  = None
            self.__slicer_   = None
            return self

        # Nothing else is implemented
        # return NotImplemented
        Qube.raise_unsupported_op('//=', self, original_arg)

    def floordiv_by_number(self, arg):
        """Internal floor division op when the arg is a Python scalar."""

        obj = self.clone(recursive=False)

        if arg == 0:
            obj._set_mask_(True)
        else:
            obj._set_values_(self.__values_ // arg)

        return obj

    def floordiv_by_scalar(self, arg):
        """Internal floor division op when the arg is a Qube with nrank == 0.

        The arg cannot have a denominator."""

        # Mask out zeros
        arg = arg.mask_where_eq(0,1)

        # Align axes
        arg_values = arg.__values_
        if np.shape(arg_values) != () and self.__rank_:
            arg_values = arg_values.reshape(arg.shape + self.__rank_ * (1,))

        # Construct object
        obj = Qube.__new__(type(self))
        obj.__init__(self.__values_ // arg_values,
                     self.__mask_ | arg.__mask_,
                     Units.div_units(self.__units_, arg.__units_),
                     derivs = {},
                     example = self)
        return obj

    ############################################################################
    # Modulus operators (with no support for derivatives)
    ############################################################################

    # Generic left modulus
    def __mod__(self, arg, recursive=True):
        """Cases of divide-by-zero become masked. Derivatives in the numerator
        are supported, but not in the denominator."""

        if Qube.is_empty(arg): return arg

        # Handle modulus by a number
        if Qube.is_real_number(arg):
            return self.mod_by_number(arg, recursive)

        # Convert arg to a Scalar if necessary
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = Qube.SCALAR_CLASS.as_scalar(arg)
            except:
                Qube.raise_unsupported_op('%', self, original_arg)

        # Check right denominator
        if arg.__drank_ > 0:
            raise ValueError("right operand denominator for '%': %s" %
                             str(arg.__denom_))

        # Modulus by scalar...
        if arg.__nrank_ == 0:
            try:
                return self.mod_by_scalar(arg, recursive)

            # Revise the exception if the arg was modified
            except:
                if arg is not original_arg:
                    Qube.raise_unsupported_op('%', self, original_arg)
                raise

        # Give up
        Qube.raise_unsupported_op('%', self, original_arg)

    # Generic right modulus
    def __rmod__(self, arg):

        # Convert arg to a Scalar and try again
        original_arg = arg
        try:
            arg = Qube.SCALAR_CLASS.as_scalar(arg)
            return arg.__mod__(self)

        # Revise the exception if the arg was modified
        except:
            if arg is not original_arg:
                Qube.raise_unsupported_op('%', original_arg, self)
            raise

    # Generic in-place modulus
    def __imod__(self, arg):
        self.require_writable()
        if Qube.is_empty(arg): return arg

        # Convert arg to a Scalar if necessary
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = Qube.SCALAR_CLASS.as_scalar(arg)
            except:
                Qube.raise_unsupported_op('%=', self, original_arg)

        # Handle modulus by a scalar
        if arg.__rank_ == 0:
            divisor = arg.mask_where_eq(0, 1)
            div_values = divisor.__values_

            # Align axes
            if self.__rank_:
                div_values = np.reshape(div_values, np.shape(div_values) +
                                                    self.__rank_ * (1,))
            self.__values_ %= div_values
            self.__mask_ = self.__mask_ | divisor.__mask_
            self.__units_ = Units.div_units(self.__units_, arg.__units_)

            self.__antimask_ = None
            self.__corners_  = None
            self.__slicer_   = None
            return self

        # Nothing else is implemented
        # return NotImplemented
        Qube.raise_unsupported_op('%=', self, original_arg)

    def mod_by_number(self, arg, recursive=True):
        """Internal modulus op when the arg is a Python scalar."""

        obj = self.clone(recursive=False)

        # Mask out zeros
        if arg == 0:
            obj._set_mask_(True)
        else:
            obj._set_values_(self.__values_ % arg)

        if recursive and self.__derivs_:
            for (key, deriv) in self.__derivs_.iteritems():
                obj.insert_deriv(key, deriv)

        return obj

    def mod_by_scalar(self, arg, recursive=True):
        """Internal modulus op when the arg is a Qube with nrank == 0 and the
        arg has no denominator."""

        # Mask out zeros
        arg = arg.without_derivs().mask_where_eq(0,1)

        # Align axes
        arg_values = arg.__values_
        if np.shape(arg_values) != () and self.__rank_:
            arg_values = arg_values.reshape(arg.shape + self.__rank_ * (1,))

        # Construct the object
        obj = Qube.__new__(type(self))
        obj.__init__(self.__values_ % arg_values,
                     self.__mask_ | arg.__mask_,
                     Units.div_units(self.__units_, arg.__units_),
                     derivs = {},
                     example = self)

        if recursive and self.__derivs_:
            for (key, deriv) in self.__derivs_.iteritems():
                obj.insert_deriv(key, deriv)

        return obj

    ############################################################################
    # Exponentiation operators
    ############################################################################

    # Generic exponentiation, PolyMath scalar to a single scalar power
    def __pow__(self, expo, recursive=True):
        if Qube.is_empty(expo): return expo

        # Arrays of exponents are not supported
        if isinstance(expo, Qube):
            if expo.__mask_:
                Qube.raise_unsupported_op('**', self, expo)

            expo = expo.__values_

        if np.shape(expo):
            Qube.raise_unsupported_op('**', self, expo)

        if self.__rank_ != 0:
            Qube.raise_unsupported_op('**', self, expo)

        # Replace zeros or negatives depending on exponent
        no_negs_allowed = (expo != int(expo))
        no_zeros_allowed = (expo < 0)

        # Without this step, negative int exponents on int values truncate to 0
        if expo == int(expo) and expo < 0:
            expo = float(expo)

        arg = self
        if no_negs_allowed:
            if no_zeros_allowed:
                arg = arg.mask_where_le(0., 1.)
            else:
                arg = arg.mask_where_lt(0., 1.)
        elif no_zeros_allowed:
                arg = arg.mask_where_eq(0., 1.)

        obj = arg.clone(recursive=False)
        obj._set_values_(arg.__values_**expo)
        obj.__units_ = Units.units_power(self.__units_, expo)

        # Evaluate the derivatives if necessary
        if recursive and self.__derivs_:
            factor = expo * arg.without_derivs()**(expo-1.)
            for (key, deriv) in self.__derivs_.iteritems():
                obj.insert_deriv(key, factor * deriv)

        return obj

    ############################################################################
    # Comparison operators, returning boolean scalars or Booleans
    ############################################################################

    def __eq__(self, arg):

        # If the subclasses cannot be unified, raise a ValueError
        if not isinstance(arg, type(self)):
            try:
                obj = Qube.__new__(type(self))
                obj.__init__(arg)
                arg = obj
            except:
                return False

        # Compare the values and masks
        compare = (self.__values_ == arg.__values_)
        for r in range(self.__rank_):
            compare = np.all(compare, axis=-1)

        both_masked = (self.__mask_ & arg.__mask_)
        one_masked  = (self.__mask_ ^ arg.__mask_)

        # Compare units for compatibility
        if not Units.can_match(self.__units_, arg.__units_):
            one_masked = True

        # Return a Python bool if the shape is ()
        if np.shape(compare) == ():
            if one_masked: return False
            if both_masked: return True
            return bool(compare)

        # Apply the mask
        if np.shape(one_masked) == ():
            if one_masked: compare.fill(False)
            if both_masked: compare.fill(True)
        else:
            compare[one_masked] = False
            compare[both_masked] = True

        result = Qube.BOOLEAN_CLASS(compare)
        result.__truth_if_all_ = True
        return result

    def __ne__(self, arg):

        # If the subclasses cannot be unified, raise a ValueError
        if not isinstance(arg, type(self)):
            try:
                obj = Qube.__new__(type(self))
                obj.__init__(arg)
                arg = obj
            except:
                return True

        # Compare the values and masks
        compare = (self.__values_ != arg.__values_)
        for r in range(self.__rank_):
            compare = np.any(compare, axis=-1)

        both_masked = (self.__mask_ & arg.__mask_)
        one_masked = (self.__mask_ ^ arg.__mask_)

        # Compare units for compatibility
        if not Units.can_match(self.__units_, arg.__units_):
            one_masked = True

        # Return a Python bool if the shape is ()
        if np.shape(compare) == ():
            if one_masked: return True
            if both_masked: return False
            return bool(compare)

        # Apply the mask
        if np.shape(one_masked) == ():
            if one_masked: compare.fill(True)
            if both_masked: compare.fill(False)
        else:
            compare[one_masked] = True
            compare[both_masked] = False

        result = Qube.BOOLEAN_CLASS(compare)
        result.__truth_if_any_ = True
        return result

    def __lt__(self, arg):
        raise ValueError('comparison operators are not supported for class ' +
                         type(self).__name__)

    def __gt__(self, arg):
        raise ValueError('comparison operators are not supported for class ' +
                         type(self).__name__)

    def __le__(self, arg):
        raise ValueError('comparison operators are not supported for class ' +
                         type(self).__name__)

    def __gt__(self, arg):
        raise ValueError('comparison operators are not supported for class ' +
                         type(self).__name__)

    def __nonzero__(self):
        """Supports 'if a == b: ...' and 'if a != b: ...' statements.

        Equality requires that every unmasked element of a and
        b be equal, and both object be masked at the same locations.

        Comparison of objects of shape () is also supported.

        Any other comparison of PolyMath object requires an explict call to
        all() or any().
        """

        if self.__truth_if_all_:
            return bool(np.all(self.as_mask_where_nonzero()))

        if self.__truth_if_any_:
            return bool(np.any(self.as_mask_where_nonzero()))

        if self.__shape_:
            raise ValueError('the truth value requires any() or all()')

        if self.__mask_:
            raise ValueError('the truth value of an entirely masked object ' +
                             'is undefined.')

        return bool(np.all(self.as_mask_where_nonzero()))

    ############################################################################
    # Any and all
    ############################################################################

    def all(self, axis=None):
        """Return True if and only if all the matching items are nonzero.

        If the result is a single scalar, it is returned as a Python bool value;
        otherwise it is returned as a Boolean.

        Input:
            axis        an integer axis or a tuple of axes. The all operatation
                        is performed across these axes, leaving any remaining
                        axes in the returned value. If None (the default), then
                        the all operation is performed across all axes if the
                        object.
        """

        self = Qube.BOOLEAN_CLASS.as_boolean(self)

        if self.shape == ():
            result = Qube.BOOLEAN_CLASS(self)

        elif not np.any(self.__mask_):
            result = Qube.BOOLEAN_CLASS(np.all(self.__values_, axis=axis),
                                        mask=False)

        elif axis is None:
            if np.shape(self.__mask_) == ():
                result = Qube.BOOLEAN_CLASS(np.all(self.__values_),
                                            mask=self.__mask_)
            elif np.all(self.__mask_):
                result = Qube.BOOLEAN_CLASS(np.all(self.__values_), mask=True)
            else:
                result = Qube.BOOLEAN_CLASS(np.all(
                                            self.__values_[self.antimask]),
                                            mask=False)

        else:

            # Create new array
            if self.__values_.dtype == np.dtype('bool'):
                new_values = self.__values_.copy()
            else:
                new_values = self.__values_.astype('bool')

            new_values[self.__mask_] = True
            new_values = np.all(new_values, axis=axis)

            # Create new mask
            if self.mask is True:
                new_mask = True
            elif self.mask is False:
                new_mask = False
            else:
                new_mask = np.all(self.__mask_, axis=axis)

            # Use the all of the unmasked values if all are masked
            if np.all(new_mask):
                unmasked_alls = np.all(self.__values_, axis=axis)
                new_values = unmasked_alls
                new_mask = True
            elif np.any(new_mask):
                unmasked_alls = np.all(self.__values_, axis=axis)
                new_values[new_mask] = unmasked_alls[new_mask]
            else:
                new_mask = False

            result = Qube.BOOLEAN_CLASS(new_values, new_mask)

        # Convert result to a Python bool if necessary
        if result.shape == () and not result.__mask_:
            if result:
                return True
            else:
                return False

        return result

    def any(self, axis=None):
        """Return True if any of the matching items are nonzero.

        If the result is a single scalar, it is returned as a Python bool value
        rather than as a Boolean.

        Input:
            axis        an integer axis or a tuple of axes. The any operatation
                        is performed across these axes, leaving any remaining
                        axes in the returned value. If None (the default), then
                        the any operation is performed across all axes if the
                        object.
        """

        self = Qube.BOOLEAN_CLASS.as_boolean(self)

        if self.shape == ():
            result = Qube.BOOLEAN_CLASS(self, derivs={})

        elif not np.any(self.__mask_):
            result = Qube.BOOLEAN_CLASS(np.any(self.__values_, axis=axis),
                                        mask=False)

        elif axis is None:
            if np.shape(self.__mask_) == ():
                result = Qube.BOOLEAN_CLASS(np.any(self.__values_),
                                            mask=self.__mask_)
            elif np.all(self.__mask_):
                result = Qube.BOOLEAN_CLASS(np.any(self.__values_), mask=True)
            else:
                result = Qube.BOOLEAN_CLASS(np.any(
                                            self.__values_[self.antimask]))

        else:

            # Create new array
            if self.__values_.dtype == np.dtype('bool'):
                new_values = self.__values_.copy()
            else:
                new_values = self.__values_.astype('bool')

            new_values[self.__mask_] = False
            new_values = np.any(new_values, axis=axis)

            # Create new mask
            if self.mask is True:
                new_mask = True
            elif self.mask is False:
                new_mask = False
            else:
                new_mask = np.all(self.__mask_, axis=axis)

            # Use the all of the unmasked values if all are masked
            if np.all(new_mask):
                unmasked_anys = np.any(self.__values_, axis=axis)
                new_values = unmasked_anys
                new_mask = True
            elif np.any(new_mask):
                unmasked_anys = np.any(self.__values_, axis=axis)
                new_values[new_mask] = unmasked_anys[new_mask]
            else:
                new_mask = False

            result = Qube.BOOLEAN_CLASS(new_values, new_mask)

        # Convert result to a Python bool if necessary
        if result.shape == () and not result.__mask_:
            if result:
                return True
            else:
                return False

        return result

    ############################################################################
    # Special operators
    ############################################################################

    def reciprocal(self, recursive=True, nozeros=False):
        """Return an object equivalent to the reciprocal of this object.

        This must be overridden by other subclasses.

        Input:
            recursive   True to return the derivatives of the reciprocal too;
                        otherwise, derivatives are removed.
            nozeros     False (the default) to mask out any zero-valued items in
                        this object prior to the divide. Set to True only if you
                        know in advance that this object has no zero-valued
                        items.
        """

        Qube.raise_unsupported_op('reciprocal()', self)

    def zero(self):
        """An object of this subclass containing all zeros.

        The returned object has the same denominator shape as this object.

        This is default behavior and may need to be overridden by some
        subclasses."""

        # Scalar case
        if self.__item_ == ():
            if self.is_float():
                new_value = 0.
            else:
                new_value = 0

        # Array case
        else:
            if self.is_float():
                new_value = np.zeros(self.__item_, dtype='float')
            else:
                new_value = np.zeros(self.__item_, dtype='int')

        # Construct the object
        obj = Qube.__new__(type(self))
        obj.__init__(new_value, False, derivs={}, example=self)

        # Return it as readonly
        return obj.as_readonly()

    def identity(self):
        """An object of this subclass equivalent to the identity.

        This must be overridden by other subclasses.
        """

        Qube.raise_unsupported_op('identity()', self)

    ############################################################################
    # Indexing operators
    ############################################################################

    def __len__(self):
        if len(self.__shape_) > 0:
            return self.__shape_[0]
        else:
            raise TypeError('len of unsized object')

    def __getitem__(self, indx):

        # Interpret and adapt the index
        # (index to apply to values array, index to apply to mask array
        #  index to define additional items to be masked)
        (vals_index, mask_index, new_mask_index) = self.prep_index(indx)

        # A shapeless object cannot be indexed except by booleans
        if self.__shape_ == ():
            if vals_index is True:
                return self

            if vals_index is False:
                return self.as_size_zero()

            raise IndexError('too many indices')

        # Apply index to values
        result_values = self.__values_[vals_index]
        result_vals_shape = np.shape(result_values)

        # Make sure we have not indexed into the item
        if len(result_vals_shape) < self.__rank_:
            raise IndexError('too many indices')

        # Apply index to mask
        if np.shape(self.__mask_) == ():
            result_mask = self.__mask_
        else:
            result_mask = self.__mask_[mask_index]

        # Apply an additional mask if provided
        # Note that this is to be applied after the first mask index.
        if new_mask_index is not None:

            # Get shape of results
            result_mask_shape = result_vals_shape[:len(result_vals_shape) -
                                                   self.__rank_]

            # If everything is newly masked...
            if new_mask_index is True:
                new_mask = True

            # If nothing is newly masked...
            elif new_mask_index is False:
                new_mask = False

            # If the new mask is an array..
            elif np.shape(new_mask_index) == result_mask_shape:
                new_mask = new_mask_index

            # If the new mask is partial..
            else:
                new_mask = np.zeros(result_mask_shape, dtype='bool')
                new_mask[new_mask_index] = True

            result_mask = result_mask | new_mask

        # Construct the object
        obj = Qube.__new__(type(self))
        obj.__init__(result_values, result_mask, derivs={}, example=self)
        obj.__readonly_ = self.__readonly_

        # Apply the same indexing to any derivatives
        for (key, deriv) in self.__derivs_.iteritems():
            obj.insert_deriv(key, deriv[mask_index])

        return obj

    def __setitem__(self, indx, arg):
        self.require_writable()

        # Interpret the arg
        arg = self.as_this_type(arg, recursive=True, nrank=self.nrank,
                                                     drank=self.drank)

        # Derivatives must match
        for key in self.__derivs_:
            if key not in arg.__derivs_:
                raise ValueError('missing derivative d_d%s in replacement' %
                                 key)

        # Interpret and adapt the index
        # (index to apply to values array, index to apply to mask array
        #  index to define additional items to be masked)
        (vals_index, mask_index, new_mask_index) = self.prep_index(indx)

        # A shapeless object cannot be indexed except by True or False
        if self.__shape_ == ():
            if vals_index is True:
                values = self.as_this_type(arg).values
                if isinstance(self.__values_, np.ndarray):
                    values = values.astype(self.__values_.dtype)
                elif isinstance(self.__values_, (bool,np.bool_)):
                    values = bool(values)
                elif isinstance(self.__values_, int):
                    values = int(values)
                else:
                    values = float(values)

                self.__values_   = values
                self.__mask_     = arg.__mask_
                self.__antimask_ = None
                self.__corners_  = None
                self.__slicer_   = None
                return self

            if vals_index is False:
                return self

            raise IndexError('too many indices')

        # Insert the replacement values
        self.__values_[vals_index] = arg.__values_

        # Update the mask if necessary

        # If mask is already an array...
        if np.shape(self.__mask_):
            self.__mask_[mask_index] = arg.__mask_

        # If the replacement is already an array...
        elif np.shape(arg.__mask_):
            maskval = self.__mask_
            self.__mask_ = np.empty(self.shape, dtype='bool')
            self.__mask_.fill(maskval)
            self.__mask_[mask_index] = arg.__mask_

        # If mask is a scalar and can remain that way...
        elif self.__mask_ == arg.__mask_ and new_mask_index is None:
            pass

        # If mask is True but needs to become an array...
        elif self.__mask_:
            self.__mask_ = np.ones(self.shape, dtype='bool')
            self.__mask_[mask_index] = arg.__mask_

        # If mask is False but needs to become an array...
        else:
            self.__mask_ = np.zeros(self.shape, dtype='bool')
            self.__mask_[mask_index] = arg.__mask_

        # Apply the new mask if any...
        if new_mask_index is not None:
            if np.shape(self.__mask_[mask_index]) == ():
                self.__mask_[mask_index] = True
            else:
                # This doesn't work because getitem preceeds setitem
                # self.__mask_[mask_index][new_mask_index] = True
                # This works instead...
                indexed_mask = self.__mask_[mask_index]
                indexed_mask[new_mask_index] = True
                self.__mask_[mask_index] = indexed_mask

        self.__antimask_ = None
        self.__corners_  = None
        self.__slicer_   = None

        # Also update the derivatives (ignoring those not in self)
        for (key, self_deriv) in self.__derivs_.iteritems():
            self.__derivs_[key][mask_index] = arg.__derivs_[key]

        return

    def prep_index(self, indx):
        """Prepare the index for application to a Qube.

        Return: A tuple (vals_index, mask_index, new_mask_index)
            vals_index      index to apply to values array.
            mask_index      index to apply to mask array.
            new_mask_index  index to define additional items to be masked.

        Note that the third index is applied after the second, so it should
        define indexing into the new mask, not the original mask.

        The index is represented by a single object or a list/tuple of objects.
        If the index contains a single Qube object, then we take the mask of
        that object into consideration, and values at the associated indices
        will become masked.

        If the index contains multiple Qube objects, then no masking is allowed.

        If the index contains a Ellipsis, we need to append additional null
        slices for the elements of the values array to account for rank > 0;
        otherwise, the axes will not align properly.

        The code also replaces a Qube with the result of its as_index_and_mask()
        method.
        """

        # If a list or tuple contains only one element, extract it
        if type(indx) in (tuple,list) and len(indx) == 1:
            indx = indx[0]

        # Replace a Qube with its index equivalent
        if isinstance(indx, Qube):
            (mask_index, new_mask_index) = indx.as_index_and_mask()

            # Note that a mask_index works as a vals_index because it operates
            # on leading dimensions of the values array, leaving individual
            # items whole.
            return (mask_index, mask_index, new_mask_index)

        # There can only a problem with indices of type list or tuple
        # By definition these are N-d array references with N > 1
        if type(indx) not in (tuple,list):

            # However, type numpy.bool_ is not a subclass of bool
            if np.shape(indx) == () and type(indx) == np.bool_:
                indx = bool(indx)

            return (indx, indx, None)

        # Search for Ellipses and Qubes
        # Note:
        #   if Ellipsis in index
        # fails when the index contains a NumPy array.

        new_index = []
        has_ellipsis = False
        new_mask_count = 0
        new_mask_index = None
        for item in indx:
            if type(item) == type(Ellipsis):
                has_ellipsis = True

            if isinstance(item, Qube):
                (qube_index, qube_mask_index) = item.as_index_and_mask()
                if qube_mask_index is not None:
                    new_mask_count += 1
                    new_mask_index = qube_mask_index
                    if new_mask_count > 1:
                        raise ValueError("illegal masked index for " +
                                         "multi-dimensional indexing")
                new_index.append(qube_index)
            else:
                new_index.append(item)

        # If an ellipsis appeared, then create safe space for the items in the
        # values array index
        if has_ellipsis:
            value_index = new_index + self.__rank_ * [slice(None)]
        else:
            value_index = new_index

        return (tuple(value_index), tuple(new_index), new_mask_index)

    ############################################################################
    # Utilities for arithmetic operations
    ############################################################################

    @staticmethod
    def raise_unsupported_op(op, obj1, obj2=None):
        """Raise a TypeError or ValueError for unsupported operations."""

        if obj2 is None:
            raise TypeError("bad operand type for %s: '%s'"
                            % (op, type(obj1).__name__))

        if (isinstance(obj1, (list,tuple,np.ndarray)) or
            isinstance(obj2, (list,tuple,np.ndarray))):
                if isinstance(obj1, Qube):
                    shape1 = obj1.__numer_
                else:
                    shape1 = np.shape(obj1)

                if isinstance(obj2, Qube):
                    shape2 = obj2.__numer_
                else:
                    shape2 = np.shape(obj2)

                raise ValueError(("unsupported operand item shapes for '%s': "
                                  "%s, %s") % (op, str(shape1),
                                                       str(shape2)))

        raise TypeError(("unsupported operand types for '%s': " +
                         "'%s', '%s'") % (op, type(obj1).__name__,
                                              type(obj2).__name__))

    ############################################################################
    # Support functions for more math operations
    #
    # These are static methods so as to avoid name conflicts with non-static
    # methods implemented for specific subclasses.
    ############################################################################

    @staticmethod
    def dot(arg1, arg2, axis1=-1, axis2=0, classes=(), recursive=True):
        """Return the dot product of two objects.

        The axes must be in the numerator, and only one of the objects can have
        a denominator (which makes this suitable for first derivatives but not
        second derivatives).

        Input:
            arg1        the first operand as a subclass of Qube.
            arg2        the second operand as a subclass of Qube.
            axis1       the item axis of this object for the dot product.
            axis2       the item axis of the arg2 object for the dot product.
            classes     a single class or list or tuple of classes. The class
                        of the object returned will be the first suitable class
                        in the list. Otherwise, a generic Qube object will be
                        returned.
            recursive   True to construct the derivatives of the dot product.
        """

        # At most one object can have a denominator.
        assert not (arg1.__drank_ and arg2.__drank_)

        # Position axis1 from left
        if axis1 >= 0:
            a1 = axis1
        else:
            a1 = axis1 + arg1.__nrank_
        if a1 < 0 or a1 >= arg1.__nrank_:
            raise ValueError('first axis is out of range (%d,%d): %d' %
                             (-arg1.__nrank_, arg1.__nrank_, axis1))
        k1 = a1 + len(arg1.__shape_)

        # Position axis2 from item left
        if axis2 >= 0:
            a2 = axis2
        else:
            a2 = axis2 + arg2.__nrank_
        if a2 < 0 or a2 >= arg2.__nrank_:
            raise ValueError('second axis out of range (%d,%d): %d' %
                             (-arg2.__nrank_, arg2.__nrank_, axis2))
        k2 = a2 + len(arg2.__shape_)

        # Confirm that the axis lengths are compatible
        if arg1.__numer_[a1] != arg2.__numer_[a2]:
            raise ValueError('axes have different lengths: %d, %d' %
                             (arg1.__numer_[a1], arg2.__numer_[a2]))

        # Re-shape the value arrays (shape, numer1, numer2, denom1, denom2)
        shape1 = (arg1.__shape_ + arg1.__numer_ + (arg2.__nrank_ - 1) * (1,) +
                  arg1.__denom_ + arg2.__drank_ * (1,))
        array1 = arg1.__values_.reshape(shape1)

        shape2 = (arg2.__shape_ + (arg1.__nrank_ - 1) * (1,) + arg2.__numer_ +
                  arg1.__drank_ * (1,) + arg2.__denom_)
        array2 = arg2.__values_.reshape(shape2)
        k2 += arg1.__nrank_ - 1

        # Roll both array axes to the right
        array1 = np.rollaxis(array1, k1, len(array1.shape))
        array2 = np.rollaxis(array2, k2, len(array2.shape))

        # Make arrays contiguous so sum will run faster
        array1 = np.ascontiguousarray(array1)
        array2 = np.ascontiguousarray(array2)
        
        # Construct the dot product
        new_values = np.sum(array1 * array2, axis=-1)

        # Construct the object and cast
        new_nrank = arg1.__nrank_ + arg2.__nrank_ - 2
        new_drank = arg1.__drank_ + arg2.__drank_

        obj = Qube(new_values, arg1.__mask_ | arg2.__mask_,
                   Units.mul_units(arg1.__units_, arg2.__units_), derivs={},
                   nrank=new_nrank, drank=new_drank, example=arg1)
        obj = obj.cast(classes)

        # Insert derivatives if necessary
        if recursive and (arg1.__derivs_ or arg2.__derivs_):
            new_derivs = {}

            if arg1.__derivs_:
                arg2_wod = arg2.without_derivs()
                for (key, arg1_deriv) in arg1.__derivs_.iteritems():
                    new_derivs[key] = Qube.dot(arg1_deriv, arg2_wod, a1, a2,
                                               classes, recursive=False)

            if arg2.__derivs_:
                arg1_wod = arg1.without_derivs()
                for (key, arg2_deriv) in arg2.__derivs_.iteritems():
                    term = Qube.dot(arg1_wod, arg2_deriv, a1, a2,
                                    classes, recursive=False)
                    if key in new_derivs:
                        new_derivs[key] += term
                    else:
                        new_derivs[key] = term

            obj.insert_derivs(new_derivs)

        return obj

    @staticmethod
    def norm(arg, axis=-1, classes=(), recursive=True):
        """Return the norm of an object along one axis.

        The axes must be in the numerator. The denominator must have zero rank.

        Input:
            arg         the object for which to calculate the norm.
            axis        the numerator axis for the norm.
            classes     a single class or list or tuple of classes. The class
                        of the object returned will be the first suitable class
                        in the list. Otherwise, a generic Qube object will be
                        returned.
            recursive   True to construct the derivatives of the norm.
        """

        assert arg.__drank_ == 0

        # Position axis from left
        if axis >= 0:
            a1 = axis
        else:
            a1 = axis + arg.__nrank_
        if a1 < 0 or a1 >= arg.__nrank_:
            raise ValueError('axis is out of range (%d,%d): %d' %
                             (-arg.__nrank_, arg.__nrank_, axis))
        k1 = a1 + len(arg.__shape_)

        # Evaluate the norm
        new_values = np.sqrt(np.sum(arg.__values_**2, axis=k1))

        # Construct the object and cast
        obj = Qube(new_values, derivs={}, nrank=arg.__nrank_-1, example=arg)
        obj = obj.cast(classes)

        # Insert derivatives if necessary
        if recursive and arg.__derivs_:
            factor = arg.without_derivs() / obj
            for (key, arg_deriv) in arg.__derivs_.iteritems():
                obj.insert_deriv(key, Qube.dot(factor, arg_deriv, a1, a1,
                                               classes, recursive=False))

        return obj

    @staticmethod
    def norm_sq(arg, axis=-1, classes=(), recursive=True):
        """Return square of the norm of an object along one axis.

        The axes must be in the numerator. The denominator must have zero rank.

        Input:
            arg         the object for which to calculate the norm-squared.
            axis        the item axis for the norm.
            classes     a single class or list or tuple of classes. The class
                        of the object returned will be the first suitable class
                        in the list. Otherwise, a generic Qube object will be
                        returned.
            recursive   True to construct the derivatives of the norm-squared.
        """

        assert arg.__drank_ == 0

        # Position axis from left
        if axis >= 0:
            a1 = axis
        else:
            a1 = axis + arg.__nrank_
        if a1 < 0 or a1 >= arg.__nrank_:
            raise ValueError('axis is out of range (%d,%d): %d' %
                             (-arg.__nrank_, arg.__nrank_, axis))
        k1 = a1 + len(arg.__shape_)

        # Evaluate the norm
        new_values = np.sum(arg.__values_**2, axis=k1)

        # Construct the object and cast
        obj = Qube(new_values, arg.__mask_,
                   Units.mul_units(arg.__units_, arg.__units_), derivs={},
                   nrank=arg.__nrank_-1, example=arg)
        obj = obj.cast(classes)

        # Insert derivatives if necessary
        if recursive and arg.__derivs_:
            factor = 2.* arg.without_derivs()
            for (key, arg_deriv) in arg.__derivs_.iteritems():
                obj.insert_deriv(key, Qube.dot(factor, arg_deriv, a1, a1,
                                               classes, recursive=False))

        return obj

    @staticmethod
    def cross(arg1, arg2, axis1=-1, axis2=0, classes=(), recursive=True):
        """Return the cross product of two objects.

        Axis lengths must be either two or three, and must be equal. At least
        one of the objects must be lacking a denominator.

        Input:
            arg1        the first operand.
            arg2        the second operand.
            axis1       the item axis of the first object.
            axis2       the item axis of the second object.
            classes     a single class or list or tuple of classes. The class
                        of the object returned will be the first suitable class
                        in the list. Otherwise, a generic Qube object will be
                        returned.
            recursive   True to construct the derivatives of the cross product.
        """

        # At most one object can have a denominator.
        assert not (arg1.__drank_ and arg2.__drank_)

        # Position axis1 from left
        if axis1 >= 0:
            a1 = axis1
        else:
            a1 = axis1 + arg1.__nrank_
        if a1 < 0 or a1 >= arg1.__nrank_:
            raise ValueError('first axis is out of range (%d,%d): %d' %
                             (-arg1.__nrank_, arg1.__nrank_, axis1))
        k1 = a1 + len(arg1.__shape_)

        # Position axis2 from item left
        if axis2 >= 0:
            a2 = axis2
        else:
            a2 = axis2 + arg2.__nrank_
        if a2 < 0 or a2 >= arg2.__nrank_:
            raise ValueError('second axis out of range (%d,%d): %d' %
                             (-arg2.__nrank_, arg2.__nrank_, axis2))
        k2 = a2 + len(arg2.__shape_)

        # Confirm that the axis lengths are compatible
        if ((arg1.__numer_[a1] != arg2.__numer_[a2]) or
            (arg1.__numer_[a1] not in (2,3))):
            raise ValueError('invalid axis length for cross product: %d, %d' %
                             (arg1.__numer_[a1], arg2.__numer_[a2]))

        # Re-shape the value arrays (shape, numer1, numer2, denom1, denom2)
        shape1 = (arg1.__shape_ + arg1.__numer_ + (arg2.__nrank_ - 1) * (1,) +
                  arg1.__denom_ + arg2.__drank_ * (1,))
        array1 = arg1.__values_.reshape(shape1)

        shape2 = (arg2.__shape_ + (arg1.__nrank_ - 1) * (1,) + arg2.__numer_ +
                  arg1.__drank_ * (1,) + arg2.__denom_)
        array2 = arg2.__values_.reshape(shape2)
        k2 += arg1.__nrank_ - 1

        # Roll both array axes to the right
        array1 = np.rollaxis(array1, k1, len(array1.shape))
        array2 = np.rollaxis(array2, k2, len(array2.shape))

        new_drank = arg1.__drank_ + arg2.__drank_

        # Construct the cross product values
        if arg1.__numer_[a1] == 3:
            new_values = Qube.cross_3x3(array1, array2)

            # Roll the new axis back to its position in arg1
            new_nrank = arg1.__nrank_ + arg2.__nrank_ - 1
            new_k1 = len(new_values.shape) - new_drank - new_nrank + a1
            new_values = np.rollaxis(new_values, -1, new_k1)

        else:
            new_values = Qube.cross_2x2(array1, array2)
            new_nrank = arg1.__nrank_ + arg2.__nrank_ - 2

        # Construct the object and cast
        obj = Qube(new_values, arg1.__mask_ | arg2.__mask_,
                   Units.mul_units(arg1.__units_, arg2.__units_), derivs={},
                   nrank=new_nrank, drank=new_drank, example=arg1)
        obj = obj.cast(classes)

        # Insert derivatives if necessary
        if recursive and (arg1.__derivs_ or arg2.__derivs_):
            new_derivs = {}

            if arg1.__derivs_:
              arg2_wod = arg2.without_derivs()
              for (key, arg1_deriv) in arg1.__derivs_.iteritems():
                new_derivs[key] = Qube.cross(arg1_deriv, arg2_wod, a1, a2,
                                             classes, recursive=False)

            if arg2.__derivs_:
              arg1_wod = arg1.without_derivs()
              for (key, arg2_deriv) in arg2.__derivs_.iteritems():
                term = Qube.cross(arg1_wod, arg2_deriv, a1, a2, classes, False)
                if key in new_derivs:
                    new_derivs[key] += term
                else:
                    new_derivs[key] = term

            obj.insert_derivs(new_derivs)

        return obj

    @staticmethod
    def cross_3x3(a,b):
        """Stand-alone method to return the cross product of two 3-vectors,
        represented as NumPy arrays."""

        (a,b) = np.broadcast_arrays(a,b)
        assert a.shape[-1] == b.shape[-1] == 3

        new_values = np.empty(a.shape)
        new_values[...,0] = a[...,1] * b[...,2] - a[...,2] * b[...,1]
        new_values[...,1] = a[...,2] * b[...,0] - a[...,0] * b[...,2]
        new_values[...,2] = a[...,0] * b[...,1] - a[...,1] * b[...,0]

        return new_values

    @staticmethod
    def cross_2x2(a,b):
        """Stand-alone method to return the cross product of two 2-vectors,
        represented as NumPy arrays."""

        (a,b) = np.broadcast_arrays(a,b)
        assert a.shape[-1] == b.shape[-1] == 2

        return a[...,0] * b[...,1] - a[...,1] * b[...,0]

    @staticmethod
    def outer(arg1, arg2, classes=(), recursive=True):
        """Return the outer product of two objects.

        The item shape of the returned object is obtained by concatenating the
        two numerators and then the two denominators, and each element is the
        product of the corresponding elements of the two objects.

        Input:
            arg1        the first operand.
            arg2        the second operand.
            classes     a single class or list or tuple of classes. The class
                        of the object returned will be the first suitable class
                        in the list. Otherwise, a generic Qube object will be
                        returned.
            recursive   True to construct the derivatives of the outer product.
        """

        # At most one object can have a denominator. This is sufficient to track
        # first derivatives
        assert not (arg1.__drank_ and arg2.__drank_)

        # Re-shape the value arrays (shape, numer1, numer2, denom1, denom2)
        shape1 = (arg1.__shape_ + arg1.__numer_ + arg2.__nrank_ * (1,) +
                  arg1.__denom_ + arg2.__drank_ * (1,))
        array1 = arg1.__values_.reshape(shape1)

        shape2 = (arg2.__shape_ + arg1.__nrank_ * (1,) + arg2.__numer_ +
                  arg1.__drank_ * (1,) + arg2.__denom_)
        array2 = arg2.__values_.reshape(shape2)

        # Construct the outer product
        new_values = array1 * array2

        # Construct the object and cast
        new_nrank = arg1.__nrank_ + arg2.__nrank_
        new_drank = arg1.__drank_ + arg2.__drank_

        obj = Qube(new_values, arg1.__mask_ | arg2.__mask_,
                   Units.mul_units(arg1.__units_, arg2.__units_), derivs={},
                   nrank=new_nrank, drank=new_drank, example=arg1)
        obj = obj.cast(classes)

        # Insert derivatives if necessary
        if recursive and (arg1.__derivs_ or arg2.__derivs_):
            new_derivs = {}

            if arg1.__derivs_:
              arg_wod = arg2.without_derivs()
              for (key, self_deriv) in arg1.__derivs_.iteritems():
                new_derivs[key] = Qube.outer(self_deriv, arg_wod, classes,
                                             recursive=False)

            if arg2.__derivs_:
              self_wod = arg1.without_derivs()
              for (key, arg_deriv) in arg2.__derivs_.iteritems():
                term = Qube.outer(self_wod, arg_deriv, classes, recursive=False)
                if key in new_derivs:
                    new_derivs[key] += term
                else:
                    new_derivs[key] = term

            obj.insert_derivs(new_derivs)

        return obj

    @staticmethod
    def as_diagonal(self, axis, classes=(), recursive=True):
        """Return a copy with one axis converted to a diagonal across two.

        Input:
            axis        the item axis to convert to two.
            classes     a single class or list or tuple of classes. The class
                        of the object returned will be the first suitable class
                        in the list. Otherwise, a generic Qube object will be
                        returned.
            recursive   True to include matching slices of the derivatives in
                        the returned object; otherwise, the returned object will
                        not contain derivatives.
        """

        # Position axis from left
        if axis >= 0:
            a1 = axis
        else:
            a1 = axis + self.__nrank_
        if a1 < 0 or a1 >= self.__nrank_:
            raise ValueError('axis is out of range (%d,%d): %d',
                             (-self.__nrank_, self.__nrank_, axis))
        k1 = a1 + len(self.__shape_)

        # Roll this axis to the end
        rolled = np.rollaxis(self.__values_, k1, len(self.__values_.shape))

        # Create the diagonal array
        new_values = np.zeros(rolled.shape + rolled.shape[-1:],
                              dtype=rolled.dtype)

        for i in range(rolled.shape[-1]):
            new_values[...,i,i] = rolled[...,i]

        # Roll the new axes back
        new_values = np.rollaxis(new_values, -1, k1)
        new_values = np.rollaxis(new_values, -1, k1)

        # Construct and cast
        new_numer = new_values.shape[len(self.__shape_):][:self.__nrank_+1]
        obj = Qube(new_values, derivs={}, nrank=self.__nrank_ + 1, example=self)
        obj = obj.cast(classes)

        # Diagonalize the derivatives if necessary
        if recursive:
          for (key, deriv) in self.__derivs_.iteritems():
            obj.insert_deriv(key, Qube.as_diagonal(deriv, axis, classes, False))

        return obj

    @classmethod
    def from_scalars(cls, *scalars, **keywords):
        """A new instance constructed from Scalars or arrays given as arguments.

        Defined as a class method so it can also be used to generate instances
        of any 1-D subclass.

        Input:
            scalars     one or more Scalars or objects that can be casted to
                        Scalars.

            recursive   True (the default) to construct the derivatives as the
                        union of the derivatives of all the components'
                        derivatives. False to return an object without
                        derivatives. Must be specified via a keyword.

            readonly    True to return a read-only object; False (the default)
                        to return something potentially writable.
        """

        # Search the keywords for "recursive" and "readonly"
        recursive = True
        if 'recursive' in keywords:
            recursive = keywords['recursive']
            del keywords['recursive']

        readonly = False
        if 'readonly' in keywords:
            readonly = keywords['readonly']
            del keywords['readonly']

        # No other keyword is allowed
        if keywords:
          raise ValueError('broadcast() got an unexpected keyword argument ' +
                           '"%s"' % keywords.keys()[0])

        # Convert to scalars and broadcast to the same shape
        args = []
        for arg in scalars:
            scalar = Qube.SCALAR_CLASS.as_scalar(arg)
            args.append(scalar)

        scalars = Qube.broadcast(*args, recursive=recursive)

        # Tabulate the properties and construct the value array
        new_mask = False
        new_units = None
        new_denom = None

        arrays = []
        deriv_dicts = []
        nderivs = 0
        for scalar in scalars:
            arrays.append(scalar.__values_)

            new_mask = new_mask | scalar.__mask_
            new_units = new_units or scalar.__units_
            Units.require_match(new_units, scalar.__units_)

            if new_denom is None:
                new_denom = scalar.__denom_
            elif new_denom != scalar.__denom_:
                raise ValueError('mixed denominator shapes')

            deriv_dicts.append(scalar.__derivs_)
            if len(scalar.__derivs_): nderivs += 1

        # Construct the values array
        new_drank = len(new_denom)
        new_values = np.array(arrays)
        new_values = np.rollaxis(new_values, 0, len(new_values.shape) -
                                                new_drank)

        # Construct the object
        obj = Qube.__new__(cls)
        obj.__init__(new_values, new_mask, new_units, drank=new_drank)

        # Insert derivatives if necessary
        if recursive and nderivs:
          count = len(scalars)
          new_derivs = {}
          for i in range(count):
            for (key, deriv) in  deriv_dicts[i].iteritems():

                # If it's already there, make sure we have a match
                if key in new_derivs:
                    if new_derivs[key].__denom_ != deriv.__denom_:
                        raise ValueError("mixed denominator shapes in 'd_d%s'" %
                                         key)
                    new_deriv = new_derivs[key]

                # Otherwise, start with an empty object
                else:
                    new_values = np.zeros(deriv.__shape_ + (count,) +
                                          deriv.__denom_)
                    empty_deriv = Qube.__new__(cls)
                    empty_deriv.__init__(zeros, nrank=deriv.__nrank_,
                                                drank=deriv.__drank_)
                    new_deriv = empty_deriv

                # Fill in the i_th entry
                index = (Ellipsis,) + (i,) + deriv.__drank_ * (slice(None),)
                new_values = new_deriv.__values_
                new_values[index] = deriv.__values_

                new_mask = new_deriv.__mask_
                if np.any(deriv.__mask_):
                    if  new_mask is False:
                        new_mask = np.zeros(deriv.shape, dtype='bool')
                    new_mask |= deriv.__mask_

                new_deriv.__values = new_values
                new_deriv.__mask = new_mask

                new_derivs[key] = new_deriv

          obj.insert_derivs(new_derivs)

        return obj

    def rms(self):
        """Return the root-mean-square values of all items as a Scalar.

        Useful for looking at the overall magnitude of the differences between
        two objects.

        Input:
            arg         the object for which to calculate the RMS.
        """

        # Evaluate the norm
        sum_sq = self.__values_**2
        for r in range(self.__rank_):
            sum_sq = np.sum(sum_sq, axis=-1)

        return Qube.SCALAR_CLASS(np.sqrt(sum_sq/self.isize), self.__mask_)

    ############################################################################
    # General shaping operations
    ############################################################################

    def reshape(self, shape, recursive=True):
        """Return a shallow copy of the object with a new leading shape.

        Input:
            shape       a tuple defining the new leading shape.
            recursive   True to apply the same shape to the derivatives.
                        Otherwise, derivatives are deleted from the returned
                        object.
        """

        shape = tuple(shape)
        if shape == self.__shape_: return self

        if np.shape(self.__values_) == ():
            new_values = np.array([self.__values_]).reshape(shape)
        else:
            new_values = self.__values_.reshape(shape + self.item)

        if np.shape(self.__mask_) == ():
            new_mask = self.__mask_
        else:
            new_mask = self.__mask_.reshape(shape)

        obj = Qube.__new__(type(self))
        obj.__init__(new_values, new_mask, derivs={}, example=self)
        obj.__readonly_ = self.__readonly_

        if recursive:
            for (key, deriv) in self.__derivs_.iteritems():
                obj.insert_deriv(key, deriv.reshape(shape, False))

        return obj

    def flatten(self, recursive=True):
        """Return a shallow copy of the object flattened to one dimension."""

        if len(self.__shape_) < 2: return self

        count = np.product(self.__shape_)
        return self.reshape((count,), recursive)

    def swap_axes(self, axis1, axis2, recursive=True):
        """Return a shallow copy of the object with two leading axes swapped.

        Input:
            axis1       the first index of the swap. Negative indices
                        Negative indices are relative to the last index before
                        the numerator items begin.
            axis2       the second index of the swap.
            recursive   True to perform the same swap on the derivatives.
                        Otherwise, derivatives are deleted from the returned
                        object.
        """

        # Validate first axis
        len_shape = len(self.__shape_)
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

        if a1 == a2: return self

        new_values = self.__values_.swapaxes(a1, a2)

        if np.shape(self.__mask_) == ():
            new_mask = self.__mask_
        else:
            new_mask = self.__mask_.swapaxes(a1, a2)

        obj = Qube.__new__(type(self))
        obj.__init__(new_values, new_mask, derivs={}, example=self)
        obj.__readonly_ = self.__readonly_

        if recursive:
            for (key, deriv) in self.__derivs_.iteritems():
                obj.insert_deriv(key, deriv.swap_axes(a1, a2, False))

        return obj

    def roll_axis(self, axis, start=0, recursive=True, rank=None):
        """Return a shallow copy of the object with the specified axis rolled to
        a new position.

        Input:
            axis        the axis to roll.
            start       the axis will be rolled to fall in front of this axis;
                        default is zero.
            recursive   True to perform the same axis roll on the derivatives.
                        Otherwise, derivatives are deleted from the returned
                        object.
            rank        rank to assume for the object, which could be larger
                        than len(self.shape) because of broadcasting
        """

        # Validate the rank
        len_shape = len(self.__shape_)
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
        if self.__shape_ == ():
            return self

        # Add missing axes if necessary
        if len_shape < rank:
            self = self.reshape((rank - len_shape) * (1,) + self.__shape_,
                                recursive=recursive)

        # Roll the values and mask of the object
        new_values = np.rollaxis(self.__values_, a1, a2)

        if np.shape(self.__mask_):
            new_mask = np.rollaxis(self.__mask_, a1, a2)
        else:
            new_mask = self.__mask_

        obj = Qube.__new__(type(self))
        obj.__init__(new_values, new_mask, derivs={}, example=self)
        obj.__readonly_ = self.__readonly_

        if recursive:
            for (key, deriv) in self.__derivs_.iteritems():
                obj.insert_deriv(key, deriv.roll_axis(a1, a2, False, rank))

        return obj

    def broadcast_into_shape(self, shape, recursive=True, sample_array=None):
        """Returns an object broadcasted to the specified shape.

        It returns self if the shape already matches. Otherwise, the returned
        object shares data with the original and both objects will be read-only.

        Input:
            shape           the shape into which the object is to be broadcast.
            recursive       True to broadcast the derivatives as well.
                            Otherwise, they are removed.
            sample_array    if specified, a NumPy ndarray with the required
                            shape. Otherwise None. Used internally.
        """

        shape = tuple(shape)

        # If no broadcast is needed, return the object
        if shape == self.__shape_:
            if recursive or not self.__derivs_: return self

        # Prepare or validate the sample array
        if sample_array is None:
            sample_array = np.empty(shape, dtype='byte')
        else:
            assert sample_array.shape == shape

        # Save the derivatives for later
        derivs = self.__derivs_

        # A broadcasted object must be read-only if __values_ is an array
        if np.shape(self.__values_):
            self.as_readonly(recursive=False)

        # Broadcast the values array
        values_shape = shape + self.__rank_ * (1,)
        new_values = np.broadcast_arrays(self.__values_,
                                         sample_array.reshape(values_shape))[0]
        Qube._array_to_readonly(new_values)

        # Broadcast the mask if necessary
        if np.shape(self.__mask_) == ():
            new_mask = self.__mask_
        else:
            new_mask = np.broadcast_arrays(self.__mask_, sample_array)[0]

        # Construct the new object
        obj = Qube.__new__(type(self))
        obj.__init__(new_values, new_mask, derivs={}, example=self)

        # Process the derivatives if necessary
        if recursive:
            for (key, deriv) in derivs.iteritems():
                obj.insert_deriv(key, deriv.broadcast_into_shape(shape, False,
                                                                 sample_array))

        return obj

    @staticmethod
    def broadcasted_shape(*objects, **keywords):
        """Return the shape defined by a broadcast across the objects.

        Input:          zero or more array objects. Values of None and Empty()
                        are assigned shape (). A list or tuple is treated as the
                        definition of an additional shape.

            item        a list or tuple to be appended to the shape. Default is
                        (). This makes it possible to use the returned shape in
                        the declaration of a NumPy array containing items that
                        are not scalars. Note that this is handled as a keyword
                        parameter in order to distinguish it from the objects.

        Return:         the broadcast shape, comprising the maximum value of
                        each corresponding axis, plus the item shape appended.
        """

        # Search the keywords for "item"
        item = ()
        if 'item' in keywords:
            item = keywords['item']
            del keywords['item']

        # No other keyword is allowed
        if keywords:
          raise TypeError(('broadcasted_shape() got an unexpected keyword ' +
                           'argument "%s"') % keywords.keys()[0])

        # Initialize the shape
        new_shape = []
        len_broadcast = 0

        # Loop through the arrays...
        for obj in objects:
            if obj is None: continue
            if Qube.is_empty(obj): continue
            if Qube.is_real_number(obj): continue

            # Get the next shape
            if type(obj) == tuple or type(obj) == list:
                shape = list(obj)
            else:
                shape = list(obj.shape)

            # Expand the shapes to the same rank
            len_shape = len(shape)

            if len_shape > len_broadcast:
                new_shape = (len_shape - len_broadcast) * [1] + new_shape
                len_broadcast = len_shape

            if len_broadcast > len_shape:
                shape = (len_broadcast - len_shape) * [1] + shape
                len_shape = len_broadcast

            # Update the broadcast shape and check for compatibility
            for i in range(len_shape):
                if new_shape[i] == 1:
                    new_shape[i] = shape[i]
                elif shape[i] == 1:
                    pass
                elif shape[i] != new_shape[i]:
                    raise ValueError('incompatible dimension on axis ' + str(i))

        return tuple(new_shape) + tuple(item)

    @staticmethod
    def broadcast(*objects, **keywords):
        """Broadcast objects to their common shape.

        Python scalars are returned unchanged because they already broadcast
        with anything.

        Returned objects must be treated as read-only because of the mechanism
        NumPy uses to broadcast arrays. The returned objects are marked
        read-only but their internal arrays are not protected.

        Input:          zero or more objects to broadcast.

            recursive   True to broadcast the derivatives to the same shape;
                        False to strip the derivatives from the returne objects.
                        Note that this is handled as a keyword argument to
                        distinguish it from the objects.

        Return:         A tuple of copies of the objects, broadcasted to a
                        common shape. Empty objects are not modified. The
                        returned objects must be treated as read-only.
        """

        # Search the keywords for "recursive"
        recursive = True
        if 'recursive' in keywords:
            recursive = keywords['recursive']
            del keywords['recursive']

        # No other keyword is allowed
        if keywords:
          raise TypeError(('broadcast() got an unexpected keyword argument ' +
                           '"%s"') % keywords.keys()[0])

        # Perform the broadcasts...
        shape = Qube.broadcasted_shape(*objects)
        sample_array = np.empty(shape, dtype='byte')

        results = []
        for obj in objects:
            if isinstance(obj, np.ndarray):
                new_obj = np.broadcast_arrays(obj, sample_array)[0]
            elif isinstance(obj, Qube):
                new_obj = obj.broadcast_into_shape(shape, recursive,
                                                          sample_array)
            else:
                new_obj = obj
            results.append(new_obj)

        return tuple(results)

    @staticmethod
    def stack(*args, **keywords):
        """Stack objects of the same class into one with a new leading axis.

        Inputs:
            args        any number of Scalars or arguments that can be casted
                        to Scalars. They need not have the same shape, but it
                        must be possible to cast them to the same shape. A value
                        of None is converted to a zero-valued Scalar that
                        matches the denominator shape of the other arguments.

            recursive   True to include all the derivatives. The returned object
                        will have derivatives representing the union of all the
                        derivatives found amongst the scalars. Default is True.

        Note that the 'recursive' input is handled as a keyword argument in
        order to distinguish it from the Qube inputs.
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

        # Get the properties of the first argument as a PolyMath subclass
        # However, do not give preference to Booleans
        denom = ()
        subclass_arg = None
        for (i,arg) in enumerate(args):
            if (isinstance(arg, Qube) and not
                isinstance(arg, Qube.BOOLEAN_CLASS)):
                denom = arg.denom
                subclass_arg = arg
                subclass_indx = i
                break

        if subclass_arg is None:
            for (i,arg) in enumerate(args):
                if isinstance(arg, Qube):
                    denom = arg.denom
                    subclass_arg = arg
                    subclass_indx = i
                    break

        if subclass_arg is None:
            raise ValueError('unidentified subclass for stack()')

        drank = len(denom)

        # Convert to subclass, identify units, select dtype
        units = None
        floats_found = False
        ints_found = False
        bools_found = False
        for (i,arg) in enumerate(args):
            if arg is None: continue        # Used as placehold for derivs

            arg = subclass_arg.as_this_type_unless_boolean(arg,
                                            recursive=recursive, drank=drank)
            args[i] = arg

            # Remember any units encountered
            if arg.units is not None:
                if units is None:
                    units = arg.units
                else:
                    arg.confirm_units(units)

            # Remember any floats, ints, bools encountered
            if arg.is_float():
                floats_found = True
            elif arg.is_int():
                ints_found = True
            elif arg.is_bool():
                bools_found = True

        # Determine the dtype
        if floats_found:
            dtype = 'float'
            for (i,arg) in enumerate(args):
                if arg is not None:
                    args[i] = arg.as_float()

        elif ints_found:
            dtype = 'int'
            for (i,arg) in enumerate(args):
                if arg is not None:
                    args[i] = arg.as_int()
        else:
            dtype = 'bool'

        # Broadcast all inputs into a common shape
        args = Qube.broadcast(*args, recursive=True)

        # Determine what type of mask is needed:
        mask_true_found = False
        mask_false_found = False
        mask_array_found = False
        for arg in args:
            if arg is None:
                continue
            elif arg.mask is True:
                mask_true_found = True
            elif arg.mask is False:
                mask_false_found = True
            else:
                mask_array_found = True

        # Construct the mask
        if  mask_array_found or (mask_false_found and mask_true_found):
            mask = np.zeros((len(args),) + args[subclass_indx].shape,
                            dtype='bool')
            for i in range(len(args)):
                if args[i] is None:
                    mask[i] = False
                else:
                    mask[i] = args[i].mask
        else:
            mask = mask_true_found

        # Construct the array
        values = np.empty((len(args),) + np.shape(args[subclass_indx].values),
                          dtype=dtype)
        for i in range(len(args)):
            if args[i] is None:
                values[i] = 0
            else:
                values[i] = args[i].values

        # Construct the result
        result = Qube.__new__(type(args[subclass_indx]))
        result.__init__(values, mask, units, drank=drank)

        # Fill in derivatives if necessary
        if recursive:
            keys = []
            for arg in args:
                if arg is None: continue
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
