################################################################################
# polymath/qube.py: Base class for all PolyMath subclasses.
#
# Mark Showalter, PDS Ring-Moon Systems Node, SETI Institute
################################################################################

from __future__ import division

from IPython import embed   ## TODO: remove
import sys
import numbers
import warnings
import numpy as np

from .units import Units

#*******************************************************************************
# Qube class
#*******************************************************************************
class Qube(object):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    The base class for all PolyMath subclasses.

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
    sqrt(-1) are masked out so that run-time errors can be avoided. See more
    about masking below.

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

    More about masked values:
    - Under normal circumstances, a masked value should be understood to mean,
      "this value does not exist." For example, a calculation of observed
      intercept points on a moon is masked if a line of sight missed the moon,
      because that line of sight does not exist.
    - Two masked values of the same class are considered equal. This is
      different from the behavior of NaN.
    - Unary and binary operations involving masked values always return masked
      values.
    - Because masked values are treated as if they do not exist, for example:
        - max() returns the maximum among the unmasked values.
        - all() returns True if all the unmasked values are True.
    - A few methods support an alternative interpretation of masked values as
      indeterminate rather than nonexistent. These follow the rules of "three-
      valued logic:
        - tvl_and() returns False if one value is False but the other is masked,
                    because the result would be False regardless of the second
                    value.
        - tvl_or()  return True if one value is True but the other is masked,
                    because the result would be True regardless of the second
                    value.
        - tvl_all() returns True only if all values are True; if any value is
                    masked, it returns False.
        - tvl_any() returns True if any value is True, irrespective of any
                    masked values.
        - tvl_eq()  returns False if both values are masked.
        - tvl_ne()  returns Masked if both values are masked.

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

    Notes about indexing

    Using an index on a Qube object is very similar to using one on a NumPy
    array, but there are a few important differences. For purposes of retrieving
    selected values from an object:

    - True and False can be applied to shapeless objects. True leaves the object
      unchanged; False masks the object.

    - For dimensional objects, an index of True selects the entire associated
      axis, equivalent to a colon or slice(None). An index of False reduces the
      associated axis to length one and masks the object entirely.

    - A Boolean object can be used as an index. If this index is unmasked, it is
      equivalent to indexing with a boolean array. If it is masked, the object
      values at the masked locations in the Boolean are returned as masked.

    - A Scalar object composed of integers can be used as an index. If this
      index is unmasked, it is equivalent to indexing with an integer or integer
      array. If it is masked, the object values at the masked locations in the
      Scalar are returned as masked.

    - A Pair object composed of integers can be used as an index. Each (i,j)
      value is treated is the index of two consecutive axes, and the associated
      value is returned. Where the Pair is masked, a masked value is returned.

    - Similarly, a Vector with three or more integer elements is treated as the
      index of three or more axes.

    - NumPy has an obscure indexing rule, which Polymath overrides to offer more
      sensible behavior. It involves how to handle multiple, non-consecutive
      arrays when used to index another array. Note, first, that when multiple
      arrays appear in an index, their resulting array shapes are broadcasted
      together. For example, in this case:
        object[<int array with shape (4,)>, <int array with shape (3,1)>, ...]
      the leading axes of the indexed object will have shape (3,4).

      When all array indices are adjacent, the location of the new axes in the
      returned object will fall at the location of the first array index. For
      example, with this index:
        object[:,<int array with shape (4,)>,<int array with shape (3,1)>]
      the second and third axes of the object returned will have shape (3,4).

      However, when array indices are not consecutive, NumPy indexing rules
      place the shape of the array axes first in the returned object, no matter
      where they appeared in the index sequence. Qube indexing rules override
      this behavior to place the axes at the location of the first array index
      instead. Suppose we have an object with shape (10,10,10,10), indexed so:
        object[:,<int array with shape (4,)>,:,<int array with shape (3,1)>]
      Under NumPy indexing rules, the shape of the result will be (3,4,10,10).
      Under Qube indexing rules, the shape will be (10,3,4,10) instead.

    When using an index to set selected values of an array, a masked index value
    prevents the corresponding value on the right-hand side of the equal sign
    from changing the corresponding array element on the left-hand side.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    #--------------------------------------------------------------------------
    # This prevents binary operations of the form:
    #   <np.ndarray> <op> <Qube>
    # from executing the ndarray operation instead of the polymath operation
    #--------------------------------------------------------------------------
    __array_priority__ = 1

    #--------------------------------------------------------------------------
    # Set this global attribute to True to restore a behavior removed on
    # 3/13/18. It allowed certain functions to return a Python value (float,
    # int, or bool) if the result had shape (), was unmasked, and had no units.
    # Currently, these functions all return Scalar or Boolean types instead.
    #
    # As an alternative to this global solution, you can use method as_builtin()
    # to convert an object to a built-in type if this is possible.
    #--------------------------------------------------------------------------
    PREFER_BUILTIN_TYPES = False

    #--------------------------------------------------------------------------
    # Global attribute to be used for testing
    #--------------------------------------------------------------------------
    DISABLE_CACHE = False

    #--------------------------------------------------------------------------
    # Default class constants, to be overridden as needed by subclasses...
    #--------------------------------------------------------------------------
    NRANK = None        # the number of numerator axes; None to leave this
                        # unconstrained.
    NUMER = None        # shape of the numerator; None to leave unconstrained.

    FLOATS_OK = True    # True to allow floating-point numbers.
    INTS_OK = True      # True to allow integers.
    BOOLS_OK = True     # True to allow booleans.

    UNITS_OK = True     # True to allow units; False to disallow them.
    DERIVS_OK = True    # True to disallow derivatives; False to allow them.

    #===========================================================================
    # __new__
    #===========================================================================
    def __new__(subtype, *values, **keywords):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Create a new, un-initialized object.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return object.__new__(subtype)
    #===========================================================================



    #===========================================================================
    # __init__
    #===========================================================================
    def __init__(self, arg, mask=False, derivs={}, units=None,
                       nrank=None, drank=None, example=None, default=None):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Default constructor.

        arg         an object to define the numeric value(s) of the returned 
                    object. It could be another PolyMath object, a number, a
                    Numpy ndarray, a NumPy MaskedArray, or anything array-like.
                    If this object is read-only, then the returned object will
                    be entirely read-only. Otherwise, the object will be
                    read-writable. The values are generally given in standard
                    units of km, seconds and radians, regardless of the
                    specified units.

        mask        the mask for the object, as a single boolean, an array, or
                    anything array-like. Use None to copy the mask from the
                    example object. False (the default) leaves the object
                    un-masked.

        derivs      a dictionary of derivatives represented as PolyMath objects.
                    Use None to employ a copy of the derivs attribute of the
                    example object, or {} (the default) for no derivatives. All
                    derivatives are broadcasted to the shape of the object if
                    necessary.

        units       the units of the object. Use None to infer the units from
                    the example object; use False to suppress units.

        nrank       optionally, the number of numerator axes in the returned
                    object; None to derive the rank from the input data and/or
                    the subclass.

        drank       optionally, the number of denominator axes in the returned
                    object; None to derive it from the input data and/or the
                    subclass.

        example     optionally, another Qube object from which to copy any input
                    arguments that have not been explicitly specified.

        default     value to use where masked. Typically a constant that will
                    not "break" most arithmetic calculations. It must be a
                    Python built-in constant or ndarray of the same shape as the
                    items. Default is None, in which case the class constant
                    DEFAULT_VALUE is used, or else it is filled with ones.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if isinstance(arg, (list, tuple)):
            arg = np.array(arg)

        if isinstance(mask, (list, tuple)):
            mask = np.array(mask)

        if self.NRANK is not None:
            if nrank is not None and nrank != self.NRANK:
                raise ValueError('invalid numerator rank for class ' +
                                 '%s: %d' % (type(self).__name__, nrank))
            nrank = self.NRANK

        if derivs and not self.DERIVS_OK:
            raise ValueError('derivatives are disallowed for class ' +
                             '%s' % type(self).__name__)

        if units and not self.UNITS_OK:
            raise TypeError('units are disallowed for class ' +
                             '%s: %s' % (type(self).__name__, str(units)))

        #------------------------------------------------------
        # Interpret the arg if it is already a PolyMath object
        #------------------------------------------------------
        if isinstance(arg, Qube):

            if mask is None:
                mask = arg.__mask_
            else:
                mask = mask | arg.__mask_

            if self.UNITS_OK and units is None:
                units = arg.__units_

            if self.DERIVS_OK and derivs is None:
                derivs = arg.__derivs_.copy()   # shallow copy

            if nrank is None:
                nrank = arg.__nrank_
            elif nrank != arg.__nrank_:         # nranks _must_ be compatible
                raise ValueError('numerator ranks are incompatible: %d, %d' %
                                 (nrank, arg.__nrank_))

            if drank is None:                   # Override drank if undefined
                drank = arg.__drank_

            if default is None:
                default = arg.__default_

            arg = arg.__values_

        #------------------------------------------------------
        # Interpret the example
        #------------------------------------------------------
        if example is not None:

            if mask is None:
                mask = example.__mask_

            if self.DERIVS_OK and derivs is None:
                derivs = example.__derivs_.copy()   # shallow copy

            if self.UNITS_OK and units is None:
                units = example.__units_

            if nrank is None:
                nrank = example.__nrank_

            if drank is None:                   # Override drank if undefined
                drank = example.__drank_

            if default is None:
                default = example.__default_

        #------------------------------------------------------
        # Interpret the arg if it is a NumPy MaskedArray
        #------------------------------------------------------
        mask_from_array = None
        if isinstance(arg, np.ma.MaskedArray):
            if arg.mask is not np.ma.nomask:
                mask_from_array = arg.mask

            arg = arg.data

        #--------------------------------------------------------
        # Fill in the denominator rank if it is still undefined
        #--------------------------------------------------------
        if drank is None:
            drank = 0

        #------------------------------------------------------
        # Fill in the numerator rank if it is still undefined
        #------------------------------------------------------
        if nrank is None:
            nrank = 0

        rank = nrank + drank

        #------------------------------------------------------
        # Check the shape against nrank and drank
        #------------------------------------------------------
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

        #------------------------------------------------------
        # Fill in the values
        #------------------------------------------------------
        if isinstance(arg, np.ndarray):
            if arg.dtype.kind == 'b':
                if self.BOOLS_OK:
                    pass
                elif self.INTS_OK:
                    arg = arg.astype(np.int_)
                else:
                    arg = np.asfarray(arg)
            elif arg.dtype.kind in {'i','u'}:
                if self.INTS_OK:
                    pass
                elif self.FLOATS_OK:
                    arg = np.asfarray(arg)
                else:
                    arg = (arg != 0)
            elif arg.dtype.kind == 'f':
                if self.FLOATS_OK:
                    pass
                elif self.INTS_OK:
                    arg = (arg // 1).astype(np.int_)
                else:
                    arg = (arg != 0)
            else:
                raise ValueError("unsupported data type: %s" % str(arg.dtype))

        else:
            if isinstance(arg, (bool, np.bool_)):
                if self.BOOLS_OK:
                    pass
                elif self.INTS_OK:
                    arg = int(arg)
                else:
                    arg = float(arg)
            elif isinstance(arg, numbers.Integral):
                if self.INTS_OK:
                    pass
                elif self.FLOATS_OK:
                    arg = float(arg)
                else:
                    arg = (arg != 0)
            elif isinstance(arg, numbers.Real):
                if self.FLOATS_OK:
                    pass
                elif self.INTS_OK:
                    arg = int(arg // 1)
                else:
                    arg = (arg != 0)
            else:
                raise ValueError("unsupported data type: '%s'" % str(arg))

        self.__values_ = arg

        #------------------------------------------------------
        # Fill in the mask
        #------------------------------------------------------
        if isinstance(mask, Qube):
            mask = mask.as_mask_where_nonzero_or_masked()

        if mask is None:
            mask = False

        if np.shape(mask):
            mask = np.asarray(mask).astype(np.bool_)

        if mask_from_array is not None:
            for r in range(rank):
                mask_from_array = np.any(mask_from_array, axis=-1)

            mask = mask | mask_from_array

        #-------------------------------------------------------------
        # Broadcast the mask to the shape of the values if necessary
        #-------------------------------------------------------------
        if np.shape(mask) not in ((), shape):
            try:
                mask = np.broadcast_to(mask, shape)
            except:
                raise ValueError(("object shape and mask shape are " +
                                  "incompatible: %s, %s") %
                                  (str(shape), str(np.shape(mask))))

        self.__mask_ = mask

        #------------------------------------------------------
        # Fill in the default
        #------------------------------------------------------
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
                self.__default_ = self.__default_.astype(np.float_)
            else:
                self.__default_ = float(self.__default_)

        elif self.is_int():
            if isinstance(self.__default_, np.ndarray):
                self.__default_ = self.__default_.astype(np.int_)
            else:
                self.__default_ = int(self.__default_)

        # Fill in the remaining shape info
        self.__rank_  = rank
        self.__nrank_ = nrank
        self.__drank_ = drank

        self.__item_  = item
        self.__numer_ = numer
        self.__denom_ = denom

        self.__shape_ = shape

        #------------------------------------------------------
        # Fill in the units
        #------------------------------------------------------
        if Qube.is_one_false(units):
            self.__units_ = None
        else:
            self.__units_ = units

        #----------------------------------------------------------------------
        # The object is read-only if the values array is read-only
        #----------------------------------------------------------------------
        self.__readonly_ = Qube._array_is_readonly(self.__values_)

        if self.__readonly_:
            Qube._array_to_readonly(self.__mask_)

        #----------------------------------------------------------------------
        # Used for anything we want to cache in association with an object
        # This cache will be cleared whenever the object is modified in any way
        #----------------------------------------------------------------------
        self.__cache_ = {}

        # Install the derivs (converting to read-only if necessary)
        self.__derivs_ = {}
        if derivs:
            self.insert_derivs(derivs)

        self.__is_deriv_ = False    # gets changed by insert_derivs()

        #------------------------------------------------------
        # Used only for if clauses
        #------------------------------------------------------
        self.__truth_if_any_ = False
        self.__truth_if_all_ = False

        return
    #===========================================================================



    #===========================================================================
    # clone
    #===========================================================================
    def clone(self, recursive=True, preserve=[]):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Fast construction of a shallow copy.

        Inputs:
            recursive   True to copy the derivatives from this object; False to
                        ignore them.
            preserve    an optional list of derivative names to include even if
                        recursive is False.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        obj = object.__new__(type(self))

        #------------------------------------------------------
        # Transfer attributes
        #------------------------------------------------------
        for (attr, value) in self.__dict__.items():
            if isinstance(value, dict):
                if not recursive and attr.endswith('__derivs_'):
                    obj.__dict__[attr] = {}
                elif attr.endswith('__cache_'):
                    obj.__dict__[attr] = {}
                else:
                    obj.__dict__[attr] = value.copy()
            elif isinstance(value, Qube):
                if not recursive and attr.startswith('d_d'):
                    pass
                else:
                    obj.__dict__[attr] = value.clone()
            else:
                obj.__dict__[attr] = value

        #------------------------------------------------------
        # Restore any derivatives to be preserved
        #------------------------------------------------------
        if isinstance(preserve, str):
            preserve = [preserve]

        for key in preserve:
            obj.insert_deriv(key, self.__derivs_[key].clone(recursive=False))

        return obj
    #===========================================================================



    ############################################################################
    # For Packrat serialization
    ############################################################################

    #===========================================================================
    # PACKRAT__args__
    #===========================================================================
    def PACKRAT__args__(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        The list of attributes to write into the Packrat file.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        args = ['_Qube__shape_', '_Qube__item_',
                '_Qube__nrank_', '_Qube__drank_',
                '_Qube__readonly_', '_Qube__default_']

        #------------------------------------------------------
        # For a fully masked object, no need to save values
        #------------------------------------------------------
        flag = Qube.as_one_bool(self.__mask_)
        if flag is True:
            args.append('_Qube__mask_')

        #--------------------------------------------------------------
        # For an unmasked object, save the array values as they are
        #--------------------------------------------------------------
        elif flag is False:
            args.append('_Qube__mask_')
            args.append('_Qube__values_')

        #----------------------------------------------------------------------
        # For a partially masked object, take advantage of the corners and also
        # only save the unmasked values
        #----------------------------------------------------------------------
        else:
            self.__corners_ = self.corners
            args.append('_Qube__corners_')

            self.__sliced_mask_ = self.__mask_[self.slicer]
            args.append('_Qube__sliced_mask_')

            self.__unmasked_values_ = self.__values_[self.antimask]
            args.append('_Qube__unmasked_values_')

        #------------------------------------------------------
        # Include derivatives and units as needed
        #------------------------------------------------------
        if self.__derivs_:
            args.append('_Qube__derivs_{"single":True}')

        if self.__units_:
            args.append('_Qube__units_')

        return args
    #===========================================================================



    #===========================================================================
    # PACKRAT__init__
    #===========================================================================
    @staticmethod
    def PACKRAT__init__(cls, **args):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Construct an object from the subobjects extracted from the XML.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        shape = args['shape']
        item  = args['item']
        nrank = args['nrank']
        drank = args['drank']
        readonly = args['readonly']
        default  = args['default']
        derivs   = args.get('derivs', {})
        units    = args.get('units', None)

        try:
            dtype = default.dtype
        except AttributeError:
            if isinstance(default, int):
                dtype = np.int_
            else:
                dtype = np.float_

        #--------------------------------------------------------------------
        # If the dictionary contains 'mask', it is either fully masked or
        # fully unmasked
        #--------------------------------------------------------------------
        try:
            mask = args['mask']
            if mask:
                values = np.empty(shape + item, dtype=dtype)
                repeater = (Ellipsis,) + len(item) * (slice(None),)
                values[repeater] = default
            else:
                values = args['values']

        #--------------------------------------------------------------------
        # Otherwise, take advantage of the corners and of the fact that only
        # unmasked values were saved
        #--------------------------------------------------------------------
        except KeyError:
    
            values = np.empty(shape + item, dtype=dtype)
            repeater = (Ellipsis,) + len(item) * (slice(None),)
            values[repeater] = default

            corners = args['corners']
            slicer = Qube.slicer_from_corners(args['corners'])
            mask = np.ones(shape, dtype=np.bool_)
            mask[slicer] = args['sliced_mask']

            values[np.logical_not(mask)] = args['unmasked_values']

        #------------------------------------------------------
        # Construct the object
        #------------------------------------------------------
        obj = Qube.__new__(cls)
        obj.__init__(values, mask, derivs=derivs, units=units,
                             nrank=nrank, drank=drank)

        #------------------------------------------------------
        # Set the readonly state as needed
        #------------------------------------------------------
        if readonly:
            obj = obj.as_readonly()

        return obj
    #===========================================================================



    ############################################################################
    # Properties and low-level access
    ############################################################################

    #===========================================================================
    # _set_values_
    #===========================================================================
    def _set_values_(self, values, mask=None, antimask=None):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Low-level method to update the values of an array.

        The read-only status of the object is defined by that of the given
        value.
        If a mask is provided, it is also updated.
        If antimask is not None, then only the array locations associated with
        the antimask are modified.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Confirm shapes
        #------------------------------------------------------
        if antimask is None:
            if np.shape(values) != np.shape(self.__values_):
                raise ValueError('value shape mismatch; old is ' +
                                 str(np.shape(self.__values_)) + '; new is ' +
                                 str(np.shape(values)))
            if isinstance(mask, np.ndarray):
                if np.shape(mask) != self.shape:
                    raise ValueError('mask shape mismatch; mask is ' +
                                      str(np.shape(mask)) + '; object is ' +
                                      str(self.shape))
        else:
            if np.shape(antimask):
                if np.shape(antimask) != self.shape:
                    raise ValueError('antimask shape mismatch; antimask is ' +
                                     str(np.shape(antimask)) + '; object is ' +
                                     str(self.shape))

        #------------------------------------------------------
        # Update values
        #------------------------------------------------------
        if antimask is None:
            self.__values_ = values
        elif np.isscalar(values):
            self.__values_ = values
        else:
            self.__values_[antimask] = values[antimask]

        self.__readonly_ = Qube._array_is_readonly(self.__values_)

        #------------------------------------------------------
        # Update the mask if necessary
        #------------------------------------------------------
        if mask is not None:

            if antimask is None:
                self.__mask_ = mask
            elif np.isscalar(mask):
                if np.isscalar(self.__mask_):
                    old_mask = self.__mask_
                    self.__mask_ = np.empty(self.shape, dtype=np.bool_)
                    self.__mask_.fill(old_mask)
                self.__mask_[antimask] = mask
            else:
                self.__mask_[antimask] = mask[antimask]

        self.__cache_.clear()

        #------------------------------------------------------
        # Set the readonly state based on the values given
        #------------------------------------------------------
        if np.shape(self.__mask_):
            if self.__readonly_:
                self.__mask_ = Qube._array_to_readonly(self.__mask_)

            elif Qube._array_is_readonly(self.__mask_):
                self.__mask_ = self.__mask_.copy()

        return self
    #===========================================================================



    #===========================================================================
    # _set_mask_
    #===========================================================================
    def _set_mask_(self, mask, antimask=None):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Low-level method to update the mask of an array.

        The read-only status of the object will be preserved.
        If antimask is not None, then only the mask locations associated with
        the antimask are modified
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        # Confirm the shape
        if not isinstance(mask, (bool,np.bool_)) and mask.shape != self.shape:
            raise ValueError('mask shape mismatch; mask is ' +
                             str(mask.shape) + '; object is ' +
                             str(self.shape))

        is_readonly = self.__readonly_

        #------------------------------------------------------
        # Update the mask
        #------------------------------------------------------
        if antimask is None:
            self.__mask_ = mask
        elif np.isscalar(mask):
            if np.isscalar(self.__mask_):
                old_mask = self.__mask_
                self.__mask_ = np.empty(self.shape, dtype=np.bool_)
                self.__mask_.fill(old_mask)
            self.__mask_[antimask] = mask
        else:
            self.__mask_[antimask] = mask[antimask]

        self.__cache_.clear()

        if isinstance(self.__mask_, np.ndarray):
            if is_readonly:
                self.__mask_ = Qube._array_to_readonly(self.__mask_)

            elif Qube._array_is_readonly(self.__mask_):
                self.__mask_ = self.__mask_.copy()

        #------------------------------------------------------
        # Avoid type np.bool_ if possible
        #------------------------------------------------------
        if np.isscalar(self.__mask_):
            if self.__mask_:
                self.__mask_ = True
                self.__cache_['antimask'] = False
            else:
                self.__mask_ = False
                self.__cache_['antimask'] = True

        return self
    #===========================================================================



    #===========================================================================
    # values
    #===========================================================================
    @property
    def values(self): return self.__values_
    #===========================================================================



    #===========================================================================
    # vals
    #===========================================================================
    @property
    def vals(self): return self.__values_        # Handy shorthand
    #===========================================================================



    #===========================================================================
    # mvals
    #===========================================================================
    @property
    def mvals(self):

        #------------------------------------------------------
        # Deal with a scalar
        #------------------------------------------------------
        if np.isscalar(self.__values_):
            if self.__mask_:
                return np.ma.masked
            else:
                return self.__values_

        #------------------------------------------------------
        # Construct something that behaves as a suitable mask
        #------------------------------------------------------
        flag = Qube.as_one_bool(self.__mask_)
        if flag is False:
            newmask = np.ma.nomask
        elif flag is True:
            newmask = np.ones(self.__values_.shape, dtype=np.bool_)
        elif self.__rank_ > 0:
            newmask = self.__mask_.reshape(self.__shape_ + self.__rank_ * (1,))
            (newmask, newvals) = np.broadcast_arrays(newmask, self.__values_)
        else:
            newmask = self.__mask_

        return np.ma.MaskedArray(self.__values_, newmask)
    #===========================================================================



    #===========================================================================
    # mask
    #===========================================================================
    @property
    def mask(self):
        if isinstance(self.__mask_, np.ndarray):
            return self.__mask_

        return bool(self.__mask_)
    #===========================================================================



    #===========================================================================
    # antimask
    #===========================================================================
    @property
    def antimask(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        The inverse of the mask, True where an element is valid.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if not Qube.DISABLE_CACHE and 'antimask' in self.__cache_:
            return self.__cache_['antimask']

        if isinstance(self.__mask_, np.ndarray):
            antimask = np.logical_not(self.__mask_)
            self.__cache_['antimask'] = antimask
            return antimask

        antimask = not self.__mask_
        self.__cache_['antimask'] = antimask
        return antimask
    #===========================================================================



    #===========================================================================
    # default
    #===========================================================================
    @property
    def default(self): return self.__default_
    #===========================================================================



    #===========================================================================
    # units
    #===========================================================================
    @property
    def units(self): return self.__units_
    #===========================================================================



    #===========================================================================
    # derivs
    #===========================================================================
    @property
    def derivs(self): return self.__derivs_
    #===========================================================================



    #===========================================================================
    # shape
    #===========================================================================
    @property
    def shape(self): return self.__shape_
    #===========================================================================



    #===========================================================================
    # rank
    #===========================================================================
    @property
    def rank(self): return self.__rank_
    #===========================================================================



    #===========================================================================
    # nrank
    #===========================================================================
    @property
    def nrank(self): return self.__nrank_
    #===========================================================================



    #===========================================================================
    # drank
    #===========================================================================
    @property
    def drank(self): return self.__drank_
    #===========================================================================



    #===========================================================================
    # item
    #===========================================================================
    @property
    def item(self): return self.__item_
    #===========================================================================



    #===========================================================================
    # numer
    #===========================================================================
    @property
    def numer(self): return self.__numer_
    #===========================================================================



    #===========================================================================
    # denom
    #===========================================================================
    @property
    def denom(self): return self.__denom_
    #===========================================================================



    #===========================================================================
    # size
    #===========================================================================
    @property
    def size(self):
        return int(np.prod(self.__shape_))
    #===========================================================================



    #===========================================================================
    # isize
    #===========================================================================
    @property
    def isize(self):
        return int(np.prod(self.__item_))
    #===========================================================================



    #===========================================================================
    # nsize
    #===========================================================================
    @property
    def nsize(self):
        return int(np.prod(self.__numer_))
    #===========================================================================



    #===========================================================================
    # dsize
    #===========================================================================
    @property
    def dsize(self):
        return int(np.prod(self.__denom_))
    #===========================================================================



    #===========================================================================
    # readonly
    #===========================================================================
    @property
    def readonly(self):
        return self.__readonly_
    #===========================================================================



    #===========================================================================
    # is_deriv
    #===========================================================================
    @property
    def is_deriv(self):
        return self.__is_deriv_
    #===========================================================================



    #===========================================================================
    # wod
    #===========================================================================
    @property
    def wod(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A shallow clone without derivatives, cached. Read-only objects
        remain read-only.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if not self.__derivs_:
            return self

        if not Qube.DISABLE_CACHE and 'wod' in self.__cache_:
            return self.__cache_['wod']

        wod = Qube.__new__(type(self))
        wod.__init__(self.__values_, self.__mask_, example=self)
        for key,attr in self.__dict__.items():
            if key.startswith('d_d'):
                pass
            elif isinstance(attr, Qube):
                wod.__dict__[key] = attr.wod
            else:
                wod.__dict__[key] = attr

        wod.__derivs_ = {}
        wod.__cache_['wod'] = wod
        self.__cache_['wod'] = wod
        return wod
    #===========================================================================



    #===========================================================================
    # find_corners
    #===========================================================================
    def find_corners(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Update the corner indices such that everything outside this defined
        "hypercube" is masked.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        shape = self.__shape_
        lshape = len(shape)
        index0 = lshape * (0,)

        if lshape == 0:
            return None

        flag = Qube.as_one_bool(self.__mask_)
        if flag is False:
            return (index0, shape)

        if Qube.is_one_true(self.__mask_):
            return (index0, index0)

        lower = []
        upper = []
        antimask = self.antimask

        for axis in range(lshape):
            other_axes = list(range(lshape))
            del other_axes[axis]

            occupied = np.any(antimask, tuple(other_axes))
            indices = np.where(occupied)[0]
            if len(indices) == 0:
                return (index0, index0)

            lower.append(indices[0])
            upper.append(indices[-1] + 1)

        return (tuple(lower), tuple(upper))
    #===========================================================================



    #===========================================================================
    # cache
    #===========================================================================
    @property
    def cache(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A dictionary for anything to be cached. The cache is cleared whenever
        the object is modified.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return self.__cache_
    #===========================================================================



    #===========================================================================
    # corners
    #===========================================================================
    @property
    def corners(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Corners of a "hypercube" that contain all the unmasked array
        elements.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if not Qube.DISABLE_CACHE and 'corners' in self.__cache_:
            return self.__cache_['corners']

        corners = self.find_corners()
        self.__cache_['corners'] = corners
        return corners
    #===========================================================================



    #===========================================================================
    # slicer_from_corners
    #===========================================================================
    @staticmethod
    def slicer_from_corners(corners):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
A slice object based on corners specified as a tuple of indices.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        slice_objects = []
        for axis in range(len(corners[0])):
            slice_objects.append(slice(corners[0][axis], corners[1][axis]))

        return tuple(slice_objects)
    #===========================================================================



    #===========================================================================
    # slicer
    #===========================================================================
    @property
    def slicer(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A slice object containing all the array elements inside the current
        corners.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if not Qube.DISABLE_CACHE and 'slicer' in self.__cache_:
            return self.__cache_['slicer']

        slicer = Qube.slicer_from_corners(self.corners)
        self.__cache_['slicer'] = slicer
        return slicer
    #===========================================================================



    #===========================================================================
    # as_one_bool
    #===========================================================================
    @staticmethod
    def as_one_bool(value):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Convert a single value to a bool; leave other values unchanged.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if isinstance(value, (bool, np.bool_)):
            return bool(value)

        return value
    #===========================================================================



    #===========================================================================
    # is_one_true
    #===========================================================================
    @staticmethod
    def is_one_true(value):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        True if the value is a single boolean True
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if isinstance(value, (bool, np.bool_)):
            return bool(value)

        return False
    #===========================================================================



    #===========================================================================
    # is_one_false
    #===========================================================================
    @staticmethod
    def is_one_false(value):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        True if the value is a single boolean False
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if isinstance(value, (bool, np.bool_)):
            return not bool(value)

        return False
    #===========================================================================



    ############################################################################
    # Derivative operations
    ############################################################################

    #===========================================================================
    # insert_deriv
    #===========================================================================
    def insert_deriv(self, key, deriv, override=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Insert or replace a derivative in this object.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if not self.DERIVS_OK:
            raise TypeError("derivatives are disallowed in class '%s'" %
                            type(self).__name__)

        #------------------------------------------------------
        # Make sure the derivative is compatible with the object
        #------------------------------------------------------
        if not isinstance(deriv, Qube):
            raise ValueError("invalid class for derivative '%s': '%s'" %
                             (key, type(deriv).__name__))

        if self.__numer_ != deriv.__numer_:
            raise ValueError(("shape mismatch for numerator of derivative " +
                              "'%s': %s, %s") % (key, str(deriv.__numer_),
                                                      str(self.__numer_)))

        if self.readonly and (key in self.__derivs_) and not override:
            raise ValueError('derivative d_d' + key + ' cannot be replaced ' +
                             'in a read-only object')

        #------------------------------------------------------
        # Prevent recursion, convert to floating point
        #------------------------------------------------------
        deriv = deriv.wod.as_float()

        #------------------------------------------------------
        # Match readonly of parent if necessary
        #------------------------------------------------------
        if self.__readonly_ and not deriv.__readonly_:
            deriv = deriv.clone(recursive=False).as_readonly()

        #------------------------------------------------------
        # Save in the derivative dictionary and as an attribute
        #------------------------------------------------------
        if deriv.__shape_ != self.__shape_:
            deriv = deriv.broadcast_into_shape(self.__shape_)

        self.__derivs_[key] = deriv
        setattr(self, 'd_d' + key, deriv)

        deriv.__is_deriv_ = True

        self.__cache_.clear()
        return self
    #===========================================================================



    #===========================================================================
    # insert_derivs
    #===========================================================================
    def insert_derivs(self, derivs, override=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Insert or replace the derivatives in this object from a dictionary.

        You cannot replace the pre-existing values of any derivative in a
        read-only object unless you explicit set override=True. However,
        inserting a new derivative into a read-only object is not prevented.

        Input:
            derivs      the dictionary of derivatives, keyed by their names.

            override    True to allow the value of a pre-existing derivative to
                        be replaced.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #-----------------------------------------------
        # Check every insert before proceeding with any
        #-----------------------------------------------
        if self.readonly and not override:
            for key in derivs:
                if key in self.__derivs_:
                    raise ValueError('derivative d_d' + key + ' cannot be ' +
                                     'replaced in a read-only object')

        #-----------------------------------------------
        # Insert derivatives
        #-----------------------------------------------
        for (key, deriv) in derivs.items():
            self.insert_deriv(key, deriv, override)
    #===========================================================================



    #===========================================================================
    # delete_deriv
    #===========================================================================
    def delete_deriv(self, key, override=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Delete a single derivative from this object, given the key.

        Derivatives cannot be deleted from a read-only object without explicitly
        setting override=True.

        Input:
            key         the name of the derivative to delete.
            override    True to allow the deleting of derivatives from a
                        read-only object.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if not override:
            self.require_writable()

        if key in self.__derivs_.keys():
            del self.__derivs_[key]
            del self.__dict__['d_d' + key]

        self.__cache_.clear()
    #===========================================================================



    #===========================================================================
    # delete_derivs
    #===========================================================================
    def delete_derivs(self, override=False, preserve=None):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Delete all derivatives from this object.

        Derivatives cannot be deleted from a read-only object without explicitly
        setting override=True.

        Input:
            override    True to allow the deleting of derivatives from a
                        read-only object.
            preserve    an optional list, tuple or set of the names of
                        derivatives to retain. All others are removed.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if not override:
            self.require_writable()

        #------------------------------------------------------
        # If something is being preserved...
        #------------------------------------------------------
        if preserve:

            #- - - - - - - - - - - - - - - - - - - - - 
            # Delete derivatives not on the list
            #- - - - - - - - - - - - - - - - - - - - - 
            for key in self.__derivs_.keys():
                if key not in preserve:
                    self.delete_deriv(key, override)

            return

        #------------------------------------------------------
        # Delete all derivatives
        #------------------------------------------------------
        for key in self.__derivs_.keys():
            delattr(self, 'd_d' + key)

        self.__derivs_ = {}
        self.__cache_.clear()
        return
    #===========================================================================



    #===========================================================================
    # without_derivs
    #===========================================================================
    def without_derivs(self, preserve=None):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A shallow copy of this object without derivatives.

        A read-only object remains read-only, and is cached for later use.

        Input:
            preserve    an optional list, tuple or set of the names of
                        derivatives to retain. All others are removed.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if not self.__derivs_:
            return self

        #------------------------------------------------------
        # If something is being preserved...
        #------------------------------------------------------
        if preserve:
            if isinstance(preserve, str):
                preserve = [preserve]

            if not any([p for p in preserve if p in self.__derivs_]):
                return self.wod

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Create a fast copy with derivatives
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            obj = self.clone(recursive=True)

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Delete derivatives not on the list
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            deletions = []
            for key in obj.__derivs_:
                if key not in preserve:
                    deletions.append(key)

            for key in deletions:
                obj.delete_deriv(key, True)

            return obj

        #------------------------------------------------------
        # Return a fast copy without derivatives
        #------------------------------------------------------
        return self.wod
    #===========================================================================



    #===========================================================================
    # without_deriv
    #===========================================================================
    def without_deriv(self, key):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A shallow copy of this object without a particular derivative.
        A read-only object remains read-only.

        Input:
            key         the key of the derivative to remove.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if key not in self.__derivs_:
            return self

        result = self.clone(recursive=True)
        del result.__derivs_[key]

        return result
    #===========================================================================



    #===========================================================================
    # with_deriv
    #===========================================================================
    def with_deriv(self, key, value, method='insert'):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A shallow copy of this object with a derivative inserted or added.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        result = self.clone(recursive=True)

        assert method in ('insert', 'replace', 'add')

        if key in result.__derivs_:
            if method == 'insert':
                raise ValueError('derivative d_d%s already exists' % key)

            if method == 'add':
                value = value + result.__derivs_[key]

        result.insert_deriv(key, value)
        return result
    #===========================================================================



    #===========================================================================
    # unique_deriv_name
    #===========================================================================
    def unique_deriv_name(self, key, *objects):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        The given name for a deriv if it does not exist in this object or any
        of the given objects; otherwise return a variant that is unique.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Make a list of all the derivative keys
        #------------------------------------------------------
        all_keys = set(self.derivs.keys())
        for obj in objects:
            if not hasattr(obj, 'derivs'):
                continue
            all_keys |= set(obj.derivs.keys())

        #------------------------------------------------------
        # Return the proposed key if it is unused
        #------------------------------------------------------
        if key not in all_keys:
            return key

        #-------------------------------------------------------------------
        # Otherwise, tack on a number and iterate until the name is unique
        #-------------------------------------------------------------------
        i = 0
        while True:
            unique = key + str(i)
            if unique not in all_keys:
                return unique

            i += 1
    #===========================================================================

    ############################################################################
    # Unit operations
    ############################################################################

    #===========================================================================
    # set_units
    #===========================================================================
    def set_units(self, units, override=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Set the units of this object.

        Input:
            units       the new units.
            override    if True, the units can be modified on a read-only
                        object.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if not self.UNITS_OK and units is not None:
            raise TypeError("units are disallowed in class '%s'" %
                            type(self).__name__)

        if not override:
            self.require_writable()

        units = Units.as_units(units)

        Units.require_compatible(units, self.__units_)
        self.__units_ = units
        self.__cache_.clear()
    #===========================================================================



    #===========================================================================
    # without_units
    #===========================================================================
    def without_units(self, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A shallow copy of this object without derivatives.

        A read-only object remains read-only. If recursive is True, derivatives
        are also stripped of their units.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if self.__units_ is None and not self.__derivs_:
            return self

        obj = self.clone(recursive)
        obj.__units_ = None
        return obj
    #===========================================================================



    #===========================================================================
    # into_units
    #===========================================================================
    def into_units(self, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A copy of this object with values scaled to its units.

        Normally, the values stored internally in standard units of km, radians,
        and seconds. This method overrides that standard and converts the
        internal values to their intended units. Used primarily for I/O.

        If this object has no units, or is already in standard units, it is
        returned as is.

        Inputs:
            recursive   if True, the derivatives are also converted; otherwise
                        derivatives are removed.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Handle easy cases first
        #------------------------------------------------------
        if recursive or not self.__derivs_:
            if self.__units_ is None:
                return self
            if self.__units_.into_units_factor == 1.:
                return self

        #------------------------------------------------------
        # Construct the new object
        #------------------------------------------------------
        obj = self.clone(recursive)
        obj._set_values_(Units.into_units(self.__units_, self.__values_))

        #------------------------------------------------------
        # Fill in derivatives if necessary
        #------------------------------------------------------
        if recursive:
            for (key, deriv) in self.__derivs_.items():
                obj.insert_deriv(key, deriv.into_units(recursive=False))

        return obj
    #===========================================================================



    #===========================================================================
    # from_units
    #===========================================================================
    def from_units(self, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A copy of this object with values scaled to standard units.

        This method undoes the conversion done by into_units().

        Inputs:
            recursive       if True, the derivatives are also converted;
                            otherwise, derivatives are removed.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Handle easy cases first
        #------------------------------------------------------
        if recursive or not self.__derivs_:
            if self.__units_ is None:
                return self
            if self.__units_.from_units_factor == 1.:
                return self

        #------------------------------------------------------
        # Construct the new object
        #------------------------------------------------------
        obj = self.clone(recursive)
        obj._set_values_(Units.from_units(self.__units_, self.__values_))

        #------------------------------------------------------
        # Fill in derivatives if necessary
        #------------------------------------------------------
        if recursive:
            for (key, deriv) in self.__derivs_.items():
                obj.insert_deriv(key, deriv.from_units(recursive=False))

        return obj
    #===========================================================================



    #===========================================================================
    # confirm_units
    #===========================================================================
    def confirm_units(self, units):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Raises a ValueError if the units are not compatible with this object.

        Input:
            units       the new units.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if not Units.can_match(self.units, units):
            raise ValueError('units are not compatible')

        return self
    #===========================================================================



    ############################################################################
    # Read-only/read-write operations
    ############################################################################

    #===========================================================================
    # _array_is_readonly
    #===========================================================================
    @staticmethod
    def _array_is_readonly(arg):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        True if the argument is a read-only NumPy ndarray.
        False means that it is either a writable array or a scalar.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        if not isinstance(arg, np.ndarray):
            return False

        return (not arg.flags['WRITEABLE'])
    #===========================================================================



    #===========================================================================
    # _array_to_readonly
    #===========================================================================
    @staticmethod
    def _array_to_readonly(arg):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Make the given array read-only. Returns the array.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if not isinstance(arg, np.ndarray):
            return arg

        arg.flags['WRITEABLE'] = False
        return arg
    #===========================================================================



    #===========================================================================
    # as_readonly
    #===========================================================================
    def as_readonly(self, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Convert this object to read-only. It is modified in place and
        returned.

        If recursive is False, the derivatives are removed. Otherwise, they are
        also converted to read-only.

        If this object is already read-only, it is returned as is. Otherwise,
        the internal __values_ and __mask_ arrays are modified as necessary.
        Once this happens, the internal arrays will also cease to be writable in
        any other object that shares them.

        Note that as_readonly() cannot be undone. Use copy() to create a
        writable copy of a readonly object.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # If it is already read-only, return
        #------------------------------------------------------
        if self.__readonly_:
            return self

        #------------------------------------------------------
        # Update the value if it is an array
        #------------------------------------------------------
        Qube._array_to_readonly(self.__values_)
        Qube._array_to_readonly(self.__mask_)
        self.__readonly_ = True

        #------------------------------------------------------
        # Update anything cached
        #------------------------------------------------------
        if not Qube.DISABLE_CACHE:
            for key,value in self.__cache_.items():
                if isinstance(value, Qube):
                    self.__cache_[key] = value.as_readonly(recursive)

        #------------------------------------------------------
        # Update the derivatives
        #------------------------------------------------------
        if recursive:
            for key in self.__derivs_:
                self.__derivs_[key].as_readonly()

        return self
    #===========================================================================



    #===========================================================================
    # match_readonly
    #===========================================================================
    def match_readonly(self, arg):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Sets the read-only status of this object equal to that of another.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if arg.__readonly_:
            return self.as_readonly()
        elif self.__readonly_:
            raise ValueError('object is read-only')

        return self
    #===========================================================================



    #===========================================================================
    # require_writable
    #===========================================================================
    def require_writable(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Raises a ValueError if the object is read-only.

        Used internally at the beginning of methods that will modify this
        object.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if self.__readonly_:
            raise ValueError('object is read-only')
    #===========================================================================



    ############################################################################
    # Copying operations and conversions
    ############################################################################

    #===========================================================================
    # copy
    #===========================================================================
    def copy(self, recursive=True, readonly=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Deep copy operation with additional options.

        Input:
            recursive   if True, derivatives will also be copied; if False,
                        the copy will not contain any derivatives.

            readonly    if True, the copy will be read-only. If the source is
                        already read-only, no array duplication is required.
                        if False, the returned copy is guaranteed to be an
                        entirely new copy, suitable for modification.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Create a shallow copy
        #------------------------------------------------------
        obj = self.clone(recursive=False)

        #------------------------------------------------------
        # Copying a readonly object is easy
        #------------------------------------------------------
        if self.__readonly_ and readonly:
            return obj

        #------------------------------------------------------
        # Copy the values
        #------------------------------------------------------
        if isinstance(self.__values_, np.ndarray):
            obj.__values_ = self.__values_.copy()
        else:
            obj.__values_ = self.__values_

        #------------------------------------------------------
        # Copy the mask
        #------------------------------------------------------
        if isinstance(self.__mask_, np.ndarray):
            obj.__mask_ = self.__mask_.copy()
        else:
            obj.__mask_ = self.__mask_

        obj.__cache_ = {}

        #------------------------------------------------------
        # Set the read-only state
        #------------------------------------------------------
        if readonly:
            obj.as_readonly()
        else:
            obj.__readonly_ = False

        #------------------------------------------------------
        # Make the derivatives read-only if necessary
        #------------------------------------------------------
        if recursive:
            for (key, deriv) in self.__derivs_.items():
                obj.insert_deriv(key, deriv.copy(recursive=False,
                                                 readonly=readonly))

        return obj
    #===========================================================================



    #===========================================================================
    # __copy__
    #===========================================================================
    def __copy__(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Python-standard copy function: A deep copy of this object unless 
        it is read-only.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return self.copy(recursive=True, readonly=False)
    #===========================================================================



    ############################################################################
    # Floats vs. integers vs. booleans
    ############################################################################

    #===========================================================================
    # is_numeric
    #===========================================================================
    def is_numeric(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        True if this object contains numbers; False if boolean.

        This method returns True. It is overridden by the Boolean subclass to
        return False.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        values = self.__values_
        if isinstance(values, np.ndarray):
            return values.dtype.kind != 'b'
        else:
            return not isinstance(values, (bool,np.bool_))
    #===========================================================================



    #===========================================================================
    # as_numeric
    #===========================================================================
    def as_numeric(self, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A numeric version of this object.

        This method normally returns the object itself without modification. It
        is overridden by the Boolean subclass to return an integer version equal
        to one where True and zero where False.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if self.is_numeric():
            if recursive:
                return self
            return self.wod

        #------------------------------------------------------
        # Array case
        #------------------------------------------------------
        if isinstance(values, np.ndarray):
            if values.dtype.kind != 'b':
                return self
            if self.INTS_OK:
                obj = Qube.__new__(type(self))
                obj.__init__(values.astype(np.int_), self.__mask_)
            elif self.FLOATS_OK:
                obj = Qube.__new__(type(self))
                obj.__init__(values.astype(np.float_), self.__mask_)
            else:
                obj = Qube(values.astype(np.int_), self.__mask_)

        #------------------------------------------------------
        # Scalar case
        #------------------------------------------------------
        else:
            if not isinstance(values, (bool,np.bool_)):
                return self
            if self.INTS_OK:
                obj = Qube.__new__(type(self))
                obj.__init__(int(values), self.__mask_)
            elif self.FLOATS_OK:
                obj = Qube.__new__(type(self))
                obj.__init__(float(values), self.__mask_)
            else:
                obj = Qube(int(values), self.__mask_)

        return obj
    #===========================================================================



    #===========================================================================
    # is_float
    #===========================================================================
    def is_float(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        True if this object contains floats; False if ints or booleans.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        values = self.__values_
        if isinstance(values, np.ndarray):
            return (values.dtype.kind == 'f')
        return isinstance(values, float)
    #===========================================================================



    #===========================================================================
    # as_float
    #===========================================================================
    def as_float(self, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A floating-point version of this object.

        If this object already contains floating-point values, it is returned
        as is. Otherwise, a copy is returned. Derivatives are not modified.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if self.is_float():
            if recursive:
                return self
            return self.wod

        values = self.__values_

        #------------------------------------------------------
        # Array case
        #------------------------------------------------------
        if isinstance(values, np.ndarray):
            if self.FLOATS_OK:
                obj = Qube.__new__(type(self))
                obj.__init__(values.astype(np.float_), self.__mask_,
                             derivs=self.__derivs_, example=self)
            else:
                raise TypeError("data type float is incompatible with class " +
                                type(self).__name__)

        #------------------------------------------------------
        # Scalar case
        #------------------------------------------------------
        else:
            if self.FLOATS_OK:
                obj = Qube.__new__(type(self))
                obj.__init__(float(values), self.__mask_,
                             derivs=self.__derivs_, example=self)
            else:
                raise TypeError("data type float is incompatible with class " +
                                type(self).__name__)

        return obj
    #===========================================================================



    #===========================================================================
    # is_int
    #===========================================================================
    def is_int(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        True if this object contains ints; False if floats or booleans.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        values = self.__values_
        if isinstance(values, np.ndarray):
            return (values.dtype.kind in 'ui')
        return (isinstance(values, numbers.Integral)
                and not isinstance(values, (bool,np.bool_)))
    #===========================================================================



    #===========================================================================
    # as_int
    #===========================================================================
    def as_int(self, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        An integer version of this object. Always round down.

        If this object already contains integers, it is returned as is.
        Otherwise, a copy is returned.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if self.is_int():
            if recursive:
                return self
            return self.wod

        values = self.__values_

        #------------------------------------------------------
        # Array case
        #------------------------------------------------------
        if isinstance(values, np.ndarray):
            if self.INTS_OK:
                obj = Qube.__new__(type(self))
                obj.__init__((values//1).astype(np.int_), self.__mask_)
            else:
                raise TypeError("data type int is incompatible with class " +
                                type(self).__name__)

        #------------------------------------------------------
        # Scalar case
        #------------------------------------------------------
        else:
            if self.INTS_OK:
                obj = Qube.__new__(type(self))
                obj.__init__(int(values//1), self.__mask_)
            else:
                raise TypeError("data type int is incompatible with class " +
                                type(self).__name__)

        return obj
    #===========================================================================



    #===========================================================================
    # is_bool
    #===========================================================================
    def is_bool(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        True if this object contains booleans; False otherwise.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        values = self.__values_
        if isinstance(values, np.ndarray):
            return (values.dtype.kind in 'b')
        return isinstance(values, (bool,np.bool_))
    #===========================================================================



    #===========================================================================
    # as_bool
    #===========================================================================
    def as_bool(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A boolean version of this object.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # If already boolean, return as is
        #------------------------------------------------------
        if self.is_bool():
            return self

        values = self.__values_

        #------------------------------------------------------
        # Array case
        #------------------------------------------------------
        if isinstance(values, np.ndarray):
            if self.BOOLS_OK:
                obj = Qube.__new__(type(self))
                obj.__init__(values.astype(np.bool_), self.__mask_)
            else:
                raise TypeError("data type bool is incompatible with class " +
                                type(self).__name__)

        #------------------------------------------------------
        # Scalar case
        #------------------------------------------------------
        else:
            if self.BOOLS_OK:
                obj = Qube.__new__(type(self))
                obj.__init__(bool(values), self.__mask_)
            else:
                raise TypeError("data type bool is incompatible with class " +
                                type(self).__name__)

        return obj
    #===========================================================================



    #===========================================================================
    # as_builtin
    #===========================================================================
    def as_builtin(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        This object as a Python built-in class (float, int, or bool) if the
        conversion can be done without loss of information.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        values = self.__values_
        if np.shape(values):
            return self
        if self.__mask_:
            return self

        if self.__units_ not in (None, Units.UNITLESS):
            return self

        if isinstance(values, (bool,np.bool_)):
            return bool(values)
        if isinstance(values, numbers.Integral):
            return int(values)
        if isinstance(values, numbers.Real):
            return float(values)

        return self     # This shouldn't happen
    #===========================================================================



    #===========================================================================
    # is_all_masked
    #===========================================================================
    def is_all_masked(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        True if this is entirely masked.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return np.all(self.__mask_)
    #===========================================================================



    ############################################################################
    # Subclass operations
    ############################################################################

    #===========================================================================
    # is_real_number
    #===========================================================================
    @staticmethod
    def is_real_number(arg):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        True if arg is of a Python numeric or NumPy numeric type.
        """ 
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return isinstance(arg, (numbers.Real, np.number))
    #===========================================================================



    #===========================================================================
    # masked_single
    #===========================================================================
    def masked_single(self, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        An object of this subclass containing one masked value.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if not self.__rank_:
            new_value = self.__default_
        else:
            new_value = self.__default_.copy()

        obj = Qube.__new__(type(self))
        obj.__init__(new_value, True, example=self)

        if recursive and self.__derivs_:
            for (key, value) in self.__derivs_.items():
                obj.insert_deriv(key, value.masked_single(recursive=False))

        obj.as_readonly()
        return obj
    #===========================================================================



    #===========================================================================
    # as_this_type
    #===========================================================================
    def as_this_type(self, arg, recursive=True, coerce=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        The argument converted to this class and data type.

        If the object is already of the correct class and type, it is returned
        unchanged. If the argument is a scalar or NumPy ndarray, a new instance
        of this object's class is created.

        The returned object will always retain its original drank.

        Input:
            arg         the object (built-in, NumPy ndarray or Qube subclass) to
                        convert to the class of this object.
            recursive   True to convert the derivatives as well.
            coerce      True to coerce the data type silently; False to leave
                        the data type unchanged.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #-------------------------------------------------------------------
        # If the classes already match, we might return the argument as is
        #-------------------------------------------------------------------
        if type(arg) == type(self):
            obj = arg
        else:
            obj = None

        #-------------------------------------------------------------------
        # Initialize the new values and mask; track other attributes
        #-------------------------------------------------------------------
        if not isinstance(arg, Qube):
            arg = Qube(arg, example=self)

        if arg.__nrank_ != self.__nrank_:
            raise ValueError('item shape mismatch')

        new_vals = arg.__values_
        new_mask = arg.__mask_
        new_units = arg.__units_
        has_derivs = bool(arg.__derivs_)
        is_readonly = arg.__readonly_

        #------------------------------------------------------
        # Convert the value types if necessary
        #------------------------------------------------------
        changed = False
        if coerce:
            if self.is_float():
                if isinstance(new_vals, np.ndarray):
                    if new_vals.dtype != np.float_:
                        new_vals = new_vals.astype(np.float_)
                        changed = True
                elif not isinstance(new_vals, float):
                    new_vals = float(new_vals)
                    changed = True

            elif self.is_int():
                if isinstance(new_vals, np.ndarray):
                    if new_vals.dtype != np.int_:
                        new_vals = new_vals.astype(np.int_)
                        changed = True
                elif not isinstance(new_vals, int):
                    new_vals = int(new_vals)
                    changed = True

            else:       # must be bool
                if isinstance(new_vals, np.ndarray):
                    if new_vals.dtype != np.bool_:
                        new_vals = new_vals.astype(np.bool_)
                        changed = True
                elif not isinstance(new_vals, bool):
                    new_vals = bool(new_vals)
                    changed = True

        #------------------------------------------------------
        # Convert the units if necessary
        #------------------------------------------------------
        if new_units and not self.UNITS_OK:
            new_units = None
            changed = True

        #------------------------------------------------------
        # Validate derivs
        #------------------------------------------------------
        if has_derivs and not self.DERIVS_OK:
            changed = True
        if has_derivs and not recursive:
            changed = True

        #------------------------------------------------------
        # Construct the new object if necessary
        #------------------------------------------------------
        if changed or obj is None:
            obj = Qube.__new__(type(self))
            obj.__init__(new_vals, new_mask, units=new_units,
                         example=self, drank=arg.__drank_)
            is_readonly = False

        #------------------------------------------------------
        # Update the derivatives if necessary
        #------------------------------------------------------
        if recursive and has_derivs:
            derivs_changed = False
            new_derivs = {}
            for (key, deriv) in arg.__derivs_.items():
                new_deriv = self.as_this_type(deriv, False, coerce=False)
                if new_deriv is not deriv:
                    derivs_changed = True
                new_derivs[key] = new_deriv

            if derivs_changed or (arg is not obj):
                if is_readonly:
                    obj = obj.copy(recursive=False)
                obj.insert_derivs(new_derivs)

        return obj
    #===========================================================================



    #===========================================================================
    # cast
    #===========================================================================
    def cast(self, classes):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A shallow copy of this object casted to another subclass.

        If a list or tuple of classes is provided, the object will be casted to
        the first suitable class in the list. If this object is already has the
        selected class, it is returned without modification.

        Input:
            classes     a single Qube subclass, or a list or tuple of Qube
                        subclasses.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Convert a single class to a tuple
        #------------------------------------------------------
        if isinstance(classes, type):
            classes = (classes,)

        #------------------------------------------------------
        # For each class in the list...
        #------------------------------------------------------
        for cls in classes:

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # If this is already the class of this object, return it as is
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if cls is type(self):
                return self

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Exclude the class if it is incompatible
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if cls.NUMER is not None and cls.NUMER != self.__numer_:
                continue
            if cls.NRANK is not None and cls.NRANK != self.__nrank_:
                continue

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Construct the new object
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            obj = Qube.__new__(cls)
            obj.__init__(self.__values_, self.__mask_, derivs=self.__derivs_,
                         example=self)
            return obj

        #------------------------------------------------------
        # If no suitable class was found, return this object unmodified
        #------------------------------------------------------
        return self
    #===========================================================================



    ############################################################################
    # Masking operations
    ############################################################################

    #===========================================================================
    # mask_where
    #===========================================================================
    def mask_where(self, mask, replace=None, remask=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A copy of this object after a mask has been applied.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Convert to boolean array if necessary
        #------------------------------------------------------
        if isinstance(mask, np.ndarray):
            mask = mask.astype(np.bool_)
        elif isinstance(mask, Qube):
            mask = mask.as_mask_where_nonzero_or_masked()
        else:
            mask = Qube.BOOLEAN_CLASS.as_boolean(mask).__values_

        #------------------------------------------------------
        # If the mask is empty, return the object as is
        #------------------------------------------------------
        if not np.any(mask):
            return self

        #------------------------------------------------------
        # Get the replacement value as this type
        #------------------------------------------------------
        if replace is not None:
            replace = self.as_this_type(replace, recursive=False)
            if replace.__shape_ not in ((), self.__shape_):
                raise ValueError('shape of replacement is incompatible with ' +
                                 'shape of object being masked: %s, %s' %
                                 (replace.__shape_, self.__shape_))

        #------------------------------------------------------
        # Shapeless case
        #------------------------------------------------------
        if np.isscalar(self.__values_):
            if np.shape(mask):
                raise ValueError('object and mask have incompatible shapes: ' +
                                 '%s, %s' % (self.__shape_, np.shape(mask)))
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

        #------------------------------------------------------
        # Construct the new mask
        #------------------------------------------------------
        if remask:
            new_mask = self.__mask_ | mask
        elif np.shape(self.__mask_):
            new_mask = self.__mask_.copy()
        else:
            new_mask = self.__mask_

        #------------------------------------------------------
        # Construct the new array of values
        #------------------------------------------------------
        if replace is None:
            new_values = self.__values_

        #------------------------------------------------------
        # If replacement is an array of values...
        #------------------------------------------------------
        elif replace.__shape_:
            new_values = self.__values_.copy()
            new_values[mask] = replace.__values_[mask]

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Update the mask if replacement values are masked
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if Qube.is_one_true(new_mask):
                pass
            elif Qube.is_one_false(replace.__mask_):
                pass
            else:
                if Qube.is_one_false(new_mask):
                    new_mask = np.zeros(self.__shape_, dtype=np.bool_)

                if Qube.is_one_true(replace.__mask_):
                    new_mask[mask] = True
                else:
                    new_mask[mask] = replace.__mask_[mask]

        #------------------------------------------------------
        # If replacement is a single value...
        #------------------------------------------------------
        else:
            new_values = self.__values_.copy()
            new_values[mask] = replace.__values_

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Update the mask if replacement values are masked
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if replace.__mask_:
                if np.shape(new_mask):
                    new_mask[mask] = True
                else:
                    new_mask = True

        #------------------------------------------------------
        # Construct the new object and return
        #------------------------------------------------------
        obj = self.clone(recursive=True)
        obj._set_values_(new_values, new_mask)
        return obj
    #===========================================================================



    #===========================================================================
    # mask_where_eq
    #===========================================================================
    def mask_where_eq(self, match, replace=None, remask=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A copy of this object with items equal to a value masked.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        match = self.as_this_type(match, recursive=False)

        mask = (self.__values_ == match.__values_)
        for r in range(self.__rank_):
            mask = np.all(mask, axis=-1)

        return self.mask_where(mask, replace, remask)
    #===========================================================================



    #===========================================================================
    # mask_where_ne
    #===========================================================================
    def mask_where_ne(self, match, replace=None, remask=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A copy of this object with items not equal to a value masked.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        match = self.as_this_type(match, recursive=False)

        mask = (self.__values_ != match.__values_)
        for r in range(self.__rank_):
            mask = np.any(mask, axis=-1)

        return self.mask_where(mask, replace, remask)
    #===========================================================================



    #===========================================================================
    # mask_where_le
    #===========================================================================
    def mask_where_le(self, limit, replace=None, remask=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A copy of this object with items <= a limit value masked.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if self.__rank_:
            raise ValueError('mask_where_le requires item rank zero')

        if isinstance(limit, Qube):
            limit = limit.__values_

        return self.mask_where(self.__values_ <= limit, replace, remask)
    #===========================================================================



    #===========================================================================
    # mask_where_ge
    #===========================================================================
    def mask_where_ge(self, limit, replace=None, remask=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A copy of this object with items >= a limit value masked.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if self.__rank_:
            raise ValueError('mask_where_ge requires item rank zero')

        if isinstance(limit, Qube):
            limit = limit.__values_

        return self.mask_where(self.__values_ >= limit, replace, remask)
    #===========================================================================



    #===========================================================================
    # mask_where_lt
    #===========================================================================
    def mask_where_lt(self, limit, replace=None, remask=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A copy with items less than a limit value masked.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if self.__rank_:
            raise ValueError('mask_where_lt requires item rank zero')

        if isinstance(limit, Qube):
            limit = limit.__values_

        return self.mask_where(self.__values_ < limit, replace, remask)
    #===========================================================================



    #===========================================================================
    # mask_where_gt
    #===========================================================================
    def mask_where_gt(self, limit, replace=None, remask=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A copy with items greater than a limit value masked.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if self.__rank_:
            raise ValueError('mask_where_gt requires item rank zero')

        if isinstance(limit, Qube):
            limit = limit.__values_

        return self.mask_where(self.__values_ > limit, replace, remask)
    #===========================================================================



    #===========================================================================
    # mask_where_between
    #===========================================================================
    def mask_where_between(self, lower, upper, mask_endpoints=False,
                                 replace=None, remask=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A copy with values between two limits masked.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if self.__rank_:
            raise ValueError('mask_where_between requires item rank zero')

        if isinstance(lower, Qube):
            lower = lower.__values_

        if isinstance(upper, Qube):
            upper = upper.__values_

        if mask_endpoints:      # end points are included in the mask
            mask = (self.__values_ >= lower) & (self.__values_ <= upper)
        else:                   # end points are not included in the mask
            mask = (self.__values_ > lower) & (self.__values_ < upper)

        return self.mask_where(mask, replace, remask)
    #===========================================================================



    #===========================================================================
    # mask_where_outside
    #===========================================================================
    def mask_where_outside(self, lower, upper, mask_endpoints=False,
                                 replace=None, remask=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A copy with values outside two limits masked.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if self.__rank_:
            raise ValueError('mask_where_outside requires item rank zero')

        if isinstance(lower, Qube):
            lower = lower.__values_

        if isinstance(upper, Qube):
            upper = upper.__values_

        if mask_endpoints:      # end points are included in the mask
            mask = (self.__values_ <= lower) | (self.__values_ >= upper)
        else:                   # end points are not included in the mask
            mask = (self.__values_ < lower) | (self.__values_ > upper)

        return self.mask_where(mask, replace, remask)
    #===========================================================================



    #===========================================================================
    # clip
    #===========================================================================
    def clip(self, lower, upper, remask=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A copy with values clipped to fall within a pair of limits.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
    #===========================================================================



    #===========================================================================
    # count_masked
    #===========================================================================
    def count_masked(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        The number of masked items in this object.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        mask = Qube.as_one_bool(self.__mask_)
        if mask is True:
            return self.size
        elif mask is False:
            return 0
        else:
            return np.count_nonzero(self.__mask_)
    #===========================================================================



    #===========================================================================
    # masked
    #===========================================================================
    def masked(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        The number of masked items in this object. DEPRECATED NAME;
        use count_masked().
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return self.count_masked()
    #===========================================================================



    #===========================================================================
    # count_unmasked
    #===========================================================================
    def count_unmasked(self):
        """
        The number of unmasked items in this object.
        """

        mask = Qube.as_one_bool(self.__mask_)
        if mask is True:
            return 0
        elif mask is False:
            return self.size
        else:
            return self.size - np.count_nonzero(self.__mask_)
    #===========================================================================



    #===========================================================================
    # unmasked
    #===========================================================================
    def unmasked(self):
        """
        The number of unmasked items in this object. DEPRECATED NAME;
        use count_unmasked()
        """

        return self.count_unmasked()
    #===========================================================================



    #===========================================================================
    # without_mask
    #===========================================================================
    def without_mask(self, recursive=True):
        """
        A shallow copy of this object without its mask.
        """

        obj = self.clone()

        if Qube.is_one_false(self.__mask_):
            return obj

        obj._set_mask_(False)

        if recursive:
            for (key,deriv) in self.__derivs_.items():
                obj.insert_deriv(key, deriv.without_mask())

        return obj
    #===========================================================================



    #===========================================================================
    # as_all_masked
    #===========================================================================
    def as_all_masked(self, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A shallow copy of this object with everything masked.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        obj = self.clone()

        if Qube.is_one_true(self.__mask_):
            return obj

        obj._set_mask_(True)

        if recursive:
            for (key,deriv) in self.__derivs_.items():
                obj.insert_deriv(key, deriv.as_all_masked(False))

        return obj
    #===========================================================================



    #===========================================================================
    # all_masked
    #===========================================================================
    def all_masked(self, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A shallow copy of this object with everything masked.
        DEPRECATED NAME; use as_all_masked()
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        return self.as_all_masked(recursive)
    #===========================================================================



    #===========================================================================
    # remask
    #===========================================================================
    def remask(self, mask, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A shallow copy of this object with a replaced mask.

        This is much quicker than masked_where(), for cases where only the mask
        is changing.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if np.shape(mask) not in (self.__shape_, ()):
            raise ValueError('mask shape is incompatible with object: ' +
                             str(np.shape(mask)) + ', ' + str(self.__shape_))

        #------------------------------------------------------
        # Construct the new object
        #------------------------------------------------------
        obj = self.clone(recursive=False)
        obj._set_mask_(mask)

        if recursive:
            for (key,deriv) in self.__derivs_.items():
                obj.insert_deriv(key, deriv.remask(mask))

        return obj
    #===========================================================================



    #===========================================================================
    # expand_mask
    #===========================================================================
    def expand_mask(self, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A shallow copy where a single mask value of True or False is
        converted to an array.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        obj = self.clone(recursive=False)

        if np.shape(self.__mask_) and not (recursive and self.__derivs_):
            return obj

        if Qube.is_one_false(obj.__mask_):
            obj._set_mask_(np.zeros(self.__shape_, dtype=np.bool_))
        elif Qube.is_one_true(obj.__mask_):
            obj._set_mask_(np.ones(self.__shape_, dtype=np.bool_))

        if recursive and self.__derivs_:
            for (key,deriv) in self.__derivs_.items():
                obj.insert_deriv(key, deriv.expand_mask(recursive=False))

        return obj
    #===========================================================================



    #===========================================================================
    # as_all_constant
    #===========================================================================
    def as_all_constant(self, constant=None, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A shallow, read-only copy of this object with constant values.

        Derivatives are all set to zero. The mask is unchanged.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if constant is None:
            constant = self.zero()

        constant = self.as_this_type(constant, recursive=False)

        obj = self.clone(recursive=False)
        obj._set_values_(Qube.broadcast(constant, obj)[0].__values_)
        obj.as_readonly()

        if recursive:
            for (key,deriv) in self.__derivs_.items():
                obj.insert_deriv(key, deriv.all_constant(recursive=False))

        return obj
    #===========================================================================



    #===========================================================================
    # all_constant
    #===========================================================================
    def all_constant(self, constant=None, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A shallow, read-only copy of this object with constant values.
        DEPRECATED NAME; use as_all_constant().

        Derivatives are all set to zero. The mask is unchanged.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return self.as_all_constant(constant, recursive)
    #===========================================================================



    #===========================================================================
    # as_size_zero
    #===========================================================================
    def as_size_zero(self, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A shallow, read-only copy of this object with size zero.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        obj = Qube.__new__(type(self))

        if self.__shape_:
            new_values = self.__values_[:0]

            if np.shape(self.__mask_):
                new_mask = self.__mask_[:0]
            else:
                new_mask = np.array([self.__mask_])[:0]

        else:
            new_values = np.array([self.__values_])[:0]
            new_mask = np.array([self.__mask_])[:0]

        obj.__init__(new_values, new_mask, example=self)

        if recursive:
            for (key,deriv) in self.__derivs_.items():
                obj.insert_deriv(key, deriv.as_size_zero(recursive=False))

        return obj
    #===========================================================================



    #===========================================================================
    # as_mask_where_nonzero
    #===========================================================================
    def as_mask_where_nonzero(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A scalar or NumPy array where values are nonzero and unmasked.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        mask = Qube.as_one_bool(self.__mask_)
        if mask is True:
            return False

        if mask is False:
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
    #===========================================================================



    #===========================================================================
    # as_mask_where_zero
    #===========================================================================
    def as_mask_where_zero(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A scalar or NumPy array where values are zero and unmasked.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        mask = Qube.as_one_bool(self.__mask_)
        if mask is True:
            return False

        if mask is False:
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
    #===========================================================================



    #===========================================================================
    # as_mask_where_nonzero_or_masked
    #===========================================================================
    def as_mask_where_nonzero_or_masked(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A scalar or NumPy array where values are nonzero or masked.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        mask = Qube.as_one_bool(self.__mask_)
        if mask is True:
            return True

        if mask is False:
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
    #===========================================================================



    #===========================================================================
    # as_mask_where_zero_or_masked
    #===========================================================================
    def as_mask_where_zero_or_masked(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A scalar or NumPy array where values are zero or masked.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        mask = Qube.as_one_bool(self.__mask_)
        if mask is True:
            return True

        if mask is False:
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
    #===========================================================================



    #===========================================================================
    # 
    # If this global is set to True, the shrink/unshrink methods are disabled.
    # Calculations done with and without shrinking should always produce the
    # same results, although they may be slower with shrinking disabled. Used
    # for testing and debugging.
    #===========================================================================
    _DISABLE_SHRINKING = False



    #===========================================================================
    # If this global is set to True, the unshrunk method will ignore any cached
    # value of its un-shrunken equivalent. Used for testing and debugging.
    #===========================================================================
    _IGNORE_UNSHRUNK_AS_CACHED = False
    #===========================================================================



    #===========================================================================
    # shrink
    #===========================================================================
    def shrink(self, antimask):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A 1-D version of this object, containing only the samples in the
        antimask provided.

        The antimask array value of True indicates that an element should be
        included; False means that is should be discarded. A scalar value of
        True or False applies to the entire object.

        The purpose is to speed up calculations by first eliminating all the
        objects that are masked. Any calculation involving un-shrunken objects
        should produce the same result if the same objects are all shrunken by
        a common antimask first, the calculation is performed, and then the
        result is un-shrunken afterward.

        Shrunken objects are always converted to read-only.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #### For testing only...
        if Qube._DISABLE_SHRINKING:
            if not self.__shape_ or Qube.is_one_true(antimask):
                return self
            return self.mask_where(np.logical_not(antimask))

        #------------------------------------------------------
        # A True antimask leaves an object unchanged
        #------------------------------------------------------
        if Qube.is_one_true(antimask):
            return self

        #------------------------------------------------------
        # If the antimask is a single False value, or if this object is already
        # entirely masked, return a single masked value
        #------------------------------------------------------
        if (Qube.is_one_true(self.__mask_) or Qube.is_one_false(antimask) or
            not np.any(antimask & self.antimask)):
                obj = self.masked_single().as_readonly()
                if not Qube.DISABLE_CACHE:
                    obj.__cache_['unshrunk'] = self
                return obj

        #------------------------------------------------------
        # If this is a shapeless object, return it as is
        #------------------------------------------------------
        if not self.__shape_:
            self.__cache_['unshrunk'] = self
            return self

        #------------------------------------------------------
        # Beyond this point, the size of the last axis in the returned object
        # will have the same number of elements as the number of True elements
        # in the antimask.
        #------------------------------------------------------

        #------------------------------------------------------
        # Ensure that this object and the antimask have compatible dimensions.
        # If the antimask has extra dimensions, broadcast self to make it work
        #------------------------------------------------------
        self_rank = len(self.__shape_)
        antimask_rank = antimask.ndim
        extras = self_rank - antimask_rank
        if extras < 0:
            self = self.broadcast_into_shape(antimask.shape, recursive=False)
            self_rank = antimask_rank
            extras = 0

        #------------------------------------------------------
        # If self has extra dimensions, these will be retained and only the
        # rightmost axes will be flattened.
        #------------------------------------------------------
        before = self.__shape_[:extras]     # shape of self retained
        after  = self.__shape_[extras:]     # shape of self to be masked

        #------------------------------------------------------
        # Make the rightmost axes of self and the mask compatible
        #------------------------------------------------------
        new_after = tuple([max(after[k],antimask.shape[k])
                           for k in range(len(after))])
        new_shape = before + new_after
        if self.__shape_ != new_shape:
            self = self.broadcast_into_shape(new_shape, recursive=False)
        if antimask.shape != new_after:
            antimask = np.broadcast_to(antimask, new_after)

        #------------------------------------------------------
        # Construct the new mask
        #------------------------------------------------------
        if Qube.is_one_false(self.__mask_):
            mask = np.zeros(antimask.shape, dtype=np.bool_)[antimask]
        else:
            mask = self.__mask_[extras * (slice(None),) + (antimask,Ellipsis)]

        if np.all(mask):
            obj = self.masked_single().as_readonly()
            obj.__cache_['unshrunk'] = self
            return obj

        if not np.any(mask):
            mask = False

        #------------------------------------------------------
        # Construct the new object
        #------------------------------------------------------
        obj = Qube.__new__(type(self))
        obj.__init__(self.__values_[extras * (slice(None),)
                                    + (antimask,Ellipsis)],
                     mask, example=self)
        obj.as_readonly()

        for (key, deriv) in self.__derivs_.items():
            obj.insert_deriv(key, deriv.shrink(antimask))

        #------------------------------------------------------
        # Cache values to speed things up later
        #------------------------------------------------------
        obj.__cache_['unshrunk'] = self
        return obj
    #===========================================================================



    #===========================================================================
    # unshrink
    #===========================================================================
    def unshrink(self, antimask, shape=()):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Convert an object to its un-shrunken shape, based on a given
        antimask.

        If this object was previously shrunken, the antimask must match the one
        used to shrink it. Otherwise, the size of this object's last axis must
        match the number of True values in the antimask.

        Input:
            antimask    the antimask to apply.
            shape       in cases where the antimask is a literal False, this
                        defines the shape of the returned object. Normally, the
                        rightmost axes of the returned object match those of
                        the antimask.

        The returned object will be read-only.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #### For testing only...
        if Qube._DISABLE_SHRINKING:
            return self

        #------------------------------------------------------
        # Get the previous unshrunk version if available and delete from cache
        #------------------------------------------------------
        if Qube.DISABLE_CACHE:
            unshrunk = None
        else:
            unshrunk = self.__cache_.get('unshrunk', None)
            if unshrunk is not None:
                del self.__cache_['unshrunk']

                if Qube._IGNORE_UNSHRUNK_AS_CACHED:
                    unshrunk = None

        #------------------------------------------------------
        # If the antimask is True, return this as is
        #------------------------------------------------------
        if Qube.is_one_true(antimask):
            return self

        #------------------------------------------------------
        # If the new object is entirely masked, return a shapeless masked object
        #------------------------------------------------------
        if not np.any(antimask) or np.all(self.__mask_):
            return self.masked_single()

        #------------------------------------------------------
        # If this object is shapeless, return it as is
        #------------------------------------------------------
        if not self.__shape_:
            return self

        #------------------------------------------------------
        # If we found a cached value, return it
        #------------------------------------------------------
        if unshrunk is not None:
            return unshrunk.mask_where(np.logical_not(antimask))

        #------------------------------------------------------
        # Create the new data array
        #------------------------------------------------------
        new_shape = self.__shape_[:-1] + antimask.shape
        indx = (len(self.__shape_)-1) * (slice(None),) + (antimask, Ellipsis)
        if isinstance(self.__values_, np.ndarray):
            default = self.__default_
            if isinstance(default, Qube):
                default = self.__default_.__values_

            new_values = np.empty(new_shape + self.__item_,
                                  self.__values_.dtype)
            new_values[...] = default

            new_values[indx] = self.__values_   # fill in non-default values

        #------------------------------------------------------
        # ...where single values can be handled by broadcasting...
        #------------------------------------------------------
        else:
            item = Scalar(self.__values_)
            new_values = item.broadcast_into_shape(new_shape).__values_

        #------------------------------------------------------
        # Create the new mask array
        #------------------------------------------------------
        new_mask = np.ones(new_shape, dtype=np.bool_)
        new_mask[indx] = self.__mask_       # insert the shrunk mask values

        #------------------------------------------------------
        # Create the new object
        #------------------------------------------------------
        obj = Qube.__new__(type(self))
        obj.__init__(new_values, new_mask, example=self)
        obj = obj.as_readonly()

        #------------------------------------------------------
        # Unshrink the derivatives
        #------------------------------------------------------
        for (key, deriv) in self.__derivs_.items():
            obj.insert_deriv(key, deriv.unshrink(antimask, shape))

        return obj
    #===========================================================================



    ############################################################################
    # I/O operations
    ############################################################################

    #===========================================================================
    # __repr__
    #===========================================================================
    def __repr__(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Express the value as a string.

        The format of the returned string is:
            Class([value, value, ...], suffices, ...)
        where the quanity inside square brackets is the result of str() applied
        to a NumPy ndarray.

        The suffices are, in order...
            - "mask" if the object has a mask
            - the name of the units of the object has units
            - the names of all the derivatives, in alphabetical order

        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return self.__str__()
    #===========================================================================



    #===========================================================================
    # __str__
    #===========================================================================
    def __str__(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Express the value as a string.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        suffix = []

        #------------------------------------------------------
        # Apply the units if necessary
        #------------------------------------------------------
        obj = self.into_units(recursive=False)

        #------------------------------------------------------
        # Indicate the denominator shape if necessary
        #------------------------------------------------------
        if self.__denom_ != ():
            suffix += ['denom=' + str(self.__denom_)]

        #------------------------------------------------------
        # Masked objects have a suffix ', mask'
        #------------------------------------------------------
        is_masked = np.any(self.__mask_)
        if is_masked:
            suffix += ['mask']

        #------------------------------------------------------
        # Objects with units include the units in the suffix
        #------------------------------------------------------
        if self.__units_ is not None and self.__units_ != Units.UNITLESS:
            suffix += [str(self.__units_)]

        #------------------------------------------------------
        # Objects with derivatives include a list of the names
        #------------------------------------------------------
        if self.__derivs_:
            keys = list(self.__derivs_.keys())
            keys.sort()
            for key in keys:
                suffix += ['d_d' + key]

        #------------------------------------------------------
        # Generate the value string
        #------------------------------------------------------
        if np.isscalar(self.__values_):
            if is_masked:
                string = '--'
            else:
                string = str(self.__values_)
        elif is_masked:
            string = str(self.mvals)[1:-1]
        else:
            string = str(self.__values_)[1:-1]

        #------------------------------------------------------
        # Add an extra set of brackets around derivatives
        #------------------------------------------------------
        if self.__denom_ != ():
            string = '[' + string + ']'

        #------------------------------------------------------
        # Concatenate the results
        #------------------------------------------------------
        if len(suffix) == 0:
            suffix = ''
        else:
            suffix = '; ' + ', '.join(suffix)

        return type(self).__name__ + '(' + string + suffix + ')'
    #===========================================================================


    
    ############################################################################
    # Numerator slicing operations
    ############################################################################

    #===========================================================================
    # extract_numer
    #===========================================================================
    def extract_numer(self, axis, index, classes=(), recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        An object extracted from one numerator axis.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Position axis from left
        #------------------------------------------------------
        if axis >= 0:
            a1 = axis
        else:
            a1 = axis + self.__nrank_
        if a1 < 0 or a1 >= self.__nrank_:
            raise ValueError('axis is out of range (%d,%d): %d',
                             (-self.__nrank_, self.__nrank_, axis))
        k1 = len(self.__shape_) + a1

        #------------------------------------------------------
        # Roll this axis to the beginning and slice it out
        #------------------------------------------------------
        new_values = np.rollaxis(self.__values_, k1, 0)
        new_values = new_values[index]

        #------------------------------------------------------
        # Construct and cast
        #------------------------------------------------------
        obj = Qube(new_values, self.__mask_, nrank=self.__nrank_ - 1,
                   example=self)
        obj = obj.cast(classes)
        obj.__readonly_ = self.__readonly_

        #------------------------------------------------------
        # Slice the derivatives if necessary
        #------------------------------------------------------
        if recursive:
            for (key, deriv) in self.__derivs_.items():
                obj.insert_deriv(key, deriv.extract_numer(a1, index, classes,
                                                          False))

        return obj
    #===========================================================================



    #===========================================================================
    # slice_numer
    #===========================================================================
    def slice_numer(self, axis, index1, index2, classes=(), recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        An object sliced from one numerator axis.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Position axis from left
        #------------------------------------------------------
        if axis >= 0:
            a1 = axis
        else:
            a1 = axis + self.__nrank_
        if a1 < 0 or a1 >= self.__nrank_:
            raise ValueError('axis is out of range (%d,%d): %d',
                             (-self.__nrank_, self.__nrank_, axis))
        k1 = len(self.__shape_) + a1

        #------------------------------------------------------
        # Roll this axis to the beginning and slice it out
        #------------------------------------------------------
        new_values = np.rollaxis(self.__values_, k1, 0)
        new_values = new_values[index1:index2]
        new_values = np.rollaxis(new_values, 0, k1+1)

        # Construct and cast
        obj = Qube(new_values, self.__mask_, example=self)
        obj = obj.cast(classes)
        obj.__readonly_ = self.__readonly_

        #------------------------------------------------------
        # Slice the derivatives if necessary
        #------------------------------------------------------
        if recursive:
            for (key, deriv) in self.__derivs_.items():
                obj.insert_deriv(key, deriv.slice_numer(a1, index1, index2,
                                                        classes, False))

        return obj
    #===========================================================================



    ############################################################################
    # Numerator shaping operations
    ############################################################################

    #===========================================================================
    # transpose_numer
    #===========================================================================
    def transpose_numer(self, axis1=0, axis2=1, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A copy of this object with two numerator axes transposed.

        Inputs:
            axis1       the first axis to transpose from among the numerator
                        axes. Negative values count backward from the last
                        numerator axis.
            axis2       the second axis to transpose.
            recursive   True to transpose the same axes of the derivatives;
                        False to return an object without derivatives.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        len_shape = len(self.__shape_)

        #------------------------------------------------------
        # Position axis1 from left
        #------------------------------------------------------
        if axis1 >= 0:
            a1 = axis1
        else:
            a1 = axis1 + self.__nrank_
        if a1 < 0 or a1 >= self.__nrank_:
            raise ValueError('first axis is out of range (%d,%d): %d',
                             (-self.__nrank_, self.__nrank_, axis1))
        k1 = len_shape + a1

        #------------------------------------------------------
        # Position axis2 from item left
        #------------------------------------------------------
        if axis2 >= 0:
            a2 = axis2
        else:
            a2 = axis2 + self.__nrank_
        if a2 < 0 or a2 >= self.__nrank_:
            raise ValueError('second axis out of range (%d,%d): %d',
                             (-self.__nrank_, self.__nrank_, axis2))
        k2 = len_shape + a2

        #------------------------------------------------------
        # Swap the axes
        #------------------------------------------------------
        new_values = np.swapaxes(self.__values_, k1, k2)

        #------------------------------------------------------
        # Construct the result
        #------------------------------------------------------
        obj = Qube.__new__(type(self))
        obj.__init__(new_values, self.__mask_, example=self)
        obj.__readonly_ = self.__readonly_

        if recursive:
            for (key, deriv) in self.__derivs_.items():
                obj.insert_deriv(key, deriv.transpose_numer(a1, a2, False))

        return obj
    #===========================================================================



    #===========================================================================
    # reshape_numer
    #===========================================================================
    def reshape_numer(self, shape, classes=(), recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        This object with a new shape for numerator items.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Validate the shape
        #------------------------------------------------------
        shape = tuple(shape)
        if self.nsize != int(np.prod(shape)):
            raise ValueError('item size must be unchanged: %s, %s' %
                             (str(self.__numer_), str(shape)))

        #------------------------------------------------------
        # Reshape
        #------------------------------------------------------
        full_shape = self.__shape_ + shape + self.__denom_
        new_values = self.__values_.reshape(full_shape)

        #------------------------------------------------------
        # Construct and cast
        #------------------------------------------------------
        obj = Qube(new_values, self.__mask_, nrank=len(shape), example=self)
        obj = obj.cast(classes)
        obj.__readonly_ = self.__readonly_

        #------------------------------------------------------
        # Reshape the derivatives if necessary
        #------------------------------------------------------
        if recursive:
          for (key, deriv) in self.__derivs_.items():
            obj.insert_deriv(key, deriv.reshape_numer(shape, classes, False))

        return obj
    #===========================================================================



    #===========================================================================
    # flatten_numer
    #===========================================================================
    def flatten_numer(self, classes=(), recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        This object with a new numerator shape such that nrank == 1.

        Input:
            classes     a single class or list or tuple of classes. The class
                        of the object returned will be the first suitable class
                        in the list. Otherwise, a generic Qube object will be
                        returned.
            recursive   True to include matching slices of the derivatives in
                        the returned object; otherwise, the returned object will
                        not contain derivatives.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return self.reshape_numer((self.nsize,), classes, recursive)
    #===========================================================================



    ############################################################################
    # Denominator shaping operations
    ############################################################################

    #===========================================================================
    # transpose_denom
    #===========================================================================
    def transpose_denom(self, axis1=0, axis2=1):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A copy of this object with two denominator axes transposed.

        Inputs:
            axis1       the first axis to transpose from among the denominator
                        axes. Negative values count backward from the last
                        axis.
            axis2       the second axis to transpose.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        len_shape = len(self.__shape_)

        #------------------------------------------------------
        # Position axis1 from left
        #------------------------------------------------------
        if axis1 >= 0:
            a1 = axis1
        else:
            a1 = axis1 + self.__drank_
        if a1 < 0 or a1 >= self.__drank_:
            raise ValueError('first axis is out of range (%d,%d): %d',
                             (-self.__drank_, self.__drank_, axis1))
        k1 = len_shape + self.__nrank_ + a1

        #------------------------------------------------------
        # Position axis2 from item left
        #------------------------------------------------------
        if axis2 >= 0:
            a2 = axis2
        else:
            a2 = axis2 + self.__drank_
        if a2 < 0 or a2 >= self.__drank_:
            raise ValueError('second axis out of range (%d,%d): %d',
                             (-self.__drank_, self.__drank_, axis2))
        k2 = len_shape + self.__nrank_ + a2

        #------------------------------------------------------
        # Swap the axes
        #------------------------------------------------------
        new_values = np.swapaxes(self.__values_, k1, k2)

        #------------------------------------------------------
        # Construct the result
        #------------------------------------------------------
        obj = Qube.__new__(type(self))
        obj.__init__(new_values, self.__mask_, example=self)
        obj.__readonly_ = self.__readonly_

        return obj
    #===========================================================================



    #===========================================================================
    # reshape_denom
    #===========================================================================
    def reshape_denom(self, shape):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        This object with a new shape for denominator items.

        Input:
            shape       the new denominator shape.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Validate the shape
        #------------------------------------------------------
        shape = tuple(shape)
        if self.dsize != int(np.prod(shape)):
            raise ValueError('denominator size must be unchanged: %s, %s' %
                             (str(self.__denom_), str(shape)))

        #------------------------------------------------------
        # Reshape
        #------------------------------------------------------
        full_shape = self.__shape_ + self.__numer_ + shape
        new_values = self.__values_.reshape(full_shape)

        #------------------------------------------------------
        # Construct and cast
        #------------------------------------------------------
        obj = Qube.__new__(type(self))
        obj.__init__(new_values, self.__mask_, drank=len(shape), example=self)
        obj.__readonly_ = self.__readonly_

        return obj
    #===========================================================================



    #===========================================================================
    # flatten_denom
    #===========================================================================
    def flatten_denom(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        This object with a new denominator shape such that drank == 1.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return self.reshape_denom((self.dsize,))
    #===========================================================================



    ############################################################################
    # Numerator/denominator operations
    ############################################################################

    #===========================================================================
    # join_items
    #===========================================================================
    def join_items(self, classes):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        The object with denominator axes joined to the numerator.

        Derivatives are removed.

        Input:
            classes     either a single subclass of Qube or a list or tuple of
                        subclasses. The returned object will be an instance of
                        the first suitable subclass in the list.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if not self.__drank_:
            return self.wod

        obj = Qube(self.__values_, self.__mask_,
                   nrank=(self.__nrank_ + self.__drank_), drank=0,
                   example=self)
        obj = obj.cast(classes)
        obj.__readonly_ = self.__readonly_

        return obj
    #===========================================================================



    #===========================================================================
    # split_items
    #===========================================================================
    def split_items(self, nrank, classes):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        The object with numerator axes converted to denominator axes.

        Derivatives are removed.

        Input:
            classes     either a single subclass of Qube or a list or tuple of
                        subclasses. The returned object will be an instance of
                        the first suitable subclass in the list.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        obj = Qube(self.__values_, self.__mask_,
                   nrank=nrank, drank=(self.__rank_ - nrank),
                   example=self)
        obj = obj.cast(classes)
        obj.__readonly_ = self.__readonly_

        return obj
    #===========================================================================



    #===========================================================================
    # swap_items
    #===========================================================================
    def swap_items(self, classes):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A new object with the numerator and denominator axes exchanged.

        Derivatives are removed.

        Input:
            classes     either a single subclass of Qube or a list or tuple of
                        subclasses. The returned object will be an instance of
                        the first suitable subclass in the list.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        new_values = self.__values_
        len_shape = new_values.ndim

        for r in range(self.__nrank_):
            new_values = np.rollaxis(new_values, -self.__drank_ - 1, len_shape)

        obj = Qube(new_values, self.__mask_,
                   nrank=self.__drank_, drank=self.__nrank_, example=self)
        obj = obj.cast(classes)
        obj.__readonly_ = self.__readonly_

        return obj
    #===========================================================================



    #===========================================================================
    # chain
    #===========================================================================
    def chain(self, arg):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Chain multiplication of this derivative by another.

        Returns the denominator of the first object times the numerator of the
        second argument. The result will be an instance of the same class. This
        operation is never recursive.

        Inputs:
            arg         the right-hand term in the chain multiplication.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        left = self.flatten_denom().join_items(Qube)
        right = arg.flatten_numer(Qube)

        return Qube.dot(left, right, -1, 0, type(self), False)
    #===========================================================================



    ############################################################################
    # Unary operators
    ############################################################################

    #===========================================================================
    # __pos__
    #===========================================================================
    def __pos__(self):
        return self.clone(recursive=True)
    #===========================================================================



    #===========================================================================
    # __neg__
    #===========================================================================
    def __neg__(self, recursive=True):

        #------------------------------------------------------
        # Construct a copy with negative values
        #------------------------------------------------------
        obj = self.clone(recursive=False)
        obj._set_values_(-self.__values_)

        #------------------------------------------------------
        # Fill in the negative derivatives
        #------------------------------------------------------
        if recursive and self.__derivs_:
            for (key, deriv) in self.__derivs_.items():
                obj.insert_deriv(key, -deriv)

        return obj
    #===========================================================================



    #===========================================================================
    # __abs__
    #===========================================================================
    def __abs__(self, recursive=True):

        #------------------------------------------------------
        # Check rank
        #------------------------------------------------------
        if self.__nrank_ != 0:
            Qube._raise_unsupported_op('abs()', self)

        #------------------------------------------------------
        # Construct a copy with absolute values
        #------------------------------------------------------
        obj = self.clone(recursive=False)
        obj._set_values_(np.abs(self.__values_))

        #------------------------------------------------------
        # Fill in the derivatives, multiplied by sign(self)
        #------------------------------------------------------
        if recursive and self.__derivs_:
            sign = self.wod.sign()
            for (key, deriv) in self.__derivs_.items():
                obj.insert_deriv(key, deriv * sign)

        return obj
    #===========================================================================



    ############################################################################
    # Addition
    ############################################################################

    #===========================================================================
    # __add__
    #===========================================================================
    def __add__(self, arg, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Default method for left addition, element by element.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Convert arg to another Qube if necessary
        #------------------------------------------------------
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = self.as_this_type(arg, coerce=False)
            except:
                Qube._raise_unsupported_op('+', self, original_arg)

        #------------------------------------------------------
        # Verify compatibility
        #------------------------------------------------------
        if not Units.can_match(self.__units_, arg.__units_):
            raise ValueError('operand units are incompatible: %s, %s' %
                             (str(self.__units_), str(arg.__units_)))

        if self.__numer_ != arg.__numer_:
            if type(self) != type(arg):
                Qube._raise_unsupported_op('+', self, original_arg)

            raise ValueError('item shapes are incompatible: %s, %s' %
                             (str(self.__numer_), str(arg.__numer_)))

        if self.__denom_ != arg.__denom_:
            raise ValueError('denominator shapes are incompatible: %s, %s' %
                             (str(self.__denom_), str(arg.__denom_)))

        #------------------------------------------------------
        # Construct the result
        #------------------------------------------------------
        obj = Qube.__new__(type(self))
        obj.__init__(self.__values_ + arg.__values_,
                     self.__mask_ | arg.__mask_,
                     units = self.__units_ or arg.__units_,
                     example=self)

        if recursive:
            obj.insert_derivs(obj._add_derivs(self, arg))

        return obj
    #===========================================================================



    #===========================================================================
    # __radd__
    #===========================================================================
    def __radd__(self, arg, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Default method for right addition, element by element.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        return self.__add__(arg, recursive)
    #===========================================================================



    #===========================================================================
    # __iadd__
    #===========================================================================
    def __iadd__(self, arg):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Default method for in-place addition, element by element.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.require_writable()

        #------------------------------------------------------
        # Convert arg to another Qube if necessary
        #------------------------------------------------------
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = self.as_this_type(arg, coerce=False)
            except:
                Qube._raise_unsupported_op('+=', self, original_arg)

        #------------------------------------------------------
        # Verify compatibility
        #------------------------------------------------------
        if not Units.can_match(self.__units_, arg.__units_):
            raise ValueError('operand units are incompatible: %s, %s' %
                             (str(self.__units_), str(arg.__units_)))

        if self.__numer_ != arg.__numer_:
            if type(self) != type(arg):
                Qube._raise_unsupported_op('+=', self, original_arg)

            raise ValueError('item shapes are incompatible: %s, %s' %
                             (str(self.__numer_), str(arg.__numer_)))

        if self.__denom_ != arg.__denom_:
            raise ValueError('denominator shapes are incompatible: %s, %s' %
                             (str(self.__denom_), str(arg.__denom_)))

        #------------------------------------------------------
        # Perform the operation
        #------------------------------------------------------
        if self.is_int() and not arg.is_int():
            raise TypeError('"+=" operation returns non-integer result')

        new_derivs = self._add_derivs(self,arg) # if this raises exception, stop
        self.__values_ += arg.__values_         # on exception, no harm done
        self.__mask_ = self.__mask_ | arg.__mask_
        self.__units_ = self.__units_ or arg.__units_
        self.insert_derivs(new_derivs)

        self.__cache_.clear()
        return self
    #===========================================================================



    #===========================================================================
    # _add_derivs
    #===========================================================================
    def _add_derivs(self, arg1, arg2):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Dictionary of added derivatives.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        set1 = set(arg1.__derivs_.keys())
        set2 = set(arg2.__derivs_.keys())
        set12 = set1 & set2
        set1 -= set12
        set2 -= set12

        new_derivs = {}
        for key in set12:
            new_derivs[key] = arg1.__derivs_[key] + arg2.__derivs_[key]
        for key in set1:
            new_derivs[key] = arg1.__derivs_[key]
        for key in set2:
            new_derivs[key] = arg2.__derivs_[key]

        return new_derivs
    #===========================================================================



    ############################################################################
    # Subtraction
    ############################################################################

    #===========================================================================
    # __sub__
    #===========================================================================
    def __sub__(self, arg, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Default method for left subtraction, element by element.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Convert arg to the same subclass and try again
        #------------------------------------------------------
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = self.as_this_type(arg, coerce=False)
            except:
                Qube._raise_unsupported_op('-', self, original_arg)

        #------------------------------------------------------
        # Verify compatibility
        #------------------------------------------------------
        if not Units.can_match(self.__units_, arg.__units_):
            raise ValueError('operand units are incompatible: %s, %s' %
                             (str(self.__units_), str(arg.__units_)))

        if self.__numer_ != arg.__numer_:
            if type(self) != type(arg):
                Qube._raise_unsupported_op('-', self, original_arg)

            raise ValueError('item shapes are incompatible: %s, %s' %
                             (str(self.__numer_), str(arg.__numer_)))

        if self.__denom_ != arg.__denom_:
            raise ValueError('denominator shapes are incompatible: %s, %s' %
                             (str(self.__denom_), str(arg.__denom_)))

        #------------------------------------------------------
        # Construct the result
        #------------------------------------------------------
        obj = Qube.__new__(type(self))
        obj.__init__(self.__values_ - arg.__values_,
                     self.__mask_ | arg.__mask_,
                     units = self.__units_ or arg.__units_,
                     example=self)

        if recursive:
            obj.insert_derivs(obj._sub_derivs(self, arg))

        return obj
    #===========================================================================



    #===========================================================================
    # __rsub__
    #===========================================================================
    def __rsub__(self, arg, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Default method for right subtraction, element by element.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Convert arg to the same subclass and try again
        #------------------------------------------------------
        if not isinstance(arg, Qube):
            arg = self.as_this_type(arg, coerce=False)
            return arg.__sub__(self, recursive)
    #===========================================================================



    #===========================================================================
    # __isub__
    #===========================================================================
    def __isub__(self, arg):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        In-place subtraction.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.require_writable()

        #------------------------------------------------------
        # Convert arg to another Qube if necessary
        #------------------------------------------------------
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = self.as_this_type(arg, coerce=False)
            except Exception as e:
                Qube._raise_unsupported_op('-=', self, original_arg)

        #------------------------------------------------------
        # Verify compatibility
        #------------------------------------------------------
        if not Units.can_match(self.__units_, arg.__units_):
            raise ValueError('operand units are incompatible: %s, %s' %
                             (str(self.__units_), str(arg.__units_)))

        if self.__numer_ != arg.__numer_:
            if type(self) != type(arg):
                Qube._raise_unsupported_op('-=', self, original_arg)

            raise ValueError('item shapes are incompatible: %s, %s' %
                             (str(self.__numer_), str(arg.__numer_)))

        if self.__denom_ != arg.__denom_:
            raise ValueError('denominator shapes are incompatible: %s, %s' %
                             (str(self.__denom_), str(arg.__denom_)))

        #------------------------------------------------------
        # Perform the operation
        #------------------------------------------------------
        if self.is_int() and not arg.is_int():
            raise TypeError('"-=" operation returns non-integer result')

        new_derivs = self._sub_derivs(self,arg) # if this raises exception, stop
        self.__values_ -= arg.__values_         # on exception, no harm done
        self.__mask_ = self.__mask_ | arg.__mask_
        self.__units_ = self.__units_ or arg.__units_
        self.insert_derivs(new_derivs)

        self.__cache_.clear()
        return self
    #===========================================================================



    #===========================================================================
    # _sub_derivs
    #===========================================================================
    def _sub_derivs(self, arg1, arg2):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Dictionary of subtracted derivatives.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        set1 = set(arg1.__derivs_.keys())
        set2 = set(arg2.__derivs_.keys())
        set12 = set1 & set2
        set1 -= set12
        set2 -= set12

        new_derivs = {}
        for key in set12:
            new_derivs[key] = arg1.__derivs_[key] - arg2.__derivs_[key]
        for key in set1:
            new_derivs[key] = arg1.__derivs_[key]
        for key in set2:
            new_derivs[key] = -arg2.__derivs_[key]

        return new_derivs
    #===========================================================================



    ############################################################################
    # Multiplication
    ############################################################################

    #===========================================================================
    # __mul__
    #===========================================================================
    def __mul__(self, arg, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Generic left multiplication.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Handle multiplication by a number
        #------------------------------------------------------
        if Qube.is_real_number(arg):
            return self._mul_by_number(arg, recursive)

        #------------------------------------------------------
        # Convert arg to a Scalar if necessary
        #------------------------------------------------------
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = Qube.SCALAR_CLASS.as_scalar(arg)
            except:
                Qube._raise_unsupported_op('*', self, original_arg)

        #------------------------------------------------------
        # Check denominators
        #------------------------------------------------------
        if self.__drank_ and arg.__drank_:
            raise ValueError("dual operand denominators for '*': %s, %s" %
                             (str(self.__denom_), str(arg.__denom_)))

        #------------------------------------------------------
        # Multiply by scalar...
        #------------------------------------------------------
        if arg.__nrank_ == 0:
            try:
                return self._mul_by_scalar(arg, recursive)
 
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Revise the exception if the arg was modified
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            except:
                if arg is not original_arg:
                    Qube._raise_unsupported_op('*', self, original_arg)
                raise

        #------------------------------------------------------
        # Swap and try again
        #------------------------------------------------------
        if self.__nrank_ == 0:
            return arg._mul_by_scalar(self, recursive)

        #------------------------------------------------------
        # Multiply by matrix...
        #------------------------------------------------------
        if self.__nrank_ == 2 and arg.__nrank_ in (1,2):
            return Qube.dot(self, arg, -1, 0, (type(arg), type(self)),
                            recursive)

        #------------------------------------------------------
        # Give up
        #------------------------------------------------------
        Qube._raise_unsupported_op('*', self, original_arg)
    #===========================================================================



    #===========================================================================
    # __rmul__
    #===========================================================================
    def __rmul__(self, arg, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Generic right multiplication.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Handle multiplication by a number
        #------------------------------------------------------
        if Qube.is_real_number(arg):
            return self._mul_by_number(arg, recursive)

        #------------------------------------------------------
        # Convert arg to a Scalar and try again
        #------------------------------------------------------
        original_arg = arg
        try:
            arg = Qube.SCALAR_CLASS.as_scalar(arg)
            return self._mul_by_scalar(arg, recursive)

        #------------------------------------------------------
        # Revise the exception if the arg was modified
        #------------------------------------------------------
        except:
            if arg is not original_arg:
                Qube._raise_unsupported_op('*', original_arg, self)
            raise
    #===========================================================================



    #===========================================================================
    # __imul__
    #===========================================================================
    def __imul__(self, arg):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Generic in-place multiplication.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.require_writable()

        #------------------------------------------------------
        # Convert arg to a Scalar if necessary
        #------------------------------------------------------
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = Qube.SCALAR_CLASS.as_scalar(arg)
            except:
                Qube._raise_unsupported_op('*=', self, original_arg)

        #------------------------------------------------------
        # Scalar case
        #------------------------------------------------------
        if arg.__rank_ == 0:

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Align axes
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            arg_values = arg.__values_
            if self.__rank_ and np.shape(arg_values):
                arg_values = arg_values.reshape(np.shape(arg_values) +
                                                self.__rank_ * (1,))

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Multiply...
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if self.is_int() and not arg.is_int():
                raise TypeError('"*=" operation returns non-integer result')

            new_derivs = self._mul_derivs(arg)   # on exception, stop
            self.__values_ *= arg_values        # on exception, no harm done
            self.__mask_ = self.__mask_ | arg.__mask_
            self.__units_ = Units.mul_units(self.__units_, arg.__units_)
            self.insert_derivs(new_derivs)

            self.__cache_.clear()
            return self

        #------------------------------------------------------
        # Matrix multiply case
        #------------------------------------------------------
        if self.__nrank_ == 2 and arg.__nrank_ == 2 and arg.__drank_ == 0:
            result = Qube.dot(self, arg, -1, 0, type(self), recursive=True)
            self._set_values_(result.__values_, result.__mask_)
            self.insert_derivs(result.__derivs_)
            return self

        #------------------------------------------------------
        # return NotImplemented
        #------------------------------------------------------
        Qube._raise_unsupported_op('*=', self, original_arg)
    #===========================================================================



    #===========================================================================
    # _mul_by_number
    #===========================================================================
    def _mul_by_number(self, arg, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Internal multiply op when the arg is a Python scalar.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        obj = self.clone(recursive=False)
        obj._set_values_(self.__values_ * arg)

        if recursive and self.__derivs_:
            for (key, deriv) in self.__derivs_.items():
                obj.insert_deriv(key, deriv._mul_by_number(arg, False))

        return obj
    #===========================================================================



    #===========================================================================
    # _mul_by_scalar
    #===========================================================================
    def _mul_by_scalar(self, arg, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Internal multiply op when the arg is a Qube with nrank == 0 and no
        more than one object has a denominator.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Align axes
        #------------------------------------------------------
        self_values = self.__values_
        self_shape = np.shape(self_values)
        if arg.__drank_ > 0 and self_shape != ():
            self_values = self_values.reshape(self_shape + arg.__drank_ * (1,))

        arg_values = arg.__values_
        arg_shape = (arg.__shape_ + self.__rank_ * (1,) + arg.__denom_)
        if np.shape(arg_values) not in ((), arg_shape):
            arg_values = arg_values.reshape(arg_shape)

        #------------------------------------------------------
        # Construct object
        #------------------------------------------------------
        obj = Qube.__new__(type(self))
        obj.__init__(self_values * arg_values,
                     self.__mask_ | arg.__mask_,
                     units = Units.mul_units(self.__units_, arg.__units_),
                     drank = max(self.__drank_, arg.__drank_),
                     example = self)

        obj.insert_derivs(self._mul_derivs(arg))

        return obj
    #===========================================================================



    #===========================================================================
    # _mul_derivs
    #===========================================================================
    def _mul_derivs(self, arg):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Dictionary of multiplied derivatives.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        new_derivs = {}
        new_shape = self.__shape_

        if self.__derivs_:
            arg_wod = arg.wod
            for (key, self_deriv) in self.__derivs_.items():
                new_derivs[key] = self_deriv * arg_wod

        if arg.__derivs_:
            self_wod = self.wod
            for (key, arg_deriv) in arg.__derivs_.items():
                if key in new_derivs:
                    new_derivs[key] = new_derivs[key] + self_wod * arg_deriv
                else:
                    new_derivs[key] = self_wod * arg_deriv

        return new_derivs
    #===========================================================================



    ############################################################################
    # Division
    ############################################################################

    #===========================================================================
    # __div__
    #===========================================================================
    def __div__(self, arg, recursive=True):
        return self.__truediv__(arg, recursive)
    #===========================================================================



    #===========================================================================
    # __rdiv__
    #===========================================================================
    def __rdiv__(self, arg, recursive=True):
        return self.__rtruediv__(arg)
    #===========================================================================



    #===========================================================================
    # __idiv__
    #===========================================================================
    def __idiv__(self, arg, recursive=True):
        return self.__itruediv__(arg)
    #===========================================================================



    #===========================================================================
    # __truediv__
    #===========================================================================
    def __truediv__(self, arg, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Generic left true division: Cases of divide-by-zero are masked.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Handle division by a number
        #------------------------------------------------------
        if Qube.is_real_number(arg):
            return self._div_by_number(arg, recursive)

        #------------------------------------------------------
        # Convert arg to a Scalar if necessary
        #------------------------------------------------------
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = Qube.SCALAR_CLASS.as_scalar(arg)
            except:
                Qube._raise_unsupported_op('/', self, original_arg)

        #------------------------------------------------------
        # Check right denominator
        #------------------------------------------------------
        if arg.__drank_ > 0:
            raise ValueError("right operand denominator for '/': %s" %
                             str(arg.__denom_))

        #------------------------------------------------------
        # Divide by scalar...
        #------------------------------------------------------
        if arg.__nrank_ == 0:
            try:
                return self._div_by_scalar(arg, recursive)
 
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Revise the exception if the arg was modified
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            except:
                if arg is not original_arg:
                    Qube._raise_unsupported_op('/', self, original_arg)
                raise

        #------------------------------------------------------
        # Swap and multiply by reciprocal...
        #------------------------------------------------------
        if self.__nrank_ == 0:
            return self.reciprocal(recursive)._mul_by_scalar(arg, recursive)

        #------------------------------------------------------
        # Matrix / matrix is multiply by inverse matrix
        #------------------------------------------------------
        if self.__rank_ == 2 and arg.__rank_ == 2:
            return self.__mul__(arg.reciprocal(recursive))

        #------------------------------------------------------
        # Give up
        #------------------------------------------------------
        Qube._raise_unsupported_op('/', self, original_arg)
    #===========================================================================



    #===========================================================================
    # __rtruediv__
    #===========================================================================
    def __rtruediv__(self, arg, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Generic right division.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Handle right division by a number
        #------------------------------------------------------
        if Qube.is_real_number(arg):
            return self.reciprocal(recursive).__mul__(arg, recursive)

        #------------------------------------------------------
        # Convert arg to a Scalar and try again
        #------------------------------------------------------
        original_arg = arg
        try:
            arg = Qube.SCALAR_CLASS.as_scalar(arg)
            return arg.__truediv__(self, recursive)

        #------------------------------------------------------
        # Revise the exception if the arg was modified
        #------------------------------------------------------
        except Exception as e:
            raise
            if arg is not original_arg:
                Qube._raise_unsupported_op('/', original_arg, self)
            raise
    #===========================================================================



    #===========================================================================
    # Generic in-place division
    #===========================================================================
    def __itruediv__(self, arg):

        if not self.is_float():
            raise TypeError('"/=" operation returns non-integer result')

        self.require_writable()

        #------------------------------------------------------
        # Convert arg to a Scalar if necessary
        #------------------------------------------------------
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = Qube.SCALAR_CLASS.as_scalar(arg)
            except:
                Qube._raise_unsupported_op('/=', self, original_arg)

        #------------------------------------------------------
        # In-place multiply by the reciprocal
        #------------------------------------------------------
        try:
            return self.__imul__(arg.reciprocal())

        #------------------------------------------------------
        # Revise the exception if the arg was modified
        #------------------------------------------------------
        except:
            if arg is not original_arg:
                Qube._raise_unsupported_op('/=', self, original_arg)
            raise
    #===========================================================================



    #===========================================================================
    # _div_by_number
    #===========================================================================
    def _div_by_number(self, arg, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Internal division op when the arg is a Python scalar.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        obj = self.clone(recursive=False)

        #------------------------------------------------------
        # Mask out zeros
        #------------------------------------------------------
        if arg == 0:
            obj._set_mask_(True)
        else:
            obj._set_values_(self.__values_ / arg)

        if recursive and self.__derivs_:
            for (key, deriv) in self.__derivs_.items():
                obj.insert_deriv(key, deriv._div_by_number(arg, False))

        return obj
    #===========================================================================



    #===========================================================================
    # _div_by_scalar
    #===========================================================================
    def _div_by_scalar(self, arg, recursive):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Internal division op when the arg is a Qube with rank == 0.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Mask out zeros
        #------------------------------------------------------
        arg = arg.mask_where_eq(0.,1.)

        #------------------------------------------------------
        # Align axes
        #------------------------------------------------------
        arg_values = arg.__values_
        if np.shape(arg_values) and self.__rank_:
            arg_values = arg_values.reshape(arg.shape + self.__rank_ * (1,))

        #------------------------------------------------------
        # Construct object
        #------------------------------------------------------
        obj = Qube.__new__(type(self))
        obj.__init__(self.__values_ / arg_values,
                     self.__mask_ | arg.__mask_,
                     units = Units.div_units(self.__units_, arg.__units_),
                     example = self)
        if recursive:
            obj.insert_derivs(self._div_derivs(arg, nozeros=True))

        return obj
    #===========================================================================



    #===========================================================================
    # _div_derivs
    #===========================================================================
    def _div_derivs(self, arg, nozeros=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Dictionary of divided derivatives.

        if nozeros is True, the arg is assumed not to contain any zeros, so
        divide-by-zero errors are not checked.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        new_derivs = {}

        if not self.__derivs_ and not arg.__derivs_:
            return new_derivs
        if not nozeros:
            arg = arg.mask_where_eq(0., 1.)

        arg_wod_inv = arg.wod.reciprocal(nozeros=True)

        for (key, self_deriv) in self.__derivs_.items():
            new_derivs[key] = self_deriv * arg_wod_inv

        if arg.__derivs_:
            self_wod = self.wod
            for (key, arg_deriv) in arg.__derivs_.items():
                term = self_wod * (arg_deriv * arg_wod_inv**2)
                if key in new_derivs:
                    new_derivs[key] -= term
                else:
                    new_derivs[key] = -term

        return new_derivs
    #===========================================================================



    ############################################################################
    # Floor Division (with no support for derivatives)
    ############################################################################

    #===========================================================================
    # __floordiv__
    #===========================================================================
    def __floordiv__(self, arg):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Generic left floor division: Cases of divide-by-zero become masked. 
        Derivatives are ignored.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Convert arg to a Scalar if necessary
        #------------------------------------------------------
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = Qube.SCALAR_CLASS.as_scalar(arg)
            except:
                Qube._raise_unsupported_op('//', self, original_arg)

        #------------------------------------------------------
        # Check right denominator
        #------------------------------------------------------
        if arg.__drank_ > 0:
            raise ValueError("right operand denominator for '//': %s" %
                             str(arg.__denom_))

        #------------------------------------------------------
        # Floor divide by scalar...
        #------------------------------------------------------
        if arg.__nrank_ == 0:
            try:
                return self._floordiv_by_scalar(arg)

            # Revise the exception if the arg was modified
            except:
                if arg is not original_arg:
                    Qube._raise_unsupported_op('//', original_arg, self)
                raise

        #------------------------------------------------------
        # Give up
        #------------------------------------------------------
        Qube._raise_unsupported_op('//', self, original_arg)
    #===========================================================================



    #===========================================================================
    # __rfloordiv__
    #===========================================================================
    def __rfloordiv__(self, arg):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Generic right floor division.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Convert arg to a Scalar and try again
        #------------------------------------------------------
        original_arg = arg
        try:
            arg = Qube.SCALAR_CLASS.as_scalar(arg)
            return arg.__floordiv__(self)

        #------------------------------------------------------
        # Revise the exception if the arg was modified
        #------------------------------------------------------
        except:
            if arg is not original_arg:
                Qube._raise_unsupported_op('//', original_arg, self)
            raise
    #===========================================================================



    #===========================================================================
    # __ifloordiv__
    #===========================================================================
    def __ifloordiv__(self, arg):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Generic in-place floor division.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.require_writable()

        #------------------------------------------------------
        # Convert arg to a Scalar if necessary
        #------------------------------------------------------
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = Qube.SCALAR_CLASS.as_scalar(arg)
            except:
                Qube._raise_unsupported_op('//=', self, original_arg)

        #------------------------------------------------------
        # Handle floor division by a scalar
        #------------------------------------------------------
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

            self.__cache_.clear()
            return self

        #------------------------------------------------------
        # Nothing else is implemented
        # return NotImplemented
        #------------------------------------------------------
        Qube._raise_unsupported_op('//=', self, original_arg)
    #===========================================================================



    #===========================================================================
    # _floordiv_by_number
    #===========================================================================
    def _floordiv_by_number(self, arg):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Internal floor division op when the arg is a Python scalar.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        obj = self.clone(recursive=False)

        if arg == 0:
            obj._set_mask_(True)
        else:
            obj._set_values_(self.__values_ // arg)

        return obj
    #===========================================================================



    #===========================================================================
    # _floordiv_by_scalar
    #===========================================================================
    def _floordiv_by_scalar(self, arg):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Internal floor division op when the arg is a Qube with nrank == 0.

        The arg cannot have a denominator.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Mask out zeros
        #------------------------------------------------------
        arg = arg.mask_where_eq(0,1)

        #------------------------------------------------------
        # Align axes
        #------------------------------------------------------
        arg_values = arg.__values_
        if np.shape(arg_values) and self.__rank_:
            arg_values = arg_values.reshape(arg.shape + self.__rank_ * (1,))

        #------------------------------------------------------
        # Construct object
        #------------------------------------------------------
        obj = Qube.__new__(type(self))
        obj.__init__(self.__values_ // arg_values,
                     self.__mask_ | arg.__mask_,
                     units = Units.div_units(self.__units_, arg.__units_),
                     example = self)
        return obj
    #===========================================================================



    ############################################################################
    # Modulus operators (with no support for derivatives)
    ############################################################################

    #===========================================================================
    # __mod__
    #===========================================================================
    def __mod__(self, arg, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Generic left modulus: Cases of divide-by-zero become masked. 
        Derivatives in the numerator are supported, but not in the 
        denominator.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Handle modulus by a number
        #------------------------------------------------------
        if Qube.is_real_number(arg):
            return self._mod_by_number(arg, recursive)

        #------------------------------------------------------
        # Convert arg to a Scalar if necessary
        #------------------------------------------------------
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = Qube.SCALAR_CLASS.as_scalar(arg)
            except:
                Qube._raise_unsupported_op('%', self, original_arg)

        #------------------------------------------------------
        # Check right denominator
        #------------------------------------------------------
        if arg.__drank_ > 0:
            raise ValueError("right operand denominator for '%': %s" %
                             str(arg.__denom_))

        #------------------------------------------------------
        # Modulus by scalar...
        #------------------------------------------------------
        if arg.__nrank_ == 0:
            try:
                return self._mod_by_scalar(arg, recursive)

            # Revise the exception if the arg was modified
            except:
                if arg is not original_arg:
                    Qube._raise_unsupported_op('%', self, original_arg)
                raise

        #------------------------------------------------------
        # Give up
        #------------------------------------------------------
        Qube._raise_unsupported_op('%', self, original_arg)
    #===========================================================================



    #===========================================================================
    # __rmod__
    #===========================================================================
    # Generic right modulus
    def __rmod__(self, arg):

        #------------------------------------------------------
        # Convert arg to a Scalar and try again
        #------------------------------------------------------
        original_arg = arg
        try:
            arg = Qube.SCALAR_CLASS.as_scalar(arg)
            return arg.__mod__(self)

        #------------------------------------------------------
        # Revise the exception if the arg was modified
        #------------------------------------------------------
        except:
            if arg is not original_arg:
                Qube._raise_unsupported_op('%', original_arg, self)
            raise
    #===========================================================================



    #===========================================================================
    # __imod__
    #===========================================================================
    # Generic in-place modulus
    def __imod__(self, arg):
        self.require_writable()

        #------------------------------------------------------
        # Convert arg to a Scalar if necessary
        #------------------------------------------------------
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = Qube.SCALAR_CLASS.as_scalar(arg)
            except:
                Qube._raise_unsupported_op('%=', self, original_arg)

        #------------------------------------------------------
        # Handle modulus by a scalar
        #------------------------------------------------------
        if arg.__rank_ == 0:
            divisor = arg.mask_where_eq(0, 1)
            div_values = divisor.__values_

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Align axes
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if self.__rank_:
                div_values = np.reshape(div_values, np.shape(div_values) +
                                                    self.__rank_ * (1,))
            self.__values_ %= div_values
            self.__mask_ = self.__mask_ | divisor.__mask_
            self.__units_ = Units.div_units(self.__units_, arg.__units_)

            self.__cache_.clear()
            return self

        #------------------------------------------------------
        # Nothing else is implemented
        # return NotImplemented
        #------------------------------------------------------
        Qube._raise_unsupported_op('%=', self, original_arg)
    #===========================================================================



    #===========================================================================
    # _mod_by_number
    #===========================================================================
    def _mod_by_number(self, arg, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Internal modulus op when the arg is a Python scalar.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        obj = self.clone(recursive=False)

        #------------------------------------------------------
        # Mask out zeros
        #------------------------------------------------------
        if arg == 0:
            obj._set_mask_(True)
        else:
            obj._set_values_(self.__values_ % arg)

        if recursive and self.__derivs_:
            for (key, deriv) in self.__derivs_.items():
                obj.insert_deriv(key, deriv)

        return obj
    #===========================================================================



    #===========================================================================
    # _mod_by_scalar
    #===========================================================================
    def _mod_by_scalar(self, arg, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Internal modulus op when the arg is a Qube with rank == 0.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Mask out zeros
        #------------------------------------------------------
        arg = arg.wod.mask_where_eq(0,1)

        #------------------------------------------------------
        # Align axes
        #------------------------------------------------------
        arg_values = arg.__values_
        if np.shape(arg_values) and self.__rank_:
            arg_values = arg_values.reshape(arg.shape + self.__rank_ * (1,))

        #------------------------------------------------------
        # Construct the object
        #------------------------------------------------------
        obj = Qube.__new__(type(self))
        obj.__init__(self.__values_ % arg_values,
                     self.__mask_ | arg.__mask_,
                     units = Units.div_units(self.__units_, arg.__units_),
                     example = self)

        if recursive and self.__derivs_:
            for (key, deriv) in self.__derivs_.items():
                obj.insert_deriv(key, deriv.broadcast_into_shape(obj.__shape_))

        return obj
    #===========================================================================



    ############################################################################
    # Exponentiation operators
    ############################################################################

    #===========================================================================
    # __pow__
    #===========================================================================
    # Generic exponentiation, PolyMath scalar to a single scalar power
    def __pow__(self, expo, recursive=True):

        if self.__rank_ != 0:
            Qube._raise_unsupported_op('**', self, expo)

        #----------------------------------------------------------
        # Interpret the exponent and mask if any
        #----------------------------------------------------------
        if isinstance(expo, Qube):
            if expo.__rank_:
                raise ValueError('exponent must be scalar')
            Units.require_unitless(expo.__units_)

            if expo.__derivs_:
                raise ValueError('derivatives in exponents are not supported')

        else:
            expo = Qube.SCALAR_CLASS(expo)

        #----------------------------------------------------------
        # 0-D case
        #----------------------------------------------------------
        if not self.__shape_ and not expo.__shape_:
            try:
                new_values = self.__values_ ** expo.__values_
            except (ValueError, ZeroDivisionError):
                return self.masked_single(recursive)

            if not isinstance(new_values, numbers.Real):
                return self.masked_single(recursive)

            new_mask = False
            new_units = Units.units_power(self.__units_, expo.__values_)

        #----------------------------------------------------------
        # Array case
        #----------------------------------------------------------
        else:

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Without this step, negative int exponents on int values truncate
            # to 0.
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if expo.is_int():
                if expo.__shape_:
                    if (expo < 0).any():
                        expo = expo.as_float()
                elif expo < 0:
                    expo = expo.as_float()

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Plow forward with the results blindly, then mask nan and inf.
            # Zero to a negative power creates a RuntTimeWarning, which needs to
            # be suppressed.
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new_values = self.__values_ ** expo.__values_

            new_mask = (self.__mask_
                        | expo.__mask_
                        | np.isnan(new_values)
                        | np.isinf(new_values))
            new_values[new_mask] = 1

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Check units and exponent
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if Units.is_unitless(self.__units_):
                new_units = None
            elif np.isscalar(expo.__values_):
                new_units = Units.units_power(self.__units_, expo.__values_)
            else:
                all_expos = np.broadcast_to(expo.__values_, new_values.shape)
                all_expos = all_expos[np.logical_not(new_mask)]
                all_expos = all_expos[all_expos != 0]
                if all_expos.size:
                  if np.any(all_expos != all_expos[0]):
                    all_expos = np.unique(all_expos)
                    all_expos = (list(all_expos[all_expos > 0]) +
                                 list(all_expos[all_expos < 0]))
                    raise ValueError('incompatible units after ' +
                                     'exponentiation: %s, %s' % (
                            Units.units_power(self.__units_, all_expos[0]),
                            Units.units_power(self.__units_, all_expos[1])))
                  else:
                    new_units = Units.units_power(self.__units_, all_expos[0])
                else:
                    new_units = None

        obj = Qube.__new__(type(self))
        obj.__init__(new_values, new_mask, units=new_units, example=self)

        #----------------------------------------------------------
        # Evaluate the derivatives if necessary
        #----------------------------------------------------------
        if recursive and self.__derivs_:
            factor = expo * self.__pow__(expo-1, recursive=False)
            for (key, deriv) in self.__derivs_.items():
                obj.insert_deriv(key, factor * deriv)

        return obj
    #===========================================================================



    ############################################################################
    # Comparison operators, returning boolean scalars or Booleans
    ############################################################################

    #===========================================================================
    # __eq__
    #===========================================================================
    def __eq__(self, arg):

        #----------------------------------------------------------
        # If the subclasses cannot be unified, raise a ValueError
        #----------------------------------------------------------
        if not isinstance(arg, type(self)):
            try:
                obj = Qube.__new__(type(self))
                obj.__init__(arg, example=self)
                arg = obj
            except:
                return False

        #----------------------------------------------------------
        # Compare the values and masks
        #----------------------------------------------------------
        if self.__item_ != arg.__item_:
            return False

        try:
            (self, arg) = Qube.broadcast(self, arg)
        except ValueError:
            return False

        compare = (self.__values_ == arg.__values_)
        for r in range(self.__rank_):
            compare = np.all(compare, axis=-1)

        both_masked = (self.__mask_ & arg.__mask_)
        one_masked  = (self.__mask_ ^ arg.__mask_)

        #----------------------------------------------------------
        # Compare units for compatibility
        #----------------------------------------------------------
        if not Units.can_match(self.__units_, arg.__units_):
            one_masked = True

        #----------------------------------------------------------
        # Return a Python bool if the shape is ()
        #----------------------------------------------------------
        if np.isscalar(compare):
            if one_masked:
                return False
            if both_masked:
                return True
            return bool(compare)

        #----------------------------------------------------------
        # Apply the mask
        #----------------------------------------------------------
        if np.isscalar(one_masked):
            if one_masked:
                compare.fill(False)
            if both_masked:
                compare.fill(True)
        else:
            compare[one_masked] = False
            compare[both_masked] = True

        result = Qube.BOOLEAN_CLASS(compare)
        result.__truth_if_all_ = True
        return result
    #===========================================================================



    #===========================================================================
    # __ne__
    #===========================================================================
    def __ne__(self, arg):

        #----------------------------------------------------------
        # If the subclasses cannot be unified, raise a ValueError
        #----------------------------------------------------------
        if not isinstance(arg, type(self)):
            try:
                obj = Qube.__new__(type(self))
                obj.__init__(arg, example=self)
                arg = obj
            except:
                return True

        #----------------------------------------------------------
        # Compare the values and masks
        #----------------------------------------------------------
        if self.__item_ != arg.__item_:
            return True

        try:
            (self, arg) = Qube.broadcast(self, arg)
        except ValueError:
            return True

        compare = (self.__values_ != arg.__values_)
        for r in range(self.__rank_):
            compare = np.any(compare, axis=-1)

        both_masked = (self.__mask_ & arg.__mask_)
        one_masked = (self.__mask_ ^ arg.__mask_)

        #----------------------------------------------------------
        # Compare units for compatibility
        #----------------------------------------------------------
        if not Units.can_match(self.__units_, arg.__units_):
            one_masked = True

        #----------------------------------------------------------
        # Return a Python bool if the shape is ()
        #----------------------------------------------------------
        if np.isscalar(compare):
            if one_masked:
                return True
            if both_masked:
                return False
            return bool(compare)

        #----------------------------------------------------------
        # Apply the mask
        #----------------------------------------------------------
        if np.shape(one_masked):
            compare[one_masked] = True
            compare[both_masked] = False
        else:
            if one_masked:
                compare.fill(True)
            if both_masked:
                compare.fill(False)

        result = Qube.BOOLEAN_CLASS(compare)
        result.__truth_if_any_ = True
        return result
    #===========================================================================



    #===========================================================================
    # __lt__
    #===========================================================================
    def __lt__(self, arg):
        raise ValueError('comparison operators are not supported for class ' +
                         type(self).__name__)
    #===========================================================================



    #===========================================================================
    # __gt__
    #===========================================================================
    def __gt__(self, arg):
        raise ValueError('comparison operators are not supported for class ' +
                         type(self).__name__)
    #===========================================================================



    #===========================================================================
    # __le__
    #===========================================================================
    def __le__(self, arg):
        raise ValueError('comparison operators are not supported for class ' +
                         type(self).__name__)
    #===========================================================================



    #===========================================================================
    # __gt__
    #===========================================================================
    def __gt__(self, arg):
        raise ValueError('comparison operators are not supported for class ' +
                         type(self).__name__)
    #===========================================================================



    #===========================================================================
    # __bool__
    #===========================================================================
    def __bool__(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Supports 'if a == b: ...' and 'if a != b: ...' statements.

        Equality requires that every unmasked element of a and
        b be equal, and both object be masked at the same locations.

        Comparison of objects of shape () is also supported.

        Any other comparison of PolyMath object requires an explict call to
        all() or any().
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
    #===========================================================================



    #===========================================================================
    # __nonzero__
    #===========================================================================
    # Needed for backward compatibility with Python 2
    def __nonzero__(self):
        return self.__bool__()

    ############################################################################
    # Boolean operators
    ############################################################################

    #===========================================================================
    # __invert__
    #===========================================================================
    def __invert__(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        (~) operator
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self = Qube.BOOLEAN_CLASS.as_boolean(self)
        return Qube.BOOLEAN_CLASS(~self.__values_, self.__mask_)
    #===========================================================================



    #===========================================================================
    # __and__
    #===========================================================================
    def __and__(self, arg):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        (&) operator
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self = Qube.BOOLEAN_CLASS.as_boolean(self)
        arg  = Qube.BOOLEAN_CLASS.as_boolean(arg)

        return Qube.BOOLEAN_CLASS(self.__values_ & arg.__values_,
                                  self.__mask_ | arg.__mask_)
    #===========================================================================



    #===========================================================================
    # __rand__
    #===========================================================================
    def __rand__(self, arg):
        return self.__and__(arg)
    #===========================================================================



    #===========================================================================
    # __or__
    #===========================================================================
    def __or__(self, arg):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        (|) operator
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self = Qube.BOOLEAN_CLASS.as_boolean(self)
        arg  = Qube.BOOLEAN_CLASS.as_boolean(arg)

        return Qube.BOOLEAN_CLASS(self.__values_ | arg.__values_,
                                  self.__mask_ | arg.__mask_)
    #===========================================================================



    #===========================================================================
    # __ror__
    #===========================================================================
    def __ror__(self, arg):
        return self.__or__(arg)
    #===========================================================================



    #===========================================================================
    # __xor__
    #===========================================================================
    def __xor__(self, arg):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        (^) operator
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self = Qube.BOOLEAN_CLASS.as_boolean(self)
        arg  = Qube.BOOLEAN_CLASS.as_boolean(arg)

        return Qube.BOOLEAN_CLASS(self.__values_ ^ arg.__values_,
                                  self.__mask_ | arg.__mask_)
    #===========================================================================



    #===========================================================================
    # __rxor__
    #===========================================================================
    def __rxor__(self, arg):
        return self.__xor__(arg)
    #===========================================================================



    #===========================================================================
    # __iand__
    #===========================================================================
    def __iand__(self, arg):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        (&=) operator
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.require_writable()

        arg = Qube.BOOLEAN_CLASS.as_boolean(arg)
        self.__values_ &= arg.__values_
        self.__mask_ |= arg.__mask_

        return self
    #===========================================================================



    #===========================================================================
    # __ior__
    #===========================================================================
    def __ior__(self, arg):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        (!=) operator
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.require_writable()

        arg = Qube.BOOLEAN_CLASS.as_boolean(arg)
        self.__values_ |= arg.__values_
        self.__mask_ |= arg.__mask_

        return self
    #===========================================================================



    #===========================================================================
    # __ixor__
    #===========================================================================
    def __ixor__(self, arg):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        (^=) operator
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self.require_writable()

        arg = Qube.BOOLEAN_CLASS.as_boolean(arg)
        self.__values_ ^= arg.__values_
        self.__mask_ |= arg.__mask_

        return self
    #===========================================================================



    ############################################################################
    # Any and all
    ############################################################################

    #===========================================================================
    # any
    #===========================================================================
    def any(self, axis=None):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        True if any of the matching items are nonzero.

        NOTE: If Qube.PREFER_BUILTIN_TYPES is True and the result is a single
        scalar, it is returned as a Python bool value; otherwise it is returned
        as a Boolean.

        Input:
            axis        an integer axis or a tuple of axes. The any operation
                        is performed across these axes, leaving any remaining
                        axes in the returned value. If None (the default), then
                        the any operation is performed across all axes if the
                        object.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self = Qube.BOOLEAN_CLASS.as_boolean(self)

        if not self.__shape_:
            result = Qube.BOOLEAN_CLASS(self)   # make a copy

        elif not np.any(self.__mask_):
            result = Qube.BOOLEAN_CLASS(np.any(self.__values_, axis=axis),
                                        mask=False)

        elif axis is None:
            if np.isscalar(self.__mask_):
                result = Qube.BOOLEAN_CLASS(np.any(self.__values_),
                                            mask=self.__mask_)
            elif np.all(self.__mask_):
                result = Qube.BOOLEAN_CLASS(np.any(self.__values_), mask=True)
            else:
                result = Qube.BOOLEAN_CLASS(np.any(
                                            self.__values_[self.antimask]))

        else:
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Create new array
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if self.__values_.dtype == np.bool_:
                new_values = self.__values_.copy()
            else:
                new_values = self.__values_.astype(np.bool_)

            new_values[self.__mask_] = False
            new_values = np.any(new_values, axis=axis)

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Create new mask
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            new_mask = Qube.as_one_bool(self.__mask_)
            if np.shape(new_mask):
                new_mask = np.all(self.__mask_, axis=axis)

            result = Qube.BOOLEAN_CLASS(new_values, new_mask)

        #----------------------------------------------------------
        # Convert result to a Python bool if necessary
        #----------------------------------------------------------
        if Qube.PREFER_BUILTIN_TYPES:
            return result.as_builtin()

        return result
    #===========================================================================



    #===========================================================================
    # all
    #===========================================================================
    def all(self, axis=None):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        True if and only if all the matching items are nonzero.

        NOTE: If Qube.PREFER_BUILTIN_TYPES is True and the result is a single
        scalar, it is returned as a Python bool value; otherwise it is returned
        as a Boolean.

        Input:
            axis        an integer axis or a tuple of axes. The all operation
                        is performed across these axes, leaving any remaining
                        axes in the returned value. If None (the default), then
                        the all operation is performed across all axes if the
                        object.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self = Qube.BOOLEAN_CLASS.as_boolean(self)

        if not self.__shape_:
            result = Qube.BOOLEAN_CLASS(self)   # make a copy

        elif not np.any(self.__mask_):
            result = Qube.BOOLEAN_CLASS(np.all(self.__values_, axis=axis),
                                        mask=False)

        elif axis is None:
            if np.isscalar(self.__mask_):
                result = Qube.BOOLEAN_CLASS(np.all(self.__values_),
                                            mask=self.__mask_)
            elif np.all(self.__mask_):
                result = Qube.BOOLEAN_CLASS(np.all(self.__values_), mask=True)
            else:
                result = Qube.BOOLEAN_CLASS(np.all(
                                            self.__values_[self.antimask]),
                                            mask=False)

        else:
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Create new array
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if self.__values_.dtype == np.dtype(np.bool_):
                new_values = self.__values_.copy()
            else:
                new_values = self.__values_.astype(np.bool_)

            new_values[self.__mask_] = True
            new_values = np.all(new_values, axis=axis)

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Create new mask
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            new_mask = Qube.as_one_bool(self.__mask_)
            if np.shape(new_mask):
                new_mask = np.all(self.__mask_, axis=axis)

            result = Qube.BOOLEAN_CLASS(new_values, new_mask)

        #----------------------------------------------------------
        # Convert result to a Python bool if necessary
        #----------------------------------------------------------
        if Qube.PREFER_BUILTIN_TYPES:
            return result.as_builtin()

        return result
    #===========================================================================



    ############################################################################
    # "Indeterminate" operators, which assume masked values are indeterminate
    ############################################################################

    #===========================================================================
    # tvl_and
    #===========================================================================
    def tvl_and(self, *args):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        'And' operation where masked values are treated as indeterminate,
        following the rules of three-valued logic:
            - False and anything = False
            - True and True = True
            - True and Masked = Masked
        Note that this behavior is different from the "and" operator "&", where
        any operation involving a masked value returns a masked value.

        If multiple inputs are given, the returned values is the and operation
        across all the inputs.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #----------------------------------------------------------
        # Truth table...
        #           False       Masked      True
        # False     False       False       False
        # Masked    False       Masked      Masked
        # True      False       Masked      True
        #----------------------------------------------------------

        self = Qube.BOOLEAN_CLASS.as_boolean(self)
        is_true = self.__values_ & self.antimask
        is_not_false = self.__values_ | self.__mask_

        for arg in args:
            arg = Qube.BOOLEAN_CLASS.as_boolean(arg)
            is_true = is_true & arg.__values_ & arg.antimask
            is_not_false = is_not_false & (arg.__values_ | arg.__mask_)

        return Qube.BOOLEAN_CLASS(is_true, is_not_false &
                                           np.logical_not(is_true))
    #===========================================================================



    #===========================================================================
    # tvl_or
    #===========================================================================
    def tvl_or(self, *args):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        'Or' operation where masked values are treated as indeterminate,
        following the rules of three-valued logic:
            - True or anything = True
            - False or False = False
            - False or Masked = Masked
        Note that this behavior is different from the "or" operator "|", where
        any operation involving a masked value returns a masked value.

        If multiple inputs are given, the returned values is the or operation
        across all the inputs.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #----------------------------------------------------------
        # Truth table...
        #           False       Masked      True
        # False     False       Masked      True
        # Masked    Masked      Masked      True
        # True      True        True        True
        #----------------------------------------------------------

        self = Qube.BOOLEAN_CLASS.as_boolean(self)
        is_true = self.__values_ & self.antimask
        is_not_false = self.__values_ | self.__mask_

        for arg in args:
            arg = Qube.BOOLEAN_CLASS.as_boolean(arg)
            is_true = is_true | (arg.__values_ & arg.antimask)
            is_not_false = is_not_false | arg.__values_ | arg.__mask_

        return Qube.BOOLEAN_CLASS(is_true, is_not_false &
                                           np.logical_not(is_true))
    #===========================================================================



    #===========================================================================
    # tvl_any
    #===========================================================================
    def tvl_any(self, axis=None):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        'Any' operation where masked values are treated as indeterminate
        rather than being ignored. This follows the rules of three-valued
        logic:
            - True if any unmasked True value appears
            - False if and only if all the items are False and unmasked.
            - otherwise, Masked.

        If Qube.PREFER_BUILTIN_TYPES is True and the result is a single scalar,
        it is returned as a Python bool value; otherwise it is returned as a
        Boolean.

        Input:
            axis        an integer axis or a tuple of axes. The any operation
                        is performed across these axes, leaving any remaining
                        axes in the returned value. If None (the default), then
                        the any operation is performed across all axes if the
                        object.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self = Qube.BOOLEAN_CLASS.as_boolean(self)

        if not self.__shape_:
            result = Qube.BOOLEAN_CLASS(self)   # make a copy

        elif not np.any(self.__mask_):
            result = Qube.BOOLEAN_CLASS(np.any(self.__values_, axis=axis),
                                        mask=False)

        elif axis is None:
            if np.isscalar(self.__mask_):
                result = Qube.BOOLEAN_CLASS(np.any(self.__values_),
                                            mask=self.__mask_)
            elif np.all(self.__mask_):
                result = Qube.BOOLEAN_CLASS(np.any(self.__values_), mask=True)
            else:
                any_true = np.any(self.__values_ & self.antimask)
                any_true_or_masked = np.any(self.__values_ | self.__mask_)

                if any_true:
                    result = Qube.BOOLEAN_CLASS.TRUE
                elif any_true_or_masked:
                    result = Qube.BOOLEAN_CLASS.MASKED
                else:
                    result = Qube.BOOLEAN_CLASS.FALSE

        else:

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Create new array
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if self.__values_.dtype == np.dtype(np.bool_):
                new_values = self.__values_.copy()
            else:
                new_values = self.__values_.astype(np.bool_)

            new_values[self.__mask_] = True
            new_values = np.any(new_values, axis=axis)

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Create new mask
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            new_mask = Qube.as_one_bool(self.__mask_)
            if np.shape(new_mask):

                #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                # False values are masked if any value is True or masked
                #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                new_mask = np.any(self.__values_ | self.__mask_, axis=axis)

                #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                # True values are unmasked if any value is True and unmasked
                #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                true_is_unmasked = np.any(self.__values_ & self.antimask,
                                          axis=axis)
                new_mask[true_is_unmasked] = False

            result = Qube.BOOLEAN_CLASS(new_values, new_mask)

        #----------------------------------------------------------
        # Convert result to a Python bool if necessary
        #----------------------------------------------------------
        if Qube.PREFER_BUILTIN_TYPES:
            return result.as_builtin()

        return result
    #===========================================================================



    #===========================================================================
    # tvl_all
    #===========================================================================
    def tvl_all(self, axis=None):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        'All' operation where masked values are treated as indeterminate
        rather than being ignored. This follows the rules of three-valued
        logic:
            - True if and only if all the items are True and unmasked.
            - False if any unmasked value of False appears.
            - otherwise, Masked.

        If Qube.PREFER_BUILTIN_TYPES is True and the result is a single scalar,
        it is returned as a Python bool value; otherwise it is returned as a
        Boolean.

        Input:
            axis        an integer axis or a tuple of axes. The all operation
                        is performed across these axes, leaving any remaining
                        axes in the returned value. If None (the default), then
                        the all operation is performed across all axes if the
                        object.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self = Qube.BOOLEAN_CLASS.as_boolean(self)

        if not self.__shape_:
            result = Qube.BOOLEAN_CLASS(self)   # make a copy

        elif not np.any(self.__mask_):
            result = Qube.BOOLEAN_CLASS(np.all(self.__values_, axis=axis),
                                        mask=False)

        elif axis is None:
            if np.isscalar(self.__mask_):
                result = Qube.BOOLEAN_CLASS(np.all(self.__values_),
                                            mask=self.__mask_)
            elif np.all(self.__mask_):
                result = Qube.BOOLEAN_CLASS(np.all(self.__values_), mask=True)
            else:
                all_true = np.all(self.__values_ & self.antimask)
                all_true_or_masked = np.all(self.__values_ | self.__mask_)

                if all_true:
                    result = Qube.BOOLEAN_CLASS.TRUE
                elif all_true_or_masked:
                    result = Qube.BOOLEAN_CLASS.MASKED
                else:
                    result = Qube.BOOLEAN_CLASS.FALSE

        else:

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Create new array
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if self.__values_.dtype == np.dtype(np.bool_):
                new_values = self.__values_.copy()
            else:
                new_values = self.__values_.astype(np.bool_)

            new_values[self.__mask_] = False
            new_values = np.all(new_values, axis=axis)

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Create new mask
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            new_mask = Qube.as_one_bool(self.__mask_)
            if np.shape(new_mask):

                #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                # False values are masked if every value is True or masked
                #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                new_mask = np.all(self.__values_ | self.__mask_, axis=axis)

                #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                # True values are unmasked if every value is True and unmasked
                #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                true_is_unmasked = np.all(self.__values_ & self.antimask,
                                          axis=axis)
                new_mask[true_is_unmasked] = False

            result = Qube.BOOLEAN_CLASS(new_values, new_mask)

        #----------------------------------------------------------
        # Convert result to a Python bool if necessary
        #----------------------------------------------------------
        if Qube.PREFER_BUILTIN_TYPES:
            return result.as_builtin()

        return result
    #===========================================================================



    #===========================================================================
    # tvl_eq
    #===========================================================================
    def tvl_eq(self, arg):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        '=' operator where masked values are treated as indeterminate. This
        follows the rules of three-valued logic. If either value is masked, the
        returned value is masked.
"""
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        try:
            arg_mask = arg.__mask_
        except AttributeError:
            arg_mask = False

        return (self == arg).mask_where(self.__mask_ | arg_mask)
    #===========================================================================



    #===========================================================================
    # tvl_ne
    #===========================================================================
    def tvl_ne(self, arg):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        '!=' operator where masked values are treated as indeterminate. This
        follows the rules of three-valued logic. If either value is masked, the
        returned value is masked.
"""
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        try:
            arg_mask = arg.__mask_
        except AttributeError:
            arg_mask = False

        return (self != arg).mask_where(self.__mask_ | arg_mask)
    #===========================================================================



    ############################################################################
    # Special operators
    ############################################################################

    #===========================================================================
    # reciprocal
    #===========================================================================
    def reciprocal(self, recursive=True, nozeros=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        An object equivalent to the reciprocal of this object.

        This must be overridden by other subclasses.

        Input:
            recursive   True to return the derivatives of the reciprocal too;
                        otherwise, derivatives are removed.
            nozeros     False (the default) to mask out any zero-valued items in
                        this object prior to the divide. Set to True only if you
                        know in advance that this object has no zero-valued
                        items.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        Qube._raise_unsupported_op('reciprocal()', self)
    #===========================================================================



    #===========================================================================
    # zero
    #===========================================================================
    def zero(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        An object of this subclass containing all zeros.

        The returned object has the same denominator shape as this object.

        This is default behavior and may need to be overridden by some
        subclasses.
"""
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #----------------------------------------------------------
        # Scalar case
        #----------------------------------------------------------
        if not self.__rank_:
            if self.is_float():
                new_value = 0.
            else:
                new_value = 0

        #----------------------------------------------------------
        # Array case
        #----------------------------------------------------------
        else:
            if self.is_float():
                new_value = np.zeros(self.__item_, dtype=np.float_)
            else:
                new_value = np.zeros(self.__item_, dtype=np.int_)

        #----------------------------------------------------------
        # Construct the object
        #----------------------------------------------------------
        obj = Qube.__new__(type(self))
        obj.__init__(new_value, False, example=self)

        #----------------------------------------------------------
        # Return it as readonly
        #----------------------------------------------------------
        return obj.as_readonly()
    #===========================================================================



    #===========================================================================
    # identity
    #===========================================================================
    def identity(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        An object of this subclass equivalent to the identity.

        This must be overridden by other subclasses.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        Qube._raise_unsupported_op('identity()', self)
    #===========================================================================



    ############################################################################
    # Indexing operators
    ############################################################################

    #===========================================================================
    # __len__
    #===========================================================================
    def __len__(self):
        if len(self.__shape_) > 0:
            return self.__shape_[0]
        else:
            raise TypeError('len of unsized object')
    #===========================================================================



    #===========================================================================
    # __getitem__
    #===========================================================================
    def __getitem__(self, indx):

        #----------------------------------------------------------
        # A single value can only be indexed with True or False
        #----------------------------------------------------------
        if not self.__shape_:
            if isinstance(indx, tuple) and len(indx) == 1:
                indx = indx[0]
            if Qube.is_one_true(indx):
                return self
            if Qube.is_one_false(indx):
                return self.as_all_masked()

            raise IndexError('too many indices')

        #----------------------------------------------------------
        # Interpret and adapt the index
        #----------------------------------------------------------
        (pre_index, post_mask, has_ellipsis,
         moved_to_front, array_shape, first_array_loc) = self._prep_index(indx)

        #----------------------------------------------------------
        # Apply index to values
        #----------------------------------------------------------
        if has_ellipsis and self.__rank_:
            vals_index = pre_index + self.__rank_ * (slice(None),)
        else:
            vals_index = pre_index
        result_values = self.__values_[vals_index]

        # Make sure we have not indexed into the item
        result_vals_shape = np.shape(result_values)
        if len(result_vals_shape) < self.__rank_:
            raise IndexError('too many indices')

        #----------------------------------------------------------
        # Apply index to mask
        #----------------------------------------------------------
        if self.__rank_:
            result_shape = result_vals_shape[:-self.__rank_]
        else:
            result_shape = result_vals_shape

        if not np.any(post_mask):           # post-mask is False
            if np.shape(self.__mask_):          # self-mask is array
                result_mask = self.__mask_[pre_index]
            else:                               # self-mask is True or False
                result_mask = post_mask or self.__mask_
        elif np.all(post_mask):             # post-mask is True
            result_mask = True
        else:                               # post-mask is array
            if np.shape(self.__mask_):          # self-mask is array
                result_mask = self.__mask_[pre_index].copy()
                result_mask[post_mask] = True
            elif self.__mask_:                  # self-mask is True
                result_mask = True
            else:                               # self-mask is False
                if post_mask.shape == result_shape:
                    result_mask = post_mask.copy()
                else:
                    result_mask = np.zeros(result_shape, dtype=np.bool_)
                    axes = len(result_shape) - post_mask.ndim
                    new_shape = post_mask.shape + axes * (1,)
                    mask = post_mask.reshape(new_shape)
                    result_mask[...] = mask

        #----------------------------------------------------------
        # Relocate the axes indexed by arrays if necessary
        #----------------------------------------------------------
        if moved_to_front:
            before = np.arange(len(array_shape))
            after = before + first_array_loc
            result_values = np.moveaxis(result_values, tuple(before),
                                                       tuple(after))
            if np.shape(result_mask):
                result_mask = np.moveaxis(result_mask, tuple(before),
                                                       tuple(after))

        #----------------------------------------------------------
        # Construct the object
        #----------------------------------------------------------
        obj = Qube.__new__(type(self))
        obj.__init__(result_values, result_mask, example=self)
        obj.__readonly_ = self.__readonly_

        #----------------------------------------------------------
        # Apply the same indexing to any derivatives
        #----------------------------------------------------------
        for (key, deriv) in self.__derivs_.items():
            obj.insert_deriv(key, deriv[indx])

        return obj
    #===========================================================================



    #===========================================================================
    # __setitem__
    #===========================================================================
    def __setitem__(self, indx, arg):
        self.require_writable()

        #----------------------------------------------------------
        # Handle shapeless objects and a single index of True or False
        #----------------------------------------------------------
        test_index = None
        if isinstance(indx, (bool, np.bool_)):
            test_index = bool(indx)
        elif (isinstance(indx, tuple)
              and len(indx) == 1
              and isinstance(indx[0], (bool, np.bool_))):
            test_index = bool(indx[0])

        if test_index is not None:
            if not test_index:          # immediate return on False
                return

            if not self.__shape_:       # immediate copy on True
                arg = self.as_this_type(arg)
                self.__values_ = arg.__values_
                self.__mask_   = arg.__mask_
                self.__cache_.clear()
                return

        #----------------------------------------------------------
        # No other index is allowed for a shapeless object
        #----------------------------------------------------------
        if not self.__shape_:
            raise IndexError('too many indices')

        #----------------------------------------------------------
        # Interpret the index
        #----------------------------------------------------------
        (pre_index, post_mask, has_ellipsis,
         moved_to_front, array_shape, first_array_loc) = self._prep_index(indx)

        #----------------------------------------------------------
        # If index is fully masked, we're done
        #----------------------------------------------------------
        if np.all(post_mask):
            return

        #----------------------------------------------------------
        # Convert the argument to this type
        #----------------------------------------------------------
        arg = self.as_this_type(arg, recursive=True)

        #----------------------------------------------------------
        # Derivatives must match
        #----------------------------------------------------------
        for key in self.__derivs_:
            if key not in arg.__derivs_:
                raise ValueError('missing derivative d_d%s in replacement' %
                                 key)

        #----------------------------------------------------------
        # Create the values index
        #----------------------------------------------------------
        if has_ellipsis and self.__rank_:
            vals_index = pre_index + self.__rank_ * (slice(None),)
        else:
            vals_index = pre_index

        #----------------------------------------------------------
        # Convert this mask to an array if necessary
        #----------------------------------------------------------
        if (np.isscalar(self.__mask_)       # in this special case, the mask
            and np.isscalar(arg.__mask_)    # is fine as is
            and self.__mask_ == arg.__mask_):
                pass

        elif np.isscalar(self.__mask_):
            if self.__mask_:
                self.__mask_ = np.ones(self.__shape_, dtype=np.bool_)
            else:
                self.__mask_ = np.zeros(self.__shape_, dtype=np.bool_)

        #------------------------------------------------------------------
        # Create a view of the arg with array-indexed axes moved to front
        #------------------------------------------------------------------
        if moved_to_front:
            rank = len(array_shape)
            arg_rank = len(arg.__shape_)
            if first_array_loc > arg_rank:
                moved_to_front = False      # arg rank does not reach array loc
            if first_array_loc + rank > arg_rank:
                after = np.arange(first_array_loc, arg_rank)
                before = tuple(after - first_array_loc)
                after = tuple(after)
            else:
                before = np.arange(rank)
                after = tuple(before + first_array_loc)
                before = tuple(before)

        if moved_to_front:
            arg_values = np.moveaxis(arg.__values_, after, before)
            if np.shape(arg.__mask_):
                arg_mask = np.moveaxis(arg.__mask_, after, before)
        else:
            arg_values = arg.__values_
            arg_mask = arg.__mask_

        #----------------------------------------------------------
        # Set the new values and mask
        #----------------------------------------------------------
        if not np.any(post_mask):           # post-mask is False

            self.__values_[vals_index] = arg.__values_
            if np.shape(self.__mask_):
                self.__mask_[pre_index] = arg.__mask_

        else:                               # post-mask is an array

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # antimask is False wherever the index is masked
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            antimask = np.logical_not(post_mask)

            selection = self.__values_[vals_index]

            if np.shape(arg.__values_):
                selection[antimask] = arg.__values_[antimask]
            else:
                selection[antimask] = arg.__values_

            self.__values_[vals_index] = selection

            if np.shape(self.__mask_):
                selection = self.__mask_[pre_index]

                if np.shape(arg.__mask_):
                    selection[antimask] = arg.__mask_[antimask]
                else:
                    selection[antimask] = arg.__mask_

                self.__mask_[pre_index] = selection

        self.__cache_.clear()

        #----------------------------------------------------------
        # Also update the derivatives (ignoring those not in self)
        #----------------------------------------------------------
        for (key, self_deriv) in self.__derivs_.items():
            self.__derivs_[key][indx] = arg.__derivs_[key]

        return
    #===========================================================================



    #===========================================================================
    # _prep_index
    #===========================================================================
    def _prep_index(self, indx):
      #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      """
      Prepare the index for this object.

      Input:
          indx            index to prepare.

      Return: A tuple (mask_index, new_mask_index, has_ellipsis)
          pre_index       index to apply to array and mask first.
          post_mask       mask to apply after indexing.
          has_ellipsis    True if the index contains an ellipsis.
          moved_to_front  True for non-consecutive array indices after first,
                          which result in axis-reordering according to NumPy's
                          rules.
          array_shape     array shape resulting from all the array indices.
          first_array_loc place to relocate the axes associated with an array,
                          if necessary.

      The index is represented by a single object or a tuple of objects.
      Out-of-bounds integer index values are replaced by masked values.
      """
      #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      try:      # catch any error and convert it to an IndexError

        #------------------------------------------------------
        # Convert a non-tuple index to a tuple
        #------------------------------------------------------
        if not isinstance(indx, (tuple,list)):
            indx = (indx,)

        #------------------------------------------------------
        # Convert tuples to Qubes of NumPy arrays;
        # Convert Vectors to individual arrays;
        # Convert all other numeric and boolean items to Qube
        #------------------------------------------------------
        expanded = []
        for item in indx:
            if isinstance(item, (list,tuple)):
                ranks = np.array([len(np.shape(k)) for k in item])
                if np.any(ranks == 0):
                    expanded += [Qube(np.array(item))]
                else:
                    expanded += [Qube(np.array(x)) for x in item]

            elif isinstance(item, Qube) and item.is_int() and item.__rank_ > 0:
                (index_list, mask_vals) = item.as_index_and_mask(purge=False,
                                                                 masked=None)
                expanded += [Qube(index_list[0], mask_vals)]
                expanded += [Qube(k) for k in index_list[1:]]

            elif isinstance(item, (bool, np.bool_, numbers.Integral,
                                   np.ndarray)):
                expanded += [Qube(item)]
            else:
                expanded += [item]

        #-----------------------------------------------------------------------
        # At this point, every item in the index list is a slice, Ellipsis,
        # None, or a Qube subclass. Ever item will consume exactly one axis of
        # the object, except for multidimensional boolean arrays, ellipses, and
        # None.
        #-----------------------------------------------------------------------

        #-----------------------------------------------------------------------
        # Identify the axis of this object consumed by each index item.
        # Initially, treat an ellipsis as consuming zero axes.
        # One item in inlocs for every item in expanded.
        #-----------------------------------------------------------------------
        inlocs = []
        ellipsis_k = -1
        inloc = 0
        for k,item in enumerate(expanded):
            inlocs += [inloc]

            if item == Ellipsis:
                if ellipsis_k >= 0:
                    raise IndexError("an index can only have a single " +
                                     "ellipsis ('...')")
                ellipsis_k = k

            elif isinstance(item, Qube) and item.__shape_ and item.is_bool():
                inloc += len(item.__shape_)

            elif item is not None:
                inloc += 1

        #------------------------------------------------------
        # Each value in inlocs is the 
        #------------------------------------------------------
        has_ellipsis = ellipsis_k >= 0
        if has_ellipsis:        # if ellipsis found
            correction = len(self.__shape_) - inloc
            if correction < 0:
                raise IndexError('too many indices for array')
            for k in range(ellipsis_k + 1, len(inlocs)):
                inlocs[k] += correction

        #------------------------------------------------------
        # Process the components of the index tuple...
        #------------------------------------------------------
        pre_index = []          # Numpy index to apply to the object
        post_mask = False       # Mask to apply after indexing, to account for
                                # masked index values.

        #------------------------------------------------------
        # Keep track of additional info about array indices
        #------------------------------------------------------
        array_inlocs = []       # inloc of each array index
        array_lengths = []      # length of axis consumed
        array_shapes = []       # shape of object returned by each array index.

        for k,item in enumerate(expanded):
            inloc = inlocs[k]

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # None consumes to input axis
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if item is None:
                pre_index += [item]
                continue

            axis_length = self.__shape_[inloc]

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Handle Qube subclasses
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if isinstance(item, Qube):

              if item.is_float():
                    raise IndexError('floating-point indexing is not permitted')

              #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
              # A Boolean index collapses one or more axes down to one, where
              # the new number of elements is equal to the number of elements
              # True or masked. After the index is applied, the entries
              # corresponding to masked index values will be masked. If no
              # values are True or masked, the axis collapses down to size zero.
              #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
              if item.is_bool():

                #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                # Boolean array
                # Consumes one index for each array dimension; returns one axis
                # with length equal to the number of occurrences of True or
                # masked; masked items leave masked elements.
                #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                if item.__shape_:

                    #- - - - - - - - - - - - - - - - - - -
                    # Validate shape
                    #- - - - - - - - - - - - - - - - - - -
                    item_shape = item.__shape_
                    for k,item_length in enumerate(item_shape):
                      if self.__shape_[inloc + k] != item_length:
                        raise IndexError((
                            'boolean index did not match indexed array along ' +
                            'dimension %d; dimension is %d but corresponding ' +
                            'boolean dimension is %d') % (inloc + k,
                                self.__shape_[inloc + k], item_length))

                    #- - - - - - - - - - - - - - - - - - -
                    # Update index and mask
                    #- - - - - - - - - - - - - - - - - - -
                    index = item.__values_ | item.__mask_   # True or masked
                    pre_index += [index]

                    if np.shape(item.__mask_):      # mask is an array
                        post_mask = post_mask | item.__mask_[index]

                    elif item.__mask_:              # mask is True
                        post_mask = True

                    array_inlocs += [inloc]
                    array_lengths += list(item_shape)
                    array_shapes += [(np.sum(index),)]

                #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                # One boolean item
                #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                else:

                    #- - - - - - - - - - - - -
                    # One masked item
                    #- - - - - - - - - - - - -
                    if item.__mask_:
                        pre_index += [slice(0,1)]   # unit-sized axis
                        post_mask = True

                    #- - - - - - - - - - - - -
                    # One True item
                    #- - - - - - - - - - - - -
                    elif item.__values_:
                        pre_index += [slice(None)]

                    #- - - - - - - - - - - - -
                    # One False item
                    #- - - - - - - - - - - - -
                    else:
                        pre_index += [slice(0,0)]   # zero-sized axis

              #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
              # Scalar index behaves like a NumPy ndarray index, except masked
              # index values yield masked array values
              #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
              elif item.__rank_ == 0:

                #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                # Scalar array
                # Consumes one axis; returns the number of axes in this array;
                # masked items leave masked elements.
                #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                if item.__shape_:
                    index_vals = item.__values_
                    mask_vals = item.__mask_

                    #- - - - - - - - - - - - - - - - -
                    # Find indices out of bounds
                    #- - - - - - - - - - - - - - - - -
                    out_of_bounds_mask = ((index_vals >= axis_length) |
                                          (index_vals < -axis_length))
                    any_out_of_bounds = np.any(out_of_bounds_mask)
                    if any_out_of_bounds:
                        mask_vals = mask_vals | out_of_bounds_mask
                        any_masked = True
                    else:
                        any_masked = np.any(mask_vals)

                    #- - - - - - - - - - - - - - - - -
                    # Find an unused index value, if any
                    #- - - - - - - - - - - - - - - - -
                    index_vals = index_vals % axis_length
                    if np.shape(mask_vals):
                        antimask = np.logical_not(mask_vals)
                        unused_set = (set(range(axis_length)) -
                                      set(index_vals[antimask]))
                    elif mask_vals:
                        unused_set = ()
                    else:
                        unused_set = (set(range(axis_length))
                                      - set(index_vals.ravel()))

                    if unused_set:
                        unused_index_value = unused_set.pop()
                    else:
                        unused_index_value = -1     # -1 = no unused element

                    #- - - - - - - - - - - - - - - - -
                    # Apply mask to index; update masked values
                    #- - - - - - - - - - - - - - - - -
                    if any_masked:
                        index_vals = index_vals.copy()
                        index_vals[mask_vals] = unused_index_value

                    pre_index += [index_vals.astype(np.intp)]

                    if np.shape(mask_vals):         # mask is also an array
                        post_mask = post_mask | mask_vals
                    elif mask_vals:                 # index is fully masked
                        post_mask = True

                    array_inlocs += [inloc]
                    array_lengths += [axis_length]
                    array_shapes += [item.__shape_]

                #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                # One scalar item
                #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                else:

                    #- - - - - - - - - - - - - - - - -
                    # Compare to allowed range
                    #- - - - - - - - - - - - - - - - -
                    index_val = item.__values_
                    mask_val = item.__mask_

                    if not mask_val:
                        if index_val < 0:
                            index_val += axis_length

                        if index_val < 0 or index_val >= axis_length:
                            mask_val = True

                    #- - - - - - - - - - - - - - - - -
                    # One masked item
                    # Remove this axis and mark everything masked
                    #- - - - - - - - - - - - - - - - -
                    if mask_val:
                        pre_index += [0]    # use 0 on a masked axis
                        post_mask = True

                    #- - - - - - - - - - - - - - - - -
                    # One unmasked item
                    #- - - - - - - - - - - - - - - - -
                    else:
                        pre_index += [index_val % axis_length]

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Handle any other index element the NumPy way, with no masking
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            elif isinstance(item, (slice, type(Ellipsis))):
                pre_index += [item]

            else:
                raise IndexError('invalid index type: ' + str(type(item)))

        #------------------------------------------------------
        # Get the shape of the array indices
        #------------------------------------------------------
        array_shape = Qube.broadcasted_shape(*array_shapes)

        #------------------------------------------------------
        # According to NumPy indexing rules, if there are non-consecutive array
        # array indices, the array indices are moved to the front of the axis
        # order in the result!
        #------------------------------------------------------
        if array_inlocs:
            first_array_loc = array_inlocs[0]
            diffs = np.diff(array_inlocs)
            moved_to_front = np.any(diffs > 1) and first_array_loc > 0
        else:
            first_array_loc = 0
            moved_to_front = False

        #------------------------------------------------------
        # Simplify the post_mask if possible
        #------------------------------------------------------
        if not all(array_shape):        # mask doesn't matter if size is zero
            post_mask = False
        elif np.all(post_mask):
            post_mask = True

        return (tuple(pre_index), post_mask, has_ellipsis,
                moved_to_front, array_shape, first_array_loc)

      except Exception as e:
        raise IndexError(e)
    #===========================================================================



    ############################################################################
    # Utilities for arithmetic operations
    ############################################################################

    #===========================================================================
    # _raise_unsupported_op
    #===========================================================================
    @staticmethod
    def _raise_unsupported_op(op, obj1, obj2=None):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Raise a TypeError or ValueError for unsupported operations.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
    #===========================================================================



    ############################################################################
    # Support functions for more math operations
    #
    # These are static methods so as to avoid name conflicts with non-static
    # methods implemented for specific subclasses.
    ############################################################################

    #===========================================================================
    # _mean_or_sum
    #===========================================================================
    @staticmethod
    def _mean_or_sum(arg, axis=None, recursive=True, _combine_as_mean=False):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        The mean or sum of the unmasked values. Internal method.

        Input:
            arg         the object for which to calculate the mean or sum.
            axis        an integer axis or a tuple of axes. The mean is
                        determined across these axes, leaving any remaining axes
                        in the returned value. If None (the default), then the
                        mean is performed across all axes if the object.
            recursive   True to construct the mean of the derivatives.
            _combine_as_mean    True to combine as a mean; False to combine as a
                                sum.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if arg.size == 0 or np.all(arg.__mask_):
            return arg.masked_single()

        #------------------------------------------------------
        # Select the NumPy function
        #------------------------------------------------------
        if _combine_as_mean:
            func = np.mean
        else:
            func = np.sum

        #------------------------------------------------------
        # Create the new axis, which is valid regardless of items
        #------------------------------------------------------
        rank = len(arg.__shape_)
        if isinstance(axis, numbers.Integral):
            new_axis = axis % rank
        elif axis is None:
            new_axis = tuple(range(rank))
        else:
            new_axis = tuple(a % rank for a in axis)

        #------------------------------------------------------
        # If there's no mask, this is easy
        #------------------------------------------------------
        if not np.any(arg.__mask_):
            obj = Qube(func(arg.__values_, axis=new_axis), False,
                       example=arg)

        #------------------------------------------------------
        # If we are averaging over all axes, this is fairly easy
        #------------------------------------------------------
        elif axis is None:
            obj = Qube(func(arg.__values_[arg.antimask], axis=0), False,
                       example=arg)

        #------------------------------------------------------
        # At this point, we have handled the cases mask==True and mask==False,
        # so the mask must be an array. Also, there must be at least one
        # unmasked value.
        #------------------------------------------------------
        else:
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Set masked items to zero, then sum across axes
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            new_values = arg.__values_.copy()
            new_values[arg.__mask_] = 0
            new_values = np.sum(new_values, axis=new_axis)

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Count the numbers of unmasked items, summed across axes
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            count = np.sum(arg.antimask, axis=new_axis)

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Convert to a mask and a mean
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            new_mask = (count == 0)
            if _combine_as_mean:
                count_reshaped = count.reshape(count.shape + arg.__rank_ * (1,))
                new_values = new_values / np.maximum(count_reshaped, 1)

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Fill in masked values with the default
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if np.any(new_mask):
                new_values[(new_mask,) +
                           arg.__rank_ * (slice(None),)] = arg.__default_
            else:
                new_mask = False

            obj = Qube(new_values, new_mask, example=arg)

        #------------------------------------------------------
        # Cast to the proper class
        #------------------------------------------------------
        obj = obj.cast(type(arg))

        #------------------------------------------------------
        # Handle derivatives
        #------------------------------------------------------
        if recursive and arg.__derivs_:
            new_derivs = {}
            for (key, deriv) in arg.__derivs_.items():
                new_derivs[key] = Qube._mean_or_sum(deriv, axis,
                                            recursive=False,
                                            _combine_as_mean=_combine_as_mean)

            obj.insert_derivs(new_derivs)

        return obj
    #===========================================================================



    #===========================================================================
    # dot
    #===========================================================================
    @staticmethod
    def dot(arg1, arg2, axis1=-1, axis2=0, classes=(), recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        The dot product of two objects.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # At most one object can have a denominator.
        #------------------------------------------------------
        if arg1.__drank_ and arg2.__drank_:
            raise ValueError('at most one object in dot() can have a ' +
                             'denominator')

        #------------------------------------------------------
        # Position axis1 from left
        #------------------------------------------------------
        if axis1 >= 0:
            a1 = axis1
        else:
            a1 = axis1 + arg1.__nrank_
        if a1 < 0 or a1 >= arg1.__nrank_:
            raise ValueError('first axis is out of range (%d,%d): %d' %
                             (-arg1.__nrank_, arg1.__nrank_, axis1))
        k1 = a1 + len(arg1.__shape_)

        #------------------------------------------------------
        # Position axis2 from item left
        #------------------------------------------------------
        if axis2 >= 0:
            a2 = axis2
        else:
            a2 = axis2 + arg2.__nrank_
        if a2 < 0 or a2 >= arg2.__nrank_:
            raise ValueError('second axis out of range (%d,%d): %d' %
                             (-arg2.__nrank_, arg2.__nrank_, axis2))
        k2 = a2 + len(arg2.__shape_)

        #------------------------------------------------------
        # Confirm that the axis lengths are compatible
        #------------------------------------------------------
        if arg1.__numer_[a1] != arg2.__numer_[a2]:
            raise ValueError('axes have different lengths: %d, %d' %
                             (arg1.__numer_[a1], arg2.__numer_[a2]))

        #------------------------------------------------------
        # Re-shape the value arrays (shape, numer1, numer2, denom1, denom2)
        #------------------------------------------------------
        shape1 = (arg1.__shape_ + arg1.__numer_ + (arg2.__nrank_ - 1) * (1,) +
                  arg1.__denom_ + arg2.__drank_ * (1,))
        array1 = arg1.__values_.reshape(shape1)

        shape2 = (arg2.__shape_ + (arg1.__nrank_ - 1) * (1,) + arg2.__numer_ +
                  arg1.__drank_ * (1,) + arg2.__denom_)
        array2 = arg2.__values_.reshape(shape2)
        k2 += arg1.__nrank_ - 1

        #------------------------------------------------------
        # Roll both array axes to the right
        #------------------------------------------------------
        array1 = np.rollaxis(array1, k1, array1.ndim)
        array2 = np.rollaxis(array2, k2, array2.ndim)

        #------------------------------------------------------
        # Make arrays contiguous so sum will run faster
        #------------------------------------------------------
        array1 = np.ascontiguousarray(array1)
        array2 = np.ascontiguousarray(array2)

        #------------------------------------------------------
        # Construct the dot product
        #------------------------------------------------------
        new_values = np.sum(array1 * array2, axis=-1)

        #------------------------------------------------------
        # Construct the object and cast
        #------------------------------------------------------
        new_nrank = arg1.__nrank_ + arg2.__nrank_ - 2
        new_drank = arg1.__drank_ + arg2.__drank_

        obj = Qube(new_values,
                   arg1.__mask_ | arg2.__mask_,
                   units=Units.mul_units(arg1.__units_, arg2.__units_),
                   nrank=new_nrank, drank=new_drank, example=arg1)
        obj = obj.cast(classes)

        #------------------------------------------------------
        # Insert derivatives if necessary
        #------------------------------------------------------
        if recursive and (arg1.__derivs_ or arg2.__derivs_):
            new_derivs = {}

            if arg1.__derivs_:
                arg2_wod = arg2.wod
                for (key, arg1_deriv) in arg1.__derivs_.items():
                    new_derivs[key] = Qube.dot(arg1_deriv, arg2_wod, a1, a2,
                                               classes, recursive=False)

            if arg2.__derivs_:
                arg1_wod = arg1.wod
                for (key, arg2_deriv) in arg2.__derivs_.items():
                    term = Qube.dot(arg1_wod, arg2_deriv, a1, a2,
                                    classes, recursive=False)
                    if key in new_derivs:
                        new_derivs[key] += term
                    else:
                        new_derivs[key] = term

            obj.insert_derivs(new_derivs)

        return obj
    #===========================================================================



    #===========================================================================
    # norm
    #===========================================================================
    @staticmethod
    def norm(arg, axis=-1, classes=(), recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        The norm of an object along one axis.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if arg.__drank_ != 0:
            raise ValueError('norm() does not allow denominators')

        #------------------------------------------------------
        # Position axis from left
        #------------------------------------------------------
        if axis >= 0:
            a1 = axis
        else:
            a1 = axis + arg.__nrank_
        if a1 < 0 or a1 >= arg.__nrank_:
            raise ValueError('axis is out of range (%d,%d): %d' %
                             (-arg.__nrank_, arg.__nrank_, axis))
        k1 = a1 + len(arg.__shape_)

        #------------------------------------------------------
        # Evaluate the norm
        #------------------------------------------------------
        new_values = np.sqrt(np.sum(arg.__values_**2, axis=k1))

        #------------------------------------------------------
        # Construct the object and cast
        #------------------------------------------------------
        obj = Qube(new_values,
                   arg.__mask_,
                   nrank=arg.__nrank_-1, example=arg)
        obj = obj.cast(classes)

        #------------------------------------------------------
        # Insert derivatives if necessary
        #------------------------------------------------------
        if recursive and arg.__derivs_:
            factor = arg.wod / obj
            for (key, arg_deriv) in arg.__derivs_.items():
                obj.insert_deriv(key, Qube.dot(factor, arg_deriv, a1, a1,
                                               classes, recursive=False))

        return obj
    #===========================================================================



    #===========================================================================
    # norm_sq
    #===========================================================================
    @staticmethod
    def norm_sq(arg, axis=-1, classes=(), recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Square of the norm of an object along one axis.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if arg.__drank_ != 0:
            raise ValueError('norm_sq() does not allow denominators')

        #------------------------------------------------------
        # Position axis from left
        #------------------------------------------------------
        if axis >= 0:
            a1 = axis
        else:
            a1 = axis + arg.__nrank_
        if a1 < 0 or a1 >= arg.__nrank_:
            raise ValueError('axis is out of range (%d,%d): %d' %
                             (-arg.__nrank_, arg.__nrank_, axis))
        k1 = a1 + len(arg.__shape_)

        #------------------------------------------------------
        # Evaluate the norm
        #------------------------------------------------------
        new_values = np.sum(arg.__values_**2, axis=k1)

        #------------------------------------------------------
        # Construct the object and cast
        #------------------------------------------------------
        obj = Qube(new_values,
                   arg.__mask_,
                   units=Units.mul_units(arg.__units_, arg.__units_),
                   nrank=arg.__nrank_-1, example=arg)
        obj = obj.cast(classes)

        #------------------------------------------------------
        # Insert derivatives if necessary
        #------------------------------------------------------
        if recursive and arg.__derivs_:
            factor = 2.* arg.wod
            for (key, arg_deriv) in arg.__derivs_.items():
                obj.insert_deriv(key, Qube.dot(factor, arg_deriv, a1, a1,
                                               classes, recursive=False))

        return obj
    #===========================================================================



    #===========================================================================
    # cross
    #===========================================================================
    @staticmethod
    def cross(arg1, arg2, axis1=-1, axis2=0, classes=(), recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        The cross product of two objects.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # At most one object can have a denominator.
        #------------------------------------------------------
        if arg1.__drank_ and arg2.__drank_:
            raise ValueError('at most one object in cross() can have a ' +
                             'denominator')

        #------------------------------------------------------
        # Position axis1 from left
        #------------------------------------------------------
        if axis1 >= 0:
            a1 = axis1
        else:
            a1 = axis1 + arg1.__nrank_
        if a1 < 0 or a1 >= arg1.__nrank_:
            raise ValueError('first axis is out of range (%d,%d): %d' %
                             (-arg1.__nrank_, arg1.__nrank_, axis1))
        k1 = a1 + len(arg1.__shape_)

        #------------------------------------------------------
        # Position axis2 from item left
        #------------------------------------------------------
        if axis2 >= 0:
            a2 = axis2
        else:
            a2 = axis2 + arg2.__nrank_
        if a2 < 0 or a2 >= arg2.__nrank_:
            raise ValueError('second axis out of range (%d,%d): %d' %
                             (-arg2.__nrank_, arg2.__nrank_, axis2))
        k2 = a2 + len(arg2.__shape_)

        #------------------------------------------------------
        # Confirm that the axis lengths are compatible
        #------------------------------------------------------
        if ((arg1.__numer_[a1] != arg2.__numer_[a2]) or
            (arg1.__numer_[a1] not in (2,3))):
            raise ValueError('invalid axis length for cross product: %d, %d' %
                             (arg1.__numer_[a1], arg2.__numer_[a2]))

        #--------------------------------------------------------------------
        # Re-shape the value arrays (shape, numer1, numer2, denom1, denom2)
        #--------------------------------------------------------------------
        shape1 = (arg1.__shape_ + arg1.__numer_ + (arg2.__nrank_ - 1) * (1,) +
                  arg1.__denom_ + arg2.__drank_ * (1,))
        array1 = arg1.__values_.reshape(shape1)

        shape2 = (arg2.__shape_ + (arg1.__nrank_ - 1) * (1,) + arg2.__numer_ +
                  arg1.__drank_ * (1,) + arg2.__denom_)
        array2 = arg2.__values_.reshape(shape2)
        k2 += arg1.__nrank_ - 1

        #------------------------------------------------------
        # Roll both array axes to the right
        #------------------------------------------------------
        array1 = np.rollaxis(array1, k1, array1.ndim)
        array2 = np.rollaxis(array2, k2, array2.ndim)

        new_drank = arg1.__drank_ + arg2.__drank_

        #------------------------------------------------------
        # Construct the cross product values
        #------------------------------------------------------
        if arg1.__numer_[a1] == 3:
            new_values = Qube.cross_3x3(array1, array2)

            # Roll the new axis back to its position in arg1
            new_nrank = arg1.__nrank_ + arg2.__nrank_ - 1
            new_k1 = new_values.ndim - new_drank - new_nrank + a1
            new_values = np.rollaxis(new_values, -1, new_k1)

        else:
            new_values = Qube.cross_2x2(array1, array2)
            new_nrank = arg1.__nrank_ + arg2.__nrank_ - 2

        #------------------------------------------------------
        # Construct the object and cast
        #------------------------------------------------------
        obj = Qube(new_values,
                   arg1.__mask_ | arg2.__mask_,
                   units=Units.mul_units(arg1.__units_, arg2.__units_),
                   nrank=new_nrank, drank=new_drank, example=arg1)
        obj = obj.cast(classes)

        #------------------------------------------------------
        # Insert derivatives if necessary
        #------------------------------------------------------
        if recursive and (arg1.__derivs_ or arg2.__derivs_):
            new_derivs = {}

            if arg1.__derivs_:
              arg2_wod = arg2.wod
              for (key, arg1_deriv) in arg1.__derivs_.items():
                new_derivs[key] = Qube.cross(arg1_deriv, arg2_wod, a1, a2,
                                             classes, recursive=False)

            if arg2.__derivs_:
              arg1_wod = arg1.wod
              for (key, arg2_deriv) in arg2.__derivs_.items():
                term = Qube.cross(arg1_wod, arg2_deriv, a1, a2, classes, False)
                if key in new_derivs:
                    new_derivs[key] += term
                else:
                    new_derivs[key] = term

            obj.insert_derivs(new_derivs)

        return obj
    #===========================================================================



    #===========================================================================
    # cross_3x3
    #===========================================================================
    @staticmethod
    def cross_3x3(a,b):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Stand-alone method to return the cross product of two 3-vectors,
        represented as NumPy arrays.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        (a,b) = np.broadcast_arrays(a,b)
        assert a.shape[-1] == b.shape[-1] == 3

        new_values = np.empty(a.shape)
        new_values[...,0] = a[...,1] * b[...,2] - a[...,2] * b[...,1]
        new_values[...,1] = a[...,2] * b[...,0] - a[...,0] * b[...,2]
        new_values[...,2] = a[...,0] * b[...,1] - a[...,1] * b[...,0]

        return new_values
    #===========================================================================



    #===========================================================================
    # cross_2x2
    #===========================================================================
    @staticmethod
    def cross_2x2(a,b):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Stand-alone method to return the cross product of two 2-vectors,
        represented as NumPy arrays.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        (a,b) = np.broadcast_arrays(a,b)
        assert a.shape[-1] == b.shape[-1] == 2

        return a[...,0] * b[...,1] - a[...,1] * b[...,0]
    #===========================================================================



    #===========================================================================
    # outer
    #===========================================================================
    @staticmethod
    def outer(arg1, arg2, classes=(), recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        The outer product of two objects.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #----------------------------------------------------------------
        # At most one object can have a denominator. This is sufficient 
        # to track first derivatives
        #----------------------------------------------------------------
        if arg1.__drank_ and arg2.__drank_:
            raise ValueError('at most one object in outer() can have a ' +
                             'denominator')

        #-------------------------------------------------------------------
        # Re-shape the value arrays (shape, numer1, numer2, denom1, denom2)
        #-------------------------------------------------------------------
        shape1 = (arg1.__shape_ + arg1.__numer_ + arg2.__nrank_ * (1,) +
                  arg1.__denom_ + arg2.__drank_ * (1,))
        array1 = arg1.__values_.reshape(shape1)

        shape2 = (arg2.__shape_ + arg1.__nrank_ * (1,) + arg2.__numer_ +
                  arg1.__drank_ * (1,) + arg2.__denom_)
        array2 = arg2.__values_.reshape(shape2)

        #------------------------------------------------------
        # Construct the outer product
        #------------------------------------------------------
        new_values = array1 * array2

        #------------------------------------------------------
        # Construct the object and cast
        #------------------------------------------------------
        new_nrank = arg1.__nrank_ + arg2.__nrank_
        new_drank = arg1.__drank_ + arg2.__drank_

        obj = Qube(new_values,
                   arg1.__mask_ | arg2.__mask_,
                   units=Units.mul_units(arg1.__units_, arg2.__units_),
                   nrank=new_nrank, drank=new_drank, example=arg1)
        obj = obj.cast(classes)

        #------------------------------------------------------
        # Insert derivatives if necessary
        #------------------------------------------------------
        if recursive and (arg1.__derivs_ or arg2.__derivs_):
            new_derivs = {}

            if arg1.__derivs_:
              arg_wod = arg2.wod
              for (key, self_deriv) in arg1.__derivs_.items():
                new_derivs[key] = Qube.outer(self_deriv, arg_wod, classes,
                                             recursive=False)

            if arg2.__derivs_:
              self_wod = arg1.wod
              for (key, arg_deriv) in arg2.__derivs_.items():
                term = Qube.outer(self_wod, arg_deriv, classes, recursive=False)
                if key in new_derivs:
                    new_derivs[key] += term
                else:
                    new_derivs[key] = term

            obj.insert_derivs(new_derivs)

        return obj
    #===========================================================================



    #===========================================================================
    # as_diagonal
    #===========================================================================
    @staticmethod
    def as_diagonal(arg, axis, classes=(), recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A copy with one axis converted to a diagonal across two.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Position axis from left
        #------------------------------------------------------
        if axis >= 0:
            a1 = axis
        else:
            a1 = axis + arg.__nrank_
        if a1 < 0 or a1 >= arg.__nrank_:
            raise ValueError('axis is out of range (%d,%d): %d',
                             (-arg.__nrank_, arg.__nrank_, axis))
        k1 = a1 + len(arg.__shape_)

        #------------------------------------------------------
        # Roll this axis to the end
        #------------------------------------------------------
        rolled = np.rollaxis(arg.__values_, k1, arg.__values_.ndim)

        #------------------------------------------------------
        # Create the diagonal array
        #------------------------------------------------------
        new_values = np.zeros(rolled.shape + rolled.shape[-1:],
                              dtype=rolled.dtype)

        for i in range(rolled.shape[-1]):
            new_values[...,i,i] = rolled[...,i]

        #------------------------------------------------------
        # Roll the new axes back
        #------------------------------------------------------
        new_values = np.rollaxis(new_values, -1, k1)
        new_values = np.rollaxis(new_values, -1, k1)

        #------------------------------------------------------
        # Construct and cast
        #------------------------------------------------------
        new_numer = new_values.shape[len(arg.__shape_):][:arg.__nrank_+1]
        obj = Qube(new_values, arg.__mask_,
                   nrank=arg.__nrank_ + 1, example=arg)
        obj = obj.cast(classes)

        #------------------------------------------------------
        # Diagonalize the derivatives if necessary
        #------------------------------------------------------
        if recursive:
          for (key, deriv) in arg.__derivs_.items():
            obj.insert_deriv(key, Qube.as_diagonal(deriv, axis, classes, False))

        return obj
    #===========================================================================



    #===========================================================================
    # from_scalars
    #===========================================================================
    @classmethod
    def from_scalars(cls, *scalars, **keywords):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A new instance constructed from Scalars or arrays given as arguments.

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

            classes     an arbitrary list defining the preferred class of the
                        returned object. The first suitable class in the list
                        will be used. Default is Vector.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Search the keywords for "recursive" and "readonly"
        #------------------------------------------------------
        recursive = True
        if 'recursive' in keywords:
            recursive = keywords['recursive']
            del keywords['recursive']

        readonly = False
        if 'readonly' in keywords:
            readonly = keywords['readonly']
            del keywords['readonly']

        classes = []
        if 'classes' in keywords:
            classes = keywords['classes']
            del keywords['classes']

        #------------------------------------------------------
        # No other keyword is allowed
        #------------------------------------------------------
        if keywords:
          raise ValueError('broadcast() got an unexpected keyword argument ' +
                           '"%s"' % keywords.keys()[0])

        #------------------------------------------------------
        # Convert to scalars and broadcast to the same shape
        #------------------------------------------------------
        args = []
        for arg in scalars:
            scalar = Qube.SCALAR_CLASS.as_scalar(arg)
            args.append(scalar)

        scalars = Qube.broadcast(*args, recursive=recursive)

        #------------------------------------------------------
        # Tabulate the properties and construct the value array
        #------------------------------------------------------
        new_units = None
        new_denom = None

        arrays = []
        masks = []
        deriv_dicts = []
        has_derivs = False
        dtype = np.int_
        for scalar in scalars:
            arrays.append(scalar.__values_)
            masks.append(scalar.__mask_)

            new_units = new_units or scalar.__units_
            Units.require_match(new_units, scalar.__units_)

            if new_denom is None:
                new_denom = scalar.__denom_
            elif new_denom != scalar.__denom_:
                raise ValueError('mixed denominator shapes')

            deriv_dicts.append(scalar.__derivs_)
            if len(scalar.__derivs_):
                has_derivs = True

            #- - - - - - - - - - - - - - - - - -
            # Remember any floats encountered
            #- - - - - - - - - - - - - - - - - -
            if scalar.is_float():
                dtype = np.float_

        #------------------------------------------------------
        # Construct the values array
        #------------------------------------------------------
        new_drank = len(new_denom)
        new_values = np.array(arrays)
        new_values = np.rollaxis(new_values, 0, new_values.ndim - new_drank)

        #------------------------------------------------------
        # Construct the mask (scalar or array)
        #------------------------------------------------------
        masks = Qube.broadcast(*masks)
        new_mask = masks[0]
        for mask in masks[1:]:
            new_mask = new_mask | mask

        #------------------------------------------------------
        # Construct the object
        #------------------------------------------------------
        obj = Qube.__new__(cls)
        obj.__init__(new_values, new_mask, units=new_units,
                                 nrank=scalars[0].__nrank_+1, drank=new_drank)
        obj = obj.cast(classes)

        #------------------------------------------------------
        # Insert derivatives if necessary
        #------------------------------------------------------
        if recursive and has_derivs:
            new_derivs = {}
            denoms = {}
            for deriv_dict in deriv_dicts:
                for key,deriv in deriv_dict.items():
                    denoms[key] = deriv.__denom_

            for key,denom in denoms.items():
                items = []
                if denom:
                    missing_deriv = Qube(np.zeros(denom), nrank=deriv.__nrank_,
                                                          drank=deriv.__drank_)
                else:
                    missing_deriv = 0.

                for i,deriv_dict in enumerate(deriv_dicts):
                    items.append(deriv_dict.get(key, missing_deriv))

                new_derivs[key] = Qube.from_scalars(*items, recursive=False,
                                                            readonly=readonly,
                                                            classes=classes)
            obj.insert_derivs(new_derivs)

        return obj
    #===========================================================================



    #===========================================================================
    # rms
    #===========================================================================
    def rms(self):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        The root-mean-square values of all items as a Scalar.

        Useful for looking at the overall magnitude of the differences between
        two objects.

        Input:
            arg         the object for which to calculate the RMS.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Evaluate the norm
        #------------------------------------------------------
        sum_sq = self.__values_**2
        for r in range(self.__rank_):
            sum_sq = np.sum(sum_sq, axis=-1)

        return Qube.SCALAR_CLASS(np.sqrt(sum_sq/self.isize), self.__mask_)
    #===========================================================================



    ############################################################################
    # General shaping operations
    ############################################################################

    #===========================================================================
    # reshape
    #===========================================================================
    def reshape(self, shape, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A shallow copy of the object with a new leading shape.

        Input:
            shape       a tuple defining the new leading shape. A value of -1
                        can appear at one location in the new shape, and the
                        size of that shape will be determined based on this
                        object's size.
            recursive   True to apply the same shape to the derivatives.
                        Otherwise, derivatives are deleted from the returned
                        object.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if np.isscalar(shape):
            shape = (shape,)

        if shape == self.__shape_:
            return self

        if np.isscalar(self.__values_):
            new_values = np.array([self.__values_]).reshape(shape)
        else:
            new_values = self.__values_.reshape(shape + self.item)

        if np.isscalar(self.__mask_):
            new_mask = self.__mask_
        else:
            new_mask = self.__mask_.reshape(shape)

        obj = Qube.__new__(type(self))
        obj.__init__(new_values, new_mask, example=self)
        obj.__readonly_ = self.__readonly_

        if recursive:
            for (key, deriv) in self.__derivs_.items():
                obj.insert_deriv(key, deriv.reshape(shape, False))

        return obj
    #===========================================================================



    #===========================================================================
    # flatten
    #===========================================================================
    def flatten(self, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A shallow copy of the object flattened to one dimension.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if len(self.__shape_) < 2:
            return self

        count = np.product(self.__shape_)
        return self.reshape((count,), recursive)
    #===========================================================================



    #===========================================================================
    # swap_axes
    #===========================================================================
    def swap_axes(self, axis1, axis2, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A shallow copy of the object with two leading axes swapped.

        Input:
            axis1       the first index of the swap. Negative indices
                        Negative indices are relative to the last index before
                        the numerator items begin.
            axis2       the second index of the swap.
            recursive   True to perform the same swap on the derivatives.
                        Otherwise, derivatives are deleted from the returned
                        object.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Validate first axis
        #------------------------------------------------------
        len_shape = len(self.__shape_)
        if axis1 < 0:
            a1 = axis1 + len_shape
        else:
            a1 = axis1

        if a1 < 0 or a1 >= len_shape:
            raise ValueError('axis1 argument out of range (%d,%d): %d' %
                             (-len_shape, len_shape, axis1))

        #------------------------------------------------------
        # Validate second axis
        #------------------------------------------------------
        if axis2 < 0:
            a2 = axis2 + len_shape
        else:
            a2 = axis2

        if a2 < 0 or a2 >= len_shape:
            raise ValueError('axis2 argument out of range (%d,%d): %d' %
                             (-len_shape, len_shape, axis2))

        if a1 == a2:
            return self

        new_values = self.__values_.swapaxes(a1, a2)

        if np.isscalar(self.__mask_):
            new_mask = self.__mask_
        else:
            new_mask = self.__mask_.swapaxes(a1, a2)

        obj = Qube.__new__(type(self))
        obj.__init__(new_values, new_mask, example=self)
        obj.__readonly_ = self.__readonly_

        if recursive:
            for (key, deriv) in self.__derivs_.items():
                obj.insert_deriv(key, deriv.swap_axes(a1, a2, False))

        return obj
    #===========================================================================



    #===========================================================================
    # roll_axis
    #===========================================================================
    def roll_axis(self, axis, start=0, recursive=True, rank=None):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        A shallow copy of the object with the specified axis rolled to a new
        position.

        Input:
            axis        the axis to roll.
            start       the axis will be rolled to fall in front of this axis;
                        default is zero.
            recursive   True to perform the same axis roll on the derivatives.
                        Otherwise, derivatives are deleted from the returned
                        object.
            rank        rank to assume for the object, which could be larger
                        than len(self.shape) because of broadcasting.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Validate the rank
        #------------------------------------------------------
        len_shape = len(self.__shape_)
        if rank is None:
            rank = len_shape
        if rank < len_shape:
            raise ValueError('roll rank %d is too small for object' % rank)

        if len_shape == 0:
            rank = 1

        #------------------------------------------------------
        # Identify the axis to roll, which could be negative
        #------------------------------------------------------
        if axis < 0:
            a1 = axis + rank
        else:
            a1 = axis

        if a1 < 0 or a1 >= rank:
            raise ValueError('roll axis %d out of range' % axis)

        #------------------------------------------------------
        # Identify the start axis, which could be negative
        #------------------------------------------------------
        if start < 0:
            a2 = start + rank
        else:
            a2 = start

        if a2 < 0 or a2 >= rank + 1:
            raise ValueError('roll axis %d out of range' % start)

        #------------------------------------------------------
        # No need to modify a shapeless object
        #------------------------------------------------------
        if not self.__shape_:
            return self

        #------------------------------------------------------
        # Add missing axes if necessary
        #------------------------------------------------------
        if len_shape < rank:
            self = self.reshape((rank - len_shape) * (1,) + self.__shape_,
                                recursive=recursive)

        #------------------------------------------------------
        # Roll the values and mask of the object
        #------------------------------------------------------
        new_values = np.rollaxis(self.__values_, a1, a2)

        if np.shape(self.__mask_):
            new_mask = np.rollaxis(self.__mask_, a1, a2)
        else:
            new_mask = self.__mask_

        obj = Qube.__new__(type(self))
        obj.__init__(new_values, new_mask, example=self)
        obj.__readonly_ = self.__readonly_

        if recursive:
            for (key, deriv) in self.__derivs_.items():
                obj.insert_deriv(key, deriv.roll_axis(a1, a2, False, rank))

        return obj
    #===========================================================================



    #===========================================================================
    # broadcast_into_shape
    #===========================================================================
    def broadcast_into_shape(self, shape, recursive=True):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        This object broadcasted to the specified shape.

        It returns self if the shape already matches. Otherwise, the returned
        object shares data with the original and both objects will be read-only.

        Input:
            shape           the shape into which the object is to be broadcast.
            recursive       True to broadcast the derivatives as well.
                            Otherwise, they are removed.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        shape = tuple(shape)

        #------------------------------------------------------
        # If no broadcast is needed, return the object
        #------------------------------------------------------
        if shape == self.__shape_:
            if recursive:
                return self
            else:
                return self.wod

        #------------------------------------------------------
        # Save the derivatives for later
        #------------------------------------------------------
        derivs = self.__derivs_

        #------------------------------------------------------
        # Special case: broadcast to ()
        #------------------------------------------------------
        if shape == ():
            if self.__rank_ == 0:
                if isinstance(self.__values_, np.ndarray):
                    new_values = self.__values_.ravel()[0]
                else:
                    new_values = self.__values_
            else:
                new_values = self.__values_.reshape(self.__item_)

            if isinstance(self.__mask_, np.ndarray):
                new_mask = bool(self.__mask_.ravel()[0])
            else:
                new_mask = bool(self.__mask_)

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Construct the new object
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            obj = Qube.__new__(type(self))
            obj.__init__(new_values, new_mask, example=self)

        else:
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Broadcast the values array
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if np.isscalar(self.__values_):
                self_values = np.array([self.__values_])
            else:
                self_values = self.__values_
                
                # An array must be read-only upon broadcast
                self.as_readonly(recursive=False)

            new_values = np.broadcast_to(self_values, shape + self.__item_)

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Broadcast the mask if necessary
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            if np.isscalar(self.__mask_):
                new_mask = self.__mask_
            else:
                new_mask = np.broadcast_to(self.__mask_, shape)
                
                # An array must be read-only upon broadcast
                self.as_readonly(recursive=False)

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Construct the new object
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            obj = Qube.__new__(type(self))
            obj.__init__(new_values, new_mask, example=self)
            obj.as_readonly(recursive=False)

        #------------------------------------------------------
        # Process the derivatives if necessary
        #------------------------------------------------------
        if recursive:
            for (key, deriv) in derivs.items():
                obj.insert_deriv(key, deriv.broadcast_into_shape(shape, False))

        return obj
    #===========================================================================



    #===========================================================================
    # broadcasted_shape
    #===========================================================================
    @staticmethod
    def broadcasted_shape(*objects, **keywords):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        The shape defined by a broadcast across the objects.

        Input:          zero or more array objects. Values of None are assigned
                        shape (). A list or tuple is treated as the definition
                        of an additional shape.

            item        a list or tuple to be appended to the shape. Default is
                        (). This makes it possible to use the returned shape in
                        the declaration of a NumPy array containing items that
                        are not scalars. Note that this is handled as a keyword
                        parameter in order to distinguish it from the objects.

        Return:         the broadcast shape, comprising the maximum value of
                        each corresponding axis, plus the item shape appended.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Search the keywords for "item"
        #------------------------------------------------------
        item = ()
        if 'item' in keywords:
            item = keywords['item']
            del keywords['item']

        #------------------------------------------------------
        # No other keyword is allowed
        #------------------------------------------------------
        if keywords:
          raise TypeError(('broadcasted_shape() got an unexpected keyword ' +
                           'argument "%s"') % keywords.keys()[0])

        #------------------------------------------------------
        # Create a list of all shapes
        #------------------------------------------------------
        shapes = []
        for obj in objects:
            if obj is None or Qube.is_real_number(obj):
                shape = ()
            elif isinstance(obj, (tuple,list)):
                shape = tuple(obj)
            else:
                shape = obj.shape

            shapes.append(shape)

        #------------------------------------------------------
        # Initialize the shape
        #------------------------------------------------------
        new_shape = []
        len_broadcast = 0
        
        #------------------------------------------------------
        # Loop through the arrays...
        #------------------------------------------------------
        for shape in shapes:
            shape = list(shape)

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Expand the shapes to the same rank
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            len_shape = len(shape)

            if len_shape > len_broadcast:
                new_shape = (len_shape - len_broadcast) * [1] + new_shape
                len_broadcast = len_shape

            if len_broadcast > len_shape:
                shape = (len_broadcast - len_shape) * [1] + shape
                len_shape = len_broadcast

            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Update the broadcast shape and check for compatibility
            #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            for i in range(len_shape):
                if new_shape[i] == 1:
                    new_shape[i] = shape[i]
                elif shape[i] == 1:
                    pass
                elif shape[i] != new_shape[i]:
                    raise ValueError('incompatible dimension on axis ' + str(i)
                                     + ': ' + str(shapes))

        return tuple(new_shape) + tuple(item)
    #===========================================================================



    #===========================================================================
    # broadcast
    #===========================================================================
    @staticmethod
    def broadcast(*objects, **keywords):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Broadcast objects to their common shape.

        Python scalars are returned unchanged because they already broadcast
        with anything.

        Returned objects must be treated as read-only because of the mechanism
        NumPy uses to broadcast arrays. The returned objects are marked
        read-only but their internal arrays are not protected.

        Input:          zero or more objects to broadcast.

            recursive   True to broadcast the derivatives to the same shape;
                        False to strip the derivatives from the returned
                        objects. Default is True.

        Return:         A tuple of copies of the objects, broadcasted to a
                        common shape. The returned objects must be treated as
                        read-only.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Search the keywords for "recursive"
        #------------------------------------------------------
        recursive = True
        if 'recursive' in keywords:
            recursive = keywords['recursive']
            del keywords['recursive']

        #------------------------------------------------------
        # No other keyword is allowed
        #------------------------------------------------------
        if keywords:
          raise TypeError(('broadcast() got an unexpected keyword argument ' +
                           '"%s"') % keywords.keys()[0])

        #------------------------------------------------------
        # Perform the broadcasts...
        #------------------------------------------------------
        shape = Qube.broadcasted_shape(*objects)
        results = []
        for obj in objects:
            if isinstance(obj, np.ndarray):
                new_obj = np.broadcast_to(obj, shape)
            elif isinstance(obj, Qube):
                new_obj = obj.broadcast_into_shape(shape, recursive=recursive)
            else:
                new_obj = obj
            results.append(new_obj)

        return tuple(results)
    #===========================================================================



    #===========================================================================
    # stack
    #===========================================================================
    @staticmethod
    def stack(*args, **keywords):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Stack objects of the same class into one with a new leading axis.

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
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #------------------------------------------------------
        # Search the keywords for "recursive"
        #------------------------------------------------------
        recursive = True
        if 'recursive' in keywords:
            recursive = keywords['recursive']
            del keywords['recursive']

        #------------------------------------------------------
        # No other keyword is allowed
        #------------------------------------------------------
        if keywords:
          raise TypeError(('stack() got an unexpected keyword argument ' +
                           '"%s"') % keywords.keys()[0])

        args = list(args)

        #------------------------------------------------------
        # Get the type and units if any
        # Only use class Qube if no suitable subclass was found
        #------------------------------------------------------
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
                denom = arg.__denom_
            elif denom != arg.__denom_:
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

            if arg.__units_ is not None:
                if units is None:
                    units = arg.__units_
                else:
                    arg.confirm_units(units)

        drank = len(denom)

        #------------------------------------------------------
        # Convert to subclass and type
        #------------------------------------------------------
        for (i,arg) in enumerate(args):
            if arg is None:                 # Used as placehold for derivs
                continue

            args[i] = args[subclass_indx].as_this_type(arg, recursive=recursive,
                                                       coerce=False)

        #------------------------------------------------------
        # Broadcast all inputs into a common shape
        #------------------------------------------------------
        args = Qube.broadcast(*args, recursive=True)

        #------------------------------------------------------
        # Determine what type of mask is needed:
        #------------------------------------------------------
        mask_true_found = False
        mask_false_found = False
        mask_array_found = False
        for arg in args:
            if arg is None:
                continue
            elif Qube.is_one_true(arg.__mask_):
                mask_true_found = True
            elif Qube.is_one_false(arg.__mask_):
                mask_false_found = True
            else:
                mask_array_found = True

        #------------------------------------------------------
        # Construct the mask
        #------------------------------------------------------
        if  mask_array_found or (mask_false_found and mask_true_found):
            mask = np.zeros((len(args),) + args[subclass_indx].shape,
                            dtype=np.bool_)
            for i in range(len(args)):
                if args[i] is None:
                    mask[i] = False
                else:
                    mask[i] = args[i].__mask_
        else:
            mask = mask_true_found

        #------------------------------------------------------
        # Construct the array
        #------------------------------------------------------
        if floats_found:
            dtype = np.float_
        elif ints_found:
            dtype = np.int_
        else:
            dtype = np.bool_

        values = np.empty((len(args),) + args[subclass_indx].__values_.shape,
                          dtype=dtype)
        for i in range(len(args)):
            if args[i] is None:
                values[i] = 0
            else:
                values[i] = args[i].__values_

        #------------------------------------------------------
        # Construct the result
        #------------------------------------------------------
        result = Qube.__new__(type(args[subclass_indx]))
        result.__init__(values, mask, units=units, drank=drank)

        #------------------------------------------------------
        # Fill in derivatives if necessary
        #------------------------------------------------------
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
    #===========================================================================

#*******************************************************************************


################################################################################
