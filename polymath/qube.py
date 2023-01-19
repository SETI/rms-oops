################################################################################
# polymath/qube.py: Base class for all PolyMath subclasses.
################################################################################

from __future__ import division
import numpy as np
import numbers

from polymath.units import Units

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
        _shape_     a tuple representing the leading axes, which are not
                    considered part of the items.

        _rank_      the number of axes belonging to the items.
        _nrank_     the number of numerator axes associated with the items.
        _drank_     the number of denominator axes associated with the items.

        _item_      a tuple representing the shape of the individual items.
        _numer_     a tuple representing the shape of the numerator items.
        _denom_     a tuple representing the shape of the denominator items.

        _values_    the array's data as a NumPy array or a Python scalar. The
                    shape of this array is object.shape + object.item. If the
                    object has units, then the values are in in default units
                    instead.
        _mask_      the array's mask as a NumPy boolean array. The array value
                    is True if the Array value at the same location is masked.
                    A scalar value of False indicates that the entire object is
                    unmasked; a scalar value of True indicates that it is
                    entirely masked.
        _units_     the units of the array, if any. None indicates no units.
        _derivs_    a dictionary of the names and values of any derivatives,
                    represented by additional PolyMath objects.

    For each of these, there exists a read-only property that has the same name
    minus the leading and trailing underscores.

    Additional attributes are filled in as needed
        _readonly_  True if the object cannot (or at least should not) be
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

    # This prevents binary operations of the form:
    #   <np.ndarray> <op> <Qube>
    # from executing the ndarray operation instead of the polymath operation
    __array_priority__ = 1

    # Set this global attribute to True to restore a behavior removed on
    # 3/13/18. It allowed certain functions to return a Python value (float,
    # int, or bool) if the result had shape (), was unmasked, and had no units.
    # Currently, these functions all return Scalar or Boolean types instead.
    #
    # As an alternative to this global solution, you can use method as_builtin()
    # to convert an object to a built-in type if this is possible.
    PREFER_BUILTIN_TYPES = False

    # Global attribute to be used for testing
    DISABLE_CACHE = False

    # If this global is set to True, the shrink/unshrink methods are disabled.
    # Calculations done with and without shrinking should always produce the
    # same results, although they may be slower with shrinking disabled. Used
    # for testing and debugging.
    _DISABLE_SHRINKING = False

    # If this global is set to True, the unshrunk method will ignore any cached
    # value of its un-shrunken equivalent. Used for testing and debugging.
    _IGNORE_UNSHRUNK_AS_CACHED = False

    # Default class constants, to be overridden as needed by subclasses...
    NRANK = None        # the number of numerator axes; None to leave this
                        # unconstrained.
    NUMER = None        # shape of the numerator; None to leave unconstrained.

    FLOATS_OK = True    # True to allow floating-point numbers.
    INTS_OK = True      # True to allow integers.
    BOOLS_OK = True     # True to allow booleans.

    UNITS_OK = True     # True to allow units; False to disallow them.
    DERIVS_OK = True    # True to allow derivatives and to allow this class to
                        # have a denominator; False to disallow them.

    #===========================================================================
    def __new__(subtype, *values, **keywords):
        """"Create a new, un-initialized object."""

        return object.__new__(subtype)

    #===========================================================================
    def __init__(self, arg, mask=False, derivs={}, units=None,
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

        mask        the mask for the object, as a single boolean, an array,
                    anything array-like, or a Boolean. Use None to copy the mask
                    from the example object. False (the default) leaves the
                    object un-masked.

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
                    arguments except derivs that have not been explicitly
                    specified.

        default     value to use where masked. Typically a constant that will
                    not "break" most arithmetic calculations. It must be a
                    Python built-in constant or ndarray of the same shape as the
                    items. Default is None, in which case the class constant
                    DEFAULT_VALUE is used, or else it is filled with ones.
        """

        # Set defaults based on a Qube input
        if isinstance(arg, Qube):

            if derivs is None:
                derivs = arg._derivs_.copy()    # shallow copy

            if units is None:
                units = arg._units_

            if nrank is None:
                nrank = arg._nrank_
            elif nrank != arg._nrank_:          # nranks _must_ be compatible
                raise ValueError('numerator ranks are incompatible: %d, %d'
                                 % (nrank, arg._nrank_))

            if drank is None:
                drank = arg._drank_
            elif drank != arg._drank_:
                raise ValueError('denominator ranks are incompatible: %d, %d'
                                 % (nrank, arg._nrank_))

            if default is None:
                default = arg._default_

        # Set defaults based on an example object
        if example is not None:

            if not isinstance(example, Qube):
                raise TypeError('example value is not a Qube subclass: '
                                + str(example))

            if mask is None:
                mask = example._mask_

            if units is None and self.UNITS_OK:
                units = example._units_

            if nrank is None and self.NRANK is None:
                nrank = example._nrank_

            if drank is None:
                drank = example._drank_

            if default is None:
                default = example._default_

        # Validate inputs
        nrank = nrank or self.NRANK or 0
        drank = drank or 0
        rank = nrank + drank

        if derivs and not self.DERIVS_OK:
            raise ValueError('derivatives are disallowed for class %s'
                             % type(self).__name__)

        if units and not self.UNITS_OK:
            raise TypeError('units are disallowed for class %s: %s'
                            % (type(self).__name__, str(units)))

        if self.NRANK is not None:
            if nrank is not None and nrank != self.NRANK:
                raise ValueError('invalid numerator rank for class ' +
                                 '%s: %d' % (type(self).__name__, nrank))

        if drank and not self.DERIVS_OK:
            raise ValueError('denominators are disallowed for class %s'
                             % type(self).__name__)

        # Get the value and check its shape
        (values, arg_mask) = Qube._as_values_and_mask(arg)
        full_shape = np.shape(values)
        if len(full_shape) < rank:
            raise ValueError(('incompatible array shape for class %s: ' +
                              '%s; minimum rank = %d + %d') %
                              (type(self).__name__, str(full_shape),
                               nrank, drank))

        dd = len(full_shape) - drank
        nn = dd - nrank
        denom = full_shape[dd:]
        numer = full_shape[nn:dd]
        item  = full_shape[nn:]
        shape = full_shape[:nn]

        # Fill in the values
        self._values_ = self._suitable_value(values, numer=numer, denom=denom)

        # Get the mask and check its shape
        mask = Qube.or_(arg_mask, Qube._as_mask(mask))
        collapse = isinstance(arg, np.ma.MaskedArray)
        self._mask_ = Qube._suitable_mask(mask, shape=shape, broadcast=True,
                                                collapse=collapse,
                                                check=False)

        # Fill in the remaining shape info
        self._rank_  = rank
        self._nrank_ = nrank
        self._drank_ = drank
        self._item_  = item
        self._numer_ = numer
        self._denom_ = denom
        self._shape_ = shape
        self._size_  = int(np.prod(shape))
        self._isize_ = int(np.prod(item))
        self._nsize_ = int(np.prod(numer))
        self._dsize_ = int(np.prod(denom))

        # Fill in the units
        if Qube.is_one_false(units):
            self._units_ = None
        else:
            self._units_ = units

        # The object is read-only if the values array is read-only
        self._readonly_ = Qube._array_is_readonly(self._values_)

        if self._readonly_:
            Qube._array_to_readonly(self._mask_)

        # Used for anything we want to cache in association with an object
        # This cache will be cleared whenever the object is modified in any way
        self._cache_ = {}

        # Install the derivs (converting to read-only if necessary)
        self._derivs_ = {}
        if derivs:
            self.insert_derivs(derivs)

        # Used only for if clauses
        self._truth_if_any_ = False
        self._truth_if_all_ = False

        # Fill in the default
        if default is not None and np.shape(default) == item:
            pass

        elif hasattr(self, 'DEFAULT_VALUE') and drank == 0:
            default = self.DEFAULT_VALUE

        elif item:
            default = np.ones(item)

        else:
            default = 1

        dtype = Qube._dtype(self._values_)
        self._default_ = Qube._casted_to_dtype(default, dtype)

    ############################################################################
    # Support functions
    ############################################################################

    @staticmethod
    def _has_qube(arg):
        """True if this is a list or tuple containing a Qube somewhere within.
        """

        if isinstance(arg, (list, tuple)):
            return (any(isinstance(item, Qube) for item in arg) or
                    any(Qube._has_qube(item) for item in arg))
        return False

    #===========================================================================
    @staticmethod
    def _has_masked_array(arg):
        """True if this is a list or tuple containing a MaskedArray somewhere
        within.
        """

        if isinstance(arg, (list, tuple)):
            return (any(isinstance(item, np.ma.MaskedArray) for item in arg) or
                    any(Qube._has_masked_array(item) for item in arg))
        return False

    #===========================================================================
    @staticmethod
    def _as_values_and_mask(arg):
        """This object converted to a scalar or Numpy array with optional mask.

        Input:
            arg         object to convert to a scalar or array.
        """

        if isinstance(arg, numbers.Real):
            return (arg, False)

        if isinstance(arg, np.ma.MaskedArray):
            return (arg.data, arg.mask)

        if isinstance(arg, np.ndarray):
            return (arg, False)

        if isinstance(arg, Qube):
            return (arg._values_, arg._mask_)

        if isinstance(arg, (list, tuple)):
            if Qube._has_qube(arg):
                merged = Qube.stack(*arg)
                return (merged._values_, merged._mask_)

            elif Qube._has_masked_array(arg):
                merged = np.ma.stack(*arg)
                return (merged.data, merged.mask)

            else:
                merged = np.array(arg)
                return (merged, False)

        if isinstance(arg, np.bool_):
            return (bool(arg), False)

        raise TypeError('invalid data type: ' + str(arg))

    #===========================================================================
    @staticmethod
    def _as_mask(arg, invert=False, masked_value=True):
        """This argument converted to a scalar bool or boolean Numpy array.

        Input:
            arg         object to convert to a mask.
            invert      True to return the logical not of the mask.
            masked_value
                        True or False, the value to use where the input argument
                        is masked. Default is True. This applies _after_ any
                        inversion.
        """

        # Handle most common cases first
        if isinstance(arg, (bool, np.bool_, type(None), numbers.Real)):
            return bool(arg) ^ invert

        if type(arg) == np.ndarray:     # exact type, not a subclass
            if arg.dtype.kind == 'b' and not invert:
                return arg
            elif invert:
                return arg == 0
            else:
                return arg != 0

        # Convert a list or tuple to something else
        if isinstance(arg, (list, tuple)):
            if Qube._has_qube(arg):
                arg = Qube.stack(*arg)

            elif Qube._has_masked_array(arg):
                arg = np.ma.stack(*arg)

            else:
                arg = np.array(arg)
                return Qube._as_mask(arg, invert=invert,
                                          masked_value=masked_value)

        # Handle an object with a possible mask
        if isinstance(arg, Qube):
            mask = arg._mask_
            arg = arg._values_
        elif isinstance(arg, np.ma.MaskedArray):
            mask = arg.mask
            arg = arg.data
        else:
            raise TypeError('invalid class for mask: ' + type(arg).__name__)

        # Handle a shapeless mask
        if isinstance(mask, (bool, np.bool_)):
            if mask:                        # entirely masked
                return bool(masked_value)
            else:                           # entirely unmasked
                return Qube._as_mask(arg, invert=invert,
                                          masked_value=masked_value)

        # Copy the arg and merge the mask
        if invert:
            merged = (arg == 0)
        else:
            merged = (arg != 0)

        merged[mask] = masked_value
        return merged

    #===========================================================================
    @staticmethod
    def _suitable_mask(arg, shape, collapse=False, broadcast=False,
                                   invert=False, masked_value=True,
                                   check=False):
        """This argument converted to a scalar bool or boolean Numpy array of
        suitable shape to use as a mask.

        Input:
            arg         object to convert to a mask.
            shape       shape of the required mask.
            collapse    True to merge the extraneous axes of a mask if its rank
                        is greater than that of the given shape.
            expand      True to broadcast this mask if its rank is less than
                        that of the given shape.
            invert      True to return the logical not of the mask.
            masked_value
                        True or False, the value to use where the input argument
                        is masked. Default is True. This applies _after_ any
                        inversion.
            check       True to check for an array containing all False values,
                        and if so, replace it by a single value of False.
        """

        mask = Qube._as_mask(arg, invert=invert, masked_value=masked_value)

        if isinstance(mask, bool):
            return mask

        if mask.shape == shape:
            if check and not np.any(mask):
                return False
            return mask

        new_rank = len(shape)
        if collapse and mask.ndim > new_rank:
            axes = tuple(range(new_rank, mask.ndim))
            mask = np.any(mask, axis=axes)
            if np.isscalar(mask):
                return bool(mask)
            if mask.shape == shape:
                return mask

        if broadcast:
            try:
                mask = np.broadcast_to(mask, shape)
            except ValueError:
                pass
            else:
                Qube._array_to_readonly(mask)
                return mask

        raise ValueError('mask shape mismatch; mask is %s; ' % str(mask.shape) +
                         'object is %s' % str(shape))

    #===========================================================================
    @staticmethod
    def _dtype_and_value(arg, masked_value=0):
        """Tuple (dtype, value), where dtype is one of "float", "int", or
        "bool".

        The value is converted to a builtin type if it is scalar; otherwise it
        is return as an array with its original dtype.
        """

        # Handle the easy and common cases first
        if isinstance(arg, (bool, np.bool_)):
            return ('bool', bool(arg))

        if isinstance(arg, numbers.Integral):
            return ('int', int(arg))

        if isinstance(arg, numbers.Real):
            return ('float', float(arg))

        if isinstance(arg, np.ndarray):
            if arg.shape == ():         # shapeless array
                return Qube._dtype_and_value(arg[()])

            kind = arg.dtype.kind
            if kind == 'f':
                return ('float', arg)

            if kind in ('i','u'):
                return ('int', arg)

            if kind == 'b':
                return ('bool', arg)

            raise ValueError('unsupported dtype: %s' % str(arg.dtype))

        # Convert a list or tuple to something else
        if isinstance(arg, (list, tuple)):
            if Qube._has_qube(arg):
                arg = Qube.stack(*arg)
            elif Qube._has_masked_array(arg):
                arg = np.ma.stack(*arg)
            else:
                arg = np.array(arg)
                return Qube._dtype_and_value(arg)

        # Handle an object with a possible mask
        if isinstance(arg, Qube):
            mask = arg._mask_
            arg = arg._values_
        elif isinstance(arg, np.ma.MaskedArray):
            mask = arg.mask
            arg = arg.data
        else:
            raise TypeError('unsupported data type: ' + type(arg).__name__)

        # Interpret the argument ignoring its mask
        (dtype, arg) = Qube._dtype_and_value(arg)

        # Handle a shapeless mask
        if isinstance(mask, (bool, np.bool_)):
            if mask:                        # entirely masked
                return (dtype, Qube._casted_to_dtype(masked_value, dtype))
            else:                           # entirely unmasked
                return (dtype, arg)

        # Mask an array value
        arg = arg.copy()
        arg[mask] = masked_value
        return (dtype, arg)

    #===========================================================================
    @staticmethod
    def _dtype(arg):
        """dtype of argument, one of "float", "int", or "bool"."""

        return Qube._dtype_and_value(arg)[0]

    #===========================================================================
    @staticmethod
    def _casted_to_dtype(arg, dtype, masked_value=0):
        """This value casted to the specified dtype, one of "float", "int", or
        "bool".

        An object that is already of the requested type is returned unchanged.

        Note that converting floats to ints is always a "floor" operation, so
        -1.5 -> -2.

        Input:
            arg         object to cast
            dtype       dtype to cast to, one of float", "int", or "bool".
            masked_value
                        value to assign to a masked item in the case where the
                        input argument is a Qube or MaskedArray. Default 0.
        """

        if isinstance(arg, (list, tuple)):
            arg = np.array(arg)

        if isinstance(arg, Qube):
            if arg._mask_ is False:
                arg = arg._values_
            else:
                mask = arg._mask_
                arg = arg.without_mask(recursive=False).copy()
                arg[mask] = masked_value
                arg = arg._values_

        elif isinstance(arg, np.ma.MaskedArray):
            if arg.mask is False:
                arg = arg.data
            else:
                mask = arg.mask
                arg = arg.data.copy()
                arg[mask] = masked_value

        if isinstance(arg, np.ndarray):
            if arg.shape == ():
                return Qube._casted_to_dtype(arg[()], dtype)

            if dtype == 'float':
                if arg.dtype.kind == 'f':
                    return arg
                return np.asfarray(arg)

            if dtype == 'int':
                if arg.dtype.kind in ('i', 'u'):
                    return arg
                return (arg // 1).astype('int')

            # must be bool
            if arg.dtype.kind == 'b':
                return arg

            return (arg != 0)

        # Handle shapeless
        if dtype == 'float':
            return float(arg)

        if dtype == 'int':
            if isinstance(arg, numbers.Integral):
                return int(arg)
            return int(arg // 1)

        # bool case
        if isinstance(arg, (bool, np.bool_)):
            return bool(arg)

        return (arg != 0)

    #===========================================================================
    @classmethod
    def _suitable_dtype(cls, dtype='float'):
        """The dtype for this class closest to a given dtype."""

        if dtype == 'float':
            if cls.FLOATS_OK:
                return 'float'
            elif cls.INTS_OK:
                return 'int'
            else:
                return 'bool'

        if dtype == 'int':
            if cls.INTS_OK:
                return 'int'
            elif cls.FLOATS_OK:
                return 'float'
            else:
                return 'bool'

        if dtype == 'bool':
            if cls.BOOLS_OK:
                return 'bool'
            elif cls.INTS_OK:
                return 'int'
            else:
                return 'float'

        # Handle a NumPy dtype
        try:
            kind = np.dtype(dtype).kind
        except (TypeError, ValueError):
            pass
        else:
            if kind == 'f':
                return cls._suitable_dtype('float')
            if kind in ('i', 'u'):
                return cls._suitable_dtype('int')
            if kind == 'b':
                return cls._suitable_dtype('bool')

        raise ValueError('dtype must be one of "float", "int", "bool": "%s"'
                         % str(dtype))

    #===========================================================================
    @classmethod
    def _suitable_numer(cls, numer=None):
        """The given numerator made suitable for this class; ValueError
        otherwise.

        None to return a default numerator
        """

        if numer is None:
            if cls.NUMER is not None:
                return cls.NUMER

            if not cls.NRANK:
                return ()

            raise ValueError('class %s does not have a default numerator'
                             % cls.__name__)

        numer = tuple(numer)

        if cls.NUMER is not None and numer != cls.NUMER:
            raise ValueError(('incompatible numerator shape for class %s: ' +
                              '%s, %s') % (cls.__name__,
                                           str(cls.NUMER), str(numer)))

        if cls.NRANK is not None and len(numer) != cls.NRANK:
            raise ValueError('invalid numerator rank for class ' +
                             '%s: %d' % (cls.__name__, len(numer)))

        return numer

    #===========================================================================
    @classmethod
    def _suitable_value(cls, arg, numer=None, denom=(), expand=True):
        """This argument converted to a suitable value for this class.

        Input:
            arg         given value.
            numer       numerator shape; None for class default.
            denom       denominator shape.
            expand      True to expand the shape of the returned argument to the
                        minimum required for the class; False to leave it with
                        its original shape.
            """

        # Convert arg to a valid dtype
        (old_dtype, arg) = Qube._dtype_and_value(arg)
        new_dtype = cls._suitable_dtype(old_dtype)
        if new_dtype != old_dtype:
            arg = Qube._casted_to_dtype(arg, new_dtype)

        # Without expansion, we're done
        if not expand:
            return arg

        # Get the valid numerator
        numer = cls._suitable_numer(numer)

        # Expand the arg shape if necessary
        item = numer + denom
        if len(np.shape(arg)) < len(item):
            temp = np.empty(item, dtype=new_dtype)
            temp[...] = arg
            arg = temp

        return arg

    #===========================================================================
    @staticmethod
    def or_(*masks):
        """The logical "or" of two or more masks, avoiding array operations if
        possible.
        """

        # Two inputs is most common
        if len(masks) == 2:
            mask0 = masks[0]
            mask1 = masks[1]

            if isinstance(mask0, (bool, np.bool_)):
                if mask0:
                    return True
                else:
                    return mask1

            if isinstance(mask1, (bool, np.bool_)):
                if mask1:
                    return True
                else:
                    return mask0

            if mask0 is mask1:          # can happen when objects share masks
                return mask0

            return mask0 | mask1

        # Handle one input
        if len(masks) == 1:
            return masks[0]

        # Handle three or more by recursion
        return Qube.or_(masks[0], Qube.or_(*masks[1:]))

    #===========================================================================
    @staticmethod
    def and_(*masks):
        """The logical "and" of two or more masks, avoiding array operations if
        possible.
        """

        # Two inputs is most common
        if len(masks) == 2:
            mask0 = masks[0]
            mask1 = masks[1]

            if isinstance(mask0, (bool, np.bool_)):
                if mask0:
                    return mask1
                else:
                    return False

            if isinstance(mask1, (bool, np.bool_)):
                if mask1:
                    return mask0
                else:
                    return False

            if mask0 is mask1:          # can happen when objects share masks
                return mask0

            return mask0 & mask1

        # Handle one input
        if len(masks) == 1:
            return masks[0]

        # Handle three or more by recursion
        return Qube.and_(masks[0], Qube.and_(*masks[1:]))

    ############################################################################
    # Alternative constructors
    ############################################################################

    def clone(self, recursive=True, preserve=[], retain_cache=False):
        """Fast construction of a shallow copy.

        Inputs:
            recursive       True to clone the derivatives of this object; False
                            to ignore them.
            preserve        an optional list of derivative names to include even
                            if recursive is False.
            retain_cache    True to retain cache except "unshrunk"; False
                            (default) to return clone with an empty cache.
        """

        obj = Qube.__new__(type(self))

        # Transfer attributes other than derivatives and cache
        for (attr, value) in self.__dict__.items():
            if attr in ('_derivs_', '_cache_'):
                obj.__dict__[attr] = {}
            elif attr.startswith('d_d'):
                continue
            elif isinstance(value, dict):
                obj.__dict__[attr] = value.copy()
            else:
                obj.__dict__[attr] = value

        # Handle derivatives recursively
        if recursive:
            new_keys = set(self._derivs_.keys())
        elif preserve:
            if isinstance(preserve, str):
                new_keys = {preserve}
            else:
                new_keys = set(preserve)
        else:
            new_keys = set()

        for key in new_keys:
            deriv = self._derivs_[key]
            new_deriv = deriv.clone(recursive=False,
                                    retain_cache=retain_cache)
            obj.insert_deriv(key, new_deriv)

        # Handle cache
        if retain_cache:
            obj._cache_ = self._cache_.copy()
            if 'shrunk' in obj._cache_:
                del obj._cache_['shrunk']
        else:
            obj._cache_ = {}

        return obj

    #===========================================================================
    @classmethod
    def zeros(cls, shape, dtype='float', numer=None, denom=(), mask=False):
        """New object of this class and shape, filled with zeros.

        Input:
            shape       shape of the object.
            dtype       one of "bool", "int", or "float", defining the data
                        type. Ignored if the class has a default dtype.
            numer       numerator shape; None to use default for class.
            denom       denominator shape.
            mask        optional mask to apply.
        """

        dtype = cls._suitable_dtype(dtype)
        numer = cls._suitable_numer(numer)

        obj = Qube.__new__(cls)
        obj.__init__(np.zeros(shape + numer + denom, dtype=dtype),
                     mask=mask, drank=len(denom))
        return obj

    #===========================================================================
    @classmethod
    def ones(cls, shape, dtype='float', numer=None, denom=(), mask=False):
        """New object of this class and shape, filled with ones.

        Input:
            shape       shape of the object.
            dtype       one of "bool", "int", or "float", defining the data
                        type. Ignored if the class has a default dtype.
            numer       numerator shape; None to use default for class.
            denom       denominator shape.
            mask        optional mask to apply.
        """

        dtype = cls._suitable_dtype(dtype)
        numer = cls._suitable_numer(numer)

        obj = Qube.__new__(cls)
        obj.__init__(np.ones(shape + numer + denom, dtype=dtype),
                     mask=mask, drank=len(denom))
        return obj

    #===========================================================================
    @classmethod
    def filled(cls, shape, fill=0, numer=None, denom=(), mask=False):
        """Internal object of this class and shape, filled with a constant.

        Input:
            shape       shape of the object.
            fill        value with which to fill the object. This can also
                        define the class and/or item shape of the object.
            numer       numerator shape; None to use default for class.
            denom       denominator shape.
            mask        optional mask to apply.
        """

        # Create example object with shape == ()
        example = Qube.__new__(cls)
        example.__init__(cls._suitable_value(fill, numer=numer, denom=denom),
                         drank=len(denom))

        # For a shapeless object, return the example
        if not shape:
            if not mask:
                return example
            example = example.remask(mask)
            return example

        # Return the filled object
        vals = np.empty(shape + example._item_, dtype=example.dtype())
        vals[...] = example._values_

        obj = Qube.__new__(cls)
        obj.__init__(vals, mask=mask, example=example, drank=len(denom))
        return obj

    ############################################################################
    # Low-level access
    ############################################################################

    def _set_values_(self, values, mask=None, antimask=None,
                                   retain_cache=False):
        """Low-level method to update the values of an array.

        The read-only status of the object is defined by that of the given
        value.
        If a mask is provided, it is also updated.
        If antimask is not None, then only the array locations associated with
        the antimask are modified.
        If retain_cache is True, then the contents of the cache are retained
        except for "unshrunk".
        """

        # Confirm shapes
        if antimask is None:
            if np.shape(values) != np.shape(self._values_):
                raise ValueError('value shape mismatch; old is ' +
                                 str(np.shape(self._values_)) + '; new is ' +
                                 str(np.shape(values)))
            if isinstance(mask, np.ndarray):
                if np.shape(mask) != self._shape_:
                    raise ValueError('mask shape mismatch; mask is ' +
                                      str(np.shape(mask)) + '; object is ' +
                                      str(self._shape_))
        else:
            if np.shape(antimask):
                if np.shape(antimask) != self._shape_:
                    raise ValueError('antimask shape mismatch; antimask is ' +
                                     str(np.shape(antimask)) + '; object is ' +
                                     str(self._shape_))

        # Update values
        if antimask is None:
            self._values_ = values
        elif np.isscalar(values):
            self._values_ = values
        else:
            self._values_[antimask] = values[antimask]

        self._readonly_ = Qube._array_is_readonly(self._values_)

        # Update the mask if necessary
        if mask is not None:

            if antimask is None:
                self._mask_ = mask
            elif np.isscalar(mask):
                if np.isscalar(self._mask_):
                    old_mask = self._mask_
                    self._mask_ = np.empty(self._shape_, dtype=np.bool_)
                    self._mask_.fill(old_mask)
                self._mask_[antimask] = mask
            else:
                self._mask_[antimask] = mask[antimask]

        # Handle the cache
        if retain_cache and mask is None:
            if 'unshrunk' in self._cache_:
                del self._cache_['unshrunk']
        else:
            self._cache_.clear()

        # Set the readonly state based on the values given
        if np.shape(self._mask_):
            if self._readonly_:
                self._mask_ = Qube._array_to_readonly(self._mask_)

            elif Qube._array_is_readonly(self._mask_):
                self._mask_ = self._mask_.copy()

        return self

    def _new_values_(self):
        """Low-level method to indicate that values have changed.

        This means "unshrunk" will be deleted from the cache if present.
        """

        if 'unshrunk' in self._cache_:
            del self._cache_['unshrunk']

    def _set_mask_(self, mask, antimask=None, check=False):
        """Low-level method to update the mask of an array.

        The read-only status of the object will be preserved.
        If antimask is not None, then only the mask locations associated with
        the antimask are modified

        Input:
            mask        new mask to apply.
            antimask    if not None, then only the mask locations associated
                        with the antimask are modified.
            check       True to check the mask for an array containing all False
                        values, and if so, replace it by a scalar False.
        """

        # Cast the mask and confirm the shape
        mask = Qube._suitable_mask(mask, self._shape_, check=check)
        is_readonly = self._readonly_

        if antimask is None:
            self._mask_ = mask
        elif np.isscalar(mask):
            if np.isscalar(self._mask_):
                old_mask = self._mask_
                self._mask_ = np.empty(self._shape_, dtype=np.bool_)
                self._mask_.fill(old_mask)
            self._mask_[antimask] = mask
        else:
            self._mask_[antimask] = mask[antimask]

        self._cache_.clear()

        if isinstance(self._mask_, np.ndarray):
            if is_readonly:
                self._mask_ = Qube._array_to_readonly(self._mask_)

            elif Qube._array_is_readonly(self._mask_):
                self._mask_ = self._mask_.copy()

        return self

    ############################################################################
    # Properties
    ############################################################################

    @property
    def values(self):
        return self._values_

    @property
    def vals(self):
        return self._values_       # Handy shorthand

    @property
    def mvals(self):
        """This object as a MaskedArray."""

        # Deal with a scalar
        if np.isscalar(self._values_):
            if self._mask_:
                return np.ma.masked
            else:
                return np.ma.MaskedArray(self._values_)

        # Deal with a scalar mask
        if isinstance(self._mask_, (bool, np.bool_)):
            if self._mask_:
                return np.ma.MaskedArray(self._values_, True)
            else:
                return np.ma.MaskedArray(self._values_)

        # For zero rank, the mask is already the right size
        if self._rank_ == 0:
            return np.ma.MaskedArray(self._values_, self._mask_)

        # Expand the mask
        mask = self._mask_.reshape(self._shape_ + self._rank_ * (1,))
        mask = np.broadcast_to(mask, self._values_.shape)
        return np.ma.MaskedArray(self._values_, mask)

    @property
    def mask(self):
        return self._mask_

    @property
    def antimask(self):
        """The inverse of the mask, True where an element is valid."""

        if not Qube.DISABLE_CACHE and 'antimask' in self._cache_:
            return self._cache_['antimask']

        if isinstance(self._mask_, np.ndarray):
            antimask = np.logical_not(self._mask_)
            self._cache_['antimask'] = antimask
            return antimask

        antimask = not self._mask_
        self._cache_['antimask'] = antimask
        return antimask

    @property
    def default(self):
        return self._default_

    @property
    def units(self):
        return self._units_

    @property
    def derivs(self):
        return self._derivs_

    @property
    def shape(self):
        return self._shape_

    @property
    def rank(self):
        return self._rank_

    @property
    def nrank(self):
        return self._nrank_

    @property
    def drank(self):
        return self._drank_

    @property
    def item(self):
        return self._item_

    @property
    def numer(self):
        return self._numer_

    @property
    def denom(self):
        return self._denom_

    @property
    def size(self):
        return self._size_

    @property
    def isize(self):
        return self._isize_

    @property
    def nsize(self):
        return self._nsize_

    @property
    def dsize(self):
        return self._dsize_

    @property
    def readonly(self):
        return self._readonly_

    @property
    def wod(self):
        """A shallow clone without derivatives, cached. Read-only objects remain
        read-only.
        """

        if not self._derivs_:
            return self

        if not Qube.DISABLE_CACHE and 'wod' in self._cache_:
            return self._cache_['wod']

        wod = Qube.__new__(type(self))
        wod.__init__(self._values_, self._mask_, example=self)
        for key,attr in self.__dict__.items():
            if key.startswith('d_d'):
                pass
            elif isinstance(attr, Qube):
                wod.__dict__[key] = attr.wod
            else:
                wod.__dict__[key] = attr

        wod._derivs_ = {}
        wod._cache_['wod'] = wod
        self._cache_['wod'] = wod

        return wod

    @property
    def cache(self):
        """A dictionary for anything to be cached. The cache is cleared whenever
        the object is modified.
        """

        return self._cache_

    def _clear_cache(self):
        """Clear the cache."""

        self._cache_.clear()

    #===========================================================================
    def _find_corners(self):
        """Update the corner indices such that everything outside this defined
        "hypercube" is masked.
        """

        shape = self._shape_
        lshape = len(shape)
        index0 = lshape * (0,)

        if lshape == 0:
            return None

        if isinstance(self._mask_, (bool, np.bool_)):
            if self._mask_:
                return (index0, index0)
            else:
                return (index0, shape)

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

    @property
    def corners(self):
        """Corners of a "hypercube" that contain all the unmasked array
        elements.
        """

        if not Qube.DISABLE_CACHE and 'corners' in self._cache_:
            return self._cache_['corners']

        corners = self._find_corners()
        self._cache_['corners'] = corners
        return corners

    @staticmethod
    def _slicer_from_corners(corners):
        """A slice object based on corners specified as a tuple of indices.
        """

        slice_objects = []
        for axis in range(len(corners[0])):
            slice_objects.append(slice(corners[0][axis], corners[1][axis]))

        return tuple(slice_objects)

    @staticmethod
    def _shape_from_corners(corners):
        """Array shape based on corner indices."""

        shape = []
        for axis in range(len(corners[0])):
            shape.append(corners[1][axis] - corners[0][axis])

        return tuple(shape)

    @property
    def _slicer(self):
        """A slice object containing all the array elements inside the current
        corners.
        """

        if not Qube.DISABLE_CACHE and 'slicer' in self._cache_:
            return self._cache_['slicer']

        slicer = Qube._slicer_from_corners(self.corners)
        self._cache_['slicer'] = slicer
        return slicer

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
            raise TypeError('derivatives are disallowed in class %s' %
                            type(self).__name__)

        # Make sure the derivative is compatible with the object
        if not isinstance(deriv, Qube):
            raise ValueError('invalid class for derivative "%s": %s' %
                             (key, type(deriv).__name__))

        if self._numer_ != deriv._numer_:
            raise ValueError(('shape mismatch for numerator of derivative ' +
                              '"%s": %s, %s') % (key, str(deriv._numer_),
                                                      str(self._numer_)))

        if self.readonly and (key in self._derivs_) and not override:
            raise ValueError('derivative d_d' + key + ' cannot be replaced ' +
                             'in a read-only object')

        # Prevent recursion, convert to floating point
        deriv = deriv.wod.as_float()

        # Match readonly of parent if necessary
        if self._readonly_ and not deriv._readonly_:
            deriv = deriv.clone(recursive=False).as_readonly()

        # Save in the derivative dictionary and as an attribute
        if deriv._shape_ != self._shape_:
            deriv = deriv.broadcast_into_shape(self._shape_)

        self._derivs_[key] = deriv
        setattr(self, 'd_d' + key, deriv)

        self._cache_.clear()
        return self

    #===========================================================================
    def insert_derivs(self, derivs, override=False):
        """Insert or replace the derivatives in this object from a dictionary.

        You cannot replace the pre-existing values of any derivative in a
        read-only object unless you explicit set override=True. However,
        inserting a new derivative into a read-only object is not prevented.

        Input:
            derivs      the dictionary of derivatives, keyed by their names.

            override    True to allow the value of a pre-existing derivative to
                        be replaced.
        """

        # Check every insert before proceeding with any
        if self.readonly and not override:
            for key in derivs:
                if key in self._derivs_:
                    raise ValueError('derivative d_d' + key + ' cannot be ' +
                                     'replaced in a read-only object')

        # Insert derivatives
        for (key, deriv) in derivs.items():
            self.insert_deriv(key, deriv, override)

    #===========================================================================
    def delete_deriv(self, key, override=False):
        """Delete a single derivative from this object, given the key.

        Derivatives cannot be deleted from a read-only object without explicitly
        setting override=True.

        Input:
            key         the name of the derivative to delete.
            override    True to allow the deleting of derivatives from a
                        read-only object.
        """

        if not override:
            self.require_writable()

        if key in self._derivs_.keys():
            del self._derivs_[key]
            del self.__dict__['d_d' + key]

        self._cache_.clear()

    #===========================================================================
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

        if not override:
            self.require_writable()

        # If something is being preserved...
        if preserve:

            # Delete derivatives not on the list
            for key in self._derivs_.keys():
                if key not in preserve:
                    self.delete_deriv(key, override)

            return

        # Delete all derivatives
        for key in self._derivs_.keys():
            delattr(self, 'd_d' + key)

        self._derivs_ = {}
        self._cache_.clear()
        return

    #===========================================================================
    def without_derivs(self, preserve=None):
        """A shallow copy of this object without derivatives.

        A read-only object remains read-only, and is cached for later use.

        Input:
            preserve    an optional list, tuple or set of the names of
                        derivatives to retain. All others are removed.
        """

        if not self._derivs_:
            return self

        # If something is being preserved...
        if preserve:
            if isinstance(preserve, str):
                preserve = [preserve]

            if not any([p for p in preserve if p in self._derivs_]):
                return self.wod

            # Create a fast copy with derivatives
            obj = self.clone(recursive=True)

            # Delete derivatives not on the list
            deletions = []
            for key in obj._derivs_:
                if key not in preserve:
                    deletions.append(key)

            for key in deletions:
                obj.delete_deriv(key, True)

            return obj

        # Return a fast copy without derivatives
        return self.wod

    #===========================================================================
    def without_deriv(self, key):
        """A shallow copy of this object without a particular derivative.

        A read-only object remains read-only.

        Input:
            key         the key of the derivative to remove.
        """

        if key not in self._derivs_:
            return self

        result = self.clone(recursive=True)
        del result._derivs_[key]

        return result

    #===========================================================================
    def with_deriv(self, key, value, method='insert'):
        """A shallow copy of this object with a derivative inserted or
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

        if method not in ('insert', 'replace', 'add'):
            raise ValueError('invalid with_deriv method: ' + repr(method))

        if key in result._derivs_:
            if method == 'insert':
                raise ValueError('derivative d_d%s already exists' % key)

            if method == 'add':
                value = value + result._derivs_[key]

        result.insert_deriv(key, value)
        return result

    #===========================================================================
    def unique_deriv_name(self, key, *objects):
        """The given name for a deriv if it does not exist in this object or any
        of the given objects; otherwise return a variant that is unique."""

        # Make a list of all the derivative keys
        all_keys = set(self._derivs_.keys())
        for obj in objects:
            if not hasattr(obj, 'derivs'):
                continue
            all_keys |= set(obj._derivs_.keys())

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
            raise TypeError('units are disallowed in class %s' %
                            type(self).__name__)

        if not override:
            self.require_writable()

        units = Units.as_units(units)

        Units.require_compatible(units, self._units_)
        self._units_ = units
        self._cache_.clear()

    #===========================================================================
    def without_units(self, recursive=True):
        """A shallow copy of this object without derivatives.

        A read-only object remains read-only. If recursive is True, derivatives
        are also stripped of their units.
        """

        if self._units_ is None and not self._derivs_:
            return self

        obj = self.clone(recursive)
        obj._units_ = None
        return obj

    #===========================================================================
    def into_units(self, recursive=True):
        """A copy of this object with values scaled to its units.

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
        if recursive or not self._derivs_:
            if self._units_ is None:
                return self
            if self._units_.into_units_factor == 1.:
                return self

        # Construct the new object
        obj = self.clone(recursive)
        obj._set_values_(Units.into_units(self._units_, self._values_))

        # Fill in derivatives if necessary
        if recursive:
            for (key, deriv) in self._derivs_.items():
                obj.insert_deriv(key, deriv.into_units(recursive=False))

        return obj

    #===========================================================================
    def from_units(self, recursive=True):
        """A copy of this object with values scaled to standard units.

        This method undoes the conversion done by into_units().

        Inputs:
            recursive       if True, the derivatives are also converted;
                            otherwise, derivatives are removed.
        """

        # Handle easy cases first
        if recursive or not self._derivs_:
            if self._units_ is None:
                return self
            if self._units_.from_units_factor == 1.:
                return self

        # Construct the new object
        obj = self.clone(recursive)
        obj._set_values_(Units.from_units(self._units_, self._values_))

        # Fill in derivatives if necessary
        if recursive:
            for (key, deriv) in self._derivs_.items():
                obj.insert_deriv(key, deriv.from_units(recursive=False))

        return obj

    #===========================================================================
    def confirm_units(self, units):
        """Raises a ValueError if the units are not compatible with this object.

        Input:
            units       the new units.
        """

        if not Units.can_match(self._units_, units):
            raise ValueError('units are not compatible')

        return self

    ############################################################################
    # Read-only/read-write operations
    ############################################################################

    @staticmethod
    def _array_is_readonly(arg):
        """True if the argument is a read-only NumPy ndarray.

        False means that it is either a writable array or a scalar."""

        if not isinstance(arg, np.ndarray):
            return False

        return (not arg.flags['WRITEABLE'])

    #===========================================================================
    @staticmethod
    def _array_to_readonly(arg):
        """Make the given array read-only. Returns the array."""

        if not isinstance(arg, np.ndarray):
            return arg

        arg.flags['WRITEABLE'] = False
        return arg

    #===========================================================================
    def as_readonly(self, recursive=True):
        """Convert this object to read-only. It is modified in place and
        returned.

        If recursive is False, the derivatives are removed. Otherwise, they are
        also converted to read-only.

        If this object is already read-only, it is returned as is. Otherwise,
        the internal _values_ and _mask_ arrays are modified as necessary.
        Once this happens, the internal arrays will also cease to be writable in
        any other object that shares them.

        Note that as_readonly() cannot be undone. Use copy() to create a
        writable copy of a readonly object.
        """

        # If it is already read-only, return
        if self._readonly_:
            return self

        # Update the value if it is an array
        Qube._array_to_readonly(self._values_)
        Qube._array_to_readonly(self._mask_)
        self._readonly_ = True

        # Update anything cached
        if not Qube.DISABLE_CACHE:
            for key,value in self._cache_.items():
                if isinstance(value, Qube):
                    self._cache_[key] = value.as_readonly(recursive)

        # Update the derivatives
        if recursive:
            for key in self._derivs_:
                self._derivs_[key].as_readonly()

        return self

    #===========================================================================
    def match_readonly(self, arg):
        """Sets the read-only status of this object equal to that of another."""

        if arg._readonly_:
            return self.as_readonly()
        elif self._readonly_:
            raise ValueError('object is read-only')

        return self

    #===========================================================================
    def require_writable(self):
        """Raises a ValueError if the object is read-only.

        Used internally at the beginning of methods that will modify this
        object.
        """

        if self._readonly_:
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
        if self._readonly_ and readonly:
            return obj

        # Copy the values
        if isinstance(self._values_, np.ndarray):
            obj._values_ = self._values_.copy()
        else:
            obj._values_ = self._values_

        # Copy the mask
        if isinstance(self._mask_, np.ndarray):
            obj._mask_ = self._mask_.copy()
        else:
            obj._mask_ = self._mask_

        obj._cache_ = {}

        # Set the read-only state
        if readonly:
            obj.as_readonly()
        else:
            obj._readonly_ = False

        # Make the derivatives read-only if necessary
        if recursive:
            for (key, deriv) in self._derivs_.items():
                obj.insert_deriv(key, deriv.copy(recursive=False,
                                                 readonly=readonly))

        return obj

    #===========================================================================
    # Python-standard copy function
    def __copy__(self):
        """A deep copy of this object unless it is read-only."""

        return self.copy(recursive=True, readonly=False)

    ############################################################################
    # Floats vs. integers vs. booleans
    ############################################################################

    def dtype(self):
        """One of "float", "int", or "bool", depending on the data type."""

        return Qube._dtype(self._values_)

    #===========================================================================
    def is_numeric(self):
        """True if this object contains numbers; False if boolean."""

        return self.dtype() != 'bool'

    #===========================================================================
    def as_numeric(self, recursive=True):
        """A numeric version of this object if it is Boolean.

        Otherwise, this is returned unchanged.
        """

        if self.is_numeric():
            if recursive:
                return self
            return self.wod

        values = self._values_

        # Array case
        if isinstance(values, np.ndarray):
            if self.INTS_OK:
                obj = Qube.__new__(type(self))
                obj.__init__(values.astype(np.int_), self._mask_)
            elif self.FLOATS_OK:
                obj = Qube.__new__(type(self))
                obj.__init__(values.astype(np.float_), self._mask_)
            else:
                obj = Qube.SCALAR_CLASS(values.astype(np.int_), self._mask_)

        # Scalar case
        else:
            if self.INTS_OK:
                obj = Qube.__new__(type(self))
                obj.__init__(int(values), self._mask_)
            elif self.FLOATS_OK:
                obj = Qube.__new__(type(self))
                obj.__init__(float(values), self._mask_)
            else:
                obj = Qube.SCALAR_CLASS(int(values), self._mask_)

        return obj

    #===========================================================================
    def is_float(self):
        """True if this object contains floats; False if ints or booleans."""

        return self.dtype() == 'float'

    #===========================================================================
    def as_float(self, recursive=True):
        """A floating-point version of this object.

        If this object already contains floating-point values, it is returned
        as is. Otherwise, a copy is returned. Derivatives are not modified.
        """

        # If already floating, return as is
        if self.is_float():
            if recursive:
                return self
            return self.wod

        values = self._values_

        # Array case
        if isinstance(values, np.ndarray):
            if self.FLOATS_OK:
                obj = Qube.__new__(type(self))
                obj.__init__(values.astype(np.float_), self._mask_,
                             derivs=self._derivs_, example=self)
            else:
                raise TypeError('data type float is incompatible with class ' +
                                type(self).__name__)

        # Scalar case
        else:
            if self.FLOATS_OK:
                obj = Qube.__new__(type(self))
                obj.__init__(float(values), self._mask_,
                             derivs=self._derivs_, example=self)
            else:
                raise TypeError('data type float is incompatible with class ' +
                                type(self).__name__)

        return obj

    #===========================================================================
    def is_int(self):
        """True if this object contains ints; False if floats or booleans."""

        return self.dtype() == 'int'

    #===========================================================================
    def as_int(self, recursive=True):
        """An integer version of this object. Always round down.

        If this object already contains integers, it is returned as is.
        Otherwise, a copy is returned.
        """

        # If already integers, return as is
        if self.is_int():
            if recursive:
                return self
            return self.wod

        values = self._values_

        # Array case
        if isinstance(values, np.ndarray):
            if self.INTS_OK:
                obj = Qube.__new__(type(self))
                obj.__init__((values//1).astype(np.int_), self._mask_)
            else:
                raise TypeError('data type int is incompatible with class ' +
                                type(self).__name__)

        # Scalar case
        else:
            if self.INTS_OK:
                obj = Qube.__new__(type(self))
                obj.__init__(int(values//1), self._mask_)
            else:
                raise TypeError('data type int is incompatible with class ' +
                                type(self).__name__)

        return obj

    #===========================================================================
    def is_bool(self):
        """True if this object contains booleans; False otherwise."""

        return self.dtype() == 'bool'

    #===========================================================================
    def as_bool(self):
        """A boolean version of this object."""

        # If already boolean, return as is
        if self.is_bool():
            return self

        values = self._values_

        # Array case
        if isinstance(values, np.ndarray):
            if self.BOOLS_OK:
                obj = Qube.__new__(type(self))
                obj.__init__(values.astype(np.bool_), self._mask_)
            else:
                raise TypeError('data type bool is incompatible with class ' +
                                type(self).__name__)

        # Scalar case
        else:
            if self.BOOLS_OK:
                obj = Qube.__new__(type(self))
                obj.__init__(bool(values), self._mask_)
            else:
                raise TypeError('data type bool is incompatible with class ' +
                                type(self).__name__)

        return obj

    #===========================================================================
    def as_builtin(self, masked=None):
        """This object as a Python built-in class (float, int, or bool) if the
        conversion can be done without loss of information."""

        values = self._values_
        if np.shape(values):
            return self
        if self._mask_:
            return self if masked is None else masked

        if self._units_ not in (None, Units.UNITLESS):
            return self

        if isinstance(values, (bool,np.bool_)):
            return bool(values)
        if isinstance(values, numbers.Integral):
            return int(values)
        if isinstance(values, numbers.Real):
            return float(values)

        return self     # This shouldn't happen

    #===========================================================================
    def is_all_masked(self):
        """True if this is entirely masked."""

        return np.all(self._mask_)

    #===========================================================================
    @staticmethod
    def as_one_bool(value):
        """Convert a single value to a bool; leave other values unchanged."""

        if isinstance(value, (bool, np.bool_)):
            return bool(value)

        return value

    #===========================================================================
    @staticmethod
    def is_one_true(value):
        """True if the value is a single boolean True."""

        if isinstance(value, (bool, np.bool_)):
            return bool(value)

        return False

    #===========================================================================
    @staticmethod
    def is_one_false(value):
        """True if the value is a single boolean False."""

        if isinstance(value, (bool, np.bool_)):
            return not bool(value)

        return False

    ############################################################################
    # Subclass operations
    ############################################################################

    @staticmethod
    def is_real_number(arg):
        """True if arg is of a Python numeric or NumPy numeric scalar."""

        return isinstance(arg, numbers.Real)

    #===========================================================================
    def masked_single(self, recursive=True):
        """An object of this subclass containing one masked value."""

        if not self._rank_:
            new_value = self._default_
        else:
            new_value = self._default_.copy()

        obj = Qube.__new__(type(self))
        obj.__init__(new_value, True, example=self)

        if recursive and self._derivs_:
            for (key, value) in self._derivs_.items():
                obj.insert_deriv(key, value.masked_single(recursive=False))

        obj.as_readonly()
        return obj

    #===========================================================================
    def as_this_type(self, arg, recursive=True, coerce=True):
        """The argument converted to this class and data type.

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

        # If the classes already match, we might return the argument as is
        if type(arg) == type(self):
            obj = arg
        else:
            obj = None

        # Initialize the new values and mask; track other attributes
        if not isinstance(arg, Qube):
            arg = Qube(arg, example=self)

        if arg._nrank_ != self._nrank_:
            raise ValueError('item shape mismatch')

        new_vals = arg._values_
        new_mask = arg._mask_
        new_units = arg._units_
        has_derivs = bool(arg._derivs_)
        is_readonly = arg._readonly_

        # Convert the value types if necessary
        changed = False
        if coerce:
            casted = Qube._casted_to_dtype(new_vals, Qube._dtype(self._values_))
            changed = casted is not new_vals
            new_vals = casted

        # Convert the units if necessary
        if new_units and not self.UNITS_OK:
            new_units = None
            changed = True

        # Validate derivs
        if has_derivs and not self.DERIVS_OK:
            changed = True
        if has_derivs and not recursive:
            changed = True

        # Construct the new object if necessary
        if changed or obj is None:
            obj = Qube.__new__(type(self))
            obj.__init__(new_vals, new_mask, units=new_units,
                         example=self, drank=arg._drank_)
            is_readonly = False

        # Update the derivatives if necessary
        if recursive and has_derivs:
            derivs_changed = False
            new_derivs = {}
            for (key, deriv) in arg._derivs_.items():
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
    def cast(self, classes):
        """A shallow copy of this object casted to another subclass.

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
            if cls is type(self):
                return self

            # Exclude the class if it is incompatible
            if cls.NUMER is not None and cls.NUMER != self._numer_:
                continue
            if cls.NRANK is not None and cls.NRANK != self._nrank_:
                continue

            # Construct the new object
            obj = Qube.__new__(cls)
            obj.__init__(self._values_, self._mask_, derivs=self._derivs_,
                         example=self)
            return obj

        # If no suitable class was found, return this object unmodified
        return self

    #===========================================================================
    def count_masked(self):
        """The number of masked items in this object."""

        if isinstance(self._mask_, (bool, np.bool_)):
            if self._mask_:
                return self._size_
            else:
                return 0

        return np.count_nonzero(self._mask_)

    #===========================================================================
    def masked(self):
        """The number of masked items in this object. DEPRECATED NAME;
        use count_masked()."""

        return self.count_masked()

    #===========================================================================
    def count_unmasked(self):
        """The number of unmasked items in this object."""

        if isinstance(self._mask_, (bool, np.bool_)):
            if self._mask_:
                return 0
            else:
                return self._size_

        return self._size_ - np.count_nonzero(self._mask_)

    #===========================================================================
    def unmasked(self):
        """The number of unmasked items in this object. DEPRECATED NAME;
        use count_unmasked()"""

        return self.count_unmasked()

    #===========================================================================
    def without_mask(self, recursive=True):
        """A shallow copy of this object without its mask."""

        obj = self.clone()

        if Qube.is_one_false(self._mask_):
            return obj

        obj._set_mask_(False)

        if recursive:
            for (key,deriv) in self._derivs_.items():
                obj.insert_deriv(key, deriv.without_mask())

        return obj

    #===========================================================================
    def as_all_masked(self, recursive=True):
        """A shallow copy of this object with everything masked."""

        obj = self.clone()

        if Qube.is_one_true(self._mask_):
            return obj

        obj._set_mask_(True)

        if recursive:
            for (key,deriv) in self._derivs_.items():
                obj.insert_deriv(key, deriv.as_all_masked(False))

        return obj

    #===========================================================================
    def all_masked(self, recursive=True):
        """A shallow copy of this object with everything masked.
        DEPRECATED NAME; use as_all_masked()"""

        return self.as_all_masked(recursive)

    #===========================================================================
    def as_one_masked(self, recursive=True):
        """This object reduced to shape (1,) and masked."""

        return self.flatten()[0].as_all_masked()

    #===========================================================================
    def remask(self, mask, recursive=True, check=True):
        """A shallow copy of this object with a replaced mask.

        This is much quicker than masked_where(), for cases where only the mask
        is changing.

        Input:
            mask        the new mask to be applied to the object.
            recursive   True to apply the same mask to any derivatives.
            check       True to check a mask array for True values, and
                        replace it with a single scalar False if it can.
        """

        mask = Qube._suitable_mask(mask, self._shape_, check=check)

        # Construct the new object
        obj = self.clone(recursive=False)
        obj._set_mask_(mask)

        if recursive:
            for (key, deriv) in self._derivs_.items():
                obj.insert_deriv(key, deriv.remask(mask, recursive=False,
                                                         check=False))

        return obj

    #===========================================================================
    def remask_or(self, mask, recursive=True, check=True):
        """A shallow copy of this object, in which the current mask is "or-ed"
        with the given mask.

        This is much quicker than masked_where(), for cases where only the mask
        is changing.

        Input:
            mask        the new mask to be "or-ed" with the object's current
                        mask.
            recursive   True to apply the same mask to any derivatives.
            check       True to check a mask array for True values, and
                        replace it with a single scalar False if it can.
        """

        mask = Qube._suitable_mask(mask, self._shape_, check=check)

        # Construct the new object
        obj = self.clone(recursive=False)
        obj._set_mask_(Qube.or_(self._mask_, mask))

        if recursive:
            for (key,deriv) in self._derivs_.items():
                obj.insert_deriv(key, deriv.remask(mask, recursive=False,
                                                         check=False))

        return obj

    #===========================================================================
    def expand_mask(self, recursive=True):
        """A shallow copy where a single mask value of True or False is
        converted to an array.

        If the object's mask is already an array, it is returned unchanged.
        """

        if np.shape(self._mask_) and not (recursive and self._derivs_):
            return self

        # Clone the object only if necessary
        obj = None
        if np.isscalar(self._mask_):
            obj = self.clone(recursive=True)
            if obj._mask_:
                obj._set_mask_(np.ones(self._shape_, dtype=np.bool_))
            else:
                obj._set_mask_(np.zeros(self._shape_, dtype=np.bool_))

        # Clone any derivs only if necessary
        new_derivs = {}
        if recursive:
            for (key, deriv) in self._derivs_.items():
                mask_before = deriv._mask_
                new_deriv = deriv.expand_mask(recursive=False)
                if mask_before is not new_deriv._mask_:
                    new_derivs[key] = new_deriv

        # If nothing has changed, return self
        if obj is None and not new_derivs:
            return self

        # Return the modified object
        if obj is None:
            obj = self.clone(recursive=True)

        for key, deriv in new_derivs.items():
            obj.insert_deriv(key, deriv, override=True)

        return obj

    #===========================================================================
    def collapse_mask(self, recursive=True):
        """A shallow copy where a mask entirely containing either True or False
        is converted to a single boolean.
        """

        if np.isscalar(self._mask_) and not (recursive and self._derivs_):
            return self

        # Clone the object only if necessary
        obj = None
        if np.shape(self._mask_):
            if not np.any(self._mask_):
                obj = self.clone(recursive=True)
                obj._set_mask_(False)
            elif np.all(self._mask_):
                obj = self.clone(recursive=True)
                obj._set_mask_(True)

        # Clone any derivs only if necessary
        new_derivs = {}
        if recursive:
            for (key, deriv) in self._derivs_.items():
                mask_before = deriv._mask_
                new_deriv = deriv.collapse_mask(recursive=False)
                if mask_before is not new_deriv._mask_:
                    new_derivs[key] = new_deriv

        # If nothing has changed, return self
        if obj is None and not new_derivs:
            return self

        # Return the modified object
        if obj is None:
            obj = self.clone(recursive=True)

        for key, deriv in new_derivs.items():
            obj.insert_deriv(deriv, override=True)

        return obj

    #===========================================================================
    def as_all_constant(self, constant=None, recursive=True):
        """A shallow, read-only copy of this object with constant values.

        Derivatives are all set to zero. The mask is unchanged.
        """

        if constant is None:
            constant = self.zero()

        constant = self.as_this_type(constant, recursive=False)

        obj = self.clone(recursive=False)
        obj._set_values_(Qube.broadcast(constant, obj)[0]._values_)
        obj.as_readonly()

        if recursive:
            for (key,deriv) in self._derivs_.items():
                obj.insert_deriv(key, deriv.all_constant(recursive=False))

        return obj

    #===========================================================================
    def all_constant(self, constant=None, recursive=True):
        """A shallow, read-only copy of this object with constant values.
        DEPRECATED NAME; use as_all_constant().

        Derivatives are all set to zero. The mask is unchanged.
        """

        return self.as_all_constant(constant, recursive=recursive)

    #===========================================================================
    def as_size_zero(self, axis=0, recursive=True):
        """A shallow, read-only copy of this object with size zero.

        Use axis = 0 to give the first axis length zero; -1 for the last axis;
        axis = None for an object collapsed to shape (0,).
        """

        obj = Qube.__new__(type(self))

        if self._shape_ == ():
            new_values = np.array([self._values_])[:0]
            new_mask = np.array([self._mask_])[:0]
        elif axis is None:
            new_values = self._values_.ravel()[:0]
            new_mask = np.asarray(self._mask_).ravel()[:0]
        else:
            if axis == 0:
                indx = slice(0,0)
            else:
                indx = (Ellipsis, slice(0,0))

            new_values = self._values_[indx]

            if np.shape(self._mask_):
                new_mask = self._mask_[indx]
            else:
                new_mask = np.array([self._mask_])[indx]

        obj.__init__(new_values, new_mask, example=self)

        if recursive:
            for (key,deriv) in self._derivs_.items():
                obj.insert_deriv(key, deriv.as_size_zero(axis=axis,
                                                         recursive=False))

        return obj

    #===========================================================================
    def as_mask_where_nonzero(self):
        """A boolean scalar or NumPy array where values are nonzero and
        unmasked."""

        return Qube._as_mask(self, masked_value=False)

    #===========================================================================
    def as_mask_where_zero(self):
        """A boolean scalar or NumPy array where values are zero and unmasked.
        """

        return Qube._as_mask(self, invert=True, masked_value=False)

    #===========================================================================
    def as_mask_where_nonzero_or_masked(self):
        """A boolean scalar or NumPy array where values are nonzero or masked.
        """

        return Qube._as_mask(self, masked_value=True)

    #===========================================================================
    def as_mask_where_zero_or_masked(self):
        """A boolean scalar or NumPy array where values are zero or masked."""

        return Qube._as_mask(self, invert=True, masked_value=True)

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

    #===========================================================================
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

        # Indicate the denominator shape if necessary
        if self._denom_ != ():
            suffix += ['denom=' + str(self._denom_)]

        # Masked objects have a suffix ', mask'
        is_masked = np.any(self._mask_)
        if is_masked:
            suffix += ['mask']

        # Objects with units include the units in the suffix
        if self._units_ is not None and self._units_ != Units.UNITLESS:
            suffix += [str(self._units_)]

        # Objects with derivatives include a list of the names
        if self._derivs_:
            keys = list(self._derivs_.keys())
            keys.sort()
            for key in keys:
                suffix += ['d_d' + key]

        # Generate the value string
        scaled = self.into_units(recursive=False)   # apply the units
        if np.isscalar(scaled._values_):
            if is_masked:
                string = '--'
            else:
                string = str(scaled._values_)
        elif is_masked:
            string = str(scaled.mvals)[1:-1]
        else:
            string = str(scaled._values_)[1:-1]

        # Add an extra set of brackets around derivatives
        if self._denom_:
            string = '[' + string + ']'

        # Concatenate the results
        if len(suffix) == 0:
            suffix = ''
        else:
            suffix = '; ' + ', '.join(suffix)

        return type(self).__name__ + '(' + string + suffix + ')'

    ############################################################################
    # Unary operators
    ############################################################################

    def __pos__(self):
        return self.clone(recursive=True)

    #===========================================================================
    def __neg__(self, recursive=True):

        # Construct a copy with negative values
        obj = self.clone(recursive=False)
        obj._set_values_(-self._values_)

        # Fill in the negative derivatives
        if recursive and self._derivs_:
            for (key, deriv) in self._derivs_.items():
                obj.insert_deriv(key, -deriv)

        return obj

    #===========================================================================
    def __abs__(self, recursive=True):

        # Check rank
        if self._nrank_ != 0:
            Qube._raise_unsupported_op('abs', self)

        # Construct a copy with absolute values
        obj = self.clone(recursive=False)
        obj._set_values_(np.abs(self._values_))

        # Fill in the derivatives, multiplied by sign(self)
        if recursive and self._derivs_:
            sign = self.wod.sign()
            for (key, deriv) in self._derivs_.items():
                obj.insert_deriv(key, deriv * sign)

        return obj

    #===========================================================================
    def __len__(self):

        if len(self._shape_) > 0:
            return self._shape_[0]
        else:
            raise TypeError('len of unsized object')

    ############################################################################
    # Addition
    ############################################################################

    # Default method for left addition, element by element
    def __add__(self, arg, recursive=True):

        # Handle a simple right-hand value...
        if self._rank_ == 0 and isinstance(arg, numbers.Real):
            obj = self.clone(recursive=recursive, retain_cache=True)
            obj._set_values_(self._values_ + arg, retain_cache=True)
            return obj

        # Convert arg to another Qube if necessary
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = self.as_this_type(arg, coerce=False)
            except (ValueError, TypeError):
                Qube._raise_unsupported_op('+', self, original_arg)

        # Verify compatibility
        if not Units.can_match(self._units_, arg._units_):
            raise ValueError('operand units are incompatible: %s, %s' %
                             (str(self._units_), str(arg._units_)))

        if self._numer_ != arg._numer_:
            if type(self) != type(arg):
                Qube._raise_unsupported_op('+', self, original_arg)

            raise ValueError('item shapes are incompatible: %s, %s' %
                             (str(self._numer_), str(arg._numer_)))

        if self._denom_ != arg._denom_:
            raise ValueError('denominator shapes are incompatible: %s, %s' %
                             (str(self._denom_), str(arg._denom_)))

        # Construct the result
        obj = Qube.__new__(type(self))
        obj.__init__(self._values_ + arg._values_,
                     Qube.or_(self._mask_, arg._mask_),
                     units = self._units_ or arg._units_,
                     example=self)

        if recursive:
            obj.insert_derivs(obj._add_derivs(self, arg))

        return obj

    #===========================================================================
    # Default method for right addition, element by element
    def __radd__(self, arg, recursive=True):
        return self.__add__(arg, recursive=recursive)

    #===========================================================================
    # Default method for in-place addition, element by element
    def __iadd__(self, arg):
        self.require_writable()

        # Handle a simple right-hand value...
        if self._rank_ == 0 and isinstance(arg, (numbers.Real, np.ndarray)):
            self._values_ += arg
            self._new_values_()
            return self

        # Convert arg to another Qube if necessary
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = self.as_this_type(arg, coerce=False)
            except (ValueError, TypeError):
                Qube._raise_unsupported_op('+=', self, original_arg)

        # Verify compatibility
        if not Units.can_match(self._units_, arg._units_):
            raise ValueError('operand units are incompatible: %s, %s' %
                             (str(self._units_), str(arg._units_)))

        if self._numer_ != arg._numer_:
            if type(self) != type(arg):
                Qube._raise_unsupported_op('+=', self, original_arg)

            raise ValueError('item shapes are incompatible: %s, %s' %
                             (str(self._numer_), str(arg._numer_)))

        if self._denom_ != arg._denom_:
            raise ValueError('denominator shapes are incompatible: %s, %s' %
                             (str(self._denom_), str(arg._denom_)))

        # Perform the operation
        if self.is_int() and not arg.is_int():
            raise TypeError('"+=" operation returns non-integer result')

        new_derivs = self._add_derivs(self,arg) # if this raises exception, stop
        self._values_ += arg._values_           # on exception, no harm done
        self._mask_ = Qube.or_(self._mask_, arg._mask_)
        self._units_ = self._units_ or arg._units_
        self.insert_derivs(new_derivs)

        self._cache_.clear()
        return self

    #===========================================================================
    def _add_derivs(self, arg1, arg2):
        """Dictionary of added derivatives."""

        set1 = set(arg1._derivs_.keys())
        set2 = set(arg2._derivs_.keys())
        set12 = set1 & set2
        set1 -= set12
        set2 -= set12

        new_derivs = {}
        for key in set12:
            new_derivs[key] = arg1._derivs_[key] + arg2._derivs_[key]
        for key in set1:
            new_derivs[key] = arg1._derivs_[key]
        for key in set2:
            new_derivs[key] = arg2._derivs_[key]

        return new_derivs

    ############################################################################
    # Subtraction
    ############################################################################

    # Default method for left subtraction, element by element
    def __sub__(self, arg, recursive=True):

        # Handle a simple right-hand value...
        if self._rank_ == 0 and isinstance(arg, numbers.Real):
            obj = self.clone(recursive=recursive, retain_cache=True)
            obj._set_values_(self._values_ - arg, retain_cache=True)
            return obj

        # Convert arg to the same subclass and try again
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = self.as_this_type(arg, coerce=False)
            except (ValueError, TypeError):
                Qube._raise_unsupported_op('-', self, original_arg)

        # Verify compatibility
        if not Units.can_match(self._units_, arg._units_):
            raise ValueError('operand units are incompatible: %s, %s' %
                             (str(self._units_), str(arg._units_)))

        if self._numer_ != arg._numer_:
            if type(self) != type(arg):
                Qube._raise_unsupported_op('-', self, original_arg)

            raise ValueError('item shapes are incompatible: %s, %s' %
                             (str(self._numer_), str(arg._numer_)))

        if self._denom_ != arg._denom_:
            raise ValueError('denominator shapes are incompatible: %s, %s' %
                             (str(self._denom_), str(arg._denom_)))

        # Construct the result
        obj = Qube.__new__(type(self))
        obj.__init__(self._values_ - arg._values_,
                     Qube.or_(self._mask_, arg._mask_),
                     units = self._units_ or arg._units_,
                     example=self)

        if recursive:
            obj.insert_derivs(obj._sub_derivs(self, arg))

        return obj

    #===========================================================================
    # Default method for right subtraction, element by element
    def __rsub__(self, arg, recursive=True):

        # Convert arg to the same subclass and try again
        if not isinstance(arg, Qube):
            arg = self.as_this_type(arg, coerce=False)
            return arg.__sub__(self, recursive=recursive)

    #===========================================================================
    # In-place subtraction
    def __isub__(self, arg):
        self.require_writable()

        # Handle a simple right-hand value...
        if self._rank_ == 0 and isinstance(arg, (numbers.Real, np.ndarray)):
            self._values_ -= arg
            self._new_values_()
            return self

        # Convert arg to another Qube if necessary
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = self.as_this_type(arg, coerce=False)
            except (ValueError, TypeError):
                Qube._raise_unsupported_op('-=', self, original_arg)

        # Verify compatibility
        if not Units.can_match(self._units_, arg._units_):
            raise ValueError('operand units are incompatible: %s, %s' %
                             (str(self._units_), str(arg._units_)))

        if self._numer_ != arg._numer_:
            if type(self) != type(arg):
                Qube._raise_unsupported_op('-=', self, original_arg)

            raise ValueError('item shapes are incompatible: %s, %s' %
                             (str(self._numer_), str(arg._numer_)))

        if self._denom_ != arg._denom_:
            raise ValueError('denominator shapes are incompatible: %s, %s' %
                             (str(self._denom_), str(arg._denom_)))

        # Perform the operation
        if self.is_int() and not arg.is_int():
            raise TypeError('"-=" operation returns non-integer result')

        new_derivs = self._sub_derivs(self,arg) # if this raises exception, stop
        self._values_ -= arg._values_           # on exception, no harm done
        self._mask_ = Qube.or_(self._mask_, arg._mask_)
        self._units_ = self._units_ or arg._units_
        self.insert_derivs(new_derivs)

        self._cache_.clear()
        return self

    #===========================================================================
    def _sub_derivs(self, arg1, arg2):
        """Dictionary of subtracted derivatives."""

        set1 = set(arg1._derivs_.keys())
        set2 = set(arg2._derivs_.keys())
        set12 = set1 & set2
        set1 -= set12
        set2 -= set12

        new_derivs = {}
        for key in set12:
            new_derivs[key] = arg1._derivs_[key] - arg2._derivs_[key]
        for key in set1:
            new_derivs[key] = arg1._derivs_[key]
        for key in set2:
            new_derivs[key] = -arg2._derivs_[key]

        return new_derivs

    ############################################################################
    # Multiplication
    ############################################################################

    # Generic left multiplication
    def __mul__(self, arg, recursive=True):

        # Handle multiplication by a number
        if Qube.is_real_number(arg):
            return self._mul_by_number(arg, recursive=recursive)

        # Convert arg to a Scalar if necessary
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = Qube.SCALAR_CLASS.as_scalar(arg)
            except (ValueError, TypeError):
                Qube._raise_unsupported_op('*', self, original_arg)

        # Check denominators
        if self._drank_ and arg._drank_:
            raise ValueError('dual operand denominators for "*": %s, %s' %
                             (str(self._denom_), str(arg._denom_)))

        # Multiply by scalar...
        if arg._nrank_ == 0:
            try:
                return self._mul_by_scalar(arg, recursive=recursive)

            # Revise the exception if the arg was modified
            except (ValueError, TypeError):
                if arg is not original_arg:
                    Qube._raise_unsupported_op('*', self, original_arg)
                raise

        # Swap and try again
        if self._nrank_ == 0:
            return arg._mul_by_scalar(self, recursive=recursive)

        # Multiply by matrix...
        if self._nrank_ == 2 and arg._nrank_ in (1,2):
            return Qube.dot(self, arg, -1, 0, (type(arg), type(self)),
                            recursive=recursive)

        # Give up
        Qube._raise_unsupported_op('*', self, original_arg)

    #===========================================================================
    # Generic right multiplication
    def __rmul__(self, arg, recursive=True):

        # Handle multiplication by a number
        if Qube.is_real_number(arg):
            return self._mul_by_number(arg, recursive=recursive)

        # Convert arg to a Scalar and try again
        original_arg = arg
        try:
            arg = Qube.SCALAR_CLASS.as_scalar(arg)
            return self._mul_by_scalar(arg, recursive=recursive)

        # Revise the exception if the arg was modified
        except (ValueError, TypeError):
            if arg is not original_arg:
                Qube._raise_unsupported_op('*', original_arg, self)
            raise

    #===========================================================================
    # Generic in-place multiplication
    def __imul__(self, arg):
        self.require_writable()

        # If a number...
        if isinstance(arg, numbers.Real):
            self._values_ *= arg
            self._new_values_()
            for key, deriv in self._derivs_.items():
                deriv._values_ *= arg
                deriv._new_values_()
            return self

        # Convert arg to a Scalar if necessary
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = Qube.SCALAR_CLASS.as_scalar(arg)
            except (ValueError, TypeError):
                Qube._raise_unsupported_op('*=', self, original_arg)

        # Scalar case
        if arg._rank_ == 0:

            # Align axes
            arg_values = arg._values_
            if self._rank_ and np.shape(arg_values):
                arg_values = arg_values.reshape(np.shape(arg_values) +
                                                self._rank_ * (1,))

            # Multiply...
            if self.is_int() and not arg.is_int():
                raise TypeError('"*=" operation returns non-integer result')

            new_derivs = self._mul_derivs(arg)  # if this raises exception, stop
            self._values_ *= arg_values         # on exception, object unchanged
            self._mask_ = Qube.or_(self._mask_, arg._mask_)
            self._units_ = Units.mul_units(self._units_, arg._units_)
            self.insert_derivs(new_derivs)

            self._cache_.clear()
            return self

        # Matrix multiply case
        if self._nrank_ == 2 and arg._nrank_ == 2 and arg._drank_ == 0:
            result = Qube.dot(self, arg, -1, 0, type(self), recursive=True)
            self._set_values_(result._values_, result._mask_)
            self.insert_derivs(result._derivs_)
            return self

        # Nothing else is implemented
        Qube._raise_unsupported_op('*=', self, original_arg)

    #===========================================================================
    def _mul_by_number(self, arg, recursive=True):
        """Internal multiply op when the arg is a Python scalar."""

        obj = self.clone(recursive=False, retain_cache=True)
        obj._set_values_(self._values_ * arg, retain_cache=True)

        if recursive and self._derivs_:
            for (key, deriv) in self._derivs_.items():
                obj.insert_deriv(key, deriv._mul_by_number(arg, False))

        return obj

    #===========================================================================
    def _mul_by_scalar(self, arg, recursive=True):
        """Internal multiply op when the arg is a Qube with nrank == 0 and no
        more than one object has a denominator."""

        # Align axes
        self_values = self._values_
        self_shape = np.shape(self_values)
        if arg._drank_ > 0 and self_shape != ():
            self_values = self_values.reshape(self_shape + arg._drank_ * (1,))

        arg_values = arg._values_
        arg_shape = (arg._shape_ + self._rank_ * (1,) + arg._denom_)
        if np.shape(arg_values) not in ((), arg_shape):
            arg_values = arg_values.reshape(arg_shape)

        # Construct object
        obj = Qube.__new__(type(self))
        obj.__init__(self_values * arg_values,
                     Qube.or_(self._mask_, arg._mask_),
                     units = Units.mul_units(self._units_, arg._units_),
                     drank = max(self._drank_, arg._drank_),
                     example = self)

        obj.insert_derivs(self._mul_derivs(arg))
        return obj

    #===========================================================================
    def _mul_derivs(self, arg):
        """Dictionary of multiplied derivatives."""

        new_derivs = {}

        if self._derivs_:
            arg_wod = arg.wod
            for (key, self_deriv) in self._derivs_.items():
                new_derivs[key] = self_deriv * arg_wod

        if arg._derivs_:
            self_wod = self.wod
            for (key, arg_deriv) in arg._derivs_.items():
                if key in new_derivs:
                    new_derivs[key] = new_derivs[key] + self_wod * arg_deriv
                else:
                    new_derivs[key] = self_wod * arg_deriv

        return new_derivs

    ############################################################################
    # Division
    ############################################################################

    def __div__(self, arg, recursive=True):
        return self.__truediv__(arg, recursive=recursive)

    def __rdiv__(self, arg, recursive=True):
        return self.__rtruediv__(arg)

    def __idiv__(self, arg, recursive=True):
        return self.__itruediv__(arg)

    #===========================================================================
    # Generic left true division
    def __truediv__(self, arg, recursive=True):
        """Cases of divide-by-zero are masked."""

        # Handle division by a number
        if Qube.is_real_number(arg):
            return self._div_by_number(arg, recursive=recursive)

        # Convert arg to a Scalar if necessary
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = Qube.SCALAR_CLASS.as_scalar(arg)
            except (ValueError, TypeError):
                Qube._raise_unsupported_op('/', self, original_arg)

        # Check right denominator
        if arg._drank_ > 0:
            raise ValueError('right operand denominator for "/": %s' %
                             str(arg._denom_))

        # Divide by scalar...
        if arg._nrank_ == 0:
            try:
                return self._div_by_scalar(arg, recursive=recursive)

            # Revise the exception if the arg was modified
            except (ValueError, TypeError):
                if arg is not original_arg:
                    Qube._raise_unsupported_op('/', self, original_arg)
                raise

        # Swap and multiply by reciprocal...
        if self._nrank_ == 0:
            return self.reciprocal(recursive)._mul_by_scalar(arg, recursive)

        # Matrix / matrix is multiply by inverse matrix
        if self._rank_ == 2 and arg._rank_ == 2:
            return self.__mul__(arg.reciprocal(recursive))

        # Give up
        Qube._raise_unsupported_op('/', self, original_arg)

    #===========================================================================
    # Generic right division
    def __rtruediv__(self, arg, recursive=True):

        # Handle right division by a number
        if Qube.is_real_number(arg):
            return self.reciprocal(recursive).__mul__(arg, recursive=recursive)

        # Convert arg to a Scalar and try again
        original_arg = arg
        try:
            arg = Qube.SCALAR_CLASS.as_scalar(arg)
            return arg.__truediv__(self, recursive=recursive)

        # Revise the exception if the arg was modified
        except (ValueError, TypeError):
            if arg is not original_arg:
                Qube._raise_unsupported_op('/', original_arg, self)
            raise

    #===========================================================================
    # Generic in-place division
    def __itruediv__(self, arg):

        if not self.is_float():
            raise TypeError('"/=" operation returns non-integer result')

        self.require_writable()

        # If a number...
        if isinstance(arg, numbers.Real) and arg != 0:
            self._values_ /= arg
            self._new_values_()
            for key, deriv in self._derivs_.items():
                deriv._values_ /= arg
                deriv._new_values_()
            return self

        # Convert arg to a Scalar if necessary
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = Qube.SCALAR_CLASS.as_scalar(arg)
            except (ValueError, TypeError):
                Qube._raise_unsupported_op('/=', self, original_arg)

        # In-place multiply by the reciprocal
        try:
            self.__imul__(arg.reciprocal())

        # Revise the exception if the arg was modified
        except (ValueError, TypeError):
            if arg is not original_arg:
                Qube._raise_unsupported_op('/=', self, original_arg)
            raise

        return self

    #===========================================================================
    def _div_by_number(self, arg, recursive=True):
        """Internal division op when the arg is a Python scalar."""

        obj = self.clone(recursive=False, retain_cache=True)

        # Mask out zeros
        if arg == 0:
            obj._set_mask_(True)
        else:
            obj._set_values_(self._values_ / arg, retain_cache=True)

        if recursive and self._derivs_:
            for (key, deriv) in self._derivs_.items():
                obj.insert_deriv(key, deriv._div_by_number(arg, False))

        return obj

    #===========================================================================
    def _div_by_scalar(self, arg, recursive):
        """Internal division op when the arg is a Qube with rank == 0."""

        # Mask out zeros
        arg = arg.mask_where_eq(0.,1.)

        # Align axes
        arg_values = arg._values_
        if np.shape(arg_values) and self._rank_:
            arg_values = arg_values.reshape(arg.shape + self._rank_ * (1,))

        # Construct object
        obj = Qube.__new__(type(self))
        obj.__init__(self._values_ / arg_values,
                     Qube.or_(self._mask_, arg._mask_),
                     units = Units.div_units(self._units_, arg._units_),
                     example = self)

        if recursive:
            obj.insert_derivs(self._div_derivs(arg, nozeros=True))

        return obj

    #===========================================================================
    def _div_derivs(self, arg, nozeros=False):
        """Dictionary of divided derivatives.

        if nozeros is True, the arg is assumed not to contain any zeros, so
        divide-by-zero errors are not checked.
        """

        new_derivs = {}

        if not self._derivs_ and not arg._derivs_:
            return new_derivs

        if not nozeros:
            arg = arg.mask_where_eq(0., 1.)

        arg_wod_inv = arg.wod.reciprocal(nozeros=True)

        for (key, self_deriv) in self._derivs_.items():
            new_derivs[key] = self_deriv * arg_wod_inv

        if arg._derivs_:
            self_wod = self.wod
            for (key, arg_deriv) in arg._derivs_.items():
                term = self_wod * (arg_deriv * arg_wod_inv*arg_wod_inv)
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

        # Convert arg to a Scalar if necessary
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = Qube.SCALAR_CLASS.as_scalar(arg)
            except (ValueError, TypeError):
                Qube._raise_unsupported_op('//', self, original_arg)

        # Check right denominator
        if arg._drank_ > 0:
            raise ValueError('right operand denominator for "//": %s' %
                             str(arg._denom_))

        # Floor divide by scalar...
        if arg._nrank_ == 0:
            try:
                return self._floordiv_by_scalar(arg)

            # Revise the exception if the arg was modified
            except (ValueError, TypeError):
                if arg is not original_arg:
                    Qube._raise_unsupported_op('//', original_arg, self)
                raise

        # Give up
        Qube._raise_unsupported_op('//', self, original_arg)

    #===========================================================================
    # Generic right floor division
    def __rfloordiv__(self, arg):

        # Convert arg to a Scalar and try again
        original_arg = arg
        try:
            arg = Qube.SCALAR_CLASS.as_scalar(arg)
            return arg.__floordiv__(self)

        # Revise the exception if the arg was modified
        except (ValueError, TypeError):
            if arg is not original_arg:
                Qube._raise_unsupported_op('//', original_arg, self)
            raise

    #===========================================================================
    # Generic in-place floor division
    def __ifloordiv__(self, arg):
        self.require_writable()

        # If a number...
        if isinstance(arg, numbers.Real) and arg != 0:
            self._values_ //= arg
            self._new_values_()
            self.delete_derivs()
            return self

        # Convert arg to a Scalar if necessary
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = Qube.SCALAR_CLASS.as_scalar(arg)
            except (ValueError, TypeError):
                Qube._raise_unsupported_op('//=', self, original_arg)

        # Handle floor division by a scalar
        if arg._rank_ == 0:
            divisor = arg.mask_where_eq(0, 1)
            div_values = divisor._values_

            # Align axes
            if self._rank_:
                div_values = np.reshape(div_values, np.shape(div_values) +
                                                    self._rank_ * (1,))
            self._values_ //= div_values
            self._mask_ = self._mask_ | divisor._mask_
            self._units_ = Units.div_units(self._units_, arg._units_)
            self.delete_derivs()

            self._cache_.clear()
            return self

        # Nothing else is implemented
        Qube._raise_unsupported_op('//=', self, original_arg)

    #===========================================================================
    def _floordiv_by_number(self, arg):
        """Internal floor division op when the arg is a Python scalar."""

        obj = self.clone(recursive=False, retain_cache=True)

        if arg == 0:
            obj._set_mask_(True)
        else:
            obj._set_values_(self._values_ // arg, retain_cache=True)

        return obj

    #===========================================================================
    def _floordiv_by_scalar(self, arg):
        """Internal floor division op when the arg is a Qube with nrank == 0.

        The arg cannot have a denominator.
        """

        # Mask out zeros
        arg = arg.mask_where_eq(0,1)

        # Align axes
        arg_values = arg._values_
        if np.shape(arg_values) and self._rank_:
            arg_values = arg_values.reshape(arg.shape + self._rank_ * (1,))

        # Construct object
        obj = Qube.__new__(type(self))
        obj.__init__(self._values_ // arg_values,
                     self._mask_ | arg._mask_,
                     units = Units.div_units(self._units_, arg._units_),
                     example = self)
        return obj

    ############################################################################
    # Modulus operators (with no support for derivatives)
    ############################################################################

    # Generic left modulus
    def __mod__(self, arg, recursive=True):
        """Cases of divide-by-zero become masked. Derivatives in the numerator
        are supported, but not in the denominator."""

        # Handle modulus by a number
        if Qube.is_real_number(arg):
            return self._mod_by_number(arg, recursive=recursive)

        # Convert arg to a Scalar if necessary
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = Qube.SCALAR_CLASS.as_scalar(arg)
            except (ValueError, TypeError):
                Qube._raise_unsupported_op('%', self, original_arg)

        # Check right denominator
        if arg._drank_ > 0:
            raise ValueError('right operand denominator for "%%": %s' %
                             str(arg._denom_))

        # Modulus by scalar...
        if arg._nrank_ == 0:
            try:
                return self._mod_by_scalar(arg, recursive=recursive)

            # Revise the exception if the arg was modified
            except (ValueError, TypeError):
                if arg is not original_arg:
                    Qube._raise_unsupported_op('%', self, original_arg)
                raise

        # Give up
        Qube._raise_unsupported_op('%', self, original_arg)

    #===========================================================================
    # Generic right modulus
    def __rmod__(self, arg):

        # Convert arg to a Scalar and try again
        original_arg = arg
        try:
            arg = Qube.SCALAR_CLASS.as_scalar(arg)
            return arg.__mod__(self)

        # Revise the exception if the arg was modified
        except (ValueError, TypeError):
            if arg is not original_arg:
                Qube._raise_unsupported_op('%', original_arg, self)
            raise

    #===========================================================================
    # Generic in-place modulus
    def __imod__(self, arg):

        self.require_writable()

        # If a number...
        if isinstance(arg, numbers.Real) and arg != 0:
            self._values_ %= arg
            self._new_values_()
            return self

        # Convert arg to a Scalar if necessary
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = Qube.SCALAR_CLASS.as_scalar(arg)
            except (ValueError, TypeError):
                Qube._raise_unsupported_op('%=', self, original_arg)

        # Handle modulus by a scalar
        if arg._rank_ == 0:
            divisor = arg.mask_where_eq(0, 1)
            div_values = divisor._values_

            # Align axes
            if self._rank_:
                div_values = np.reshape(div_values, np.shape(div_values) +
                                                    self._rank_ * (1,))
            self._values_ %= div_values
            self._mask_ = self._mask_ | divisor._mask_
            self._units_ = Units.div_units(self._units_, arg._units_)

            self._cache_.clear()
            return self

        # Nothing else is implemented
        Qube._raise_unsupported_op('%=', self, original_arg)

    #===========================================================================
    def _mod_by_number(self, arg, recursive=True):
        """Internal modulus op when the arg is a Python scalar."""

        obj = self.clone(recursive=False, retain_cache=True)

        # Mask out zeros
        if arg == 0:
            obj._set_mask_(True)
        else:
            obj._set_values_(self._values_ % arg, retain_cache=True)

        if recursive and self._derivs_:
            for (key, deriv) in self._derivs_.items():
                obj.insert_deriv(key, deriv)

        return obj

    #===========================================================================
    def _mod_by_scalar(self, arg, recursive=True):
        """Internal modulus op when the arg is a Qube with rank == 0."""

        # Mask out zeros
        arg = arg.wod.mask_where_eq(0,1)

        # Align axes
        arg_values = arg._values_
        if np.shape(arg_values) and self._rank_:
            arg_values = arg_values.reshape(arg.shape + self._rank_ * (1,))

        # Construct the object
        obj = Qube.__new__(type(self))
        obj.__init__(self._values_ % arg_values,
                     self._mask_ | arg._mask_,
                     units = Units.div_units(self._units_, arg._units_),
                     example = self)

        if recursive and self._derivs_:
            for (key, deriv) in self._derivs_.items():
                obj.insert_deriv(key, deriv.broadcast_into_shape(obj._shape_))

        return obj

    ############################################################################
    # Exponentiation operator
    #
    # Default is to support single integer powers between -15 and 15, using
    # repeated multiplications. This will handle any class that supports __mul__
    # and reciprocal(), such as Matrix objects and Quaternions.
    #
    # Overridden by Scalar for normal behavior of the "**" operator.
    ############################################################################

    def __pow__(self, arg):

        if not isinstance(arg, numbers.Real):
            try:
                arg = Qube.SCALAR_CLASS.as_scalar(arg)
            except (ValueError, TypeError):
                Qube._raise_unsupported_op('**', self, arg)

            if arg._shape_:
                Qube._raise_unsupported_op('**', self, arg)

            if arg._mask_:
                return self.as_fully_masked(recursive=True)

            arg = arg._values_

        expo = int(arg)
        if expo != arg:
            Qube._raise_unsupported_op('**', self, arg)

        # At this point, expo is an int

        # Check range
        if expo < -15 or expo > 15:
            raise ValueError('exponent is limited to range (-15,15)')

        # Handle zero
        if expo == 0:
            item = self.identity()
            result = self.filled(self._shape_, item, numer=self._numer_,
                                 mask=self._mask_)
            for key, deriv in self._derivs_.items():
                new_deriv = deriv.zeros(deriv._shape_, numer=deriv._numer_,
                                        denom=deriv._denom_, mask=deriv._mask_)
                result.insert_deriv(key, new_deriv)

            return result

        # Handle negative exponent
        if expo < 0:
            x = self.reciprocal(recursive=True)
            expo = -expo
        else:
            x = self

        # Handle one
        if expo == 1:
            return x

        # Handle 2 through 15
        # Note powers[0] is not a copy!
        # Note derivatives and units are included in multiplies
        powers = [x, x * x]
        if expo >= 4:
            powers.append(powers[-1] * powers[-1])
        if expo >= 8:
            powers.append(powers[-1] * powers[-1])

        # Select the powers needed for this exponent
        x_powers = []
        for k, e in enumerate((1,2,4,8)):
            if (expo & e):
                x_powers.append(powers[k])

        # Multiply the items together
        result = x_powers[-1]
            # x_powers[0] might not be a copy, but x_powers[-1] must be, because
            # we have already already handled expo == 1.
        for x_power in x_powers[:-1]:
            result *= x_power

        return result

    ############################################################################
    # Comparison operators, returning boolean scalars or Booleans
    #   Masked values are treated as equal to one another. Masked and unmasked
    #   values are always unequal.
    ############################################################################

    def _compatible_arg(self, arg):
        """None if it is impossible for self and arg to be equal; otherwise,
        the argument made compatible with self.
        """

        # If the subclasses cannot be unified, raise a ValueError
        if not isinstance(arg, type(self)):
            try:
                obj = Qube.__new__(type(self))
                obj.__init__(arg, example=self)
                arg = obj
            except (ValueError, TypeError):
                return None

        else:
            # Compare units for compatibility
            if not Units.can_match(self._units_, arg._units_):
                return None

            # Compare item shapes
            if self._item_ != arg._item_:
                return None

        # Check for compatible shapes
        try:
            (self, arg) = Qube.broadcast(self, arg)
        except ValueError:
            return None

        return arg

    #===========================================================================
    def __eq__(self, arg):

        # Try to make argument compatible
        arg = self._compatible_arg(arg)
        if arg is None:
            return False        # an incompatible argument is not equal

        # Compare...
        compare = (self._values_ == arg._values_)
        if self._rank_:
            compare = np.all(compare, axis=tuple(range(-self._rank_,0)))

        both_masked = (self._mask_ & arg._mask_)
        one_masked  = (self._mask_ ^ arg._mask_)

        # Return a Python bool if the shape is ()
        if np.isscalar(compare):
            if one_masked:
                return False
            if both_masked:
                return True
            return bool(compare)

        # Apply the mask
        if np.isscalar(one_masked):
            if one_masked:
                compare.fill(False)
            if both_masked:
                compare.fill(True)
        else:
            compare[one_masked] = False
            compare[both_masked] = True

        result = Qube.BOOLEAN_CLASS(compare)
        result._truth_if_all_ = True
        return result

    #===========================================================================
    def __ne__(self, arg):

        # Try to make argument compatible
        arg = self._compatible_arg(arg)
        if arg is None:
            return True         # an incompatible argument is not equal

        # Compare...
        compare = (self._values_ != arg._values_)
        if self._rank_:
            compare = np.any(compare, axis=tuple(range(-self._rank_,0)))

        both_masked = (self._mask_ & arg._mask_)
        one_masked  = (self._mask_ ^ arg._mask_)

        # Compare units for compatibility
        if not Units.can_match(self._units_, arg._units_):
            compare = True
            one_masked = True

        # Return a Python bool if the shape is ()
        if np.isscalar(compare):
            if one_masked:
                return True
            if both_masked:
                return False
            return bool(compare)

        # Apply the mask
        if np.shape(one_masked):
            compare[one_masked] = True
            compare[both_masked] = False
        else:
            if one_masked:
                compare.fill(True)
            if both_masked:
                compare.fill(False)

        result = Qube.BOOLEAN_CLASS(compare)
        result._truth_if_any_ = True
        return result

    #===========================================================================
    def __lt__(self, arg):
        raise ValueError('comparison operators are not supported for class ' +
                         type(self).__name__)

    #===========================================================================
    def __gt__(self, arg):
        raise ValueError('comparison operators are not supported for class ' +
                         type(self).__name__)

    #===========================================================================
    def __le__(self, arg):
        raise ValueError('comparison operators are not supported for class ' +
                         type(self).__name__)

    #===========================================================================
    def __ge__(self, arg):
        raise ValueError('comparison operators are not supported for class ' +
                         type(self).__name__)

    #===========================================================================
    def __bool__(self):
        """Supports 'if a == b: ...' and 'if a != b: ...' statements.

        Equality requires that every unmasked element of a and
        b be equal, and both object be masked at the same locations.

        Comparison of objects of shape () is also supported.

        Any other comparison of PolyMath object requires an explict call to
        all() or any().
        """

        if self._truth_if_all_:
            return bool(np.all(self.as_mask_where_nonzero()))

        if self._truth_if_any_:
            return bool(np.any(self.as_mask_where_nonzero()))

        if self._shape_:
            raise ValueError('the truth value requires any() or all()')

        if self._mask_:
            raise ValueError('the truth value of an entirely masked object ' +
                             'is undefined.')

        return bool(np.all(self.as_mask_where_nonzero()))

    #===========================================================================
    # Needed for backward compatibility with Python 2
    def __nonzero__(self):
        return self.__bool__()

    ############################################################################
    # Boolean operators
    ############################################################################

    # (~) operator
    def __invert__(self):
        return Qube.BOOLEAN_CLASS(self._values_ == 0, self._mask_)

    #===========================================================================
    # (&) operator
    def __and__(self, arg):

        if isinstance(arg, np.ma.MaskedArray):
            arg = Qube.BOOLEAN_CLASS(arg != 0)

        if isinstance(arg, Qube):
            return Qube.BOOLEAN_CLASS(
                                (self._values_ != 0) & (arg._values_ != 0),
                                Qube.or_(self._mask_, arg._mask_))

        return Qube.BOOLEAN_CLASS((self._values_ != 0) & (arg != 0),
                                   self._mask_)

    def __rand__(self, arg):
        return self.__and__(arg)

    #===========================================================================
    # (|) operator
    def __or__(self, arg):

        if isinstance(arg, np.ma.MaskedArray):
            arg = Qube.BOOLEAN_CLASS(arg != 0)

        if isinstance(arg, Qube):
            return Qube.BOOLEAN_CLASS(
                                (self._values_ != 0) | (arg._values_ != 0),
                                Qube.or_(self._mask_, arg._mask_))

        return Qube.BOOLEAN_CLASS((self._values_ != 0) | (arg != 0),
                                   self._mask_)

    def __ror__(self, arg):
        return self.__or__(arg)

    #===========================================================================
    # (^) operator
    def __xor__(self, arg):

        if isinstance(arg, np.ma.MaskedArray):
            arg = Qube.BOOLEAN_CLASS(arg != 0)

        if isinstance(arg, Qube):
            return Qube.BOOLEAN_CLASS(
                                (self._values_ != 0) ^ (arg._values_ != 0),
                                Qube.or_(self._mask_, arg._mask_))

        return Qube.BOOLEAN_CLASS((self._values_ != 0) ^ (arg != 0),
                                   self._mask_)

    def __rxor__(self, arg):
        return self.__xor__(arg)

    #===========================================================================
    # (&=) operator
    def __iand__(self, arg):
        self.require_writable()

        if isinstance(arg, np.ma.MaskedArray):
            arg = Qube.BOOLEAN_CLASS(arg != 0)

        if isinstance(arg, Qube):
            self._values_ &= (arg._values_ != 0)
            self._mask_ = Qube.or_(self._mask_, arg._mask_)
        else:
            self._values_ &= (arg != 0)

        return self

    #===========================================================================
    # (|=) operator
    def __ior__(self, arg):
        self.require_writable()

        if isinstance(arg, np.ma.MaskedArray):
            arg = Qube.BOOLEAN_CLASS(arg != 0)

        if isinstance(arg, Qube):
            self._values_ |= (arg._values_ != 0)
            self._mask_ = Qube.or_(self._mask_, arg._mask_)
        else:
            self._values_ |= (arg != 0)

        return self

    #===========================================================================
    # (^=) operator
    def __ixor__(self, arg):
        self.require_writable()

        if isinstance(arg, np.ma.MaskedArray):
            arg = Qube.BOOLEAN_CLASS(arg != 0)

        if isinstance(arg, Qube):
            self._values_ ^= (arg._values_ != 0)
            self._mask_ = Qube.or_(self._mask_, arg._mask_)
        else:
            self._values_ ^= (arg != 0)

        return self

    #===========================================================================
    def logical_not(self):
        """The negation of this object, True where it is zero or False."""

        if self._rank_:
            values = np.any(self._values_, axis=tuple(range(-self._rank_,0)))
        else:
            values = self._values_

        return Qube.BOOLEAN_CLASS(np.logical_not(values), self._mask_)

    ############################################################################
    # Any and all
    ############################################################################

    def any(self, axis=None, builtins=None, masked=None, out=None):
        """True if any of the unmasked items are nonzero.

        Input:
            axis        an integer axis or a tuple of axes. The any operation
                        is performed across these axes, leaving any remaining
                        axes in the returned value. If None (the default), then
                        the any operation is performed across all axes of the
                        object.
            builtins    if True and the result is a single scalar True or False,
                        the result is returned as a Python boolean instead of
                        an instance of Boolean. Default is that specified by
                        Qube.PREFER_BUILTIN_TYPES.
            masked      value to return if builtins is True but the returned
                        value is masked. Default is to return a masked value
                        instead of a builtin type.
            out         Ignored. Enables "np.any(Qube)" to work.
        """

        self = Qube.BOOLEAN_CLASS.as_boolean(self)

        if not self._shape_:
            args = (self,)                  # make a copy

        elif np.isscalar(self._mask_):
            args = (np.any(self._values_, axis=axis), self._mask_)

        else:
            # True where a value is True AND its antimask is True
            bools = self._values_ & self.antimask
            args = (np.any(bools, axis=axis), np.all(self._mask_, axis=axis))

        result = Qube.BOOLEAN_CLASS(*args)

        # Convert result to a Python bool if necessary
        if builtins is None:
            builtins = Qube.PREFER_BUILTIN_TYPES

        if builtins:
            return result.as_builtin(masked=masked)

        return result

    #===========================================================================
    def all(self, axis=None, builtins=None, masked=None, out=None):
        """True if all the unmasked items are nonzero.

        Input:
            axis        an integer axis or a tuple of axes. The all operation
                        is performed across these axes, leaving any remaining
                        axes in the returned value. If None (the default), then
                        the all operation is performed across all axes of the
                        object.
            builtins    if True and the result is a single scalar True or False,
                        the result is returned as a Python boolean instead of
                        an instance of Boolean. Default is that specified by
                        Qube.PREFER_BUILTIN_TYPES.
            masked      value to return if builtins is True but the returned
                        value is masked. Default is to return a masked value
                        instead of a builtin type.
            out         Ignored. Enables "np.all(Qube)" to work.
        """

        self = Qube.BOOLEAN_CLASS.as_boolean(self)

        if not self._shape_:
            args = (self,)                  # make a copy

        elif np.isscalar(self._mask_):
            args = (np.all(self._values_, axis=axis), self._mask_)

        else:
            # True where a value is True OR its mask is True
            bools = Qube.or_(self._values_, self._mask_)
            args = (np.all(bools, axis=axis), np.all(self._mask_, axis=axis))

        result = Qube.BOOLEAN_CLASS(*args)

        # Convert result to a Python bool if necessary
        if builtins is None:
            builtins = Qube.PREFER_BUILTIN_TYPES

        if builtins:
            return result.as_builtin(masked=masked)

        return result

    #===========================================================================
    def any_true_or_masked(self, axis=None, builtins=None):
        """True if any of the items are nonzero or masked.

        This differs from the any() method in how it handles the case of every
        value being masked. This method returns True, whereas any() returns a
        masked Boolean value.

        Input:
            axis        an integer axis or a tuple of axes. The any operation
                        is performed across these axes, leaving any remaining
                        axes in the returned value. If None (the default), then
                        the operation is performed across all axes of the
                        object.
            builtins    if True and the result is a single scalar True or False,
                        the result is returned as a Python boolean instead of
                        an instance of Boolean. Default is that specified by
                        Qube.PREFER_BUILTIN_TYPES.
        """

        self = Qube.BOOLEAN_CLASS.as_boolean(self)

        if not self._shape_:
            args = (self,)                  # make a copy

        else:
            # True where a value is True OR its mask is True
            bools = Qube.or_(self._values_, self._mask_)
            args = (np.any(bools, axis=axis), False)

        result = Qube.BOOLEAN_CLASS(*args)

        # Convert result to a Python bool if necessary
        if builtins is None:
            builtins = Qube.PREFER_BUILTIN_TYPES

        if builtins:
            return result.as_builtin()

        return result

    #===========================================================================
    def all_true_or_masked(self, axis=None, builtins=None):
        """True if all of the items are nonzero or masked.

        This differs from the all() method in how it handles the case of every
        value being masked. This method returns True, whereas all() returns a
        masked Boolean value.

        Input:
            axis        an integer axis or a tuple of axes. The any operation
                        is performed across these axes, leaving any remaining
                        axes in the returned value. If None (the default), then
                        the operation is performed across all axes of the
                        object.
            builtins    if True and the result is a single scalar True or False,
                        the result is returned as a Python boolean instead of
                        an instance of Boolean. Default is that specified by
                        Qube.PREFER_BUILTIN_TYPES.
        """

        self = Qube.BOOLEAN_CLASS.as_boolean(self)

        if not self._shape_:
            args = (self,)                  # make a copy

        else:
            # True where a value is True OR its mask is True
            bools = Qube.or_(self._values_, self._mask_)
            args = (np.all(bools, axis=axis), False)

        result = Qube.BOOLEAN_CLASS(*args)

        # Convert result to a Python bool if necessary
        if builtins is None:
            builtins = Qube.PREFER_BUILTIN_TYPES

        if builtins:
            return result.as_builtin()

        return result

    ############################################################################
    # Special operators
    ############################################################################

    def reciprocal(self, recursive=True, nozeros=False):
        """An object equivalent to the reciprocal of this object.

        This must be overridden by other subclasses.

        Input:
            recursive   True to return the derivatives of the reciprocal too;
                        otherwise, derivatives are removed.
            nozeros     False (the default) to mask out any zero-valued items in
                        this object prior to the divide. Set to True only if you
                        know in advance that this object has no zero-valued
                        items.
        """

        Qube._raise_unsupported_op('reciprocal()', self)

    #===========================================================================
    def zero(self):
        """An object of this subclass containing all zeros.

        The returned object has the same denominator shape as this object.

        This is default behavior and may need to be overridden by some
        subclasses.
        """

        # Scalar case
        if not self._rank_:
            if self.is_float():
                new_value = 0.
            else:
                new_value = 0

        # Array case
        else:
            if self.is_float():
                new_value = np.zeros(self._item_, dtype=np.float_)
            else:
                new_value = np.zeros(self._item_, dtype=np.int_)

        # Construct the object
        obj = Qube.__new__(type(self))
        obj.__init__(new_value, False, derivs={}, example=self)

        # Return it as readonly
        return obj.as_readonly()

    #===========================================================================
    def identity(self):
        """An object of this subclass equivalent to the identity.

        This must be overridden by other subclasses.
        """

        Qube._raise_unsupported_op('identity()', self)

    #===========================================================================
    def sum(self, axis=None, recursive=True, builtins=None, masked=None,
                  out=None):
        """The sum of the unmasked values along the specified axis.

        Input:
            axis        an integer axis or a tuple of axes. The sum is
                        determined across these axes, leaving any remaining axes
                        in the returned value. If None (the default), then the
                        sum is performed across all axes if the object.
            recursive   True to include the sums of the derivatives inside the
                        returned Scalar.
            builtins    if True and the result is a single unmasked scalar, the
                        result is returned as a Python int or float instead of
                        as an instance of Qube. Default is that specified by
                        Qube.PREFER_BUILTIN_TYPES.
            masked      value to return if builtins is True but the returned
                        value is masked. Default is to return a masked value
                        instead of a builtin type.
            out         Ignored. Enables "np.sum(Qube)" to work.
        """

        result = self._mean_or_sum(axis, recursive, _combine_as_mean=False)

        # Convert result to a Python type if necessary
        if builtins is None:
            builtins = Qube.PREFER_BUILTIN_TYPES

        if builtins:
            return result.as_builtin(masked=masked)

        return result

    #===========================================================================
    def mean(self, axis=None, recursive=True, builtins=None, masked=None,
                   dtype=None, out=None):
        """The mean of the unmasked values along the specified axis.

        Input:
            axis        an integer axis or a tuple of axes. The mean is
                        determined across these axes, leaving any remaining axes
                        in the returned value. If None (the default), then the
                        mean is performed across all axes if the object.
            recursive   True to include the means of the derivatives inside the
                        returned Scalar.
            builtins    if True and the result is a single unmasked scalar, the
                        result is returned as a Python int or float instead of
                        as an instance of Scalar. Default is that specified by
                        Qube.PREFER_BUILTIN_TYPES.
            masked      value to return if builtins is True but the returned
                        value is masked. Default is to return a masked value
                        instead of a builtin type.
            dtype, out  Ignored. Enable "np.mean(Qube)" to work.
        """

        result = self._mean_or_sum(axis, recursive, _combine_as_mean=True)

        # Convert result to a Python type if necessary
        if builtins is None:
            builtins = Qube.PREFER_BUILTIN_TYPES

        if builtins:
            return result.as_builtin(masked=masked)

        return result

    #===========================================================================
    @staticmethod
    def _raise_unsupported_op(op, obj1, obj2=None):
        """Raise a TypeError or ValueError for unsupported operations."""

        if obj2 is None:
            raise TypeError('bad operand type for "%s": %s'
                            % (op, type(obj1).__name__))

        if (isinstance(obj1, (list,tuple,np.ndarray)) or
            isinstance(obj2, (list,tuple,np.ndarray))):
                if isinstance(obj1, Qube):
                    shape1 = obj1._numer_
                else:
                    shape1 = np.shape(obj1)

                if isinstance(obj2, Qube):
                    shape2 = obj2._numer_
                else:
                    shape2 = np.shape(obj2)

                raise ValueError(('unsupported operand item shapes for "%s": ' +
                                  '%s, %s') % (op, str(shape1), str(shape2)))

        raise TypeError(('unsupported operand types for "%s": ' +
                         '%s, %s') % (op, type(obj1).__name__,
                                          type(obj2).__name__))

    ############################################################################
    # Broadcast operations
    ############################################################################

    def broadcast_into_shape(self, shape, recursive=True, _protected=True):
        """This object broadcasted to the specified shape. DEPRECATED name;
        use broadcast_to.

        It returns self if the shape already matches. Otherwise, the returned
        object shares data with the original and both objects will be read-only.

        Input:
            shape           the shape into which the object is to be broadcast.
            recursive       True to broadcast the derivatives as well.
                            Otherwise, they are removed.
        """

        return self.broadcast_to(shape, recursive=recursive,
                                        _protected=_protected)

    def broadcast_to(self, shape, recursive=True, _protected=True):
        """This object broadcasted to the specified shape.

        It returns self if the shape already matches. Otherwise, the returned
        object shares data with the original and both objects will be read-only.

        Input:
            shape           the shape into which the object is to be broadcast.
            recursive       True to broadcast the derivatives as well.
                            Otherwise, they are removed.
        """

        # Set the "un-documented" _protected option to False to prevent setting
        # the arrays as readonly. You have been warned.

        shape = tuple(shape)

        # If no broadcast is needed, return the object
        if shape == self._shape_:
            if recursive:
                return self
            else:
                return self.wod

        # Save the derivatives for later
        derivs = self._derivs_

        # Special case: broadcast to ()
        if shape == ():
            if self._rank_ == 0:
                if isinstance(self._values_, np.ndarray):
                    new_values = self._values_.ravel()[0]
                else:
                    new_values = self._values_
            else:
                new_values = self._values_.reshape(self._item_)

            if isinstance(self._mask_, np.ndarray):
                new_mask = bool(self._mask_.ravel()[0])
            else:
                new_mask = bool(self._mask_)

            # Construct the new object
            obj = Qube.__new__(type(self))
            obj.__init__(new_values, new_mask, example=self)

        else:

            # Broadcast the values array
            if np.isscalar(self._values_):
                self_values = np.array([self._values_])
            else:
                self_values = self._values_

                # An array should be read-only upon broadcast
                if _protected:
                    self.as_readonly(recursive=False)

            new_values = np.broadcast_to(self_values, shape + self._item_)

            # Broadcast the mask if necessary
            if np.isscalar(self._mask_):
                new_mask = self._mask_
            else:
                new_mask = np.broadcast_to(self._mask_, shape)

                # An array should be read-only upon broadcast
                if _protected:
                    self.as_readonly(recursive=False)

            # Construct the new object
            obj = Qube.__new__(type(self))
            obj.__init__(new_values, new_mask, example=self)
            obj.as_readonly(recursive=False)

        # Process the derivatives if necessary
        if recursive:
            for (key, deriv) in derivs.items():
                obj.insert_deriv(key, deriv.broadcast_into_shape(shape, False,
                                                                 _protected))

        return obj

    #===========================================================================
    @staticmethod
    def broadcasted_shape(*objects, **keywords):
        """The shape defined by a broadcast across the objects.

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

        # Search the keywords for "item"
        item = ()
        if 'item' in keywords:
            item = keywords['item']
            del keywords['item']

        # No other keyword is allowed
        if keywords:
          raise TypeError(('broadcasted_shape() got an unexpected keyword ' +
                           'argument "%s"') % keywords.keys()[0])

        # Create a list of all shapes
        shapes = []
        for obj in objects:
            if obj is None or Qube.is_real_number(obj):
                shape = ()
            elif isinstance(obj, (tuple,list)):
                shape = tuple(obj)
            else:
                shape = obj.shape

            shapes.append(shape)

        # Initialize the shape
        new_shape = []
        len_broadcast = 0

        # Loop through the arrays...
        for shape in shapes:
            shape = list(shape)

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
                    raise ValueError('incompatible dimension on axis ' + str(i)
                                     + ': ' + str(shapes))

        return tuple(new_shape) + tuple(item)

    #===========================================================================
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
                        False to strip the derivatives from the returned
                        objects. Default is True.

        Return:         A tuple of copies of the objects, broadcasted to a
                        common shape. The returned objects must be treated as
                        read-only.
        """

        # Set the "un-documented" _protected option to False to prevent setting
        # the arrays as readonly. You have been warned.

        # Search the keywords for "recursive"
        recursive = True
        if 'recursive' in keywords:
            recursive = keywords['recursive']
            del keywords['recursive']

        _protected = True
        if '_protected' in keywords:
            _protected = keywords['_protected']
            del keywords['_protected']

        # No other keyword is allowed
        if keywords:
          raise TypeError(('broadcast() got an unexpected keyword argument ' +
                           '"%s"') % keywords.keys()[0])

        # Perform the broadcasts...
        shape = Qube.broadcasted_shape(*objects)
        results = []
        for obj in objects:
            if isinstance(obj, np.ndarray):
                new_obj = np.broadcast_to(obj, shape)
            elif isinstance(obj, Qube):
                new_obj = obj.broadcast_into_shape(shape, recursive=recursive,
                                                          _protected=_protected)
            else:
                new_obj = obj
            results.append(new_obj)

        return tuple(results)

    ############################################################################
    # Class method from_scalars
    ############################################################################

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

            classes     an arbitrary list defining the preferred class of the
                        returned object. The first suitable class in the list
                        will be used. Default is Vector.
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

        classes = []
        if 'classes' in keywords:
            classes = keywords['classes']
            del keywords['classes']

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
        new_units = None
        new_denom = None

        arrays = []
        masks = []
        deriv_dicts = []
        has_derivs = False
        dtype = np.int_
        for scalar in scalars:
            arrays.append(scalar._values_)
            masks.append(scalar._mask_)

            new_units = new_units or scalar._units_
            Units.require_match(new_units, scalar._units_)

            if new_denom is None:
                new_denom = scalar._denom_
            elif new_denom != scalar._denom_:
                raise ValueError('mixed denominator shapes')

            deriv_dicts.append(scalar._derivs_)
            if len(scalar._derivs_):
                has_derivs = True

            # Remember any floats encountered
            if scalar.is_float():
                dtype = np.float_

        # Construct the values array
        new_drank = len(new_denom)
        new_values = np.array(arrays, dtype=dtype)
        new_values = np.rollaxis(new_values, 0, new_values.ndim - new_drank)

        # Construct the mask (scalar or array)
        masks = Qube.broadcast(*masks)
        new_mask = Qube.or_(*masks)

        # Construct the object
        obj = Qube.__new__(cls)
        obj.__init__(new_values, new_mask, units=new_units,
                                 nrank=scalars[0]._nrank_+1, drank=new_drank)
        obj = obj.cast(classes)

        # Insert derivatives if necessary
        if recursive and has_derivs:
            new_derivs = {}

            # Find one example of each derivative
            examples = {}
            for deriv_dict in deriv_dicts:
                for key, deriv in deriv_dict.items():
                    examples[key] = deriv

            for key, example in examples.items():
                items = []
                if example._item_:
                    missing_deriv = Qube(np.zeros(example._item_),
                                         nrank=example._nrank_,
                                         drank=example._drank_)
                else:
                    missing_deriv = 0.

                for deriv_dict in deriv_dicts:
                    items.append(deriv_dict.get(key, missing_deriv))

                new_derivs[key] = Qube.from_scalars(*items, recursive=False,
                                                            readonly=readonly,
                                                            classes=classes)
            obj.insert_derivs(new_derivs)

        return obj

################################################################################
