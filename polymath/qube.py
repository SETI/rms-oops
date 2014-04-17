################################################################################
# polymath/modules/qube.py: Base class for all PolyMath subclasses.
#
# Mark Showalter, PDS Rings Node, SETI Institute, February 2014
################################################################################

from __future__ import division
import numpy as np
import numbers

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
                       nrank=None, drank=None, example=None):
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
        """

        if type(arg) in (list, tuple):
            arg = np.array(arg)

        if type(mask) in (list, tuple):
            mask = np.array(mask)

        if self.NRANK is not None: nrank = self.NRANK

        # Interpret the example
        if example is not None:
            if arg    is None: arg    = example.__values_
            if mask   is None: mask   = example.__mask_
            if units  is None: units  = example.__units_
            if derivs is None: derivs = example.__derivs_
            if nrank  is None: nrank  = example.__nrank_
            if drank  is None: drank  = example.__drank_

        # Interpret the arg if it is a PolyMath object
        if isinstance(arg, Qube):
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
            if isinstance(arg, bool):
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

        obj.__shape_ = self.__shape_

        obj.__rank_ = self.__rank_
        obj.__nrank_ = self.__nrank_
        obj.__drank_ = self.__drank_

        obj.__item_ = self.__item_
        obj.__numer_ = self.__numer_
        obj.__denom_ = self.__denom_

        obj.__units_ = self.__units_

        obj.__values_ = self.__values_
        obj.__mask_ = self.__mask_

        obj.__readonly_ = self.__readonly_

        # Install the derivs
        obj.__derivs_ = {}
        if recursive:
            obj.insert_derivs(self.__derivs_)
        elif preserve:
            for (key,deriv) in self.__derivs_.iteritems():
                if key not in preserve:
                    obj.insert_deriv(key, deriv)

        # Used only for if clauses
        obj.__truth_if_any_ = self.__truth_if_any_
        obj.__truth_if_all_ = self.__truth_if_all_

        return obj

    ############################################################################
    # Properties and low-level access
    ############################################################################

    def __set_values_(self, values, mask=None):
        """Low-level method to update the values of an array.

        The read-only status of the object will be modified accordingly.

        If a mask is provided, it is also updated.
        """

        # Confirm shapes
        assert np.shape(values) == np.shape(self.__values_)
        if type(mask) == np.ndarray:
            assert np.shape(mask) == self.shape

        # Determine the new read-only state
        new_readonly = Qube._array_is_readonly(values)

        # Update values
        self.__values_ = values

        # Update the mask if necessary
        if mask is not None:

            # Mask must match the read-only state of the object
            if new_readonly:
                Qube._array_to_readonly(mask)

            self.__mask_ = mask

        # If the object is newly read-only, confirm the mask and derivs are also
        if new_readonly and not self.__readonly_:
            if mask is None:
                Qube._array_to_readonly(self.__mask_)

            for derivs in self.__derivs_.values():
                derivs.as_readonly()

        # Update the internal read-only state
        self.__readonly_ = new_readonly

    def __set_mask_(self, mask):
        """Low-level method to update the mask of an array.

        If the object is read-only, then the mask will be set to read-only.
        """

        # Confirm the shape
        assert type(mask)==bool or mask.shape == self.shape

        # Mask must match the read-only state of the object
        if self.__readonly_:
            Qube._array_to_readonly(mask)

        self.__mask_ = mask

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
        if self.__mask_ is False:
            newmask = np.ma.nomask
        elif self.__mask_ is True:
            newmask = np.ones(self.__values_.shape, dtype='bool')
        elif self.__rank_ > 0:
            newmask = self.__mask_.reshape(self.__shape_ + self.__rank_ * (1,))
            (newmask, newvals) = np.broadcast_arrays(newmask, self.__values_)
        else:
            newmask = self.__mask_

        return np.ma.MaskedArray(self.__values_, newmask)

    @property
    def mask(self): return self.__mask_

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
            deriv = deriv.broadcast_into_shape(self.shape, False).copy()
            deriv = deriv.as_readonly()
        elif self.__readonly_ and not deriv.__readonly_:
            deriv = deriv.clone().as_readonly()

        # Save in the derivative dictionary and as an attribute
        if self.readonly and (key in self.__derivs_) and not override:
            raise ValueError('derivative d_d' + key + ' cannot be replaced ' +
                             'in a read-only object')

        self.__derivs_[key] = deriv
        setattr(self, 'd_d' + key, deriv)

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
            obj = self.clone(True)

            # Delete derivatives not on the list
            deletions = []
            for key in obj.__derivs_:
                if key not in preserve:
                    deletions.append(key)

            for key in deletions:
                obj.delete_deriv(key, True)

            return obj

        # Return a fast copy without derivatives
        return self.clone(False)

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
        obj.__set_values_(Units.into_units(self.__units_, self.__values_))

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
        obj.__set_values_(Units.from_units(self.__units_, self.__values_))

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

    def as_writable(self, recursive=True):
        """Convert this object to read-writeable. It is modified and returned.

        If this object is already writable, it is returned as is. Otherwise,
        the internal __values_ and __mask_ arrays are copied.
        """

        # If it is already writable, return
        if not self.__readonly_: return self

        # Update the value if it is an array
        if type(self.__values_) == np.ndarray:
            if Qube._array_is_readonly(self.__values_):
                self.__values_ = self.__values_.copy()

        # Update the mask if it is an array
        if type(self.__mask_) == np.ndarray:
            if Qube._array_is_readonly(self.__mask_):
                self.__mask_ = self.__mask_.copy()

        # Update the derivatives
        if recursive:
            for key in self.__derivs_:
                self.__derivs_[key].as_writable()

        return self

    def match_readonly(self, arg):
        """Sets the read-only status of this object equal to that of another."""

        if arg.__readonly_:
            return self.as_readonly()
        else:
            return self.as_writeable()

    def require_writable(self):
        """Raises a ValueError if the object is read-only.

        Used internally at the beginning of methods that will modify this
        object.
        """

        if self.readonly:
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
                        entirely new copy, suitable for modification. If None,
                        then the copy will have the read-only state of this
                        object.
        """

        # Create a shallow copy
        obj = self.clone(False)

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

        # Set the read-only state
        if readonly is True:
            obj.as_readonly()
        elif readonly is False:
            obj.__readonly_ = False
        else:
            obj.match_readonly(self)

       # Make the derivatives read-only if necessary
        if recursive:
            for (key, deriv) in self.__derivs_.iteritems():
                obj.insert_deriv(key, deriv.copy(recursive=False,
                                                 readonly=readonly))

        return obj

    # Python-standard copy function
    def __copy__(self):
        """Return a deep copy of this object unless it is read-only."""

        return self.copy(readonly=self.readonly, recursive=True)

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
        obj.__set_values_(new_values)
        return obj

    def is_int(self):
        """True if this object contains ints; False if floats or booleans."""

        # Array case
        if isinstance(self.__values_, np.ndarray):
            return np.issubdtype(self.__values_.dtype, int)

        # Scalar case
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
        obj.__set_values_(new_values)
        return obj

    ############################################################################
    # Subclass operations
    ############################################################################

    @staticmethod
    def is_empty(arg):
        return (type(arg) == Qube.EMPTY_CLASS)

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
                    obj.insert_deriv(key, self.as_this_type(deriv,False))

        else:
            obj.__init__(arg, mask=False, units=None, derivs={},
                              nrank=nrank, drank=drank, example=self)

        return obj

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
            replace         a single replacement value, inserted into the values
                            array at every masked location. Use None (the
                            default) to leave values unchanged.
            remask          True to include the new mask into the object's mask;
                            False to replace the values but leave them unmasked.
        """

        # Convert to boolean array if necessary
        if isinstance(mask, np.ndarray):
            mask = mask.astype('bool')
        else:
            mask = Qube.BOOLEAN_CLASS.as_boolean(mask).values

        # If the mask is empty, return the object as is
        if not np.any(mask): return self

        # Get the replacement value as a scalar or NumPy ndarray
        if isinstance(replace, Qube):
            replace = replace.__values_
        elif np.shape(replace) != ():
            replace = np.asarray(replace)

        # Shapeless case
        if np.shape(self.__values_) == ():
            if replace is not None:
                new_values = replace
            else:
                new_values = self.__values_

            if remask:
                new_mask = True
            else:
                new_mask = self.__mask_

            obj = self.clone(True)
            obj.__set_values_(new_values, new_mask)
            return obj

        # Insert replacement values if necessary
        mask_union = self.__mask_ | mask    # change masked values too

        if replace is not None:
            new_values = self.__values_.copy()
            new_values[mask_union] = replace
        else:
            new_values = self.__values_

        # Update the mask if necessary
        if remask:
            new_mask = mask_union
        else:
            new_mask = self.__mask_

        # Construct the object
        obj = self.clone(True)
        obj.__set_values_(new_values, new_mask)
        return obj

    def mask_where_eq(self, match, replace=None, remask=True):
        """Return a copy of this object with items equal to a value masked.

        Instead of or in addition to masking the items, the values can be
        replaced. If no items need to be masked, this object is returned
        unchanged.

        Inputs:
            match           the item value to match.
            replace         a single replacement value, inserted into the values
                            array at every masked location. Use None (the
                            default) to leave values unchanged.
            remask          True to include the new mask into the object's mask;
                            False to replace the values but leave them unmasked.
        """

        match = self.as_this_type(match, recursive=False)

        mask = (self.values == match.values)
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
            replace         a single replacement value, inserted into the values
                            array at every masked location. Use None (the
                            default) to leave values unchanged.
            remask          True to include the new mask into the object's mask;
                            False to replace the values but leave them unmasked.
        """

        match = self.as_this_type(match, recursive=False)

        mask = (self.values != match.values)
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
            replace         a single replacement value, inserted into the values
                            array at every masked location. Use None (the
                            default) to leave values unchanged.
            remask          True to include the new mask into the object's mask;
                            False to replace the values but leave them unmasked.
        """

        assert self.item == ()
        return self.mask_where((self.__values_ <= limit), replace, remask)

    def mask_where_ge(self, limit, replace=None, remask=True):
        """Return a copy of this object with items >= a limit value masked.

        Instead of or in addition to masking the items, the values can be
        replaced. If no items need to be masked, this object is returned
        unchanged.

        Inputs:
            limit           the limiting value.
            replace         a single replacement value, inserted into the values
                            array at every masked location. Use None (the
                            default) to leave values unchanged.
            remask          True to include the new mask into the object's mask;
                            False to replace the values but leave them unmasked.
        """

        assert self.item == ()
        return self.mask_where((self.__values_ >= limit), replace, remask)

    def mask_where_lt(self, limit, replace=None, remask=True):
        """Return a copy with items less than a limit value masked.

        Instead of or in addition to masking the items, the values can be
        replaced. If no items need to be masked, this object is returned
        unchanged.

        Inputs:
            limit           the limiting value.
            replace         a single replacement value, inserted into the values
                            array at every masked location. Use None (the
                            default) to leave values unchanged.
            remask          True to include the new mask into the object's mask;
                            False to replace the values but leave them unmasked.
        """

        assert self.item == ()
        return self.mask_where((self.__values_ < limit), replace, remask)

    def mask_where_gt(self, limit, replace=None, remask=True):
        """Return a copy with items greater than a limit value masked.

        Instead of or in addition to masking the items, the values can be
        replaced. If no items need to be masked, this object is returned
        unchanged.

        Inputs:
            limit           the limiting value.
            replace         a single replacement value, inserted into the values
                            array at every masked location. Use None (the
                            default) to leave values unchanged.
            remask          True to include the new mask into the object's mask;
                            False to replace the values but leave them unmasked.
        """

        assert self.item == ()
        return self.mask_where((self.__values_ > limit), replace, remask)

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
            replace         a single replacement value, inserted into the values
                            array at every masked location. Use None (the
                            default) to leave values unchanged.
            remask          True to include the new mask into the object's mask;
                            False to replace the values but leave them unmasked.
        """

        assert self.item == ()

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
            replace         a single replacement value, inserted into the values
                            array at every masked location. Use None (the
                            default) to leave values unchanged.
            remask          True to include the new mask into the object's mask;
                            False to replace the values but leave them unmasked.
        """

        assert self.item == ()

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
            lower           the lower limit.
            upper           the upper limit.
            remask          True to include the new mask into the object's mask;
                            False to replace the values but leave them unmasked.
        """

        temp   = self.mask_where((self.__values_ < lower), lower, remask)
        result = temp.mask_where((self.__values_ > upper), upper, remask)
        return result

    def masked(self):
        """Return the number of masked items in this object."""

        if self.mask is True:
            return self.size
        else:
            return np.sum(self.mask)

    def unmasked(self):
        """Return the number of unmasked items in this object."""

        if self.mask is True:
            return 0
        else:
            return self.size - np.sum(self.mask)

    def without_mask(self, recursive=True):
        """Return a shallow copy of this object without its mask."""

        if self.mask is False: return self

        obj = self.clone(False)
        obj.__set_mask_(False)

        if recursive:
            for (key,deriv) in self.__derivs_.iteritems():
                obj.insert_deriv(key, deriv.without_mask())

        return obj

    def all_masked(self, recursive=True):
        """Return a shallow copy of this object with everything masked."""

        if self.mask is True: return self

        obj = self.clone(False)
        obj.__set_mask_(True)

        if recursive:
            for (key,deriv) in self.__derivs_.iteritems():
                obj.insert_deriv(key, deriv.all_masked())

        return obj

    def remask(self, mask, recursive=True):
        """Return a shallow copy of this object with a replaced.

        This is much quicker than masked_where(), for cases where only the mask
        is changing.
        """

        if np.shape(mask) not in (self.shape, ()):
            raise ValueError('mask shape is incompatible with object: ' +
                             str(np.shape(mask)) + ', ' + str(self.shape))

        # Construct the new object
        obj = self.clone(False)
        obj.__set_mask_(mask)

        if recursive:
            for (key,deriv) in self.__derivs_.iteritems():
                obj.insert_deriv(key, deriv.remask(mask))

        return obj

    def all_constant(self, constant=None, recursive=True):
        """Return a shallow, read-only copy of this object with constant values.

        Derivatives are all set to zero. The mask is unchanged.
        """

        if constant is None:
            constant = self.zero()

        constant = self.as_this_type(constant, recursive=False)

        obj = self.clone(False)
        obj.__set_values_(Qube.broadcast(constant, obj)[0].__values_)
        obj.as_readonly()

        if recursive:
            for (key,deriv) in self.__derivs_.iteritems():
                obj.insert_deriv(key, deriv.all_constant(recursive=False))

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
            - "mask" if the object has a mask
            - the name of the units of the object has units
            - the names of all the derivatives, in alphabetical order
        """

        suffix = []

        # Apply the units if necessary
        obj = self.into_units(recursive=False)

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
        """Return an object sliced from one numerator axis.

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
        return self.clone()

    def __neg__(self, recursive=True):

        # Construct a copy with negative values
        obj = self.clone(False)
        obj.__set_values_(-self.__values_)

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
        obj = self.clone(False)
        obj.__set_values_(np.abs(self.__values_))

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
        new_derivs = self.add_derivs(arg)   # if this raises exception, stop
        self.__values_ += arg.__values_     # on exception here, no harm done
        self.__mask_ = self.__mask_ | arg.__mask_
        self.__units_ = self.__units_ or arg.__units_
        self.insert_derivs(new_derivs)
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
        new_derivs = self.sub_derivs(arg)   # if this raises exception, stop
        self.__values_ -= arg.__values_     # on exception here, no harm done
        self.__mask_ = self.__mask_ | arg.__mask_
        self.__units_ = self.__units_ or arg.__units_
        self.insert_derivs(new_derivs)
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
        if isinstance(arg, numbers.Real):
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
        if isinstance(arg, numbers.Real):
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
            new_derivs = self.mul_derivs(arg)   # on exception, stop
            self.__values_ *= arg_values        # on exception, no harm done
            self.__mask_ = self.__mask_ | arg.__mask_
            self.__units_ = Units.mul_units(self.__units_, arg.__units_)
            self.insert_derivs(new_derivs)
            return self

        # Matrix multiply case
        if self.__nrank_ == 2 and arg.__nrank_ == 2 and arg.__drank_ == 0:
            result = Qube.dot(self, arg, -1, 0, type(self), recursive=True)
            self.__set_values_(result.__values_, result.__mask_)
            self.insert_derivs(result.__derivs_)
            return self

        # return NotImplemented
        Qube.raise_unsupported_op('*=', self, original_arg)

    def mul_by_number(self, arg, recursive=True):
        """Internal multiply op when the arg is a Python scalar."""

        obj = self.clone(False)
        obj.__set_values_(self.__values_ * arg)

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
        if isinstance(arg, numbers.Real):
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
        if isinstance(arg, numbers.Real):
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
        self.require_writable()
        if Qube.is_empty(arg): return arg

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

        obj = self.clone(False)

        # Mask out zeros
        if arg == 0:
            obj.__set_mask_(True)
        else:
            obj.__set_values_(self.__values_ / arg)

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
            return self

        # Nothing else is implemented
        # return NotImplemented
        Qube.raise_unsupported_op('//=', self, original_arg)

    def floordiv_by_number(self, arg):
        """Internal floor division op when the arg is a Python scalar."""

        obj = self.clone(False)

        if arg == 0:
            obj.__set_mask_(True)
        else:
            obj.__set_values_(self.__values_ // arg)

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
        if isinstance(arg, numbers.Real):
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

            for (key,deriv) in self.derivs.iteritems():
                deriv.__values_ /= div_values
                deriv.__mask_ = deriv.__mask_ | divisor.__mask_
                deriv.__units_ = Units.div_units(self.__units_, arg.__units_)

            return self

        # Nothing else is implemented
        # return NotImplemented
        Qube.raise_unsupported_op('%=', self, original_arg)

    def mod_by_number(self, arg, recursive=True):
        """Internal modulus op when the arg is a Python scalar."""

        obj = self.clone(False)

        # Mask out zeros
        if arg == 0:
            obj.__set_mask_(True)
        else:
            obj.__set_values_(self.__values_ % arg)

        if recursive and self.__derivs_:
            for (key, deriv) in self.__derivs_.iteritems():
                obj.insert_deriv(key, deriv.div_by_number(arg, False))

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

        if recursive:
            obj.insert_derivs(self.div_derivs(arg, nozeros=True))

        return obj

    ############################################################################
    # Exponentiation operators
    ############################################################################

    # Generic exponentiation, PolyMath scalar to a single scalar power
    def __pow__(self, expo, recursive=True):
        if Qube.is_empty(expo): return expo

        # Arrays of exponents are not supported
        if isinstance(expo, Qube):
            if expo.mask:
                Qube.raise_unsupported_op('**', self, expo)

            expo = expo.values

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

        obj = arg.clone(False)
        obj.__set_values_(arg.__values_**expo)
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

        # Return a Python scalar if the shape is ()
        if np.shape(compare) == ():
            if one_masked: return False
            if both_masked: return True
            return compare

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

        # Return a Python scalar if the shape is ()
        if np.shape(compare) == ():
            if one_masked: return True
            if both_masked: return False
            return compare

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

    def __nonzero__(self):
        """Supports 'if a == b: ...' and 'if a != b: ...' statements.

        Equality requires that every unmasked element of a and
        b be equal, and both object be masked at the same locations.

        Comparison of objects of shape () is also supported.

        Any other comparison of PolyMath object requires an explict call to
        all() or any().
        """

        if self.__truth_if_all_:
            return bool(np.all(self.__values_))
        if self.__truth_if_any_:
            return bool(np.any(self.__values_))

        if self.__shape_:
            raise ValueError('the truth value requires any() or all()')

        if self.__mask_:
            raise ValueError('the truth value of an entirely masked object ' +
                             'is undefined.')

        return bool(np.all(self.__values_))

    def all(self):
        """Returns True if every unmasked value is nonzero.

        The truth value of an entirely masked object is undefined."""

        if self.__mask_ is False:
            return bool(np.all(self.__values_))

        if np.all(self.__mask_):
            raise ValueError('the truth value of an entirely masked object ' +
                             'is undefined.')

        return bool(np.all(self.__values_[~self.__mask_]))

    def any(self):
        """Returns True if any unmasked value is nonzero.

        The truth value of an entirely masked object is undefined."""

        if self.__mask_ is False:
            return bool(np.any(self.__values_))

        if np.all(self.__mask_):
            raise ValueError('the truth value of an entirely masked object ' +
                             'is undefined.')
 
        return bool(np.any(self.__values_[~self.__mask_]))

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

    def __getitem__(self, i):

        # A shapeless object cannot be indexed
        if self.__shape_ == ():
            raise IndexError('too many indices')

        # Interpret and adapt the index
        (i,imask,idxmask) = self.prep_index(i, remove_masked=False)

        # Apply index
        item_value = self.__values_[i]

        if np.shape(self.__mask_) == ():
            item_mask = self.__mask_
        else:
            item_mask = self.__mask_[imask]

        if idxmask is not None:
            val_shape = np.shape(item_value)
            mask_shape = val_shape[:len(val_shape)-self.drank-self.nrank]
            if np.shape(idxmask) != mask_shape:
                # Index is not of the same rank as the values
                if np.shape(idxmask) == ():
                    if idxmask: # Single True
                        new_idxmask = np.ones(mask_shape, dtype='bool')
                    else:
                        new_idxmask = idxmask # Single False
                else: # Need to reshape
                    new_idxmask = np.zeros(mask_shape, dtype='bool')
                    new_idxmask[idxmask] = True
                idxmask = new_idxmask
            item_mask |= idxmask

        # Make sure we have not indexed into the item
        item_shape = np.shape(item_value)
        if len(item_shape) < self.__rank_:
            raise IndexError('too many indices')

        # Construct the object
        obj = Qube.__new__(type(self))
        obj.__init__(item_value, item_mask, derivs={}, example=self)
        obj.__readonly_ = self.__readonly_

        # Apply the same indexing to any derivatives
        for (key, deriv) in self.__derivs_.iteritems():
            obj.insert_deriv(key, deriv[imask])

        return obj

    def __setitem__(self, i, arg):
        self.require_writable()

        # A shapeless object cannot be indexed
        if self.__shape_ == ():
            raise IndexError('too many indices')

        # Interpret the arg
        arg = self.as_this_type(arg, recursive=True, nrank=self.nrank,
                                                     drank=self.drank)

        # Derivatives must match
        for key in self.__derivs_:
            if key not in arg.__derivs_:
                raise ValueError('missing derivative d_d%s in replacement' %
                                 key)

        # Interpret and adapt the index
        (i,imask,idxmask) = self.prep_index(i, remove_masked=True)

        if i is None or (np.shape(i) != () and None in i):
            # Fully flattened index along at least one dimension; nothing to do
            return

        # Insert the values
        if (idxmask is None or (np.shape(idxmask) == () and idxmask == False) or
            np.shape(arg.__values_) == ()):
            self.__values_[i] = arg.__values_
        else:
            self.__values_[i] = new_vals[idxmask]

        # Update the mask if necessary
        if np.shape(self.__mask_):
            self.__mask_[imask] = arg.__mask_
        elif np.all(self.__mask_ == arg.__mask_):
            pass
        elif self.__mask_:
            self.__mask_ = np.ones(self.shape, dtype='bool')
            self.__mask_[imask] = arg.__mask_
        else:
            self.__mask_ = np.zeros(self.shape, dtype='bool')
            self.__mask_[imask] = arg.__mask_

        # Also update the derivatives (ignoring those not in self)
        for (key, self_deriv) in self.__derivs_.iteritems():
            self.__derivs_[key][imask] = arg.__derivs_[key]

        return

    def prep_index(self, index, remove_masked):
        """Repair the index for use in a Qube, returning one for the values and
        one for the mask.

        If remove_masked is False:
            If the index is a single Qube object, then we also allow the return
            of a mask indicating which index values are masked. However, if
            the index is multiple objects, then no masked indices are allowed.
        If remove_masked is True:
            If the index is a single Qube object, then any masked items are
            removed.
        
        If the index contains a Ellipsis, we need to append additional null
        slices for the item elements of the array in order to make the axes
        align properly.

        This code also works around a "feature" that makes the simple construct
            if Ellipsis in index
        fail when the index contains a NumPy array.

        The code also replaces a Qube with the result of its
        as_index_and_mask() method.
        """

        # Replace a Qube with its index equivalent
        if isinstance(index, Qube):
            index, idxmask = index.as_index_and_mask(remove_masked=remove_masked)
            return (index, index, idxmask)

        # There's only a problem with indices of type list or tuple
        # By definition these are N-d array references with N>1
        if type(index) not in (tuple,list):
            return (index, index, None) # NOTE we don't handle NP MaskedArrays

        # Search for Ellipses and Qubes
        new_index = []
        has_ellipsis = False
        for item in index:
            if type(item) == type(Ellipsis):
                has_ellipsis = True

            if isinstance(item, Qube):
                index, idxmask = item.as_index_and_mask(remove_masked=remove_masked)
                if idxmask is not None:
                    raise ValueError("illegal masked index for " +
                                     "multi-dimensional indexing")
                new_index.append(index)
            else:
                new_index.append(item)

        if has_ellipsis:
            value_index = new_index + self.__rank_ * [slice(None)]
        else:
            value_index = new_index

        return (tuple(value_index), tuple(new_index), None)

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
          raise TypeError(("broadcast() got an unexpected keyword argument " +
                           "'%s'") % keywords.keys()[0])

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

        return Qube.SCALAR_CLASS(np.sqrt(sum_sq/self.isize), self.mask)

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
          raise TypeError(("broadcasted_shape() got an unexpected keyword " +
                           "argument '%s'") % keywords.keys()[0])

        # Initialize the shape
        new_shape = []
        len_broadcast = 0

        # Loop through the arrays...
        for obj in objects:
            if obj is None: continue
            if Qube.is_empty(obj): continue
            if isinstance(obj, numbers.Real): continue

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
          raise TypeError(("broadcast() got an unexpected keyword argument " +
                           "'%s'") % keywords.keys()[0])

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

################################################################################
