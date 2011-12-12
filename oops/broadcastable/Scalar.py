import numpy as np
import unittest

################################################################################
################################################################################
# Superclass of Scalar, Pair, Vector3, Matrix3 and Tuple. All represent
# N-dimensional arrays of intrinsically dimensional objects.
#
# Modified 12/12/2011 (BSW) - added help comments to most methods
################################################################################
################################################################################

class Array(object):
    """A class defining an arbitrary Array of possibly multidimensional items.
    Unlike numpy ndarrays, this class makes a clear distinction between the
    dimensions associated with the items and any additional, leading dimensions
    which describe the set of such items. The shape is defined by the leading
    axes only, so a 2x2 array of 3x3 matrices would have shape (2,2,3,3)
    according to numpy but has shape (2,2) according to Array.

    The Array object is designed as a lightweight "wrapper" on the numpy
    ndarray. All standard mathematical operators and indexing/slicing options
    are defined. One can mix Array arithmetic with scalars, numpy arrays, or
    anything array-like. The standard numpy rules of broadcasting apply to
    operations performed on Arrays of different shapes.

    Array objects have the following attributes:
        shape       a list (not tuple) representing the leading dimensions.
        rank        the number of trailing dimensions belonging to the
                    individual items.
        item        a list (not tuple) representing the shape of the
                    individual items.
        vals        the array's actual data as a numpy array. The shape of this
                    array is object.shape + object.item.
    """

    # A constant, defined for all Array subclasses, overridden by subclass Empty
    IS_EMPTY = False

    def __new__(subtype, *arguments, **keywords):
        obj = object.__new__(subtype)
        return obj

    def __repr__(self):
        """show values of Array or subtype. repr() call returns array at start
            of string, replace with object type, else just prefix entire string
            with object type."""
        string = repr(self.vals)
        if string[:5] == "array":
            return type(self).__name__ + string[5:]
        else:
            return type(self).__name__ + "(" + string + ")"

    def __str__(self):
        """show values of Array or subtype. repr() call returns array at start
            of string, replace with object type, else just prefix entire string
            with object type."""
        string = str(self.vals)
        if string[:5] == "array":
            return type(self).__name__ + string[5:]
        else:
            return type(self).__name__ + "(" + string + ")"

    ####################################
    # Indexing operators
    ####################################

    def __getitem__(self, i):
        """returns the item value at a specific index. copies data to new
            location in memory. if self has no items, i.e. - no shape in the
            object-sense, then raise IndexError. called from x = obj[i]"""
        if self.shape == []: raise IndexError("too many indices")

        if np.size(self.vals[i]) == 1: return self.vals[i]

        obj = Array.__new__(type(self))
        obj.__init__(self.vals[i])
        return obj

    def __setitem__(self, i, arg):
        """sets the item value at a specific index. if self has no items,
            i.e. - no shape in the object-sense, then raise IndexError. called
            from obj[i] = arg."""
        if self.shape == []: raise IndexError("too many indices")

        if isinstance(arg, Array): arg = arg.vals
        self.vals[i] = arg
        return self

    def __getslice__(self, i, j):
        """returns slice of items. copies data to new location in memory. if
            self has no items, i.e. - no shape in the object-sense, then raise
            IndexError. called from x = obj[i:j]."""
        if self.shape == []: raise IndexError("too many indices")

        obj = Array.__new__(type(self))
        obj.__init__(self.vals[i:j])
        return obj

    def __setslice__(self, i, j, arg):
        """sets slice of items' values to values of arg. if self has no items,
            i.e. - no shape in the object-sense, then raise IndexError. called
            from obj[i:j] = arg."""
        if self.shape == []: raise IndexError("too many indices")

        if isinstance(arg, Array): arg = arg.vals
        self.vals[i:j] = arg
        return self

    ####################################
    # Unary arithmetic operators
    ####################################

    def __pos__(self): return self

    def __neg__(self):
        """returns newly created instance of object with values equal to self
            except the negative. called from x = -obj."""
        obj = Array.__new__(type(self))
        obj.__init__(-self.vals)
        return obj

    def __nonzero__(self):
        """returns true if all values of each item are non-zero, else false.
            called from if(obj != 0) of if(boj)."""
        return bool(np.all(self.vals))

    ####################################
    # Default binary arithmetic operators
    ####################################

    def __add__(self, arg):
        """returns newly created instance of same type as self, adding self and
            arg together. returns object of class Empty if either object is of
            class Empty. raises type mismatch error if objects are of different
            types, and item mismatch if objects have different shape. for
            optimal performance, immediately add object elements together and
            return if of same type, else convert second operand to np array and
            then add elements. called from x = obj + arg.
            
            Input:
            self      left operand.
            arg       right operand.
            
            Return:   Array with elements of self + arg elements.
            """
        # If the operands are of the same subclass, operate element-by-element
        if isinstance(arg, type(self)):
            obj = Array.__new__(type(self))
            obj.__init__(self.vals + arg.vals)
            return obj

        # Any operation with Empty returns Empty
        if isinstance(self, Empty): return self
        if isinstance(arg, Empty): return arg

        # If the operands are of different Array subclasses, raise a TypeError
        if isinstance(arg, Array): self.raise_type_mismatch("+", arg)

        # Convert the second operand to numpy ndarray if necessary
        arg = np.asarray(arg)

        # If the item shapes do not match, raise a ValueError
        if self.rank > 0:
            if (len(arg.shape) < self.rank or
                arg.shape[-self.rank:] != tuple(self.item)):
                    self.raise_item_mismatch("+", arg)

        # Operate element-by-element
        obj = Array.__new__(type(self))
        obj.__init__(self.vals + arg)
        return obj

    def __sub__(self, arg):
        """returns newly created instance of same type as self, subtracting arg
            (second operand) from self (first operand). returns object of class
            Empty if either object is ofclass Empty. raises type mismatch error
            if objects are of different types, and item mismatch if objects have
            different shape. for optimal performance, immediately subtract arg
            elements from self elements and return if of same type, else convert
            second operand to np array and then subtract arg elements from self
            elements. called from x = obj - arg.
            
            Input:
            self      left operand.
            arg       right operand.
            
            Return:   Array with elements of self - arg elements.
            """
        # If the operands are of the same subclass, operate element-by-element
        if isinstance(arg, type(self)):
            obj = Array.__new__(type(self))
            obj.__init__(self.vals - arg.vals)
            return obj

        # Any operation with Empty returns Empty
        if isinstance(self, Empty): return self
        if isinstance(arg, Empty): return arg

        # If the operands are of different Array subclasses, raise a TypeError
        if isinstance(arg, Array): self.raise_type_mismatch("-", arg)

        # Convert the second operand to numpy ndarray if necessary
        arg = np.asarray(arg)

        # If the item shapes do not match, raise a ValueError
        if self.rank > 0:
            if (len(arg.shape) < self.rank or
                arg.shape[-self.rank:] != tuple(self.item)):
                    self.raise_item_mismatch("-", arg)

        # Operate element-by-element
        obj = Array.__new__(type(self))
        obj.__init__(self.vals - arg)
        return obj

    def __mul__(self, arg):
        """returns newly created instance of same type as self, multiplying
            elements of self (first operand) with elements of arg (second
            operand). called from x = obj * arg.
            
            Input:
            self      left operand.
            arg       right operand.
            
            Return:   Array with elements of self * arg elements.
            """
        # If the operands are of the same subclass, operate element-by-element
        if isinstance(arg, type(self)):
            obj = Array.__new__(type(self))
            obj.__init__(self.vals * arg.vals)
            return obj

        # Any operation with Empty returns Empty
        if isinstance(self, Empty): return self
        if isinstance(arg, Empty): return arg

        # If the operands are of different Array subclasses...
        if isinstance(arg, Array):

            # For the special case of (1-D * 0-D), scale the first operand
            if self.rank == 1 and arg.rank == 0:
                obj = Array.__new__(type(self))
                obj.__init__(self.vals * np.asarray(arg.vals)[..., np.newaxis])
                return obj

            # For the special case of (0-D * 1-D), scale the second operand
            if self.rank == 0 and arg.rank == 1:
                obj = Array.__new__(type(arg))
                obj.__init__(arg.vals * np.asarray(self.vals)[..., np.newaxis])
                return obj

            # Otherwise, raise a TypeError
            self.raise_type_mismatch("*", arg)

        # If the second operand is array-like, handle 0-D and 1-D operations...
        if self.rank <= 1:

            # Convert it to a numpy ndarray if necessary
            arg = np.asarray(arg)

            # If the arg does not match the shape of a 1-D item, add an axis.
            # This causes an item-by-item operation to occur for matching items,
            # but an overall scale factor to be applied otherwise.
            if (self.rank == 1 and arg.shape != ()
                               and self.item[0] != arg.shape[-1]):
                arg = arg[..., np.newaxis]

            obj = Array.__new__(type(self))
            obj.__init__(self.vals * arg)
            return obj

        # No other operations are supported by default, so raise a ValueError
        self.raise_type_mismatch("*", arg)

    def __div__(self, arg):
        """returns newly created instance of same type as self, dividing
            elements of self (first operand) by elements of arg (second
            operand). called from x = obj / arg.
            
            Input:
            self      left operand.
            arg       right operand.
            
            Return:   Array with elements of self / arg elements.
            """
        # If the operands are of the same subclass, operate element-by-element
        if isinstance(arg, type(self)):
            obj = Array.__new__(type(self))
            obj.__init__(self.vals / arg.vals)
            return obj

        # Any operation with Empty returns Empty
        if isinstance(self, Empty): return self
        if isinstance(arg, Empty): return arg

        # If the operands are of different Array subclasses...
        if isinstance(arg, Array):

            # For the special case of (1-D / 0-D), scale the first operand
            if self.rank == 1 and arg.rank == 0:
                obj = Array.__new__(type(self))
                obj.__init__(self.vals / np.asarray(arg.vals)[..., np.newaxis])
                return obj

            # Otherwise, raise a TypeError
            self.raise_type_mismatch("/", arg)

        # If the second operand is array-like, handle 0-D and 1-D operations...
        if self.rank <= 1:

            # Convert it to a numpy ndarray if necessary
            arg = np.asarray(arg)

            # If the arg does not match the shape of a 1-D item, add an axis.
            # This causes an item-by-item operation to occur for matching items,
            # but an overall scale factor to be applied otherwise.
            if (self.rank == 1 and arg.shape != ()
                               and self.item[0] != arg.shape[-1]):
                arg = arg[..., np.newaxis]

            obj = Array.__new__(type(self))
            obj.__init__(self.vals / arg)
            return obj

        # No other operations are supported by default, so raise a ValueError
        self.raise_type_mismatch("/", arg)

    ####################################
    # Default arithmetic-in-place operators
    ####################################

    def __iadd__(self, arg):
        """add arg to self and return self. raises type mismatch error if
            objects are of different types, and item mismatch if objects have
            different shape. for optimal performance, immediately add object
            elements of arg to those of self and return if of same type, else
            convert second operand to np array and then add elements. called
            from self += arg.
            
            Input:
            self      left operand.
            arg       right operand.
            
            Return:   self.  Operates on left operand.
            """

        # If the operands are of the same subclass, operate element-by-element
        if isinstance(arg, type(self)):
            self.vals += arg.vals
            return self

        # If the operands are of different Array subclasses, raise a TypeError
        if isinstance(arg, Array): self.raise_type_mismatch("+=", arg)

        # Convert the second operand to numpy ndarray if necessary
        arg = np.asarray(arg)

        # If the item shapes do not match, raise a ValueError
        if self.rank > 0:
            if (len(arg.shape) < self.rank or
                arg.shape[-self.rank:] != tuple(self.item)):
                    self.raise_item_mismatch("+=", arg)

        # Operate element-by-element
        self.vals += arg
        return self

    def __isub__(self, arg):
        """subtract arg from self and return self. raises type mismatch error if
            objects are of different types, and item mismatch if objects have
            different shape. for optimal performance, immediately subtract
            object elements of arg from that of self and return if of same type,
            else convert second operand to np array and then subtract elements.
            called from self -= arg.
            
            Input:
            self      left operand.
            arg       right operand.
            
            Return:   self.  Operates on left operand.
            """
        # If the operands are of the same subclass, operate element-by-element
        if isinstance(arg, type(self)):
            self.vals -= arg.vals
            return self

        # If the operands are of different Array subclasses, raise a TypeError
        if isinstance(arg, Array): self.raise_type_mismatch("-=", arg)

        # Convert the second operand to numpy ndarray if necessary
        arg = np.asarray(arg)

        # If the item shapes do not match, raise a ValueError
        if self.rank > 0:
            if (len(arg.shape) < self.rank or
                arg.shape[-self.rank:] != tuple(self.item)):
                    self.raise_item_mismatch("-=", arg)

        # Operate element-by-element
        self.vals -= arg
        return self

    def __imul__(self, arg):
        """multiply arg with self and return self. raises type mismatch error if
            objects are of different types, and item mismatch if objects have
            different shape. for optimal performance, immediately multiply
            object elements of arg to those of self and return if of same type,
            else convert second operand to np array and then multiply elements.
            called from self *= arg.
            
            Input:
            self      left operand.
            arg       right operand.
            
            Return:   self.  Operates on left operand.
            """
        # If the operands are of the same subclass, operate element-by-element
        if isinstance(arg, type(self)):
            self.vals *= arg.vals
            return self

        # If the operands are of different Array subclasses...
        if isinstance(arg, Array):

            # For the special case of (1-D * 0-D), scale the first operand
            if self.rank == 1 and arg.rank == 0:
                self.vals *= np.asarray(arg.vals)[..., np.newaxis]
                return self

            # Otherwise, raise a ValueError
            self.raise_type_mismatch("*=", arg)

        # If the second operand is array-like, scale a 0-D or 1-D subclass
        if self.rank == 0:
            self.vals *= np.asarray(arg)
            return self

        if self.rank == 1:
            arg = np.asarray(arg)
            if arg.shape != () and self.item[0] != arg.shape[-1]:
                arg = arg[..., np.newaxis]

            self.vals *= arg
            return self

        # No other operations are supported by default, so raise a ValueError
        self.raise_type_mismatch("*=", arg)

    def __idiv__(self, arg):
        """divide self by arg and return self. raises type mismatch error if
            objects are of different types, and item mismatch if objects have
            different shape. for optimal performance, immediately divide
            object elements of arg by those of self and return if of same type,
            else convert second operand to np array and then divide elements.
            called from self /= arg.
            
            Input:
            self      left operand.
            arg       right operand.
            
            Return:   self.  Operates on left operand.
            """
        # If the operands are of the same subclass, operate element-by-element
        if isinstance(arg, type(self)):
            self.vals /= arg.vals
            return self

        # If the operands are of different Array subclasses...
        if isinstance(arg, Array):

            # For the special case of (1-D / 0-D), scale the first operand
            if self.rank == 1 and arg.rank == 0:
                self.vals /= np.asarray(arg.vals)[..., np.newaxis]
                return self

            # Otherwise, raise a ValueError
            self.raise_type_mismatch("*=", arg)

        # If the second operand is array-like, scale a 0-D or 1-D subclass
        if self.rank == 0:
            self.vals /= np.asarray(arg)
            return self

        if self.rank == 1:
            arg = np.asarray(arg)
            if arg.shape != () and self.item[0] != arg.shape[-1]:
                arg = arg[..., np.newaxis]

            self.vals /= arg
            return self

        # No other operations are supported by default, so raise a ValueError
        self.raise_type_mismatch("/=", arg)

    ####################################
    # Default comparison operators
    ####################################

    def __eq__(self, arg):
        """return list of bools describing whether each item within list has all
            equal elements. if arguments are not of same instance, convert to
            nparray and test each element. if we do not have sufficient size in
            arg to test all elements of at least one item of self, or shape of
            elemnts in trailing dimensions of arg do not have identical shape to
            that of items in self, then return false. called by (self == arg).
            
            Input:
            self        left operand
            arg         right operand
            
            Return:     list of bools indicating equivalence of items.
            """
        # If second operand has the same subclass, operate item-by-item
        if isinstance(arg, type(self)):

            # First operate element-by-element
            bools = (self.vals == arg.vals)

            # Collapse rightmost axes depending on rank
            for iter in range(self.rank):
                bools = np.all(bools, axis=-1)      # equal if all are equal

            # Return as scalar
            return Scalar(bools)

        # If second operand is a different Array subclass, objects are not equal
        if isinstance(arg, Array): return False

        # Convert to numpy ndarray if necessary
        arg = np.asarray(arg)

        # Make sure item shapes match; unequal on mismatch
        if self.rank > 0:
            if (len(arg.shape) < self.rank or
                arg.shape[-self.rank:] != tuple(self.item)):
                    return False

        # Operate item-by-item
        bools = (self.vals == arg)

        # Collapse rightmost axes depending on rank
        for iter in range(self.rank):
            bools = np.all(bools, axis=-1)

        # Return as scalar
        return Scalar(bools)

    def __ne__(self, arg):
        """return list of bools describing whether each item within list has any
            unequal elements. if arguments are not of same instance, convert to
            nparray and test each element. if we do not have sufficient size in
            arg to test all elements of at least one item of self, or shape of
            elemnts in trailing dimensions of arg do not have identical shape to
            that of items in self, then return true. called by (self != arg).
            
            Input:
            self        left operand
            arg         right operand
            
            Return:     list of bools indicating unequivalence of items.
            """
        # If second operand has the same subclass, operate item-by-item
        if isinstance(arg, type(self)):

            # First operate element-by-element
            bools = (self.vals != arg.vals)

            # Collapse rightmost axes depending on rank
            for iter in range(self.rank):
                bools = np.any(bools, axis=-1)  # unequal if any are unequal

            # Return as scalar
            return Scalar(bools)

        # If second operand is a different Array subclass, objects are not equal
        if isinstance(arg, Array): return True

        # Convert to numpy ndarray if necessary
        arg = np.asarray(arg)

        # Make sure item shapes match; unequal on mismatch
        if self.rank > 0:
            if (len(arg.shape) < self.rank or
                arg.shape[-self.rank:] != tuple(self.item)):
                    return True

        # Operate item-by-item
        bools = (self.vals != arg)

        # Collapse rightmost axes depending on rank
        for iter in range(self.rank):
            bools = np.any(bools, axis=-1)  # unequal if any are unequal

        # Return as scalar
        return Scalar(bools)

    ####################################
    # Miscellaneous functions
    ####################################

    def __copy__(self):
        """describes how python should handle copying of Array types. if vals
            are instance of type nparray then create new instance of Array with
            a copy of the values, else just use the memory location of the
            values.
            
            Return:     a new instance of type Array.
            """
        obj = Array.__new__(type(self))
        if isinstance(self.vals, np.ndarray):
            obj.__init__(self.vals.copy())
        else:
            obj.__init__(self.vals)

        return obj

    def copy(self): return self.__copy__()

    def astype(self, dtype):
        """converts dtype of elements of Array to said type. creates new
            instance of Array and returns it.
            
            Input:
            dtype:      type, such as int, float32, etc.
            
            Return:     new instance of Array with converted elements.
            """
        obj = Array.__new__(type(self))

        if isinstance(self.vals, np.ndarray):
            if self.vals.dtype == dtype: return self
            obj.__init__(self.vals.astype(dtype))
        else:
            value = np.array([self.vals], dtype=dtype)
            obj.__init__(value[0])

        return obj

    def swapaxes(self, axis1, axis2):
        """returns a new instance of Array with the elements in axis1 and axis2
            swapped. use nparray.swapaxes on Array vals. if axis1 or axis2
            is greater than rank of self then raise ValueError. note that new
            data is not created, but a new Array shell is created that indexes
            its axes in a swapped fashion, therefore changing values of original
            Array will change values of newly created Array.
            
            Input:
            axis1       first axis to swap from/to.
            axis2       second axis to swap to/from.
            
            Return:     new instance of Array with axes swapped.
            """
        if axis1 >= len(self.shape):
            raise ValueError("bad axis1 argument to swapaxes")
        if axis2 >= len(self.shape):
            raise ValueError("bad axis2 argument to swapaxes")

        obj = Array.__new__(type(self))
        obj.__init__(self.vals.swapaxes(axis1,axis2))
        return obj

    def reshape(self, shape):
        """returns a new instance of Array with the list of items reshaped to
            that of shape. note that the argument shape refers to the shape of
            Array, not that of the nparray. note that new data is not created,
            but a new Array shell is created that indexes its elemnts according
            to the new shape, therefore changing values of the original
            Array will change values of newly created Array.
            
            Input:
            shape       shape of Array
            
            Return:     new instance of Array with new shape.
            """
        obj = Array.__new__(type(self))
        obj.__init__(self.vals.reshape(list(shape) + self.item))
        return obj

    def flatten(self):
        """returns a new instance of Array with the items flattened into a one-
            dimensional nparray of Arrays. note that new data is not created,
            but a new Array shell is created that indexes its elemnts according
            to the new shape, therefore changing values of the original
            Array will change values of newly created Array.
            
            Return:     new instance of Array with 1D shape.
            """
        if len(self.shape) < 2: return self

        count = np.product(self.shape)
        obj = Array.__new__(type(self))
        obj.__init__(self.vals.reshape([count] + self.item))
        return obj

    def ravel(self): return self.flatten()

    def reorder_axes(self, axes):
        """Puts the leading axes into the specified order. Item axes are
        unchanged."""

        allaxes = axes + range(len(self.shape), len(self.vals.shape))

        obj = Array.__new__(type(self))
        obj.__init__(self.vals.transpose(allaxes))
        return obj

    def append_axes(self, axes):
        """Appends the specified number of unit axes to the end of the shape."""

        if axes == 0: return self

        obj = Array.__new__(type(self))
        obj.__init__(self.vals.reshape(list(shape) + axes*[1]))
        return obj

    def prepend_axes(self, axes):
        """Prepends the specified number of unit axes to the end of the shape.
        """

        if axes == 0: return self

        obj = Array.__new__(type(self))
        obj.__init__(self.vals.reshape(axes*[1] + list(shape)))
        return obj

    def strip_axes(self, axes):
        """Removes unit axes from the beginning of an Array's shape."""

        newshape = self.shape
        while len(newshape) > 0 and newshape[0] == 1: newshape = newshape[1:]

        obj = Array.__new__(type(self))
        obj.__init__(self.vals.reshape(newshape + self.item))
        return obj

    def rotate_axes(self, axis):
        """Rotates the axes until the specified axis comes first; leading axes
        are moved to the end but the item axes are unchanged."""

        allaxes = (range(axis, axis + len(self.shape)) +
                   range(axis) +
                   range(len(self.shape), len(self.vals.shape)))

        obj = Array.__new__(type(self))
        obj.__init__(self.vals.transpose(allaxes))
        return obj

    def prepend_rotate_strip(self, axes, rank):
        """This method prepends unit axes if necessary until the leading Array
        dimensions reach the specified rank. Then it rotates the axes to put the
        specified axis first. Then it strips away any leading unit axes.

        This procedure can be used to rotate the axes of multiple Arrays in a
        consistent way such that the same axes broadcast together both before
        and after the operation."""

        return self.prepend_axes(rank-len(self.shape)).rotate(axes).strip_axes()

    def rebroadcast(self, newshape):
        """Returns an Array broadcasted to the specified shape. It returns self
        if the shape already matches. Otherwise, the returned object shares data
        with the original and should be treated as read-only."""

        newshape = list(newshape)
        if newshape == self.shape: return self

        temp = np.empty(newshape + self.item, dtype="byte")
        vals = np.broadcast_arrays(self.vals, temp)[0]

        obj = Array.__new__(type(self))
        obj.__init__(vals)
        return obj

    @staticmethod
    def broadcast_shape(arrays, item=[]):
        """This static method returns the shape that would result from a
        broadcast across the provided set of Array objects. It raises a
        ValueError if the shapes cannot be broadcasted.

        Input:
            arrays      a list or tuple containing zero or more array objects.
                        Values of None and Empty() are ignored. Anything (such
                        as a numpy array) that has an intrinsic shape attribute
                        can be used. A list or tuple is treated as the
                        definition of an additional shape.

            item        a list or tuple to be appended to the shape. Default is
                        []. Makes it possible to use the returned shape in the
                        declaration of a numpy array containing items that are
                        not scalars.

        Return:         the broadcast shape, comprising the maximum value of
                        each corresponding axis, plus the item shape if any.
        """

        # Initialize the shape
        broadcast = []
        len_broadcast = 0

        # Loop through the arrays...
        for array in arrays:
            if array == None: continue

            # Get the next shape
            try:
                shape = list(array.shape)
            except AttributeError:
                shape = list(array)

            # Expand the shapes to the same rank
            len_shape = len(shape)

            if len_shape > len_broadcast:
                broadcast = [1] * (len_shape - len_broadcast) + broadcast
                len_broadcast = len_shape

            if len_broadcast > len_shape:
                shape = [1] * (len_broadcast - len_shape) + shape
                len_shape = len_broadcast

            # Update the broadcast shape and check for compatibility
            for i in range(len_shape):
                if broadcast[i] == 1:
                    broadcast[i] = shape[i]
                elif shape[i] == 1:
                    pass
                elif shape[i] != broadcast[i]:
                    raise ValueError("shape mismatch: two or more arrays " +
                        "have incompatible dimensions on axis " + str(i))

        return broadcast + item

    @staticmethod
    def broadcast_arrays(arrays):
        """This static method returns a list of Array objects all broadcasted to
        the same shape. It raises a ValueError if the shapes cannot be
        broadcasted.

        Input:
            arrays      a list or tuple containing zero or more array objects.
                        Values of None and Empty() are returned as is. Anything
                        scalar or array-like is first converted to a Scalar.

        Return:         A list of new Array objects, broadcasted to the same
                        shape. The array data should be treated as read-only.
                        In the returned array, the function arguments come
                        before the elements of the "arrays" list.
        """

        newshape = Array.broadcast_shape(arrays)

        results = []
        for array in arrays:
            if array is None:
                results.append(None)
            else:
                results.append(Array.rebroadcast(array, newshape))

        return results

    @staticmethod
    def shape(arg):
        """Returns the inferred shape of the given argument, regardless of its
        class, as a list."""

        if isinstance(arg, Array): return arg.shape
        if isinstance(arg, Group): return arg.shape
        return list(np.shape(arg))

    @staticmethod
    def item(arg):
        """Returns the inferred item shape of the given argument, regardless of
        its class, as a list."""

        if isinstance(arg, Array): return arg.item
        return []

    @staticmethod
    def rank(arg):
        """Returns the inferred dimensions of the items comprising the given
        argument, regardless of its class, as a list."""

        if isinstance(arg, Array): return arg.rank
        return 0

    @staticmethod
    def vals(arg):
        """Returns the value array from an Array object. For anything else it
        returns the object itself. In either case, the result is returned as a
        numpy ndarray."""

        if isinstance(Array): arg = arg.vals
        return np.asarray(arg)

    # Support methods for raising errors...

    def raise_type_mismatch(self, op, arg):
        """Raises a TypeError with text indicating that the operand types are
        unsupported."""

        raise TypeError("unsupported operand types for " + op +
                        ": '"     + type(self).__name__ +
                        "' and '" + type(arg).__name__  + "'")

    def raise_item_mismatch(self, op, arg):
        """Raises a ValueError with text indicating that the operand shapes are
        incompatible."""

        raise ValueError("incompatible operand types for " + op +
                         ": '"  + type(self).__name__ +
                         "' and 'ndarray' of shape " + str(np.shape(arg)))

################################################################################
################################################################################
# Empty
################################################################################
################################################################################

class Empty(Array):
    """An empty array, needed in some situations so the moral equivalent of a
    "None" can still respond to basic Array operations."""

    OOPS_CLASS = "Empty"
    IS_EMPTY = True

    def __init__(self, vals=None):

        self.shape = []
        self.rank  = 0
        self.item  = []
        self.vals  = np.array(0)

    # Overrides of standard Array methods
    def __repr__(self): return "Empty()"

    def __str__(self): return "Empty()"

    def __nonzero__(self): return False

    def __getitem__(self, i): return self

    def __getslice__(self, i, j): return self

################################################################################
################################################################################
# Scalar
################################################################################
################################################################################

class Scalar(Array):
    """An arbitrary Array of scalars."""

    OOPS_CLASS = "Scalar"

    def __init__(self, arg):

        if isinstance(arg, Scalar): arg = arg.vals

        if np.shape(arg) == ():
            self.vals  = arg
            self.rank  = 0
            self.item  = []
            self.shape = []
        else:
            self.vals  = np.asarray(arg)
            self.rank  = 0
            self.item  = []
            self.shape = list(self.vals.shape)

        return

    @staticmethod
    def as_scalar(arg):
        if isinstance(arg, Scalar): return arg
        return Scalar(arg)

    @staticmethod
    def as_float_scalar(arg):
        if isinstance(arg, Scalar) and arg.vals.dtype == np.dtype("float"):
            return arg

        return Scalar.as_scalar(arg) * 1.

    def int(self):
        """Returns the integer (floor) component of each value."""

        return Scalar(np.floor(self.vals).astype("int"))

    def frac(self):
        """Returns the fractional component of each value."""

        return Scalar(self.vals - np.floor(self.vals))

    # abs() operator
    def __abs__(self):
        return Scalar(np.abs(self.vals))

    # (%) operator
    def __mod__(self, arg):
        if isinstance(arg, Scalar): arg = arg.vals
        return Scalar(self.vals % arg)

    # (%=) operator
    def __imod__(self, arg):
        if isinstance(arg, Scalar): arg = arg.vals
        self.vals %= arg
        return self

    # (<) operator
    def __lt__(self, arg):
        if isinstance(arg, Scalar): arg = arg.vals
        return Scalar._scalar_unless_shapeless(self.vals < arg)

    # (>) operator
    def __gt__(self, arg):
        if isinstance(arg, Scalar): arg = arg.vals
        return Scalar._scalar_unless_shapeless(self.vals > arg)

    # (<=) operator
    def __le__(self, arg):
        if isinstance(arg, Scalar): arg = arg.vals
        return Scalar._scalar_unless_shapeless(self.vals <= arg)

    # (>=) operator
    def __ge__(self, arg):
        if isinstance(arg, Scalar): arg = arg.vals
        return Scalar._scalar_unless_shapeless(self.vals >= arg)

    # (~) operator
    def __invert__(self):
        return Scalar._scalar_unless_shapeless(~self.vals)

    # (&) operator
    def __and__(self, arg):
        if isinstance(arg, Scalar): arg = arg.vals
        return Scalar._scalar_unless_shapeless(self.vals & arg)

    # (|) operator
    def __or__(self, arg):
        if isinstance(arg, Scalar): arg = arg.vals
        return Scalar._scalar_unless_shapeless(self.vals | arg)

    # (^) operator
    def __xor__(self, arg):
        if isinstance(arg, Scalar): arg = arg.vals
        return Scalar._scalar_unless_shapeless(self.vals ^ arg)

    # (&=) operator
    def __iand__(self, arg):
        if isinstance(arg, Scalar): arg = arg.vals
        self.vals &= arg
        return self

    # (|=) operator
    def __ior__(self, arg):
        if isinstance(arg, Scalar): arg = arg.vals
        self.vals |= arg
        return self

    # (^=) operator
    def __ixor__(self, arg):
        if isinstance(arg, Scalar): arg = arg.vals
        self.vals ^= arg
        return self

    # This is needed for Scalars of booleans; it ensures that "if" tests execute
    # properly, because otherwise "if Scalar(False)" executes
    @staticmethod
    def _scalar_unless_shapeless(values):
        if np.shape(values) == (): return values
        return Scalar(values)

########################################
# UNIT TESTS
########################################

class Test_Scalar(unittest.TestCase):

    def runTest(self):

        # Arithmetic operations
        ints = Scalar((1,2,3))
        test = Scalar(np.array([1,2,3]))
        self.assertEqual(ints, test)

        test = Scalar(test)
        self.assertEqual(ints, test)

        self.assertEqual(ints, (1,2,3))
        self.assertEqual(ints, [1,2,3])

        self.assertEqual(ints.shape, [3])

        self.assertEqual(-ints, [-1,-2,-3])
        self.assertEqual(+ints, [1,2,3])

        self.assertEqual(ints, abs(ints))
        self.assertEqual(ints, abs(Scalar(( 1, 2, 3))))
        self.assertEqual(ints, abs(Scalar((-1,-2,-3))))

        self.assertEqual(ints * 2, [2,4,6])
        self.assertEqual(ints / 2., [0.5,1,1.5])
        self.assertEqual(ints / 2, [0,1,1])
        self.assertEqual(ints + 1, [2,3,4])
        self.assertEqual(ints - 0.5, (0.5,1.5,2.5))
        self.assertEqual(ints % 2, (1,0,1))

        self.assertEqual(ints + Scalar([1,2,3]), [2,4,6])
        self.assertEqual(ints - Scalar((1,2,3)), [0,0,0])
        self.assertEqual(ints * [1,2,3], [1,4,9])
        self.assertEqual(ints / [1,2,3], [1,1,1])
        self.assertEqual(ints % [1,3,3], [0,2,0])

        self.assertRaises(ValueError, ints.__add__, (4,5))
        self.assertRaises(ValueError, ints.__sub__, (4,5))
        self.assertRaises(ValueError, ints.__mul__, (4,5))
        self.assertRaises(ValueError, ints.__div__, (4,5))
        self.assertRaises(ValueError, ints.__mod__, (4,5))

        self.assertRaises(ValueError, ints.__add__, Scalar((4,5)))
        self.assertRaises(ValueError, ints.__sub__, Scalar((4,5)))
        self.assertRaises(ValueError, ints.__mul__, Scalar((4,5)))
        self.assertRaises(ValueError, ints.__div__, Scalar((4,5)))
        self.assertRaises(ValueError, ints.__mod__, Scalar((4,5)))

        ints += 1
        self.assertEqual(ints, [2,3,4])

        ints -= 1
        self.assertEqual(ints, [1,2,3])

        ints *= 2
        self.assertEqual(ints, [2,4,6])

        ints /= 2
        self.assertEqual(ints, [1,2,3])

        ints *= (3,2,1)
        self.assertEqual(ints, [3,4,3])

        ints /= (1,2,3)
        self.assertEqual(ints, [3,2,1])

        ints += (1,2,3)
        self.assertEqual(ints, 4)
        self.assertEqual(ints, [4])
        self.assertEqual(ints, [4,4,4])
        self.assertEqual(ints, Scalar([4,4,4]))

        ints -= (3,2,1)
        self.assertEqual(ints, [1,2,3])

        test = Scalar((10,10,10))
        test %= 4
        self.assertEqual(test, 2)

        test = Scalar((10,10,10))
        test %= (4,3,2)
        self.assertEqual(test, [2,1,0])

        test = Scalar((10,10,10))
        test %= Scalar((5,4,3))
        self.assertEqual(test, [0,2,1])

        self.assertRaises(ValueError, ints.__iadd__, (4,5))
        self.assertRaises(ValueError, ints.__isub__, (4,5))
        self.assertRaises(ValueError, ints.__imul__, (4,5))
        self.assertRaises(ValueError, ints.__idiv__, (4,5))
        self.assertRaises(ValueError, ints.__imod__, (4,5))

        self.assertRaises(ValueError, ints.__iadd__, Scalar((4,5)))
        self.assertRaises(ValueError, ints.__isub__, Scalar((4,5)))
        self.assertRaises(ValueError, ints.__imul__, Scalar((4,5)))
        self.assertRaises(ValueError, ints.__idiv__, Scalar((4,5)))
        self.assertRaises(ValueError, ints.__imod__, Scalar((4,5)))

        # Logical operations
        self.assertEqual(ints > 1,  [False, True,  True ])
        self.assertEqual(ints < 2,  [True,  False, False])
        self.assertEqual(ints >= 1, [True,  True,  True ])
        self.assertEqual(ints <= 2, [True,  True,  False])
        self.assertEqual(ints != 1, [False, True,  True ])
        self.assertEqual(ints == 2, [False, True,  False])

        self.assertEqual(ints >  Scalar(1),  [False, True,  True ])
        self.assertEqual(ints <  Scalar(2),  [True,  False, False])
        self.assertEqual(ints >= Scalar((1,4,3)), [True,  False, True ])
        self.assertEqual(ints <= Scalar((2,2,1)), [True,  True,  False])
        self.assertEqual(ints != [2,2,3], [True,  False, False])
        self.assertEqual(ints == (3,2,1), [False, True,  False])


        self.assertEqual(ints == (4,5), False)
        self.assertEqual(ints != (4,5), True)
        self.assertRaises(ValueError, ints.__gt__, (4,5))
        self.assertRaises(ValueError, ints.__lt__, (4,5))
        self.assertRaises(ValueError, ints.__ge__, (4,5))
        self.assertRaises(ValueError, ints.__le__, (4,5))

        self.assertRaises(ValueError, ints.__gt__, Scalar([4,5]))
        self.assertRaises(ValueError, ints.__lt__, Scalar([4,5]))
        self.assertRaises(ValueError, ints.__ge__, Scalar([4,5]))
        self.assertRaises(ValueError, ints.__le__, Scalar([4,5]))

        bools = Scalar(([True, True, True],[False, False, False]))
        self.assertEqual(bools.shape, [2,3])
        self.assertEqual(bools[0], True)
        self.assertEqual(bools[1], False)
        self.assertEqual(bools[:,0], (True, False))
        self.assertEqual(bools[:,:].swapaxes(0,1), (True, False))
        self.assertEqual(bools[:].swapaxes(0,1),   (True, False))
        self.assertEqual(bools.swapaxes(0,1),      (True, False))

        self.assertEqual(~bools.swapaxes(0,1), (False, True))
        self.assertEqual(~bools, ((False, False, False), (True, True, True)))
        self.assertEqual(~bools, Scalar([(False, False, False),
                                         (True,  True,  True )]))

        self.assertEqual(bools & True,  bools)
        self.assertEqual(bools & False, False)
        self.assertEqual(bools & (True,False,True), [[True, False,True ],
                                                     [False,False,False]])
        self.assertEqual(bools & Scalar(True),  bools)
        self.assertEqual(bools & Scalar(False), False)
        self.assertEqual(bools & Scalar((True,False,True)),[[True, False,True ],
                                                           [False,False,False]])

        self.assertEqual(bools | True,  True)
        self.assertEqual(bools | False, bools)
        self.assertEqual(bools | (True,False,True), [[True, True, True ],
                                                     [True, False,True ]])
        self.assertEqual(bools | Scalar(True),  True)
        self.assertEqual(bools | Scalar(False), bools)
        self.assertEqual(bools | Scalar((True,False,True)),[[True, True, True ],
                                                           [True, False,True ]])

        self.assertEqual((bools ^ True).swapaxes(0,1), (False,True))
        self.assertEqual(bools ^ True, ~bools)
        self.assertEqual(bools ^ False, bools)
        self.assertEqual(bools ^ (True,False,True), [[False,True, False],
                                                     [True, False,True ]])
        self.assertEqual((bools ^ Scalar(True)).swapaxes(0,1), (False,True))
        self.assertEqual( bools ^ Scalar(True), ~bools)
        self.assertEqual( bools ^ Scalar(False), bools)
        self.assertEqual( bools ^ Scalar((True,False,True)),[[False,True,False],
                                                            [True,False,True ]])

        self.assertEqual(bools == Scalar([True,True]), False)
        self.assertEqual(bools != Scalar([True,True]), True)

        bools &= bools
        self.assertEqual(bools, [[True, True, True],[False, False, False]])

        bools |= bools
        self.assertEqual(bools, [[True, True, True],[False, False, False]])

        test = bools.copy().swapaxes(0,1)
        test |= test
        self.assertEqual(test, [[True, False],[True, False],[True, False]])

        test ^= bools.swapaxes(0,1)
        self.assertEqual(test,  False)
        self.assertEqual(test,  (False,False))

        test[0] = True
        self.assertEqual(test, [[True, True],[False, False],[False, False]])

        test[1:,1] ^= True
        self.assertEqual(test, [[True, True],[False, True],[False, True]])

        test[1:,0] |= test[1:,1]
        self.assertEqual(test, True)

        self.assertRaises(ValueError, bools.__ior__,  (True, False))
        self.assertRaises(ValueError, bools.__iand__, (True, False))
        self.assertRaises(ValueError, bools.__ixor__, (True, False))

        self.assertRaises(ValueError, bools.__ior__,  Scalar((True, False)))
        self.assertRaises(ValueError, bools.__iand__, Scalar((True, False)))
        self.assertRaises(ValueError, bools.__ixor__, Scalar((True, False)))

        # Generic Array operations
        self.assertEqual(ints[0], 1)

        floats = ints.astype("float")
        self.assertEqual(floats[0], 1.)

        strings = ints.astype("string")
        self.assertEqual(strings[1], "2")

        six = Scalar([1,2,3,4,5,6])
        self.assertEqual(six.shape, [6])

        test = six.copy().reshape((3,1,2))
        self.assertEqual(test.shape, [3,1,2])
        self.assertEqual(test, [[[1,2]],[[3,4]],[[5,6]]])
        self.assertEqual(test.swapaxes(0,1).shape, [1,3,2])
        self.assertEqual(test.swapaxes(0,2).shape, [2,1,3])
        self.assertEqual(test.ravel().shape, [6])
        self.assertEqual(test.flatten().shape, [6])

        four = Scalar([1,2,3,4]).reshape((2,2))
        self.assertEqual(four, [[1,2],[3,4]])

        self.assertEqual(Array.broadcast_shape((four,test)), [3,2,2])
        self.assertEqual(four.rebroadcast((3,2,2)), [[[1,2],[3,4]],
                                                     [[1,2],[3,4]],
                                                     [[1,2],[3,4]]])
        self.assertEqual(test.rebroadcast((3,2,2)), [[[1,2],[1,2]],
                                                     [[3,4],[3,4]],
                                                     [[5,6],[5,6]]])

        ten = four + test
        self.assertEqual(ten.shape, [3,2,2])
        self.assertEqual(ten, [[[2, 4], [4, 6]],
                               [[4, 6], [6, 8]],
                               [[6, 8], [8,10]]])

        x24 = four * test
        self.assertEqual(x24.shape, [3,2,2])
        self.assertEqual(x24, [[[1, 4], [ 3, 8]],
                               [[3, 8], [ 9,16]],
                               [[5,12], [15,24]]])

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
