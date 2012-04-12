################################################################################
# oops_/array/array_.py: Abstract class Array
#
# Superclass of Empty, Scalar, Pair, Vector3, Matrix3 and Tuple. All represent
# N-dimensional arrays of intrinsically dimensional objects.
#
# Modified 12/12/2011 (BSW) - added help comments to most methods
#
# Modified 1/2/11 (MRS) -- Refactored Scalar.py to eliminate circular loading
#   dependencies with Array and Empty classes. Array.py, Scalar.py and Empty.py
#   are now separate files.
# Modified 2/8/12 (MRS) -- Supports array masks and units; uses a cleaner set of
#   definitions for math operators.
# 3/2/12 MRS: Supports subfields; further cleanup of arithmetic operators.
################################################################################

import numpy as np
import numpy.ma as ma

from oops_.units import Units

class Array(object):
    """A class defining an arbitrary Array of possibly multidimensional items.
    Unlike numpy ndarrays, this class makes a clear distinction between the
    dimensions associated with the items and any additional, leading dimensions
    which describe the set of such items. The shape is defined by the leading
    axes only, so a 2x2 array of 3x3 matrices would have shape (2,2,3,3)
    according to numpy but has shape (2,2) according to Array.

    The Array object is designed as a lightweight "wrapper" on the numpy
    ndarray and numpy.ma.MaskedArray. All standard mathematical operators and
    indexing/slicing options are defined. One can mix Array arithmetic with
    scalars, numpy arrays, masked arrays, or anything array-like. The standard
    numpy rules of broadcasting apply to operations performed on Arrays of
    different shapes.

    Array objects have the following attributes:
        shape       a list (not tuple) representing the leading dimensions.
        rank        the number of trailing dimensions belonging to the
                    individual items.
        item        a list (not tuple) representing the shape of the
                    individual items.
        vals        the array's data as a numpy array. The shape of this array
                    is object.shape + object.item. It the object has units,
                    these values are in the specified units.
        mask        the array's mask as a numpy boolean array. The array value
                    is True if the Array value at the same location is masked.
                    A single value of False indicates that the array is not
                    masked.
        units       the units of the array, if any. None indicates no units.

        subfields   a dictionary of the names and values of any subfields.
                    Subfields contain additonal information about an array,
                    typically its derivatives with respect to other quantities.
                    Each subfield also becomes accessible as an attribute of the
                    object, with the same name.
        subfield_math
                    True for subfields to be included in mathematical operations
                    and copies; False to ignore them. Default is True.

        mvals       a read-only property that presents tha vals and the mask as
                    a masked array.
    """

    def __new__(subtype, *arguments, **keywords):
        """Creates a pre-initialized object of the specified type."""
        obj = object.__new__(subtype)
        return obj

    def __init__(self, arg, mask, units, rank, item=None,
                            float=False, dimensionless=False):
        """Default constructor."""

        self.subfields = {}
        self.subfield_math = True

        # Convert the mask to boolean and to an array if necessary
        if np.shape(mask) == ():
            mask = bool(mask)
        else:
            mask = np.asarray(mask).astype("bool")

        # Interpret the data if it is something already of class Array
        if isinstance(arg, Array):
            if arg.rank == rank:
                mask = arg.mask | mask
                if units is None: units = arg.units
                self.subfields = arg.subfields
                arg = arg.vals
            else:
                raise ValueError("class " + type(arg).__name__ +
                                 " cannot be converted to class " +
                                 type(self).__name__)

        # Interpret the data as a masked array
        elif isinstance(arg, ma.MaskedArray):
            if arg.mask != ma.nomask:

                # Collapse the mask to the proper shape
                tempmask = arg.mask
                for r in range(rank):
                    tempmask = np.any(tempmask, axis=-1)
                mask = mask | tempmask

            arg = arg.data

        # Check the overall shape
        ashape = list(np.shape(arg))
        if len(ashape) < rank:
            raise ValueError("data array of shape " + str(ashape) +
                             " is incompatible with " + type(self).__name__ +
                             " of rank " + str(rank))

        # Fill in the vals field
        if ashape == []:
            if float:
                self.vals = float(arg)
            else:
                self.vals = arg
        else:
            if float:
                self.vals = np.asfarray(arg)
            else:
                self.vals = np.asarray(arg)

        # Fill in the remaining shape info
        self.rank  = rank
        if rank == 0:
            self.item = []
            self.shape = ashape
        else:
            self.item  = ashape[-rank:]
            self.shape = ashape[:-rank]

        # Validate the item shape if necessary
        if item is not None and self.item != item:
            raise ValueError("data array of shape " + str(ashape) +
                             " is incompatible with " + type(self).__name__ +
                             " with items of shape " + str(item))

        # Fill in the mask and confirm shape compatibility
        self.mask = mask
        if np.shape(self.mask) != () and list(self.mask.shape) != self.shape:
            raise ValueError("mask array of shape " + str(self.mask.shape) +
                             " is incompatible with " + type(self).__name__ +
                             " shape " + str(self.shape))

        # Fill in the units and confirm compatibility
        self.units = units
        if dimensionless and self.units is not None:
            raise ValueError("a " + type(self).__name__ +
                             " object cannot have units")

        return

    @property
    def mvals(self):
        # Construct something that behaves as a suitable mask
        if self.mask is False:
            newmask = ma.nomask
        elif self.mask is True:
            newmask = np.ones((len(self.shape) + self.rank) * [1], dtype="bool")
            (newmask, newvals) = np.broadcast_arrays(newmask, self.vals)
        elif self.rank > 0:
            newmask = self.mask.reshape(self.shape + self.rank * [1])
            (newmask, newvals) = np.broadcast_arrays(newmask, self.vals)
        else:
            newmask = self.mask

        # Return the masked array
        return ma.MaskedArray(self.vals, newmask)

    def expand_mask(self):
        """Expands the mask to an array if it is currently just a boolean."""

        if np.shape(self.mask) == ():
            if self.mask:
                self.mask = np.ones(self.shape, dtype="bool")
            else:
                self.mask = np.zeros(self.shape, dtype="bool")

    def collapse_mask(self):
        """Reduces the mask to a single boolean if possible."""

        if not np.any(self.mask):
            self.mask = False
        elif np.all(self.mask):
            self.mask = True

    @staticmethod
    def is_empty(arg):
        return isinstance(arg, Array.EMPTY_CLASS)

    @staticmethod
    def into_scalar(arg):
        """A static Array method to turn something into a Scalar."""
        return Array.SCALAR_CLASS.as_scalar(arg)

    # Conversions to specific classes, which will only work if items match
    def as_scalar(self):
        """Casts any Array subclass to a Scalar if possible."""
        return Array.SCALAR_CLASS.as_scalar(self)

    def as_pair(self):
        """Casts any Array subclass to a Pair if possible."""
        return Array.PAIR_CLASS.as_pair(self)

    def as_tuple(self):
        """Casts any Array subclass to a Tuple if possible."""
        return Array.TUPLE_CLASS.as_pair(self)

    def as_vector3(self):
        """Casts any Array subclass to a Vector3 if possible."""
        return Array.VECTOR3_CLASS.as_vector3(self)

    def as_vectorn(self):
        """Casts any Array subclass to a VectorN if possible."""
        return Array.VECTORN_CLASS.as_vectorn(self)

    def as_matrix3(self):
        """Casts any Array subclass to a Matrix3 if possible."""
        return Array.MATRIX3_CLASS.as_matrix3(self)

    def as_matrixn(self):
        """Casts any Array subclass to a MatrixN if possible."""
        return Array.MATRIXN_CLASS.as_matrixn(self)

    def masked_version(self):
        """Retuns on Array of the same subclass and shape, containing all
        masked values."""

        obj = Array.__new__(type(self))

        if type(self.vals) == type(np.ndarray):
            vals = np.zeros(self.item, dtype=self.vals.dtype)
            vals = np.broadcast_arrays(vals, self.vals)[0]
        elif type(self.vals) == type(0):
            vals = 0
        else:
            vals = 0.

        obj.__init__(vals, True, self.units)
        return obj

    @classmethod
    def all_masked(cls, shape=[], item=None):
        """Returns an entirely masked object of the specified subclass and
        shape. The shape is emulated via broadcasting so the object should be
        treated as immutable."""

        temp = Array.__new__(cls)
        if item is None: item = [3,3]   # Any subclass can be initialized with a
                                        # 3x3 array. This is a bit of a kluge.
        temp.__init__(np.ones(item))

        obj = Array.__new__(cls)
        obj.__init__(np.zeros(temp.item), mask=True)
        return obj.rebroadcast(shape)

    def __repr__(self):
        """show values of Array or subtype. repr() call returns array at start
            of string, replace with object type, else just prefix entire string
            with object type."""

        return self.__str__()

    def __str__(self):
        """show values of Array or subtype. repr() call returns array at start
            of string, replace with object type, else just prefix entire string
            with object type."""

        suffix = ""

        is_masked = np.any(self.mask)
        if is_masked:
            suffix += ", mask"

        if self.units is not None:
            suffix += ", " + str(self.units)

        if np.shape(self.vals) == ():
            if is_masked:
                string = "--"
            else:
                string = str(self.vals)

        elif is_masked:
            masked_array = self.vals.view(ma.MaskedArray)

            if self.rank == 0:
                masked_array.mask = np.asarray(self.mask)
            else:
                temp_mask = np.asarray(self.mask)
                temp_mask = temp_mask.reshape(temp_mask.shape +
                                              self.rank * (1,))
                masked_array.mask = np.empty(self.vals.shape, dtype="bool")
                masked_array.mask[...] = temp_mask

            string = str(masked_array)
        else:
            string = str(self.vals)

        if string[0] == "[":
            return type(self).__name__ + string[:-1] + suffix + "]"
        else:
            return type(self).__name__ + "(" + string + suffix + ")"

    def __copy__(self):

        if isinstance(self.vals, np.ndarray):
            vals = self.vals.copy()
        else:
            vals = self.vals

        if isinstance(self.mask, np.ndarray):
            mask = self.mask.copy()
        else:
            mask = self.mask

        obj = Array.__new__(type(self))
        obj.__init__(vals, mask, self.units)

        if self.subfield_math:
            for key in self.subfields.keys():
                obj.insert_subfield(key, self.subfields[key].__copy__())

        return obj

    def copy(self, subfields=True):
        """If optional argument subfields is False, subfields will not be
        copied, overriding the internal subfield_math flag."""

        if subfields or not self.subfield_math: return self.__copy__()

        self.subfield_math = False
        result = self.__copy__()
        self.subfield_math = True

        return result

    def plain(self):
        """Returns a shallow copy of the object without subfields."""

        if len(self.subfields) == 0: return self

        obj = Array.__new__(type(self))
        obj.__init__(self.vals, self.mask, self.units)
        return obj

    def unmasked(self):
        """Returns a shallow copy of the object without a mask."""

        if self.mask is False: return self

        obj = Array.__new__(type(self))
        obj.__init__(self.vals, False, self.units)
        return obj

    ####################################################
    # Subarray support methods
    ####################################################

    def insert_subfield(self, key, value):
        """Inserts or replaces a subfield."""

        self.subfields[key] = value
        self.__dict__[key] = value

    def add_to_subfield(self, key, value):
        """Adds the given value to a subfield if it is already present;
        otherwise, it creates a new subfield."""

        if key in self.subfields.keys():
            self.subfields[key] = self.subfields[key] + value
        else:
            self.subfields[key] = value

        self.__dict__[key] = self.subfields[key]

    def insert_subfield_if_new(self, key, value):
        """Inserts the subfield if it does not already exist; otherwise, it
        leaves the subfield alone."""

        if key in self.subfields.keys(): return
        self.insert_subfield(key, value)

    def delete_subfield(self, key):
        """Deletes a subfield."""

        if key in self.subfields.keys():
            del self.subfields[key]
            del self.__dict__[key]

    def delete_subfields(self):
        """Deletes all subfields."""

        for key in self.subfields.keys():
            del self.__dict__[key]

        self.subfields = {}

    ####################################################
    # Arithmetic support methods
    ####################################################

    def as_my_type(self, arg, mask=False, units=None):
        """This method converts an operand to the same class as self, and then
        returns the operand."""

        if not isinstance(arg, type(self)):
            obj = Array.__new__(type(self))
            obj.__init__(arg, mask, units)
            arg = obj

        return arg

    def as_if_my_type(self, arg):
        """This method converts an operand to the same class as self, and then
        returns a tuple (values, mask, units)."""

        arg = self.as_my_type(arg)
        return (arg.vals, arg.mask, arg.units)

    def as_if_my_rank(self, arg):
        """This method converts an operand to the same rank as self, making it
        suitable for Numpy multiplication or division with self. It returns a
        a tuple (values, mask, units)."""

        arg = Array.into_scalar(arg)

        if self.rank > 0:
            vals = np.reshape(arg.vals, arg.shape + self.rank * [1])
        else:
            vals = arg.vals

        return (vals, arg.mask, arg.units)

    def as_if_my_units(self, vals, units):
        """This method converts an operand to the same units as self, making it
        suitable for Numpy addition or subtraction with self. It returns a
        a tuple (values, mask, units)."""

        # Find the common units
        if units is None:
            return (vals, self.units)

        if self.units is None:
            return (vals, units)

        return (units.convert(vals, self.units), self.units)

    ####################################################
    # Unary operators
    ####################################################

    def __pos__(self):
        return self

    def __neg__(self):
        obj = Array.__new__(type(self))
        obj.__init__(-self.vals, self.mask, self.units)

        if self.subfield_math:
            for key in self.subfields.keys():
                obj.insert_subfield(key, -self.subfields[key])

        return obj

    def __abs__(self):
        obj = Array.__new__(type(self))
        obj.__init__(np.abs(self.vals), self.mask, self.units)

        # This operation does not preserve subfields

        return obj

    ####################################################
    # Addition
    ####################################################

    def __add__(self, arg):
        if Array.is_empty(arg): return arg
        (vals, mask, units) = self.as_if_my_type(arg)
        (vals, units) = self.as_if_my_units(vals, units)
        obj = Array.__new__(type(self))
        obj.__init__(self.vals + vals, self.mask | mask, units)
        obj.units = units
        obj.add_subfields(self, arg)
        return obj

    def __radd__(self, arg): return self.__add__(arg)

    def __iadd__(self, arg):
        (vals, mask, units) = self.as_if_my_type(arg)
        (vals, units) = self.as_if_my_units(vals, units)
        self.vals += vals
        self.mask |= mask
        self.units = units
        self.iadd_subfields(arg)
        return self

    def add_subfields(self, arg1, arg2):
        if not arg1.subfield_math: return
        if isinstance(arg2, Array):
            if not arg2.subfield_math: return

            set1 = set(arg1.subfields)
            set2 = set(arg2.subfields)
            set12 = set1 & set2
            set1 -= set12
            set2 -= set12
            for key in set12:
                self.insert_subfield(key, arg1.subfields[key] +
                                          arg2.subfields[key])
            for key in set1:
                self.insert_subfield(key, arg1.subfields[key].copy())
            for key in set2:
                self.insert_subfield(key, arg2.subfields[key].copy())
        else:
            for key in arg1.subfields.keys():
                self.insert_subfield(key, arg1.subfields[key].copy())

    def iadd_subfields(self, arg2):
        if not self.subfield_math: return
        if isinstance(arg2, Array):
            if not arg2.subfield_math: return

            set1 = set(self.subfields)
            set2 = set(arg2.subfields)
            set12 = set1 & set2
            set2 -= set12
            for key in set12:
                self.subfields[key] += arg2.subfields[key]
                self.__dict__[key] = self.subfields[key]
            for key in set2:
                self.insert_subfield(key, arg2.subfields[key])

    ####################################################
    # Subtraction
    ####################################################

    def __sub__(self, arg):
        if Array.is_empty(arg): return arg
        (vals, mask, units) = self.as_if_my_type(arg)
        (vals, units) = self.as_if_my_units(vals, units)
        obj = Array.__new__(type(self))
        obj.__init__(self.vals - vals, self.mask | mask, units)
        obj.sub_subfields(self, arg)
        return obj

    def __rsub__(self, arg):
        if Array.is_empty(arg): return arg
        (vals, mask, units) = self.as_if_my_type(arg)
        (vals, units) = self.as_if_my_units(vals, units)
        obj = Array.__new__(type(self))
        obj.__init__(vals - self.vals, self.mask | mask, units)
        obj.add_subfields(-self, arg)
        return obj

    def __isub__(self, arg):
        (vals, mask, units) = self.as_if_my_type(arg)
        (vals, units) = self.as_if_my_units(vals, units)
        self.vals -= vals
        self.mask |= mask
        self.units = units
        self.isub_subfields(arg)
        return self

    def sub_subfields(self, arg1, arg2):
        if not arg1.subfield_math: return
        if isinstance(arg2, Array):
            if not arg2.subfield_math: return

            set1 = set(arg1.subfields)
            set2 = set(arg2.subfields)
            set12 = set1 & set2
            set1 -= set12
            set2 -= set12
            for key in set12:
                self.insert_subfield(key, arg1.subfields[key] -
                                          arg2.subfields[key])
            for key in set1:
                self.insert_subfield(key, arg1.subfields[key].copy())
            for key in set2:
                self.insert_subfield(key, -arg2.subfields[key].copy())
        else:
            for key in arg1.subfields.keys():
                self.insert_subfield(key, arg1.subfields[key].copy())

    def isub_subfields(self, arg2):
        if not self.subfield_math: return
        if isinstance(arg2, Array):
            if not arg2.subfield_math: return

            set1 = set(self.subfields)
            set2 = set(arg2.subfields)
            set12 = set1 & set2
            set2 -= set12
            for key in set12:
                self.subfields[key] -= arg2.subfields[key]
                self.__dict__[key] = self.subfields[key]
            for key in set2:
                self.insert_subfield(key, -arg2.subfields[key])

    ####################################################
    # Multiplication
    ####################################################

    def __mul__(self, arg):
        if Array.is_empty(arg): return arg

        try:
            (vals, mask, units) = self.as_if_my_type(arg)
        except Exception as e:
            try:
                return Array.into_scalar(arg).__mul__(self)
            except:
                raise e

        obj = Array.__new__(type(self))
        obj.__init__(self.vals * vals,
                     self.mask | mask,
                     Units.mul_units(self.units, units))

        obj.mul_subfields(self, arg)

        return obj

    # Reverse-multiply if forward multiply fails
    # This takes care of 2 * Array, for example
    def __rmul__(self, arg):
        return self.__mul__(arg)

    def __imul__(self, arg):
        try:
            (vals, mask, units) = self.as_if_my_type(arg)
        except Exception as e:
            try:
                (vals, mask, units) = self.as_if_my_rank(arg)
            except:
                raise e

        self.vals *= vals
        self.mask |= mask
        self.units = Units.mul_units(self.units, units)

        self.imul_subfields(arg)

        return self

    def mul_subfields(self, arg1, arg2):
        if not arg1.subfield_math: return
        if isinstance(arg2, Array):
            if not arg2.subfield_math: return

            set1 = set(arg1.subfields)
            set2 = set(arg2.subfields)
            set12 = set1 & set2
            set1 -= set12
            set2 -= set12
            for key in set12:
                self.insert_subfield(key, arg1.subfields[key] *
                                          arg2.subfields[key])
            for key in set1:
                self.insert_subfield(key, arg1.subfields[key].copy())
            for key in set2:
                self.insert_subfield(key, arg2.subfields[key].copy())
        else:
            for key in arg1.subfields.keys():
                self.insert_subfield(key, arg1.subfields[key].copy())

    def imul_subfields(self, arg2):
        if not self.subfield_math: return
        if isinstance(arg2, Array):
            if not arg2.subfield_math: return

            set1 = set(self.subfields)
            set2 = set(arg2.subfields)
            set12 = set1 & set2
            set2 -= set12
            for key in set12:
                self.subfields[key] *= arg2.subfields[key]
                self.__dict__[key] = self.subfields[key]
            for key in set2:
                self.insert_subfield(key, arg2.subfields[key])

    ####################################################
    # Division
    ####################################################

    def __div__(self, arg):
        if Array.is_empty(arg): return arg

        try:
            (vals, mask, units) = self.as_if_my_type(arg)
        except Exception as e:
            try:
                return Array.into_scalar(arg).__rdiv__(self)
            except:
                raise e

        # Mask any items to be divided by zero
        div_by_zero = (vals == 0)
        if np.any(div_by_zero):
            if np.shape(vals) == ():
                vals = 1
            else:
                vals = vals.copy()
                vals[div_by_zero] = 1
                # Collapse mask down to one element per item of Array
                for iters in range(self.rank):
                    div_by_zero = np.any(div_by_zero, axis=-1)
        else:
            div_by_zero = False

        obj = Array.__new__(type(self))
        obj.__init__(self.vals / vals,
                     self.mask | mask | div_by_zero,
                     Units.div_units(self.units, units))

        obj.div_subfields(self, arg)

        return obj

    # Reverse-divide works element-by-element if rank is 0 or 1.
    def __rdiv__(self, arg):
        if self.rank == 2: return NotImplemented

        try:
            arg = self.as_my_type(arg)
            return arg / self
        except: pass

        (vals, mask, units) = self.as_if_my_rank(arg)
        vals = np.broadcast_arrays(vals, self.vals)[0]
        arg = self.as_my_type(vals, mask, units)
        return arg / self

    def __idiv__(self, arg):
        try:
            (vals, mask, units) = self.as_if_my_type(arg)
        except Exception as e:
            try:
                (vals, mask, units) = self.as_if_my_rank(arg)
            except:
                raise e

        # Mask any items to be divided by zero
        div_by_zero = (vals == 0)
        if np.any(div_by_zero):
            if np.shape(vals) == ():
                vals = 1
            else:
                vals = vals.copy()
                vals[div_by_zero] = 1
                # Collapse mask down to one element per item of Array
                for iters in range(self.rank):
                    div_by_zero = np.any(div_by_zero, axis=-1)
        else:
            div_by_zero = False

        self.vals /= vals
        self.mask |= (mask | div_by_zero)
        self.units = Units.div_units(self.units, units)

        self.idiv_subfields(arg)

        return self

    def div_subfields(self, arg1, arg2):
        if not arg1.subfield_math: return
        if isinstance(arg2, Array):
            if not arg2.subfield_math: return

            set1 = set(arg1.subfields)
            set2 = set(arg2.subfields)
            set12 = set1 & set2
            set1 -= set12
            set2 -= set12
            for key in set12:
                self.insert_subfield(key, arg1.subfields[key] /
                                          arg2.subfields[key])
            for key in set1:
                self.insert_subfield(key, arg1.subfields[key].copy())
            for key in set2:
                self.insert_subfield(key, 1./arg2.subfields[key].copy())
        else:
            for key in arg1.subfields.keys():
                self.insert_subfield(key, arg1.subfields[key].copy())

    def idiv_subfields(self, arg2):
        if not self.subfield_math: return
        if isinstance(arg2, Array):
            if not arg2.subfield_math: return

            set1 = set(self.subfields)
            set2 = set(arg2.subfields)
            set12 = set1 & set2
            set2 -= set12
            for key in set12:
                self.subfields[key] /= arg2.subfields[key]
                self.__dict__[key] = self.subfields[key]
            for key in set2:
                self.insert_subfield(key, 1./arg2.subfields[key])

    ####################################################
    # Other operators
    #
    # These operations never preserve subfields
    ####################################################

    def __mod__(self, arg):
        if Array.is_empty(arg): return arg

        try:
            (vals, mask, units) = self.as_if_my_type(arg)
        except Exception as e:
            try:
                return Array.into_scalar(arg).__rmod__(self)
            except:
                raise e

        # Mask any items to be divided by zero
        div_by_zero = (vals == 0)
        if np.any(div_by_zero):
            if np.shape(vals) == ():
                vals = 1
            else:
                vals = vals.copy()
                vals[div_by_zero] = 1
                # Collapse mask down to one element per item of Array
                for iters in range(self.rank):
                    div_by_zero = np.any(div_by_zero, axis=-1)
        else:
            div_by_zero = False

        obj = Array.__new__(type(self))
        obj.__init__(self.vals % vals,
                     self.mask | mask | div_by_zero,
                     Units.div_units(self.units, units))

        obj.div_subfields(self, arg)

        return obj

    def __imod__(self, arg):
        try:
            (vals, mask, units) = self.as_if_my_type(arg)
        except Exception as e:
            try:
                (vals, mask, units) = self.as_if_my_rank(arg)
            except:
                raise e

        div_by_zero = (vals == 0)
        if np.any(div_by_zero):
            if np.shape(vals) == ():
                vals = 1
            else:
                vals = vals.copy()
                vals[div_by_zero] = 1
        else:
            div_by_zero = False

        self.vals %= vals
        self.mask |= (mask | div_by_zero)
        self.units = Units.div_units(self.units, units)

        self.idiv_subfields(arg)

        return self

    def __pow__(self, arg):
        if Array.is_empty(arg): return arg

        new_mask = False
        if arg <= 0:
            new_mask = new_mask | (self.vals == 0.)
        if type(arg) != type(0):
            new_mask = new_mask | (self.vals < 0.)

        if np.any(new_mask):
            if np.shape(self.vals) == ():
                vals = 1.
            else:
                vals = self.vals.copy()
                vals[new_mask] = 1.
        else:
            vals = self.vals

        obj = Array.__new__(type(self))
        obj.__init__(self.vals**arg, self.mask | new_mask,
                                     Units.units_power(self.units, arg))
        return obj

    ####################################
    # Default comparison operators
    ####################################

    def __eq__(self, arg):

        # If the subclasses cannot be unified, the objects are unequal
        if not isinstance(arg, type(self)):
            try:
                obj = Array.__new__(type(self))
                obj.__init__(arg)
                arg = obj
            except:
                return False

        # If the units are incompatible, the objects are unequal
        # If units are compatible, convert to the same units
        if self.units is not None and arg.units is not None:
            if self.units.exponents != arg.units.exponents:
                return False
            arg = arg.convert_units(self.units)
        else:
            if self.units != arg.units:
                return False

        # The comparison is easy if the shape is []
        if np.shape(self.vals) == () and np.shape(arg.vals) == ():
            if self.mask and arg.mask: return True
            if self.mask != arg.mask: return False
            return self.vals == arg.vals

        # Compare the values
        compare = (self.vals == arg.vals)

        # Collapse the rightmost axes based on rank
        for i in range(self.rank):
            compare = np.all(compare, axis=-1)

        # Quick test: If both masks are empty, just return the comparison
        if (not np.any(self.mask) and not np.any(arg.mask)):
            return Array.into_scalar(compare)

        # Otherwise, perform the detailed comparison
        compare[self.mask & arg.mask] = True
        compare[self.mask ^ arg.mask] = False
        return Array.into_scalar(compare)

    def __ne__(self, arg):

        # If the subclasses cannot be unified, the objects are unequal
        if not isinstance(arg, type(self)):
            try:
                obj = Array.__new__(type(self))
                obj.__init__(arg)
                arg = obj
            except:
                return True

        # If the units are incompatible, the objects are unequal
        # If units are compatible, convert to the same units
        if self.units is not None and arg.units is not None:
            if self.units.exponents != arg.units.exponents: return True
            arg = arg.convert_units(self.units)
        else:
            if self.units != arg.units: return False

        # The comparison is easy if the shape is []
        if np.shape(self.vals) == () and np.shape(arg.vals) == ():
            if self.mask and arg.mask: return False
            if self.mask != arg.mask: return True
            return self.vals != arg.vals

        # Compare the values
        compare = (self.vals != arg.vals)

        # Collapse the rightmost axes based on rank
        for i in range(self.rank):
            compare = np.any(compare, axis=-1)

        # Quick test: If both masks are empty, just return the comparison
        if (not np.any(self.mask) and not np.any(arg.mask)):
            return Array.into_scalar(compare)

        # Otherwise, perform the detailed comparison
        compare[self.mask & arg.mask] = False
        compare[self.mask ^ arg.mask] = True
        return Array.into_scalar(compare)

    def __nonzero__(self):
        """This is the test performed by an if clause."""

        if self.mask is False:
            return bool(np.all(self.vals))
        elif np.all(self.mask):
            raise ValueError("the truth value of an entirely masked object " +
                             "is undefined.")
        return bool(np.all(self.vals[~self.mask]))

    ####################################
    # Indexing operators
    ####################################

    def __getitem__(self, i):

        # Handle a shapeless Array
        if self.shape == []:
            if i is True: return self
            if i is False: return Array.masked_version(self)
            raise IndexError("too many indices")

        # Handle an index as a boolean
        if i is True: return self
        if i is False: return Array.masked_version(self)

        # Get the value and mask
        vals = self.vals[i]

        if np.shape(self.mask) == ():
            mask = self.mask
        else:
            mask = self.mask[i]

        # Make sure we have not penetrated the components of a 1-D or 2-D item
        icount = 1
        if type(i) == type(()): icount = len(i)
        if icount > len(self.shape): raise IndexError("too many indices")

        # If the result is a single, unmasked, unitless value, return it as a
        # number
        if np.shape(vals) == () and not np.any(mask) and self.units is None:
            return vals

        # Construct the object and return it
        obj = Array.__new__(type(self))
        obj.__init__(vals, mask, self.units)
        return obj

    def __getslice__(self, i, j):

        # Get the values and mask
        vals = self.vals[i:j]

        if np.shape(self.mask) == ():
            mask = self.mask
        else:
            mask = self.mask[i:j]

        # Make sure we have not penetrated the components of a 1-D or 2-D item
        icount = 1
        if type(i) == type(()): icount = len(i)
        if icount > len(self.shape): raise IndexError("too many indices")

        # If the result is a single, unmasked, unitless value, return it as a
        # number
        if np.shape(vals) == () and not np.any(mask) and self.units is None:
            return vals

        # Construct the object and return it
        obj = Array.__new__(type(self))
        obj.__init__(vals, mask, self.units)
        return obj

#     def __setitem__(self, i, arg):
# 
#         # Handle a single boolean index
#         if i is False:
#             return self
#         if i is True:
#             obj = Array.__new__(type(self))
#             obj.__init__(arg)
# 
#             if np.shape(self.vals) == ():
#                 self.vals = obj.vals
#                 self.mask = obj.mask
#             else:
#                 self.vals[...] = obj.vals
#                 if np.shape(self.mask) == ():
#                     self.mask = obj.mask
#                 else:
#                     self.mask[...] = obj.mask
# 
#             self.units = obj.units
#             return self
# 
#         # Get the values and mask after converting arg to the same subclass
#         (vals, mask, units) = self.as_if_my_type(arg)
#         (vals, units) = self.as_if_my_units(vals, units)
# 
#         # Replace the value(s)
#         self.vals[i] = vals
# 
#         # If the mask is already an array, replace the mask value
#         if np.shape(self.mask) != ():
#             self.mask[i] = mask
# 
#         # Otherwise, if the mask values disagree...
#         elif np.any(self.mask != mask):
# 
#             # Replace the mask with a boolean array, then fill in the new mask
#             newmask = np.empty(self.shape, dtype="bool")
#             newmask[...] = self.mask
#             newmask[i] = mask
#             self.mask = newmask
# 
#         return self
# 
#     def __setslice__(self, i, j, arg):
# 
#         # Get the values and mask after converting arg to the same subclass
#         (vals, mask, units) = self.as_if_my_type(arg)
#         (vals, units) = self.as_if_my_units(vals, units)
# 
#         # Replace the value(s)
#         self.vals[i:j] = vals
# 
#         # If the mask is already an array, replace the mask value
#         if np.shape(self.mask) != ():
#             self.mask[i:j] = mask
# 
#         # Otherwise, if the mask values disagree...
#         elif np.any(self.mask != mask):
# 
#             # Replace the mask with a boolean array, then fill in the new mask
#             newmask = np.empty(self.shape, dtype="bool")
#             newmask[...] = self.mask
#             newmask[i:j] = mask
#             self.mask = newmask
# 
#         return self

    ####################################
    # Value Transformations
    ####################################

    def astype(self, dtype):
        """converts dtype of elements of Array to said type. creates new
            instance of Array and returns it.
            
            Input:
            dtype:      type, such as int, float32, etc.
            
            Return:     new instance of Array with converted elements.
            """

        if isinstance(self.vals, np.ndarray):
            if self.vals.dtype == dtype: return self
            vals = self.vals.astype(dtype)
        else:
            vals = np.array([self.vals], dtype=dtype)[0]

        obj = Array.__new__(type(self))
        obj.__init__(vals, self.mask, self.units)
        return obj

    # Note that these three methods have slightly different methods for how to
    # handle units of None.

    def convert_units(self, units):
        """Returns the same Array but with the given target units. If the Array
        does not have any units, its units are assumed to be standard units
        of km, seconds and radians, which are converted to the target units."""

        if self.units == units: return self

        obj = Array.__new__(type(self))

        if units is None:
            obj.__init__(self.units.to_standard(self.vals), self.mask, None)
        elif self.units is None:
            obj.__init__(units.to_units(self.vals), self.mask, units)
        else:
            obj.__init__(self.units.convert(self.vals, units), self.mask, units)

        return obj

    def attach_units(self, units):
        """Returns the same Array but with the given target units. If the Array
        does not have any units, the units are assumed to be target units
        already and the values are returned unchanged. Arrays with different
        units are converted to the target units."""

        if self.units == units: return self

        obj = Array.__new__(type(self))

        if self.units is None or units is None:
            obj.__init__(self.vals, self.mask, units)
        else:
            obj.__init__(self.units.convert(self.vals, units), self.mask, units)

        return obj

    def confirm_units(self, units):
        """Returns the same Array but with the given target units. If the Array
        does not have any units, the values are assumed to be unitless. Arrays
        with different units are converted to the target units."""

        if self.units == units: return self

        obj = Array.__new__(type(self))

        self_units = self.units
        if self_units is None: self_units = Units.UNITLESS

        obj.__init__(self_units.convert(self.vals, units), self.mask, units)
        return obj

    ####################################
    # Shaping functions
    ####################################

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

        if axis1 == axis2: return self

        vals = self.vals.swapaxes(axis1, axis2)

        if np.shape(self.mask) == ():
            mask = self.mask
        else:
            mask = self.mask.swapaxes(axis1, axis2)

        obj = Array.__new__(type(self))
        obj.__init__(vals, mask, self.units)

        for key in self.subfields.keys():
            subfield = self.subfields[key]
            if isinstance(subfield, Array):
                obj.insert_subfield(key, subfield.swapaxes(axis1,axis2))
            else:
                obj.insert_subfield(key, subfield)

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

        vals = self.vals.reshape(list(shape) + self.item)

        if np.shape(self.mask) == ():
            mask = self.mask
        else:
            mask = self.mask.reshape(shape)

        obj = Array.__new__(type(self))
        obj.__init__(vals, mask, self.units)

        for key in self.subfields.keys():
            subfield = self.subfields[key]
            if isinstance(subfield, Array):
                obj.insert_subfield(key, subfield.reshape(shape))
            else:
                obj.insert_subfield(key, subfield)

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
        vals = self.vals.reshape([count] + self.item)

        if np.shape(self.mask) == ():
            mask = self.mask
        else:
            mask = self.mask.reshape((count))

        obj = Array.__new__(type(self))
        obj.__init__(vals, mask, self.units)

        for key in self.subfields.keys():
            subfield = self.subfields[key]
            if isinstance(subfield, Array):
                obj.insert_subfield(key, subfield.flatten())
            else:
                obj.insert_subfield(key, subfield)

        return obj

    def ravel(self): return self.flatten()

    def reorder_axes(self, axes):
        """Puts the leading axes into the specified order. Item axes are
        unchanged."""

        allaxes = axes + range(len(self.shape), len(self.vals.shape))

        vals = self.vals.transpose(allaxes)

        if np.shape(self.mask) == ():
            mask = self.mask
        else:
            mask = self.mask.transpose(allaxes[:len(self.shape)])

        obj = Array.__new__(type(self))
        obj.__init__(vals, mask, self.units)

        for key in self.subfields.keys():
            subfield = self.subfields[key]
            if isinstance(subfield, Array):
                obj.insert_subfield(key, subfield.reorder_axes(axes))
            else:
                obj.insert_subfield(key, subfield)

        return obj

    def append_axes(self, axes):
        """Appends the specified number of unit axes to the end of the shape."""

        if axes == 0: return self

        vals = self.vals.reshape(self.shape + axes*[1] + self.item)

        if np.shape(self.mask) == ():
            mask = self.mask
        else:
            mask = self.mask.reshape(self.shape + axes*[1])

        obj = Array.__new__(type(self))
        obj.__init__(vals, mask, self.units)

        for key in self.subfields.keys():
            subfield = self.subfields[key]
            if isinstance(subfield, Array):
                obj.insert_subfield(key, subfield.append_axes(axes))
            else:
                obj.insert_subfield(key, subfield)

        return obj

    def prepend_axes(self, axes):
        """Prepends the specified number of unit axes to the end of the shape.
        """

        if axes == 0: return self

        vals = self.vals.reshape(axes*[1] + self.shape + self.item)

        if np.shape(self.mask) == ():
            mask = self.mask
        else:
            mask = self.mask.reshape(axes*[1] + self.shape)

        obj = Array.__new__(type(self))
        obj.__init__(vals, mask, self.units)

        for key in self.subfields.keys():
            subfield = self.subfields[key]
            if isinstance(subfield, Array):
                obj.insert_subfield(key, subfield.prepend_axes(axes))
            else:
                obj.insert_subfield(key, subfield)

        return obj

    def strip_axes(self, axes):
        """Removes unit axes from the beginning of an Array's shape."""

        newshape = self.shape
        while len(newshape) > 0 and newshape[0] == 1: newshape = newshape[1:]

        vals = self.vals.reshape(newshape + self.item)

        if np.shape(self.mask) == ():
            mask = self.mask
        else:
            mask = self.mask.reshape(newshape)

        obj = Array.__new__(type(self))
        obj.__init__(vals, mask, self.units)

        for key in self.subfields.keys():
            subfield = self.subfields[key]
            if isinstance(subfield, Array):
                obj.insert_subfield(key, subfield.strip_axes(axes))
            else:
                obj.insert_subfield(key, subfield)

        return obj

    def rotate_axes(self, axis):
        """Rotates the axes until the specified axis comes first; leading axes
        are moved to the end but the item axes are unchanged."""

        allaxes = (range(axis, axis + len(self.shape)) +
                   range(axis) +
                   range(len(self.shape), len(self.vals.shape)))

        vals = self.vals.transpose(allaxes)

        if np.shape(self.mask) == ():
            mask = self.mask
        else:
            mask = self.mask.transpose(allaxes[:len(self.shape)])

        obj = Array.__new__(type(self))
        obj.__init__(vals, mask, self.units)

        for key in self.subfields.keys():
            subfield = self.subfields[key]
            if isinstance(subfield, Array):
                obj.insert_subfield(key, subfield.rotate_axes(axis))
            else:
                obj.insert_subfield(key, subfield)

        return obj

    def rebroadcast(self, newshape):
        """Returns an Array broadcasted to the specified shape. It returns self
        if the shape already matches. Otherwise, the returned object shares data
        with the original and should be treated as read-only."""

        newshape = list(newshape)
        if newshape == self.shape: return self

        temp = np.empty(newshape + self.item, dtype="byte")
        vals = np.broadcast_arrays(self.vals, temp)[0]

        if isinstance(self.mask, np.ndarray):
            temp = np.empty(newshape, dtype="byte")
            mask = np.broadcast_arrays(self.mask, temp)[0]
        else:
            mask = self.mask

        obj = Array.__new__(type(self))
        obj.__init__(vals, mask, self.units)

        for key in self.subfields.keys():
            subfield = self.subfields[key]
            if isinstance(subfield, Array):
                obj.insert_subfield(key, subfield.rebroadcast(newshape))
            else:
                obj.insert_subfield(key, subfield)

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
                        definition of an additional shape. Scalars have shape
                        ().

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
            if array is None: continue

            # Get the next shape
            try:
                shape = list(array.shape)
            except AttributeError:
                if type(array) == type(()) or type(array) == type([]):
                    shape = list(array)
                else:
                    shape = list(np.shape(array))

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

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Array(unittest.TestCase):

    # No tests here. Everything is tested by subclasses

    def runTest(self):
        pass

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
