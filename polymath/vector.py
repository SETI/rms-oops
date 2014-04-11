################################################################################
# polymath/modules/vector.py: Vector subclass of PolyMath base class
#
# Mark Showalter, PDS Rings Node, SETI Institute, February 2014
################################################################################

from __future__ import division
import numpy as np

from qube   import Qube
from scalar import Scalar
from units  import Units

class Vector(Qube):
    """A PolyMath subclass containing 1-D vectors of arbitrary length.
    """

    NRANK = 1           # the number of numerator axes.
    NUMER = None        # shape of the numerator.

    FLOATS_OK = True    # True to allow floating-point numbers.
    INTS_OK = True      # True to allow integers.
    BOOLS_OK = False    # True to allow booleans.

    UNITS_OK = True     # True to allow units; False to disallow them.
    MASKS_OK = True     # True to allow masks; False to disallow them.
    DERIVS_OK = True    # True to disallow derivatives; False to allow them.

    @staticmethod
    def as_vector(arg, recursive=True):
        """Return the argument converted to Scalar if possible.

        If recursive is True, derivatives will also be converted. However, note
        that derivatives are not necessarily removed when recursive is False.
        """

        if type(arg) == Vector:
            if recursive: return arg
            return arg.without_derivs()

        if isinstance(arg, Qube):

            # Convert any 1-D object
            if arg.nrank == 1:
                return arg.flatten_numer(Vector, recursive)

            # Collapse a 1xN or Nx1 MatrixN down to a Vector
            if arg.nrank == 2 and (arg.numer[0] == 1 or arg.numer[1] == 1):
                return arg.flatten_numer(Vector, recursive)

            # Convert Scalar to shape (1,)
            if arg.nrank == 0:
                return arg.flatten_numer(Vector, recursive)

            # For any other Qube, move numerator items to the denominator
            if arg.rank > 1:
                return arg.split_items(1, Vector)

            arg = Vector(arg, example=arg)
            if recursive: return arg
            return arg.without_derivs()

        return Vector(arg)

    def to_scalar(self, axis, recursive=True):
        """Return one of the components of a Vector as a Scalar.

        Input:
            axis        axis index.
            recursive   True to extract the derivatives as well.
        """

        return self.extract_numer(0, axis, Scalar, recursive)

    def to_scalars(self, recursive=True):
        """Return the components of a Vector as a tuple of Scalars.

        Input:
            recursive   True to include the derivatives.
        """

        results = []
        for i in range(self.numer[0]):
            results.append(self.extract_numer(0, i, Scalar, recursive))

        return tuple(results)

    def to_pair(self, axes=(0,1), recursive=True):
        """Return a Pair containing two selected components of a Vector.

        Overrides the default method to include an 'axes' argument, which can
        extract any two components of a Vector very efficiently.
        """

        i0 = axes[0]
        di = axes[1] - axes[0]
        if di < 0:
            i0 -= self.item[0]
        i1 = i0 + 2 * di

        index = (Ellipsis, slice(i0,i1,di)) + self.drank * (slice(None),)

        result = Qube.PAIR_CLASS(self.values[index], example=self)

        if recursive and self.derivs:
            for (key,deriv) in self.derivs.iteritems():
                result.insert_deriv(key, deriv.to_pair(axes,False))

        return result

    @staticmethod
    def from_scalars(*args, **keywords):
        """A Vector or subclass constructed by combining scalars.

        Inputs:
            args        any number of Scalars or arguments that can be casted
                        to Scalars. They need not have the same shape, but it
                        must be possible to cast them to the same shape. A value
                        of None is converted to a zero-valued Scalar that
                        matches the denominator shape of the other arguments.

            recursive   True to include all the derivatives. The returned object
                        will have derivatives representing the union of all the
                        derivatives found amongst the scalars. Default is True.

            classes     an arbitrary list of scalars defining the preferred
                        class of the returned object. The first suitable class
                        in the list will be used. Default is Vector.

        Note that the 'recursive' and 'classes' inputs are handled as keyword
        arguments in order to distinguish them from the scalar inputs.
        """

        # Search the keywords for "recursive" and "classes"
        recursive = True
        classes = []
        if 'recursive' in keywords:
            recursive = keywords['recursive']
            del keywords['recursive']

        if 'classes' in keywords:
            classes = keywords['classes']
            del keywords['classes']

        # No other keyword is allowed
        if keywords:
          raise TypeError(("broadcast() got an unexpected keyword argument " +
                           "'%s'") % keywords.keys()[0])

        args = list(args)

        # Get the properties of the first argument as a PolyMath subclass
        denom = ()
        for arg in args:
            if isinstance(arg, Qube):
                denom = arg.denom
                break

        drank = len(denom)

        # Convert to Scalars, identify units, select dtype
        units = None
        dtype = 'int'
        for (i,arg) in enumerate(args):

            # Convert None to a zero-valued scalar of the proper denom shape
            if arg is None:
                args[i] = Scalar(np.zeros(denom, dtype='int'), drank=drank)

            # Cast any Qube to a Scalar
            elif isinstance(arg, Qube):
                args[i] = Scalar.as_scalar(arg)

                # Make sure denominator shape matches
                if arg.denom != denom:
                    raise ValueError('incompatible denominator shapes: ' +
                                     str(denom) + ', ' + str(arg.denom))

                # Remember any units encountered
                if arg.units is not None:
                    units = arg.units

                # Remember any floats encountered
                if arg.is_float():
                    dtype = 'float'

            # Otherwise, convert to Scalar
            else:
                args[i] = Scalar(arg, drank=drank)

                # Remember any floats encountered
                if args[i].is_float():
                    dtype = 'float'

        # Make one more pass to confirm compatible units
        if units is not None:
            for arg in args:
                arg.confirm_units(units)

        # Broadcast all inputs into a common shape
        args = Qube.broadcast(*args, recursive=True)

        # Assemble the values into an array, merge the masks
        mask = False
        array = np.empty(args[0].shape + denom + (len(args),), dtype=dtype)

        for (i,arg) in enumerate(args):
            mask = mask | arg.mask
            array[...,i] = arg.values

        if len(denom):
            array = np.rollaxis(array, -1, len(array.shape) - drank - 1)

        # Construct the result
        result = Qube(array, mask, units, nrank=1, drank=drank)
        result = result.cast(list(classes) + [Vector])

        # Fill in derivatives if necessary
        if recursive:
            derivs = {}
            for (i,arg) in enumerate(args):
                for (key,deriv) in arg.derivs.iteritems():

                    # Create a buffer empty except for one component
                    deriv_args = len(args) * [None]
                    deriv_args[i] = deriv
                    full_deriv = Vector.from_scalars(*deriv_args,
                                                     recursive=False,
                                                     classes=classes)
                    if key in derivs:
                        derivs[key] += full_deriv
                    else:
                        derivs[key] = full_deriv

            result.insert_derivs(derivs)

        return result

    def as_index(self, remove_masked=False, masked=None):
        """An object suitable for indexing an N-dimensional array and a mask.

        The returned object is a tuple of NumPy arrays, each of the same shape.
        Each array contains indices along the corresponding axis of the array
        being indexed.

        Input:
            remove_masked True if the index should have masked values removed.
            masked        the index or list/tuple/array of indices to insert in
                          the place of a masked item. If None and the object
                          contains masked elements, the array will be flattened
                          and masked elements will be skipped over.

        Return: (valueidx, mask)
            valueidx    the indices corresponding to this Vector's value
            mask        a boolean array indicating which indices are masked
        """

        if (self.drank > 0):
            raise ValueError('an indexing object cannot have a denominator')

        obj = self.as_int()
        mask = None
        
        if not np.any(self.mask):
            values = obj.values

        if masked is None:
            if remove_masked:
                if np.all(obj.mask): # All masked
                    return None, None
                elif not np.any(obj.mask): # None masked
                    values = obj.values
                    mask = obj.mask
                else: # Some masked
                    obj = obj.flatten()
                    values = obj[~obj.mask].values
                    mask = obj.mask
            else:
                obj = obj.copy()
                values = obj.values
                mask = obj.mask
                if np.shape(values) == ():
                    if mask:
                        values = 0
                else:
                    if np.shape(mask) == ():
                        if mask:
                            values[:] = 0 # All masked
                    else: # Partially masked
                        values[mask] = 0 # Always a safe index
        else:
            obj = obj.copy()
            obj.values[obj.mask] = masked
            values = obj.values

        return tuple(np.rollaxis(values, -1, 0)), mask

    def as_column(self, recursive=True):
        """Convert the Vector to an Nx1 column matrix.

        Input:
            recursive   True to include the derivatives.
        """

        return self.reshape_numer(self.numer + (1,), Qube.MATRIX_CLASS,
                                  recursive)

    def as_row(self, recursive=True):
        """Convert the Vector to a 1xN row matrix.

        Input:
            recursive   True to include the derivatives.
        """

        return self.reshape_numer((1,) + self.numer, Qube.MATRIX_CLASS,
                                  recursive)

    def as_diagonal(self, recursive=True):
        """Convert the vector to a diagonal matrix.

        Input:
            recursive   True to include the derivatives.
        """

        return Qube.as_diagonal(self, 0, Qube.MATRIX_CLASS, recursive)

    def dot(self, arg, recursive=True):
        """Return the dot product of this vector and another as a Scalar.

        Input:
            recursive   True to include the derivatives.
        """

        arg = self.as_this_type(arg, recursive)
        return Qube.dot(self, arg, 0, 0, Scalar, recursive)

    def norm(self, recursive=True):
        """Return the length of this Vector as a Scalar.

        Input:
            recursive   True to include the derivatives.
        """

        return Qube.norm(self, 0, Scalar, recursive)

    def norm_sq(self, recursive=True):
        """Return the squared length of this Vector as a Scalar.

        Input:
            recursive   True to include the derivatives.
        """

        return Qube.norm_sq(self, 0, Scalar, recursive)

    def unit(self, recursive=True):
        """Return this vector converted to unit length.

        The returned object is an instance of the same subclass as this object.

        Input:
            recursive   True to include the derivatives.
        """

        if not recursive:
            self = self.without_derivs()

        return self / self.norm(recursive)

    def cross(self, arg, recursive=True):
        """Return the cross product of this vector with another.

        The returned object is an instance of the same subclass as this object.

        Input:
            recursive   True to include the derivatives.
        """

        arg = self.as_this_type(arg, recursive)

        # type(self) is for 3-vectors, Scalar is for 2-vectors...
        return Qube.cross(self, arg, 0, 0, (type(self), Scalar), recursive)

    def ucross(self, arg, recursive=True):
        """Return the unit vector in the direction of the cross product.

        Works only for vectors of length 3. The returned object is an instance
        of the same subclass as this object.

        Input:
            recursive   True to include the derivatives.
        """

        return self.cross(arg, recursive).unit(recursive)

    def outer(self, arg, recursive=True):
        """Perform an outer multiply of two Vectors, returning a Matrix.

        Input:
            recursive   True to include the derivatives.
        """

        arg = Vector.as_vector(arg, recursive)
        return Qube.outer(self, arg, Qube.MATRIX_CLASS, recursive)

    def perp(self, arg, recursive=True):
        """Return the component of this vector perpendicular to another.

        The returned object is an instance of the same subclass as this object.

        Input:
            recursive   True to include the derivatives.
        """

        # Convert arg to a unit vector
        if recursive:
            arg = self.as_this_type(arg, recursive).unit()
        else:
            arg = self.as_this_type(arg).without_derivs().unit()
            self = self.without_derivs()

        # Return the component of this vector perpendicular to the arg
        return self - arg * self.dot(arg, recursive)

    def proj(self, arg, recursive=True):
        """Return the component of this Vector projected into another Vector.

        The returned object is an instance of the same subclass as this object.

        Input:
            recursive   True to include the derivatives.
        """

        # Convert arg to a unit vector
        if recursive:
            arg = self.as_this_type(arg, recursive).unit()
        else:
            arg = self.as_this_type(arg).without_derivs().unit()

        # Return the component of this vector projected into the arg
        return arg * self.dot(arg, recursive)

    def sep(self, arg, recursive=True):
        """Return the separation angle between this vector and another.

        The returned object is an instance of the same subclass as this object.
        Works for vectors of length 2 or 3.

        Input:
            recursive   True to include the derivatives.
        """

        # Translated from the SPICE source code for VSEP().

        # Convert to unit vectors a and b. These define an isoceles triangle.
        a = self.unit(recursive)
        b = self.as_this_type(arg,recursive).unit(recursive)

        # This is the separation angle:
        #   angle = 2 * arcsin(|a-b| / 2)
        # However, this formula becomes less accurate for angles near pi. For
        # these angles, we reverse b and calculate the supplementary angle.

        sign = a.dot(b).sign().mask_where_eq(0, 1, remask=False)
        b = b * sign

        arg = 0.5 * (a - b).norm()
        angle = 2. * sign * arg.arcsin(check=False) + (sign < 0.) * np.pi

        return angle

    def cross_product_as_matrix(self, recursive=True):
        """The Matrix whose multiply equals a cross product with this object.

        This object must have length 3.
        """

        if self.numer != (3,):
            raise ValueError('shape must be (3,)')

        if self.denom != ():
            raise NotImplementedError('method not implemented for derivatives')

        # Roll the numerator axis to the end if necessary
        if self.drank == 0:
            old_values = self.values
        else:
            old_values = np.rollaxis(self.values, -self.drank-1,
                                     len(self.values.shape))

        # Fill in the matrix elements
        new_values = np.zeros(self.shape + self.denom + (3,3),
                              dtype = self.values.dtype)
        new_values[...,0,1] = -old_values[...,2]
        new_values[...,0,2] =  old_values[...,1]
        new_values[...,1,2] = -old_values[...,0]
        new_values[...,1,0] =  old_values[...,2]
        new_values[...,2,0] = -old_values[...,1]
        new_values[...,2,1] =  old_values[...,0]

        # Roll the denominator axes back to the end
        for i in range(self.drank):
            new_values = np.rollaxis(new_values, -3, len(new_values.shape))

        obj = Qube.MATRIX_CLASS(new_values, derivs={}, example=self)

        if recursive:
            for (key, deriv) in self.derivs.iteritems():
                obj.insert_deriv(key, deriv.cross_product_as_matrix(False))

        return obj

    def element_mul(self, arg, recursive=True):
        """Perform an element-by-element multiply of two vectors.

        The returned object is an instance of the same subclass as this object.

        Input:
            recursive   True to include the derivatives.
        """

        # Convert to this class if necessary
        original_arg = arg
        arg = self.as_this_type(arg, recursive)

        # If it had no units originally, it should not have units now
        if not isinstance(original_arg, Qube):
            arg = arg.without_units()

        # Validate
        if arg.numer != self.numer:
            raise ValueError(("incompatible numerator shapes: " +
                              "%s, %s") % (str(self.numer), str(arg.numer)))

        if self.drank > 0 and arg.drank > 0:
            raise ValueError(("dual operand denominators for element_mul(): " +
                              "%s, %s") % (str(self.denom), str(arg.denom)))

        # Reshape arrays as needed
        if arg.drank:
            self_values = self.values.reshape(self.values.shape +
                                              arg.drank * (1,))
        else:
            self_values = self.values

        if self.drank:
            arg_values = arg.values.reshape(arg.values.shape +
                                            self.drank * (1,))
        else:
            arg_values = arg.values

        # Construct the object
        obj = Qube.__new__(type(self))
        obj.__init__(self_values * arg_values,
                     self.mask | arg.mask,
                     Units.mul_units(self.units, arg.units),
                     derivs = {},
                     drank = self.drank + arg.drank,
                     example=self)

        # Insert derivatives if necessary
        if recursive:
            new_derivs = {}
            if self.derivs:
                arg_wod = arg.without_derivs()
                for (key, self_deriv) in self.derivs.iteritems():
                    new_derivs[key] = self_deriv.element_mul(arg_wod, False)

            if arg.derivs:
                self_wod = self.without_derivs()
                for (key, arg_deriv) in arg.derivs.iteritems():
                    term = self_wod.element_mul(arg_deriv, False)
                    if key in new_derivs:
                        new_derivs[key] += term
                    else:
                        new_derivs[key] = term

            obj.insert_derivs(new_derivs)

        return obj

    def element_div(self, arg, recursive=True):
        """Perform an element-by-element division of two vectors.

        The returned object is an instance of the same subclass as this object.

        Input:
            recursive   True to include the derivatives.
        """

        # Convert to this class if necessary
        if not isinstance(arg, Qube):
            arg = self.as_this_type(arg, recursive, drank=0)
            arg = arg.without_units()

        # Validate
        if arg.numer != self.numer:
            raise ValueError(("incompatible numerator shapes: " +
                              "%s, %s") % (str(self.numer), str(arg.numer)))

        if arg.drank > 0:
            raise ValueError(("right operand denominator for element_div(): " +
                              "%s") % str(arg.denom))

        # Mask out zeros in divisor
        zero_mask = (arg.values == 0.)

        if np.any(zero_mask):
            if np.shape(arg.values) == ():
                divisor = 1.
            else:
                divisor = arg.values.copy()
                divisor[zero_mask] = 1.
        else:
            divisor = arg.values

        # Update the divisor mask
        for r in range(self.rank):  # if any element is zero, mask the vector
            zero_mask = np.any(zero_mask, axis=-1)

        divisor_mask = arg.mask | zero_mask

        # Re-shape the divisor array if necessary to match the dividend shape
        if self.drank:
            divisor = divisor.reshape(divisor.shape + self.drank * (1,))

        # Construct the ratio object
        obj = Qube.__new__(type(self))
        obj.__init__(self.values / divisor,
                     self.mask | divisor_mask,
                     Units.div_units(self.units, arg.units))

        # Insert the derivatives if necessary
        if recursive:
            new_derivs = {}

            if self.derivs:
                arg_inv = Qube.__new__(type(self))
                arg_inv.__init__(1. / divisor, divisor_mask,
                                 Units.units_power(arg.units,-1))

                for (key, self_deriv) in self.derivs.iteritems():
                    new_derivs[key] = self_deriv.element_mul(arg_inv)

            if arg.derivs:
                arg_inv_sq = Qube.__new__(type(self))
                arg_inv_sq.__init__(divisor**(-2), divisor_mask,
                                    Units.units_power(arg.units,-1))
                factor = self.without_derivs().element_mul(arg_inv_sq)

                for (key, arg_deriv) in arg.derivs.iteritems():
                    term = arg_deriv.element_mul(factor)

                    if key in new_derivs:
                        new_derivs[key] -= term
                    else:
                        new_derivs[key] = -term

            obj.insert_derivs(new_derivs)

        return obj

    @classmethod
    def combos(cls, *args):
        """A vector with every combination of components of given scalars.

        Masks are also combined in the analogous manner.

        The returned object will have a shape defined by concatenating the
        shapes of all the arguments. Units and derivatives are ignored.
        """

        scalars = []
        newshape = []
        dtype = 'int'
        for arg in args:
            scalar = Scalar.as_scalar(arg)
            if scalar.drank:
                raise ValueError('scalar cannot have denominator in combos()')

            scalars.append(scalar)
            newshape += list(scalar.shape)
            if scalar.is_float(): dtype = 'float'

        newshape = tuple(newshape)
        newrank = len(newshape)
        data = np.empty(newshape + (len(args),), dtype=dtype)
        mask = np.zeros(newshape, dtype='bool')

        before = 0
        after = newrank
        for (i,scalar) in enumerate(scalars):
            shape = scalar.shape
            rank = len(shape)
            scalar = scalar.reshape(before * (1,) + shape + (after-rank) * (1,))
            data[...,i] = scalar.values
            mask |= scalar.mask

            before += rank
            after -= rank

        return cls(data, mask)

    ############################################################################
    # Overrides of superclass operators
    ############################################################################

    def __abs__(self, recursive=True):
        return self.norm(recursive)

    def identity(self):
        Qube.raise_unsupported_op('identity()', self)

    def reciprocal(self, recursive=True, nozeros=False):
        """Treat a Vector subclass with drank=1 as a Matrix inversion."""

        if self.drank != 1:
            Qube.raise_unsupported_op('reciprocal()', self)

        matrix = self.join_items([Qube.MATRIX_CLASS])
        inverse = matrix.reciprocal()

        return inverse.split_items(1, [type(self)])

# A set of useful class constants
Vector.ZERO3   = Vector((0.,0.,0.)).as_readonly()
Vector.XAXIS3  = Vector((1.,0.,0.)).as_readonly()
Vector.YAXIS3  = Vector((0.,1.,0.)).as_readonly()
Vector.ZAXIS3  = Vector((0.,0.,1.)).as_readonly()
Vector.MASKED3 = Vector((1,1,1), True).as_readonly()

Vector.ZERO2   = Vector((0.,0.)).as_readonly()
Vector.XAXIS2  = Vector((1.,0.)).as_readonly()
Vector.YAXIS2  = Vector((0.,1.)).as_readonly()
Vector.MASKED2 = Vector((1,1), True).as_readonly()

################################################################################
# Once defined, register with Qube class
################################################################################

Qube.VECTOR_CLASS = Vector

################################################################################
