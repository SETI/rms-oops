################################################################################
# polymath/vector.py: Vector subclass of PolyMath base class
################################################################################

from __future__ import division
import numpy as np

from polymath.qube   import Qube
from polymath.scalar import Scalar
from polymath.units  import Units

class Vector(Qube):
    """
    A PolyMath subclass containing 1-D vectors of arbitrary length."""

    NRANK = 1           # the number of numerator axes.
    NUMER = None        # shape of the numerator.

    FLOATS_OK = True    # True to allow floating-point numbers.
    INTS_OK = True      # True to allow integers.
    BOOLS_OK = False    # True to allow booleans.

    UNITS_OK = True     # True to allow units; False to disallow them.
    DERIVS_OK = True    # True to allow derivatives and to allow this class to
                        # have a denominator; False to disallow them.

    #===========================================================================
    def __init__(self, arg, mask=False, derivs={}, units=None,
                       nrank=None, drank=None, example=None, default=None):
        """Tweak the default constructor to convert a Python scalar to an array
        of shape (1,).
        """

        if isinstance(arg, (float,int)):
            arg = np.array([arg])

        super(Vector,self).__init__(arg, mask=mask, derivs=derivs, units=units,
                                    nrank=nrank, drank=drank, example=example,
                                    default=default)

    #===========================================================================
    @staticmethod
    def as_vector(arg, recursive=True):
        """The argument converted to Scalar if possible.

        If recursive is True, derivatives will also be converted. However, note
        that derivatives are not necessarily removed when recursive is False.
        """

        if isinstance(arg, Vector):
            if recursive:
                return arg

            return arg.wod

        if isinstance(arg, Qube):

            # Convert any 1-D object
            if arg._nrank_ == 1:
                return arg.flatten_numer(Vector, recursive=recursive)

            # Collapse a 1xN or Nx1 MatrixN down to a Vector
            if arg._nrank_ == 2 and (arg._numer_[0] == 1
                                     or arg._numer_[1] == 1):
                return arg.flatten_numer(Vector, recursive=recursive)

            # Convert Scalar to shape (1,)
            if arg._nrank_ == 0:
                if np.isscalar(arg._values_):
                    new_values = np.array([arg._values_])
                else:
                    new_values = arg._values_.reshape(arg._shape_ + (1,) +
                                                      arg.item)

                result = Vector(new_values, arg._mask_, nrank=1,
                                drank=arg._drank_, derivs={}, example=arg)

                if recursive and arg._derivs_:
                    for (key, value) in arg._derivs_.items():
                        result.insert_deriv(key, Vector.as_vector(value, False))
                return result

            # For any other Qube, move numerator items to the denominator
            if arg.rank > 1:
                return arg.split_items(1, Vector)

            arg = Vector(arg)
            if recursive:
                return arg

            return arg.wod

        return Vector(arg)

    #===========================================================================
    def to_scalar(self, indx, recursive=True):
        """One of the components of a Vector as a Scalar.

        Input:
            indx        index of the vector component.
            recursive   True to extract the derivatives as well.
        """

        return self.extract_numer(0, indx, Scalar, recursive=recursive)

    #===========================================================================
    def to_scalars(self, recursive=True):
        """All the components of a Vector as a tuple of Scalars.

        Input:
            recursive   True to include the derivatives.
        """

        results = []
        for i in range(self._numer_[0]):
            results.append(self.extract_numer(0, i, Scalar,
                                              recursive=recursive))

        return tuple(results)

    #===========================================================================
    def to_pair(self, axes=(0,1), recursive=True):
        """A Pair containing two selected components of a Vector.

        Overrides the default method to include an 'axes' argument, which can
        extract any two components of a Vector very efficiently.
        """

        i0 = axes[0]
        di = axes[1] - axes[0]
        if di < 0:
            i0 -= self.item[0]
        i1 = i0 + 2 * di
        idx = (Ellipsis, slice(i0,i1,di)) + self._drank_ * (slice(None),)

        result = Qube.PAIR_CLASS(self._values_[idx], self._mask_, derivs={},
                                                              example=self)

        if recursive and self._derivs_:
            for (key,deriv) in self._derivs_.items():
                result.insert_deriv(key, deriv.to_pair(axes,False))

        return result

    #===========================================================================
    @staticmethod
    def from_scalars(*args, **keywords):
        """A Vector constructed by combining scalars.

        Inputs:
            x, y, z     Three Scalars defining the vector's x, y and z
                        components. They need not have the same shape, but it
                        must be possible to cast them to the same shape. A value
                        of None is converted to a zero-valued Scalar that
                        matches the denominator shape of the other arguments.

            recursive   True to include all the derivatives. The returned object
                        will have derivatives representing the union of all the
                        derivatives found among x, y and z. Default is True.

            readonly    True to return a read-only object; False (the default)
                        to return something potentially writable.
        """

        return Qube.from_scalars(*args, classes=[Vector], **keywords)

    #===========================================================================
    def as_index(self, masked=None):
        """This object made suitable for indexing an N-dimensional NumPy array.

        The returned object is a tuple of NumPy arrays, each of the same shape.
        Each array contains indices along the corresponding axis of the array
        being indexed.

        Input:
            masked      the index or list/tuple/array of indices to insert in
                        the place of a masked item. If None and the object
                        contains masked elements, the array will be flattened
                        and masked elements will be skipped over.
        """

        (index, mask) = self.as_index_and_mask((masked is None), masked)
        return index

    #===========================================================================
    def as_index_and_mask(self, purge=False, masked=None):
        """This object made suitable for indexing and masking an N-dimensional
        array.

        Input:
            purge           True to eliminate masked elements from the index;
                            False to retain them but leave them masked.
            masked          the index value to insert in place of any masked.
                            item. This may be needed because each value in the
                            returned index array must be an integer and in
                            range. If None (the default), then masked values
                            in the index will retain their unmasked values when
                            the index is applied.
        """

        if self.is_float():
            raise IndexError('floating-point indexing is not permitted')

        if self._drank_:
            raise ValueError('an index cannot have a denominator')

        # If nothing is masked, this is easy
        if not np.any(self._mask_):
            return (tuple(np.rollaxis(self._values_.astype(np.intp), -1, 0)),
                    False)

        # If purging...
        if purge:
            # If all masked...
            if Qube.is_one_true(self._mask_):
                return ((), False)

            # If partially masked...
            new_values = self._values_[self.antimask]
            return (tuple(np.rollaxis(new_values.astype(np.intp), -1, 0)),
                    False)

        # Without a replacement...
        if masked is None:
            new_values = self._values_.astype(np.intp)

        # If all masked...
        elif Qube.is_one_true(self._mask_):
            new_values = np.empty(self._values_._shape_, dtype=np.intp)
            new_values[...] = masked

        # If partially masked...
        else:
            new_values = self._values_.copy().astype(np.intp)
            new_values[self._mask_] = masked

        return (tuple(np.rollaxis(new_values, -1, 0)), self._mask_)

    #===========================================================================
    def int(self, top=None, remask=False, clip=False, inclusive=True,
                  shift=None):
        """An integer (floor) version of this Vector or subclass.

        If this object already contains integers, it is returned as is.
        Otherwise, a copy is returned. Derivatives are always removed. Units
        are disallowed.

        Inputs:
            top         Optional tuple of nominal maximum integer values,
                        equivalent to the array shape.
            remask      If True, values less than zero or greater than the
                        specified top values (if provided) are masked.
            clip        If True, values less than zero or greater than the
                        specified top values are clipped. Use a tuple of
                        booleans to handle the axes differently.
            inclusive   True to leave the top limits unmasked; False to mask
                        them. Use a tuple of booleans to handle the axes
                        differently.
            shift       True to shift any occurrences of the top limit down by
                        one; False to leave them unchanged. Use a tuple of
                        booleans to handle the axes differently. shift=True
                        implies inclusive=True. Default None lets shift match
                        the input value of inclusive.
        """

        def _as_tuple(item):
            # Quick internal method to make sure clip, inclusive, and shift are
            # tuples or lists of the correct length.
            if isinstance(item, (list, tuple)):
                assert len(item) == len(top)
            else:
                item = len(top) * (item,)
            return item

        Units.require_unitless(self._units_)
        if self._denom_:
            raise ValueError('denominators are not supported in Vector.int')

        if top is not None:
            assert len(top) == self.item[-1]

            clip = _as_tuple(clip)
            inclusive = _as_tuple(inclusive)
            if shift is None:
                shift = inclusive
            else:
                shift = _as_tuple(shift)

            # Convert to int; be sure it's a copy before modifying
            if self.is_int():
                copied = self.copy()
                values = copied._values_
                mask = copied._mask_
            else:
                values = self.wod.as_int()._values_
                mask = self._mask_
                if not np.isscalar(mask):
                    mask = mask.copy()

            # For each axis...
            for k in range(self.item[-1]):
                if shift[k] and not clip[k]: # shift is unneeded if clip is True
                    top_value = (self._values_[...,k] == top[k])
                    if self._shape_:
                        values[top_value, k] -= 1
                    elif top_value:
                        values[k] -= 1

                if remask:
                    is_outside = Scalar.is_outside(self._values_[...,k],
                                                   0, top[k], inclusive[k])

                if clip[k]:
                    values[...,k] = np.clip(values[...,k], 0, top[k]-1)

                if remask:
                    mask |= is_outside

            result = Qube.__new__(type(self))
            result.__init__(values, mask, example=self)

        else:
            result = self.wod.as_int()
            if clip:
                result = result.mask_where_lt(0, replace=0, remask=remask)

            elif remask:
                result = result.mask_where_lt(0, remask=remask)

        return result

    #===========================================================================
    def as_column(self, recursive=True):
        """Convert the Vector to an Nx1 column matrix.

        Input:
            recursive   True to include the derivatives.
        """

        return self.reshape_numer(self._numer_ + (1,), Qube.MATRIX_CLASS,
                                  recursive=recursive)

    #===========================================================================
    def as_row(self, recursive=True):
        """Convert the Vector to a 1xN row matrix.

        Input:
            recursive   True to include the derivatives.
        """

        return self.reshape_numer((1,) + self._numer_, Qube.MATRIX_CLASS,
                                  recursive=recursive)

    #===========================================================================
    def as_diagonal(self, recursive=True):
        """Convert the vector to a diagonal matrix.

        Input:
            recursive   True to include the derivatives.
        """

        return Qube.as_diagonal(self, 0, Qube.MATRIX_CLASS, recursive=recursive)

    #===========================================================================
    def dot(self, arg, recursive=True):
        """The dot product of this vector and another as a Scalar.

        Input:
            recursive   True to include the derivatives.
        """

        arg = self.as_this_type(arg, recursive=recursive, coerce=False)
        return Qube.dot(self, arg, 0, 0, Scalar, recursive=recursive)

    #===========================================================================
    def norm(self, recursive=True):
        """The length of this Vector as a Scalar.

        Input:
            recursive   True to include the derivatives.
        """

        return Qube.norm(self, 0, Scalar, recursive=recursive)

    #===========================================================================
    def norm_sq(self, recursive=True):
        """The squared length of this Vector as a Scalar.

        Input:
            recursive   True to include the derivatives.
        """

        return Qube.norm_sq(self, 0, Scalar, recursive=recursive)

    #===========================================================================
    def unit(self, recursive=True):
        """This vector converted to unit length.

        The returned object is an instance of the same subclass as this object.

        Input:
            recursive   True to include the derivatives.
        """

        if recursive:
            return self / self.norm(recursive=True)
        else:
            return self.wod / self.norm(recursive=False)

    #===========================================================================
    def with_norm(self, norm=1., recursive=True):
        """This vector scaled to the specified length.

        The returned object is an instance of the same subclass as this object.

        Input:
            recursive   True to include the derivatives.
        """

        norm = Scalar.as_scalar(norm, recursive=recursive)

        if recursive:
            return self * (norm / self.norm(recursive=True))
        else:
            return self.wod * (norm / self.norm(recursive=False))

    #===========================================================================
    def cross(self, arg, recursive=True):
        """The cross product of this vector with another.

        The returned object is an instance of the same subclass as this object.

        Input:
            recursive   True to include the derivatives.
        """

        arg = self.as_this_type(arg, recursive=recursive, coerce=False)

        # type(self) is for 3-vectors, Scalar is for 2-vectors...
        return Qube.cross(self, arg, 0, 0, (type(self), Scalar),
                                recursive=recursive)

    #===========================================================================
    def ucross(self, arg, recursive=True):
        """The unit vector in the direction of the cross product.

        Works only for vectors of length 3. The returned object is an instance
        of the same subclass as this object.

        Input:
            recursive   True to include the derivatives.
        """

        return self.cross(arg, recursive=recursive).unit(recursive=recursive)

    #===========================================================================
    def outer(self, arg, recursive=True):
        """The outer multiply of two Vectors, returning a Matrix.

        Input:
            recursive   True to include the derivatives.
        """

        arg = Vector.as_vector(arg, recursive=recursive)
        return Qube.outer(self, arg, Qube.MATRIX_CLASS, recursive=recursive)

    #===========================================================================
    def perp(self, arg, recursive=True):
        """The component of this vector perpendicular to another.

        The returned object is an instance of the same subclass as this object.

        Input:
            recursive   True to include the derivatives.
        """

        # Convert arg to a unit vector
        arg = self.as_this_type(arg, recursive=recursive, coerce=False).unit()
        if not recursive:
            self = self.wod

        # Return the component of this vector perpendicular to the arg
        return self - arg * self.dot(arg, recursive=recursive)

    #===========================================================================
    def proj(self, arg, recursive=True):
        """The component of this Vector projected into another Vector.

        The returned object is an instance of the same subclass as this object.

        Input:
            recursive   True to include the derivatives.
        """

        # Convert arg to a unit vector
        arg = self.as_this_type(arg, recursive=recursive, coerce=False).unit()

        # Return the component of this vector projected into the arg
        return arg * self.dot(arg, recursive=recursive)

    #===========================================================================
    def sep(self, arg, recursive=True):
        """The separation angle between this vector and another.

        The returned object is an instance of the same subclass as this object.
        Works for vectors of length 2 or 3.

        Input:
            recursive   True to include the derivatives.
        """

        # Translated from the SPICE source code for VSEP().

        # Convert to unit vectors a and b. These define an isoceles triangle.
        a = self.unit(recursive)
        b = self.as_this_type(arg, recursive=recursive, coerce=False).unit()

        # This is the separation angle:
        #   angle = 2 * arcsin(|a-b| / 2)
        # However, this formula becomes less accurate for angles near pi. For
        # these angles, we reverse b and calculate the supplementary angle.
        sign = a.dot(b).sign().mask_where_eq(0, 1, remask=False)
        b = b * sign

        arg = 0.5 * (a - b).norm()
        angle = 2. * sign * arg.arcsin() + (sign < 0.) * np.pi

        return angle

    #===========================================================================
    def cross_product_as_matrix(self, recursive=True):
        """The Matrix whose multiply equals a cross product with this object.

        This object must have length 3.
        """

        if self._numer_ != (3,):
            raise ValueError('shape must be (3,)')

        if self._drank_:
            raise ValueError('denominators are not supported in ' +
                             type(self).__name__ + '.cross_product_as_matrix')

        # Roll the numerator axis to the end if necessary
        if self._drank_ == 0:
            old_values = self._values_
        else:
            old_values = np.rollaxis(self._values_, -self._drank_-1,
                                     len(self._values_._shape_))

        # Fill in the matrix elements
        new_values = np.zeros(self._shape_ + self._denom_ + (3,3),
                              dtype = self._values_.dtype)
        new_values[...,0,1] = -old_values[...,2]
        new_values[...,0,2] =  old_values[...,1]
        new_values[...,1,2] = -old_values[...,0]
        new_values[...,1,0] =  old_values[...,2]
        new_values[...,2,0] = -old_values[...,1]
        new_values[...,2,1] =  old_values[...,0]

        # Roll the denominator axes back to the end
        for i in range(self._drank_):
            new_values = np.rollaxis(new_values, -3, len(new_values._shape_))

        obj = Qube.MATRIX_CLASS(new_values, self._mask_, derivs={},
                                example=self)

        if recursive:
            for (key, deriv) in self._derivs_.items():
                obj.insert_deriv(key, deriv.cross_product_as_matrix(False))

        return obj

    #===========================================================================
    def element_mul(self, arg, recursive=True):
        """The element-by-element multiply of two vectors.

        The returned object is an instance of the same subclass as this object.

        Input:
            recursive   True to include the derivatives.
        """

        # Convert to this class if necessary
        original_arg = arg
        arg = self.as_this_type(arg, recursive=recursive, coerce=False)

        # If it had no units originally, it should not have units now
        if not isinstance(original_arg, Qube):
            arg = arg.without_units()

        # Validate
        if arg._numer_ != self._numer_:
            raise ValueError(("incompatible numerator shapes: " +
                              "%s, %s") % (str(self._numer_), str(arg._numer_)))

        if self._drank_ > 0 and arg._drank_ > 0:
            raise ValueError(("dual operand denominators for element_mul(): " +
                              "%s, %s") % (str(self._denom_), str(arg._denom_)))

        # Reshape value arrays as needed
        if arg._drank_:
            self_values = self._values_.reshape(self._values_.shape +
                                                arg._drank_ * (1,))
        else:
            self_values = self._values_

        if self._drank_:
            arg_values = arg._values_.reshape(arg._values_.shape +
                                              self._drank_ * (1,))
        else:
            arg_values = arg._values_

        # Construct the object
        obj = Qube.__new__(type(self))
        obj.__init__(self_values * arg_values,
                     Qube.or_(self._mask_, arg._mask_),
                     derivs = {},
                     units = Units.mul_units(self._units_, arg._units_),
                     drank = self._drank_ + arg._drank_,
                     example=self)

        # Insert derivatives if necessary
        if recursive:
            new_derivs = {}
            for (key, self_deriv) in self._derivs_.items():
                new_derivs[key] = self_deriv.element_mul(arg.wod, False)

            for (key, arg_deriv) in arg._derivs_.items():
                term = self.wod.element_mul(arg_deriv, False)
                if key in new_derivs:
                    new_derivs[key] += term
                else:
                    new_derivs[key] = term

            obj.insert_derivs(new_derivs)

        return obj

    #===========================================================================
    def element_div(self, arg, recursive=True):
        """The element-by-element division of two vectors.

        The returned object is an instance of the same subclass as this object.

        Input:
            recursive   True to include the derivatives.
        """

        # Convert to this class if necessary
        if not isinstance(arg, Qube):
            arg = self.as_this_type(arg, recursive=recursive, coerce=False)
            arg = arg.without_units()

        # Validate
        if arg._numer_ != self._numer_:
            raise ValueError(("incompatible numerator shapes: " +
                              "%s, %s") % (str(self._numer_), str(arg._numer_)))

        if arg._drank_ > 0:
            raise ValueError(("right operand denominator for element_div(): " +
                              "%s") % str(arg._denom_))

        # Mask out zeros in divisor
        zero_mask = (arg._values_ == 0.)
        if np.any(zero_mask):
            if np.isscalar(arg._values_):
                divisor = 1.
            else:
                divisor = arg._values_.copy()
                divisor[zero_mask] = 1.

            # Reduce the zero mask over the item axes
            zero_mask = np.any(zero_mask, axis=tuple(range(-self._rank_,0)))
            divisor_mask = Qube.or_(arg._mask_, zero_mask)

        else:
            divisor = arg._values_
            divisor_mask = arg._mask_

        # Re-shape the divisor array if necessary to match the dividend shape
        if self._drank_:
            divisor = divisor.reshape(divisor._shape_ + self._drank_ * (1,))

        # Construct the ratio object
        obj = Qube.__new__(type(self))
        obj.__init__(self._values_ / divisor,
                     Qube.or_(self._mask_, divisor_mask),
                     units = Units.div_units(self._units_, arg._units_))

        # Insert the derivatives if necessary
        if recursive:
            new_derivs = {}

            if self._derivs_:
                arg_inv = Qube.__new__(type(self))
                arg_inv.__init__(1. / divisor, divisor_mask,
                                 units = Units.units_power(arg._units_, -1))

                for (key, self_deriv) in self._derivs_.items():
                    new_derivs[key] = self_deriv.element_mul(arg_inv)

            if arg._derivs_:
                arg_inv_sq = Qube.__new__(type(self))
                arg_inv_sq.__init__(divisor**(-2), divisor_mask,
                                    units = Units.units_power(arg._units_, -1))
                factor = self.wod.element_mul(arg_inv_sq)

                for (key, arg_deriv) in arg._derivs_.items():
                    term = arg_deriv.element_mul(factor)

                    if key in new_derivs:
                        new_derivs[key] -= term
                    else:
                        new_derivs[key] = -term

            obj.insert_derivs(new_derivs)

        return obj

    #===========================================================================
    def vector_scale(self, factor, recursive=True):
        """Stretch this Vector along a direction defined by a given scaling
        vector and and by an amount equal to the magnitude of this vector.

        Components of the vector perpendicular to the scaling vector are
        unchanged.

        Input:
            factor      a Vector defining the direction and magnitude of the
                        scaling.
            recursive   True to include the derivatives.

        Return:         a copy of this Vector scaled according to the scaling
                        vector
        """

        projected = self.proj(factor, recursive=recursive)

        if recursive:
            return self + (projected.norm() - 1) * projected
        else:
            return self.wod + (projected.norm() - 1) * projected

    #===========================================================================
    def vector_unscale(self, factor, recursive=True):
        """Un-stretch this Vector along a direction defined by a given scaling
        vector and and by an amount equal to the magnitude of this vector.

        Components of the vector perpendicular to the scaling vector are
        unchanged.

        Input:
            factor      a Vector defining the direction and magnitude of the
                        scaling.
            recursive   True to include the derivatives.

        Return:         a copy of this Vector scaled according to the scaling
                        vector
        """

        return self.vector_scale(factor/factor.norm_sq(recursive=recursive),
                                 recursive=recursive)

    #===========================================================================
    @classmethod
    def combos(cls, *args):
        """A vector with every combination of components of given scalars.

        Masks are also combined in the analogous manner.

        The returned object will have a shape defined by concatenating the
        shapes of all the arguments. Units and derivatives are ignored.
        """

        scalars = []
        newshape = []
        dtype = np.int_
        for arg in args:
            scalar = Scalar.as_scalar(arg)
            if scalar._drank_:
                raise ValueError('denominators are not supported in combos()')

            scalars.append(scalar)
            newshape += list(scalar._shape_)
            if scalar.is_float():
                dtype = np.float_

        newshape = tuple(newshape)
        newrank = len(newshape)
        data = np.empty(newshape + (len(args),), dtype=dtype)
        mask = np.zeros(newshape, dtype='bool')

        before = 0
        after = newrank
        for (i,scalar) in enumerate(scalars):
            shape = scalar._shape_
            rank = len(shape)
            scalar = scalar.reshape(before * (1,) + shape + (after-rank) * (1,))
            data[...,i] = scalar._values_
            mask |= scalar._mask_

            before += rank
            after -= rank

        if not np.any(mask):
            mask = False

        return cls(data, mask)

    #===========================================================================
    def mask_where_component_le(self, axis, limit, replace=None, remask=True):
        """A copy of this object where values of a specified component <= a
        limit value are masked.

        Instead of or in addition to masking the items, the values can be
        replaced. If no items need to be masked, this object is returned
        unchanged.

        Inputs:
            axis            the index of the component to use for comparison.
            limit           the limiting value or a Scalar of limiting values.
            replace         a single replacement value or an array of
                            replacement values, inserted into the returned
                            object at every masked location. Use None (the
                            default) to leave values unchanged.
            remask          True to include the new mask into the object's mask;
                            False to replace the values but leave them unmasked.
        """

        scalar = self.to_scalar(axis)
        return self.mask_where(scalar <= limit, replace=replace, remask=remask)

    #===========================================================================
    def mask_where_component_ge(self, axis, limit, replace=None, remask=True):
        """A copy of this object where values of a specified component >= a
        limit value are masked.

        Instead of or in addition to masking the items, the values can be
        replaced. If no items need to be masked, this object is returned
        unchanged.

        Inputs:
            axis            the index of the component to use for comparison.
            limit           the limiting value or a Scalar of limiting values.
            replace         a single replacement value or an array of
                            replacement values, inserted into the returned
                            object at every masked location. Use None (the
                            default) to leave values unchanged.
            remask          True to include the new mask into the object's mask;
                            False to replace the values but leave them unmasked.
        """

        scalar = self.to_scalar(axis)
        return self.mask_where(scalar >= limit, replace=replace, remask=remask)

    #===========================================================================
    def mask_where_component_lt(self, axis, limit, replace=None, remask=True):
        """A copy of this object where values of a specified component < a
        limit value are masked.

        Instead of or in addition to masking the items, the values can be
        replaced. If no items need to be masked, this object is returned
        unchanged.

        Inputs:
            axis            the index of the component to use for comparison.
            limit           the limiting value or a Scalar of limiting values.
            replace         a single replacement value or an array of
                            replacement values, inserted into the returned
                            object at every masked location. Use None (the
                            default) to leave values unchanged.
            remask          True to include the new mask into the object's mask;
                            False to replace the values but leave them unmasked.
        """

        scalar = self.to_scalar(axis)
        return self.mask_where(scalar < limit, replace=replace, remask=remask)

    #===========================================================================
    def mask_where_component_gt(self, axis, limit, replace=None, remask=True):
        """A copy of this object where values of a specified component > a
        limit value are masked.

        Instead of or in addition to masking the items, the values can be
        replaced. If no items need to be masked, this object is returned
        unchanged.

        Inputs:
            axis            the index of the component to use for comparison.
            limit           the limiting value or a Scalar of limiting values.
            replace         a single replacement value or an array of
                            replacement values, inserted into the returned
                            object at every masked location. Use None (the
                            default) to leave values unchanged.
            remask          True to include the new mask into the object's mask;
                            False to replace the values but leave them unmasked.
        """

        scalar = self.to_scalar(axis)
        return self.mask_where(scalar > limit, replace=replace, remask=remask)

    #===========================================================================
    def clip_component(self, axis, lower, upper, remask=False):
        """A copy of this object where values of a specified component outside
        a given range are shifted to the closest in-range value.

        Optionally, the clipped items can also be masked.

        If no items need to be clipped, this object is returned unchanged.

        Inputs:
            axis            the index of the component to use for comparison.
            lower           the lower limit for clipping; None to ignore. This
                            can be a single scalar or a Scalar object of the
                            same shape as the object.
            upper           the upper limit for clipping; None to ignore. This
                            can be a single scalar or a Scalar object of the
                            same shape as the object.
            remask          True to include the new mask into the object's mask;
                            False to replace the values but leave them unmasked.
        """

        if self._drank_:
            raise ValueError('clip_component() requires a Vector without a ' +
                             'denominator')

        vector = self.copy()
        mask = vector._mask_
        compt = vector.to_scalar(axis)      # shares memory with vector

        if lower is not None:
            lower = Scalar.as_scalar(lower)
            clipping_mask = (compt._values_
                             < lower._values_) & lower.antimask
            if np.shape(lower._values_):
                compt._values_[clipping_mask] = lower._values_[clipping_mask]
            elif vector._shape_:
                compt._values_[clipping_mask] = lower._values_
            elif clipping_mask:
                vector._values_[axis] = lower._values_

            if remask:
                mask = Qube.or_(mask, clipping_mask)

        if upper is not None:
            upper = Scalar.as_scalar(upper)
            clipping_mask = (compt._values_
                             > upper._values_) & upper.antimask
            if np.shape(upper._values_):
                compt._values_[clipping_mask] = upper._values_[clipping_mask]
            elif vector._shape_:
                compt._values_[clipping_mask] = upper._values_
            elif clipping_mask:
                vector._values_[axis] = upper

            if remask:
                mask = Qube.or_(mask, clipping_mask)

        if remask and np.any(mask):
            vector._set_mask_(mask)

        return vector

    ############################################################################
    # Overrides of superclass operators
    ############################################################################

    def __abs__(self, recursive=True):
        return self.norm(recursive)

    def identity(self):
        Qube._raise_unsupported_op('identity()', self)

    def reciprocal(self, recursive=True, nozeros=False):
        """Treat a Vector subclass with drank=1 as a Matrix inversion."""

        if self._drank_ != 1:
            Qube._raise_unsupported_op('reciprocal()', self)

        matrix = self.join_items([Qube.MATRIX_CLASS])
        inverse = matrix.reciprocal()

        return inverse.split_items(1, [type(self)])

################################################################################
# A set of useful class constants
################################################################################

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
