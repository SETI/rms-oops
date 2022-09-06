################################################################################
# polymath/matrix.py: Matrix subclass ofse PolyMath base class
################################################################################

from __future__ import division, print_function
import numpy as np

from .qube    import Qube
from .scalar  import Scalar
from .boolean import Boolean
from .vector  import Vector
from .vector3 import Vector3
from .units   import Units

class Matrix(Qube):
    """A Qube of arbitrary 2-D matrices."""

    NRANK = 2           # the number of numerator axes.
    NUMER = None        # shape of the numerator.

    FLOATS_OK = True    # True to allow floating-point numbers.
    INTS_OK = False     # True to allow integers.
    BOOLS_OK = False    # True to allow booleans.

    UNITS_OK = True     # True to allow units; False to disallow them.
    DERIVS_OK = True    # True to disallow derivatives; False to allow them.

    DEBUG = False       # Set to True for some debugging tasks
    DELTA = np.finfo(float).eps * 3     # Cutoff used in unary()

    #===========================================================================
    @staticmethod
    def as_matrix(arg, recursive=True):
        """The argument converted to Matrix if possible."""

        if type(arg) == Matrix:
            if recursive:
                return arg
            return arg.wod

        if isinstance(arg, Qube):

            # Convert a Vector with drank=1 to a Matrix
            if isinstance(arg, Vector) and arg.drank == 1:
                return arg.join_items([Matrix])

            arg = Matrix(arg._values_, arg._mask_, example=arg)
            if recursive:
                return arg
            return arg.wod

        return Matrix(arg)

    #===========================================================================
    def row_vector(self, row, recursive=True, classes=(Vector3,Vector)):
        """The selected row of a Matrix as a Vector.

        If the Matrix is M x N, then this will return a Vector of length N. By
        default, if N == 3, it will return a Vector3 object instead.

        Input:
            row         index of the row to return.
            recursive   True to return corresponding vectors of derivatives.
            classes     a list of classes; an instance of the first suitable
                        class is returned. Default is to return a Vector3 if
                        the length is 3, otherwise a Vector.
        """

        return self.extract_numer(0, row, classes, recursive)

    #===========================================================================
    def row_vectors(self, recursive=True, classes=(Vector3,Vector)):
        """A tuple of Vector objects, one for each row of this Matrix.

        If the Matrix is M x N, then this will return M Vectors of length N. By
        default, if N == 3, it will return Vector3 objects instead.

        Input:
            recursive   True to return corresponding vectors of derivatives.
            classes     a list of classes; instances of the first suitable
                        class are returned. Default is to return Vector3 objects
                        if the length is 3, otherwise a Vector.
        """

        vectors = []
        for row in range(self.numer[0]):
            vectors.append(self.extract_numer(0, row, classes, recursive))

        return tuple(vectors)

    #===========================================================================
    def column_vector(self, column, recursive=True, classes=(Vector3,Vector)):
        """The selected column of a Matrix as a Vector.

        If the Matrix is M x N, then this will return a Vector of length M. By
        default, if M == 3, it will return a Vector3 object instead.

        Input:
            column      index of the column to return.
            recursive   True to return corresponding vectors of derivatives.
            classes     a list of classes; an instance of the first suitable
                        class is returned. Default is to return a Vector3 if
                        the length is 3, otherwise a Vector.
        """

        return self.extract_numer(1, column, classes, recursive)

    #===========================================================================
    def column_vectors(self, recursive=True, classes=(Vector3,Vector)):
        """A tuple of Vector objects, one for each column of this Matrix.

        If the Matrix is M x N, then this will return N Vectors of length M. By
        default, if M == 3, it will return Vector3 objects instead.

        Input:
            recursive   True to return corresponding vectors of derivatives.
            classes     a list of classes; instances of the first suitable
                        class are returned. Default is to return Vector3 objects
                        if the length is 3, otherwise a Vector.
        """

        vectors = []
        for col in range(self.numer[1]):
            vectors.append(self.extract_numer(1, col, classes, recursive))

        return tuple(vectors)

    #===========================================================================
    def to_vector(self, axis, indx, classes=[], recursive=True):
        """One of the components of a Matrix as a Vector.

        Input:
            axis        axis index from which to extract vector.
            indx        index of the vector along this axis.
            classes     a list of the Vector subclasses to return. The first
                        valid one will be used. Default is Vector.
            recursive   True to extract the derivatives as well.
        """

        return self.extract_numer(axis, indx, list(classes) + [Vector],
                                  recursive)

    #===========================================================================
    def to_scalar(self, indx0, indx1, recursive=True):
        """One of the elements of a Matrix as a Scalar.

        Input:
            indx0       index along the first matrix axis.
            indx1       index along the second matrix axis.
            recursive   True to extract the derivatives as well.
        """

        vector = self.extract_numer(0, axis0, Vector, recursive)
        return   self.extract_numer(0, axis1, Scalar, recursive)

    #===========================================================================
    @staticmethod
    def from_scalars(*args, **keywords):
        """A Matrix or subclass constructed by combining scalars.

        Inputs:
            args        any number of Scalars or arguments that can be casted
                        to Scalars. They need not have the same shape, but it
                        must be possible to cast them to the same shape. A value
                        of None is converted to a zero-valued Scalar that
                        matches the denominator shape of the other arguments.

            recursive   True to include all the derivatives. The returned object
                        will have derivatives representing the union of all the
                        derivatives found amongst the scalars. Default is True.

            shape       The Matrix's item shape. If not specified but the number
                        of Scalars is a perfect square, a square matrix is
                        returned.

            classes     an arbitrary list defining the preferred class of the
                        returned object. The first suitable class in the list
                        will be used. Default is Matrix.

        Note that the 'recursive' and 'classes' inputs are handled as keyword
        arguments in order to distinguish them from the scalar inputs.
        """

        # Search for keyword "shape" and "classes"
        # Pass "recursive" to the next function
        item = None
        if 'shape' in keywords:
            item = keywords['shape']
            del keywords['shape']

        classes = []
        if 'classes' in keywords:
            classes = keywords['classes']
            del keywords['classes']

        # Create the Vector object
        vector = Vector.from_scalars(*args, **keywords)

        # Int matrices are disallowed
        if vector.is_int():
            raise TypeError('Matrix objects must be of type float')

        # Determine the shape
        if item is not None:
            if len(item) != 2:
                raise ValueError('invalid Matrix shape %s' % str(item))

            size = item[0] * item[1]
            if len(args) != item:
                raise ValueError('incorrect number of Scalars to create ' +
                                 'Matrix of shape %s' % str(item))
            item = tuple(item)

        else:
            dim = int(np.sqrt(len(args)))
            size = dim*dim
            if size != len(args):
                raise ValueError('incorrect number of Scalars to construct ' +
                                 'a square Matrix')
            item = (dim, dim)

        result = vector.reshape_numer(item, list(classes) + [Matrix],
                                            recursive=True)

    #===========================================================================
    def is_diagonal(self, delta=0.):
        """A Boolean equal to True where the matrix is diagonal.

        Masked matrices return True.

        Input:
            delta           the fractional limit on what can be treated as a
                            equivalent to zero in the off-diagonal terms. It is
                            scaled by the RMS value of all the elements in the
                            matrix.

        """

        size = self.item[0]
        if size != self.item[1]:
            raise ValueError('a diagonal matrix must be square')

        if self.drank:
            raise ValueError('diagonal matrix test is not supported for a '
                             'Matrix with a denominator')

        # If necessary, calculate the matrix RMS
        if delta != 0.:
            # rms, scaled to be unity for an identity matrix
            rms = (np.sqrt(np.sum(np.sum(self._values_**2, axis=-1), axis=-1)) /
                                                                        size)

        # Flatten the value array
        values = self._values_.reshape(self.shape + (size*size,))

        # Slice away the last element
        sliced = values[...,:-1]

        # Reshape so that only elemenents in the first column can be nonzero
        reshaped = sliced.reshape(self.shape + (size-1, size+1))

        # Slice away the first column
        sliced = reshaped[...,1:]

        # Convert back to 1-D items
        reshaped = sliced.reshape(self.shape + ((size-1) * size,))

        # Compare
        if delta == 0:
            compare = (reshaped == 0.)
        else:
            compare = (np.abs(reshaped) <= (delta * rms)[...,np.newaxis])

        compare = np.all(compare, axis=-1)

        # Apply mask
        if np.shape(compare) == ():
            if self._mask_:
                compare = True
        elif np.shape(self._mask_) == ():
            if self._mask_:
                compare.fill(True)
        else:
            compare[self._mask_] = True

        return Boolean(compare)

    #===========================================================================
    def transpose(self, recursive=True):
        """Transpose of this matrix.

        Input:
            recursive   True to include the transposed derivatives; False to
                        return an object without derivatives.
        """

        return self.transpose_numer(0, 1, recursive)

    #===========================================================================
    @property
    def T(self):
        """Shorthand notation for the transpose of a rotation matrix."""

        return self.transpose_numer(0, 1, recursive=True)

    #===========================================================================
    def inverse(self, recursive=True):
        """Inverse of this matrix.

        The returned object will have the same subclass as this object.

        Input:
            recursive   True to include the derivatives of the inverse.
        """

        # Validate array
        if self.numer[0] != self.numer[1]:
            raise ValueError("only square matrices can be inverted: shape is " +
                             str(self.numer))

        if self.drank:
            raise ValueError("a matrix with denominators cannot be inverted")

        # Check determinant
        det = np.linalg.det(self._values_)

        # Mask out univertible matrices and replace with diagonal matrix values
        mask = (det == 0.)
        if np.any(mask):
            self._values_[mask] = np.diag(np.ones(self.numer[0]))
            new_mask = self._mask_ | mask
        else:
            new_mask = self._mask_

        # Invert the arrray
        new_values = np.linalg.inv(self._values_)

        # Construct the result
        obj = Matrix(new_values, new_mask,
                     units = Units.units_power(self.units,-1))

        # Fill in derivatives
        if recursive and self.derivs:
            new_derivs = {}

            # -M^-1 * dM/dt * M^-1
            for (key, deriv) in self.derivs.items():
                new_derivs[key] = -obj * deriv * obj

            obj.insert_derivs(new_derivs)

        return obj

    #===========================================================================
    def unitary(self):
        """The nearest unitary matrix as a Matrix3."""

        # Algorithm from
        #    wikipedia.org/wiki/Orthogonal_matrix#Nearest_orthogonal_matrix

        MAX_ITERS = 10      # Adequate iterations unless convergence is failing

        m0 = self.wod
        if m0.drank:
            raise ValueError('a denominator is not supported for unitary()')

        if m0.numer != (3,3):
            raise ValueError('matrix shape must be 3x3 for unitary()')

        # Iterate...
        next_m = m0
        for i in range(MAX_ITERS):
            m = next_m
            next_m = 2. * m0 * (m.inverse() * m0 + m0.T * m).inverse()
            rms = Qube.rms(next_m * next_m.T - Matrix.IDENTITY3)

            if Matrix.DEBUG:
                sorted = np.sort(rms._values_.ravel())
                print(i, sorted[-4:])

            if rms.max() <= Matrix.DELTA:
                break

        new_mask = (rms._values_ > Matrix.DELTA)
        return Qube.MATRIX3_CLASS(next_m._values_, self._mask_ | new_mask)

# Algorithm has been validated but code has not been tested
#     def solve(self, values, recursive=True):
#         """Solve for the Vector X that satisfies A X = B, for this square matrix
#         A and a Vector B of results."""
#
#         b = Vector.as_vector(values, recursive=True)
#
#         size = self.item[0]
#         if size != self.item[1]:
#             raise ValueError('solver requires a square Matrix')
#
#         if self.drank:
#             raise ValueError('solver does not suppart a Matrix with a ' +
#                              'denominator')
#
#         if size != b.item[0]:
#             raise ValueError('Matrix and Vector have incompatible sizes')
#
#         # Easy cases: X = A-1 B
#         if size <= 3:
#             if recursive:
#                 return self.inverse(True) * b
#             else:
#                 return self.inverse(False) * b.wod
#
#         new_shape = Qube.broadcasted_shape(self.shape, b.shape)
#
#         # Algorithm is simpler with matrix indices rolled to front
#         # Also, Vector b's elements are placed after the elements of Matrix a
#
#         ab_vals = np.empty((size,size+1) + new_shape)
#         rolled = np.rollaxis(self._values_, -1, 0)
#         rolled = np.rollaxis(rolled, -1, 0)
#
#         ab_vals[:,:-1] = rolled
#         ab_vals[:,-1] = b._values_
#
#         for k in range(size-1):
#             # Zero out the leading coefficients from each row at each iteration
#             ab_saved = ab_vals[k+1:,k:k+1]
#             ab_vals[k+1:,k:] *= ab_vals[k,k:k+1]
#             ab_vals[k+1:,k:] -= ab_vals[k,k:] * ab_saved
#
#         # Now work backward solving for values, replacing Vector b
#         for k in range(size,0):
#             ab_vals[ k,-1] /= ab_vals[k,k]
#             ab_vals[:k,-1] -= ab_vals[k,-1] * ab_vals[:k,k]
#
#         ab_vals[0,-1] /= ab_vals[0,0]
#
#         x = np.rollaxis(ab_vals[:,-1], 0, len(shape))
#
#         x = Vector(x, self._mask_ | b._mask_, derivs={},
#                       units=Units.units_div(self.units, b.units))
#
#         # Deal with derivatives if necessary
#         # A x = B
#         # A dx/dt + dA/dt x = dB/dt
#         # A dx/dt = dB/dt - dA/dt x
#
#         if recursive and (self.derivs or b.derivs):
#             derivs = {}
#             for key in self.derivs:
#                 if key in b.derivs:
#                     values = b.derivs[key] - self.derivs[key] * x
#                 else:
#                     values = -self.derivs[k] * x
#
#             derivs[key] = self.solve(values, recursive=False)
#
#             for key in b.derivs:
#                 if key not in self.derivs:
#                     derivs[key] = self.solve(b.derivs[k], recursive=False)
#
#             self.insert_derivs(derivs)
#
#         return x

    ############################################################################
    # Overrides of superclass operators
    ############################################################################

    def __abs__(self):
        Qube._raise_unsupported_op('abs()', self)

    def __floordiv__(self, arg):
        Qube._raise_unsupported_op('//', self, arg)

    def __rfloordiv__(self, arg):
        Qube._raise_unsupported_op('//', arg, self)

    def __ifloordiv__(self, arg):
        Qube._raise_unsupported_op('//=', self, arg)

    def __mod__(self, arg):
        Qube._raise_unsupported_op('%', self, arg)

    def __rmod__(self, arg):
        Qube._raise_unsupported_op('%', arg, self)

    def __imod__(self, arg):
        Qube._raise_unsupported_op('%=', self, arg)

    def identity(self):
        """An identity matrix of the same size and subclass as this."""

        size = self.numer[0]

        if self.numer[1] != size:
            raise ValueError('Matrix is not square')

        values = np.zeros((size,size))
        for i in range(size):
            values[i,i] = 1.

        obj = Qube.__new__(type(self))
        obj.__init__(values)

        return obj.as_readonly()

    ############################################################################
    # Overrides of arithmetic operators
    ############################################################################

    def reciprocal(self, recursive=True, nozeros=False):
        """An object equivalent to the reciprocal of this object.

        Input:
            recursive   True to return the derivatives of the reciprocal too;
                        otherwise, derivatives are removed.
            nozeros     False (the default) to mask out any zero-valued items in
                        this object prior to the divide. Set to True only if you
                        know in advance that this object has no zero-valued
                        items.
        """

        return self.inverse(recursive=recursive)

################################################################################
# Useful class constants
################################################################################

Matrix.IDENTITY2 = Matrix([[1,0,],[0,1,]]).as_readonly()
Matrix.IDENTITY3 = Matrix([[1,0,0],[0,1,0],[0,0,1]]).as_readonly()

Matrix.MASKED2 = Matrix([[1,1],[1,1]], True).as_readonly()
Matrix.MASKED3 = Matrix([[1,1,1],[1,1,1],[1,1,1]], True).as_readonly()

Matrix.ZERO33 = Matrix([[0,0,0],[0,0,0],[0,0,0]]).as_readonly()
Matrix.UNIT33 = Matrix([[1,0,0],[0,1,0],[0,0,1]]).as_readonly()

Matrix.ZERO3_ROW = Matrix([[0,0,0]]).as_readonly()
Matrix.XAXIS_ROW = Matrix([[1,0,0]]).as_readonly()
Matrix.YAXIS_ROW = Matrix([[0,1,0]]).as_readonly()
Matrix.ZAXIS_ROW = Matrix([[0,0,1]]).as_readonly()

Matrix.ZERO3_COL = Matrix([[0],[0],[0]]).as_readonly()
Matrix.XAXIS_COL = Matrix([[1],[0],[0]]).as_readonly()
Matrix.YAXIS_COL = Matrix([[0],[1],[0]]).as_readonly()
Matrix.ZAXIS_COL = Matrix([[0],[0],[1]]).as_readonly()

################################################################################
# Once defined, register with base class
################################################################################

Qube.MATRIX_CLASS = Matrix

################################################################################
