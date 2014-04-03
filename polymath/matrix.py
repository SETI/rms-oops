################################################################################
# polymath/modules/matrix.py: Matrix subclass ofse PolyMath base class
#
# Mark Showalter, PDS Rings Node, SETI Institute, February 2014
################################################################################

from __future__ import division
import numpy as np

from qube    import Qube
from scalar  import Scalar
from boolean import Boolean
from vector  import Vector
from vector3 import Vector3
from units   import Units

################################################################################
# Matrix Subclass...
################################################################################

class Matrix(Qube):
    """A Qube of arbitrary 2-D matrices."""

    NRANK = 2           # the number of numerator axes.
    NUMER = None        # shape of the numerator.

    FLOATS_OK = True    # True to allow floating-point numbers.
    INTS_OK = False     # True to allow integers.
    BOOLS_OK = False    # True to allow booleans.

    UNITS_OK = True     # True to allow units; False to disallow them.
    MASKS_OK = True     # True to allow masks; False to disallow them.
    DERIVS_OK = True    # True to disallow derivatives; False to allow them.

    DEBUG = False       # Set to True for some debugging tasks
    DELTA = np.finfo(float).eps * 3     # Cutoff used in unary()

    @staticmethod
    def as_matrix(arg, recursive=True):

        if type(arg) == Matrix:
            if recursive: return arg
            return arg.without_derivs()

        # Convert a Vector with drank=1 to a Matrix
        if isinstance(arg, Vector) and arg.drank == 1:
            return arg.join_items([Matrix])

        arg = Matrix(arg)
        if recursive: return arg
        return arg.without_derivs()

    def row_vector(self, row, recursive=True, classes=(Vector3,Vector)):
        """Return the selected row of a Matrix as a Vector.

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

    def row_vectors(self, recursive=True, classes=(Vector3,Vector)):
        """Return a tuple of Vector objects, one for each row of this Matrix.

        If the Matrix is M x N, then this will return M Vectors of length N. By
        default, if N == 3, it will return Vector3 objects instead.

        Input:
            recursive   True to return corresponding vectors of derivatives.
            classes     a list of classes; instances of the first suitable
                        class are returned. Default is to return Vector3 objects
                        if the length is 3, otherwise a Vector.
        """

        list = []
        for row in range(self.numer[0]):
            list.append(self.extract_numer(0, row, classes, recursive))

        return tuple(list)

    def column_vector(self, column, recursive=True, classes=(Vector3,Vector)):
        """Return the selected column of a Matrix as a Vector.

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

    def column_vectors(self, recursive=True, classes=(Vector3,Vector)):
        """Return a tuple of Vector objects, one for each column of this Matrix.

        If the Matrix is M x N, then this will return N Vectors of length M. By
        default, if M == 3, it will return Vector3 objects instead.

        Input:
            recursive   True to return corresponding vectors of derivatives.
            classes     a list of classes; instances of the first suitable
                        class are returned. Default is to return Vector3 objects
                        if the length is 3, otherwise a Vector.
        """

        list = []
        for col in range(self.numer[1]):
            list.append(self.extract_numer(1, col, classes, recursive))

        return tuple(list)

    def is_diagonal(self, delta=0.):
        """Return Boolean equal to True where the matrix is diagonal.

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
            rms = (np.sqrt(np.sum(np.sum(self.values**2, axis=-1), axis=-1)) /
                                                                        size)

        # Flatten the value array
        values = self.values.reshape(self.shape + (size*size,))

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
            if self.mask:
                compare = True
        elif np.shape(self.mask) == ():
            if self.mask:
                compare.fill(True)
        else:
            compare[self.mask] = True

        return Boolean(compare)

    def transpose(self, recursive=True):
        """Return the matrix transposed.

        Input:
            recursive   True to include the transposed derivatives; False to
                        return an object without derivatives.
        """

        return self.transpose_numer(0, 1, recursive)

    @property
    def T(self):
        """Shorthand notation for the transpose of a rotation matrix."""

        return self.transpose_numer(0, 1, recursive=True)

    def inverse(self, recursive=True, delta=0.):
        """Return the inverse of a matrix, for 2x2, 3x3 and diagonal matrices.

        The returned object will have the same subclass as this object.

        Input:
            recursive   True to include the derivatives of the inverse.
            delta       Relative upper limit on the off-diagonal elements of a
                        matrix such that the matrix is still considered
                        diagonal.
        """

        if self.numer[0] != self.numer[1]:
            raise ValueError("only square matrices can be inverted: shape is " +
                             str(self.numer))

        if self.drank:
            raise ValueError("a matrix with denominators cannot be inverted")

        # 3 x 3 case
        if self.numer[0] == 3:
            (new_values, new_mask) = Matrix.inverse_3x3(self.values)

        # 2 x 2 case
        elif self.numer[0] == 2:
            (new_values, new_mask) = Matrix.inverse_2x2(self.values)

        # All-diagonal case
        elif self.is_diagonal(delta=delta).all():
            (new_values, new_mask) = Matrix.inverse_diag(self.values)

        # A single item
        elif self.shape == ():
            return Matrix(np.array(np.matrix(self.values).I))
            
        # Remainder are TBD
        else:
            raise NotImplementedError("inversion of non-diagonal matrices " +
                                      "larger than 3x3 is not implemented")

        # Construct inverse matrix
        obj = Matrix(new_values, new_mask,
                     units = Units.units_power(self.units,-1),
                     derivs = {},
                     example = self)
        if self.readonly: obj = obj.as_readonly()

        # Fill in derivatives
        if recursive and self.derivs:
            new_derivs = {}

            # -M^-1 * dM/dt * M^-1
            for (key, deriv) in self.derivs.iteritems():
                new_derivs[key] = -obj * deriv * obj

            obj.insert_derivs(new_derivs, nocopy='vm')

        return obj

    def unitary(self):
        """Return a the nearest unitary matrix as a Matrix3."""

        # Algorithm from
        #    wikipedia.org/wiki/Orthogonal_matrix#Nearest_orthogonal_matrix

        MAX_ITERS = 10      # Adequate iterations unless convergence is failing

        m0 = self.without_derivs()
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
                sorted = np.sort(rms.values.ravel())
                print i, sorted[-4:]

            if rms.max() <= Matrix.DELTA: break

        new_mask = (rms.values > Matrix.DELTA)
        obj = Qube.MATRIX3_CLASS(next_m.values, self.mask | new_mask)

        if obj.readonly:
            obj = obj.as_readonly()

        return obj

    ############################################################################
    # Overrides of superclass operators
    ############################################################################

    def __abs__(self):
        Qube.raise_unsupported_op('abs()', self)

    def __floordiv__(self, arg):
        Qube.raise_unsupported_op('//', self, arg)

    def __rfloordiv__(self, arg):
        Qube.raise_unsupported_op('//', arg, self)

    def __ifloordiv__(self, arg):
        Qube.raise_unsupported_op('//=', self, arg)

    def __mod__(self, arg):
        Qube.raise_unsupported_op('%', self, arg)

    def __rmod__(self, arg):
        Qube.raise_unsupported_op('%', arg, self)

    def __imod__(self, arg):
        Qube.raise_unsupported_op('%=', self, arg)

    def identity(self):
        """Return an identity matrix of the same size and subclass as this."""

        size = self.numer[0]

        if self.numer[1] != size:
            raise ValueError('Matrix is not square')

        values = np.zeros((size,size))
        for i in range(size):
            values[i,i] = 1.

        obj = Qube.__new__(type(self))
        obj.__init__(values)

        return obj.as_readonly(nocopy='vm')

    ############################################################################
    # Overrides of arithmetic operators
    ############################################################################

    def reciprocal(self, recursive=True, nozeros=False):
        """Return an object equivalent to the reciprocal of this object.

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
# Matrix inverse functions
################################################################################

# For 3x3 matrix multiply
# From http://www.dr-lex.be/random/matrix_inv.html
#
# |a11 a12 a13|-1          |  a33a22-a32a23  -(a33a12-a32a13)   a23a12-a22a13 |
# |a21 a22 a23|  = 1/DET * |-(a33a21-a31a23)   a33a11-a31a13  -(a23a11-a21a13)|
# |a31 a32 a33|            |  a32a21-a31a22  -(a32a11-a31a12)   a22a11-a21a12 |
# 
# with DET  =  a11(a33a22-a32a23)-a21(a33a12-a32a13)+a31(a23a12-a22a13)
#
# Decrement indices by one
#
# |a00 a01 a02|-1            |a11a22-a12a21  a21a02-a22a01  a12a01-a11a02|
# |a10 a11 a12|   =  1/DET * |a12a20-a10a22  a22a00-a20a02  a10a02-a12a00|
# |a20 a21 a22|              |a10a21-a11a20  a20a01-a21a00  a11a00-a10a01|
# 
# with DET  =  a00(a11a22-a21a12)-a12(a22a01-a21a02)+a20(a12a01-a11a02)

    I1A = np.array([[1,2,1],[1,0,1],[1,2,1]])
    J1A = np.array([[1,1,2],[2,0,0],[0,0,1]])
    I1B = np.array([[2,0,0],[2,2,0],[2,0,0]])
    J1B = np.array([[2,2,1],[0,2,2],[1,1,0]])
    I2A = np.array([[1,2,1],[1,2,1],[1,2,1]])
    J2A = np.array([[2,2,1],[0,0,2],[1,1,0]])
    I2B = np.array([[2,0,0],[2,0,0],[2,0,0]])
    J2B = np.array([[1,1,2],[2,2,0],[0,0,1]])

    @staticmethod
    def inverse_3x3(mats):
        """Invert an arbitrary array of 3x3 matrices."""

        inverses = (mats[..., Matrix.I1A, Matrix.J1A] *
                    mats[..., Matrix.I1B, Matrix.J1B] -
                    mats[..., Matrix.I2A, Matrix.J2A] *
                    mats[..., Matrix.I2B, Matrix.J2B])

        det = (mats[...,0,0] * inverses[...,0,0] +
               mats[...,0,1] * inverses[...,1,0] +
               mats[...,0,2] * inverses[...,2,0])

        mask = (det == 0.)
        if np.any(mask):
            if np.shape(mask) == ():
                det = np.array(1.)
                mask = True
            else:
                det[mask] = 1.

        return (inverses/det[...,np.newaxis,np.newaxis], mask)

    @staticmethod
    def inverse_2x2(mats):
        """Invert of an arbitrary array of 2x2 matrices."""

        inverses = np.empty(mats.shape)
        inverses[...,0,0] =  mats[...,1,1]
        inverses[...,0,1] = -mats[...,0,1]
        inverses[...,1,0] = -mats[...,1,0]
        inverses[...,1,1] =  mats[...,0,0]

        det = mats[...,0,0] * mats[...,1,1] - mats[...,0,1] * mats[...,1,0]

        mask = (det == 0.)
        if np.any(mask):
            if np.shape(mask) == ():
                det = np.array(1.)
                mask = True
            else:
                det[mask] = 1.

        return (inverses/det[...,np.newaxis,np.newaxis], mask)

    @staticmethod
    def inverse_diag(mats):
        """Invert of an arbitrary array of equal-sized diagonal matrices."""

        diags = np.copy(mats[...,0])
        for i in range(1, mats.shape[-1]):
            diags[...,i] = mats[...,i,i]

        mask = (diags == 0.)
        if np.any(mask):
            if np.shape(mask) == ():
                diags = 1.
                mask = True
            else:
                diags[mask] = 1.

        reciprocals = 1. / diags

        inverses = np.zeros(mats.shape)
        for i in range(mats.shape[-1]):
            inverses[...,i,i] = reciprocals[...,i]

        mask = np.any(mask, axis=-1)

        return (inverses, mask)

# Useful class constants

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
