################################################################################
# polymath/modules/matrix3.py: Matrix3 subclass of PolyMath Matrix class
#
# Mark Showalter, PDS Rings Node, SETI Institute, February 2014
################################################################################

from __future__ import division
import numpy as np

from qube    import Qube
from scalar  import Scalar
from vector  import Vector
from vector3 import Vector3
from matrix  import Matrix
from units   import Units

class Matrix3(Matrix):
    """A Qube of 3x3 rotation matrices."""

    NRANK = 2           # the number of numerator axes.
    NUMER = (3,3)       # shape of the numerator.

    FLOATS_OK = True    # True to allow floating-point numbers.
    INTS_OK = False     # True to allow integers.
    BOOLS_OK = False    # True to allow booleans.

    UNITS_OK = False    # True to allow units; False to disallow them.
    MASKS_OK = True     # True to allow masks; False to disallow them.
    DERIVS_OK = True    # True to disallow derivatives; False to allow them.

    @staticmethod
    def as_matrix3(arg, recursive=True):
        """Convert to Matrix3. The result is not checked to be unitary.

        Quaternions are converted to matrices.

        Inputs:
            arg         the object to convert.
            recursive   True to include derivatives in the returned result.
        """

        if type(arg) == Matrix3:
            if recursive: return arg
            return arg.without_derivs()

        if isinstance(arg, Qube):
            if isinstance(arg, Qube.QUATERNION_CLASS):
                return arg.to_matrix3(recursive)

            arg = Matrix3(arg, example=arg)
            if recursive: return arg
            return arg.without_derivs()

        return Matrix3(arg)

    @staticmethod
    def twovec(vector1, axis1, vector2, axis2):
        """Return a rotation matrix defined by two vectors.

        The returned matrix rotates to a coordinate frame having vector1
        pointing along a specified axis (axis1=0 for X, 1 for Y, 2 for Z) and
        vector2 pointing into the half-plane defined by (axis1,axis2).

        This function does not support derivatives.
        """

        # Based on the SPICE source code for TWOVEC()

        unit1 = Vector3.as_vector3(vector1).without_derivs().unit()
        vector2 = Vector3.as_vector3(vector2).without_derivs()
        (unit1, vector2) = Qube.broadcast(unit1, vector2)

        new_values = np.empty(unit1.values.shape + (3,))
        new_values[..., axis1, :] = unit1.values

        axis3 = 3 - axis1 - axis2
        if (3 + axis2 - axis1) % 3 == 1:      # if (0,1), (1,2) or (2,0)
            unit3 = unit1.ucross(vector2)
            new_values[..., axis3, :] = unit3.values
            new_values[..., axis2, :] = unit3.ucross(unit1).values
        else:
            unit3 = vector2.ucross(unit1)
            new_values[..., axis3, :] = unit3.values
            new_values[..., axis2, :] = unit1.ucross(unit3).values

        return Matrix3(new_values, vector1.mask | vector2.mask)

    # from https://en.wikipedia.org/wiki/Rotation_matrix
    # These are rotations of a vector counterclockwise about an axis
    # The same matrices rotate a coordinate system clockwise about the axis!

    @staticmethod
    def x_rotation(angle, recursive=True):
        """Rotation matrix about X-axis.

        The returned matrix rotates a vector counterclockwise about the X-axis
        bythe specified angle in radians. The same matrix rotates a coordinate
        system clockwise by the same angle.
        """

        angle = Scalar.as_scalar(angle)
        Units.require_angle(angle.units)

        cos_angle = np.cos(angle.values)
        sin_angle = np.sin(angle.values)

        values = np.zeros(angle.shape + (3,3))
        values[...,1,1] =  cos_angle
        values[...,1,2] =  sin_angle
        values[...,2,1] = -sin_angle
        values[...,2,2] =  cos_angle
        values[...,0,0] =  1.

        obj = Matrix3(values.reshape(angle.shape + (3,3)))

        if recursive and angle.derivs:
            matrix = np.zeros(angle.shape + (3,3))
            matrix[...,1,1] = -sin_angle
            matrix[...,1,2] =  cos_angle
            matrix[...,2,1] = -cos_angle
            matrix[...,2,2] = -sin_angle

            for (key, deriv) in self.derivs.iteritems():
                obj.insert_deriv(key, Matrix(matrix * deriv))

        return obj

    @staticmethod
    def y_rotation(angle, recursive=True):
        """Rotation matrix about Y-axis.

        The returned matrix rotates a vector counterclockwise about the Y-axis
        by the specified angle in radians. The same matrix rotates a coordinate
        system clockwise by the same angle.
        """

        angle = Scalar.as_scalar(angle)
        Units.require_angle(angle.units)

        cos_angle = np.cos(angle.values)
        sin_angle = np.sin(angle.values)

        values = np.zeros(angle.shape + (3,3))
        values[...,0,0] =  cos_angle
        values[...,0,2] =  sin_angle
        values[...,2,0] = -sin_angle
        values[...,2,2] =  cos_angle
        values[...,1,1] =  1.

        obj = Matrix3(values.reshape(angle.shape + (3,3)))

        if recursive and angle.derivs:
            matrix = np.zeros(angle.shape + (3,3))
            matrix[...,0,0] = -sin_angle
            matrix[...,0,2] =  cos_angle
            matrix[...,2,0] = -cos_angle
            matrix[...,2,2] = -sin_angle

            for (key, deriv) in self.derivs.iteritems():
                obj.insert_deriv(key, Matrix(matrix * deriv))

        return obj

    @staticmethod
    def z_rotation(angle, recursive=True):
        """Rotation matrix about Z-axis.

        The returned matrix rotates a vector counterclockwise about the Z-axis
        by the specified angle in radians. The same matrix rotates a coordinate
        system clockwise by the same angle.
        """

        angle = Scalar.as_scalar(angle)
        Units.require_angle(angle.units)

        cos_angle = np.cos(angle.values)
        sin_angle = np.sin(angle.values)

        values = np.zeros(angle.shape + (3,3))
        values[...,0,0] =  cos_angle
        values[...,0,1] = -sin_angle
        values[...,1,0] =  sin_angle
        values[...,1,1] =  cos_angle
        values[...,2,2] =  1.

        obj = Matrix3(values.reshape(angle.shape + (3,3)))

        if recursive and angle.derivs:
            matrix = np.zeros(angle.shape + (3,3))
            matrix[...,0,0] = -sin_angle
            matrix[...,0,1] = -cos_angle
            matrix[...,1,0] =  cos_angle
            matrix[...,1,1] = -sin_angle

            for (key, deriv) in self.derivs.iteritems():
                obj.insert_deriv(key, Matrix(matrix * deriv))

        return obj

    @staticmethod
    def axis_rotation(angle, axis=2, recursive=True):
        """Rotation about one of the three primary axes.

        The returned matrix rotates a vector counterclockwise by the specified
        angle about the specified axis (0 for X, 1 for Y, 2 for Z). The same
        matrix rotates a coordinate system clockwise by the same angle.
        """

        axis = axis % 3

        if axis == 2:
            return Matrix3.z_rotation(angle, recursive)

        if axis == 0:
            return Matrix3.x_rotation(angle, recursive)

        return Matrix3.y_rotation(angle, recursive)

    # This matrix rotates J2000 coordinates to another inertial frame,
    # placing the Z-axis along the pole and the X-axis along the J2000
    # ascending node.
    @staticmethod
    def pole_rotation(ra, dec):
        """Rotation matrix to a frame defined by right ascension and
        declination.

        The returned matrix rotates coordinates into a frame where the Z-axis is
        defined by (ra,dec) and the X-axis points along the new equatorial
        plane's ascending node on the original equator.

        Derivatives are not supported.
        """

        ra = Scalar.as_scalar(ra)
        Units.require_angle(ra.units)

        cos_ra = np.cos(ra.values)
        sin_ra = np.sin(ra.values)

        dec = Scalar.as_scalar(dec)
        Units.require_angle(dec.units)

        cos_dec = np.cos(dec.values)
        sin_dec = np.sin(dec.values)

        values = np.stack([-sin_ra,            cos_ra,           0.,
                           -cos_ra * sin_dec, -sin_ra * sin_dec, cos_dec,
                            cos_ra * cos_dec,  sin_ra * cos_dec, sin_dec],
                           axis=-1)
        return Matrix3(values.reshape(values.shape[:-1] + (3,3)))

    def rotate(self, arg, recursive=True):
        """Rotate by this Matrix3, returning an instance of the same subclass.

        Input:
            recursive   if True, the rotated derivatives are included in the
                        object returned.
        """

        # Rotation of a vector or matrix
        if arg.nrank > 0:
            return Qube.dot(self, arg, -1, 0, type(arg), recursive)

        # Rotation of a scalar leaves it unchanged
        else: return arg

    def unrotate(self, arg, recursive=True):
        """Rotate by the inverse of this Matrix3, returning the same subclass.

        Input:
            recursive   if True, the un-rotated derivatives are included in the
                        object returned.
        """

        # Rotation of a vector or matrix
        if arg.nrank > 0:
            return self.dot(self, arg, -2, 0, type(arg), recursive)

        # Rotation of a scalar leaves it unchanged
        else: return arg

    ############################################################################
    # Overrides of arithmetic operators
    ############################################################################

    # Left multiplication
    def __mul__(self, arg, recursive=True):
        """Matrix3 times Scalar returns the same Scalar.

        This overrides the default result of a Matrix times a Scalar.
        """

        # Convert arg to a Scalar if necessary
        original_arg = arg
        if not isinstance(arg, Qube):
            try:
                arg = Scalar.as_scalar(arg)
            except:
                Qube.raise_unsupported_op('*', self, original_arg)

        # Rotate a scalar, returning the scalar unchanged except for new derivs
        if arg.nrank == 0:
            if not recursive: return arg.without_derivs()
            return arg

        # For every other purpose, use the default multiply
        return Qube.__mul__(self, original_arg)

    # In-place multiplication only works for a Matrix3
    def __imul__(self, arg):
        self.require_writable()
        if Qube.is_empty(arg): return arg

        # Attempt a conversion to Matrix3
        original_arg = arg
        try:
            arg = Matrix3.as_matrix3(arg)
        except:
            Qube.raise_unsupported_op('*=', self, original_arg)

        return Qube.__imul__(self, arg)

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

        return self.transpose(recursive=recursive)

################################################################################
# Decomposition into rotations
#
# From: http://www.lfd.uci.edu/~gohlke/code/transformations.py.html
#
# A triple of Euler angles can be applied/interpreted in 24 ways, which can
# be specified using a 4 character string or encoded 4-tuple:
# 
#   *Axes 4-string*: e.g. 'sxyz' or 'ryxy'
# 
#   - first character : rotations are applied to 's'tatic or 'r'otating frame
#   - remaining characters : successive rotation axis 'x', 'y', or 'z'
# 
#   *Axes 4-tuple*: e.g. (0, 0, 0, 0) or (1, 1, 1, 1)
# 
#   - inner axis: code of axis ('x':0, 'y':1, 'z':2) of rightmost matrix.
#   - parity : even (0) if inner axis 'x' is followed by 'y', 'y' is followed
#     by 'z', or 'z' is followed by 'x'. Otherwise odd (1).
#   - repetition : first and last axis are same (1) or different (0).
#   - frame : rotations are applied to static (0) or rotating (1) frame.
################################################################################

    # axis sequences for Euler angles
    _NEXT_AXIS = [1, 2, 0, 1]

    # map axes strings to/from tuples of inner axis, parity, repetition, frame
    _AXES2TUPLE = {
        'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
        'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
        'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
        'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
        'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
        'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
        'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
        'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

    _TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

    EPSILON = 1.e-15
    TWOPI = 2. * np.pi

    @staticmethod
    def from_euler(ai, aj, ak, axes='rzxz'):
        """Return homogeneous rotation matrix from Euler angles and axis
        sequence.

        ai, aj, ak : Euler's roll, pitch and yaw angles
        axes : One of 24 axis sequences as string or encoded tuple

        >>> R = euler_matrix(1, 2, 3, 'syxz')
        >>> np.allclose(np.sum(R[0]), -1.34786452)
        True
        >>> R = euler_matrix(1, 2, 3, (0, 1, 0, 1))
        >>> np.allclose(np.sum(R[0]), -0.383436184)
        True
        >>> ai, aj, ak = (4*np.pi) * (np.random.random(3) - 0.5)
        >>> for axes in _AXES2TUPLE.keys():
        ...    R = euler_matrix(ai, aj, ak, axes)
        >>> for axes in _TUPLE2AXES.keys():
        ...    R = euler_matrix(ai, aj, ak, axes)

        """

        ai = Scalar.as_scalar(ai)
        aj = Scalar.as_scalar(aj)
        ak = Scalar.as_scalar(ak)
        Units.require_angle(ai.units)
        Units.require_angle(aj.units)
        Units.require_angle(ak.units)

        (ai,aj,ak) = Qube.broadcast(ai,aj,ak)

        axes = axes.lower()
        try:
            (firstaxis, parity, repetition, frame) = Matrix3._AXES2TUPLE[axes]
        except (AttributeError, KeyError):
            Matrix3._TUPLE2AXES[axes]  # validation
            firstaxis, parity, repetition, frame = axes

        i = firstaxis
        j = Matrix3._NEXT_AXIS[i+parity]
        k = Matrix3._NEXT_AXIS[i-parity+1]

        if frame:
            (ai, ak) = (ak, ai)

        if parity:
            (ai, aj, ak) = (-ai, -aj, -ak)

        si = ai.sin().values
        sj = aj.sin().values
        sk = ak.sin().values

        ci = ai.cos().values
        cj = aj.cos().values
        ck = ak.cos().values

        cc = ci * ck
        cs = ci * sk

        sc = si * ck
        ss = si * sk

        matrix = np.empty(ai.shape + (3,3))
        if repetition:
            matrix[...,i,i] =  cj
            matrix[...,i,j] =  sj * si
            matrix[...,i,k] =  sj * ci
            matrix[...,j,i] =  sj * sk
            matrix[...,j,j] = -cj * ss + cc
            matrix[...,j,k] = -cj * cs - sc
            matrix[...,k,i] = -sj * ck
            matrix[...,k,j] =  cj * sc + cs
            matrix[...,k,k] =  cj * cc - ss
        else:
            matrix[...,i,i] =  cj * ck
            matrix[...,i,j] =  sj * sc - cs
            matrix[...,i,k] =  sj * cc + ss
            matrix[...,j,i] =  cj * sk
            matrix[...,j,j] =  sj * ss + cc
            matrix[...,j,k] =  sj * cs - sc
            matrix[...,k,i] = -sj
            matrix[...,k,j] =  cj * si
            matrix[...,k,k] =  cj * ci

        return Matrix3(matrix, ai.mask | aj.mask | ak.mask)

    def to_euler(self, axes='rzxz'):
        """Return three Scalars of Euler angles from this Matrix3, given a
        specified axis sequence.

        axes : One of 24 axis sequences as string or encoded tuple

        Note that many Euler angle triplets can describe one matrix.

        >>> R0 = euler_matrix(1, 2, 3, 'syxz')
        >>> al, be, ga = euler_from_matrix(R0, 'syxz')
        >>> R1 = euler_matrix(al, be, ga, 'syxz')
        >>> np.allclose(R0, R1)
        True
        >>> angles = (4*np.pi) * (np.random.random(3) - 0.5)
        >>> for axes in _AXES2TUPLE.keys():
        ...    R0 = euler_matrix(axes=axes, *angles)
        ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
        ...    if not np.allclose(R0, R1): print(axes, "failed")

        """

        try:
            (firstaxis, parity, repetition,
                                frame) = Matrix3._AXES2TUPLE[axes.lower()]

        except (AttributeError, KeyError):
            Matrix3._TUPLE2AXES[axes]  # validation
            firstaxis, parity, repetition, frame = axes

        i = firstaxis
        j = Matrix3._NEXT_AXIS[i+parity]
        k = Matrix3._NEXT_AXIS[i-parity+1]

        matvals = self.values[np.newaxis]
        if repetition:
            sy = np.sqrt(matvals[...,i,j]**2 + matvals[...,i,k]**2)

            ax = np.arctan2(matvals[...,i,j],  matvals[...,i,k])
            ay = np.arctan2(sy,                matvals[...,i,i])
            az = np.arctan2(matvals[...,j,i], -matvals[...,k,i])

            mask = (sy <= Matrix3.EPSILON)
            if np.any(mask):
                ax[mask] = np.arctan2(-matvals[...,j,k], matvals[...,j,j])
                ay[mask] = np.arctan2( sy,               matvals[...,i,i])
                az[mask] = 0.

        else:
            cy = np.sqrt(matvals[...,i,i]**2 + matvals[...,j,i]**2)

            ax = np.arctan2( matvals[...,k,j], matvals[...,k,k])
            ay = np.arctan2(-matvals[...,k,i], cy)
            az = np.arctan2( matvals[...,j,i], matvals[...,i,i])

            mask = (cy <= Matrix3.EPSILON)
            if np.any(mask):
                ax[mask] = np.arctan2(-matvals[...,j,k], matvals[...,j,j])[mask]
                ay[mask] = np.arctan2(-matvals[...,k,i], cy)[mask]
                az[mask] = 0.

        if parity:
            ax, ay, az = -ax, -ay, -az
        if frame:
            ax, az = az, ax

        return (Scalar(ax[0] % Matrix3.TWOPI, self.mask),
                Scalar(ay[0] % Matrix3.TWOPI, self.mask),
                Scalar(az[0] % Matrix3.TWOPI, self.mask))

    def to_quaternion(self):
        """Converts this Matrix3 to an equivalent Quaternion."""

        return Qube.QUATERNION_CLASS.from_matrix3(self)

# Useful class constants

Matrix3.IDENTITY = Matrix3([[1,0,0],[0,1,0],[0,0,1]]).as_readonly()
Matrix3.MASKED = Matrix3([[1,0,0],[0,1,0],[0,0,1]], True).as_readonly()

################################################################################
# Once defined, register with Qube class
################################################################################

Qube.MATRIX3_CLASS = Matrix3

################################################################################
