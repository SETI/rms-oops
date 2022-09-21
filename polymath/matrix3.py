################################################################################
# polymath/matrix3.py: Matrix3 subclass of PolyMath Matrix class
################################################################################

from __future__ import division
import numpy as np

from .qube    import Qube
from .scalar  import Scalar
from .vector3 import Vector3
from .matrix  import Matrix
from .units   import Units

class Matrix3(Matrix):
    """A Qube of 3x3 rotation matrices."""

    NRANK = 2           # the number of numerator axes.
    NUMER = (3,3)       # shape of the numerator.

    FLOATS_OK = True    # True to allow floating-point numbers.
    INTS_OK = False     # True to allow integers.
    BOOLS_OK = False    # True to allow booleans.

    UNITS_OK = False    # True to allow units; False to disallow them.
    DERIVS_OK = True    # True to disallow derivatives; False to allow them.

    DEFAULT_VALUE = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])

    #===========================================================================
    @staticmethod
    def as_matrix3(arg, recursive=True):
        """Convert to Matrix3. The result is not checked to be unitary.

        Quaternions are converted to matrices.

        Inputs:
            arg         the object to convert.
            recursive   True to include derivatives in the returned result.
        """

        if isinstance(arg, Matrix3):
            if recursive:
                return arg
            return arg.wod

        if isinstance(arg, Qube):
            if isinstance(arg, Qube.QUATERNION_CLASS):
                return arg.to_matrix3(recursive)

            arg = Matrix3(arg._values_, arg._mask_, example=arg)
            if recursive:
                return arg
            return arg.wod

        return Matrix3(arg)

    #===========================================================================
    @staticmethod
    def twovec(vector1, axis1, vector2, axis2, recursive=True):
        """A rotation matrix defined by two vectors.

        The returned matrix rotates to a right-handed coordinate frame having
        vector1 pointing along a specified axis (axis1=0 for X, 1 for Y, 2 for
        Z) and vector2 pointing into the half-plane defined by (axis1,axis2).
        """

        # Based on the SPICE source code for TWOVEC()

        # Make shapes and types consistent
        unit1 = Vector3.as_vector3(vector1).unit(recursive)
        vector2 = Vector3.as_vector3(vector2, recursive)
        (unit1, vector2) = Qube.broadcast(unit1, vector2)

        # Denominators are disallowed
        assert unit1.denom   == (), 'denominator is disallowed'
        assert vector2.denom == (), 'denominator is disallowed'

        # Define the remaining two columns of the matrix
        axis3 = 3 - axis1 - axis2
        if (3 + axis2 - axis1) % 3 == 1:        # if (0,1), (1,2) or (2,0)
            unit3 = unit1.ucross(vector2, recursive)
            unit2 = unit3.ucross(unit1, recursive)
        else:
            unit3 = vector2.ucross(unit1, recursive)
            unit2 = unit1.ucross(unit3, recursive)

        # Assemble the values into an array
        array = np.empty(unit1.shape + (3,3))
        array[...,axis1,:] = unit1._values_
        array[...,axis2,:] = unit2._values_
        array[...,axis3,:] = unit3._values_

        # Construct the result
        result = Matrix3(array, unit1._mask_ | vector2._mask_)

        # Fill in derivatives if necessary
        if recursive and (unit1.derivs or vector2.derivs):

            # Find all the derivatives and their denominator shapes
            denoms = {}
            for (key,deriv) in unit1.derivs.items():
                denoms[key] = deriv.denom
            for (key,deriv) in vector2.derivs.items():
                if key in denoms:
                    if deriv.denom != denoms[key]:
                        raise ValueError('denominator shape mismatch: %s, %s' %
                                         (denoms[key], deriv.denom))
                else:
                    denoms[key] = vector2.derivs[key].denom

            derivs = {}
            for (key,denom) in denoms.items():
                drank = len(denom)
                deriv = np.zeros(unit1.shape + (3,3) + denom)

                suffix = (drank + 1) * (slice(None),)
                if key in unit1.derivs:
                    deriv[(Ellipsis,axis1)+suffix] = unit1.derivs[key]._values_
                if key in unit2.derivs:
                    deriv[(Ellipsis,axis2)+suffix] = unit2.derivs[key]._values_
                if key in unit3.derivs:
                    deriv[(Ellipsis,axis3)+suffix] = unit3.derivs[key]._values_

                derivs[key] = Matrix3(deriv, mask=result._mask_, drank=drank)

            result.insert_derivs(derivs)

        if unit1.readonly and vector2.readonly:
            result = result.as_readonly()

        return result

    # from https://en.wikipedia.org/wiki/Rotation_matrix
    # These are rotations of a vector counterclockwise about an axis
    # The same matrices rotate a coordinate system clockwise about the axis!

    @staticmethod
    def x_rotation(angle, recursive=True):
        """Rotation matrix about X-axis.

        The returned matrix rotates a vector counterclockwise about the X-axis
        by the specified angle in radians. The same matrix rotates a coordinate
        system clockwise by the same angle.
        """

        angle = Scalar.as_scalar(angle)
        Units.require_angle(angle.units)

        cos_angle = np.cos(angle._values_)
        sin_angle = np.sin(angle._values_)

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

            for (key, deriv) in angle.derivs.items():
                obj.insert_deriv(key, Matrix(matrix * deriv))

        return obj

    #===========================================================================
    @staticmethod
    def y_rotation(angle, recursive=True):
        """Rotation matrix about Y-axis.

        The returned matrix rotates a vector counterclockwise about the Y-axis
        by the specified angle in radians. The same matrix rotates a coordinate
        system clockwise by the same angle.
        """

        angle = Scalar.as_scalar(angle)
        Units.require_angle(angle.units)

        cos_angle = np.cos(angle._values_)
        sin_angle = np.sin(angle._values_)

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

            for (key, deriv) in angle.derivs.items():
                obj.insert_deriv(key, Matrix(matrix * deriv))

        return obj

    #===========================================================================
    @staticmethod
    def z_rotation(angle, recursive=True):
        """Rotation matrix about Z-axis.

        The returned matrix rotates a vector counterclockwise about the Z-axis
        by the specified angle in radians. The same matrix rotates a coordinate
        system clockwise by the same angle.
        """

        angle = Scalar.as_scalar(angle)
        Units.require_angle(angle.units)

        cos_angle = np.cos(angle._values_)
        sin_angle = np.sin(angle._values_)

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

            for (key, deriv) in angle.derivs.items():
                obj.insert_deriv(key, Matrix(matrix * deriv))

        return obj

    #===========================================================================
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

    #===========================================================================
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

        cos_ra = np.cos(ra._values_)
        sin_ra = np.sin(ra._values_)

        dec = Scalar.as_scalar(dec)
        Units.require_angle(dec.units)

        cos_dec = np.cos(dec._values_)
        sin_dec = np.sin(dec._values_)

        values = np.stack([-sin_ra,            cos_ra,           0.,
                           -cos_ra * sin_dec, -sin_ra * sin_dec, cos_dec,
                            cos_ra * cos_dec,  sin_ra * cos_dec, sin_dec],
                           axis=-1)
        return Matrix3(values.reshape(values.shape[:-1] + (3,3)))

    #===========================================================================
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
        else:
            return arg

    #===========================================================================
    def unrotate(self, arg, recursive=True):
        """Rotate by the inverse of this Matrix3, returning the same subclass.

        Input:
            recursive   if True, the un-rotated derivatives are included in the
                        object returned.
        """

        # Rotation of a vector or matrix
        if arg.nrank > 0:
            return Qube.dot(self, arg, -2, 0, type(arg), recursive)

        # Rotation of a scalar leaves it unchanged
        else:
            return arg

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
            if not recursive:
                return arg.wod
            return arg

        # For every other purpose, use the default multiply
        return Qube.__mul__(self, original_arg)

    # In-place multiplication only works for a Matrix3
    def __imul__(self, arg):
        self.require_writable()

        # Attempt a conversion to Matrix3
        original_arg = arg
        try:
            arg = Matrix3.as_matrix3(arg)
        except:
            Qube.raise_unsupported_op('*=', self, original_arg)

        return Qube.__imul__(self, arg)

    #===========================================================================
    def reciprocal(self, recursive=True, nozeros=False):
        """A object equivalent to the reciprocal of this object.

        Input:
            recursive   True to return the derivatives of the reciprocal too;
                        otherwise, derivatives are removed.
            nozeros     False (the default) to mask out any zero-valued items in
                        this object prior to the divide. Set to True only if you
                        know in advance that this object has no zero-valued
                        items.
        """

        return self.transpose(recursive=recursive)

    ############################################################################
    # Decomposition into rotations
    #
    # From: http://www.lfd.uci.edu/~gohlke/code/transformations.py.html
    #
    # A triple of Euler angles can be applied/interpreted in 24 ways, which can
    # be specified using a 4 character string or encoded 4-tuple:
    #
    #   *Axes 4-string*: e.g. 'sxyz' or 'ryxy'
    #
    #   - first character : rotations are applied to 's'tatic or 'r'otating
    #     frame
    #   - remaining characters : successive rotation axis 'x', 'y', or 'z'
    #
    #   *Axes 4-tuple*: e.g. (0, 0, 0, 0) or (1, 1, 1, 1)
    #
    #   - inner axis: code of axis ('x':0, 'y':1, 'z':2) of rightmost matrix.
    #   - parity : even (0) if inner axis 'x' is followed by 'y', 'y' is
    #     followed by 'z', or 'z' is followed by 'x'. Otherwise odd (1).
    #   - repetition : first and last axis are same (1) or different (0).
    #   - frame : rotations are applied to static (0) or rotating (1) frame.
    ############################################################################

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

    #===========================================================================
    @staticmethod
    def from_euler(ai, aj, ak, axes='rzxz'):
        """A homogeneous rotation matrix from Euler angles and axis sequence.

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

        si = ai.sin()._values_
        sj = aj.sin()._values_
        sk = ak.sin()._values_

        ci = ai.cos()._values_
        cj = aj.cos()._values_
        ck = ak.cos()._values_

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

        return Matrix3(matrix, ai._mask_ | aj._mask_ | ak._mask_)

    #===========================================================================
    def to_euler(self, axes='rzxz'):
        """Three Scalars of Euler angles from this Matrix3, given a specified
        axis sequence.

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

        matvals = self._values_[np.newaxis]
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

        return (Scalar(ax[0] % Matrix3.TWOPI, self._mask_),
                Scalar(ay[0] % Matrix3.TWOPI, self._mask_),
                Scalar(az[0] % Matrix3.TWOPI, self._mask_))

    #===========================================================================
    def to_quaternion(self):
        """Converts this Matrix3 to an equivalent Quaternion."""

        return Qube.QUATERNION_CLASS.from_matrix3(self)

################################################################################
# Useful class constants
################################################################################

Matrix3.IDENTITY = Matrix3([[1,0,0],[0,1,0],[0,0,1]]).as_readonly()
Matrix3.MASKED = Matrix3([[1,0,0],[0,1,0],[0,0,1]], True).as_readonly()

################################################################################
# Once defined, register with Qube class
################################################################################

Qube.MATRIX3_CLASS = Matrix3

################################################################################
