################################################################################
# oops/transform.py: Class Transform
################################################################################

import numpy as np

from polymath import Qube, Scalar, Vector3, Matrix3

class Transform(object):
    """An object describing a coordinate transformation, defined by a rotation
    matrix plus an optional angular rotation vector indicating how that frame is
    rotating. The components are interpreted as follows:
        matrix          rotates coordinates from the reference coordinate frame
                        into the target frame.
        omega           the angular rotation vector of the target frame
                        relative to the reference, and specified in the
                        reference coordinate frame.

    Given a state vector (pos,vel) in the reference coordinate frame, we can
    convert to a state vector in the target frame as follows:
        pos_target = matrix * pos_ref
        vel_target = matrix * (vel_ref - omega x pos_ref)

    The inverse transformation is:
        pos_ref = matrix-T * pos_target
        vel_ref = matrix-T * vel_target + omega x pos_ref
                = matrix-T * (vel_target - omega1 x pos_target)
    where
        omega1     = -matrix * omega
                   = the negative of the rotation vector, transformed into the
                     target frame.

    With this definition, a transform can also be used to describe the
    orientation and rotation rate of a planetary body:
        matrix[0,:] = the instantaneous X-axis in reference coordinates
        matrix[1,:] = the instantaneous Y-axis in reference coordinates
        matrix[2,:] = the instantaneous Z-axis in reference coordinates
        omega       = the body's rotation vector in reference coordinates.

    A Tranform object also has these attributes that describe the transform it
    is performing:
        frame_id        the registered ID of the target frame.
        reference_id    the registered ID of the reference frame. The transform
                        rotates coordinates in the reference frame into
                        coordinates in the target frame. The reference frame
                        must have shape [].
        origin_id       the path ID of the origin if this is a rotating frame;
                        None otherwise.

    This is a static property, generated only if needed
        shape           the intrinsic shape of the transform.
    """

    ############################################################################
    # Note:
    # The class constants are defined at the end of __init__.py:
    #   Transform.FRAME_CLASS
    #   Transform.IDENTITY
    ############################################################################

    def __init__(self, matrix, omega, frame, reference, origin=None):
        """Constructor for a Transform object.

        Input:
            matrix      the Matrix3 object that is used to rotate coordinates
                        from the reference frame into the new frame.
            omega       the spin vector for the coordinate frame, given in
                        coordinates of the reference frame.
            frame       the frame or frame ID into which this Transform rotates.
            reference   the frame or frame ID from which this Transform rotates.
            origin      the path or path ID of the center of rotation. If None,
                        it is derived from the reference frame.
        """

        self.matrix = Matrix3.as_matrix3(matrix)
        self.omega  = Vector3.as_vector3(omega)

        self.is_fixed = (self.omega == Vector3.ZERO)

        self.frame     = Transform.FRAME_CLASS.as_wayframe(frame)
        self.reference = Transform.FRAME_CLASS.as_wayframe(reference)

        if origin is not None:
            self.origin = origin
        elif reference is not None:
            self.origin = self.reference.origin
        else:
            self.origin = None

        self.filled_shape = None            # filled in only when needed
        self.filled_omega1 = None
        self.filled_matrix_with_deriv = None
        self.filled_inverse_matrix = None
        self.filled_inverse_with_deriv = None

    def __getstate__(self):
        return (self.matrix, self.omega, self.frame, self.reference,
                self.origin)

    def __setstate__(self, state):
        self.__init__(*state)

    #===========================================================================
    @property
    def shape(self):
        """The intrinsic shape of the Transform.

        This is a bit expensive to generate and used rarely, so it is
        implemented as a property rather than an attribute.
        """

        if self.filled_shape is None:
            self.filled_shape =  Qube.broadcasted_shape(self.matrix,
                                                        self.omega,
                                                        self.frame,
                                                        self.reference)
        return self.filled_shape

    #===========================================================================
    @property
    def omega1(self):
        """The negative rotation matrix transformed into the target frame.

        Used for the inverse transform.
        """

        if self.filled_omega1 is None:
            self.filled_omega1 = -self.matrix * self.omega

        return self.filled_omega1

    #===========================================================================
    @property
    def matrix_with_deriv(self):
        """The rotation matrix with its time-derivative filled in."""

        if self.filled_matrix_with_deriv is None:
            self.filled_matrix_with_deriv = self.matrix.clone()

            d_dt = -self.matrix * self.omega.cross_product_as_matrix()
            self.filled_matrix_with_deriv.insert_deriv('t', d_dt, override=True)

        return self.filled_matrix_with_deriv

    #===========================================================================
    @property
    def inverse_matrix(self):
        """The inverse rotation matrix."""

        if self.filled_inverse_matrix is None:
            self.filled_inverse_matrix = self.matrix.transpose()

        return self.filled_inverse_matrix

    #===========================================================================
    @property
    def inverse_with_deriv(self):
        """The inverse rotation matrix with its time-derivative filled in."""

        if self.filled_inverse_with_deriv is None:
            inverse = self.matrix_with_deriv.inverse(recursive=True)
            self.filled_inverse_with_deriv = inverse

        return self.filled_inverse_with_deriv

    #===========================================================================
    def __str__(self):
        return ('Transform(shape=' +
                repr(self.shape).replace(' ', '') + '/' +
                repr(self.frame.frame_id) + ')')

    #===========================================================================
    def __repr__(self):
        return self.__str__()

    #===========================================================================
    @staticmethod
    def identity(frame):
        """An identity transform from a frame to itself."""

        return Transform(Matrix3.IDENTITY, Vector3.ZERO, frame, frame)

    ############################################################################
    # Vector operations
    ############################################################################

    def rotate(self, pos, derivs=True):
        """Rotate the coordinates of a position or matrix.

        Optionally, it also rotates any derivatives.

        Input:
            pos         a Vector3, Vector or Matrix object. The size of the
                        leading axis must be 3. Anything not a subclass of Qube
                        (e.g., a list or tuple) is converted to a Vector3 first.

            derivs      True to calculate the time-derivative as well.

        Return:         an equivalent Vector3 position transformed into the
                        target frame.

                        If derivs is True, then the returned position has a
                        time derivative.
        """

        if pos is None:
            return None

        if not isinstance(pos, Qube):
            pos = Vector3.as_vector3(pos)

        if derivs:
            return self.matrix_with_deriv * pos
        else:
            return self.matrix * pos.wod

    #===========================================================================
    def rotate_pos_vel(self, pos, vel):
        """Rotate the coordinates of a position and velocity.

        This function ignores derivatives. It does correctly allow for the
        artificial component of the velocity for a position off the origin in a
        rotating frame.

        Input:
            pos         position as a Vector3, in the reference frame.

            vel         velocity as a Vector3, in the reference frame.

        Return:         a tuple containing the same Vector3 position and
                        velocity transformed into the target frame.
        """

        pos = Vector3.as_vector3(pos)
        vel = Vector3.as_vector3(vel)

        # pos_target = matrix * pos_ref
        # vel_target = matrix * (vel_ref - omega x pos_ref)

        pos_target = self.matrix * pos

        velocity_is_easy = self.is_fixed or (pos == Vector3.ZERO)
        if velocity_is_easy:
            vel_target = self.matrix * vel
        else:
            vel_target = self.matrix * (vel - self.omega.cross(pos))

        return (pos_target, vel_target)

    #===========================================================================
    def unrotate(self, pos, derivs=True):
        """Un-rotate the coordinates of a position into the reference frame.

        Input:
            pos         a Vector3, VectorN or Matrix object. The size of the
                        leading axis must be 3. Anything not a subclass of Array
                        (e.g., a list or tuple) is converted to a Vector3 first.
                        Velocity is always assumed zero.

            derivs      True to calculate dpos/dpos and dpos/dt as well.

        Return:         the same Vector3 position transformed back into the
                        reference frame.

                        If derivs is True, then the returned position has a
                        subfield "d_dt", a Vector3 representing the partial
                        derivatives with respect to time.
        """

        if pos is None:
            return None

        if not isinstance(pos, Qube):
            pos = Vector3.as_vector3(pos)

        if derivs:
            return self.inverse_with_deriv * pos
        else:
            return self.inverse_matrix * pos.wod

    #===========================================================================
    def unrotate_pos_vel(self, pos, vel):
        """Un-rotates the coordinates of a position and velocity.

        Derivatives are not supported.

        Input:
            pos         position as a Vector3, in the target frame.

            vel         velocity as a Vector3, in the target frame.

        Return:         a tuple containing the same Vector3 position and
                        velocity transformed back into the reference frame.
        """

        pos = Vector3.as_vector3(pos)
        vel = Vector3.as_vector3(vel)

        # pos_ref = matrix-T * pos_target
        # vel_ref = matrix-T * vel_target + omega x pos_ref

        pos_ref = self.matrix.unrotate(pos)

        velocity_is_easy = self.is_fixed or pos == Vector3.ZERO
        if velocity_is_easy:
            vel_ref = self.matrix.unrotate(vel)
        else:
            vel_ref = self.matrix.unrotate(vel) + self.omega.cross(pos_ref)

        return (pos_ref, vel_ref)

    ############################################################################
    # Operations on Transforms
    ############################################################################

    def invert(self):
        """The inverse transformation."""

        return Transform(self.matrix.reciprocal(), self.omega1,
                         self.reference, self.frame, self.origin)

    #===========================================================================
    def rotate_transform(self, arg):
        """Apply this transform to another, as a left-multiply.

        The result is a single transform that converts coordinates in the
        reference frame of the argument transform into the frame of this
        transform.
        """

        # Two tranforms
        #   P1 = M P0; V1 = M (V0 - omega x P0)
        #   P2 = N P1; V2 = N (V1 - kappa x P1)
        #
        # Combine...
        #   P2 = [N M] P0
        #
        #   V2 = N [M (V0 - omega x P0) - kappa x M P0]
        #      = N [M (V0 - omega x P0) - M MT (kappa x M P0)
        #      = N M [(V0 - omega x P0) - MT ([M MT kappa] x M P0)]
        #      = N M [(V0 - omega x P0) - MT ([M MT kappa] x M P0)]
        #      = N M [(V0 - omega x P0) - MT M ([MT kappa] x P0)]
        #      = N M [(V0 - [omega + MT kappa] x P0)]

        assert self.reference == arg.frame

        if self.origin is None:
            origin = arg.origin
        elif arg.origin is None:
            origin = self.origin
        else:
            origin = self.origin
            # assert self.origin_id == arg.origin_id

        return Transform(self.matrix.rotate(arg.matrix),
                         arg.matrix.unrotate(self.omega) + arg.omega,
                         self.frame, arg.reference, origin)

    #===========================================================================
    def unrotate_transform(self, arg):
        """Apply the inverse of this transform to another, as a left-multiply.

        The result is a single transform that applies the convert coordinates
        in the parent frame of the argument transform into the parent frame of
        this transform. I.e., if arg rotates A to B and self rotates C to B,
        then the result rotates A to C.
        """

        return self.invert().rotate_transform(arg)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Transform(unittest.TestCase):

    def runTest(self):

        np.random.seed(5819)

        # Additional imports needed for testing
        from oops.frame import Frame, Wayframe

        # Fake out the FRAME REGISTRY with something that has .shape = ()
        Frame.WAYFRAME_REGISTRY["TEST"] = Wayframe("J2000")
        Frame.WAYFRAME_REGISTRY["SPIN"] = Wayframe("J2000")

        tr = Transform(Matrix3(np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])),
                       Vector3(np.array([0.,0.,0.])), "J2000", "J2000")

        p = Vector3(np.random.rand(2,1,4,3))
        v = Vector3(np.random.rand(  3,4,3))

        self.assertEqual(tr.rotate_pos_vel(p,v)[0], p)
        self.assertEqual(tr.rotate_pos_vel(p,v)[1], v)

        self.assertEqual(tr.unrotate_pos_vel(p,v)[0], p)
        self.assertEqual(tr.unrotate_pos_vel(p,v)[1], v)

        tr = tr.invert()

        self.assertEqual(tr.rotate_pos_vel(p,v)[0], p)
        self.assertEqual(tr.rotate_pos_vel(p,v)[1], v)

        self.assertEqual(tr.unrotate_pos_vel(p,v)[0], p)
        self.assertEqual(tr.unrotate_pos_vel(p,v)[1], v)

        tr = Transform(Matrix3([[1,0,0],[0,1,0],[0,0,1]]),
                       Vector3([0,0,1]), "SPIN", "J2000")

        self.assertEqual(tr.unrotate_pos_vel(p,v)[0], p)
        self.assertEqual(Scalar(tr.rotate_pos_vel(p,v)[1].mvals[...,2]), Scalar(v.mvals[...,2]))
        self.assertEqual(Scalar(tr.rotate_pos_vel(p,v)[1].mvals[...,0]),
                                                Scalar(v.mvals[...,0]) + Scalar(p.mvals[...,1]))
        self.assertEqual(Scalar(tr.rotate_pos_vel(p,v)[1].mvals[...,1]),
                                                Scalar(v.mvals[...,1]) - Scalar(p.mvals[...,0]))

        tr = tr.invert()

        self.assertEqual(tr.rotate(p), p)
        self.assertEqual(Scalar(tr.rotate_pos_vel(p,v)[1].mvals[...,2]), Scalar(v.mvals[...,2]))
        self.assertEqual(Scalar(tr.rotate_pos_vel(p,v)[1].mvals[...,0]),
                                                Scalar(v.mvals[...,0]) - Scalar(p.mvals[...,1]))
        self.assertEqual(Scalar(tr.rotate_pos_vel(p,v)[1].mvals[...,1]),
                                                Scalar(v.mvals[...,1]) + Scalar(p.mvals[...,0]))

        a = Vector3(np.random.rand(3,1,3))
        b = Vector3(np.random.rand(1,1,3))
        m = Matrix3.twovec(a,0,b,1)
        omega = Vector3(np.random.rand(3,1,3))

        tr = Transform(m, omega, "TEST", "J2000")

#         self.assertEqual(tr.unrotate(p), tr.invert().rotate(p))
#         self.assertEqual(tr.rotate(p), tr.invert().unrotate(p))
        eps = 1.e-15
        self.assertTrue(np.all(np.abs(tr.unrotate(p).vals - tr.invert().rotate(p).vals)) < eps)
        self.assertTrue(np.all(np.abs(tr.rotate(p).vals - tr.invert().unrotate(p).vals)) < eps)

        eps = 1.e-15
        diff = tr.unrotate_pos_vel(p,v)[1] - tr.invert().rotate_pos_vel(p,v)[1]
        self.assertTrue(np.all(diff.vals > -eps))
        self.assertTrue(np.all(diff.vals <  eps))

        diff = tr.rotate_pos_vel(p,v)[1] - tr.invert().unrotate_pos_vel(p,v)[1]
        self.assertTrue(np.all(diff.vals > -eps))
        self.assertTrue(np.all(diff.vals <  eps))

        # Transform derivatives are unit tested as part of the SpinFrame tests

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
