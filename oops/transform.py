################################################################################
# oops_/transform.py: Class Transform
#
# 2/5/12 Modified (MRS): Revised for consistent style.
# 3/12/12 MRS - Implemented derivative support.
################################################################################

import numpy as np

from oops_.array.all import *
import oops_.registry as registry

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

    ZERO_MATRIXN = VectorN([0,0,0]).as_column()
    ZERO_VECTOR3 = Vector3([0,0,0])

    def __init__(self, matrix, omega, frame_id, reference_id, origin_id=None):
        """Constructor for a Transform object.

        Input:
            matrix      the Matrix3 object that is used to rotate coordinates
                        from the reference frame into the new frame.
            omega       the spin vector for the coordinate frame, given in
                        coordinates of the reference frame.
            frame_id    the ID of the frame into which this Transform rotates.
            reference_id the ID of the frame from which this Transform rotates.
            origin_id   the path ID of the center of rotation. If None, it is
                        derived from the reference frame ID.
        """

        self.matrix = Matrix3.as_matrix3(matrix)
        self.omega  = Vector3.as_vector3(omega)

        self.is_fixed = (self.omega == Transform.ZERO_VECTOR3)
        self.matrixn = MatrixN(self.matrix)

        self.frame_id     = frame_id
        self.reference_id = reference_id

        if origin_id is not None:
            self.origin_id = origin_id
        elif reference_id is not None:
            frame = registry.FRAME_REGISTRY[self.reference_id]
            self.origin_id = frame.origin_id
        else:
            self.origin_id = None

        self.filled_shape = None            # filled in only when needed
        self.filled_omega1 = None
        self.filled_dmatrix_dt = None

    @property
    def shape(self):
        """Returns the intrinsic shape of the Transform. This is a bit expensive
        to generate and used rarely, so it is implemented as a property rather
        than an attribute."""

        if self.filled_shape is None:
            self.filled_shape =  Array.broadcast_shape((
                                          self.matrix,
                                          self.omega,
                                          registry.as_frame(self.frame_id),
                                          registry.as_frame(self.reference_id)))
        return self.filled_shape

    @property
    def omega1(self):
        """Returns the negative rotation matrix transformed into the target
        frame. Sometimes used for the inverse transform."""

        if self.filled_omega1 is None:
            self.filled_omega1 = -self.matrix * self.omega

        return self.filled_omega1

    @property
    def dmatrix_dt(self):
        """Returns the time-derivative of the rotation matrix."""

        if self.filled_dmatrix_dt is None:
            self.filled_dmatrix_dt = (-self.matrix *
                                       self.omega.cross_product_as_matrix())

        return self.filled_dmatrix_dt

    # string operations
    def __str__(self):
        return ("Transform(shape=" +
                repr(self.shape).replace(' ', '') + "/" +
                repr(self.frame_id) + ")")

    def __repr__(self): return self.__str__()

################################################################################
# Vector operations
################################################################################

    def rotate(self, pos, derivs=False):
        """Rotates the coordinates of a position or matrix forward into the
        target frame. It also rotates any subarrays.

        Input:
            pos         a Vector3, VectorN or MatrixN object. The size of the
                        leading axis must be 3. Anything not a subclass of Array
                        (e.g., a list or tuple) is converted to a Vector3 first.
                        Velocity is always assumed zero.

            derivs      True to calculate the time-derivative as well.

        Return:         an equivalent Vector3 position transformed into the
                        target frame.

                        If derivs is True, then the returned position has a
                        subfield "d_dt", a Vector3 representing the partial
                        derivatives with respect to time.
        """

        if pos is None: return None
        if pos == Empty(): return pos

        if not isinstance(pos, Array):
            pos = Vector3.as_vector3(pos)

        self.matrix.subfield_math = derivs
        self.omega.subfield_math = derivs

        pos_target = self.matrix * pos

        # Calculate/update the time-derivative if necessary
        if derivs:
            if self.is_fixed or pos == Transform.ZERO_VECTOR3:
                pos_target.insert_subfield_if_new("d_dt",
                                        Transform.ZERO_MATRIXN)
            else:
                pos_target.add_to_subfield("d_dt",
                                        self.dmatrix_dt.rotate(pos).as_column())

        return pos_target

    def rotate_pos_vel(self, pos, vel):
        """Rotates the coordinates of a position and velocity forward into the
        target frame, also allowing for the artificial component of the velocity
        for a position off the origin in a rotating frame.

        Input:
            pos         position as a Vector3, in the reference frame.

            vel         velocity as a Vector3, in the reference frame.

        Return:         a tuple containing the same Vector3 position and
                        velocity transformed into the target frame.

        Note that this method has no "derivs" argument because derivatives are
        handled automatically in the velocity component.
        """

        pos = Vector3.as_vector3(pos)
        vel = Vector3.as_vector3(vel)

        # pos_target = matrix * pos_ref
        # vel_target = matrix * (vel_ref - omega x pos_ref)

        pos_target = self.matrix * pos

        velocity_is_easy = self.is_fixed or (pos == Transform.ZERO_VECTOR3)
        if velocity_is_easy:
            vel_target = self.matrix * vel
        else:
            vel_target = self.matrix * (vel - self.omega.cross(pos))

        return (pos_target, vel_target)

    def unrotate(self, pos, derivs=False):
        """Un-rotates the coordinates of a position backward into the reference
        frame.

        Input:
            pos         a Vector3, VectorN or MatrixN object. The size of the
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

        if pos is None: return None
        if pos == Empty(): return pos

        if not isinstance(pos, Array):
            pos = Vector3.as_vector3(pos)

        self.matrix.subfield_math = derivs
        self.omega.subfield_math = derivs

        pos_ref = self.matrix.unrotate(pos)

        # Calculate/update the time-derivative if necessary
        if derivs:
            if self.is_fixed or pos == Transform.ZERO_VECTOR3:
                pos_ref.insert_subfield_if_new("d_dt",
                                    Transform.ZERO_MATRIXN)
            else:
                pos_ref.add_to_subfield("d_dt",
                                    self.dmatrix_dt.unrotate(pos).as_column())

        return pos_ref

    def unrotate_pos_vel(self, pos, vel):
        """Un-rotates the coordinates of a position and velocity backward into
        the reference frame.

        Input:
            pos         position as a Vector3, in the target frame.

            vel         velocity as a Vector3, in the target frame.

        Return:         a tuple containing the same Vector3 position and
                        velocity transformed back into the reference frame.

        Note that this method has no "derivs" argument because derivatives are
        handled automatically in the velocity component.
        """

        pos = Vector3.as_vector3(pos)
        vel = Vector3.as_vector3(vel)

        # pos_ref = matrix-T * pos_target
        # vel_ref = matrix-T * vel_target + omega x pos_ref

        pos_ref = self.matrix.unrotate(pos)

        velocity_is_easy = self.is_fixed or pos == Transform.ZERO_VECTOR3
        if velocity_is_easy:
            vel_ref = self.matrix.unrotate(vel)
        else:
            vel_ref = self.matrix.unrotate(vel) + self.omega.cross(pos_ref)

        return (pos_ref, vel_ref)

################################################################################
# Operations on Transforms
################################################################################

    def invert(self):
        """Returns the inverse transformation."""

        return Transform(self.matrix.invert(), self.omega1,
                         self.reference_id, self.frame_id, self.origin_id)

    @staticmethod
    def null_transform(frame_id, origin_id):
        """Returns a transform that leaves all coordinates unchanged."""

        return Transform([[1,0,0],[0,1,0],[0,0,1]], [0,0,0],
                         frame_id, frame_id, origin_id)

    def rotate_transform(self, arg):
        """Applies this transform to another, as a left-multiply. The result is
        a single transform that converts coordinates in the reference frame of
        the argument transform into the frame of this transform."""

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

        if self.origin_id is None:
            origin_id = arg.origin_id
        elif arg.origin_id is None:
            origin_id = self.origin_id
        else:
            origin_id = self.origin_id
            # assert self.origin_id == arg.origin_id

        return Transform(self.matrix.rotate_matrix3(arg.matrix),
                         arg.matrix.unrotate_vector3(self.omega) + arg.omega,
                         self.frame_id, arg.reference_id, origin_id)

    def unrotate_transform(self, arg):
        """Applies the inverse of one transform to another transform, as a
        left-multiply. The result is a single transform that applies the
        convert coordinates in the parent frame of the argument transform into
        the parent frame of this transform. I.e., if arg rotates A to B and
        self rotates C to B, then the result rotates A to C.
        """

        return self.invert().rotate_transform(arg)

################################################################################
# UNIT TESTS
################################################################################

import unittest

class Test_Transform(unittest.TestCase):

    def runTest(self):

        # Additional imports needed for testing
        from oops_.frame.frame_ import Frame, Wayframe
        from oops_.frame.spinframe import SpinFrame

        # Fake out the FRAME REGISTRY with something that has .shape = []
        registry.FRAME_REGISTRY["TEST"] = Wayframe("J2000")
        registry.FRAME_REGISTRY["SPIN"] = Wayframe("J2000")

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
        self.assertEqual(tr.rotate_pos_vel(p,v)[1].as_scalar(2), v.as_scalar(2))
        self.assertEqual(tr.rotate_pos_vel(p,v)[1].as_scalar(0),
                                                v.as_scalar(0) + p.as_scalar(1))
        self.assertEqual(tr.rotate_pos_vel(p,v)[1].as_scalar(1),
                                                v.as_scalar(1) - p.as_scalar(0))

        tr = tr.invert()

        self.assertEqual(tr.rotate(p), p)
        self.assertEqual(tr.rotate_pos_vel(p,v)[1].as_scalar(2), v.as_scalar(2))
        self.assertEqual(tr.rotate_pos_vel(p,v)[1].as_scalar(0),
                                                v.as_scalar(0) - p.as_scalar(1))
        self.assertEqual(tr.rotate_pos_vel(p,v)[1].as_scalar(1),
                                                v.as_scalar(1) + p.as_scalar(0))

        a = Vector3(np.random.rand(3,1,3))
        b = Vector3(np.random.rand(1,1,3))
        m = Matrix3.twovec(a,0,b,1)
        omega = Vector3(np.random.rand(3,1,3))

        tr = Transform(m, omega, "TEST", "J2000")

        self.assertEqual(tr.unrotate(p), tr.invert().rotate(p))
        self.assertEqual(tr.rotate(p), tr.invert().unrotate(p))

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
