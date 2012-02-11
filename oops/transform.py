import numpy as np
import unittest

import oops

################################################################################
# Transform
################################################################################

class Transform(object):
    """An object describing a coordinate transformation, defined by a rotation
    matrix plus an optional angular rotation vector indicating how that frame is
    rotating. The components are interpreted as follows:
        matrix          rotates coordinates from the reference coordinate frame
                        into the target frame.
        omega           the angular rotation vector of the target frame relative
                        to the reference, and specified in the reference
                        coordinate frame. Default is a zero vector.

    Given a state vector (pos,vel) in the reference coordinate frame, we can
    convert to a state vector in the target frame as follows:
        pos_target = matrix *  pos_ref
        vel_target = matrix * (vel_ref - cross(omega, pos_ref))

    The inverse transformation is:
        pos_ref = matrix-T *  pos_target
        vel_ref = matrix-T *  vel_target + cross(omega_ref, pos_ref))
        vel_ref = matrix-T * (vel_target + cross(omega_target, pos_target))

    With this definition, a transform can also be used to describe the
    orientation and rotation rate of a planetary body:
        matrix[0,:] = the instantaneous X-axis in reference coordinates
        matrix[1,:] = the instantaneous Y-axis in reference coordinates
        matrix[2,:] = the instantaneous Z-axis in reference coordinates
        omega       = the body's rotation vector in reference coordinates.

    A Tranform object also has these attributes that describe the transform it
    is performing:
        omega1          the omega rotation vector transformed into the target
                        frame.
        frame_id        the registered ID of the destination frame.
        reference_id    the registered ID of the reference frame. The transform
                        rotates coordinates in the reference frame into
                        coordinates in the destination frame. The reference
                        frame must have shape [].
        shape           the intrinsic shape of the transform.
    """

    OOPS_CLASS = "Transform"

    def __init__(self, matrix, omega, frame_id, reference_id, omega1=None):
        """Constructor for a Transform object.

        Input:
            matrix      the Matrix3 object that is used to rotate coordinates
                        from the reference frame into the new frame.
            omega       the spin vector for the coordinate frame, given in
                        coordinates of the reference frame.
            frame       the Frame object from which this Transform was
                        generated.
            omega1      optional Vector3 containing matrix.rotate(omega).
        """

        self.matrix = oops.Matrix3.as_matrix3(matrix)
        self.omega  = oops.Vector3.as_vector3(omega)
                                                # spin vector in old frame

        if omega1 is None:                      # spin vector in the new frame
            self.omega1 = self.matrix.rotate(omega)
        else:
            self.omega1 = omega1

        self.frame_id     = frame_id
        self.reference_id = reference_id

        self.shape = oops.Array.broadcast_shape((self.matrix, self.omega,
                                    oops.as_frame(self.frame_id),
                                    oops.as_frame(self.reference_id)))

############################################
# Shape operations
############################################

    def rotate_axes(self, axes):
        """Rotates the axes of a transform object so that the specified axis
        number comes first. This handles the case where the Array components may
        be of different shapes. The same axes will still broadcast together
        properly after the operation."""

        if axes == 0: return self

        rank = len(self.shape)
        transform = Transform(self.matrix.prepend_rotate_strip(axes,rank),
                              self.omega.prepend_rotate_strip(axes,rank),
                              self.frame_id, self.reference_id,
                              self.omega1.prepend_rotate_strip(axes,rank))

        if self.lt is not None:
            result.lt = self.lt.prepend_rotate_strip(axes,rank)

        if self.perp is not None:
            result.perp = self.perp.prepend_rotate_strip(axes,rank)

        if self.arr is not None:
            result.arr = self.arr.prepend_rotate_strip(axes,rank)

        if self.dep is not None:
            result.dep = self.dep.prepend_rotate_strip(axes,rank)

        if self.vlocal != oops.Vector3((0.,0.,0.)):
            result.vlocal = self.vlocal.prepend_rotate_strip(axes,rank)

        return Event

################################################################################
# Arithmetic operations
################################################################################

############################
# Operations on Transforms
############################

    def invert(self):
        """Returns the inverse transformation."""

        return Transform(self.matrix.invert(), -self.omega1,
                         self.reference_id, self.frame_id, omega1=-self.omega)

    @staticmethod
    def null_transform(frame_id):
        """Returns a transform that leaves all coordinates unchanged."""

        return Transform([[1,0,0],[0,1,0],[0,0,1]], [0,0,0], frame_id, frame_id)

    def rotate_transform(self, arg):
        """Applies this transform to another, as a left-multiply. The result is
        a single transform that converts coordinates in the reference frame of
        the argument transform into the frame of this transform."""

        # Two tranforms
        #   P1 = M0 P0; V1 = M0(V0 - omega0 x P0)
        #   P2 = M1 P1; V2 = M1(V1 - omega1 x P1)
        #
        # Combine...
        #   P2 = [M1 M0] P0
        #
        #   V2 = M1 [M0(V0 - omega0 x P0) - omega1 x M0 P0]
        #      = M1 M0 (V0 - omega0 x P0) - M1 M0 M0T (omega1 x M0 P0)
        #      = [M1 M0] (V0 - [M0T omega1 + omega0] x P0)

        return Transform(self.matrix * arg.matrix,
                         arg.matrix.unrotate(self.omega) + arg.omega,
                         self.frame_id,
                         arg.reference_id)

    def unrotate_transform(self, arg):
        """Applies the inverse of one transform to another transform, as a
        left-multiply. The result is a single transform that applies the
        convert coordinates in the parent frame of the argument transform into
        the parent frame of this transform. I.e., if arg rotates A to B and
        self rotates C to B, then the result rotates A to C.
        """

        return self.invert().rotate_transform(arg)

############################
# Operations on vectors
############################

    def rotate(self, pos):
        """Rotates the coordinates of a position forward into the new frame.

        Input:
            pos         position as a Vector3, in the reference frame. A value
                        of None returns None.

        Return:         the same positions in the destination frame.
        """

        if pos is None: return None
        return self.matrix.rotate(pos)

    def rotate_pos(self, pos):
        """Rotates the coordinates of a position forward into the new frame.

        Input:
            pos         position as a Vector3, in the reference frame.

        Return:         the same positions in the destination frame.
        """

        return self.rotate(pos)

    def rotate_vel(self, vel, pos, omega_x_pos=None):
        """Rotates the coordinates of a velocity forward into the new frame,
        also allowing for the artificial component of the velocity relative to
        the frame's rotation.

        Input:
            vel         velocity as a Vector3, in the reference frame.
            pos         position as a Vector3, in the reference frame.
            omega_x_pos optional cross product of omega and position, used to
                        speed up multiple calculations using the same position
                        vectors. This cross product must be defined within the
                        reference coordinate frame.

        Return:         the velocities relative to the target frame.
        """

        if omega_x_pos is None:
            omega_x_pos = self.omega.cross(pos)

        return self.matrix.rotate(vel - omega_x_pos)

    def unrotate(self, pos):
        """Un-rotates the coordinates of a position backward into the reference
        frame.

        Input:
            pos         position as a Vector3, in the destination frame. A value
                        of None returns None.

        Return:         the same positions in the reference frame.
        """

        if pos is None: return None
        return self.matrix.unrotate(pos)

    def unrotate_pos(self, pos):
        """Un-rotates the coordinates of a position backward into the reference
        frame.

        Input:
            pos         position as a Vector3, in the destination frame.

        Return:         the same positions in the reference frame.
        """

        return self.unrotate(pos)

    def unrotate_vel(self, vel, pos, omega1_x_pos=None):
        """Un-rotates the coordinates of a position and velocity backward into
        the reference frame.

        Input:
            vel         velocity as a Vector3, in the destination frame.
            pos         position as a Vector3, in the destination frame.
            omega1_x_pos
                        optional cross product of omega and position, used to
                        speed up multiple calculations using the same position
                        vectors. This cross product must be defined within the
                        destination coordinate frame.
        """

        if omega1_x_pos is None:
            omega1_x_pos = self.omega1.cross(pos)

        return self.matrix.unrotate(vel + omega1_x_pos)

############################
# Arithmetic operators
############################

    # Unary invert operator "~"
    def __invert__(self, arg):
        return self.invert()

    # Binary multiply operator "*"
    def __mul__(self, arg):

        if arg.OOPS_Class == "Transform":
            return self.rotate_transform(arg)

        elif arg.OOPS_Class == "Event":
            return self.rotate_event(arg)

        oops.raise_type_mismatch(self, "*", arg)

    # Binary divide operator "/"
    def __div__(self, arg):

        if arg.OOPS_Class == "Transform":
            return self.rotate_transform(arg.invert())

        oops.raise_type_mismatch(self, "*", arg)

    # string operations
    def __str__(self):
        return ("Transform(shape=" +
                repr(self.shape).replace(' ', '') + "/" +
                repr(self.frame_id) + ")")

    def __repr__(self): return self.__str__()

################################################################################
# UNIT TESTS
################################################################################

class Test_Transform(unittest.TestCase):

    def runTest(self):

        # Fake out the FRAME REGISTRY with something that has .shape = []
        oops.FRAME_REGISTRY["TEST"] = oops.Scalar(0.)
        oops.FRAME_REGISTRY["SPIN"] = oops.Scalar(0.)

        tr = Transform(oops.Matrix3(np.array([[1.,0.,0.],
                                              [0.,1.,0.],[0.,0.,1.]])),
                       oops.Vector3(np.array([0.,0.,0.])), "J2000", "J2000")

        p = oops.Vector3(np.random.rand(2,1,4,3))
        v = oops.Vector3(np.random.rand(  3,4,3))

        self.assertEqual(tr.rotate_pos(p),   p)
        self.assertEqual(tr.rotate_vel(v,p), v)

        self.assertEqual(tr.unrotate_pos(p),   p)
        self.assertEqual(tr.unrotate_vel(v,p), v)

        tr = tr.invert()

        self.assertEqual(tr.rotate_pos(p),   p)
        self.assertEqual(tr.rotate_vel(v,p), v)

        self.assertEqual(tr.unrotate_pos(p),   p)
        self.assertEqual(tr.unrotate_vel(v,p), v)

        tr = Transform(oops.Matrix3([[1,0,0],[0,1,0],[0,0,1]]),
                       oops.Vector3([0,0,1]), "SPIN", "J2000")

        self.assertEqual(tr.rotate_pos(p), p)
        self.assertEqual(tr.rotate_vel(v,p).as_scalar(2), v.as_scalar(2))
        self.assertEqual(tr.rotate_vel(v,p).as_scalar(0), v.as_scalar(0) +
                                                          p.as_scalar(1))
        self.assertEqual(tr.rotate_vel(v,p).as_scalar(1), v.as_scalar(1) -
                                                          p.as_scalar(0))

        tr = tr.invert()

        self.assertEqual(tr.rotate_pos(p), p)
        self.assertEqual(tr.rotate_vel(v,p).as_scalar(2), v.as_scalar(2))
        self.assertEqual(tr.rotate_vel(v,p).as_scalar(0), v.as_scalar(0) -
                                                          p.as_scalar(1))
        self.assertEqual(tr.rotate_vel(v,p).as_scalar(1), v.as_scalar(1) +
                                                          p.as_scalar(0))

        a = oops.Vector3(np.random.rand(3,1,3))
        b = oops.Vector3(np.random.rand(1,1,3))
        m = oops.Matrix3.twovec(a,0,b,1)
        omega = oops.Vector3(np.random.rand(3,1,3))

        tr = Transform(m, omega, "TEST", "J2000")

        self.assertEqual(tr.unrotate_pos(p), tr.invert().rotate_pos(p))
        self.assertEqual(tr.rotate_pos(p), tr.invert().unrotate_pos(p))

        eps = 1.e-15
        diff = tr.unrotate_vel(v,p) - tr.invert().rotate_vel(v,p)
        self.assertTrue(np.all(diff.vals > -eps))
        self.assertTrue(np.all(diff.vals <  eps))

        diff = tr.rotate_vel(v,p) - tr.invert().unrotate_vel(v,p)
        self.assertTrue(np.all(diff.vals > -eps))
        self.assertTrue(np.all(diff.vals <  eps))

        self.assertEqual(tr.rotate_vel(v,p),
                         tr.rotate_vel(v,p,tr.omega.cross(p)))

        self.assertEqual(tr.unrotate_vel(v,p),
                         tr.unrotate_vel(v,p,tr.omega1.cross(p)))

################################################################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
