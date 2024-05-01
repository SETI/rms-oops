################################################################################
# tests/test_transform.py
################################################################################

import numpy as np
import unittest

from polymath   import Scalar, Vector3, Matrix3
from oops       import Transform
from oops.frame import Frame, Wayframe


class Test_Transform(unittest.TestCase):

    def runTest(self):

        np.random.seed(5819)

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
