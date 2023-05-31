################################################################################
# Vector3.spin(), Vector3.offset_angles()
################################################################################

import numpy as np
import unittest

from polymath import Qube, Boolean, Scalar, Vector, Vector3, Pair, Units

class Test_Vector3_spin(unittest.TestCase):

  # runTest
    def runTest(self):

        np.random.seed(9431)

        DPR = np.pi / 180.
        X = Vector3((1,0,0))
        Y = Vector3((0,1,0))
        Z = Vector3((0,0,1))

        deg20 = 20 * DPR
        cos20 = np.cos(deg20)
        sin20 = np.sin(deg20)

        deg40 = 40 * DPR
        cos40 = np.cos(deg40)
        sin40 = np.sin(deg40)

        v1 = Vector3(np.random.randn(3))
        v9 = Vector3(np.random.randn(9,3))

        EPS = 5.e-15
        self.assertTrue(np.all(abs((v1.spin(X,0.) - v1).vals) < EPS))
        self.assertTrue(np.all(abs((v9.spin(X,0.) - v9).vals) < EPS))
        self.assertTrue(np.all(abs((v9.spin(X+Y,0.) - v9).vals) < EPS))

        self.assertTrue(np.all(abs((v9.spin(X,np.pi) - v9).vals[:,0]) < EPS))
        self.assertTrue(np.all(abs((v9.spin(X,np.pi) + v9).vals[:,1:]) < EPS))

        angles = np.random.rand(22) * Scalar.PI
        self.assertTrue(np.all(abs(X.spin(X, angles) - X).vals < EPS))

        self.assertTrue(np.all(abs(X.spin(Z,  np.pi/2) - Y).vals < EPS))
        self.assertTrue(np.all(abs(X.spin(Z,  np.pi  ) + X).vals < EPS))
        self.assertTrue(np.all(abs(X.spin(Z, -np.pi/2) + Y).vals < EPS))

        self.assertTrue(np.all(abs(Z.spin(X, deg20) - (0., -sin20, cos20))).vals < EPS)
        self.assertTrue(np.all(abs(Z.spin(Y, deg20) - (sin20,  0., cos20))).vals < EPS)

        # offset_angles()
        self.assertEqual(Z.offset_angles(Z), (0.,0.))

        target = Vector3([0., sin20, cos20])
        self.assertEqual(Z.offset_angles(target), (0., -deg20))
        test = Z.spin(X, -deg20)
        self.assertTrue(np.all(abs(test - target).vals < EPS))

        target = Vector3([sin20, 0., cos20])
        self.assertEqual(Z.offset_angles(target), (deg20, 0.))
        test = Z.spin(Y, deg20)
        self.assertTrue(np.all(abs(test - target).vals < EPS))

        start  = Vector3([0., -sin20, cos20])
        target = Vector3([0.,  sin20, cos20])
        angles = start.offset_angles(target)
        self.assertEqual(angles[0], 0.)
        self.assertAlmostEqual(angles[1], -deg40, 15)

        start  = Vector3([-sin20, 0., cos20])
        target = Vector3([ sin20, 0., cos20])
        angles = start.offset_angles(target)
        self.assertAlmostEqual(angles[0], deg40, 15)
        self.assertEqual(angles[1], 0.)

        start_vals = 0.5 * np.random.randn(1,1,4,3)
        start_vals[...,2] = 1.
        start = Vector3(start_vals).unit()

        target_vals = 0.5 * np.random.randn(1,3,4,3)
        target_vals[...,2] = 1.
        target = Vector3(target_vals).unit()

        (yrot, xrot) = start.offset_angles(target)
        test = start.spin(Y, yrot).spin(X, xrot)
        self.assertTrue(np.all(abs(test - target.unit()).vals < EPS))

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
