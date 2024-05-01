################################################################################
# tests/surface/test_ansa.py
################################################################################

import numpy as np
import unittest

from polymath          import Scalar, Vector3
from oops.frame.frame_ import Frame
from oops.path.path_   import Path
from oops.surface.ansa import Ansa

from oops.constants import PI, HALFPI


class Test_Ansa(unittest.TestCase):

    def runTest(self):

        np.random.seed(7742)

        surface = Ansa('SSB', 'J2000')

        # intercept()
        obs = Vector3( np.random.rand(10,3) * 1.e5)
        los = Vector3(-np.random.rand(10,3))

        (pos,t) = surface.intercept(obs, los)
        pos_xy = pos.element_mul((1,1,0))
        los_xy = los.element_mul((1,1,0))

        self.assertTrue(abs(pos_xy.sep(los_xy) - HALFPI).max() < 1.e-8)
        self.assertTrue(abs(obs + t * los - pos).max() < 1.e-8)

        # coords_from_vector3()
        obs = Vector3(np.random.rand(100,3) * 1.e6)
        pos = Vector3(np.random.rand(100,3) * 1.e5)

        (r,z) = surface.coords_from_vector3(pos, obs, axes=2)

        pos_xy = pos.element_mul(Vector3((1,1,0)))
        pos_z  = pos.to_scalar(2)
        self.assertTrue(abs(pos_xy.norm() - abs(r)).max() < 1.e-8)
        self.assertTrue(abs(pos_z - z).max() < 1.e-8)

        (r,z,theta) = surface.coords_from_vector3(pos, obs, axes=3)

        pos_xy = pos.element_mul(Vector3((1,1,0)))
        pos_z  = pos.to_scalar(2)
        self.assertTrue(abs(pos_xy.norm() - abs(r)).max() < 1.e-8)
        self.assertTrue(abs(pos_z - z).max() < 1.e-8)
        self.assertTrue(abs(theta).max() <= PI)

        # vector3_from_coords()
        obs = Vector3(1.e-5 + np.random.rand(100,3) * 1.e6)
        r = Scalar(1.e-4 + np.random.rand(100) * 9e-4)
        z = Scalar((2 * np.random.rand(100) - 1) * 1.e5)
        theta = Scalar(np.random.rand(100))

        pos = surface.vector3_from_coords((r,z), obs)

        pos_xy = pos.element_mul(Vector3((1,1,0)))
        pos_z  = pos.to_scalar(2)
        self.assertTrue(abs(pos_xy.norm() - abs(r)).max() < 1.e-8)
        self.assertTrue(abs(pos_z - z).max() < 1.e-8)

        obs_xy = obs.element_mul(Vector3((1,1,0)))
        self.assertTrue(abs(pos_xy.sep(obs_xy - pos_xy) - HALFPI).max() < 1.e-5)

        pos1 = surface.vector3_from_coords((r,z,theta), obs)
        pos1_xy = pos1.element_mul(Vector3((1,1,0)))
        self.assertTrue(abs(pos1_xy.sep(pos_xy) - theta).max() < 1.e-5)

        pos1 = surface.vector3_from_coords((r,z,-theta), obs)
        pos1_xy = pos1.element_mul(Vector3((1,1,0)))
        self.assertTrue(abs(pos1_xy.sep(pos_xy) - theta).max() < 1.e-5)

        pos = surface.vector3_from_coords((-r,z), obs)
        pos_xy = pos.element_mul(Vector3((1,1,0)))

        pos1 = surface.vector3_from_coords((-r,z,-theta), obs)
        pos1_xy = pos1.element_mul(Vector3((1,1,0)))
        self.assertTrue(abs(pos1_xy.sep(pos_xy) - theta).max() < 1.e-5)

        pos1 = surface.vector3_from_coords((-r,z,theta), obs)
        pos1_xy = pos1.element_mul(Vector3((1,1,0)))
        self.assertTrue(abs(pos1_xy.sep(pos_xy) - theta).max() < 1.e-5)

        # vector3_from_coords() & coords_from_vector3()
        obs = Vector3((1.e6,0,0))
        r = Scalar(1.e4 + np.random.rand(100) * 9.e4)
        r *= np.sign(2 * np.random.rand(100) - 1)
        z = Scalar((2 * np.random.rand(100) - 1) * 1.e5)
        theta = Scalar((2 * np.random.rand(100) - 1) * 1.)

        pos = surface.vector3_from_coords((r,z,theta), obs)
        coords = surface.coords_from_vector3(pos, obs, axes=3)
        self.assertTrue(abs(r - coords[0]).max() < 1.e-5)
        self.assertTrue(abs(z - coords[1]).max() < 1.e-5)
        self.assertTrue(abs(theta - coords[2]).max() < 1.e-8)

        obs = Vector3(np.random.rand(100,3) * 1.e6)
        pos = Vector3(np.random.rand(100,3) * 1.e5)
        coords = surface.coords_from_vector3(pos, obs, axes=3)
        test_pos = surface.vector3_from_coords(coords, obs)
        self.assertTrue(abs(test_pos - pos).max() < 1.e-5)

        # intercept() derivatives
        obs = Vector3(np.random.rand(10,3))
        obs.insert_deriv('obs', Vector3.IDENTITY)
        los = Vector3(-np.random.rand(10,3))
        los.insert_deriv('los', Vector3.IDENTITY)
        (pos0,t0) = surface.intercept(obs, los, derivs=True)

        eps = 1e-6
        (pos1,t1) = surface.intercept(obs + (eps,0,0), los, derivs=False)
        dpos_dobs_test = (pos1 - pos0) / eps
        dt_dobs_test = (t1 - t0) / eps
        self.assertTrue(abs(dpos_dobs_test - pos0.d_dobs.vals[...,0]).max() < 1.e-6)
        self.assertTrue(abs(dt_dobs_test - t0.d_dobs.vals[...,0]).max() < 1.e-6)

        (pos1,t1) = surface.intercept(obs + (0,eps,0), los, derivs=False)
        dpos_dobs_test = (pos1 - pos0) / eps
        dt_dobs_test = (t1 - t0) / eps
        self.assertTrue(abs(dpos_dobs_test - pos0.d_dobs.vals[...,1]).max() < 1.e-5)
        self.assertTrue(abs(dt_dobs_test - t0.d_dobs.vals[...,1]).max() < 1.e-6)

        (pos1,t1) = surface.intercept(obs + (0,0,eps), los, derivs=False)
        dpos_dobs_test = (pos1 - pos0) / eps
        dt_dobs_test = (t1 - t0) / eps
        self.assertTrue(abs(dpos_dobs_test - pos0.d_dobs.vals[...,2]).max() < 1.e-5)
        self.assertTrue(abs(dt_dobs_test - t0.d_dobs.vals[...,2]).max() < 1.e-6)

        eps = 1e-6
        (pos1,t1) = surface.intercept(obs, los + (eps,0,0), derivs=False)
        dpos_dlos_test = (pos1 - pos0) / eps
        dt_dlos_test = (t1 - t0) / eps
        self.assertTrue(abs(dpos_dlos_test - pos0.d_dlos.vals[...,0]).max() < 1.e-2)
        self.assertTrue(abs(dt_dlos_test - t0.d_dlos.vals[...,0]).max() < 1.e-2)

        (pos1,t1) = surface.intercept(obs, los + (0,eps,0), derivs=False)
        dpos_dlos_test = (pos1 - pos0) / eps
        dt_dlos_test = (t1 - t0) / eps
        self.assertTrue(abs(dpos_dlos_test - pos0.d_dlos.vals[...,1]).max() < 1.e-2)
        self.assertTrue(abs(dt_dlos_test - t0.d_dlos.vals[...,1]).max() < 1.e-2)

        (pos1,t1) = surface.intercept(obs, los + (0,0,eps), derivs=False)
        dpos_dlos_test = (pos1 - pos0) / eps
        dt_dlos_test = (t1 - t0) / eps
        self.assertTrue(abs(dpos_dlos_test - pos0.d_dlos.vals[...,2]).max() < 1.e-2)
        self.assertTrue(abs(dt_dlos_test - t0.d_dlos.vals[...,2]).max() < 1.e-2)

        Path.reset_registry()
        Frame.reset_registry()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
