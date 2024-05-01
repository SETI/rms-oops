################################################################################
# tests/surface/test_ringplane.py
################################################################################

import numpy as np
import unittest

from polymath               import Vector3
from oops.constants         import TWOPI
from oops.frame.frame_      import Frame
from oops.path.path_        import Path
from oops.surface.ringplane import RingPlane


class Test_RingPlane(unittest.TestCase):

    def runTest(self):

        from oops.gravity import Gravity
        from oops.event import Event

        np.random.seed(8829)

        plane = RingPlane(Path.SSB, Frame.J2000)

        # Coordinate/vector conversions
        obs = np.random.rand(2,4,3,3)

        (r,theta,z) = plane.coords_from_vector3(obs,axes=3)
        self.assertTrue((theta >= 0.).all())
        self.assertTrue((theta < TWOPI).all())
        self.assertTrue((r >= 0.).all())

        test = plane.vector3_from_coords((r,theta,z))
        self.assertTrue(np.all(np.abs(test.vals - obs) < 1.e-15))

        # Ring intercepts
        los = np.random.rand(2,4,3,3)
        obs[...,2] =  np.abs(obs[...,2])
        los[...,2] = -np.abs(los[...,2])

        (pts, factors) = plane.intercept(obs, los)
        self.assertTrue(abs(pts.to_scalar(2)).max() < 1.e-15)

        angles = pts - obs
        self.assertTrue((angles.sep(los) > -1.e-12).all())
        self.assertTrue((angles.sep(los) <  1.e-12).all())

        # Intercepts that point away from the ring plane
        self.assertTrue(np.all(factors.vals > 0.))

        ########################################################################
        # Test of radial modes
        ########################################################################

        # Coordinate/vector conversions
        refplane = RingPlane(Path.SSB, Frame.J2000)

        plane = RingPlane(Path.SSB, Frame.J2000,
                          modes=[(10, 1000., 0., 0.)], epoch=0.)

        obs = 10.e3 * np.random.rand(2,4,3,3)

        (a,theta,z) = plane.coords_from_vector3(obs, time=0., axes=3)
        test = plane.vector3_from_coords((a,theta,z), time=0.)
        self.assertTrue(np.all(np.abs(test.vals - obs) < 1.e-11))

        test = plane.vector3_from_coords((a,theta,z), time=1.e8)
        self.assertTrue(np.all(np.abs(test.vals - obs) < 1.e-11))

        plane = RingPlane(Path.SSB, Frame.J2000,
                          modes=[(10, 1000., 0., 2*np.pi/100.)], epoch=0.)

        obs = 10.e3 * np.random.rand(2,4,3,3)

        (a,theta,z) = plane.coords_from_vector3(obs, time=0., axes=3)
        test = plane.vector3_from_coords((a,theta,z), time=0.)
        self.assertTrue(np.all(np.abs(test.vals - obs) < 1.e-11))

        test = plane.vector3_from_coords((a,theta,z), time=100.)
        self.assertTrue(np.all(np.abs(test.vals - obs) < 1.e-11))

        # longitudes are the same in both maps
        (a0,theta0,z0) = refplane.coords_from_vector3(obs, time=0., axes=3)
        self.assertEqual(theta0, theta)

        # radial offsets are out of phase when time=50.
        diff1 = a - a0
        (a,theta,z) = plane.coords_from_vector3(obs, time=50., axes=3)
        diff2 = a - a0
        self.assertTrue(abs(diff1 + diff2).max() < 1.e-11)

        ########################################################################
        # Test of velocities
        ########################################################################

        pos = 10.e3 * np.random.rand(200,3)
        pos[...,2] = 0.     # set Z-coordinate to zero
        pos = Vector3(pos)

        # No gravity, no modes
        refplane = RingPlane(Path.SSB, Frame.J2000)

        vels = refplane.velocity(obs)
        self.assertEqual(vels, (0.,0.,0.))

        # No gravity, motionless mode
        plane = RingPlane(Path.SSB, Frame.J2000,
                          modes=[(10, 1000., 0., 0.)], epoch=0.)

        vels = plane.velocity(obs)
        self.assertEqual(vels, (0.,0.,0.))

        # No gravity, modes (10 cycles, 100 km amplitude, period = 10,000 s)
        plane = RingPlane(Path.SSB, Frame.J2000,
                          modes=[(10, 100., 0., 2.*np.pi/1.e4)], epoch=0.)

        TIME = 0.
        (a0,theta0) = plane.coords_from_vector3(pos, time=TIME - 0.5)
        (a ,theta ) = plane.coords_from_vector3(pos, time=TIME)
        (a1,theta1) = plane.coords_from_vector3(pos, time=TIME + 0.5)
        self.assertEqual(theta, theta0)
        self.assertEqual(theta, theta1)

        vels = plane.velocity(pos, time=TIME)
        v_angular = vels.perp(pos)
        v_radial = vels - v_angular

        sign = v_radial.dot(pos).sign()
        speed2 = sign * v_radial.norm()
        speed1 = a0 - a1
        self.assertTrue(abs(speed1 - speed2).max() < 2.e-9)

        # Gravity, no modes
        plane = RingPlane(Path.SSB, Frame.J2000, gravity=Gravity.SATURN)

        (a, theta) = plane.coords_from_vector3(pos)

        vels = plane.velocity(pos)
        sep = vels.sep(pos)
        self.assertTrue(abs(sep - np.pi/2.).max() < 1.e-14)

        speed1 = vels.norm()
        rate = np.minimum(Gravity.SATURN.n(a.vals), plane.max_rate)
        diff = (a * rate - speed1) / speed1
        self.assertTrue(abs(diff).max() < 1.e-15)

        ########################################################################
        # coords_of_event, event_from_coords
        ########################################################################

        plane = RingPlane(Path.SSB, Frame.J2000)

        pos = Vector3(np.random.rand(2,4,3,3))
        vel = Vector3(np.random.rand(2,4,3,3))
        pos.insert_deriv('t', vel)

        event = Event(0., pos, Path.SSB, Frame.J2000)
        coords = plane.coords_of_event(event)
        test = plane.event_at_coords(0., coords)

        self.assertTrue(np.all(np.abs(test.pos.vals - pos.vals) < 1.e-15))
        self.assertTrue(np.all(np.abs(test.vel.vals - vel.vals) < 1.e-15))

        ########################################################################
        # Note: Additional unit testing is performed in orbitplane.py
        ########################################################################

        Path.reset_registry()
        Frame.reset_registry()

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
