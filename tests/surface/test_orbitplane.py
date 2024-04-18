################################################################################
# tests/surface/test_orbitplane.py
################################################################################

import numpy as np
import unittest

from polymath                import Scalar, Vector3
from oops.constants          import PI, HALFPI, TWOPI, RPD
from oops.event              import Event
from oops.surface.orbitplane import OrbitPlane


class Test_OrbitPlane(unittest.TestCase):

    def runTest(self):

        # elements = (a, lon, n)

        # Circular orbit, no derivatives, forward
        elements = (1, 0, 1)
        epoch = 0
        orbit = OrbitPlane(elements, epoch, 'SSB', 'J2000', 'TEST')

        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])
        (r,l,z) = orbit.coords_from_vector3(pos, None, axes=3, derivs=False)

        r_true = Scalar([1,2,1,1])
        l_true = Scalar([0, 0, PI, HALFPI])
        z_true = Scalar([0,0,0,0.1])

        self.assertTrue(abs(r - r_true).max() < 1.e-12)
        self.assertTrue(abs(l - l_true).max() < 1.e-12)
        self.assertTrue(abs(z - z_true).max() < 1.e-12)

        # Circular orbit, no derivatives, reverse
        pos2 = orbit.vector3_from_coords((r, l, z), None, derivs=False)

        self.assertTrue((pos - pos2).norm().max() < 1.e-10)

        # Circular orbit, with derivatives, forward
        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])
        pos.insert_deriv('pos', Vector3.IDENTITY, override=True)
        eps = 1.e-6
        delta = 1.e-4

        for step in ([eps,0,0], [0,eps,0], [0,0,eps]):
            dpos = Vector3(step)
            (r,l,z) = orbit.coords_from_vector3(pos + dpos, None, axes=3,
                                                derivs=True)

            r_test = r + r.d_dpos.chain(dpos)
            l_test = l + l.d_dpos.chain(dpos)
            z_test = z + z.d_dpos.chain(dpos)

            self.assertTrue(abs(r - r_test).max() < delta)
            self.assertTrue(abs(l - l_test).max() < delta)
            self.assertTrue(abs(z - z_test).max() < delta)

        # Circular orbit, with derivatives, reverse
        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])
        (r,l,z) = orbit.coords_from_vector3(pos, None, axes=3, derivs=False)
        eps = 1.e-6
        delta = 1.e-5

        r.insert_deriv('r', Scalar.ONE, override=True)
        l.insert_deriv('l', Scalar.ONE, override=True)
        z.insert_deriv('z', Scalar.ONE, override=True)
        pos0 = orbit.vector3_from_coords((r, l, z), None, derivs=True)

        pos1 = orbit.vector3_from_coords((r + eps, l, z), None, derivs=False)
        pos1_test = pos0 + eps * pos0.d_dr
        self.assertTrue((pos1_test - pos1).norm().max() < delta)

        pos1 = orbit.vector3_from_coords((r, l + eps, z), None, derivs=False)
        pos1_test = pos0 + eps * pos0.d_dl
        self.assertTrue((pos1_test - pos1).norm().max() < delta)

        pos1 = orbit.vector3_from_coords((r, l, z + eps), None, derivs=False)
        pos1_test = pos0 + eps * pos0.d_dz
        self.assertTrue((pos1_test - pos1).norm().max() < delta)

        # elements = (a, lon, n, e, peri, prec)

        # Eccentric orbit, no derivatives, forward
        ae = 0.1
        prec = 0.1
        elements = (1, 0, 1, ae, 0, prec)
        epoch = 0
        orbit = OrbitPlane(elements, epoch, 'SSB', 'J2000', 'TEST')
        eps = 1.e-6
        delta = 1.e-5

        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])
        event = Event(0., pos, 'SSB', 'J2000')
        (r,l,z) = orbit.coords_of_event(event, derivs=False)

        r_true = Scalar([1. + ae, 2. + ae, 1 - ae, np.sqrt(1. + ae**2)])
        l_true = Scalar([TWOPI, TWOPI, PI, np.arctan2(1,ae)])
        z_true = Scalar([0,0,0,0.1])

        self.assertTrue(abs(r - r_true).max() < delta)
        self.assertTrue(abs(l - l_true).max() < delta)
        self.assertTrue(abs(z - z_true).max() < delta)

        # Eccentric orbit, no derivatives, reverse
        event2 = orbit.event_at_coords(event.time, (r,l,z)).wrt_ssb()
        self.assertTrue((pos - event2.pos).norm().max() < 1.e-10)
        self.assertTrue((event2.vel).norm().max() < 1.e-10)

        # Eccentric orbit, with derivatives, forward
        ae = 0.1
        prec = 0.1
        elements = (1, 0, 1, ae, 0, prec)
        epoch = 0
        orbit = OrbitPlane(elements, epoch, 'SSB', 'J2000')
        eps = 1.e-6
        delta = 3.e-5

        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])

        for v in ([0,0,0], [0.1,0,0], [0,0.1,0], [0,0,0.1]):
            vel = Vector3(v)
            event = Event(0., (pos, vel), 'SSB', 'J2000')
            (r,l,z) = orbit.coords_of_event(event, derivs=True)

            event = Event(eps, (pos + vel*eps, vel), 'SSB', 'J2000')
            (r1,l1,z1) = orbit.coords_of_event(event, derivs=False)
            dr_dt_test = (r1 - r) / eps
            dl_dt_test = (l1 - l) / eps
            dz_dt_test = (z1 - z) / eps

            self.assertTrue(abs(r.d_dt - dr_dt_test).max() < delta)
            self.assertTrue(abs(z.d_dt - dz_dt_test).max() < delta)

            d_dl_dt = ((l.d_dt*eps - dl_dt_test*eps + PI) % TWOPI - PI) / eps
            self.assertTrue(abs(d_dl_dt).max() < delta)

        # Eccentric orbit, with derivatives, reverse
        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])
        (r,l,z) = orbit.coords_from_vector3(pos, axes=3, derivs=False)
        eps = 1.e-6
        delta = 1.e-5

        r.insert_deriv('r', Scalar.ONE)
        l.insert_deriv('l', Scalar.ONE)
        z.insert_deriv('z', Scalar.ONE)
        pos0 = orbit.vector3_from_coords((r, l, z), derivs=True)

        pos1 = orbit.vector3_from_coords((r + eps, l, z), derivs=False)
        pos1_test = pos0 + eps * pos0.d_dr
        self.assertTrue((pos1_test - pos1).norm().max() < delta)

        pos1 = orbit.vector3_from_coords((r, l + eps, z), derivs=False)
        pos1_test = pos0 + eps * pos0.d_dl
        self.assertTrue((pos1_test - pos1).norm().max() < delta)

        pos1 = orbit.vector3_from_coords((r, l, z + eps), derivs=False)
        pos1_test = pos0 + eps * pos0.d_dz
        self.assertTrue((pos1_test - pos1).norm().max() < delta)

        # elements = (a, lon, n, e, peri, prec, i, node, regr)

        # Inclined orbit, no eccentricity, no derivatives, forward
        inc = 0.1
        regr = -0.1
        node = -HALFPI
        sini = np.sin(inc)
        cosi = np.cos(inc)

        elements = (1, 0, 1, 0, 0, 0, inc, node, regr)
        epoch = 0
        orbit = OrbitPlane(elements, epoch, 'SSB', 'J2000')
        eps = 1.e-6
        delta = 1.e-5

        dz = 0.1
        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,dz)])
        event = Event(0., pos, 'SSB', 'J2000')
        (r,l,z) = orbit.coords_of_event(event, derivs=False)

        r_true = Scalar([cosi, 2*cosi, cosi, np.sqrt(1 + (dz*sini)**2)])
        l_true = Scalar([TWOPI, TWOPI, PI, np.arctan2(1,dz*sini)])
        z_true = Scalar([-sini, -2*sini, sini, dz*cosi])

        self.assertTrue(abs(r - r_true).max() < delta)
        self.assertTrue(abs(l - l_true).max() < delta)
        self.assertTrue(abs(z - z_true).max() < delta)

        # Inclined orbit, no derivatives, reverse
        event2 = orbit.event_at_coords(event.time, (r,l,z)).wrt_ssb()
        self.assertTrue((pos - event2.pos).norm().max() < 1.e-10)
        self.assertTrue(event2.vel.norm().max() < 1.e-10)

        # Inclined orbit, with derivatives, forward
        inc = 0.1
        regr = -0.1
        node = -HALFPI
        sini = np.sin(inc)
        cosi = np.cos(inc)

        elements = (1, 0, 1, 0, 0, 0, inc, node, regr)
        epoch = 0
        orbit = OrbitPlane(elements, epoch, 'SSB', 'J2000')
        eps = 1.e-6
        delta = 1.e-5

        dz = 0.1
        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,dz)])

        for v in ([0,0,0], [0.1,0,0], [0,0.1,0], [0,0,0.1]):
            vel = Vector3(v)
            event = Event(0., (pos, vel), 'SSB', 'J2000')
            (r,l,z) = orbit.coords_of_event(event, derivs=True)

            event = Event(eps, (pos + vel*eps, vel), 'SSB', 'J2000')
            (r1,l1,z1) = orbit.coords_of_event(event, derivs=False)
            dr_dt_test = (r1 - r) / eps
            dl_dt_test = ((l1 - l + PI) % TWOPI - PI) / eps
            dz_dt_test = (z1 - z) / eps

            self.assertTrue(abs(r.d_dt - dr_dt_test).max() < delta)
            self.assertTrue(abs(l.d_dt - dl_dt_test).max() < delta)
            self.assertTrue(abs(z.d_dt - dz_dt_test).max() < delta)

        # Inclined orbit, with derivatives, reverse
        pos = Vector3([(1,0,0), (2,0,0), (-1,0,0), (0,1,0.1)])
        (r,l,z) = orbit.coords_from_vector3(pos, axes=3, derivs=False)
        eps = 1.e-6
        delta = 1.e-5

        r.insert_deriv('r', Scalar.ONE)
        l.insert_deriv('l', Scalar.ONE)
        z.insert_deriv('z', Scalar.ONE)
        pos0 = orbit.vector3_from_coords((r, l, z), derivs=True)

        pos1 = orbit.vector3_from_coords((r + eps, l, z), derivs=False)
        pos1_test = pos0 + eps * pos0.d_dr
        self.assertTrue((pos1_test - pos1).norm().max() < delta)

        pos1 = orbit.vector3_from_coords((r, l + eps, z), derivs=False)
        pos1_test = pos0 + eps * pos0.d_dl
        self.assertTrue((pos1_test - pos1).norm().max() < delta)

        pos1 = orbit.vector3_from_coords((r, l, z + eps), derivs=False)
        pos1_test = pos0 + eps * pos0.d_dz
        self.assertTrue((pos1_test - pos1).norm().max() < delta)

        # From/to mean anomaly
        elements = (1, 0, 1, 0.1, 0, 0.1)
        epoch = 0
        orbit = OrbitPlane(elements, epoch, 'SSB', 'J2000', 'TEST')

        l = np.arange(361) * RPD
        anoms = orbit.to_mean_anomaly(l)

        lons = orbit.from_mean_anomaly(anoms)
        self.assertTrue(abs(lons - l).max() < 1.e-15)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
