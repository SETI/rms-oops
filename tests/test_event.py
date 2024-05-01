################################################################################
# tests/test_event.py
################################################################################

import numpy as np
import unittest

from polymath       import Vector3
from oops           import Event
from oops.constants import C, RPD


class Test_Event(unittest.TestCase):

    def setUp(self):
        from oops.body import Body
        Body.reset_registry()
        Body.define_solar_system('1990-01-01', '2010-01-01')

    def tearDown(self):
        pass

    def runTest(self):
        import cspyce

        np.random.seed(5531)

        # This is the exact formula for stellar aberration
        #   beta = v/c
        #   angle is measured from the direction of motion to the actual (not
        #       time-reversed) direction of the incoming ray.
        def aberrate(angle, beta):
            tan_half_angle_prime = np.sqrt((1.+beta) /
                                           (1.-beta)) * np.tan(angle/2.)
            return 2. * np.arctan(tan_half_angle_prime)

        def unaberrate(angle_prime, beta):
            tan_half_angle = np.sqrt((1.+beta) /
                                     (1.-beta)) * np.tan(angle_prime/2.)
            return 2. * np.arctan(tan_half_angle)

        # Test against the approximation sin(delta) = beta * sin(angle)
        # where angle_prime = angle + delta
        BETA = 0.001
        angles = np.arange(181.) * RPD
        exact_prime = aberrate(angles, BETA)
        delta = exact_prime - angles
        for k in range(181):
            self.assertTrue(abs(np.sin(delta[k]) - BETA * np.sin(angles[k])) <
                            1.e-6)

        ########################################################################
        # Test aberration magnitudes and directions to first order
        ########################################################################

        BETA = 0.001
        DEL = 3.e-9
        SPEED = BETA * C        # largest speed we care about is 300 km/s
        HALFPI = np.pi/2

        # Incoming aberration in the forward direction
        ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), 'SSB', 'J2000')
        ev.arr = -Vector3.ZAXIS
        self.assertEqual(Vector3.ZAXIS.sep(ev.neg_arr_ap), 0.)

        # Incoming aberration in the side direction
        ev = Event(0., (Vector3.ZERO, SPEED * Vector3.XAXIS), 'SSB', 'J2000')
        ev.arr = -Vector3.YAXIS
        self.assertTrue(abs(Vector3.XAXIS.sep(ev.neg_arr_ap) - (HALFPI-BETA)) < DEL)

        # Outgoing aberration in the forward direction
        ev = Event(0., (Vector3.ZERO, SPEED * Vector3.XAXIS), 'SSB', 'J2000')
        ev.dep = Vector3.XAXIS
        self.assertEqual(Vector3.XAXIS.sep(ev.dep_ap), 0.)

        # Incoming aberration in the side direction
        ev = Event(0., (Vector3.ZERO, SPEED * Vector3.XAXIS), 'SSB', 'J2000')
        ev.dep = Vector3.YAXIS
        self.assertTrue(abs(Vector3.XAXIS.sep(ev.dep_ap) - (HALFPI+BETA)) < DEL)

        # Incoming aberration in the forward direction
        ev = Event(0., (Vector3.ZERO, SPEED * Vector3.XAXIS), 'SSB', 'J2000')
        ev.arr_ap = -Vector3.XAXIS
        self.assertEqual(Vector3.XAXIS.sep(ev.neg_arr_ap), 0.)

        # Incoming aberration in the side direction
        ev = Event(0., (Vector3.ZERO, SPEED * Vector3.XAXIS), 'SSB', 'J2000')
        ev.arr_ap = -Vector3.YAXIS
        self.assertTrue(abs(Vector3.XAXIS.sep(ev.neg_arr) - (HALFPI+BETA)) < DEL)

        # Outgoing aberration in the forward direction
        ev = Event(0., (Vector3.ZERO, SPEED * Vector3.XAXIS), 'SSB', 'J2000')
        ev.dep = Vector3.XAXIS
        self.assertEqual(Vector3.XAXIS.sep(ev.dep_ap), 0.)

        # Incoming aberration in the side direction
        ev = Event(0., (Vector3.ZERO, SPEED * Vector3.XAXIS), 'SSB', 'J2000')
        ev.dep_ap = Vector3.YAXIS
        self.assertTrue(abs(Vector3.XAXIS.sep(ev.dep) - (HALFPI-BETA)) < DEL)

        ########################################################################
        # Test compatibility with SPICE toolkit and with the exact calculation
        ########################################################################

        angles = np.arange(181.)
        cspyce_arr_ap = []
        cspyce_dep_ap = []
        for angle in angles:
            vobs = np.array([SPEED, 0., 0.])

            # Note the sign change on pobj, because we consider the photon's
            # direction, not the direction to the target
            pobj = np.array([-np.cos(angle * RPD),
                             -np.sin(angle * RPD), 0.])
            appobj = cspyce.stelab(pobj, vobs)
            cspyce_arr_ap.append(np.arctan2(-appobj[1], -appobj[0]))

            pobj = np.array([np.cos(angle * RPD),
                             np.sin(angle * RPD), 0.])
            appobj = cspyce.stlabx(pobj, vobs)
            cspyce_dep_ap.append(np.arctan2(appobj[1], appobj[0]))

        ev = Event(0., (Vector3.ZERO, SPEED * Vector3.XAXIS), 'SSB', 'J2000')
        ray = Vector3.from_scalars(np.cos(angles * RPD),
                                   np.sin(angles * RPD), 0.)
        ev.arr = ray
        ev.dep = ray

        exact_arr_ap = aberrate(angles * RPD, BETA)
        exact_dep_ap = aberrate(angles * RPD, BETA)

        for k in range(181):
            arr_ap = np.arctan2(ev.arr_ap[k].vals[1], ev.arr_ap[k].vals[0])
            self.assertTrue(abs(cspyce_arr_ap[k] - exact_arr_ap[k]) < 1.e-6)
            self.assertTrue(abs(arr_ap - exact_arr_ap[k]) < 1.e-15)

        for k in range(181):
            dep_ap = np.arctan2(ev.dep_ap[k].vals[1], ev.dep_ap[k].vals[0])
            self.assertTrue(abs(cspyce_dep_ap[k] - exact_dep_ap[k]) < 1.e-6)
            self.assertTrue(abs(dep_ap - exact_dep_ap[k]) < 1.e-15)

        ########################################################################
        # Test aberration inversions
        ########################################################################

        COUNT = 2000
        ev1 = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), 'SSB', 'J2000')
        ev1.arr_ap = Vector3.from_scalars(np.random.randn(COUNT),
                                          np.random.randn(COUNT),
                                          np.random.randn(COUNT))
        ev1.dep_ap = Vector3.from_scalars(np.random.randn(COUNT),
                                          np.random.randn(COUNT),
                                          np.random.randn(COUNT))

        ev2 = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), 'SSB', 'J2000')
        ev2.arr = ev1.arr
        ev2.dep = ev1.dep

        self.assertTrue((ev2.arr_ap.unit() -
                         ev1.arr_ap.unit()).norm().max() < 1.e-15)

        ev1 = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), 'SSB', 'J2000')
        ev1.arr = Vector3.from_scalars(np.random.randn(COUNT),
                                       np.random.randn(COUNT),
                                       np.random.randn(COUNT))
        ev1.dep = Vector3.from_scalars(np.random.randn(COUNT),
                                       np.random.randn(COUNT),
                                       np.random.randn(COUNT))

        ev2 = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), 'SSB', 'J2000')
        ev2.arr_ap = ev1.arr_ap
        ev2.dep_ap = ev1.dep_ap

        self.assertTrue((ev2.arr_ap.unit() -
                         ev1.arr_ap.unit()).norm().max() < 1.e-15)

        ########################################################################
        # Subfield checks
        ########################################################################

        for (origin, frame) in [('SSB', 'J2000'),
                                ('EARTH', 'IAU_EARTH'),
                                ('PLUTO', 'IAU_EARTH')]:

            ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
            self.assertIsNone(ev._arr_)
            self.assertIsNone(ev._arr_ap_)
            self.assertIsNone(ev._neg_arr_)
            self.assertIsNone(ev._neg_arr_ap_)
            self.assertIsNone(ev._ssb_)

            ########################
            # Define arr
            ########################

            ev.arr = (1,2,3)
            self.assertEqual(ev._arr_, Vector3((1.,2.,3.)))
            self.assertIsNone(ev._arr_ap_)
            self.assertIsNone(ev._neg_arr_)
            self.assertIsNone(ev._neg_arr_ap_)
            self.assertIsNone(ev._ssb_)

            self.assertIsNone(ev._neg_arr_)
            self.assertEqual(ev.neg_arr, Vector3((-1.,-2.,-3.)))
            self.assertIs(ev.neg_arr, ev._neg_arr_)
            self.assertIsNone(ev._neg_arr_ap_)
            self.assertIsNone(ev._ssb_)

            try:
                ev.arr_ap = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            self.assertIsNone(ev._arr_ap_)
            self.assertIsNone(ev._neg_arr_ap_)
            self.assertIsNone(ev._ssb_)

            try:
                ev.neg_arr_ap = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            self.assertIsNone(ev._arr_ap_)
            self.assertIsNone(ev._neg_arr_ap_)
            self.assertIsNone(ev._ssb_)

            # Let arr_ap and ssb be filled in
            _ = ev.arr_ap
            self.assertTrue((ev.arr_ap - ev.arr).norm() < 5*BETA)
            self.assertEqual(ev.neg_arr, -ev.arr)
            self.assertEqual(ev.neg_arr_ap, -ev.arr_ap)
            self.assertEqual(ev.neg_arr_j2000, -ev.arr_j2000)
            self.assertEqual(ev.neg_arr_ap_j2000, -ev.arr_ap_j2000)

            if (origin, frame) == ('SSB', 'J2000'):
                self.assertIs(ev, ev._ssb_)
                self.assertEqual(ev.arr_j2000, ev.arr)
                self.assertEqual(ev.arr_ap_j2000, ev.arr_ap)
            else:
                self.assertIsNotNone(ev._ssb_)
                self.assertIsNotNone(ev._ssb_.arr)
                self.assertIsNotNone(ev._ssb_.arr_ap)

            ########################
            # Define arr_ap
            ########################

            ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
            self.assertIsNone(ev._arr_)
            self.assertIsNone(ev._arr_ap_)
            self.assertIsNone(ev._neg_arr_)
            self.assertIsNone(ev._neg_arr_ap_)
            self.assertIsNone(ev._ssb_)

            ev.arr_ap = (1,2,3)
            self.assertEqual(ev._arr_ap_, Vector3((1.,2.,3.)))
            self.assertIsNone(ev._arr_)
            self.assertIsNone(ev._neg_arr_)
            self.assertIsNone(ev._neg_arr_ap_)
            self.assertIsNone(ev._ssb_)

            self.assertEqual(ev.neg_arr_ap, Vector3((-1.,-2.,-3.)))
            self.assertEqual(ev._arr_ap_, Vector3((1.,2.,3.)))
            self.assertIsNone(ev._arr_)
            self.assertIsNone(ev._neg_arr_)
            self.assertIsNone(ev._ssb_)

            try:
                ev.arr = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            self.assertIsNone(ev._arr_)
            self.assertIsNone(ev._neg_arr_)
            self.assertIsNone(ev._ssb_)

            # Let arr and ssb be filled in
            _ = ev.arr
            self.assertTrue((ev.arr_ap - ev.arr).norm() < 5*BETA)
            self.assertEqual(ev.neg_arr, -ev.arr)
            self.assertEqual(ev.neg_arr_ap, -ev.arr_ap)
            self.assertEqual(ev.neg_arr_j2000, -ev.arr_j2000)
            self.assertEqual(ev.neg_arr_ap_j2000, -ev.arr_ap_j2000)

            if (origin, frame) == ('SSB', 'J2000'):
                self.assertIs(ev, ev._ssb_)
                self.assertIs(ev.arr_j2000, ev.arr)
                self.assertIs(ev.arr_ap_j2000, ev.arr_ap)
            else:
                self.assertIsNotNone(ev._ssb_)
                self.assertIsNotNone(ev._ssb_.arr)
                self.assertIsNotNone(ev._ssb_.arr_ap)

            ########################
            # Define arr_j2000
            ########################

            ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
            self.assertIsNone(ev._arr_)
            self.assertIsNone(ev._arr_ap_)
            self.assertIsNone(ev._neg_arr_)
            self.assertIsNone(ev._neg_arr_ap_)
            self.assertIsNone(ev._ssb_)

            ev.arr_j2000 = (1,2,3)
            self.assertIsNotNone(ev._ssb_)
            self.assertEqual(ev.ssb._arr_, Vector3((1.,2.,3.)))
            self.assertIsNone(ev.ssb._arr_ap_)
            self.assertIsNone(ev.ssb._neg_arr_)
            self.assertIsNone(ev.ssb._neg_arr_ap_)

            self.assertIsNotNone(ev._arr_)
            self.assertIsNone(ev._arr_ap_)
            self.assertIsNone(ev._neg_arr_)
            self.assertIsNone(ev._neg_arr_ap_)

            try:
                ev.arr = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            try:
                ev.arr_ap = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            self.assertIsNotNone(ev._arr_)
            self.assertIsNone(ev._arr_ap_)
            self.assertIsNone(ev._neg_arr_)
            self.assertIsNone(ev._neg_arr_ap_)

            try:
                ev.neg_arr = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            self.assertIsNone(ev._arr_ap_)
            self.assertIsNone(ev._neg_arr_)
            self.assertIsNone(ev._neg_arr_ap_)

            # Let arr_ap and ssb be filled in
            _ = ev.arr_ap
            self.assertTrue((ev.arr_ap - ev.arr).norm() < 5*BETA)
            self.assertEqual(ev.neg_arr, -ev.arr)
            self.assertEqual(ev.neg_arr_ap, -ev.arr_ap)
            self.assertEqual(ev.neg_arr_j2000, -ev.arr_j2000)
            self.assertEqual(ev.neg_arr_ap_j2000, -ev.arr_ap_j2000)

            if (origin, frame) == ('SSB', 'J2000'):
                self.assertIs(ev, ev._ssb_)
                self.assertEqual(ev.arr_j2000, ev.arr)
                self.assertEqual(ev.arr_ap_j2000, ev.arr_ap)
            else:
                self.assertIsNotNone(ev._ssb_)
                self.assertIsNotNone(ev._ssb_.arr)
                self.assertIsNotNone(ev._ssb_.arr_ap)

            ########################
            # Define arr_ap_j2000
            ########################

            ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
            self.assertIsNone(ev._arr_)
            self.assertIsNone(ev._arr_ap_)
            self.assertIsNone(ev._neg_arr_)
            self.assertIsNone(ev._neg_arr_ap_)
            self.assertIsNone(ev._ssb_)

            ev.arr_ap_j2000 = (1,2,3)
            self.assertIsNotNone(ev._ssb_)
            self.assertEqual(ev.ssb._arr_ap_, Vector3((1.,2.,3.)))
            self.assertIsNone(ev.ssb._arr_)
            self.assertIsNone(ev.ssb._neg_arr_)
            self.assertIsNone(ev.ssb._neg_arr_ap_)

            self.assertIsNotNone(ev._arr_ap_)
            self.assertIsNone(ev._arr_)
            self.assertIsNone(ev._neg_arr_)
            self.assertIsNone(ev._neg_arr_ap_)

            try:
                ev.arr = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            try:
                ev.arr_ap = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            self.assertIsNotNone(ev._arr_ap_)
            self.assertIsNone(ev._arr_)
            self.assertIsNone(ev._neg_arr_)
            self.assertIsNone(ev._neg_arr_ap_)

            try:
                ev.neg_arr_ap = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            self.assertIsNone(ev._arr_)
            self.assertIsNone(ev._neg_arr_)
            self.assertIsNone(ev._neg_arr_ap_)

            # Let arr and ssb be filled in
            _ = ev.arr_ap
            self.assertTrue((ev.arr_ap - ev.arr).norm() < 5*BETA)
            self.assertEqual(ev.neg_arr, -ev.arr)
            self.assertEqual(ev.neg_arr_ap, -ev.arr_ap)
            self.assertEqual(ev.neg_arr_j2000, -ev.arr_j2000)
            self.assertEqual(ev.neg_arr_ap_j2000, -ev.arr_ap_j2000)

            if (origin, frame) == ('SSB', 'J2000'):
                self.assertIs(ev, ev._ssb_)
                self.assertEqual(ev.arr_j2000, ev.arr)
                self.assertEqual(ev.arr_ap_j2000, ev.arr_ap)
            else:
                self.assertIsNotNone(ev._ssb_)
                self.assertIsNotNone(ev._ssb_.arr)
                self.assertIsNotNone(ev._ssb_.arr_ap)

            ########################
            # Define neg_arr
            ########################

            ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
            self.assertIsNone(ev._arr_)
            self.assertIsNone(ev._arr_ap_)
            self.assertIsNone(ev._neg_arr_)
            self.assertIsNone(ev._neg_arr_ap_)
            self.assertIsNone(ev._ssb_)

            ev.neg_arr = (-1,-2,-3)
            self.assertEqual(ev._arr_, Vector3((1.,2.,3.)))
            self.assertEqual(ev._neg_arr_, Vector3((-1.,-2.,-3.)))
            self.assertIsNone(ev._arr_ap_)
            self.assertIsNone(ev._neg_arr_ap_)
            self.assertIsNone(ev._ssb_)

            try:
                ev.arr = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            try:
                ev.arr_ap = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            self.assertIsNone(ev._arr_ap_)
            self.assertIsNone(ev._neg_arr_ap_)
            self.assertIsNone(ev._ssb_)

            try:
                ev.neg_arr_ap = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            self.assertIsNone(ev._arr_ap_)
            self.assertIsNone(ev._neg_arr_ap_)
            self.assertIsNone(ev._ssb_)

            # Let arr_ap and ssb be filled in
            _ = ev.arr_ap
            self.assertTrue((ev.arr_ap - ev.arr).norm() < 5*BETA)
            self.assertEqual(ev.neg_arr, -ev.arr)
            self.assertEqual(ev.neg_arr_ap, -ev.arr_ap)
            self.assertEqual(ev.neg_arr_j2000, -ev.arr_j2000)
            self.assertEqual(ev.neg_arr_ap_j2000, -ev.arr_ap_j2000)

            if (origin, frame) == ('SSB', 'J2000'):
                self.assertIs(ev, ev._ssb_)
                self.assertEqual(ev.arr_j2000, ev.arr)
                self.assertEqual(ev.arr_ap_j2000, ev.arr_ap)
            else:
                self.assertIsNotNone(ev._ssb_)
                self.assertIsNotNone(ev._ssb_.arr)
                self.assertIsNotNone(ev._ssb_.arr_ap)

            ########################
            # Define neg_arr_ap
            ########################

            ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
            self.assertIsNone(ev._arr_)
            self.assertIsNone(ev._arr_ap_)
            self.assertIsNone(ev._neg_arr_)
            self.assertIsNone(ev._neg_arr_ap_)
            self.assertIsNone(ev._ssb_)

            ev.neg_arr_ap = (-1,-2,-3)
            self.assertEqual(ev._arr_ap_, Vector3((1.,2.,3.)))
            self.assertEqual(ev._neg_arr_ap_, Vector3((-1.,-2.,-3.)))
            self.assertIsNone(ev._arr_)
            self.assertIsNone(ev._neg_arr_)
            self.assertIsNone(ev._ssb_)

            try:
                ev.arr = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            try:
                ev.arr_ap = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            self.assertIsNone(ev._arr_)
            self.assertIsNone(ev._neg_arr_)
            self.assertIsNone(ev._ssb_)

            try:
                ev.neg_arr_ap = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            self.assertIsNone(ev._arr_)
            self.assertIsNone(ev._neg_arr_)
            self.assertIsNone(ev._ssb_)

            # Let arr and ssb be filled in
            _ = ev.arr
            self.assertTrue((ev.arr_ap - ev.arr).norm() < 5*BETA)
            self.assertEqual(ev.neg_arr, -ev.arr)
            self.assertEqual(ev.neg_arr_ap, -ev.arr_ap)
            self.assertEqual(ev.neg_arr_j2000, -ev.arr_j2000)
            self.assertEqual(ev.neg_arr_ap_j2000, -ev.arr_ap_j2000)

            if (origin, frame) == ('SSB', 'J2000'):
                self.assertIs(ev, ev._ssb_)
                self.assertEqual(ev.arr_j2000, ev.arr)
                self.assertEqual(ev.arr_ap_j2000, ev.arr_ap)
            else:
                self.assertIsNotNone(ev._ssb_)
                self.assertIsNotNone(ev._ssb_.arr)
                self.assertIsNotNone(ev._ssb_.arr_ap)

            ########################
            # Define neg_arr_j2000
            ########################

            ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
            self.assertIsNone(ev._arr_)
            self.assertIsNone(ev._arr_ap_)
            self.assertIsNone(ev._neg_arr_)
            self.assertIsNone(ev._neg_arr_ap_)
            self.assertIsNone(ev._ssb_)

            ev.neg_arr_j2000 = (-1,-2,-3)
            self.assertIsNotNone(ev._ssb_)
            self.assertEqual(ev.ssb._arr_, Vector3((1.,2.,3.)))
            self.assertEqual(ev.ssb._neg_arr_, Vector3((-1.,-2.,-3.)))
            self.assertIsNone(ev.ssb._arr_ap_)
            self.assertIsNone(ev.ssb._neg_arr_ap_)

            try:
                ev.arr = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            try:
                ev.arr_ap = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            try:
                ev.neg_arr = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            # Let arr_ap and ssb be filled in
            _ = ev.arr_ap
            self.assertTrue((ev.arr_ap - ev.arr).norm() < 5*BETA)
            self.assertEqual(ev.neg_arr, -ev.arr)
            self.assertEqual(ev.neg_arr_ap, -ev.arr_ap)
            self.assertEqual(ev.neg_arr_j2000, -ev.arr_j2000)
            self.assertEqual(ev.neg_arr_ap_j2000, -ev.arr_ap_j2000)

            if (origin, frame) == ('SSB', 'J2000'):
                self.assertIs(ev, ev._ssb_)
                self.assertEqual(ev.arr_j2000, ev.arr)
                self.assertEqual(ev.arr_ap_j2000, ev.arr_ap)
            else:
                self.assertIsNotNone(ev._ssb_)
                self.assertIsNotNone(ev._ssb_.arr)
                self.assertIsNotNone(ev._ssb_.arr_ap)

            ########################
            # Define neg_arr_ap_j2000
            ########################

            ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
            self.assertIsNone(ev._arr_)
            self.assertIsNone(ev._arr_ap_)
            self.assertIsNone(ev._neg_arr_)
            self.assertIsNone(ev._neg_arr_ap_)
            self.assertIsNone(ev._ssb_)

            ev.neg_arr_ap_j2000 = (-1,-2,-3)
            self.assertIsNotNone(ev._ssb_)
            self.assertEqual(ev.ssb._arr_ap_, Vector3((1.,2.,3.)))
            self.assertEqual(ev.ssb._neg_arr_ap_, Vector3((-1.,-2.,-3.)))
            self.assertIsNotNone(ev.ssb._arr_ap_)
            self.assertIsNotNone(ev.ssb._neg_arr_ap_)

            try:
                ev.arr = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            try:
                ev.arr_ap = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            try:
                ev.neg_arr = (1,2,3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            # Let arr and ssb be filled in
            _ = ev.arr
            self.assertTrue((ev.arr_ap - ev.arr).norm() < 5*BETA)
            self.assertEqual(ev.neg_arr, -ev.arr)
            self.assertEqual(ev.neg_arr_ap, -ev.arr_ap)
            self.assertEqual(ev.neg_arr_j2000, -ev.arr_j2000)
            self.assertEqual(ev.neg_arr_ap_j2000, -ev.arr_ap_j2000)

            if (origin, frame) == ('SSB', 'J2000'):
                self.assertIs(ev, ev._ssb_)
                self.assertEqual(ev.arr_j2000, ev.arr)
                self.assertEqual(ev.arr_ap_j2000, ev.arr_ap)
            else:
                self.assertIsNotNone(ev._ssb_)
                self.assertIsNotNone(ev._ssb_.arr)
                self.assertIsNotNone(ev._ssb_.arr_ap)

            ########################
            # Define dep
            ########################

            ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
            self.assertIsNone(ev._dep_)
            self.assertIsNone(ev._dep_ap_)

            ev.dep = (-1,2,-3)
            self.assertEqual(ev._dep_, Vector3((-1.,2.,-3.)))
            self.assertIsNone(ev._dep_ap_)
            self.assertIsNone(ev._ssb_)

            try:
                ev.dep_ap = (-1,2,-3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            self.assertIsNone(ev._dep_ap_)
            self.assertIsNone(ev._ssb_)

            # Fill in dep_ap and ssb
            _ = ev.dep_ap
            self.assertTrue((ev.dep_ap - ev.dep).norm() < 5*BETA)

            if (origin, frame) == ('SSB', 'J2000'):
                self.assertIs(ev, ev._ssb_)
                self.assertEqual(ev.dep_j2000, ev.dep)
                self.assertEqual(ev.dep_ap_j2000, ev.dep_ap)
            else:
                self.assertIsNotNone(ev._ssb_)
                self.assertIsNotNone(ev._ssb_.dep)
                self.assertIsNotNone(ev._ssb_.dep_ap)

            ########################
            # Define dep_ap
            ########################

            ev = Event(0., (Vector3.ZERO, SPEED * Vector3.ZAXIS), origin, frame)
            self.assertIsNone(ev._dep_)
            self.assertIsNone(ev._dep_ap_)

            ev.dep_ap = (-1,2,-3)
            self.assertEqual(ev._dep_ap_, Vector3((-1.,2.,-3.)))
            self.assertIsNone(ev._dep_)
            self.assertIsNone(ev._ssb_)

            try:
                ev.dep_ap = (-1,2,-3)
                self.assertTrue(False, msg='ValueError not raised')
            except ValueError:
                pass

            self.assertIsNone(ev._dep_)
            self.assertIsNone(ev._ssb_)

            # Fill in dep and ssb
            _ = ev.dep
            self.assertTrue((ev.dep_ap - ev.dep).norm() < 5*BETA)

            if (origin, frame) == ('SSB', 'J2000'):
                self.assertIs(ev, ev._ssb_)
                self.assertEqual(ev.dep_j2000, ev.dep)
                self.assertEqual(ev.dep_ap_j2000, ev.dep_ap)
            else:
                self.assertIsNotNone(ev._ssb_)
                self.assertIsNotNone(ev._ssb_.dep)
                self.assertIsNotNone(ev._ssb_.dep_ap)

#########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
