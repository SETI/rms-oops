################################################################################
# tests/frame/test_poleframe.py
################################################################################

import numpy as np
import unittest

import cspyce

from polymath   import Scalar, Vector3
from oops.body  import Body
from oops.event import Event
from oops.frame import Frame, PoleFrame, RingFrame, SpiceFrame
from oops.path  import Path, SpicePath
from oops.unittester_support import TEST_SPICE_PREFIX


class Test_PoleFrame(unittest.TestCase):

    def setUp(self):
        paths = TEST_SPICE_PREFIX.retrieve(['naif0009.tls', 'pck00010.tpc',
                                            'de421.bsp'])
        for path in paths:
            cspyce.furnsh(path)
        Path.reset_registry()
        Frame.reset_registry()

    def tearDown(self):
        pass

    def runTest(self):

        np.random.seed(1152)

        _ = SpicePath('MARS', 'SSB')
        planet = SpiceFrame('IAU_MARS', 'J2000')
        self.assertEqual(Frame.as_wayframe('IAU_MARS'), planet.wayframe)

        # This invariable pole is aligned with the planet's pole, so this
        # should behave just like a RingFrame
        for aries in (False, True):
            pole = planet.transform_at_time(0.).matrix.inverse() * Vector3.ZAXIS
            poleframe = PoleFrame(planet, pole, cache_size=0, aries=aries)
            ringframe = RingFrame(planet, epoch=0., aries=aries)
            self.assertEqual(Frame.as_wayframe('IAU_MARS_POLE'), poleframe.wayframe)
            vectors = Vector3(np.random.rand(3,4,2,3)).unit()

            ring_vecs = ringframe.transform_at_time(0.).rotate(vectors)
            pole_vecs = poleframe.transform_at_time(0.).rotate(vectors)
            diffs = ring_vecs - pole_vecs
            self.assertTrue(diffs.norm().max() < 1.e-15)

            posvel = np.random.rand(3,4,2,6)
            event = Event(0., (posvel[...,0:3], posvel[...,3:6]), 'SSB', 'J2000')
            rotated = event.wrt_frame('IAU_MARS')
            fixed   = event.wrt_frame(poleframe)

            # Confirm Z axis is tied to planet's pole
            diffs = Scalar(rotated.pos.vals[...,2]) - Scalar(fixed.pos.vals[...,2])
            self.assertTrue(diffs.abs().max() < 1.e-15)

            # Confirm X-axis is tied to the J2000 equator
            xaxis = Event(0., Vector3.XAXIS, 'SSB', poleframe)
            test = xaxis.wrt_frame('J2000').pos
            self.assertTrue(abs(test.values[2]) < 1.e-15)

            # Confirm it's the ascending node
            xaxis = Event(0., (1,1.e-8,0), 'SSB', poleframe)
            test = xaxis.wrt_frame('J2000').pos
            self.assertTrue(test.values[2] > 0.)

        # Test reference angles, Aries = True vs. False
        vectors = Vector3(np.random.rand(100,3)).unit()
        poleframe1 = PoleFrame(planet, pole, cache_size=0, aries=True)
        poleframe2 = PoleFrame(planet, pole, cache_size=0, aries=False)
        pole1_vecs = poleframe1.transform_at_time(0.).rotate(vectors)
        pole2_vecs = poleframe2.transform_at_time(0.).rotate(vectors)
        (x1,y1,z1) = pole1_vecs.to_scalars()
        (x2,y2,z2) = pole2_vecs.to_scalars()

        # Z axes are the same
        self.assertTrue((z1 - z2).abs().max() < 1.e-15)

        # Longitudes have a fixed, nonzero offset
        dlon = (y1.arctan2(x1) - y2.arctan2(x2)) % (2.*np.pi)
        self.assertTrue(dlon[0] != 0.)
        self.assertTrue((dlon - dlon[0]).abs().max() < 1.e-15)

        diff = dlon[0] - poleframe1.invariable_node_lon
        diff = (diff - np.pi) % (2.*np.pi) - np.pi
        self.assertTrue(diff.abs() < 1.e-15)

        # Now try for Neptune
        _ = SpicePath('NEPTUNE', 'SSB')
        planet = SpiceFrame('IAU_NEPTUNE', 'J2000')

        # This invariable pole is aligned with the planet's pole, so this
        # should behave just like a RingFrame
        for aries in (False, True):
            pole = planet.transform_at_time(0.).matrix.inverse() * Vector3.ZAXIS
            poleframe = PoleFrame(planet, pole, cache_size=0, aries=aries)
            ringframe = RingFrame(planet, epoch=0., aries=aries)

            vectors = Vector3(np.random.rand(3,4,2,3)).unit()

            ring_vecs = ringframe.transform_at_time(0.).rotate(vectors)
            pole_vecs = poleframe.transform_at_time(0.).rotate(vectors)
            diffs = ring_vecs - pole_vecs
            self.assertTrue(diffs.norm().max() < 3.e-15)

            posvel = np.random.rand(3,4,2,6)
            event = Event(0., (posvel[...,0:3], posvel[...,3:6]), 'SSB', 'J2000')
            rotated = event.wrt_frame('IAU_NEPTUNE')
            fixed   = event.wrt_frame(poleframe)

            # Confirm Z axis is tied to planet's pole
            diffs = Scalar(rotated.pos.vals[...,2]) - Scalar(fixed.pos.vals[...,2])
            self.assertTrue(diffs.abs().max() < 1.e-15)

            # Confirm X-axis is tied to the J2000 equator
            xaxis = Event(0., Vector3.XAXIS, 'SSB', poleframe)
            test = xaxis.wrt_frame('J2000').pos
            self.assertTrue(abs(test.values[2]) < 1.e-15)

            # Confirm it's the ascending node
            xaxis = Event(0., (1,1.e-8,0), 'SSB', poleframe)
            test = xaxis.wrt_frame('J2000').pos
            self.assertTrue(test.values[2] > 0.)

        # Test reference angles, Aries = True vs. False
        vectors = Vector3(np.random.rand(100,3)).unit()
        poleframe1 = PoleFrame(planet, pole, cache_size=0, aries=True)
        poleframe2 = PoleFrame(planet, pole, cache_size=0, aries=False)
        pole1_vecs = poleframe1.transform_at_time(0.).rotate(vectors)
        pole2_vecs = poleframe2.transform_at_time(0.).rotate(vectors)
        (x1,y1,z1) = pole1_vecs.to_scalars()
        (x2,y2,z2) = pole2_vecs.to_scalars()

        # Z axes are the same
        self.assertTrue((z1 - z2).abs().max() < 1.e-15)

        # Longitudes have a fixed, nonzero offset
        dlon = (y1.arctan2(x1) - y2.arctan2(x2)) % (2.*np.pi)
        self.assertTrue(dlon[0] != 0.)
        self.assertTrue((dlon - dlon[0]).abs().max() < 1.e-15)

        diff = dlon[0] - poleframe1.invariable_node_lon
        diff = (diff - np.pi) % (2.*np.pi) - np.pi
        self.assertTrue(diff.abs() < 1.e-15)

        # Neptune at multiple times, with actual polar precession
        times = Scalar(np.arange(1000) * 86400. * 365.)     # 1000 years
        for aries in (False, True):
            ra  = cspyce.bodvrd('NEPTUNE', 'POLE_RA')[0]  * np.pi/180
            dec = cspyce.bodvrd('NEPTUNE', 'POLE_DEC')[0] * np.pi/180
            pole = Vector3.from_ra_dec_length(ra,dec)
            poleframe = PoleFrame(planet, pole, cache_size=0, aries=aries)

            # Make sure Z-axis tracks Neptune pole
            pole_vecs = poleframe.transform_at_time(times).unrotate(Vector3.ZAXIS)
            test_vecs = planet.transform_at_time(times).unrotate(Vector3.ZAXIS)
            diffs = pole_vecs - test_vecs
            self.assertTrue(diffs.norm().max() < 1.e-15)

            # Make sure Z-axis circles the pole at uniform distance
            seps = pole_vecs.sep(pole)
            sep_mean = seps.mean()
            self.assertTrue((seps - sep_mean).abs().max() < 3.e-5)

            # Make sure the X-axis stays close to the ecliptic
            if not aries:
                node_vecs = poleframe.transform_at_time(times).unrotate(Vector3.XAXIS)
                min_node_z = np.min(node_vecs.values[:,2])
                max_node_z = np.max(node_vecs.values[:,2])
                self.assertTrue(min_node_z > -0.0062)
                self.assertTrue(max_node_z <  0.0062)
                self.assertTrue(abs(min_node_z + max_node_z) < 1.e-8)

            # Make sure the X-axis stays in a generally fixed direction
            diffs = node_vecs - node_vecs[0]
            self.assertTrue(diffs.norm().max() < 0.02)

        # Test cache
        poleframe = PoleFrame(planet, pole, cache_size=3)
        self.assertTrue(poleframe.cache_size == 4)
        self.assertTrue(poleframe.trim_size == 1)
        self.assertTrue(len(poleframe.cache) == 0)

        pole_vecs = poleframe.transform_at_time(times).unrotate(Vector3.ZAXIS)
        self.assertTrue(len(poleframe.cache) == 0)  # don't cache vectors
        self.assertFalse(poleframe.cached_value_returned)

        pole_vecs = poleframe.transform_at_time(100.).unrotate(Vector3.ZAXIS)
        self.assertTrue(len(poleframe.cache) == 1)
        self.assertTrue(100. in poleframe.cache)
        self.assertFalse(poleframe.cached_value_returned)

        pole_vecs = poleframe.transform_at_time(100.).unrotate(Vector3.ZAXIS)
        self.assertTrue(len(poleframe.cache) == 1)
        self.assertTrue(poleframe.cached_value_returned)

        pole_vecs = poleframe.transform_at_time(200.).unrotate(Vector3.ZAXIS)
        self.assertTrue(len(poleframe.cache) == 2)

        pole_vecs = poleframe.transform_at_time(300.).unrotate(Vector3.ZAXIS)
        self.assertTrue(len(poleframe.cache) == 3)

        pole_vecs = poleframe.transform_at_time(400.).unrotate(Vector3.ZAXIS)
        self.assertTrue(len(poleframe.cache) == 4)

        pole_vecs = poleframe.transform_at_time(500.).unrotate(Vector3.ZAXIS)
        self.assertTrue(len(poleframe.cache) == 4)
        self.assertTrue(100. not in poleframe.cache)

        pole_vecs = poleframe.transform_at_time(200.).unrotate(Vector3.ZAXIS)
        self.assertTrue(len(poleframe.cache) == 4)
        self.assertTrue(poleframe.cached_value_returned)

        pole_vecs = poleframe.transform_at_time(100.).unrotate(Vector3.ZAXIS)
        self.assertTrue(len(poleframe.cache) == 4)
        self.assertFalse(poleframe.cached_value_returned)
        self.assertTrue(300. not in poleframe.cache)

########################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
