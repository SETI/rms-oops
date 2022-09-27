################################################################################
# oops/backplane/unittester.py
################################################################################

from __future__ import print_function

import numpy as np
import os.path

from polymath import Vector3

from oops.backplane    import Backplane
from oops.body         import Body
from oops.event        import Event
from oops.meshgrid     import Meshgrid
from oops.surface.ansa import Ansa
import oops.config as config
from oops.backplane.exercise_backplanes import exercise_backplanes


################################################################################
# UNIT TESTS
################################################################################

import unittest
from oops.config import ABERRATION
from oops.unittester_support import TESTDATA_PARENT_DIRECTORY

UNITTEST_SATURN_FILESPEC = os.path.join(TESTDATA_PARENT_DIRECTORY,
                                      'cassini/ISS/W1573721822_1.IMG')
UNITTEST_RHEA_FILESPEC = os.path.join(TESTDATA_PARENT_DIRECTORY,
                                      'cassini/ISS/N1649465464_1.IMG')
UNITTEST_UNDERSAMPLE = 16

class Test_Backplane(unittest.TestCase):

    OLD_RHEA_SURFACE = None

    #===========================================================================
    def setUp(self):
        global OLD_RHEA_SURFACE

        from oops.surface.ellipsoid  import Ellipsoid

        Body.reset_registry()
        Body.define_solar_system('2000-01-01', '2020-01-01')

        # Distort Rhea's shape for better Ellipsoid testing
        rhea = Body.as_body('RHEA')
        OLD_RHEA_SURFACE = rhea.surface
        old_rhea_radii = OLD_RHEA_SURFACE.radii

        new_rhea_radii = tuple(np.array([1.1, 1., 0.9]) * old_rhea_radii)
        new_rhea_surface = Ellipsoid(rhea.path, rhea.frame, new_rhea_radii)
        Body.as_body('RHEA').surface = new_rhea_surface

  #      config.LOGGING.on('   ')
        config.EVENT_CONFIG.collapse_threshold = 0.
        config.SURFACE_PHOTONS.collapse_threshold = 0.

    #===========================================================================
    def tearDown(self):
        global OLD_RHEA_SURFACE

        config.LOGGING.off()
        config.EVENT_CONFIG.collapse_threshold = 3.
        config.SURFACE_PHOTONS.collapse_threshold = 3.

        # Restore Rhea's shape
        Body.as_body('RHEA').surface = OLD_RHEA_SURFACE

        ABERRATION.old = False

    #===========================================================================
    def runTest(self):
      import hosts.cassini.iss as iss

      from oops.surface.centricspheroid import CentricSpheroid
      from oops.surface.graphicspheroid import GraphicSpheroid
      from oops.surface.centricellipsoid import CentricEllipsoid
      from oops.surface.graphicellipsoid import GraphicEllipsoid

      for ABERRATION.old in (False, True):

        snap = iss.from_file(UNITTEST_SATURN_FILESPEC, fast_distortion=False)
        meshgrid = Meshgrid.for_fov(snap.fov, undersample=UNITTEST_UNDERSAMPLE,
                                    swap=True)
        #meshgrid = Meshgrid(snap.fov, (512,512))
        uv0 = meshgrid.uv
        bp = Backplane(snap, meshgrid)

        # Actual (ra,dec)
        ra = bp.right_ascension(apparent=False)
        dec = bp.declination(apparent=False)

        ev = Event(snap.midtime, Vector3.ZERO, snap.path, snap.frame)
        ev.neg_arr_j2000 = Vector3.from_ra_dec_length(ra, dec)
        uv = snap.fov.uv_from_los(ev.neg_arr_ap)
        diff = uv - uv0
        self.assertTrue(diff.norm().max() < 1.e-9)

        # Apparent (ra,dec)  # test doesn't work for ABERRATION=old
        if not ABERRATION.old:
            ra = bp.right_ascension(apparent=True)
            dec = bp.declination(apparent=True)

            ev = Event(snap.midtime, Vector3.ZERO, snap.path, snap.frame)
            ev.neg_arr_ap_j2000 = Vector3.from_ra_dec_length(ra, dec)
            uv = snap.fov.uv_from_los(ev.neg_arr_ap)

            diff = uv - uv0
        self.assertTrue(diff.norm().max() < 1.e-9)

        # RingPlane (rad, lon)
        rad = bp.ring_radius('saturn:ring')
        lon = bp.ring_longitude('saturn:ring', reference='node')

        ev = Event(snap.midtime, Vector3.ZERO, snap.path, snap.frame)
        body = Body.as_body('SATURN_RING_PLANE')
        (surface_ev, ev) = body.surface.photon_to_event_by_coords(ev, (rad,lon))

        uv = snap.fov.uv_from_los(ev.neg_arr_ap)
        diff = uv - uv0
        self.assertTrue(diff.norm().max() < 1.e-8)

        # Ansa (rad, alt)
        rad = bp.ansa_radius('saturn:ansa', radius_type='right')
        alt = bp.ansa_altitude('saturn:ansa')

        ev = Event(snap.midtime, Vector3.ZERO, snap.path, snap.frame)
        body = Body.as_body('SATURN_RING_PLANE')
        surface = Ansa.for_ringplane(body.surface)
        (surface_ev, ev) = surface.photon_to_event_by_coords(ev, (rad,alt))

        uv = snap.fov.uv_from_los(ev.neg_arr_ap)
        diff = uv - uv0
        self.assertTrue(diff.norm().max() < 1.e-8)

        # Spheroid (lon,lat)
        lat = bp.latitude('saturn', lat_type='squashed')
        lon = bp.longitude('saturn', reference='iau', direction='east',
                                     lon_type='centric')

        ev = Event(snap.midtime, Vector3.ZERO, snap.path, snap.frame)
        body = Body.as_body('SATURN')
        (surface_ev, ev) = body.surface.photon_to_event_by_coords(ev, (lon,lat))

        uv = snap.fov.uv_from_los(ev.neg_arr_ap)
        diff = uv - uv0
        self.assertTrue(diff.norm().max() < 1.e-8)

        # CentricSpheroid (lon,lat)
        lat = bp.latitude('saturn', lat_type='centric')
        lon = bp.longitude('saturn', reference='iau', direction='east',
                                     lon_type='centric')

        ev = Event(snap.midtime, Vector3.ZERO, snap.path, snap.frame)
        body = Body.as_body('SATURN')
        surface = CentricSpheroid(body.path, body.frame, body.surface.radii)
        (surface_ev, ev) = surface.photon_to_event_by_coords(ev, (lon,lat))

        uv = snap.fov.uv_from_los(ev.neg_arr_ap)
        diff = uv - uv0
        self.assertTrue(diff.norm().max() < 1.e-8)

        # GraphicSpheroid (lon,lat)
        lat = bp.latitude('saturn', lat_type='graphic')
        lon = bp.longitude('saturn', reference='iau', direction='east',
                                     lon_type='centric')

        ev = Event(snap.midtime, Vector3.ZERO, snap.path, snap.frame)
        body = Body.as_body('SATURN')
        surface = GraphicSpheroid(body.path, body.frame, body.surface.radii)
        (surface_ev, ev) = surface.photon_to_event_by_coords(ev, (lon,lat))

        uv = snap.fov.uv_from_los(ev.neg_arr_ap)
        diff = uv - uv0
        self.assertTrue(diff.norm().max() < 1.e-8)

        # Rhea tests, with Rhea modified
        body = Body.as_body('RHEA')
        snap = iss.from_file(UNITTEST_RHEA_FILESPEC, fast_distortion=False)
        meshgrid = Meshgrid.for_fov(snap.fov, undersample=UNITTEST_UNDERSAMPLE,
                                    swap=True)

        uv0 = meshgrid.uv
        bp = Backplane(snap, meshgrid)

        # Ellipsoid (lon,lat)
        lat = bp.latitude('rhea', lat_type='squashed')
        lon = bp.longitude('rhea', reference='iau', direction='east',
                                     lon_type='squashed')

        ev = Event(snap.midtime, Vector3.ZERO, snap.path, snap.frame)
        body = Body.as_body('RHEA')
        (surface_ev, ev) = body.surface.photon_to_event_by_coords(ev, (lon,lat))

        uv = snap.fov.uv_from_los(ev.neg_arr_ap)
        diff = uv - uv0
        #print(diff.norm().min(), diff.norm().max())
        self.assertTrue(diff.norm().max() < 2.e-7)

        # CentricEllipsoid (lon,lat)
        lat = bp.latitude('rhea', lat_type='centric')
        lon = bp.longitude('rhea', reference='iau', direction='east',
                                     lon_type='centric')

        ev = Event(snap.midtime, Vector3.ZERO, snap.path, snap.frame)
        body = Body.as_body('RHEA')
        surface = CentricEllipsoid(body.path, body.frame, body.surface.radii)
        (surface_ev, ev) = surface.photon_to_event_by_coords(ev, (lon,lat))

        uv = snap.fov.uv_from_los(ev.neg_arr_ap)
        diff = uv - uv0
        #print(diff.norm().min(), diff.norm().max())
        self.assertTrue(diff.norm().max() < 2.e-7)

        # GraphicEllipsoid (lon,lat)
        lat = bp.latitude('rhea', lat_type='graphic')
        lon = bp.longitude('rhea', reference='iau', direction='east',
                                     lon_type='graphic')

        ev = Event(snap.midtime, Vector3.ZERO, snap.path, snap.frame)
        body = Body.as_body('RHEA')
        surface = GraphicEllipsoid(body.path, body.frame, body.surface.radii)
        (surface_ev, ev) = surface.photon_to_event_by_coords(ev, (lon,lat))

        uv = snap.fov.uv_from_los(ev.neg_arr_ap)
        diff = uv - uv0
        #print(diff.norm().min(), diff.norm().max())
        self.assertTrue(diff.norm().max() < 2.e-7)

########################################

class Test_Backplane_Exercises(unittest.TestCase):

    #===========================================================================
    def runTest(self):

        import hosts.cassini.iss as iss
        iss.initialize(asof='2019-09-01', mst_pck=True)

        filespec = os.path.join(TESTDATA_PARENT_DIRECTORY,
                                'cassini/ISS/W1573721822_1.IMG')

#         filespec = os.path.join(TESTDATA_PARENT_DIRECTORY,
#                                 'cassini/ISS/W1575632515_1.IMG')
# TARGET = SATURN

#         filespec = os.path.join(TESTDATA_PARENT_DIRECTORY,
#                                 'cassini/ISS/N1573845439_1.IMG')
# TARGET = 'ENCELADUS'

        TEST_LEVEL = 3

        logging = False         # Turn on for efficiency testing

        if TEST_LEVEL == 3:     # long and slow, creates images, logging off
            printing = True
            saving = True
            undersample = 1
        elif TEST_LEVEL == 2:   # faster, prints info, no images, undersampled
            printing = True
            saving = False
            undersample = 16
        elif TEST_LEVEL == 1:   # executes every routine but does no printing
            printing = False
            saving = False
            undersample = 32

        if TEST_LEVEL > 0:

            snap = iss.from_file(filespec)
            bp = exercise_backplanes(snap, printing, logging, saving,
                                     undersample,
                                     use_inventory=True, inventory_border=4)

        else:
            print('test skipped')

########################################

if __name__ == '__main__':

    unittest.main(verbosity=2)

################################################################################
