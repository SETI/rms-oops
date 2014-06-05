################################################################################
# oops/inst/nh/lorri.py
#
# 2/9/14 Created (RSF)
################################################################################

import numpy as np
import julian
try:
    import astropy.io.fits as pyfits
except ImportError:
    import pyfits
import solar

import oops
from oops.inst.nh.nh_  import NewHorizons

################################################################################
# Standard class methods
################################################################################

def from_file(filespec, geom='spice', pointing='spice', **parameters):
    """Return a Snapshot object based on a given NewHorizons LORRI image file.

    If parameters["data"] is False, no data or associated arrays are loaded.
    If parameters["calibration"] is False, no calibration objects are created.
    If parameters["headers"] is False, no header dictionary is returned.

    If parameters["astrometry"] is True, this is equivalent to data=False,
    calibration=False, headers=False.

    If parameters["solar_range"] is specified, it overrides the distance from
    the Sun to the target body (in AU) for calibration purposes.
    """

    assert geom in {'spice', 'fits'}
    assert pointing in {'spice', 'fits'}

    LORRI.initialize()    # Define everything the first time through

    # Load the FITS file
    nh_file = pyfits.open(filespec)
    filename = os.path.split(filespec)[1]

    # Get key information from the header
    texp = nh_file[0].header['EXPTIME']
    tdb_midtime = nh_file[0].header['SPCSCET']
    tstart = tdb_midtime - texp/2

    binning_mode = nh_file[0].header['SFORMAT']
    fov = LORRI.fovs[binning_mode]

    target_name = nh_file[0].header['SPCCBTNM']
    if target_name.strip() == '---':
        target_name = 'PLUTO'

    try:
        target_body = oops.Body.lookup(target_name)
    except:
        target_body = None

    # Make sure the SPICE kernels are loaded
    NewHorizons.load_cks( tstart, tstart + texp)
    NewHorizons.load_spks(tstart, tstart + texp)

    if geom != 'spice':

        # First construct a path from the Sun to NH
        posx = -nh_file[0].header['SPCSSCX']
        posy = -nh_file[0].header['SPCSSCY']
        posz = -nh_file[0].header['SPCSSCZ']

        velx = -nh_file[0].header['SPCSSCVX']
        vely = -nh_file[0].header['SPCSSCVY']
        velz = -nh_file[0].header['SPCSSCVZ']

        # The path_id has to be unique to this observation
        sun_path = oops.Path.as_waypoint('SUN')
        path_id = '.NH_PATH_' + filename
        sc_path = oops.path.LinearPath((oops.Vector3([posx, posy, posz]),
                                        oops.Vector3([velx, vely, velz])),
                                       tdb_midtime, sun_path,
                                       oops.Frame.J2000,
                                       id=path_id)
        path = oops.Path.as_waypoint(sc_path)
    else:
        path = oops.Path.as_waypoint('NEW HORIZONS')

    if pointing != 'spice':

        # Next create a frame based on the boresight
        ra = nh_file[0].header['SPCBRRA']
        dec = nh_file[0].header['SPCBRDEC']

        # OH, THE HORROR
        year = int(nh_file[0].header['SPCUTCID'][:4])
        if year <= 2012:
            print 'before 2012!'
            north_clk = nh_file[0].header['SPCEMEN'] + 90.
        else:
            print 'after 2012!'
            north_clk = nh_file[0].header['SPCEMEN']

        scet = nh_file[0].header['SPCSCET']

        frame_id = '.NH_FRAME_' + filename
        lorri_frame = oops.frame.Cmatrix.from_ra_dec(ra, dec, north_clk,
                                                     oops.Frame.J2000,
                                                     id=frame_id)
        frame = oops.Frame.as_wayframe(lorri_frame)
    else:
        frame = oops.Frame.as_wayframe('NH_LORRI')

    # Create a Snapshot
    snapshot = oops.obs.Snapshot(('v','u'), tstart, texp, fov, path, frame,
                                 target = target_name,
                                 instrument = 'LORRI')

    # Interpret loader options
    if ('astrometry' in parameters) and parameters['astrometry']:
        include_data = False
        include_calibration = False
        include_headers = False

    else:
        include_data = ('data' not in parameters) or parameters['data']
        include_calibration = (('calibration' not in parameters) or
                               parameters['calibration'])
        include_headers = ('headers' not in parameters) or parameters['headers']

    if include_data:
        data = nh_file[0].data
        error = None
        quality = None

        try:
            error = nh_file[1].data
            quality = nh_file[2].data
        except IndexError:
            pass

        snapshot.insert_subfield('data', data)
        snapshot.insert_subfield('error', error)
        snapshot.insert_subfield('quality', quality)

    if include_calibration:
        spectral_name = target_name
        if parameters.has_key('calib_body'):
            spectral_name = parameters['calib_body']

        # Look up the solar range...
        try:
            solar_range = parameters['solar_range']
        except KeyError:
            solar_range = None

        # If necessary, get the solar range from the target name
        if solar_range is None and target_body is not None:
            target_sun_path = oops.Path.as_waypoint(target_name).wrt('SUN')
            # Paths of the relevant bodies need to be defined in advance!

            sun_event = target_sun_path.event_at_time(tdb_midtime)
            solar_range = sun_event.pos.norm().vals / solar.AU

        if solar_range is None:
            raise IOError("Calibration can't figure out range from Sun to " +
                          "target body " + target_name + " in file " +
                          filespec)

        extended_calib = {}
        point_calib = {}

        for spectral_name in ['SOLAR', 'PLUTO', 'PHOLUS', 'CHARON', 'JUPITER']:

            # Extended source
            spectral_radiance = nh_file[0].header['R' + spectral_name]

            # Point source
            spectral_irradiance = nh_file[0].header['P' + spectral_name]

            F_solar = 176.  # pivot 6076.2 A at 1 AU

            # Conversion to I/F
            extended_factor = (1. / texp / spectral_radiance * np.pi *
                               solar_range**2 / F_solar)
            point_factor = (1. / texp / spectral_irradiance / fov.uv_area *
                            np.pi * solar_range**2 / F_solar)

            extended_calib[spectral_name] = oops.calib.ExtendedSource('I/F',
                                                            extended_factor)
            point_calib[spectral_name] = oops.calib.PointSource('I/F',
                                                            point_factor, fov)

        snapshot.insert_subfield('point_calib', point_calib)
        snapshot.insert_subfield('extended_calib', extended_calib)

    if include_headers:
        headers = []
        for objects in nh_file:
            headers.append(objects.header)

        snapshot.insert_subfield('headers', headers)

    return snapshot

################################################################################

class LORRI(object):
    """A instance-free class to hold NewHorizons LORRI instrument parameters."""

    instrument_kernel = None
    fovs = {}
    initialized = False

    # Create a master version of the LORRI distortion models from
    #   Owen Jr., W.M., 2011. New Horizons LORRI Geometric Calibration of
    #   August 2006. JPL IOM 343L-11-002.
    
    LORRI_F = 2619.008    # mm
    LORRI_E2 = 2.696e-5   # / mm2
    LORRI_E5 = 1.988e-5   # / mm
    LORRI_E6 = -2.864e-5  # / mm
    LORRI_KX = 76.9231    # samples/mm
    LORRI_KY = -76.9231   # lines/mm
    
    LORRI_COEFF = np.zeros((4,4,2))
    LORRI_COEFF[1,0,0] = LORRI_KX          * LORRI_F
    LORRI_COEFF[3,0,0] = LORRI_KX*LORRI_E2 * LORRI_F**3
    LORRI_COEFF[1,2,0] = LORRI_KX*LORRI_E2 * LORRI_F**3
    LORRI_COEFF[1,1,0] = LORRI_KX*LORRI_E5 * LORRI_F**2
    LORRI_COEFF[2,0,0] = LORRI_KX*LORRI_E6 * LORRI_F**2
    
    LORRI_COEFF[0,1,1] = LORRI_KY          * LORRI_F
    LORRI_COEFF[2,1,1] = LORRI_KY*LORRI_E2 * LORRI_F**3
    LORRI_COEFF[0,3,1] = LORRI_KY*LORRI_E2 * LORRI_F**3
    LORRI_COEFF[0,2,1] = LORRI_KY*LORRI_E5 * LORRI_F**2
    LORRI_COEFF[1,1,1] = LORRI_KY*LORRI_E6 * LORRI_F**2

    @staticmethod
    def initialize():
        """Fill in key information about LORRI. Must be called first.
        """

        # Quick exit after first call
        if LORRI.initialized: return

        NewHorizons.initialize()
        NewHorizons.load_instruments()

        # Load the instrument kernel
        kernels = NewHorizons.spice_instrument_kernel('LORRI')
        LORRI.instrument_kernel = kernels[0]

        # Construct a Polynomial FOV
        info = LORRI.instrument_kernel['INS']['NH_LORRI_1X1']#[-98301]

        # Full field of view
        lines = info['PIXEL_LINES']
        samples = info['PIXEL_SAMPLES']

        xfov = info['FOV_REF_ANGLE']
        yfov = info['FOV_CROSS_ANGLE']
        assert info['FOV_ANGLE_UNITS'] == 'DEGREES'

        # Display directions: [u,v] = [right,down]
        full_fov = oops.fov.Polynomial(LORRI.LORRI_COEFF,
                                       (samples,lines), xy_to_uv=True)

        # Load the dictionary, include the subsampling modes
        LORRI.fovs['1X1'] = full_fov
        LORRI.fovs['4X4'] = oops.fov.Subsampled(full_fov, 4)

        # Construct a SpiceFrame
        lorri_flipped = oops.frame.SpiceFrame('NH_LORRI', id='NH_LORRI_FLIPPED')

        # The SPICE IK gives the boresight along -Z, so flip axes
        flipxyz = oops.Matrix3([[ 1, 0, 0],
                                [ 0,-1, 0],
                                [ 0, 0,-1]])
        ignore = oops.frame.Cmatrix(flipxyz, lorri_flipped, 'NH_LORRI')

        LORRI.initialized = True

    @staticmethod
    def reset():
        """Reset the internal NewHorizons LORRI parameters.

        Can be useful for debugging."""

        LORRI.instrument_kernel = None
        LORRI.fovs = {}
        LORRI.initialized = False

        NewHorizons.reset()

################################################################################
# Initialize at load time
################################################################################

LORRI.initialize()

################################################################################
# UNIT TESTS
################################################################################

import unittest
import os.path

class Test_NewHorizons_LORRI(unittest.TestCase):

    def runTest(self):

        from oops.unittester_support    import TESTDATA_PARENT_DIRECTORY
        import cspice

        snapshot = from_file(os.path.join(TESTDATA_PARENT_DIRECTORY,
                                  "nh/LORRI/LOR_0034969199_0X630_SCI_1.FIT"),
                             astrometry=True)
        self.assertFalse(snapshot.__dict__.has_key("data"))
        self.assertFalse(snapshot.__dict__.has_key("quality"))
        self.assertFalse(snapshot.__dict__.has_key("error"))
        self.assertFalse(snapshot.__dict__.has_key("point_calib"))
        self.assertFalse(snapshot.__dict__.has_key("extended_calib"))
        self.assertFalse(snapshot.__dict__.has_key("headers"))

        snapshot = from_file(os.path.join(TESTDATA_PARENT_DIRECTORY,
                                  "nh/LORRI/LOR_0034969199_0X630_SCI_1.FIT"),
                             data=False, calibration=True)
        self.assertFalse(snapshot.__dict__.has_key("data"))
        self.assertFalse(snapshot.__dict__.has_key("quality"))
        self.assertFalse(snapshot.__dict__.has_key("error"))
        self.assertTrue(snapshot.__dict__.has_key("point_calib"))
        self.assertTrue(snapshot.__dict__.has_key("extended_calib"))
        self.assertTrue(snapshot.__dict__.has_key("headers"))

        snapshot = from_file(os.path.join(TESTDATA_PARENT_DIRECTORY,
                                  "nh/LORRI/LOR_0034969199_0X630_SCI_1.FIT"),
                             data=True, calibration=False)
        self.assertTrue(snapshot.__dict__.has_key("data"))
        self.assertTrue(snapshot.__dict__.has_key("quality"))
        self.assertTrue(snapshot.__dict__.has_key("error"))
        self.assertFalse(snapshot.__dict__.has_key("point_calib"))
        self.assertFalse(snapshot.__dict__.has_key("extended_calib"))
        self.assertTrue(snapshot.__dict__.has_key("headers"))

        snapshot = from_file(os.path.join(TESTDATA_PARENT_DIRECTORY,
                                  "nh/LORRI/LOR_0034969199_0X630_SCI_1.FIT"),
                             headers=False)
        self.assertTrue(snapshot.__dict__.has_key("data"))
        self.assertTrue(snapshot.__dict__.has_key("quality"))
        self.assertTrue(snapshot.__dict__.has_key("error"))
        self.assertTrue(snapshot.__dict__.has_key("point_calib"))
        self.assertTrue(snapshot.__dict__.has_key("extended_calib"))
        self.assertFalse(snapshot.__dict__.has_key("headers"))

        snapshot = from_file(os.path.join(TESTDATA_PARENT_DIRECTORY,
                                  "nh/LORRI/LOR_0034969199_0X630_SCI_1.FIT"))
        self.assertTrue(snapshot.__dict__.has_key("data"))
        self.assertTrue(snapshot.__dict__.has_key("quality"))
        self.assertTrue(snapshot.__dict__.has_key("error"))
        self.assertTrue(snapshot.__dict__.has_key("point_calib"))
        self.assertTrue(snapshot.__dict__.has_key("extended_calib"))
        self.assertTrue(snapshot.__dict__.has_key("headers"))

        self.assertTrue(snapshot.data.shape == (1024,1024))
        self.assertTrue(snapshot.quality.shape == (1024,1024))
        self.assertTrue(snapshot.error.shape == (1024,1024))

        self.assertAlmostEqual(snapshot.time[1]-snapshot.time[0],
                               snapshot.texp)
        self.assertAlmostEqual(snapshot.time[0]+snapshot.texp/2,
                               cspice.utc2et(snapshot.headers[0]["SPCUTCID"]),
                               places=3)
        self.assertEqual(snapshot.target, "EUROPA")

        fov_1024 = snapshot.fov

        for pointing, geom in [('spice', 'fits'),
                               ('fits', 'spice'),
                               ('fits', 'fits')]:
            snapshot_fits = from_file(os.path.join(TESTDATA_PARENT_DIRECTORY,
                            "nh/LORRI/LOR_0034969199_0X630_SCI_1.FIT"),
                            geom=geom, pointing=pointing)

            self.assertEqual(snapshot.time, snapshot_fits.time)
            self.assertEqual(snapshot.texp, snapshot_fits.texp)

            meshgrid = oops.Meshgrid.for_fov(snapshot.fov, (0,0), limit=(0,0))
            bp = oops.Backplane(snapshot, meshgrid=meshgrid)
            bp_fits = oops.Backplane(snapshot_fits, meshgrid=meshgrid)
            ra =          bp.right_ascension().vals.astype('float')
            ra_fits =     bp_fits.right_ascension().vals.astype('float')
            dec =         bp.declination().vals.astype('float')
            dec_fits =    bp_fits.declination().vals.astype('float')
            europa =      bp.where_intercepted("europa").vals
            europa_fits = bp_fits.where_intercepted("europa").vals

            self.assertAlmostEqual(ra, ra_fits, places=3)
            self.assertAlmostEqual(dec, dec_fits, places=3)
            self.assertEqual(europa, 0.0)
            self.assertEqual(europa_fits, 0.0)

            # Adjust offset as CSPICE kernels change
            europa_uv = (385,510)
            meshgrid = oops.Meshgrid.for_fov(snapshot.fov, europa_uv,
                                             limit=europa_uv, swap=True)
            meshgrid_fits = oops.Meshgrid.for_fov(snapshot_fits.fov, europa_uv,
                                                  limit=europa_uv, swap=True)
            orig_fov = snapshot.fov
            orig_fits_fov = snapshot_fits.fov
            snapshot.fov = oops.fov.OffsetFOV(orig_fov, (63,-80))
            snapshot_fits.fov = oops.fov.OffsetFOV(orig_fits_fov, (46,-124))
            bp = oops.Backplane(snapshot, meshgrid=meshgrid)
            bp_fits = oops.Backplane(snapshot_fits, meshgrid=meshgrid)
            long =        bp.longitude("europa").vals.astype('float')
            long_fits =   bp_fits.longitude("europa").vals.astype('float')
            lat =         bp.latitude("europa").vals.astype('float')
            lat_fits =    bp_fits.latitude("europa").vals.astype('float')
            europa =      bp.where_intercepted("europa").vals
            europa_fits = bp_fits.where_intercepted("europa").vals
            snapshot.fov = orig_fov
            snapshot_fits.fov = orig_fits_fov

            self.assertAlmostEqual(long, long_fits, places=-1)
            self.assertAlmostEqual(lat, lat_fits, places=0)
            self.assertEqual(europa, 1.0)
            self.assertEqual(europa_fits, 1.0)

        europa_ext_iof = (snapshot.extended_calib["CHARON"].
                          value_from_dn(snapshot.data[440,606])).vals
        europa_pt_iof = (snapshot.point_calib["CHARON"].
                          value_from_dn(snapshot.data[440,606],
                                        (440,606))).vals
        self.assertGreater(europa_ext_iof, 0.35)
        self.assertLess(europa_ext_iof, 0.6)
        self.assertAlmostEqual(europa_ext_iof, europa_pt_iof, 1)

        snapshot = from_file(os.path.join(TESTDATA_PARENT_DIRECTORY,
                                  "nh/LORRI/LOR_0030710290_0x633_SCI_1.FIT"),
                             calibration=False)
        self.assertTrue(snapshot.data.shape == (256,256))
        self.assertTrue(snapshot.quality.shape == (256,256))
        self.assertTrue(snapshot.error.shape == (256,256))

        fov_256 = snapshot.fov

        self.assertAlmostEqual(fov_256.uv_scale.vals[0]/4,
                               fov_1024.uv_scale.vals[0])
        self.assertAlmostEqual(fov_256.uv_scale.vals[1]/4,
                               fov_1024.uv_scale.vals[1])

############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
