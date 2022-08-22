################################################################################
# oops/inst/nh/lorri.py
#
# 2/9/14 Created (RSF)
################################################################################

import numpy as np
import julian
import pdstable

try:
    import astropy.io.fits as pyfits
except ImportError:
    import pyfits
import solar

import oops
from oops.inst.nh.nh_ import NewHorizons

################################################################################
# Standard routines for interpreting WCS parameters, adapted from STSCI source.
#
# Note: Here (u,v) = (1,1) refers to the center of the lower left pixel. This
# differs from the oops definition in which (0,0) falls at the corner of an
# image and pixel centers have half-integer values.
################################################################################

#===============================================================================
# radec_from_uv
#===============================================================================
def radec_from_uv(u, v, header):
    dx = u - header['CRPIX1']
    dy = v - header['CRPIX2']

    dra_dx = header['CD1_1']
    dra_dy = header['CD1_2']
    ddec_dx = header['CD2_1']
    ddec_dy = header['CD2_2']

    xi  = (dra_dx * dx + dra_dy * dy) / oops.DPR
    eta = (ddec_dx * dx + ddec_dy * dy) / oops.DPR

    ra0  = header['CRVAL1'] / oops.DPR
    dec0 = header['CRVAL2'] / oops.DPR

    denom = np.cos(dec0) - eta * np.sin(dec0)

    ra  = oops.DPR * (np.arctan(xi / denom) + ra0)
    dec = oops.DPR * np.arctan((eta * np.cos(dec0) + np.sin(dec0)) /
                               (np.sqrt(denom**2 + xi**2)))

    return (ra, dec)
#===============================================================================



#===============================================================================
# uv_from_radec
#===============================================================================
def uv_from_radec(ra, dec, header):
    dra = ra - header['CRVAL1']
    ddec = dec - header['CRVAL2']

    dra_dx = header['CD1_1']
    dra_dy = header['CD1_2']
    ddec_dx = header['CD2_1']
    ddec_dy = header['CD2_2']

    det = dra_dx * ddec_dy - dra_dy * ddec_dx
    dx_dra = ddec_dy / det
    dx_ddec = -dra_dy / det
    dy_dra = -ddec_dx / det
    dy_ddec = dra_dx / det

    ra0 = header['CRVAL1'] / oops.DPR
    dec0 = header['CRVAL2'] / oops.DPR

    ra = ra / oops.DPR
    dec = dec / oops.DPR
    dra = ra - ra0

    denom = (np.sin(dec) * np.sin(dec0) +
             np.cos(dec) * np.cos(dec0) * np.cos(dra))

    xi = np.cos(dec) * np.sin(dra) / denom
    eta = (np.sin(dec) * np.cos(dec0) -
           np.cos(dec) * np.sin(dec0) * np.cos(dra)) / denom

    u = oops.DPR * (dx_dra * xi + dx_ddec * eta) + header['CRPIX1']
    v = oops.DPR * (dy_dra * xi + dy_ddec * eta) + header['CRPIX2']

    return (u,v)
#===============================================================================



#===============================================================================
# to_xms
#===============================================================================
def to_xms(x):
    if x < 0.:
        x_sign = -1
        x = -x
    else:
        x_sign = 1

    x = abs(x)
    h = int(x)
    x -= h
    x *= 60
    m = int(x)
    x -= m
    x *= 60
    return (int(x_sign) * h, m, x)
#===============================================================================



################################################################################
# Standard class methods
################################################################################

#===============================================================================
# from_file
#===============================================================================
def from_file(filespec, geom='spice', pointing='spice', fov_type='fast',
              asof=None, meta=None, **parameters):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Return a Snapshot object based on a given NewHorizons LORRI image file.

    If parameters["data"] is False, no data or associated arrays are loaded.
    If parameters["calibration"] is False, no calibration objects are created.
    If parameters["headers"] is False, no header dictionary is returned.

    If parameters["astrometry"] is True, this is equivalent to data=False,
    calibration=False, headers=False.

    If parameters["solar_range"] is specified, it overrides the distance from
    the Sun to the target body (in AU) for calibration purposes.

    Inputs:
        geom        'spice'     to use a SPICE SPK for the geometry;
                    'fits'      to read the geoemtry info from the header.
        pointing    'spice'     to use a SPICE CK for the pointing;
                    'fits'      to use the pointing info in the FITS header;
                    'fitsapp'   to use the pointing info in the FITS header, but
                                to treat it as apparent pointing rather than
                                pointing in the SSB frame.
                    'fits90'    to use the pointing info in the FITS header but
                                add a 90 degree rotation; this is needed to
                                correct an error in the first PDS delivery of
                                the data set.
        fov_type    'fast'      to use a separate numerically inverted
                                polynomial FOV for camera distortion;
                    'slow'      to invert the polynomial FOV using Newton's
                                method;
                    'flat'      to use a flat FOV model.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    assert geom in {'spice', 'fits'}
    assert pointing in {'spice', 'fits', 'fitsapp', 'fits90'}
    assert fov_type in {'fast', 'slow', 'flat'}

    LORRI.initialize(asof=asof, meta=meta)

    #-------------------------
    # Load the FITS file
    #-------------------------
    nh_file = pyfits.open(filespec)
    filename = os.path.split(filespec)[1]
    header = nh_file[0].header

    #-----------------------------------------
    # Get key information from the header
    #-----------------------------------------
    texp = header['EXPTIME']
    tdb_midtime = header['SPCSCET']
    tstart = tdb_midtime - texp/2
    shape = nh_file[0].data.shape

#     binning_mode = header['SFORMAT']
    if shape[0] == 1024:
        binning_mode = '1X1'
    elif shape[0] == 256:
        binning_mode = '4X4'
    else:
        raise ValueError('Unrecognized binning mode; shape =', str(shape))

    fov = LORRI.fovs[binning_mode, fov_type]

    target_name = header['SPCCBTNM']
    if target_name.strip() == '---':
        target_name = 'PLUTO'
    if target_name.startswith('PLUTO'):     # fixes some weird cases
        target_name = 'PLUTO'

    try:
        target_body = oops.Body.lookup(target_name)
    except:
        target_body = None

    if geom == 'spice':
        path = oops.Path.as_waypoint('NEW HORIZONS')
    else:
        #- - - - - - - - - - - - - - - - - - - - - - - -
        # First construct a path from the Sun to NH
        #- - - - - - - - - - - - - - - - - - - - - - - -
        posx = -header['SPCSSCX']
        posy = -header['SPCSSCY']
        posz = -header['SPCSSCZ']

        velx = -header['SPCSSCVX']
        vely = -header['SPCSSCVY']
        velz = -header['SPCSSCVZ']

        #- - - - - - - - - - - - - - - - - - - - - - - - - - -
        # The path_id has to be unique to this observation
        #- - - - - - - - - - - - - - - - - - - - - - - - - - -
        sun_path = oops.Path.as_waypoint('SUN')
        path_id = '.NH_PATH_' + filename
        sc_path = oops.path.LinearPath((oops.Vector3([posx, posy, posz]),
                                        oops.Vector3([velx, vely, velz])),
                                       tdb_midtime, sun_path,
                                       oops.Frame.J2000,
                                       id=path_id)
        path = oops.Path.as_waypoint(sc_path)

    if pointing == 'spice':
        frame = oops.Frame.as_wayframe('NH_LORRI')
    else:
        #- - - - - - - - - - - - - - - - - - - - -
        # Create a frame based on the boresight
        #- - - - - - - - - - - - - - - - - - - - -
        u_center = shape[1]/2. + 0.5    # offset to put [1,1] at center of pixel
        v_center = shape[0]/2. + 0.5
        (ra_deg, dec_deg) = radec_from_uv(u_center, v_center, header)
        north_clock_deg = header['SPCEMEN']

        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Apply the correction for apparent geometry if necessary
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if pointing == 'fitsapp':
            ra  = ra_deg  * oops.RPD
            dec = dec_deg * oops.RPD

            event = path.event_at_time(tdb_midtime)

            # Insert apparent vector as actual to reverse the aberration effect
            event.neg_arr_j2000 = oops.Vector3.from_ra_dec_length(ra, dec,
                                                                recursive=False)
            (ra, dec) = event.ra_and_dec(apparent=True)
            ra_deg  = ra  * oops.DPR
            dec_deg = dec * oops.DPR

        #- - - - - - - - - - - - - - - - - - - - - - - -
        # Apply the 90-degree rotation if necessary
        #- - - - - - - - - - - - - - - - - - - - - - - -
        elif pointing == 'fits90':
            year = int(header['SPCUTCID'][:4])
            if year <= 2012:
                north_clock_deg += 90.

        scet = header['SPCSCET']

        frame_id = '.NH_FRAME_' + filename
        lorri_frame = oops.frame.Cmatrix.from_ra_dec(ra_deg, dec_deg,
                                                     north_clock_deg,
                                                     oops.Frame.J2000,
                                                     id=frame_id)
        frame = oops.Frame.as_wayframe(lorri_frame)

        event = oops.Event(tdb_midtime, oops.Vector3.ZERO, path, frame)
        event.neg_arr_ap = oops.Vector3.ZAXIS
        los = event.neg_arr_ap_j2000

    #----------------------
    # Create a Snapshot
    #----------------------
    snapshot = oops.obs.Snapshot(('v','u'), tstart, texp, fov, path, frame,
                                 target = target_name,
                                 instrument = 'LORRI')

    #------------------------------
    # Interpret loader options
    #------------------------------
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
        if 'calib_body' in parameters:
            spectral_name = parameters['calib_body']

        #- - - - - - - - - - - - - - - - -
        # Look up the solar range...
        #- - - - - - - - - - - - - - - - -
        try:
            solar_range = parameters['solar_range']
        except KeyError:
            solar_range = None

        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # If necessary, get the solar range from the target name
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
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

            #- - - - - - - - - - -
            # Extended source
            #- - - - - - - - - - -
            spectral_radiance = header['R' + spectral_name]

            #- - - - - - - - -
            # Point source
            #- - - - - - - - -
            spectral_irradiance = header['P' + spectral_name]

            F_solar = 176.  # pivot 6076.2 A at 1 AU

            #- - - - - - - - - - - -
            # Conversion to I/F
            #- - - - - - - - - - - -
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
#===============================================================================



#===============================================================================
# from_index
#===============================================================================
def from_index(filespec, fov_type='fast', asof=None, meta=None, **parameters):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    Return a list of Snapshot objects, one for each row in a supplemental
    index file.

    Inputs:
        fov_type    'fast'      to use a separate numerically inverted
                                polynomial FOV for camera distortion;
                    'slow'      to invert the polynomial FOV using Newton's
                                method;
                    'flat'      to use a flat FOV model.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    LORRI.initialize(asof=asof, meta=meta)

    #------------------------------------------------------------------------
    # Read the index file
    #------------------------------------------------------------------------
    COLUMNS = []                # Return all columns
    TIMES = ['START_TIME']      # Convert this one to TAI
    table = pdstable.PdsTable(filespec, columns=COLUMNS, times=TIMES)
    row_dicts = table.dicts_by_row()

    #------------------------------------------------------------------------
    # Create a list of Snapshot objects
    #------------------------------------------------------------------------
    snapshots = []
    for dict in row_dicts:

        tstart = julian.tdb_from_tai(dict['START_TIME'])
        texp = max(0.0005, dict['EXPOSURE_DURATION'])
        fov = LORRI.fovs[dict['BINNING_MODE'].upper(), fov_type]
        target_name = dict['TARGET_NAME']

        #- - - - - - - - - - - - -
        # Create a Snapshot
        #- - - - - - - - - - - - -
        item = oops.obs.Snapshot(('v','u'), tstart, texp,
                                 fov, 'NEW HORIZONS', 'NH_LORRI',
                                 dict = dict,
                                 index_dict = dict,
                                 target = target_name,
                                 instrument = 'LORRI')

        snapshots.append(item)

    return snapshots
#===============================================================================



#*******************************************************************************
# LORRI
#*******************************************************************************
class LORRI(object):
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    A instance-free class to hold NewHorizons LORRI instrument parameters.
    """
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    instrument_kernel = None
    fovs = {}
    initialized = False
    asof = None
    meta = None

    #------------------------------------------------------------------------
    # Create a master version of the LORRI distortion models from
    #   Owen Jr., W.M., 2011. New Horizons LORRI Geometric Calibration of
    #   August 2006. JPL IOM 343L-11-002.
    # These polynomials convert from X,Y (radians) to U,V (pixels).
    #------------------------------------------------------------------------
    LORRI_F  =  2619.008    # mm
    LORRI_E2 =  2.696e-5    # / mm2
    LORRI_E5 =  1.988e-5    # / mm
    LORRI_E6 = -2.864e-5    # / mm
    LORRI_KX =  76.9231     # samples/mm
    LORRI_KY = -76.9231     # lines/mm

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

    #------------------------------------------------------------------------
    # Create a master version of the inverse distortion model.
    # These coefficients were computed by numerically solving the above
    # polynomials.
    # These polynomials convert from U,V (pixels) to X,Y (radians).
    # Maximum errors from applying the original distortion model and
    # then inverting:
    #   U DIFF MIN MAX -0.00261017184761 0.0018410196501
    #   V DIFF MIN MAX -0.00263250108583 0.00209864359397
    #------------------------------------------------------------------------
    LORRI_INV_COEFF = np.zeros((4,4,2))
    LORRI_INV_COEFF[:,:,0] = [
        [  5.62144901e-10,  1.80741669e-15, -3.62872755e-15, -7.65201036e-21],
        [  4.96369475e-06,  1.27370808e-12, -2.24498555e-14,  0.00000000e+00],
        [  1.83505816e-12,  1.87593608e-18,  0.00000000e+00,  0.00000000e+00],
        [ -2.24714913e-14,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]
    LORRI_INV_COEFF[:,:,1] = [
        [ -3.90196341e-10, -4.96369475e-06, -1.27377633e-12,  2.24721956e-14],
        [ -1.96689309e-15, -1.83495964e-12, -1.87499462e-18,  0.00000000e+00],
        [  2.51861242e-15,  2.24491504e-14,  0.00000000e+00,  0.00000000e+00],
        [  7.85425602e-21,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]

    #===========================================================================
    # initialize
    #===========================================================================
    @staticmethod
    def initialize(asof=None, time=None, meta=None):
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Fill in key information about LORRI. Must be called first.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        #- - - - - - - - - - - - - - - -
        # Update kernels if necessary
        #- - - - - - - - - - - - - - - -
        NewHorizons.initialize(asof=asof, time=time, meta=meta)

        #- - - - - - - - - - - - - - - -
        # Quick exit after first call
        #- - - - - - - - - - - - - - - -
        if LORRI.initialized and LORRI.asof == asof: return

        #- - - - - - - - - - - - - - - -
        # Load the instrument kernel
        #- - - - - - - - - - - - - - - -
        kernels = NewHorizons.spice_instrument_kernel('LORRI')
        LORRI.instrument_kernel = kernels[0]

        #- - - - - - - - - - - - - - - -
        # Construct a Polynomial FOV
        #- - - - - - - - - - - - - - - -
        info = LORRI.instrument_kernel['INS']['NH_LORRI_1X1']

        #- - - - - - - - - - - -
        # Full field of view
        #- - - - - - - - - - - -
        lines = info['PIXEL_LINES']
        samples = info['PIXEL_SAMPLES']

        xfov = info['FOV_REF_ANGLE']
        yfov = info['FOV_CROSS_ANGLE']
        assert info['FOV_ANGLE_UNITS'] == 'DEGREES'

        #- - - - - - - - - - - - - - - - - - - - - - -
        # Display directions: [u,v] = [right,down]
        #- - - - - - - - - - - - - - - - - - - - - - -
        full_fov = oops.fov.Polynomial((samples,lines),
                                       coefft_uv_from_xy=LORRI.LORRI_COEFF,
                                       coefft_xy_from_uv=None)
        full_fov_fast = oops.fov.Polynomial((samples,lines),
                                       coefft_uv_from_xy=LORRI.LORRI_COEFF,
                                       coefft_xy_from_uv=LORRI.LORRI_INV_COEFF)

        full_fov_flat = oops.fov.FlatFOV((4.96e-6,-4.96e-6), (1024,1024))

        #- - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Load the dictionary, include the subsampling modes
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - -
        LORRI.fovs['1X1', 'slow'] = full_fov
        LORRI.fovs['1X1', 'fast'] = full_fov_fast
        LORRI.fovs['1X1', 'flat'] = full_fov_flat
        LORRI.fovs['4X4', 'slow'] = oops.fov.Subsampled(full_fov, 4)
        LORRI.fovs['4X4', 'fast'] = oops.fov.Subsampled(full_fov_fast, 4)
        LORRI.fovs['4X4', 'flat'] = oops.fov.Subsampled(full_fov_flat, 4)

        #- - - - - - - - - - - - - - -
        # Construct a SpiceFrame
        #- - - - - - - - - - - - - - -
        lorri_flipped = oops.frame.SpiceFrame('NH_LORRI', id='NH_LORRI_FLIPPED')

        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # The SPICE IK gives the boresight along -Z, so flip axes
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        flipxyz = oops.Matrix3([[ 1, 0, 0],
                                [ 0,-1, 0],
                                [ 0, 0,-1]])
        ignore = oops.frame.Cmatrix(flipxyz, lorri_flipped, 'NH_LORRI')

        LORRI.initialized = True
        LORRI.asof = asof
        LORRI.meta = meta
    #===========================================================================



    #===========================================================================
    # reset
    #===========================================================================
    @staticmethod
    def reset():
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        Reset the internal NewHorizons LORRI parameters.

        Can be useful for debugging.
        """
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        LORRI.instrument_kernel = None
        LORRI.fovs = {}
        LORRI.initialized = False
        LORRI.asof = asof
        LORRI.meta = meta

        NewHorizons.reset()
    #===========================================================================



#*******************************************************************************



################################################################################
# Initialize at load time
################################################################################

LORRI.initialize()

################################################################################
# UNIT TESTS
################################################################################

import unittest
import os.path

#*******************************************************************************
# Test_NewHorizons_LORRI
#*******************************************************************************
class Test_NewHorizons_LORRI(unittest.TestCase):

    #===========================================================================
    # runTest
    #===========================================================================
    def runTest(self):

        from oops.unittester_support    import TESTDATA_PARENT_DIRECTORY
        import cspyce

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
                               cspyce.utc2et(snapshot.headers[0]["SPCUTCID"]),
                               places=3)
        self.assertEqual(snapshot.target, "EUROPA")

        fov_1024 = snapshot.fov

        for geom, pointing, offset in [('spice', 'fits90', (-49,-28)),
                                       ('fits', 'spice', (-4,-12)),
                                       ('fits', 'fits90', (-48,-27))]:
            snapshot_fits = from_file(os.path.join(TESTDATA_PARENT_DIRECTORY,
                            "nh/LORRI/LOR_0034969199_0X630_SCI_1.FIT"),
                            geom=geom, pointing=pointing, fast_distortion=True)

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

            self.assertAlmostEqual(ra, ra_fits, places=2)
            self.assertAlmostEqual(dec, dec_fits, places=2)
            self.assertEqual(europa, 0.0)
            self.assertEqual(europa_fits, 0.0)

            # Adjust offset as SPICE kernels change
            orig_fov = snapshot.fov
            orig_fits_fov = snapshot_fits.fov
            snapshot.fov = oops.fov.OffsetFOV(orig_fov, (-4,-13))
            snapshot_fits.fov = oops.fov.OffsetFOV(orig_fits_fov, offset)

            europa_uv = (500,440)
            meshgrid = oops.Meshgrid.for_fov(snapshot.fov, europa_uv,
                                             limit=europa_uv, swap=True)
            meshgrid_fits = oops.Meshgrid.for_fov(snapshot_fits.fov, europa_uv,
                                                  limit=europa_uv, swap=True)
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

#             self.assertAlmostEqual(long, long_fits, places=1)
#             self.assertAlmostEqual(lat, lat_fits, places=1)
            self.assertEqual(europa, True)
            self.assertEqual(europa_fits, True)

        europa_ext_iof = (snapshot.extended_calib["CHARON"].
                          value_from_dn(snapshot.data[440,606])).vals
        europa_pt_iof = (snapshot.point_calib["CHARON"].
                          value_from_dn(snapshot.data[440,606],
                                        (440,606))).vals
        self.assertGreater(europa_ext_iof, 0.35)
        self.assertLess(europa_ext_iof, 0.6)
        self.assertAlmostEqual(europa_ext_iof, europa_pt_iof, 1)

        snapshot = from_file(os.path.join(TESTDATA_PARENT_DIRECTORY,
                                  "nh", "LORRI", "LOR_0030710290_0x633_SCI_1.FIT"),
                             calibration=False)
        self.assertTrue(snapshot.data.shape == (256,256))
        self.assertTrue(snapshot.quality.shape == (256,256))
        self.assertTrue(snapshot.error.shape == (256,256))

        fov_256 = snapshot.fov

        self.assertAlmostEqual(fov_256.uv_scale.vals[0]/4,
                               fov_1024.uv_scale.vals[0])
        self.assertAlmostEqual(fov_256.uv_scale.vals[1]/4,
                               fov_1024.uv_scale.vals[1])
    #===========================================================================



#*******************************************************************************



############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
################################################################################
